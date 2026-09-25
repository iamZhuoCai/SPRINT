import contextlib
import math
from dataclasses import dataclass
from itertools import permutations
from logging import getLogger
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from genrec.constants import MAX_ITEM_SEQ_LEN, N_CANDIDATES
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer
from genrec.utils import log
from genrec.models.SPRINT.diff_transformer import (
    DecoderBlock,
    EncoderBlock,
    make_norm,
)
from genrec.models.SPRINT.mean_flow import (
    TimestepEmbedder,
    discrete_mask_at_r,
    get_inference_time_schedule,
    sample_tr,
)


@dataclass
class SPRINTOutput:
    loss: torch.Tensor
    ghost_loss: torch.Tensor = None  # raw ghost InfoNCE (pre-weight)
    ghost_weighted: torch.Tensor = None  # what actually entered ``loss``
    # Dual-level breakdown (mf_dual_level only; None otherwise).
    dual_mix_loss: torch.Tensor = None  # NLL of the mixed distribution
    dual_tok_loss: torch.Tensor = None  # token-level branch NLL (aux)
    dual_item_loss: torch.Tensor = None  # item-level branch NLL (aux)
    dual_alpha: torch.Tensor = None  # current mixing weight on the token branch


class SPRINT(AbstractModel):

    def __init__(self, config: dict, dataset: AbstractDataset,
                 tokenizer: AbstractTokenizer):
        super(SPRINT, self).__init__(config, dataset, tokenizer)

        self.logger = getLogger()

        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        self.posValidTokens = self.get_posValidToken().to(
            self.config['device'])

        self.mask_token_id = tokenizer.mask_token
        num_r_tokens = config.get('mf_num_r_tokens', 4)
        num_t_tokens = config.get('mf_num_t_tokens', 4)
        self.num_r_tokens = num_r_tokens
        self.num_t_tokens = num_t_tokens
        # No r/t conditioning: r only sets mask_prob = 1 - r.
        # ---- DiffGRM-style encoder–decoder backbone ----
        self.d_model = config['d_model']
        self.n_head = config['n_heads']
        self.n_digit = self.tokenizer.n_digit
        if num_r_tokens != self.n_digit or num_t_tokens != self.n_digit:
            raise ValueError(
                f"mf_num_r/t_tokens must equal n_digit={self.n_digit} for "
                f"item-like slot compression, got r={num_r_tokens}, "
                f"t={num_t_tokens}")
        # Flat digits: skip item_mlp / rt_mlp; feed SID digit tokens and the
        # raw r/t slot tokens straight into the transformer.
        self.mf_flat_digits = bool(config.get('mf_flat_digits', False))
        # Ablation: keep r conditioning tokens, drop t (still sample t for
        # masking / data_proportion; just do not inject t emb into the net).
        self.mf_drop_t_emb = bool(config.get('mf_drop_t_emb', False))
        if config.get('n_inner') is not None:
            self.n_inner = int(config['n_inner'])
        else:
            self.n_inner = int(self.d_model * config.get('mlp_ratio', 4))
        self.encoder_n_layer = 1
        self.decoder_n_layer = 4
        dropout = float(config.get('dropout_rate', 0.1))
        norm_type = 'rmsnorm'
        norm_eps = float(config.get('norm_eps', 1e-5))

        inner_tok = getattr(tokenizer, 'tokenizer', tokenizer)
        self.sid_offset = int(getattr(inner_tok, 'sid_offset', 3))
        if hasattr(inner_tok, 'codebook_size'):
            self.codebook_size = int(inner_tok.codebook_size)
        else:
            self.codebook_size = int(tokenizer.codebook_sizes[0])

        self.embedding = nn.Embedding(tokenizer.vocab_size, self.d_model)
        # Per-digit [MASK] embeddings; decoder logits are dot products with
        # the shared token embedding; no absolute digit position embedding.
        self.mask_emb_table = nn.Embedding(self.n_digit, self.d_model)
        # item_mlp / rt_mlp: compress concat(n_digit embeds) → d,
        # nd*d → d → ReLU → d.
        in_dim = self.n_digit * self.d_model
        self.item_mlp = self._make_compress_mlp(
            in_dim, self.d_model, [self.d_model])
        self.rt_mlp = self._make_compress_mlp(
            in_dim, self.d_model, [self.d_model])
        hist_len = MAX_ITEM_SEQ_LEN
        self.pos_emb_enc = nn.Embedding(hist_len, self.d_model)
        # keep for informative-prior digit-level item pos compatibility
        self.item_pos_emb = self.pos_emb_enc

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(self.d_model, self.n_head, self.n_inner, dropout,
                         dropout, act='gelu', norm_type=norm_type,
                         norm_eps=norm_eps)
            for _ in range(self.encoder_n_layer)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(self.d_model, self.n_head, self.n_inner, dropout,
                         dropout, act='gelu', norm_type=norm_type,
                         norm_eps=norm_eps)
            for _ in range(self.decoder_n_layer)
        ])
        self.ln_f = make_norm(norm_type, self.d_model, norm_eps)
        self.ln_dec = make_norm(norm_type, self.d_model, norm_eps)
        self.drop = nn.Dropout(dropout)

        self.output_adapter = nn.Identity()

        freq_dim = config.get('mf_freq_dim', 256)
        self.r_embedder = TimestepEmbedder(self.d_model, freq_dim)
        self.t_embedder = TimestepEmbedder(self.d_model, freq_dim)
        self.r_prefix_tokens = nn.Parameter(
            torch.randn(1, self.num_r_tokens, self.d_model) * 0.02)
        self.t_prefix_tokens = nn.Parameter(
            torch.randn(1, self.num_t_tokens, self.d_model) * 0.02)

        # ---- CVAE latent z: q(z|history,target) at train, p(z|history) at test.
        # z is added onto [MASK] label slots so latent bias applies only where
        # the model must predict; self-attn still couples digits in the decoder.
        self.apply(self._init_weights)

        # Every training row is forced to (r=0, t=1), i.e. all digits masked;
        # sample_tr still draws r ~ U[0,1], t ~ U[r,1] first (then overrides
        # them), which keeps the RNG stream of the reference runs.
        self.mf_data_proportion = 1.0
        # Mask schedule: bernoulli (default mean-flow) | llada (exact count)
        self.mf_mask_schedule = str(
            config.get('mf_mask_schedule', 'bernoulli')).lower()
        if self.mf_mask_schedule not in ('bernoulli', 'llada'):
            raise ValueError(
                f"mf_mask_schedule must be 'bernoulli' or 'llada', "
                f"got {self.mf_mask_schedule}")
        # Enforce t ∈ [r + mf_t_min_gap, 1] when > 0 (0 keeps t >= r only).
        self.mf_t_min_gap = float(config.get('mf_t_min_gap', 0.0))
        if self.mf_t_min_gap < 0 or self.mf_t_min_gap >= 1.0:
            raise ValueError(
                f"mf_t_min_gap must be in [0, 1), got {self.mf_t_min_gap}")
        self.mf_shuffle_label_digits = config.get(
            'mf_shuffle_label_digits', False)
        self.mf_shuffle_label_digits_prob = float(
            config.get('mf_shuffle_label_digits_prob', 1.0))
        self.mf_shuffle_all_digit_views = config.get(
            'mf_shuffle_all_digit_views', False)
        # Canonical view + this many random non-identity digit-order views.
        # Ignored when mf_shuffle_all_digit_views=True (uses all n_digit!).
        self.mf_num_extra_digit_views = int(
            config.get('mf_num_extra_digit_views', 0))
        self.mf_digit_view_chunk = int(config.get('mf_digit_view_chunk', 4))
        self.mf_informative_prior = config.get('mf_informative_prior', False)
        self.mf_filter_invalid_sids = config.get('mf_filter_invalid_sids',
                                                 False)
        self.mf_item_align_weight = float(
            config.get('mf_item_align_weight', 0.0))
        self.mf_item_align_loss = str(
            config.get('mf_item_align_loss', 'mse')).lower()
        if self.mf_item_align_loss not in ('mse', 'cosine', 'softmax', 'kl'):
            raise ValueError(
                f"mf_item_align_loss must be 'mse', 'cosine', 'softmax', or "
                f"'kl', got {self.mf_item_align_loss}")
        self.mf_item_align_neg_mode = str(
            config.get('mf_item_align_neg_mode', 'random')).lower()
        if self.mf_item_align_neg_mode not in ('random', 'catalog', 'inbatch'):
            raise ValueError(
                f"mf_item_align_neg_mode must be 'random', 'catalog', or "
                f"'inbatch', got {self.mf_item_align_neg_mode}")
        self.mf_item_align_num_neg = int(
            config.get('mf_item_align_num_neg', 100))
        self.mf_item_align_temperature = float(
            config.get('mf_item_align_temperature', 0.1))
        # Scale on the primary masked-digit CE term in mean_flow_loss.
        # X_r / X_t logits consistency (same target, coupled intermediate states).
        self.mf_rt_consist_weight = float(
            config.get('mf_rt_consist_weight', 0.0))
        self.mf_rt_consist_loss = str(
            config.get('mf_rt_consist_loss', 'mse')).lower()
        if self.mf_rt_consist_loss not in ('mse', 'kl', 'cosine'):
            raise ValueError(
                f"mf_rt_consist_loss must be 'mse', 'kl', or 'cosine', "
                f"got {self.mf_rt_consist_loss}")
        self.mf_rt_consist_detach = str(
            config.get('mf_rt_consist_detach', 't')).lower()
        if self.mf_rt_consist_detach not in ('t', 'r', 'none'):
            raise ValueError(
                f"mf_rt_consist_detach must be 't', 'r', or 'none', "
                f"got {self.mf_rt_consist_detach}")
        self.mf_rt_consist_on = str(
            config.get('mf_rt_consist_on', 'all')).lower()
        if self.mf_rt_consist_on not in ('all', 'masked_r'):
            raise ValueError(
                f"mf_rt_consist_on must be 'all' or 'masked_r', "
                f"got {self.mf_rt_consist_on}")
        # Ghost penalty = the dual-level (token + item) loss below. It is the
        # whole training loss (weight 1, on from epoch 1, no label smoothing).
        # Catalog negatives per example; only the positive's own SID is
        # dropped from them.
        self.mf_ghost_penalty_num_neg = int(
            config.get('mf_ghost_penalty_num_neg', 0))
        # ---- Dual-level (token + item) contrastive ghost -------------------
        # Token level is Σ_d log p_d(c_d); item level is cos(u, v_c) with
        # u = item_mlp(concat decoder hidden) and v_c = item_mlp(concat E(c))
        # (shared item_mlp, stop-grad into E), i.e. a joint SID representation
        # that does NOT factorise over digits. Each branch is softmax-
        # normalised over the same catalog candidates (token τ = 1, item τ =
        # mf_dual_item_tau). Training loss is L_tok + L_item; inference ranks
        # the catalog by P_tok + w·P_item with a fixed w = 1.
        self.mf_dual_item_tau = float(config.get('mf_dual_item_tau', 0.07))
        self._catalog_emb_cache = None
        # Kept as a (frozen) parameter so checkpoints keep the same keys.
        self.dual_item_weight = nn.Parameter(
            torch.tensor(1.0), requires_grad=False)

        # ---- Jump-composition consistency (discrete MeanFlow identity) -----
        # One big jump must equal two small ones:
        #     f(X_r, r→t)  ≡  f( f(X_r, r→m), m→t ),   r < m < t
        # Teacher: two no-grad hops through a waypoint m (it re-conditions
        # after partially unmasking, so it models the joint better).
        # Student: the single hop r→t, which is what gen_steps=1 deploys.
        #
        # Why this is the piece that makes (r,t) matter: the plain x0-prediction
        # target is provably t-independent (given X_r, the posterior over clean
        # data does not depend on how far you intend to jump), which is why the
        # mf_rt_condition=none ablation costs ~0 and why conditioning alpha on
        # (r,t) buys interpretability but no accuracy. This loss instead ties
        # DIFFERENT intervals to each other, so competence at small jumps is
        # forced into the (0,1) operator that inference actually uses.
        self.mf_jump_consist_weight = float(
            config.get('mf_jump_consist_weight', 0.0))
        self.mf_jump_consist_loss = str(
            config.get('mf_jump_consist_loss', 'kl')).lower()
        if self.mf_jump_consist_loss not in ('kl', 'mse'):
            raise ValueError(
                f"mf_jump_consist_loss must be 'kl' or 'mse', "
                f"got {self.mf_jump_consist_loss}")
        # Fraction of rows the teacher rollout runs on (cost control).
        self.mf_jump_consist_ratio = float(
            config.get('mf_jump_consist_ratio', 1.0))
        # How the waypoint state is filled: argmax (deterministic) or sample.
        self.mf_jump_consist_fill = str(
            config.get('mf_jump_consist_fill', 'argmax')).lower()
        if self.mf_jump_consist_fill not in ('argmax', 'sample'):
            raise ValueError(
                f"mf_jump_consist_fill must be 'argmax' or 'sample', "
                f"got {self.mf_jump_consist_fill}")
        self.mf_jump_consist_start_epoch = int(
            config.get('mf_jump_consist_start_epoch', 0))
        # Minimum separation for the waypoint so neither hop is degenerate.
        self.mf_jump_consist_min_gap = float(
            config.get('mf_jump_consist_min_gap', 0.0))
        # ---- X_t prediction: predict the INTERMEDIATE state, not clean x0 ---
        # Default target is the clean SID, i.e. the model learns
        # p(x0 | x_r, history). Given x_r that posterior does not depend on t
        # at all, which is why t currently receives no gradient and the
        # mf_rt_condition=none ablation is free.
        #
        # With this term the target becomes X_t (r < t, t is the CLEANER end).
        # For a position masked at r, the coupled path gives
        #     the true code   with prob (t-r)/(1-r)
        #     [MASK]          with prob (1-t)/(1-r)
        # so the output distribution needs a MASK class and its mass is an
        # explicit function of (r, t). The model cannot produce it without
        # reading r and t -> the time conditioning becomes load-bearing.
        #
        # Caveat worth knowing before reading results: under the absorbing
        # kernel the optimal solution factorises as
        #     lambda * p(x0 | x_r)  +  (1-lambda) * delta_MASK,  lambda=(t-r)/(1-r)
        # where p(x0|x_r) is still t-independent and lambda is closed-form. So
        # (r,t) become NECESSARY inputs, but at inference (r=0,t=1) lambda=1
        # and the prediction reduces to the usual x0 one. Expect the rtnone
        # ablation to degrade; do not necessarily expect the metric to move.
        self.mf_xt_pred_weight = float(config.get('mf_xt_pred_weight', 0.0))
        # Restrict the loss to positions masked at r (the ones being resolved).
        self.mf_xt_pred_on_masked_only = bool(
            config.get('mf_xt_pred_on_masked_only', True))
        # What counts as the target on those positions:
        #   with_mask     - codebook + one extra MASK class; positions still
        #                   masked at t are labelled MASK. The MASK mass is
        #                   (1-t)/(1-r), which is what makes the target depend
        #                   on (r,t) and gives the time conditioning gradient.
        #   revealed_only - only positions the path REVEALS by t are supervised
        #                   (plain codebook CE, no MASK class); positions still
        #                   masked at t are dropped from the loss.
        #                   Cleaner head: at inference lambda=1 so MASK is never
        #                   wanted, and letting it compete for softmax mass can
        #                   only blunt the code predictions. But it also removes
        #                   the (r,t) dependence of the target -- given X_r the
        #                   identity of the true code does not depend on t, so
        #                   expect the rtnone ablation to go flat again.
        self.mf_xt_pred_target = str(
            config.get('mf_xt_pred_target', 'with_mask')).lower()
        if self.mf_xt_pred_target not in ('with_mask', 'revealed_only'):
            raise ValueError(
                f"mf_xt_pred_target must be 'with_mask' or 'revealed_only', "
                f"got {self.mf_xt_pred_target}")
        # EMA teacher decay; 0 disables (teacher = live weights).
        # Self-distillation against the LIVE model is a moving target: every
        # optimiser step changes the teacher, so the student chases something
        # that runs away from it. At tiny weights the feedback is too weak to
        # matter, but it showed up as loss spikes at w=30 and divergence at
        # w=100. An EMA teacher changes ~1/(1-decay) times slower, which is
        # the same reason DQN uses a target network.
        self.mf_jump_consist_ema = float(
            config.get('mf_jump_consist_ema', 0.0))
        # Ramp the decay early on, when the EMA still lags random init.
        self.mf_jump_consist_ema_warmup = bool(
            config.get('mf_jump_consist_ema_warmup', True))
        # Plain dict, deliberately NOT registered: this is a training-time aid
        # and should not travel in checkpoints (it would double their size and
        # is meaningless without the optimiser state).
        self._ema_state = {}
        self._ema_steps = 0

        # lengths of padded digit codebooks in posValidTokens
        self.pos_valid_counts = (self.posValidTokens != 0).sum(dim=1)

        self.log(
            f"backbone: DiffGRM encoder–decoder "
            f"(enc={self.encoder_n_layer}, dec={self.decoder_n_layer}, "
            f"d={self.d_model}, heads={self.n_head}, inner={self.n_inner}, "
            f"norm={norm_type}, eps={norm_eps}, hist_len={hist_len}, "
            f"mask_emb_table=on, output=shared_emb_dot_product, "
            f"decoder_pos_emb=off)")
        if self.mf_flat_digits:
            hist_repr = "hist=flat_digit_tokens(no_item_mlp)"
        else:
            hist_repr = (f"hist=item_mlp({self.n_digit}d→{self.d_model}→d,"
                         f"ReLU)")
        self.log(
            f"generation: discrete mean flow, 1 sampling step at (r=0, t=1), "
            f"mask_schedule={self.mf_mask_schedule}, "
            f"train at (r=0, t=1) for every row (all digits masked), "
            f"t_min_gap={self.mf_t_min_gap}, "
            f"{hist_repr}")
        self.log(
            f"ghost penalty: num_neg={self.mf_ghost_penalty_num_neg}")
        self.log(
            f"dual level: train=L_tok+L_item, infer=P_tok+w*P_item (w=1), "
            f"neg=catalog, item=cos(u,v_c) via shared item_mlp, "
            f"tok_tau=1.0, item_tau={self.mf_dual_item_tau}")

        self.catalog_tokens = torch.unique(self.item_id2tokens[1:], dim=0)
        self.catalog_packed = self._pack_sids(self.catalog_tokens)
        if self.mf_filter_invalid_sids:
            self.log(
                f"SID post-filter: generate digit combos then drop SIDs not in "
                f"catalog ({self.catalog_tokens.shape[0]} unique real SIDs)")
        self.log(
            f"constrained decode: rank full catalog "
            f"({self.catalog_tokens.shape[0]} unique real SIDs) by "
            f"P_tok + w*P_item; illegal rate is 0 by construction")

    def _score_sids_from_log_probs(
        self,
        log_probs: torch.Tensor,
        sid_tokens: torch.Tensor,
        per_example: bool = False,
        digit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score SIDs under batch digit distributions: ``Σ_d log p_d``.

        Args:
            log_probs: ``(B, n_digit, V)``
            sid_tokens:
              - ``per_example=True``: ``(B, n_digit)`` one SID per row → ``(B,)``
              - ``per_example=False`` + 2D: ``(N, n_digit)`` shared catalog → ``(B, N)``
              - 3D: ``(B, N, n_digit)`` per-row candidates → ``(B, N)``
            per_example: interpret 2D tokens as one SID per batch row.
            digit_mask: optional ``(B, n_digit)`` bool; when set, only masked
                digits contribute (unmasked → 0).
        """
        n_digit = self.n_digit
        if sid_tokens.dim() == 3:
            logs = [
                log_probs[:, d, :].gather(1, sid_tokens[:, :, d])
                for d in range(n_digit)
            ]
            stacked = torch.stack(logs, dim=-1)
        elif sid_tokens.dim() != 2 or sid_tokens.shape[-1] != n_digit:
            raise ValueError(
                f"sid_tokens must be (B, n_digit), (N, n_digit), or "
                f"(B, N, n_digit); got {tuple(sid_tokens.shape)}")
        elif per_example:
            if sid_tokens.shape[0] != log_probs.shape[0]:
                raise ValueError(
                    f"per_example SID batch {sid_tokens.shape[0]} != "
                    f"log_probs batch {log_probs.shape[0]}")
            logs = [
                log_probs[:, d, :].gather(
                    1, sid_tokens[:, d:d + 1]).squeeze(1)
                for d in range(n_digit)
            ]
            stacked = torch.stack(logs, dim=-1)
        else:
            batch_size = log_probs.shape[0]
            logs = [
                log_probs[:, d, :].gather(
                    1, sid_tokens[:, d].unsqueeze(0).expand(batch_size, -1))
                for d in range(n_digit)
            ]
            stacked = torch.stack(logs, dim=-1)

        if digit_mask is not None:
            if digit_mask.shape != (log_probs.shape[0], n_digit):
                raise ValueError(
                    f"digit_mask must be (B, n_digit)="
                    f"{(log_probs.shape[0], n_digit)}, "
                    f"got {tuple(digit_mask.shape)}")
            # stacked is (B, n_digit) or (B, N, n_digit)
            if stacked.dim() == 2:
                stacked = stacked.masked_fill(~digit_mask, 0.0)
            else:
                stacked = stacked.masked_fill(
                    ~digit_mask.unsqueeze(1), 0.0)
        return stacked.sum(dim=-1)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

    def _map_item_tokens(self) -> torch.Tensor:
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(
                self.tokenizer.item2tokens[item])
        return item_id2tokens

    def get_posValidToken(self) -> torch.Tensor:
        posValidTokens = {}

        for item in self.tokenizer.item2tokens:
            cur_tokens = self.tokenizer.item2tokens[item]
            for pos in range(len(cur_tokens)):
                if pos not in posValidTokens.keys():
                    posValidTokens[pos] = set()
                posValidTokens[pos].add(cur_tokens[pos])
        max_token_num = 0

        for pos in posValidTokens:
            posValidTokens[pos] = sorted(list(posValidTokens[pos]))
            max_token_num = max(max_token_num, len(posValidTokens[pos]))

        posValidTokens_pt = torch.zeros((len(posValidTokens), max_token_num),
                                        dtype=torch.long)

        for pos in posValidTokens:
            posValidTokens_pt[
                pos, :len(posValidTokens[pos])] = torch.LongTensor(
                    posValidTokens[pos])

        return posValidTokens_pt

    @property
    def n_parameters(self) -> str:
        total_params = sum(p.numel() for p in self.parameters()
                           if p.requires_grad)
        emb_params = sum(p.numel() for p in self.embedding.parameters()
                         if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def _apply_mask_emb_table(
        self,
        embeds: torch.Tensor,
        token_ids: torch.Tensor,
        digit_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replace masked digit slots with per-digit mask_emb_table vectors.

        ``digit_order[b, pos]`` is the original digit index at ``pos`` (identity
        when unset). Mask embeddings are keyed by original digit, not sequence
        position, so shuffled digit views stay consistent.
        """
        out = embeds.clone()
        if digit_order is None:
            for d in range(self.n_digit):
                mask_emb = self.mask_emb_table.weight[d]
                is_masked = token_ids[:, d] == self.mask_token_id
                out[:, d, :] = torch.where(is_masked.unsqueeze(-1), mask_emb,
                                           out[:, d, :])
            return out

        for pos in range(self.n_digit):
            is_masked = token_ids[:, pos] == self.mask_token_id
            if not is_masked.any():
                continue
            digs = digit_order[:, pos]
            for dig in range(self.n_digit):
                sel = is_masked & (digs == dig)
                if sel.any():
                    out[sel, pos, :] = self.mask_emb_table.weight[dig]
        return out

    def _compute_digit_logits(
        self,
        hidden_last: torch.Tensor,
        digit: int,
    ) -> torch.Tensor:
        """Shared-embedding dot-product logits for one digit codebook."""
        start = self.sid_offset + digit * self.codebook_size
        end = start + self.codebook_size
        e_sub = self.embedding.weight[start:end]
        h = self.output_adapter(hidden_last)
        return torch.matmul(h, e_sub.t())

    @staticmethod
    def _make_compress_mlp(
        in_dim: int,
        out_dim: int,
        hiddens: list,
    ) -> nn.Module:
        """Linear(in→h1)→ReLU→…→Linear(hk→out). Single Linear if no hiddens."""
        dims = [in_dim] + list(hiddens) + [out_dim]
        layers: list = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        if len(layers) == 1:
            return layers[0]
        return nn.Sequential(*layers)

    def _history_item_embeds(
        self,
        history_tokens: torch.Tensor,
        history_attention_mask: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build encoder history sequence embeddings.

        Default: compress each item's ``n_digit`` SID embeds via ``item_mlp``
        → ``(B, S, d)`` with item-level validity.
        Flat (``mf_flat_digits``): feed digit embeds directly →
        ``(B, S*n_digit, d)``; item-level pos emb is broadcast to that item's
        digits; validity is per digit token.

        When ``mf_rt_condition=add_digits`` and ``r,t`` are given, add the
        timestep embeddings onto each SID digit embedding (before item_mlp
        when not flat).
        """
        batch_size = history_tokens.shape[0]
        n_digit = self.n_digit
        n_items = history_tokens.shape[1] // n_digit
        hist = history_tokens.view(batch_size, n_items, n_digit)
        hist_mask = history_attention_mask.view(batch_size, n_items,
                                                n_digit).bool()
        item_valid = hist_mask.any(dim=-1)

        tok_emb = self.embedding(hist)  # (B, S, n_digit, d)

        pos_ids = torch.arange(n_items, device=history_tokens.device)
        pos_emb = self.pos_emb_enc(pos_ids).unsqueeze(0)  # (1, S, d)

        if self.mf_flat_digits:
            # (B, S, n_digit, d) + broadcast item pos → flatten to digit seq
            digit_emb = tok_emb + pos_emb.unsqueeze(2)
            digit_emb = digit_emb.reshape(batch_size, n_items * n_digit, -1)
            digit_valid = hist_mask.reshape(batch_size, n_items * n_digit)
            digit_emb = digit_emb * digit_valid.unsqueeze(-1).to(
                digit_emb.dtype)
            return digit_emb, digit_valid

        item_emb = self.item_mlp(tok_emb.reshape(batch_size, n_items, -1))
        item_emb = item_emb + pos_emb
        item_emb = item_emb * item_valid.unsqueeze(-1).to(item_emb.dtype)
        return item_emb, item_valid

    def _history_digit_prior_embeds(
        self,
        history_tokens: torch.Tensor,
        history_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean embedding of history tokens at each digit position."""
        n_digit = self.n_digit
        batch_size = history_tokens.shape[0]
        device = history_tokens.device
        n_items = history_tokens.shape[1] // n_digit

        hist = history_tokens.view(batch_size, n_items, n_digit)
        hist_mask = history_attention_mask.view(batch_size, n_items,
                                                n_digit).bool()
        item_valid = hist_mask.any(dim=-1)

        hist_flat = hist.reshape(batch_size, -1)
        hist_emb = self.embedding(hist_flat).view(batch_size, n_items, n_digit,
                                                  -1)
        hist_emb = hist_emb * item_valid[:, :, None, None].to(hist_emb.dtype)

        counts = item_valid.sum(dim=1).clamp(min=1).to(hist_emb.dtype)
        prior = hist_emb.sum(dim=1) / counts[:, None, None]

        no_hist = ~item_valid.any(dim=1)
        if no_hist.any():
            mask_emb = self.mask_emb_table.weight.mean(dim=0, keepdim=True)
            prior = torch.where(no_hist[:, None, None], mask_emb, prior)
        return prior

    def _build_label_embeds(
        self,
        label_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        history_attention_mask: torch.Tensor,
        prior_mask: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        digit_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Digit-level label embeds; optional informative prior for masked slots."""
        label_embeds = self.embedding(label_tokens)
        label_embeds = self._apply_mask_emb_table(
            label_embeds, label_tokens, digit_order=digit_order)

        if (self.mf_informative_prior and prior_mask is not None
                and prior_mask.any()):
            prior = self._history_digit_prior_embeds(history_tokens,
                                                     history_attention_mask)
            label_embeds = torch.where(prior_mask.unsqueeze(-1), prior,
                                       label_embeds)

        # DiffGRM decoder has no absolute digit position embedding.
        return label_embeds

    def _encode(
        self,
        history_tokens: torch.Tensor,
        history_attention_mask: torch.Tensor,
        r: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode history items (no r/t tokens)."""
        hist_emb, item_valid = self._history_item_embeds(
            history_tokens, history_attention_mask, r=r, t=t)
        n_items = hist_emb.shape[1]
        encoder_hidden = self.drop(hist_emb)
        attn_mask = item_valid[:, None, None, :].expand(-1, 1, n_items, -1)
        for block in self.encoder_blocks:
            encoder_hidden = block(encoder_hidden, attention_mask=attn_mask)
        return self.ln_f(encoder_hidden)

    def _decode_to_logits(self, label_embeds: torch.Tensor,
                          encoder_hidden: torch.Tensor,
                          digit_order: Optional[torch.Tensor] = None,
                          ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode label embeds; return (logits, decoder_hidden H).

        Logits are dot products with the shared token embedding; each sequence
        position is filled with only that digit's codebook logits. When digits
        are shuffled, ``digit_order[b, pos]`` selects which original digit
        codebook belongs at ``pos`` (defaults to identity / canonical order).
        """
        x = self.drop(label_embeds)
        for block in self.decoder_blocks:
            x = block(x, encoder_hidden=encoder_hidden)
        x = self.ln_dec(x)
        batch_size, n_digit, _ = x.shape
        logits = torch.full(
            (batch_size, n_digit, self.tokenizer.vocab_size),
            float('-inf'),
            device=x.device,
            dtype=x.dtype,
        )
        if digit_order is None:
            for digit in range(n_digit):
                digit_logits = self._compute_digit_logits(
                    x[:, digit, :], digit)
                start = self.sid_offset + digit * self.codebook_size
                end = start + self.codebook_size
                logits[:, digit, start:end] = digit_logits
        else:
            # digit_order may differ across rows (multi-view chunks).
            for pos in range(n_digit):
                digs = digit_order[:, pos]
                for dig in range(n_digit):
                    sel = digs == dig
                    if not sel.any():
                        continue
                    digit_logits = self._compute_digit_logits(
                        x[sel, pos, :], dig)
                    start = self.sid_offset + dig * self.codebook_size
                    end = start + self.codebook_size
                    logits[sel, pos, start:end] = digit_logits
        return logits, x

    def _to_canonical_digit_order(
        self,
        x: torch.Tensor,
        digit_order: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Scatter ``x`` from position order back to canonical digit order.

        ``digit_order[b, pos]`` = original digit whose content sits at ``pos``.
        Works for ``x`` shaped ``(B, n_digit, ...)``.
        """
        if digit_order is None:
            return x
        out = torch.empty_like(x)
        batch_size, n_digit = digit_order.shape
        batch_idx = torch.arange(
            batch_size, device=x.device).unsqueeze(1).expand(-1, n_digit)
        out[batch_idx, digit_order] = x
        return out

    def _item_mlp_from_sid_tokens(
        self,
        sid_tokens: torch.Tensor,
        detach_token_emb: bool = True,
    ) -> torch.Tensor:
        """Map SID tokens → joint item emb via ``item_mlp``.

        Args:
            sid_tokens: (..., n_digit) token ids (canonical digit order for
                catalog / packed-SID matching).
            detach_token_emb: if True, stop-grad into ``embedding`` (item_mlp
                still trains).
        Returns:
            (..., d_model) joint embeddings.
        """
        tok_emb = self.embedding(sid_tokens)
        if detach_token_emb:
            tok_emb = tok_emb.detach()
        leading = sid_tokens.shape[:-1]
        return self.item_mlp(tok_emb.reshape(*leading, -1))

    # ------------------------------------------------------------------
    # Dual-level (token + item) contrastive ghost.
    # ------------------------------------------------------------------

    @staticmethod
    def _pos_nll(log_probs: torch.Tensor) -> torch.Tensor:
        """NLL of the positive; ``log_probs`` is ``(B, 1+N)``, positive at 0."""
        return -log_probs[:, 0].mean()

    def _dual_user_h(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """Decoder hidden → item-tower user vector ``(B, d)`` (not L2-normalised)."""
        batch_size = decoder_hidden.shape[0]
        return self.item_mlp(
            decoder_hidden.reshape(batch_size, 1, -1)).squeeze(1)

    def _dual_user_emb(self, decoder_hidden: torch.Tensor) -> torch.Tensor:
        """Decoder hidden states → one L2-normalised user vector (B, d)."""
        return F.normalize(self._dual_user_h(decoder_hidden), dim=-1)

    def _dual_item_emb(self, sid_tokens: torch.Tensor) -> torch.Tensor:
        """SID tokens → L2-normalised joint item vectors ``(..., d)``."""
        v = self._item_mlp_from_sid_tokens(sid_tokens, detach_token_emb=True)
        return F.normalize(v, dim=-1)

    def _catalog_item_emb(self) -> torch.Tensor:
        """Item-level embeddings for the whole catalog, ``(N, d)``.

        Recomputed every call while training (the MLP is being updated);
        cached in eval so full-catalog scoring costs one matmul per batch.
        Call ``invalidate_catalog_emb_cache()`` before each eval pass.
        """
        catalog = self.catalog_tokens.to(self.embedding.weight.device)
        if self.training:
            return self._dual_item_emb(catalog)
        if self._catalog_emb_cache is None:
            with torch.no_grad():
                self._catalog_emb_cache = self._dual_item_emb(catalog)
        return self._catalog_emb_cache

    def invalidate_catalog_emb_cache(self) -> None:
        """Drop the cached catalog item embeddings (weights changed)."""
        self._catalog_emb_cache = None

    def _dual_item_scores(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Raw ``cos(u, v)`` from L2-normalised user/item vectors.

        ``mf_dual_item_tau`` is applied at the train/infer softmax.
        """
        return user_emb @ item_emb.transpose(-1, -2)

    def _dual_item_sid_scores(
        self,
        user_emb: torch.Tensor,
        sid_tokens: torch.Tensor,
        *,
        per_example: bool = False,
        joint_item_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Item SID scores from joint MLP embeddings.

        Returns ``(B, 1)`` when ``per_example`` else ``(B, N)``.
        """
        if joint_item_emb is None:
            joint_item_emb = self._dual_item_emb(sid_tokens)
        if per_example:
            return self._dual_item_scores(
                user_emb.unsqueeze(1),
                joint_item_emb.unsqueeze(1)).reshape(user_emb.shape[0], 1)
        return self._dual_item_scores(user_emb, joint_item_emb)

    def _dual_mix_log_probs(
        self,
        s_tok: torch.Tensor,
        s_item: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-branch log-probs and their unnormalised mix.

        ``log_mixed = log(P_tok + w·P_item)`` with the fixed item weight ``w``
        (not divided by ``1 + w``). Returns ``(log_mixed, log_p_tok,
        log_p_item)``, all ``(..., n_cand)``.
        """
        log_p_tok = F.log_softmax(s_tok, dim=-1)
        log_p_item = F.log_softmax(
            s_item / max(self.mf_dual_item_tau, 1e-6), dim=-1)

        finite = torch.isfinite(log_p_tok) & torch.isfinite(log_p_item)
        p_tok = torch.where(finite, log_p_tok.exp(),
                            torch.zeros_like(log_p_tok))
        p_item = torch.where(finite, log_p_item.exp(),
                             torch.zeros_like(log_p_item))
        mixed = p_tok + self.dual_item_weight * p_item
        mixed = torch.where(finite, mixed.clamp(min=1e-12),
                            torch.zeros_like(mixed))
        log_mixed = torch.log(mixed.clamp(min=1e-12))
        log_mixed = torch.where(
            finite, log_mixed,
            torch.full_like(log_mixed, float('-inf')))
        return log_mixed, log_p_tok, log_p_item

    def _dual_level_candidate_scores(
        self,
        log_probs: torch.Tensor,
        decoder_hidden: torch.Tensor,
        target_label_tokens: torch.Tensor,
        digit_order: Optional[torch.Tensor] = None,
        digit_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token- and item-level scores over ``[target | negatives]``.

        ``log_probs`` must already be in canonical digit order. Both branches
        always get the SAME candidate set and the SAME ``-inf`` mask, so the
        mixture stays a distribution over one support.

        When ``digit_mask`` is set, token-level scores sum log-probs only over
        masked digits. Item-level is the joint ``cos(u, v_c)``.

        Negatives are unique real-item SIDs. If ``mf_ghost_penalty_num_neg >
        0``, ``randint``-sample that many per row first, then score only those
        candidates (the positive's own SID masked); item scores gather from one
        catalog emb table. Otherwise score the full catalog.

        Returns ``(s_tok, s_item)``, each ``(B, 1 + N')``.
        """
        device = log_probs.device

        # -- positives ------------------------------------------------------
        pos_tok = self._score_sids_from_log_probs(
            log_probs, target_label_tokens, per_example=True,
            digit_mask=digit_mask)  # (B,)
        hidden_c = self._to_canonical_digit_order(decoder_hidden, digit_order)
        user_emb = self._dual_user_emb(hidden_c)  # (B, d)
        tgt_emb = self._dual_item_emb(target_label_tokens)  # (B, d)
        pos_item = self._dual_item_sid_scores(
            user_emb, target_label_tokens, per_example=True,
            joint_item_emb=tgt_emb)

        # -- catalog negatives ----------------------------------------------
        catalog = self.catalog_tokens.to(device)  # (N, n_digit)
        n_neg = int(self.mf_ghost_penalty_num_neg)
        n_catalog = int(catalog.shape[0])

        # Fast path: randint (B,k) indices, score only those candidates.
        # Avoids (B,N) rand+topk and avoids re-running item MLP on B*k
        # SIDs (gather from one catalog emb table instead).
        if 0 < n_neg < n_catalog:
            top_idx, too_near = self._sample_catalog_neg_indices(
                target_label_tokens, catalog, n_neg)
            cand = catalog[top_idx]  # (B, k, n_digit)
            neg_tok = self._score_sids_from_log_probs(
                log_probs, cand, digit_mask=digit_mask)
            cat_joint = self._catalog_item_emb()  # (N, d)
            neg_emb = cat_joint[top_idx]  # (B, k, d)
            neg_item = self._dual_item_scores(
                user_emb.unsqueeze(1), neg_emb).squeeze(1)
        else:
            # Full catalog: score all SIDs (n_neg<=0 or n_neg>=N).
            neg_tok = self._score_sids_from_log_probs(
                log_probs, catalog, digit_mask=digit_mask)
            neg_item = self._dual_item_sid_scores(
                user_emb, catalog, joint_item_emb=self._catalog_item_emb())
            # Packed equality avoids (B,N,n_digit) hamming (OOM on
            # large catalogs, e.g. Electronics).
            cat_packed = self.catalog_packed.to(device)
            tgt_packed = self._pack_sids(target_label_tokens)
            too_near = (cat_packed.unsqueeze(0)
                        == tgt_packed.unsqueeze(1))

        neg_tok = neg_tok.masked_fill(too_near, float('-inf'))
        neg_item = neg_item.masked_fill(too_near, float('-inf'))

        s_tok = torch.cat([pos_tok.unsqueeze(1), neg_tok], dim=1)
        s_item = torch.cat([pos_item, neg_item], dim=1)
        return s_tok, s_item

    def _dual_ghost_loss(
        self,
        logits: torch.Tensor,
        decoder_hidden: torch.Tensor,
        target_label_tokens: torch.Tensor,
        label_attention_mask: Optional[torch.Tensor] = None,
        digit_order: Optional[torch.Tensor] = None,
        masked_indices: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict]:
        """Dual-level contrastive loss ``L = NLL(P_tok) + NLL(P_item)``.

        Both branches are softmaxes over ``[target | catalog negatives]``.
        The unnormalised mix NLL ``-log(P_tok(y) + w·P_item(y))`` is only
        logged.

        Returns ``(loss, logs)``; ``logs`` holds the detached breakdown.
        """
        # Canonical digit order, matching ``catalog_tokens``.
        batch_size, n_digit = logits.shape[0], logits.shape[1]
        device = logits.device
        if digit_order is not None:
            canon_logits = torch.empty_like(logits)
            batch_idx = torch.arange(
                batch_size, device=device).unsqueeze(1).expand(-1, n_digit)
            canon_logits[batch_idx, digit_order] = logits
            logits = canon_logits
            if masked_indices is not None:
                mask_c = torch.zeros_like(masked_indices)
                mask_c[batch_idx, digit_order] = masked_indices
                masked_indices = mask_c
            if label_attention_mask is not None:
                lam = torch.zeros_like(label_attention_mask)
                lam[batch_idx, digit_order] = label_attention_mask
                label_attention_mask = lam
        log_probs = F.log_softmax(logits, dim=-1)

        # Token-level scores only on masked digits (CE convention).
        if label_attention_mask is None:
            label_attention_mask = torch.ones(
                batch_size, n_digit, dtype=torch.bool, device=device)
        if masked_indices is None:
            digit_mask = label_attention_mask.bool()
        else:
            digit_mask = masked_indices.bool() & label_attention_mask.bool()

        s_tok, s_item = self._dual_level_candidate_scores(
            log_probs, decoder_hidden, target_label_tokens,
            digit_order=digit_order, digit_mask=digit_mask)

        valid = label_attention_mask.all(dim=-1) & digit_mask.any(dim=-1)
        if not valid.any():
            return logits.new_zeros(()), {}
        s_tok, s_item = s_tok[valid], s_item[valid]
        log_mixed, log_p_tok, log_p_item = self._dual_mix_log_probs(
            s_tok, s_item)

        l_mix = self._pos_nll(log_mixed)
        l_tok = self._pos_nll(log_p_tok)
        l_item = self._pos_nll(log_p_item)
        loss = l_tok + l_item
        logs = {
            'mix': l_mix.detach(),
            'tok': l_tok.detach(),
            'item': l_item.detach(),
            'alpha': self.dual_item_weight.detach().mean(),
        }
        return loss, logs

    # ------------------------------------------------------------------
    # Jump-composition consistency. Opt-in via mf_jump_consist_weight.
    # ------------------------------------------------------------------

    def _sample_catalog_neg_indices(
        self,
        target_label_tokens: torch.Tensor,
        catalog: torch.Tensor,
        n_neg: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Uniform catalog indices via ``randint`` — O(B·k), not O(B·N).

        With-replacement draws from ``[0, N)``. Slots that hit the positive
        SID are resampled a few times, then any residual collisions are
        marked ``invalid`` (caller masks to ``-inf``).

        Returns:
            top_idx: ``(B, k)`` long indices into ``catalog``
            invalid: ``(B, k)`` bool
        """
        batch_size = target_label_tokens.shape[0]
        device = target_label_tokens.device
        n_catalog = int(catalog.shape[0])
        k = min(n_neg, n_catalog)
        with torch.no_grad():
            top_idx = torch.randint(
                0, n_catalog, (batch_size, k), device=device)
            cat_packed = self.catalog_packed.to(device)
            tgt_packed = self._pack_sids(target_label_tokens)
            for _ in range(3):
                hit = cat_packed[top_idx] == tgt_packed.unsqueeze(1)
                if not bool(hit.any()):
                    break
                top_idx = torch.where(
                    hit,
                    torch.randint(
                        0, n_catalog, (batch_size, k), device=device),
                    top_idx)
            invalid = cat_packed[top_idx] == tgt_packed.unsqueeze(1)
        return top_idx, invalid

    def _forward_with_time_condition(
        self,
        input_tokens: torch.Tensor,
        input_attention_mask: torch.Tensor,
        label_tokens: torch.Tensor,
        label_attention_mask: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        prior_mask: Optional[torch.Tensor] = None,
        return_hidden: bool = False,
        digit_order: Optional[torch.Tensor] = None,
    ):
        encoder_hidden = self._encode(
            input_tokens,
            input_attention_mask,
            r=r,
            t=t,
        )
        label_embeds = self._build_label_embeds(
            label_tokens,
            input_tokens,
            input_attention_mask,
            prior_mask=prior_mask,
            r=r,
            t=t,
            digit_order=digit_order,
        )
        logits, hidden = self._decode_to_logits(
            label_embeds,
            encoder_hidden,
            digit_order=digit_order,
        )
        if return_hidden:
            return logits, hidden
        return logits

    def mean_flow_loss(
        self,
        input_tokens,
        input_attention_mask,
        label_tokens,
        label_attention_mask,
    ):
        """Returns (total, ghost_raw, ghost_weighted, dual_logs)."""
        batch_size = label_tokens.shape[0]
        device = label_tokens.device

        canonical_label_tokens = label_tokens

        digit_order = None  # digits always stay in canonical order

        t, r = sample_tr(
            batch_size,
            device,
            data_proportion=self.mf_data_proportion,
            noise_dist='uniform',
            t_min_gap=self.mf_t_min_gap,
        )
        h = (t - r).clamp(min=0.0)

        masked_label_tokens, masked_indices, p_mask = discrete_mask_at_r(
            label_tokens,
            label_attention_mask,
            r,
            self.mask_token_id,
        )

        prior_mask = masked_indices if self.mf_informative_prior else None

        logits, hidden = self._forward_with_time_condition(
            input_tokens,
            input_attention_mask,
            masked_label_tokens,
            label_attention_mask,
            r,
            t,
            prior_mask=prior_mask,
            return_hidden=True,
            digit_order=digit_order,
        )
        logits = self._constrain_label_logits(logits, digit_order=digit_order)

        # `logits` carries -inf on illegal digits, so never seed the total
        # from it (-inf * 0 = nan); start from a clean zero scalar.
        loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
        # Token level (= the ghost score) + item level, weight 1.
        ghost_raw, dual_logs = self._dual_ghost_loss(
            logits,
            hidden,
            canonical_label_tokens,
            label_attention_mask,
            digit_order=digit_order,
            masked_indices=masked_indices,
        )
        loss = loss + ghost_raw

        return loss, ghost_raw, ghost_raw, dict(dual_logs)

    def forward(self, batch: dict):
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_tokens = input_tokens.reshape((input_tokens.shape[0], -1))
        input_attention_mask = input_tokens != 0

        assert 'labels' in batch, 'The batch must contain the labels.'
        label_tokens = self.item_id2tokens[batch['labels'].squeeze(1)]
        label_tokens = label_tokens.reshape((label_tokens.shape[0], -1))
        label_attention_mask = torch.ones_like(
            label_tokens, device=label_tokens.device).bool()

        (total_loss, ghost_loss, ghost_weighted,
         dual_logs) = self.mean_flow_loss(
             input_tokens, input_attention_mask, label_tokens,
             label_attention_mask)

        return SPRINTOutput(
            loss=total_loss,
            ghost_loss=ghost_loss,
            ghost_weighted=ghost_weighted,
            dual_mix_loss=dual_logs.get('mix'),
            dual_tok_loss=dual_logs.get('tok'),
            dual_item_loss=dual_logs.get('item'),
            dual_alpha=dual_logs.get('alpha'),
        )

    def _constrain_label_logits(
        self,
        logits: torch.Tensor,
        digit_order: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Restrict each label position to its digit codebook vocab.

        Args:
            logits: (B, n_digit, vocab)
            digit_order: optional (B, n_digit); digit_order[b, pos] is the
                original digit whose valid tokens are allowed at ``pos``.
                Defaults to identity (canonical order).
        """
        constrained = torch.full_like(logits, float('-inf'))
        batch_size, n_digit, _ = logits.shape
        if digit_order is None:
            for pos in range(n_digit):
                valid = self.posValidTokens[pos]
                valid = valid[valid != 0]
                constrained[:, pos, valid] = logits[:, pos, valid]
            return constrained

        # digit_order[b, pos] -> which original digit's vocab sits at pos
        selected = self.posValidTokens[digit_order]  # (B, n_digit, M)
        valid_mask = selected != 0
        b_idx = torch.arange(
            batch_size, device=logits.device)[:, None, None].expand_as(selected)
        p_idx = torch.arange(
            n_digit, device=logits.device)[None, :, None].expand_as(selected)
        constrained[b_idx[valid_mask], p_idx[valid_mask],
                    selected[valid_mask]] = logits[b_idx[valid_mask],
                                                  p_idx[valid_mask],
                                                  selected[valid_mask]]
        return constrained

    def _pack_sids(self, sids: torch.Tensor) -> torch.Tensor:
        """Pack (..., n_digit) SID tokens into a unique int64 key."""
        base = int(self.tokenizer.vocab_size)
        packed = torch.zeros(sids.shape[:-1],
                             dtype=torch.long,
                             device=sids.device)
        for d in range(sids.shape[-1]):
            packed = packed * base + sids[..., d]
        return packed

    def _topk_catalog_sids_from_logits(
        self,
        logits: torch.Tensor,
        k: int,
        decoder_hidden: Optional[torch.Tensor] = None,
        r: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
        fixed_tokens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank only real-item SIDs.

        With ``decoder_hidden`` given, the score is ``P_tok + w·P_item`` over
        the catalog; otherwise the token-only ``Σ_d log p_d``.

        ``fixed_tokens`` (B, n_digit): non-``[MASK]`` positions must match the
        catalog SID (used by multi-step decode after committing some digits).
        """
        log_probs = F.log_softmax(logits, dim=-1)
        catalog = self.catalog_tokens.to(logits.device)  # (N, n_digit)
        n_items = catalog.shape[0]

        scores = self._score_sids_from_log_probs(log_probs, catalog)

        if decoder_hidden is not None:
            user_emb = self._dual_user_emb(decoder_hidden)
            item_scores = self._dual_item_scores(
                user_emb, self._catalog_item_emb().to(logits.device))
            p_tok = F.softmax(scores, dim=-1)
            p_item = F.softmax(
                item_scores / max(self.mf_dual_item_tau, 1e-6), dim=-1)
            scores = p_tok + self.dual_item_weight * p_item

        if fixed_tokens is not None:
            # (B, 1, D) vs (1, N, D): keep catalog rows matching committed digits.
            fixed = fixed_tokens.unsqueeze(1)
            is_fixed = fixed != self.mask_token_id
            match = (~is_fixed) | (catalog.unsqueeze(0) == fixed)
            scores = scores.masked_fill(~match.all(dim=-1), float('-inf'))

        k = min(k, n_items)
        top_scores, top_idx = torch.topk(scores, k, dim=1)
        label_ids = catalog[top_idx]  # (B, k, n_digit)
        return label_ids, top_scores

    def generate(self, batch, n_return_sequences=1):
        outputs = self.generate_sids(batch, N_CANDIDATES, n_return_sequences)
        # The catalog ranker already returns legal SIDs, ordered.
        return outputs[:, :n_return_sequences]

    def generate_sids(self, batch, n_candidates, top_k):
        """One-step masked-diffusion decode.

        All four digits come from a single forward at (r=0, t=1); the catalog
        ranker then returns the ``max(n_candidates, top_k)`` best legal SIDs.
        There is no autoregressive search: ``n_candidates`` only widens the
        intermediate top-k, it does not change which items rank highest.
        """

        input_ids: torch.Tensor = self.item_id2tokens[batch['input_ids']]
        input_ids = input_ids.reshape((input_ids.shape[0], -1))
        attention_mask: torch.Tensor = input_ids != 0

        batch_size = input_ids.shape[0]
        n_digit = self.tokenizer.n_digit
        device = input_ids.device

        r_schedule, h_schedule = get_inference_time_schedule(1, device)

        label_ids = torch.full(
            (batch_size, n_digit),
            self.mask_token_id,
            device=device,
            dtype=torch.long,
        )
        label_attention_mask = torch.ones(
            (batch_size, n_digit), device=device, dtype=torch.bool)

        r_batch = r_schedule[0].expand(batch_size)
        t_batch = (r_schedule[0] + h_schedule[0]).expand(batch_size)

        # The item-level branch scores with the decoder hidden state.
        logits, decoder_hidden = self._forward_with_time_condition(
            input_ids,
            attention_mask,
            label_ids,
            label_attention_mask,
            r_batch,
            t_batch,
            return_hidden=True,
        )
        logits = self._constrain_label_logits(logits)
        k = max(n_candidates, top_k)
        # Rank the full real-item catalog by per-digit log-prob sum: exact MAP
        # over valid SIDs, so every returned item is legal.
        label_ids, _ = self._topk_catalog_sids_from_logits(
            logits, k, decoder_hidden=decoder_hidden, r=r_batch, t=t_batch)
        return label_ids

    def log(self, message, level='info'):
        return log(message,
                   self.config['accelerator'],
                   self.logger,
                   level=level)
