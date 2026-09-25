"""DiffGRM-style encoder–decoder Transformer blocks (bidirectional + cross-attn)."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(norm_type: str, dim: int, eps: float):
    if (norm_type or 'layernorm').lower() == 'rmsnorm':
        return nn.RMSNorm(dim, eps=eps)
    return nn.LayerNorm(dim, eps=eps)


class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, n_head, attn_drop=0.1, resid_drop=0.1):
        super().__init__()
        assert emb_dim % n_head == 0
        self.n_head = n_head
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // n_head

        self.qkv = nn.Linear(emb_dim, 3 * emb_dim, bias=False)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.attn_dropout = nn.Dropout(attn_drop)
        self.resid_dropout = nn.Dropout(resid_drop)

        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self,
                x,
                attention_mask=None,
                key_value=None,
                kv_prefix: Optional[torch.Tensor] = None,
                q_prefix: Optional[torch.Tensor] = None,
                q_bias: Optional[torch.Tensor] = None):
        """Self- or cross-attn; optional K/V or Q-side r/t prefixes.

        Args:
            x: (B, T, C) query content.
            attention_mask: optional bool/float mask broadcastable to
                (B, 1, T_q, T_kv_content) for the content K/V only; prefix
                columns are auto-appended as fully visible when
                ``kv_prefix`` is set.
            key_value: optional (B, T_kv, 2C) precomputed [K|V] for cross-attn.
            kv_prefix: optional (B, P, C) tokens projected into K/V and
                prepended before content keys/values. Queries stay content-only.
            q_prefix: optional (B, P, C) tokens projected into Q-space, mean-
                pooled over P, and added to every content query (modulates
                attention weights without expanding K/V).
            q_bias: optional (B, C) or (B, 1, C) added to queries after Q
                projection (broadcast over query length).
        """
        B, T, C = x.size()

        if key_value is not None:
            # Cross-attn: Q from x; K,V from precomputed [K|V] = 2*emb_dim.
            # Callers that want KV prefixes should prepend them onto
            # ``key_value`` / encoder states themselves (DiffGRM-style).
            q = self.qkv(x)[:, :, :self.emb_dim]
            k, v = key_value.chunk(2, dim=-1)
            T_kv = k.size(1)
            if attention_mask is not None and attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
        else:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            T_kv = T
            if kv_prefix is not None:
                # Self-attn: project prefixes and prepend only to K/V.
                pref_qkv = self.qkv(kv_prefix)
                _, k_p, v_p = pref_qkv.chunk(3, dim=-1)
                k = torch.cat([k_p, k], dim=1)
                v = torch.cat([v_p, v], dim=1)
                if attention_mask is not None:
                    if attention_mask.dim() == 3:
                        attention_mask = attention_mask.unsqueeze(1)
                    p = kv_prefix.size(1)
                    pref_mask = attention_mask.new_ones(
                        attention_mask.size(0), attention_mask.size(1),
                        attention_mask.size(2), p)
                    attention_mask = torch.cat([pref_mask, attention_mask],
                                               dim=-1)
                T_kv = k.size(1)

        if q_prefix is not None:
            # Dual of kv_prefix: inject r/t into Query only (pooled add).
            pref_q = self.qkv(q_prefix)[:, :, :self.emb_dim]
            q = q + pref_q.mean(dim=1, keepdim=True)

        if q_bias is not None:
            if q_bias.dim() == 2:
                q_bias = q_bias.unsqueeze(1)
            q = q + q_bias

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, self.head_dim).transpose(1, 2)

        scale = 1.0 / (self.head_dim**0.5)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)
            att = att.masked_fill(attention_mask == 0, float('-inf'))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = torch.matmul(att, v)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):

    def __init__(self, emb_dim, n_inner, resid_drop=0.1, act='gelu'):
        super().__init__()
        self.c_fc = nn.Linear(emb_dim, n_inner)
        self.c_proj = nn.Linear(n_inner, emb_dim)
        self.dropout = nn.Dropout(resid_drop)
        self.act = F.gelu if act == 'gelu' else F.relu

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class EncoderBlock(nn.Module):

    def __init__(self,
                 emb_dim,
                 n_head,
                 n_inner,
                 attn_drop=0.1,
                 resid_drop=0.1,
                 act='gelu',
                 norm_type='layernorm',
                 norm_eps=1e-5):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.attn = MultiHeadAttention(emb_dim, n_head, attn_drop, resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self,
                x,
                attention_mask=None,
                kv_prefix: Optional[torch.Tensor] = None,
                q_prefix: Optional[torch.Tensor] = None):
        x = x + self.attn(self.ln_1(x),
                          attention_mask=attention_mask,
                          kv_prefix=kv_prefix,
                          q_prefix=q_prefix)
        x = x + self.mlp(self.ln_2(x))
        return x


class DecoderBlock(nn.Module):
    """Bidirectional self-attn + cross-attn (no causal mask), DiffGRM-style."""

    def __init__(self,
                 emb_dim,
                 n_head,
                 n_inner,
                 attn_drop=0.1,
                 resid_drop=0.1,
                 act='gelu',
                 norm_type='layernorm',
                 norm_eps=1e-5):
        super().__init__()
        self.ln_1 = make_norm(norm_type, emb_dim, norm_eps)
        self.self_attn = MultiHeadAttention(emb_dim, n_head, attn_drop,
                                            resid_drop)
        self.ln_2 = make_norm(norm_type, emb_dim, norm_eps)
        self.cross_attn = MultiHeadAttention(emb_dim, n_head, attn_drop,
                                             resid_drop)
        self.ln_3 = make_norm(norm_type, emb_dim, norm_eps)
        self.mlp = FeedForward(emb_dim, n_inner, resid_drop, act)

    def forward(self,
                x,
                encoder_hidden=None,
                self_attention_mask=None,
                kv_prefix: Optional[torch.Tensor] = None,
                q_prefix: Optional[torch.Tensor] = None,
                q_bias: Optional[torch.Tensor] = None):
        x = x + self.self_attn(self.ln_1(x),
                               attention_mask=self_attention_mask,
                               kv_prefix=kv_prefix,
                               q_prefix=q_prefix)
        if encoder_hidden is not None:
            # DiffGRM: reuse encoder states directly as K/V (concat).
            # Optional r/t tokens are prepended onto the K/V sequence only.
            # Optional q_prefix / q_bias modulate cross-attn queries.
            enc_for_kv = encoder_hidden
            if kv_prefix is not None:
                enc_for_kv = torch.cat([kv_prefix, enc_for_kv], dim=1)
            encoder_kv = torch.cat([enc_for_kv, enc_for_kv], dim=-1)
            x = x + self.cross_attn(self.ln_2(x),
                                    key_value=encoder_kv,
                                    q_prefix=q_prefix,
                                    q_bias=q_bias)
        x = x + self.mlp(self.ln_3(x))
        return x
