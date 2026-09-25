"""Discrete Mean Flow: continuous time r,t in (0, 1) with mask probability 1 - r."""

import math
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn


class TimestepEmbedder(nn.Module):
    """Sinusoidal time embedding + MLP (ELF-style)."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(
        t: torch.Tensor,
        dim: int,
        max_period: float = 10000.0,
    ) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, device=t.device, dtype=t.dtype)
            / half,
        )
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


def mask_prob_at_r(
    r: torch.Tensor,
    min_mask_prob: float = 1e-4,
) -> torch.Tensor:
    """Mask probability at time r: r=0 -> full mask, r=1 -> no mask."""
    return (1.0 - r).clamp(min=min_mask_prob, max=1.0)


def sample_tr(
    batch_size: int,
    device: torch.device,
    data_proportion: float = 0.75,
    noise_dist: Literal["logit_normal", "uniform"] = "uniform",
    p_mean: float = -0.4,
    p_std: float = 1.0,
    t_min_gap: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample (t, r) with t >= r in [0, 1].

    For a fraction ``data_proportion`` of the batch, force r = 0 and t = 1
    (fully masked); the rest keep sampled r, t with t >= r.

    ``noise_dist='uniform'``: r ~ Uniform[0, 1], then t ~ Uniform[r, 1].
    ``noise_dist='logit_normal'``: two logit-normal draws, ordered so t >= r.

    If ``t_min_gap`` > 0, enforce t ∈ [r + t_min_gap, 1] by clamping
    r ≤ 1 - t_min_gap and resampling t ~ Uniform[r + t_min_gap, 1]
    (applied after the full-mask override as well, so r=0,t=1 stays valid).
    """
    if noise_dist == "logit_normal":
        rnd = torch.randn(batch_size, device=device, dtype=torch.float32)
        samples = torch.sigmoid(rnd * p_std + p_mean)
        t = samples
        rnd_r = torch.randn(batch_size, device=device, dtype=torch.float32)
        r = torch.sigmoid(rnd_r * p_std + p_mean)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
    elif noise_dist == "uniform":
        # True Uniform[0,1] on r (not min of two uniforms).
        r = torch.rand(batch_size, device=device, dtype=torch.float32)
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        t = r + u * (1.0 - r)
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")

    gap = float(t_min_gap)
    if gap > 0:
        if gap >= 1.0:
            raise ValueError(f"t_min_gap must be in [0, 1), got {gap}")
        # Keep a non-empty interval [r+gap, 1].
        r = torch.minimum(r, torch.full_like(r, 1.0 - gap))
        t_low = r + gap
        u = torch.rand(batch_size, device=device, dtype=torch.float32)
        t = t_low + u * (1.0 - t_low)

    # Full-mask override last so r=0,t=1 is preserved (gap=1 >= t_min_gap).
    data_size = int(batch_size * data_proportion)
    if data_size > 0:
        full_mask = torch.arange(batch_size, device=device) < data_size
        r = torch.where(full_mask, torch.zeros_like(r), r)
        t = torch.where(full_mask, torch.ones_like(t), t)

    return t, r


def discrete_mask_at_r(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    r: torch.Tensor,
    mask_token_id: int,
    min_mask_prob: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask each valid token independently with probability 1 - r."""
    batch_size, seq_len = input_ids.shape
    mask_prob = mask_prob_at_r(r, min_mask_prob=min_mask_prob)
    mask_prob = mask_prob.view(batch_size, 1).expand(batch_size, seq_len)
    masked_indices = torch.bernoulli(mask_prob).to(dtype=torch.bool)
    masked_indices &= attention_mask.bool()

    noisy_ids = input_ids.clone()
    noisy_ids[masked_indices] = mask_token_id
    noisy_ids = noisy_ids.masked_fill(~attention_mask, 0)
    masked_indices = masked_indices.masked_fill(~attention_mask, False)
    return noisy_ids, masked_indices, mask_prob


def get_inference_time_schedule(
    num_sampling_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inference schedule on r in [0, 1]: r=0 is fully masked, r=1 is clean.

    Returns per-step (r, h) with h = delta r to the next boundary.
    """
    if num_sampling_steps == 1:
        r_vals = torch.zeros(1, device=device, dtype=torch.float32)
        h_vals = torch.ones(1, device=device, dtype=torch.float32)
        return r_vals, h_vals

    boundaries = torch.linspace(
        0.0,
        1.0,
        num_sampling_steps + 1,
        device=device,
        dtype=torch.float32,
    )
    r_vals = boundaries[:-1]
    h_vals = boundaries[1:] - boundaries[:-1]
    return r_vals, h_vals

