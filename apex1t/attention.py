"""Multi-Head Latent Attention (MLA) with full decoupled RoPE.

DeepSeek-V3-style attention: queries and keys/values are each routed through
a low-rank down-projection ("compression") before being expanded back up per
head, cutting the KV-cache footprint in real deployments. Positional
information is carried by a *separate* rotary path decoupled from the
compressed content path:

  - Queries get a per-head rotary component (`q_rope`), produced by the same
    up-projection as the content component (`q_nope`).
  - Keys get a single rotary component (`k_rope`) shared/broadcast across all
    heads, produced directly by the down-projection (not per-head) — this is
    what makes the compressed KV cache cheap: only `kv_lora_rank +
    qk_rope_head_dim` values need to be cached per token, not a full
    per-head key.

The five linear projections here (q_down, q_up, kv_down, kv_up, out) are
exactly the ones counted by `apex1t.param_counter._mla_attention_params` —
no bias terms and no internal normalization layers, so a real instantiated
module's parameter count matches the analytical counter exactly (needed for
ticket 07's cross-check).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from apex1t.config import ModelConfig


def _rope_cache(
    seq_len: int, rope_dim: int, base: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for rotary embeddings, shape (seq_len, rope_dim)."""
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, device=device, dtype=torch.float32) / rope_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, rope_dim / 2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, rope_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (..., seq_len, rope_dim); cos/sin broadcastable to the same shape."""
    return x * cos + _rotate_half(x) * sin


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: ModelConfig, rope_base: float = 10_000.0):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.nope_dim = config.qk_nope_head_dim
        self.rope_dim = config.qk_rope_head_dim
        self.v_dim = config.v_head_dim
        self.rope_base = rope_base

        self.q_down_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_up_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * (self.nope_dim + self.rope_dim), bias=False
        )
        self.kv_down_proj = nn.Linear(
            config.hidden_size, config.kv_lora_rank + self.rope_dim, bias=False
        )
        self.kv_up_proj = nn.Linear(
            config.kv_lora_rank, self.num_heads * (self.nope_dim + self.v_dim), bias=False
        )
        self.out_proj = nn.Linear(self.num_heads * self.v_dim, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        cos, sin = _rope_cache(seq_len, self.rope_dim, self.rope_base, x.device, x.dtype)
        cos = cos.view(1, 1, seq_len, self.rope_dim)
        sin = sin.view(1, 1, seq_len, self.rope_dim)

        # --- query path: per-head content (nope) + per-head rotary (rope) ---
        q_compressed = self.q_down_proj(x)
        q_full = self.q_up_proj(q_compressed)
        q_full = q_full.view(batch, seq_len, self.num_heads, self.nope_dim + self.rope_dim)
        q_full = q_full.transpose(1, 2)  # (batch, heads, seq, nope + rope)
        q_nope, q_rope = q_full.split([self.nope_dim, self.rope_dim], dim=-1)
        q_rope = _apply_rope(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # --- key/value path: per-head content (nope) + shared rotary (rope) ---
        kv_combined = self.kv_down_proj(x)
        kv_compressed, k_rope_shared = kv_combined.split(
            [self.config.kv_lora_rank, self.rope_dim], dim=-1
        )
        k_rope_shared = k_rope_shared.view(batch, 1, seq_len, self.rope_dim)
        k_rope_shared = _apply_rope(k_rope_shared, cos, sin)
        k_rope_shared = k_rope_shared.expand(-1, self.num_heads, -1, -1)

        kv_full = self.kv_up_proj(kv_compressed)
        kv_full = kv_full.view(batch, seq_len, self.num_heads, self.nope_dim + self.v_dim)
        kv_full = kv_full.transpose(1, 2)  # (batch, heads, seq, nope + v)
        k_nope, v = kv_full.split([self.nope_dim, self.v_dim], dim=-1)

        k = torch.cat([k_nope, k_rope_shared], dim=-1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(batch, seq_len, self.num_heads * self.v_dim)

        return self.out_proj(attn_out)
