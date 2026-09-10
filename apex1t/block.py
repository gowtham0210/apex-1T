"""Transformer block: pre-norm MLA attention + pre-norm FFN, residual around
each sublayer. The FFN is a plain dense SwiGLU for the first
`config.n_dense_layers` layers (the "dense stem") and the sparse MoE for
every layer after that, per `layer_idx`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apex1t.attention import MultiHeadLatentAttention
from apex1t.config import ModelConfig
from apex1t.ffn import SwiGLUFFN
from apex1t.moe import SparseMoE
from apex1t.norm import RMSNorm


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        if not 0 <= layer_idx < config.n_layers:
            raise ValueError(
                f"layer_idx must be in [0, {config.n_layers}), got {layer_idx}"
            )

        self.layer_idx = layer_idx
        self.use_moe = layer_idx >= config.n_dense_layers

        self.attn_norm = RMSNorm(config.hidden_size)
        self.attention = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
        self.ffn: nn.Module = (
            SparseMoE(config)
            if self.use_moe
            else SwiGLUFFN(config.hidden_size, config.dense_intermediate_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
