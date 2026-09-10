"""SwiGLU feed-forward block.

Shared by the MoE module's experts (ticket 04) and the dense-stem layers
that later get wired in during transformer block assembly (ticket 05) — one
building block, reused rather than duplicated. Three linear projections, no
bias, matching `apex1t.param_counter._swiglu_ffn_params` exactly (3 *
hidden_size * intermediate_size).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
