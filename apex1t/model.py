"""End-to-end Apex-1T model: token embedding -> transformer block stack ->
final norm -> output head. A single class, parameterized entirely by
`ModelConfig` — the same code instantiates both `DEBUG_CONFIG` (in real
memory) and `FULL_CONFIG` (on `torch.device("meta")`, since ~1T parameters
does not fit in this machine's memory).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from apex1t.block import TransformerBlock
from apex1t.config import ModelConfig
from apex1t.norm import RMSNorm


class Apex1T(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            TransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        )
        self.final_norm = RMSNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return self.output_head(x)
