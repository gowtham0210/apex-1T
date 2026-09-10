"""Sparse Mixture-of-Experts feed-forward layer with auxiliary-loss-free,
bias-based load balancing (DeepSeek-V3 style).

Every token passes through every shared expert (always on) plus its top-k
selected routed experts. Routing is chosen by adding a per-expert bias to
the router's logits before top-k selection — but the *weighting* of the
selected experts' outputs uses the original, unbiased router probabilities,
so the bias only steers *which* experts get picked, not how much a token
trusts the one it picked.

The bias itself is not a gradient parameter: it is a plain buffer, nudged by
a small fixed step after each forward pass based on whether an expert was
over- or under-loaded relative to the average, exactly the rule DeepSeek-V3
uses instead of an auxiliary load-balancing loss term.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from apex1t.config import ModelConfig
from apex1t.ffn import SwiGLUFFN


class SparseMoE(nn.Module):
    def __init__(self, config: ModelConfig, bias_update_rate: float = 1e-3):
        super().__init__()
        self.config = config
        self.top_k = config.top_k
        self.n_routed_experts = config.n_routed_experts
        self.bias_update_rate = bias_update_rate

        self.shared_experts = nn.ModuleList(
            SwiGLUFFN(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_shared_experts)
        )
        self.routed_experts = nn.ModuleList(
            SwiGLUFFN(config.hidden_size, config.moe_intermediate_size)
            for _ in range(config.n_routed_experts)
        )
        self.router = nn.Linear(config.hidden_size, config.n_routed_experts, bias=False)

        # Auxiliary-loss-free load-balancing bias — a buffer, not a
        # Parameter: it never receives a gradient, only the fixed-step
        # update in `_update_routing_bias`.
        self.register_buffer("routing_bias", torch.zeros(config.n_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)
        n_tokens = x_flat.shape[0]

        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x_flat)

        router_logits = self.router(x_flat)  # (tokens, n_routed_experts)
        routing_probs = F.softmax(router_logits, dim=-1)

        # Bias only affects *which* experts are picked (top-k selection),
        # not the weight assigned to whichever ones are picked.
        biased_logits = router_logits + self.routing_bias.unsqueeze(0)
        _, topk_idx = biased_logits.topk(self.top_k, dim=-1)  # (tokens, top_k)
        topk_probs = torch.gather(routing_probs, 1, topk_idx)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        load_count = torch.zeros(self.n_routed_experts, device=x.device)

        for expert_id, expert in enumerate(self.routed_experts):
            # A given expert appears at most once per token's top-k row, so
            # summing the mask across the top-k axis yields either 0 (not
            # selected) or that token's weight for this expert.
            selection_mask = topk_idx == expert_id  # (tokens, top_k) bool
            token_mask = selection_mask.any(dim=-1)  # (tokens,)
            if not torch.any(token_mask):
                continue

            weight = (selection_mask.to(topk_probs.dtype) * topk_probs).sum(dim=-1)  # (tokens,)
            expert_out = expert(x_flat[token_mask])
            routed_out[token_mask] = routed_out[token_mask] + expert_out * weight[
                token_mask
            ].unsqueeze(-1)
            load_count[expert_id] = token_mask.sum()

        if self.training:
            self._update_routing_bias(load_count, n_tokens)

        output = shared_out + routed_out
        return output.view(batch, seq_len, hidden_size)

    def _update_routing_bias(self, load_count: torch.Tensor, n_tokens: int) -> None:
        with torch.no_grad():
            target_load = n_tokens * self.top_k / self.n_routed_experts
            overloaded = load_count > target_load
            underloaded = load_count < target_load
            self.routing_bias[overloaded] -= self.bias_update_rate
            self.routing_bias[underloaded] += self.bias_update_rate
