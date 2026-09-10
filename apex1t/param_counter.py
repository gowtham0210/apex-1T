"""Analytical parameter and FLOP counter for `ModelConfig`.

Pure functions of the config only — no `torch`, no model instantiation. This
lets ticket 06/07 cross-check these numbers against the real, instantiated
module tree without the counter depending on the model existing yet.

Conventions used throughout:
  - "Total" parameters means every weight that exists in the model, whether
    or not a given token's forward pass touches it (i.e. every routed
    expert, even ones not selected for a particular token).
  - "Active" parameters means every weight that participates in compute for
    a *given token*: attention, shared experts, the router, dense-stem FFNs,
    the embedding table, and the output head are always active; only the
    top-k selected routed experts (not all of them) count as active.
  - FLOPs per token is estimated with the standard forward-pass
    approximation FLOPs ≈ 2 * active_params (the multiply-accumulate cost of
    a dense matmul over the active weights). This intentionally ignores the
    smaller, sequence-length-dependent attention-score term
    (O(seq_len * hidden_size) per layer), which is negligible next to the
    O(hidden_size^2)-scale projection cost at this parameter count.
"""

from __future__ import annotations

from dataclasses import dataclass

from apex1t.config import ModelConfig


def _swiglu_ffn_params(hidden_size: int, intermediate_size: int) -> int:
    """Gate, up, and down projections of a SwiGLU FFN."""
    return 3 * hidden_size * intermediate_size


def _mla_attention_params(config: ModelConfig) -> int:
    """Multi-Head Latent Attention: down/up projections for Q and KV, plus output."""
    q_down = config.hidden_size * config.q_lora_rank
    q_up = config.q_lora_rank * config.num_heads * (
        config.qk_nope_head_dim + config.qk_rope_head_dim
    )
    # KV down-projection produces the compressed content latent AND the
    # shared decoupled-RoPE key component in one projection.
    kv_down = config.hidden_size * (config.kv_lora_rank + config.qk_rope_head_dim)
    kv_up = config.kv_lora_rank * config.num_heads * (
        config.qk_nope_head_dim + config.v_head_dim
    )
    out_proj = config.num_heads * config.v_head_dim * config.hidden_size
    return q_down + q_up + kv_down + kv_up + out_proj


def _norm_params(config: ModelConfig) -> int:
    """RMSNorm scale vectors: 2 per layer (pre-attn, pre-FFN) + 1 final."""
    return (2 * config.n_layers + 1) * config.hidden_size


@dataclass(frozen=True)
class ComponentCounts:
    total: int
    active: int


@dataclass(frozen=True)
class ParamReport:
    config_name: str
    total_params: int
    active_params: int
    flops_per_token: int
    breakdown: dict[str, ComponentCounts]

    @property
    def activation_ratio(self) -> float:
        return self.active_params / self.total_params


def count_params(config: ModelConfig, config_name: str = "config") -> ParamReport:
    expert_params = _swiglu_ffn_params(config.hidden_size, config.moe_intermediate_size)
    dense_ffn_params = _swiglu_ffn_params(config.hidden_size, config.dense_intermediate_size)
    attn_params = _mla_attention_params(config)
    router_params = config.hidden_size * config.n_routed_experts

    n_moe = config.n_moe_layers
    n_dense = config.n_dense_layers

    embedding_total = config.vocab_size * config.hidden_size
    output_head_total = config.vocab_size * config.hidden_size  # untied from embedding
    norms_total = _norm_params(config)

    attention_total = config.n_layers * attn_params
    dense_ffn_total = n_dense * dense_ffn_params
    router_total = n_moe * router_params
    shared_experts_total = n_moe * config.n_shared_experts * expert_params
    routed_experts_total = n_moe * config.n_routed_experts * expert_params

    total_params = (
        embedding_total
        + output_head_total
        + norms_total
        + attention_total
        + dense_ffn_total
        + router_total
        + shared_experts_total
        + routed_experts_total
    )

    # Active: identical to total for every component except routed experts,
    # where only top_k (not all n_routed_experts) are active per token.
    routed_experts_active = n_moe * config.top_k * expert_params

    active_params = (
        embedding_total
        + output_head_total
        + norms_total
        + attention_total
        + dense_ffn_total
        + router_total
        + shared_experts_total
        + routed_experts_active
    )

    flops_per_token = 2 * active_params

    breakdown = {
        "embedding": ComponentCounts(embedding_total, embedding_total),
        "output_head": ComponentCounts(output_head_total, output_head_total),
        "norms": ComponentCounts(norms_total, norms_total),
        "attention": ComponentCounts(attention_total, attention_total),
        "dense_ffn": ComponentCounts(dense_ffn_total, dense_ffn_total),
        "router": ComponentCounts(router_total, router_total),
        "shared_experts": ComponentCounts(shared_experts_total, shared_experts_total),
        "routed_experts": ComponentCounts(routed_experts_total, routed_experts_active),
    }

    return ParamReport(
        config_name=config_name,
        total_params=total_params,
        active_params=active_params,
        flops_per_token=flops_per_token,
        breakdown=breakdown,
    )


def _fmt(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def format_report(report: ParamReport) -> str:
    lines = [
        f"=== {report.config_name} ===",
        f"{'component':<18} {'total':>10} {'active':>10}",
        "-" * 40,
    ]
    for name, counts in report.breakdown.items():
        lines.append(f"{name:<18} {_fmt(counts.total):>10} {_fmt(counts.active):>10}")
    lines.append("-" * 40)
    lines.append(f"{'TOTAL':<18} {_fmt(report.total_params):>10} {_fmt(report.active_params):>10}")
    lines.append(f"activation ratio: {report.activation_ratio:.2%}")
    lines.append(f"FLOPs/token (fwd, approx): {_fmt(report.flops_per_token)}")
    return "\n".join(lines)


if __name__ == "__main__":
    from apex1t.config import DEBUG_CONFIG, FULL_CONFIG

    debug_report = count_params(DEBUG_CONFIG, "DebugConfig")
    full_report = count_params(FULL_CONFIG, "FullConfig")

    print(format_report(debug_report))
    print()
    print(format_report(full_report))
