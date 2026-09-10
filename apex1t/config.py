"""Config schema for Apex-1T (MLA + auxiliary-loss-free sparse MoE).

`ModelConfig` fully parameterizes both the debug-scale and full-scale (~1T
total / ~37B active) versions of the architecture. The same dataclass and the
same model code (later tickets) are used for both — only the field values
differ.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    # --- global shape ---
    vocab_size: int
    max_context_length: int
    hidden_size: int  # d_model
    n_layers: int
    n_dense_layers: int  # leading layers that use a plain dense FFN, not MoE

    # --- attention (Multi-Head Latent Attention) ---
    num_heads: int
    qk_nope_head_dim: int  # non-positional part of each query/key head
    qk_rope_head_dim: int  # decoupled rotary part of each query/key head
    v_head_dim: int
    q_lora_rank: int  # query down-projection (compression) rank
    kv_lora_rank: int  # key/value down-projection (compression) rank

    # --- feed-forward ---
    dense_intermediate_size: int  # SwiGLU intermediate size for dense-stem layers
    moe_intermediate_size: int  # SwiGLU intermediate size for each expert
    n_routed_experts: int
    n_shared_experts: int
    top_k: int  # routed experts selected per token

    def __post_init__(self) -> None:
        if self.n_dense_layers > self.n_layers:
            raise ValueError("n_dense_layers cannot exceed n_layers")
        if self.top_k > self.n_routed_experts:
            raise ValueError("top_k cannot exceed n_routed_experts")

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers


# ~34M total params. Small enough to instantiate in real memory and run
# forward + backward on an 8GB Apple M2 in seconds. Same architecture shape
# as FullConfig, just narrower/shallower.
DEBUG_CONFIG = ModelConfig(
    vocab_size=2_000,
    max_context_length=256,
    hidden_size=384,
    n_layers=6,
    n_dense_layers=2,
    num_heads=4,
    qk_nope_head_dim=32,
    qk_rope_head_dim=16,
    v_head_dim=32,
    q_lora_rank=96,
    kv_lora_rank=48,
    dense_intermediate_size=768,
    moe_intermediate_size=384,
    n_routed_experts=16,
    n_shared_experts=1,
    top_k=2,
)

# Target: ~1T total params, ~37B active params per token (~3.7% activation
# ratio, matching DeepSeek-V3's ratio). 120 layers, DeepSeek-V3-scaled.
# 128K vocab / 128K context, matching current (2025/2026) frontier defaults.
FULL_CONFIG = ModelConfig(
    vocab_size=128_000,
    max_context_length=128_000,
    hidden_size=6_144,
    n_layers=120,
    n_dense_layers=3,
    num_heads=48,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    q_lora_rank=1_536,
    kv_lora_rank=512,
    dense_intermediate_size=16_384,
    moe_intermediate_size=2_048,
    n_routed_experts=224,
    n_shared_experts=1,
    top_k=5,
)
