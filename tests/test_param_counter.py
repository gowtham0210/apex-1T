import pytest

from apex1t.config import DEBUG_CONFIG, FULL_CONFIG, ModelConfig
from apex1t.param_counter import count_params


def test_debug_config_total_within_indicative_range():
    report = count_params(DEBUG_CONFIG, "DebugConfig")
    assert 10_000_000 <= report.total_params <= 50_000_000


def test_full_config_total_near_one_trillion():
    report = count_params(FULL_CONFIG, "FullConfig")
    assert 0.9e12 <= report.total_params <= 1.1e12


def test_full_config_active_near_37_billion():
    report = count_params(FULL_CONFIG, "FullConfig")
    assert 30e9 <= report.active_params <= 45e9


@pytest.mark.parametrize("config", [DEBUG_CONFIG, FULL_CONFIG])
def test_breakdown_sums_to_reported_totals(config: ModelConfig):
    report = count_params(config)
    assert sum(c.total for c in report.breakdown.values()) == report.total_params
    assert sum(c.active for c in report.breakdown.values()) == report.active_params


@pytest.mark.parametrize("config", [DEBUG_CONFIG, FULL_CONFIG])
def test_active_never_exceeds_total(config: ModelConfig):
    report = count_params(config)
    assert report.active_params <= report.total_params
    for counts in report.breakdown.values():
        assert counts.active <= counts.total


@pytest.mark.parametrize("config", [DEBUG_CONFIG, FULL_CONFIG])
def test_only_routed_experts_differ_between_total_and_active(config: ModelConfig):
    report = count_params(config)
    for name, counts in report.breakdown.items():
        if name == "routed_experts":
            assert counts.active < counts.total
        else:
            assert counts.active == counts.total


@pytest.mark.parametrize("config", [DEBUG_CONFIG, FULL_CONFIG])
def test_flops_per_token_is_twice_active_params(config: ModelConfig):
    report = count_params(config)
    assert report.flops_per_token == 2 * report.active_params


def test_top_k_greater_than_routed_experts_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(
            vocab_size=100,
            max_context_length=64,
            hidden_size=32,
            n_layers=2,
            n_dense_layers=0,
            num_heads=2,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            q_lora_rank=16,
            kv_lora_rank=8,
            dense_intermediate_size=64,
            moe_intermediate_size=64,
            n_routed_experts=4,
            n_shared_experts=1,
            top_k=8,
        )


def test_dense_layers_greater_than_total_layers_is_rejected():
    with pytest.raises(ValueError):
        ModelConfig(
            vocab_size=100,
            max_context_length=64,
            hidden_size=32,
            n_layers=2,
            n_dense_layers=4,
            num_heads=2,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            q_lora_rank=16,
            kv_lora_rank=8,
            dense_intermediate_size=64,
            moe_intermediate_size=64,
            n_routed_experts=4,
            n_shared_experts=1,
            top_k=2,
        )
