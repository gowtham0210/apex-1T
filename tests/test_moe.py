import torch

from apex1t.config import DEBUG_CONFIG
from apex1t.moe import SparseMoE
from apex1t.param_counter import _swiglu_ffn_params


def _tiny_input(batch: int = 2, seq_len: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, seq_len, DEBUG_CONFIG.hidden_size, requires_grad=True)


def test_forward_output_shape_and_finiteness():
    moe = SparseMoE(DEBUG_CONFIG)
    x = _tiny_input()

    out = moe(x)

    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_backward_produces_gradients_on_shared_experts_and_router():
    moe = SparseMoE(DEBUG_CONFIG)
    x = _tiny_input()

    out = moe(x)
    out.sum().backward()

    for expert in moe.shared_experts:
        for name, param in expert.named_parameters():
            assert param.grad is not None, f"shared expert {name} got no gradient"
            assert param.grad.abs().sum().item() > 0, f"shared expert {name} grad is all-zero"

    assert moe.router.weight.grad is not None
    assert moe.router.weight.grad.abs().sum().item() > 0


def test_backward_produces_gradients_on_every_selected_routed_expert():
    moe = SparseMoE(DEBUG_CONFIG)
    x = _tiny_input(batch=4, seq_len=16)  # enough tokens to likely touch every expert

    router_logits = moe.router(x.reshape(-1, DEBUG_CONFIG.hidden_size).detach())
    _, topk_idx = router_logits.topk(moe.top_k, dim=-1)
    selected_experts = set(topk_idx.flatten().tolist())

    out = moe(x)
    out.sum().backward()

    for expert_id in selected_experts:
        expert = moe.routed_experts[expert_id]
        for name, param in expert.named_parameters():
            assert param.grad is not None, f"routed expert {expert_id}.{name} got no gradient"
            assert param.grad.abs().sum().item() > 0, (
                f"routed expert {expert_id}.{name} grad is all-zero"
            )


def test_routing_bias_updates_after_forward_pass_when_training():
    moe = SparseMoE(DEBUG_CONFIG, bias_update_rate=1e-2)
    moe.train()
    x = _tiny_input(batch=4, seq_len=16)

    bias_before = moe.routing_bias.clone()
    moe(x)
    bias_after = moe.routing_bias.clone()

    assert not torch.equal(bias_before, bias_after)


def test_routing_bias_does_not_update_in_eval_mode():
    moe = SparseMoE(DEBUG_CONFIG, bias_update_rate=1e-2)
    moe.eval()
    x = _tiny_input(batch=4, seq_len=16)

    bias_before = moe.routing_bias.clone()
    with torch.no_grad():
        moe(x)
    bias_after = moe.routing_bias.clone()

    assert torch.equal(bias_before, bias_after)


def test_routing_bias_is_not_a_trainable_parameter():
    moe = SparseMoE(DEBUG_CONFIG)
    param_names = {name for name, _ in moe.named_parameters()}
    assert "routing_bias" not in param_names
    assert "routing_bias" in dict(moe.named_buffers())


def test_parameter_count_matches_analytical_formula():
    moe = SparseMoE(DEBUG_CONFIG)

    expert_params = _swiglu_ffn_params(DEBUG_CONFIG.hidden_size, DEBUG_CONFIG.moe_intermediate_size)
    expected = (
        DEBUG_CONFIG.n_shared_experts * expert_params
        + DEBUG_CONFIG.n_routed_experts * expert_params
        + DEBUG_CONFIG.hidden_size * DEBUG_CONFIG.n_routed_experts  # router
    )

    actual = sum(p.numel() for p in moe.parameters())

    assert actual == expected


def test_runnable_standalone_without_full_transformer_block():
    moe = SparseMoE(DEBUG_CONFIG)
    x = torch.randn(1, 4, DEBUG_CONFIG.hidden_size)

    out = moe(x)

    assert out.shape == (1, 4, DEBUG_CONFIG.hidden_size)
