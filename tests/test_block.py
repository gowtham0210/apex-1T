import pytest
import torch
import torch.nn as nn

from apex1t.block import TransformerBlock
from apex1t.config import DEBUG_CONFIG
from apex1t.param_counter import count_params


def _stack() -> nn.ModuleList:
    return nn.ModuleList(
        TransformerBlock(DEBUG_CONFIG, layer_idx=i) for i in range(DEBUG_CONFIG.n_layers)
    )


def _tiny_input(batch: int = 2, seq_len: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, seq_len, DEBUG_CONFIG.hidden_size, requires_grad=True)


def test_first_n_dense_layers_use_swiglu_not_moe():
    blocks = _stack()
    for i, block in enumerate(blocks):
        if i < DEBUG_CONFIG.n_dense_layers:
            assert not block.use_moe
        else:
            assert block.use_moe


def test_stack_forward_shape_and_finiteness():
    blocks = _stack()
    x = _tiny_input()

    for block in blocks:
        x = block(x)

    assert x.shape == (2, 8, DEBUG_CONFIG.hidden_size)
    assert torch.isfinite(x).all()


def test_stack_backward_produces_gradients_in_every_layer_dense_and_moe():
    blocks = _stack()
    x = _tiny_input()

    out = x
    for block in blocks:
        out = block(out)
    out.sum().backward()

    for i, block in enumerate(blocks):
        for name, param in block.named_parameters():
            if "routed_experts" in name:
                # Sparse MoE: only the experts selected for this tiny batch
                # receive gradient — checked directly in test_moe.py, not
                # required for every routed expert here.
                continue
            assert param.grad is not None, f"layer {i} ({name}) got no gradient"
            assert param.grad.abs().sum().item() > 0, f"layer {i} ({name}) grad is all-zero"


def test_layer_idx_out_of_range_is_rejected():
    with pytest.raises(ValueError):
        TransformerBlock(DEBUG_CONFIG, layer_idx=DEBUG_CONFIG.n_layers)
    with pytest.raises(ValueError):
        TransformerBlock(DEBUG_CONFIG, layer_idx=-1)


def test_stack_parameter_count_matches_analytical_breakdown():
    blocks = _stack()
    report = count_params(DEBUG_CONFIG)
    b = report.breakdown

    expected = (
        b["attention"].total
        + b["dense_ffn"].total
        + b["router"].total
        + b["shared_experts"].total
        + b["routed_experts"].total
        + (b["norms"].total - DEBUG_CONFIG.hidden_size)  # exclude the model's final norm
    )
    actual = sum(p.numel() for block in blocks for p in block.parameters())

    assert actual == expected
