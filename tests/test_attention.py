import torch

from apex1t.attention import MultiHeadLatentAttention
from apex1t.config import DEBUG_CONFIG
from apex1t.param_counter import _mla_attention_params


def _tiny_input(batch: int = 2, seq_len: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, seq_len, DEBUG_CONFIG.hidden_size, requires_grad=True)


def test_forward_output_shape_and_finiteness():
    attn = MultiHeadLatentAttention(DEBUG_CONFIG)
    x = _tiny_input(batch=2, seq_len=8)

    out = attn(x)

    assert out.shape == (2, 8, DEBUG_CONFIG.hidden_size)
    assert torch.isfinite(out).all()


def test_backward_produces_nonzero_gradients_on_every_projection():
    attn = MultiHeadLatentAttention(DEBUG_CONFIG)
    x = _tiny_input(batch=2, seq_len=8)

    out = attn(x)
    out.sum().backward()

    for name, param in attn.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert param.grad.abs().sum().item() > 0, f"{name} gradient is all-zero"


def test_parameter_count_matches_analytical_formula():
    attn = MultiHeadLatentAttention(DEBUG_CONFIG)

    actual = sum(p.numel() for p in attn.parameters())
    expected = _mla_attention_params(DEBUG_CONFIG)

    assert actual == expected


def test_runnable_standalone_without_full_transformer_block():
    # No block/model context required — just the module and a raw tensor.
    attn = MultiHeadLatentAttention(DEBUG_CONFIG)
    x = torch.randn(1, 4, DEBUG_CONFIG.hidden_size)

    out = attn(x)

    assert out.shape == (1, 4, DEBUG_CONFIG.hidden_size)
