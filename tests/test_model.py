import torch

from apex1t.attention import _rope_cache
from apex1t.config import DEBUG_CONFIG, FULL_CONFIG
from apex1t.model import Apex1T
from apex1t.param_counter import count_params


def _tiny_token_ids(batch: int = 2, seq_len: int = 8) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, DEBUG_CONFIG.vocab_size, (batch, seq_len))


def test_debug_config_forward_shape_and_finiteness():
    model = Apex1T(DEBUG_CONFIG)
    input_ids = _tiny_token_ids()

    logits = model(input_ids)

    assert logits.shape == (2, 8, DEBUG_CONFIG.vocab_size)
    assert torch.isfinite(logits).all()


def test_debug_config_backward_reaches_embedding_and_head():
    model = Apex1T(DEBUG_CONFIG)
    input_ids = _tiny_token_ids()

    logits = model(input_ids)
    targets = _tiny_token_ids()
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, DEBUG_CONFIG.vocab_size), targets.view(-1)
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.embed_tokens.weight.grad is not None
    assert model.embed_tokens.weight.grad.abs().sum().item() > 0
    assert model.output_head.weight.grad is not None
    assert model.output_head.weight.grad.abs().sum().item() > 0
    assert model.final_norm.weight.grad is not None
    assert model.final_norm.weight.grad.abs().sum().item() > 0


def test_debug_config_total_param_count_matches_analytical_counter():
    model = Apex1T(DEBUG_CONFIG)
    expected = count_params(DEBUG_CONFIG).total_params
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected


def test_full_config_instantiates_on_meta_device_without_error():
    with torch.device("meta"):
        model = Apex1T(FULL_CONFIG)

    assert len(model.layers) == FULL_CONFIG.n_layers
    assert model.embed_tokens.weight.device.type == "meta"


def test_full_config_meta_param_count_is_near_one_trillion():
    with torch.device("meta"):
        model = Apex1T(FULL_CONFIG)

    total = sum(p.numel() for p in model.parameters())
    assert 0.9e12 <= total <= 1.1e12


def test_rope_cache_supports_full_config_context_length():
    # Cheap structural proof that RoPE isn't hardcoded to a shorter max
    # position: build the cos/sin tables at the full 128K context length
    # without needing a full attention forward pass (which would require
    # O(seq_len^2) memory this machine doesn't have).
    cos, sin = _rope_cache(
        FULL_CONFIG.max_context_length,
        FULL_CONFIG.qk_rope_head_dim,
        base=10_000.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert cos.shape == (FULL_CONFIG.max_context_length, FULL_CONFIG.qk_rope_head_dim)
    assert sin.shape == (FULL_CONFIG.max_context_length, FULL_CONFIG.qk_rope_head_dim)
    assert torch.isfinite(cos).all()
    assert torch.isfinite(sin).all()
