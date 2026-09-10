"""Validation suite (ticket 07): the three seam tests from SPEC.md, run
against the fully assembled `Apex1T` model rather than individual
components.

Runnable as a script (`python -m apex1t.validate`) for a clear pass/fail
report, and as a library of functions each covered by its own pytest test
in `tests/test_validate.py`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from apex1t.config import DEBUG_CONFIG, FULL_CONFIG
from apex1t.model import Apex1T
from apex1t.param_counter import count_params, format_report


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def check_debug_forward_backward() -> CheckResult:
    """Seam 1: debug model runs a real forward+backward pass; every
    parameter expected to receive gradient does, and the MoE routing bias
    buffers actually move.
    """
    torch.manual_seed(0)
    model = Apex1T(DEBUG_CONFIG)
    model.train()

    bias_before = {
        f"layers.{i}": block.ffn.routing_bias.clone()
        for i, block in enumerate(model.layers)
        if block.use_moe
    }

    input_ids = torch.randint(0, DEBUG_CONFIG.vocab_size, (2, 8))
    targets = torch.randint(0, DEBUG_CONFIG.vocab_size, (2, 8))

    logits = model(input_ids)
    if not torch.isfinite(logits).all():
        return CheckResult("debug_forward_backward", False, "logits contain NaN/Inf")

    loss = F.cross_entropy(logits.view(-1, DEBUG_CONFIG.vocab_size), targets.view(-1))
    loss.backward()

    missing_grad = []
    for name, param in model.named_parameters():
        if "routed_experts" in name:
            continue  # only experts selected for this batch receive gradient
        if param.grad is None or param.grad.abs().sum().item() == 0:
            missing_grad.append(name)
    if missing_grad:
        return CheckResult(
            "debug_forward_backward", False, f"no/zero gradient on: {missing_grad[:5]}"
        )

    selected_without_grad = []
    for i, block in enumerate(model.layers):
        if not block.use_moe:
            continue
        for expert_id, expert in enumerate(block.ffn.routed_experts):
            has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in expert.parameters())
            was_selected = any(p.grad is not None for p in expert.parameters())
            if was_selected and not has_grad:
                selected_without_grad.append(f"layers.{i}.routed_experts.{expert_id}")
    if selected_without_grad:
        return CheckResult(
            "debug_forward_backward", False, f"selected expert with zero grad: {selected_without_grad[:5]}"
        )

    unchanged_bias = [
        name
        for name, before in bias_before.items()
        if torch.equal(before, dict(model.named_modules())[name].ffn.routing_bias)
    ]
    if unchanged_bias:
        return CheckResult(
            "debug_forward_backward", False, f"routing_bias unchanged for: {unchanged_bias}"
        )

    return CheckResult("debug_forward_backward", True, "gradients present, routing bias updated")


def check_debug_param_count_exact() -> CheckResult:
    """Seam 2: analytical counter matches the real, materialized debug model exactly."""
    model = Apex1T(DEBUG_CONFIG)
    expected = count_params(DEBUG_CONFIG).total_params
    actual = sum(p.numel() for p in model.parameters())
    if actual != expected:
        return CheckResult(
            "debug_param_count_exact", False, f"counter={expected} actual={actual}"
        )
    return CheckResult("debug_param_count_exact", True, f"{actual:,} params (exact match)")


def check_full_meta_instantiation() -> CheckResult:
    """Seam 3a: the real 1T-param config constructs on the meta device with no shape errors."""
    try:
        with torch.device("meta"):
            model = Apex1T(FULL_CONFIG)
    except Exception as exc:  # noqa: BLE001 - report any construction failure as a failed check
        return CheckResult("full_meta_instantiation", False, f"{type(exc).__name__}: {exc}")

    if len(model.layers) != FULL_CONFIG.n_layers:
        return CheckResult(
            "full_meta_instantiation", False, f"expected {FULL_CONFIG.n_layers} layers, got {len(model.layers)}"
        )
    return CheckResult("full_meta_instantiation", True, f"{len(model.layers)} layers constructed on meta device")


def check_full_param_count_exact() -> CheckResult:
    """Seam 3b: the meta-instantiated full model's real param count matches the analytical counter exactly."""
    with torch.device("meta"):
        model = Apex1T(FULL_CONFIG)
    actual = sum(p.numel() for p in model.parameters())
    expected = count_params(FULL_CONFIG).total_params
    if actual != expected:
        return CheckResult(
            "full_param_count_exact", False, f"counter={expected} actual={actual}"
        )
    return CheckResult("full_param_count_exact", True, f"{actual:,} params (exact match)")


def run_all() -> tuple[bool, list[CheckResult]]:
    checks = [
        check_debug_forward_backward(),
        check_debug_param_count_exact(),
        check_full_meta_instantiation(),
        check_full_param_count_exact(),
    ]
    return all(c.passed for c in checks), checks


if __name__ == "__main__":
    all_passed, results = run_all()

    print("=== Apex-1T validation suite ===")
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")

    print()
    print("=== Validated FullConfig numbers (for README, ticket 08) ===")
    print(format_report(count_params(FULL_CONFIG, "FullConfig")))

    print()
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    raise SystemExit(0 if all_passed else 1)
