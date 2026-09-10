from apex1t.validate import (
    check_debug_forward_backward,
    check_debug_param_count_exact,
    check_full_meta_instantiation,
    check_full_param_count_exact,
    run_all,
)


def test_debug_forward_backward_check_passes():
    result = check_debug_forward_backward()
    assert result.passed, result.detail


def test_debug_param_count_exact_check_passes():
    result = check_debug_param_count_exact()
    assert result.passed, result.detail


def test_full_meta_instantiation_check_passes():
    result = check_full_meta_instantiation()
    assert result.passed, result.detail


def test_full_param_count_exact_check_passes():
    result = check_full_param_count_exact()
    assert result.passed, result.detail


def test_run_all_reports_every_check_passing():
    all_passed, results = run_all()
    assert all_passed, [r for r in results if not r.passed]
    assert {r.name for r in results} == {
        "debug_forward_backward",
        "debug_param_count_exact",
        "full_meta_instantiation",
        "full_param_count_exact",
    }
