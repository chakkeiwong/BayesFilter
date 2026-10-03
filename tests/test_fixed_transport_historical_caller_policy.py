"""Static policy contracts for historical fixed-transport benchmark callers."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_ROOT = ROOT / "docs" / "benchmarks"

# These callers were identified by the ordinary-HMC migration review.  They
# retain old one-L or directional tuning semantics and must say so explicitly.
HISTORICAL_CALLERS = (
    "run_weighted_neutra_three_mode_hmc_2026_08_12.py",
    "run_neutra_paper_d100_hmc_2026_08_13.py",
    "run_weighted_neutra_german_reverse_hmc_2026_08_13.py",
    "run_weighted_neutra_strong_smooth_hmc_2026_08_12.py",
    "run_defensive_weighted_neutra_analytic_hmc_2026_08_12.py",
    "run_neutra_banana_hmc_repair_2026_08_16.py",
    "run_neutra_replication_hmc_campaign_2026_08_16.py",
    "run_ssl_lstm_q20_fixed_hmc_api_cpu_xla_validation_2026_08_02.py",
    "run_ssl_lstm_q20_neutra_global_mixing_hmc_2026_08_19.py",
    "run_ssl_lstm_q20_neutra_global_mixing_continuation_2026_08_20.py",
    "run_ssl_lstm_q20_seed_b_terminal_six_l_tuning_2026_08_07.py",
    "run_ssl_lstm_q20_chart_a_six_l_fixed_hmc_tuning_2026_08_03.py",
)


def _config_calls(path: Path) -> tuple[ast.Call, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return tuple(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "FixedTransportHMCKernelTuningConfig"
    )


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def test_review_named_callers_select_explicit_legacy_policy() -> None:
    for filename in HISTORICAL_CALLERS:
        path = BENCHMARK_ROOT / filename
        calls = _config_calls(path)
        assert calls, filename
        source = path.read_text(encoding="utf-8")
        assert "FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY" in source
        assert "diagnostic" in source.lower()
        for call in calls:
            policy = _keyword(call, "tuning_policy")
            assert isinstance(policy, ast.Name)
            assert policy.id == "FIXED_TRANSPORT_HMC_LEGACY_DIAGNOSTIC_POLICY"
            selector = _keyword(call, "selection_policy")
            assert isinstance(selector, ast.Constant)
            assert selector.value == "acceptance_target_distance"


def test_every_benchmark_fixed_transport_config_declares_a_policy() -> None:
    for path in sorted(BENCHMARK_ROOT.glob("*.py")):
        for call in _config_calls(path):
            policy = _keyword(call, "tuning_policy")
            assert policy is not None, f"bare fixed-transport policy: {path}:{call.lineno}"
            policy_text = ast.unparse(policy)
            if policy_text == "'measured_joint_grid_v1'":
                assert _keyword(call, "step_size_candidates") is not None


def test_historical_callers_do_not_claim_verified_fixed_transport_handoff() -> None:
    forbidden = "build_verified_fixed_transport_hmc_handoff_from_tuning_result"
    for filename in HISTORICAL_CALLERS:
        source = (BENCHMARK_ROOT / filename).read_text(encoding="utf-8")
        assert forbidden not in source, filename
