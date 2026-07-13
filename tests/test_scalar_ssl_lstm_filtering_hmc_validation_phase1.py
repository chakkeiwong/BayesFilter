from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase1",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed_row(
    *,
    status: str = "passed_short_smoke",
    acceptance_rate: float = 0.75,
    native_available: bool = False,
    native_count: int = 0,
):
    native = (
        {"available": True, "count": native_count}
        if native_available
        else {
            "available": False,
            "status": "not_exposed_by_kernel",
            "nonclaim": "unavailable native divergence telemetry is not zero divergences",
        }
    )
    return {
        "status": status,
        "hard_vetoes": [],
        "hmc_error": None,
        "samples_summary": {
            "finite_sample_count": 16,
            "nonfinite_sample_count": 0,
        },
        "trace_summary": {
            "acceptance_rate": acceptance_rate,
            "target_log_prob": {
                "finite": True,
                "finite_count": 16,
                "nonfinite_count": 0,
            },
            "log_accept_ratio": {
                "finite_count": 16,
                "nonfinite_count": 0,
                "max_abs_finite": 1200.0,
            },
            "native_divergence": native,
        },
    }


def test_phase1_settings_match_reviewed_contract() -> None:
    harness = _load_harness()
    base = harness.load_base_module()
    settings = harness.phase1_settings(base)

    assert settings.num_leapfrog_steps == 4
    assert settings.step_size == 0.3925
    assert settings.trajectory_length == 1.57
    assert settings.num_results == 16
    assert settings.num_burnin_steps == 4
    assert settings.seeds == (
        (20260709, 6101),
        (20260709, 6102),
        (20260709, 6103),
    )


def test_phase1_gate_passes_unavailable_native_divergence_without_zero_claim() -> None:
    harness = _load_harness()
    rows = [_seed_row(acceptance_rate=rate) for rate in (0.75, 0.875, 0.9375)]

    gate = harness.evaluate_phase1_gate(rows, expected_seed_count=3)

    assert gate["passed"] is True
    assert gate["vetoes"] == ()
    assert gate["native_divergence_unavailable_count"] == 3
    assert gate["zero_divergence_claim_made"] is False
    assert gate["log_accept_threshold_used_as_native_divergence"] is False
    assert gate["native_divergence_interpretation"].endswith("not zero divergences")


def test_phase1_gate_rejects_boundary_acceptance() -> None:
    harness = _load_harness()
    rows = [
        _seed_row(acceptance_rate=0.75),
        _seed_row(acceptance_rate=1.0),
        _seed_row(acceptance_rate=0.875),
    ]

    gate = harness.evaluate_phase1_gate(rows, expected_seed_count=3)

    assert gate["passed"] is False
    assert "seed_1_acceptance_outside_phase1_screen" in gate["vetoes"]


def test_phase1_gate_rejects_positive_native_divergence_when_available() -> None:
    harness = _load_harness()
    rows = [
        _seed_row(native_available=True, native_count=0),
        _seed_row(native_available=True, native_count=1),
        _seed_row(native_available=True, native_count=0),
    ]

    gate = harness.evaluate_phase1_gate(rows, expected_seed_count=3)

    assert gate["passed"] is False
    assert "seed_1_native_divergence_detected" in gate["vetoes"]
    assert gate["native_divergence_positive_count"] == 1


def test_phase1_markdown_preserves_zero_divergence_nonclaim() -> None:
    harness = _load_harness()
    payload = {
        "decision": {
            "phase1_short_chain_screen_passed": True,
            "vetoes": (),
            "passed_seed_count": 3,
            "seed_count": 3,
            "zero_divergence_claim_made": False,
            "next_justified_action": "refresh Phase 2",
        },
        "phase1_gate": harness.evaluate_phase1_gate(
            [_seed_row(), _seed_row(), _seed_row()],
            expected_seed_count=3,
        ),
        "aggregate_summary": {},
        "seed_rows": [],
        "inference_status": {
            "zero_divergence_claim": "not made",
        },
        "run_manifest": {
            "command": "fake command",
            "git": {"commit": "abc", "dirty": True},
            "environment": {"python": "3.x", "tensorflow": "2.x"},
            "native_divergence_telemetry_status": ["not_exposed_by_kernel"] * 3,
            "native_divergence_interpretation": (
                "native divergence unavailable for at least one seed; unavailable is not zero divergences"
            ),
        },
        "nonclaims": harness.NONCLAIMS,
    }

    markdown = harness.render_markdown(payload)

    assert "zero_divergence_claim_made: `False`" in markdown
    assert "not a zero-divergence claim" in markdown
    assert "## Run Manifest" in markdown
    assert "native_divergence_telemetry_status" in markdown
