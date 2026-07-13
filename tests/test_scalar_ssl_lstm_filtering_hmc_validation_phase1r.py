from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase1r_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase1r",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase1r_settings_keep_kernel_and_seeds_but_extend_draws() -> None:
    harness = _load_harness()
    base = harness.load_base_module()
    settings = harness.phase1r_settings(base)

    assert settings.num_leapfrog_steps == 4
    assert settings.step_size == 0.3925
    assert settings.trajectory_length == 1.57
    assert settings.num_results == 64
    assert settings.num_burnin_steps == 4
    assert settings.seeds == (
        (20260709, 6101),
        (20260709, 6102),
        (20260709, 6103),
    )


def test_phase1r_repair_policy_forbids_tuning() -> None:
    harness = _load_harness()

    policy = harness.phase1r_repair_policy()

    assert policy["no_tuning"] is True
    assert policy["settings_changed"] == ("num_results",)
    assert "step_size" in policy["settings_held_fixed"]
    assert "acceptance_thresholds" in policy["settings_held_fixed"]
    assert "num_burnin_steps" in policy["settings_held_fixed"]


def test_phase1r_markdown_preserves_nonclaims_and_native_boundary() -> None:
    harness = _load_harness()
    payload = {
        "decision": {
            "phase1r_acceptance_repair_screen_passed": True,
            "vetoes": (),
            "passed_seed_count": 3,
            "seed_count": 3,
            "zero_divergence_claim_made": False,
            "next_justified_action": "refresh Phase 2",
        },
        "phase1r_gate": {
            "acceptance_rates": [0.9, 0.8, 0.7],
            "acceptance_min": 0.7,
            "acceptance_max": 0.9,
            "native_divergence_statuses": ["not_exposed_by_kernel"] * 3,
            "native_divergence_interpretation": (
                "native divergence unavailable for at least one seed; unavailable is not zero divergences"
            ),
            "log_accept_threshold_used_as_native_divergence": False,
        },
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
    assert "unavailable is not zero divergences" in markdown
    assert "## Run Manifest" in markdown
    assert "native_divergence_telemetry_status" in markdown
