from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2r_localization",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase2_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2_local_quadratic_reference.v1",
        "decision": {
            "phase2_local_quadratic_reference_agreement_passed": False,
            "vetoes": ["mean_abs_error_above_0p5"],
        },
        "reference": {
            "precision_u": np.eye(4).tolist(),
            "linear_u": [0.0, 0.0, 0.0, 0.0],
            "mean_u": [0.0, 0.0, 0.0, 0.0],
            "precision_u_identity_max_abs_error": 0.0,
            "covariance_u_identity_max_abs_error": 0.0,
        },
        "hmc_summary": {
            "pooled_mean_u": [5.0, 0.0, 0.0, 0.0],
            "seed_mean_u": [[5.0, 0.0, 0.0, 0.0]],
        },
        "precondition": {"coordinate_contract": {"scale": [1.0, 1.0, 1.0, 1.0]}},
        "telemetry_policy": {
            "native_divergence_statuses": ["not_exposed_by_kernel"],
        },
    }


def _geometry_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_geometry.v1",
        "settings": {"low_rank_trust_radius": 0.30},
    }


def _mass_payload():
    return {"schema_version": "scalar_ssl_lstm.filtering_mass_handoff.v1"}


def test_localization_selects_outside_trust_region_when_norm_and_drop_large() -> None:
    harness = _load_harness()

    payload = harness.run_phase2r_localization(
        _geometry_payload(),
        _mass_payload(),
        _phase2_payload(),
        replay_target=False,
    )

    assert payload["decision"]["phase2r_localization_passed"] is True
    assert payload["decision"]["selected_outcome"] == "outside_geometry_trust_region"
    assert "pooled_hmc_mean" in payload["localization_diagnostics"]["outside_trust_region_points"]
    assert "pooled_hmc_mean" in payload["localization_diagnostics"]["large_quadratic_drop_points"]


def test_transform_mismatch_has_priority_over_geometry_outcome() -> None:
    harness = _load_harness()
    phase2 = _phase2_payload()
    phase2["reference"]["precision_u_identity_max_abs_error"] = 1.0

    payload = harness.run_phase2r_localization(
        _geometry_payload(),
        _mass_payload(),
        phase2,
        replay_target=False,
    )

    assert payload["decision"]["phase2r_localization_passed"] is False
    assert "transform_identity_check_failed" in payload["decision"]["vetoes"]
    assert payload["outcome"]["selected_outcome"] == "transform_bookkeeping_mismatch"


def test_markdown_preserves_nonclaims() -> None:
    harness = _load_harness()
    payload = harness.run_phase2r_localization(
        _geometry_payload(),
        _mass_payload(),
        _phase2_payload(),
        replay_target=False,
    )

    markdown = harness.render_markdown(payload)

    assert "selected_outcome" in markdown
    assert "not HMC readiness evidence" in markdown
    assert "not a zero-divergence claim" in markdown
