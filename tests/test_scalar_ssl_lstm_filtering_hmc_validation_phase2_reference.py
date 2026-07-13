from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2_local_quadratic_reference_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2_reference",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _geometry_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_geometry.v1",
        "decision": {"geometry_sanity_passed": True},
        "low_rank_geometry": {
            "linear_term": [0.1, -0.2, 0.3, -0.4],
        },
    }


def _mass_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_mass_handoff.v1",
        "decision": {"mass_handoff_passed": True},
        "mass_handoff": {
            "regularized_precision_K_z": np.eye(4).tolist(),
            "factor": (2.0 * np.eye(4)).tolist(),
        },
        "coordinate_contract": {
            "tfp_hmc_coordinate_u": "z = u @ chol(M_z).T",
        },
    }


def _phase1r_payload():
    rows = []
    for index in range(3):
        rows.append(
            {
                "samples_summary": {
                    "mean_u": [0.05, -0.1, 0.15, -0.2],
                    "std_u": [0.5, 0.5, 0.5, 0.5],
                    "finite_sample_count": 64,
                },
                "trace_summary": {
                    "acceptance_rate": 0.75,
                    "log_accept_ratio": {"max_abs_finite": 1.0},
                    "native_divergence": {
                        "available": False,
                        "status": "not_exposed_by_kernel",
                    },
                },
                "seed_index": index,
            }
        )
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase1r.v1",
        "decision": {
            "phase1r_acceptance_repair_screen_passed": True,
            "vetoes": [],
        },
        "seed_rows": rows,
        "telemetry_policy": {
            "native_divergence_statuses": ["not_exposed_by_kernel"] * 3,
            "native_divergence_interpretation": (
                "native divergence unavailable for at least one seed; unavailable is not zero divergences"
            ),
        },
        "settings": {"seeds": [[20260709, 6101], [20260709, 6102], [20260709, 6103]]},
        "target_scope": "toy",
    }


def test_local_quadratic_reference_uses_general_u_mean_formula() -> None:
    harness = _load_harness()

    reference = harness.build_local_quadratic_reference(_geometry_payload(), _mass_payload())

    np.testing.assert_allclose(reference["precision_u"], 4.0 * np.eye(4))
    np.testing.assert_allclose(reference["covariance_u"], 0.25 * np.eye(4))
    np.testing.assert_allclose(reference["mean_u"], np.array([0.05, -0.1, 0.15, -0.2]))
    assert reference["vetoes"] == ()


def test_evaluate_agreement_rejects_large_mean_error() -> None:
    harness = _load_harness()
    reference = {
        "mean_u": np.zeros(4),
        "std_u": np.ones(4),
    }
    hmc = {
        "pooled_mean_u": np.array([0.0, 0.0, 0.6, 0.0]),
        "pooled_std_u": np.ones(4),
    }

    agreement = harness.evaluate_agreement(reference, hmc)

    assert agreement["passed"] is False
    assert "mean_abs_error_above_0p5" in agreement["vetoes"]


def test_phase2_payload_preserves_local_reference_nonclaims() -> None:
    harness = _load_harness()

    payload = harness.run_phase2_reference_agreement(
        _geometry_payload(),
        _mass_payload(),
        _phase1r_payload(),
    )

    assert payload["schema_version"] == harness.SCHEMA_VERSION
    assert payload["decision"]["phase2_local_quadratic_reference_agreement_passed"] is True
    assert payload["decision"]["zero_divergence_claim_made"] is False
    assert payload["reference"]["formula"]["mean_u"] == "m_u = C_u @ F.T @ l_z"
    assert any("not an exact posterior reference" in item for item in payload["nonclaims"])
