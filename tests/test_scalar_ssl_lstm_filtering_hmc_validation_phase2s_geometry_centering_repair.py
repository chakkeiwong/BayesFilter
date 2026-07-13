from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2s_geometry_centering_repair",
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
        "center": {
            "free_parameter_values": [0.35, -0.08, 0.65, 0.05],
        },
    }


def _mass_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_mass_handoff.v1",
        "decision": {"mass_handoff_passed": True},
        "coordinate_contract": {
            "scale": [0.35, 0.35, 0.35, 0.35],
        },
        "mass_handoff": {
            "factor": np.eye(4).tolist(),
        },
    }


def _phase1r_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase1r.v1",
        "decision": {
            "phase1r_acceptance_repair_screen_passed": True,
            "vetoes": [],
        },
        "telemetry_policy": {
            "native_divergence_statuses": ["not_exposed_by_kernel"] * 3,
        },
        "settings": {"seeds": [[20260709, 6101], [20260709, 6102], [20260709, 6103]]},
    }


def _phase2_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2_local_quadratic_reference.v1",
        "decision": {
            "phase2_local_quadratic_reference_agreement_passed": False,
            "vetoes": ["mean_abs_error_above_0p5"],
        },
        "reference": {"mean_u": [0.1, -0.2, 0.3, -0.4]},
        "hmc_summary": {"pooled_mean_u": [1.0, 2.0, 3.0, 4.0]},
    }


def _phase2r_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2r_localization.v1",
        "decision": {
            "phase2r_localization_passed": True,
            "selected_outcome": "outside_geometry_trust_region",
            "zero_divergence_claim_made": False,
        },
        "transform_checks": {"passed": True},
    }


def _usable_initializer_payload(*, locator_accepted: bool = True, holdout_count: int = 20):
    return {
        "accepted": True,
        "status": "usable",
        "map_candidate": [0.1, 0.2, 0.3, 0.4],
        "map_candidate_role": "quadratic_surrogate_map_candidate",
        "locator_position": [0.1, 0.2, 0.3, 0.4],
        "locator_diagnostics": {
            "accepted_optimizer_position": locator_accepted,
            "uses_optimizer_inverse_hessian": False,
        },
        "geometry": {
            "accepted": True,
            "status": "usable",
            "diagnostics": {
                "regression_parameter_count": 9,
                "required_finite_samples": 45,
                "finite_sample_count": 90,
                "holdout_count": holdout_count,
                "holdout_passed": True,
            },
        },
        "precision_eigen_summary": {
            "finite": True,
            "positive": True,
            "condition_number": 10.0,
        },
        "covariance_eigen_summary": {
            "finite": True,
            "positive": True,
            "condition_number": 10.0,
        },
        "mass_matrix": {
            "regularization_report": {
                "diagonal_fallback_used": False,
            },
        },
    }


def test_precondition_requires_phase2r_outside_geometry_outcome() -> None:
    harness = _load_harness()
    phase2r = _phase2r_payload()
    phase2r["decision"]["selected_outcome"] = "inconclusive_needs_longer_cpu_chain"

    precondition = harness.validate_inputs(
        _geometry_payload(),
        _mass_payload(),
        _phase1r_payload(),
        _phase2_payload(),
        phase2r,
    )

    assert precondition["passed"] is False
    assert "phase2r_outcome_not_outside_geometry_trust_region" in precondition["vetoes"]


def test_diagnostic_points_transform_u_to_free_parameters() -> None:
    harness = _load_harness()

    points = harness.build_diagnostic_points(
        _geometry_payload(),
        _mass_payload(),
        _phase2_payload(),
    )

    assert points["passed"] is True
    expected_reference = np.array([0.35, -0.08, 0.65, 0.05]) + 0.35 * np.array(
        [0.1, -0.2, 0.3, -0.4]
    )
    np.testing.assert_allclose(
        points["points"]["phase2_reference_mean_initial"]["free"],
        expected_reference,
    )


def test_phase2s_gate_vetoes_locator_fallback_even_with_usable_geometry() -> None:
    harness = _load_harness()
    precondition = {"vetoes": ()}
    points = {"vetoes": ()}
    target_replay = {"computed": True, "vetoes": ()}
    handoff = {"factor_z": np.eye(4)}

    gate = harness.evaluate_phase2s_gate(
        precondition,
        points,
        _usable_initializer_payload(locator_accepted=False),
        target_replay,
        handoff,
    )

    assert gate["decision"]["phase2s_geometry_centering_repair_passed"] is False
    assert "locator_fallback_or_not_accepted" in gate["decision"]["vetoes"]


def test_phase2s_gate_requires_nonzero_holdout() -> None:
    harness = _load_harness()
    precondition = {"vetoes": ()}
    points = {"vetoes": ()}
    target_replay = {"computed": True, "vetoes": ()}
    handoff = {"factor_z": np.eye(4)}

    gate = harness.evaluate_phase2s_gate(
        precondition,
        points,
        _usable_initializer_payload(holdout_count=0),
        target_replay,
        handoff,
    )

    assert gate["decision"]["phase2s_geometry_centering_repair_passed"] is False
    assert "holdout_count_zero" in gate["decision"]["vetoes"]


def test_markdown_preserves_phase2s_nonclaims() -> None:
    harness = _load_harness()
    payload = {
        "decision": {
            "phase2s_geometry_centering_repair_passed": False,
            "vetoes": ("locator_fallback_or_not_accepted",),
            "viable_for_map_local_reference_subplan": False,
            "zero_divergence_claim_made": False,
            "next_justified_action": "write Phase 2S result",
        },
        "initializer": _usable_initializer_payload(locator_accepted=False),
        "inference_status": {"hmc_readiness": "not assessed"},
        "nonclaims": harness.NONCLAIMS,
    }

    markdown = harness.render_markdown(payload)

    assert "not HMC readiness evidence" in markdown
    assert "not a certified global MAP" in markdown
    assert "zero_divergence_claim_made" in markdown
