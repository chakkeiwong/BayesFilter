from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2t_map_local_reference_handoff",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase2s_payload(*, bad_theta_transform: bool = False):
    scale = np.array([0.5, 2.0, 1.0, 0.25])
    factor_z = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.2, 1.2, 0.0, 0.0],
            [-0.1, 0.3, 0.8, 0.0],
            [0.05, -0.2, 0.4, 0.9],
        ]
    )
    covariance_z = factor_z @ factor_z.T
    precision_z = np.linalg.inv(covariance_z)
    inv_scale = 1.0 / scale
    precision_theta = inv_scale[:, None] * precision_z * inv_scale[None, :]
    covariance_theta = scale[:, None] * covariance_z * scale[None, :]
    if bad_theta_transform:
        precision_theta = precision_theta.copy()
        precision_theta[0, 0] += 0.1
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2s_geometry_centering_repair.v1",
        "decision": {
            "phase2s_geometry_centering_repair_passed": True,
            "vetoes": [],
        },
        "initializer": {
            "accepted": True,
            "status": "usable",
            "map_candidate_role": "locator_position_geometry_covariance_only",
            "locator_diagnostics": {
                "accepted_optimizer_position": True,
                "uses_optimizer_inverse_hessian": False,
            },
        },
        "map_local_handoff": {
            "center_free_parameter_values": [0.1, 0.2, 0.3, 0.4],
            "scale": scale.tolist(),
            "precision_z": precision_z.tolist(),
            "covariance_z": covariance_z.tolist(),
            "factor_z": factor_z.tolist(),
            "precision_theta": precision_theta.tolist(),
            "covariance_theta": covariance_theta.tolist(),
            "coordinate_formula": "free = center_free_parameter_values + scale * (factor_z @ u_new)",
        },
        "target_replay": {
            "computed": True,
            "values": {
                "map_candidate": {
                    "status": "finite",
                    "value": -1.0,
                },
                "phase1r_pooled_hmc_mean": {
                    "status": "finite",
                    "value": -2.0,
                },
            },
        },
        "telemetry_policy": {
            "native_divergence_statuses": ["not_exposed_by_kernel"],
            "native_divergence_interpretation": (
                "native divergence unavailable for at least one seed; unavailable is not zero divergences"
            ),
        },
    }


def test_map_local_matrix_checks_include_theta_z_transforms() -> None:
    harness = _load_harness()

    matrices = harness.validate_map_local_matrices(_phase2s_payload())

    assert matrices["passed"] is True
    diagnostics = matrices["diagnostics"]
    assert diagnostics["precision_theta_scale_transform_max_abs_error"] == 0.0
    assert diagnostics["covariance_theta_scale_transform_max_abs_error"] == 0.0


def test_theta_transform_mismatch_vetoes_handoff() -> None:
    harness = _load_harness()

    matrices = harness.validate_map_local_matrices(
        _phase2s_payload(bad_theta_transform=True)
    )

    assert matrices["passed"] is False
    assert "precision_theta_scale_transform_failed" in matrices["vetoes"]


def test_old_geometry_projection_is_excluded_from_pass_fail() -> None:
    harness = _load_harness()

    projection = harness.old_geometry_projection_diagnostic(_phase2s_payload())

    assert projection["computed"] is True
    assert projection["included_in_pass_fail"] is False
    assert projection["promotion_criterion"] is False


def test_phase2u_next_contract_predeclares_grid_and_selection() -> None:
    harness = _load_harness()

    contract = harness.phase2u_next_subplan_contract()

    assert contract["candidate_grid_predeclared"] is True
    assert contract["all_trajectory_lengths_equal_1p57"] is True
    assert contract["selection_policy_predeclared"] is True
    assert contract["acceptance_envelope"] == {
        "lower_exclusive": 0.05,
        "upper_exclusive": 0.99,
    }


def test_phase2t_full_gate_passes_for_consistent_payload() -> None:
    harness = _load_harness()

    payload = harness.run_phase2t_map_local_reference_handoff(_phase2s_payload())

    assert payload["decision"]["phase2t_map_local_reference_handoff_passed"] is True
    assert payload["decision"]["zero_divergence_claim_made"] is False
    assert payload["old_geometry_summary_projection"]["included_in_pass_fail"] is False
    assert "not HMC readiness evidence" in payload["nonclaims"]
