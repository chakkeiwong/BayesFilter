from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase2s_payload(*, bad_factor: bool = False):
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
    if bad_factor:
        factor_z = factor_z.copy()
        factor_z[0, 0] += 0.2
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2s_geometry_centering_repair.v1",
        "decision": {
            "phase2s_geometry_centering_repair_passed": True,
            "vetoes": [],
        },
        "map_local_handoff": {
            "center_free_parameter_values": [0.1, 0.2, 0.3, 0.4],
            "scale": scale.tolist(),
            "precision_z": precision_z.tolist(),
            "covariance_z": covariance_z.tolist(),
            "factor_z": factor_z.tolist(),
            "precision_theta": precision_theta.tolist(),
            "covariance_theta": covariance_theta.tolist(),
        },
        "telemetry_policy": {
            "native_divergence_statuses": ["not_exposed_by_kernel"],
        },
    }


def _phase2t_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2t_map_local_reference_handoff.v1",
        "decision": {
            "phase2t_map_local_reference_handoff_passed": True,
            "vetoes": [],
        },
        "phase2u_next_subplan_contract": {
            "candidate_grid": [
                {
                    "num_leapfrog_steps": 2,
                    "step_size": 0.785,
                    "trajectory_length_L_times_epsilon": 1.57,
                },
                {
                    "num_leapfrog_steps": 4,
                    "step_size": 0.3925,
                    "trajectory_length_L_times_epsilon": 1.57,
                },
                {
                    "num_leapfrog_steps": 8,
                    "step_size": 0.19625,
                    "trajectory_length_L_times_epsilon": 1.57,
                },
                {
                    "num_leapfrog_steps": 16,
                    "step_size": 0.098125,
                    "trajectory_length_L_times_epsilon": 1.57,
                },
            ],
            "selection_policy_predeclared": True,
            "all_trajectory_lengths_equal_1p57": True,
        },
        "telemetry_policy": {
            "native_divergence_statuses": ["not_exposed_by_kernel"],
        },
    }


def test_phase2u_settings_predeclare_equal_trajectory_grid_and_seeds() -> None:
    harness = _load_harness()

    settings = harness.Phase2UScreenSettings()
    payload = settings.payload()

    assert payload["candidate_grid"] == [
        {
            "num_leapfrog_steps": 2,
            "step_size": 0.785,
            "trajectory_length_L_times_epsilon": 1.57,
        },
        {
            "num_leapfrog_steps": 4,
            "step_size": 0.3925,
            "trajectory_length_L_times_epsilon": 1.57,
        },
        {
            "num_leapfrog_steps": 8,
            "step_size": 0.19625,
            "trajectory_length_L_times_epsilon": 1.57,
        },
        {
            "num_leapfrog_steps": 16,
            "step_size": 0.098125,
            "trajectory_length_L_times_epsilon": 1.57,
        },
    ]
    assert payload["seeds"] == [
        (20260709, 6301),
        (20260709, 6302),
        (20260709, 6303),
        (20260709, 6304),
    ]
    assert payload["selection_policy"] == "first_passing_candidate_in_predeclared_order"


def test_map_local_matrix_validation_keeps_factor_nonsymmetric() -> None:
    harness = _load_harness()

    checks = harness.validate_map_local_handoff_matrices(_phase2s_payload())

    assert checks["passed"] is True
    diagnostics = checks["diagnostics"]
    assert diagnostics["factor_z_reconstructs_covariance_z_max_abs_error"] == 0.0
    np.testing.assert_allclose(
        np.asarray(diagnostics["adapter_factor"]),
        np.diag([0.5, 2.0, 1.0, 0.25])
        @ np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.2, 1.2, 0.0, 0.0],
                [-0.1, 0.3, 0.8, 0.0],
                [0.05, -0.2, 0.4, 0.9],
            ]
        ),
    )


def test_bad_factor_reconstruction_vetoes_handoff() -> None:
    harness = _load_harness()

    checks = harness.validate_map_local_handoff_matrices(
        _phase2s_payload(bad_factor=True)
    )

    assert checks["passed"] is False
    assert "factor_z_reconstructs_covariance_z_failed" in checks["vetoes"]


def test_handoff_artifact_validation_checks_phase2t_candidate_grid() -> None:
    harness = _load_harness()

    precondition = harness.validate_handoff_artifacts(
        _phase2s_payload(),
        _phase2t_payload(),
    )

    assert precondition["passed"] is True


def test_candidate_gate_selects_first_passing_candidate_without_ranking() -> None:
    harness = _load_harness()
    settings = harness.Phase2UScreenSettings()
    rows = [
        {
            "candidate_index": 0,
            "status": "passed_hard_vetoes",
            "hard_vetoes": (),
            "num_leapfrog_steps": 2,
            "step_size": 0.785,
            "trajectory_length_L_times_epsilon": 1.57,
            "acceptance_rate": 1.0,
        },
        {
            "candidate_index": 1,
            "status": "passed_hard_vetoes",
            "hard_vetoes": (),
            "num_leapfrog_steps": 4,
            "step_size": 0.3925,
            "trajectory_length_L_times_epsilon": 1.57,
            "acceptance_rate": 0.7,
        },
        {
            "candidate_index": 2,
            "status": "passed_hard_vetoes",
            "hard_vetoes": (),
            "num_leapfrog_steps": 8,
            "step_size": 0.19625,
            "trajectory_length_L_times_epsilon": 1.57,
            "acceptance_rate": 0.6,
        },
        {
            "candidate_index": 3,
            "status": "failed_hard_vetoes",
            "hard_vetoes": ("nonfinite_log_accept_ratio",),
            "num_leapfrog_steps": 16,
            "step_size": 0.098125,
            "trajectory_length_L_times_epsilon": 1.57,
            "acceptance_rate": 0.8,
        },
    ]

    gate = harness.evaluate_candidate_gate(rows, settings)

    assert gate["selected_candidate"]["candidate_index"] == 1
    assert gate["selection_policy"] == "first_passing_candidate_in_predeclared_order"
    assert "candidate_3_nonfinite_log_accept_ratio" in gate["vetoes"]


def test_telemetry_policy_keeps_unavailable_native_divergence_nonclaim() -> None:
    harness = _load_harness()
    rows = [
        {
            "trace_summary": {
                "native_divergence": {
                    "available": False,
                    "status": "not_exposed_by_kernel",
                }
            }
        }
    ]

    telemetry = harness.telemetry_policy_payload(
        rows,
        _phase2s_payload(),
        _phase2t_payload(),
    )

    assert telemetry["zero_divergence_claim_made"] is False
    assert telemetry["unavailable_native_divergence_is_zero_divergence"] is False
    assert "unavailable is not zero divergences" in telemetry["native_divergence_interpretation"]


def test_json_ready_converts_tensorflow_tensors() -> None:
    harness = _load_harness()

    payload = harness.json_ready(
        {
            "scalar": tf.constant(1.25, dtype=tf.float64),
            "vector": tf.constant([1.0, 2.0], dtype=tf.float64),
        }
    )

    assert payload == {"scalar": 1.25, "vector": [1.0, 2.0]}
