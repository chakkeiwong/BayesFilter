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
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2w_importance_reference_agreement",
        HARNESS_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _phase2s_payload():
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


def _phase2u_payload(*, passed: bool = True, selected_index: int = 0):
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1",
        "decision": {
            "phase2u_retuned_map_local_hmc_screen_passed": passed,
            "vetoes": [],
            "selected_candidate": {
                "candidate_index": selected_index,
                "num_leapfrog_steps": 2,
                "step_size": 0.785,
                "trajectory_length_L_times_epsilon": 1.57,
            },
            "viable_for_phase3_gpu_xla_subplan": False,
        },
        "candidate_rows": [
            {
                "candidate_index": 0,
                "initial": {"u_new": [0.0, 0.0, 0.0, 0.0]},
            }
        ],
    }


def _phase2v_payload(*, mean=None, std=None, native_available: bool = False):
    if mean is None:
        mean = [0.1, -0.2, 0.3, -0.4]
    if std is None:
        std = [1.0, 1.1, 1.2, 1.3]
    native = (
        {"available": True, "count": 0, "values": [False]}
        if native_available
        else {
            "available": False,
            "status": "not_exposed_by_kernel",
            "nonclaim": "unavailable native divergence telemetry is not zero divergences",
        }
    )
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2v_longer_selected_map_local_screen.v1",
        "decision": {
            "phase2v_longer_selected_map_local_screen_passed": True,
            "vetoes": [],
            "selected_kernel": {
                "num_leapfrog_steps": 2,
                "phase2u_selected_candidate_index": 0,
                "step_size": 0.785,
                "trajectory_length_L_times_epsilon": 1.57,
            },
            "viable_for_phase3_gpu_xla_subplan": False,
            "viable_for_scalar_reference_posterior_agreement_subplan": True,
            "zero_divergence_claim_made": False,
        },
        "selected_kernel_row": {
            "status": "passed_hard_vetoes",
            "hard_vetoes": [],
            "acceptance_rate": 0.4,
            "initial": {"u_new": [0.0, 0.0, 0.0, 0.0]},
            "samples_summary": {
                "finite_sample_count": 128,
                "nonfinite_sample_count": 0,
                "mean_u_new": mean,
                "std_u_new": std,
            },
            "trace_summary": {
                "native_divergence": native,
            },
        },
    }


class _StandardNormalAdapter:
    parameter_dim = 4
    target_scope = "test:standard_normal"

    def log_prob_and_grad(self, z):
        tensor = tf.convert_to_tensor(z, dtype=tf.float64)
        return -0.5 * tf.reduce_sum(tf.square(tensor)), -tensor


def test_settings_lock_phase2w_reference_contract() -> None:
    harness = _load_harness()

    payload = harness.Phase2WReferenceSettings().payload()

    assert payload["proposal_sample_count"] == 1024
    assert payload["antithetic_base_sample_count"] == 512
    assert payload["seed"] == (20260709, 6501)
    assert payload["reference_ess_min"] == 128.0
    assert payload["reference_ess_ratio_min"] == 0.125
    assert payload["mean_threshold_formula"] == (
        "max(mean_abs_floor, mean_mcse_multiplier * reference_mean_mcse)"
    )


def test_antithetic_proposal_is_reproducible_and_paired() -> None:
    harness = _load_harness()
    settings = harness.Phase2WReferenceSettings(proposal_sample_count=8)

    proposal = harness.generate_antithetic_standard_normal_proposal(settings)

    samples = np.asarray(proposal["samples"], dtype=float)
    assert proposal["generated"] is True
    assert samples.shape == (8, 4)
    np.testing.assert_allclose(samples[:4] + samples[4:], 0.0)
    assert proposal["antithetic_pairing"]["max_abs_pair_sum"] == 0.0


def test_importance_reference_uses_sqrt_variance_over_ess_for_mcse() -> None:
    harness = _load_harness()
    settings = harness.Phase2WReferenceSettings(
        proposal_sample_count=8,
        reference_ess_min=1.0,
        reference_ess_ratio_min=0.01,
    )
    proposal = harness.generate_antithetic_standard_normal_proposal(settings)

    reference = harness.compute_importance_reference(
        _StandardNormalAdapter(),
        proposal,
        settings,
    )

    variance = np.asarray(reference["second_moment_variance_u_new"], dtype=float)
    mcse = np.asarray(reference["mean_mcse_u_new"], dtype=float)
    expected = np.sqrt(variance / float(reference["ess"]))
    np.testing.assert_allclose(mcse, expected)
    assert reference["reference_valid"] is True


def test_phase2w_handoff_vetoes_invalid_phase2u_selected_candidate() -> None:
    harness = _load_harness()

    precondition = harness.validate_phase2w_handoff(
        _phase2s_payload(),
        _phase2t_payload(),
        _phase2u_payload(selected_index=1),
        _phase2v_payload(),
        harness.Phase2WReferenceSettings(),
    )

    assert precondition["passed"] is False
    assert "phase2u_handoff_phase2u_selected_candidate_index_mismatch" in precondition["vetoes"]


def test_phase2v_payload_validation_requires_hmc_moments() -> None:
    harness = _load_harness()
    phase2v = harness.load_phase2v_module()

    validity = harness.validate_phase2v_payload(
        _phase2v_payload(mean=[np.nan, 0.0, 0.0, 0.0]),
        phase2v.Phase2VScreenSettings(),
    )

    assert validity["passed"] is False
    assert "hmc_mean_missing_or_nonfinite" in validity["vetoes"]


def test_agreement_gate_passes_mean_and_std_screens() -> None:
    harness = _load_harness()
    settings = harness.Phase2WReferenceSettings()
    reference = {
        "reference_valid": True,
        "mean_u_new": [0.0, 0.0, 0.0, 0.0],
        "std_u_new": [1.0, 1.0, 1.0, 1.0],
        "mean_mcse_u_new": [0.1, 0.1, 0.1, 0.1],
    }

    agreement = harness.compare_hmc_to_reference(
        _phase2v_payload(mean=[0.5, -0.5, 0.25, -0.25], std=[0.75, 1.25, 1.0, 1.5]),
        reference,
        settings,
    )

    assert agreement["passed"] is True
    assert agreement["vetoes"] == []


def test_agreement_gate_vetoes_mean_and_std_failures() -> None:
    harness = _load_harness()
    settings = harness.Phase2WReferenceSettings()
    reference = {
        "reference_valid": True,
        "mean_u_new": [0.0, 0.0, 0.0, 0.0],
        "std_u_new": [1.0, 1.0, 1.0, 1.0],
        "mean_mcse_u_new": [0.1, 0.1, 0.1, 0.1],
    }

    agreement = harness.compare_hmc_to_reference(
        _phase2v_payload(mean=[0.8, 0.0, 0.0, 0.0], std=[0.4, 1.0, 1.0, 1.0]),
        reference,
        settings,
    )

    assert agreement["passed"] is False
    assert "hmc_mean_component_0_outside_threshold" in agreement["vetoes"]
    assert "hmc_std_component_0_ratio_outside_interval" in agreement["vetoes"]


def test_agreement_not_interpreted_when_reference_invalid() -> None:
    harness = _load_harness()

    agreement = harness.compare_hmc_to_reference(
        _phase2v_payload(),
        {"reference_valid": False},
        harness.Phase2WReferenceSettings(),
    )

    assert agreement["evaluated"] is False
    assert "reference_invalid_agreement_not_interpreted" in agreement["vetoes"]


def test_telemetry_policy_keeps_unavailable_native_divergence_nonclaim() -> None:
    harness = _load_harness()

    telemetry = harness.telemetry_policy_payload(_phase2v_payload())

    assert telemetry["zero_divergence_claim_made"] is False
    assert telemetry["unavailable_native_divergence_is_zero_divergence"] is False
    assert telemetry["log_accept_threshold_used_as_native_divergence"] is False
