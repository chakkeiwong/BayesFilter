from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
HARNESS_PATH = (
    ROOT
    / "docs/benchmarks/"
    "benchmark_scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair_2026_07_09.py"
)


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "scalar_ssl_lstm_filtering_hmc_validation_phase2x_shifted_mixture_reference_repair",
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


def _phase2u_payload():
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2u_retuned_map_local_hmc_screen.v1",
        "decision": {
            "phase2u_retuned_map_local_hmc_screen_passed": True,
            "vetoes": [],
            "selected_candidate": {
                "candidate_index": 0,
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


def _phase2v_payload():
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
                "mean_u_new": [0.0, 0.0, 0.0, 0.0],
                "std_u_new": [1.0, 1.0, 1.0, 1.0],
            },
            "trace_summary": {
                "native_divergence": {
                    "available": False,
                    "status": "not_exposed_by_kernel",
                },
            },
        },
    }


def _phase2w_payload(*, extra_veto: bool = False, agreement_evaluated: bool = False):
    vetoes = ["reference_ess_below_threshold", "reference_ess_ratio_below_threshold"]
    if extra_veto:
        vetoes.append("target_log_prob_nonfinite")
    return {
        "schema_version": "scalar_ssl_lstm.filtering_hmc_validation_phase2w_importance_reference_agreement.v1",
        "decision": {
            "phase2w_importance_reference_agreement_passed": False,
            "reference_valid": False,
            "agreement_passed": False,
            "vetoes": vetoes,
            "reference_ess": 22.894679726459746,
            "reference_ess_ratio": 0.022358085670370845,
            "zero_divergence_claim_made": False,
        },
        "importance_reference": {
            "reference_valid": False,
            "vetoes": vetoes,
            "ess": 22.894679726459746,
            "ess_ratio": 0.022358085670370845,
            "mean_u_new": [
                0.16900152112527375,
                0.34590014590251295,
                0.47216707577215133,
                -0.3362900480743778,
            ],
            "std_u_new": [
                1.1289232726542155,
                1.3947178163994365,
                1.7877962561383989,
                1.7764811837333756,
            ],
            "log_weight_summary": {
                "finite_count": 1024,
                "nonfinite_count": 0,
            },
            "target_score_norm_summary": {
                "nonfinite_count": 0,
            },
        },
        "hmc_reference_agreement": {
            "evaluated": agreement_evaluated,
        },
    }


def test_phase2x_settings_lock_mixture_contract() -> None:
    harness = _load_harness()

    settings = harness.Phase2XReferenceSettings()
    payload = settings.payload()

    assert payload["proposal_sample_count"] == 2048
    assert payload["standard_component_count"] == 512
    assert payload["shifted_component_count"] == 1536
    assert payload["seed"] == [20260709, 6601]
    assert payload["reference_ess_min"] == 256.0
    assert payload["reference_ess_ratio_min"] == 0.125
    np.testing.assert_allclose(
        payload["shifted_scale"],
        np.clip(1.25 * np.asarray(payload["phase2w_pilot_std"], dtype=float), 0.75, 3.0),
    )


def test_shifted_mixture_proposal_has_predeclared_counts_and_pairing() -> None:
    harness = _load_harness()
    settings = harness.Phase2XReferenceSettings(proposal_sample_count=16)

    proposal = harness.generate_shifted_mixture_proposal(settings)

    assert proposal["generated"] is True
    assert proposal["component_counts"] == {"standard": 4, "shifted": 12}
    samples = np.asarray(proposal["samples"], dtype=float)
    assert samples.shape == (16, 4)
    assert proposal["antithetic_pairing"]["standard_max_abs_pair_sum"] == 0.0
    assert proposal["antithetic_pairing"]["shifted_centered_max_abs_pair_sum"] < 1.0e-12
    assert proposal["proposal_parameter_source"] == "phase2w_importance_reference_pilot_only_not_hmc_moments"


def test_shifted_mixture_log_prob_matches_manual_logsumexp_at_center() -> None:
    harness = _load_harness()
    settings = harness.Phase2XReferenceSettings()
    sample = harness.PHASE2W_REFERENCE_MEAN.reshape(1, 4)

    actual = harness.shifted_mixture_log_prob(sample, settings)[0]

    standard = harness.phase2w.standard_normal_log_prob(sample)[0]
    scale = harness.shifted_scale_from_settings(settings)
    shifted = -np.sum(np.log(scale)) - 0.5 * 4 * np.log(2.0 * np.pi)
    a = np.log(0.25) + standard
    b = np.log(0.75) + shifted
    expected = max(a, b) + np.log(np.exp(a - max(a, b)) + np.exp(b - max(a, b)))
    np.testing.assert_allclose(actual, expected)


def test_phase2x_handoff_accepts_only_phase2w_ess_failure() -> None:
    harness = _load_harness()

    precondition = harness.validate_phase2x_handoff(
        _phase2s_payload(),
        _phase2t_payload(),
        _phase2u_payload(),
        _phase2v_payload(),
        _phase2w_payload(),
        harness.Phase2XReferenceSettings(),
    )

    assert precondition["passed"] is True
    assert precondition["proposal_parameter_contract"]["uses_hmc_moments"] is False


def test_phase2x_handoff_vetoes_phase2w_extra_failure() -> None:
    harness = _load_harness()

    precondition = harness.validate_phase2x_handoff(
        _phase2s_payload(),
        _phase2t_payload(),
        _phase2u_payload(),
        _phase2v_payload(),
        _phase2w_payload(extra_veto=True),
        harness.Phase2XReferenceSettings(),
    )

    assert precondition["passed"] is False
    assert "phase2w_failure_not_limited_to_ess_thresholds" in precondition["vetoes"]


def test_phase2x_handoff_vetoes_interpreted_phase2w_agreement() -> None:
    harness = _load_harness()

    precondition = harness.validate_phase2x_handoff(
        _phase2s_payload(),
        _phase2t_payload(),
        _phase2u_payload(),
        _phase2v_payload(),
        _phase2w_payload(agreement_evaluated=True),
        harness.Phase2XReferenceSettings(),
    )

    assert precondition["passed"] is False
    assert "phase2w_agreement_was_interpreted" in precondition["vetoes"]
