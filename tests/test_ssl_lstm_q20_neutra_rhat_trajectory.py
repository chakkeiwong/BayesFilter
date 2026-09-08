"""Focused tests for the SSL-LSTM q=20 R-hat trajectory diagnostic."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_neutra_rhat_trajectory_diagnostic_2026_08_21.py"
)


def _module():
    spec = importlib.util.spec_from_file_location(
        "ssl_lstm_q20_neutra_rhat_trajectory", RUNNER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _trajectory_row(draws: int, value: float) -> dict:
    return {
        "checkpoint_draws": draws,
        "cumulative": {"observation_weight_rhat": {"maximum": value}},
        "recent_window": {
            "observation_weight_rhat": {"maximum": value + 0.01}
        },
    }


def test_frozen_argument_and_budget_contract() -> None:
    runner = _module()
    args = SimpleNamespace(
        device="1",
        time_cap_seconds=runner.INTERNAL_WORK_CAP_SECONDS,
        output_root=runner.DEFAULT_OUTPUT,
    )
    runner._validate_args(args)
    forecast = runner._forecast()
    assert forecast["requested_states_per_chain"] == 4064
    assert forecast["remaining_aggregate_grant_seconds"] == pytest.approx(
        41519.345206590995
    )
    assert forecast["aggregate_wall_at_external_cap_seconds"] == pytest.approx(
        59280.654793409005
    )
    assert forecast["fits_internal_with_closeout"] is True
    assert forecast["fits_prior_hmc_envelope"] is True
    assert forecast["fits_aggregate_grant"] is True

    drifted = SimpleNamespace(**vars(args))
    drifted.time_cap_seconds = 40000.0
    with pytest.raises(SystemExit, match="cap is frozen"):
        runner._validate_args(drifted)
    drifted = SimpleNamespace(**vars(args))
    drifted.output_root = runner.R2_ROOT
    with pytest.raises(SystemExit, match="output root is frozen"):
        runner._validate_args(drifted)


def test_immutable_r2_inputs_and_failed_kernel_identity() -> None:
    runner = _module()
    continuation, training, baseline, budget = runner._validated_inputs()
    assert continuation.TARGET_SIGNATURE
    assert training["status"] == "TRAINING_SCREEN_AND_FROZEN_AUDIT_COMPLETED"
    assert baseline["r2_result"]["status"] == "HMC_NO_CANDIDATE_ADMITTED"
    assert baseline["tuning"]["final_status"] == "no_viable_candidate"
    assert baseline["candidate"]["hard_vetoes"] == [
        "verification_modern_rank_folded_rhat_failed"
    ]
    assert baseline["kernel"]["step_size"] == runner.STEP_SIZE
    assert baseline["kernel"]["num_leapfrog_steps"] == runner.NUM_LEAPFROG_STEPS
    assert baseline["kernel"]["mass_policy"] == "fixed_identity_z"
    assert baseline["kernel"]["proposal_dynamics_identity"] == (
        "exact_transformed_gradient"
    )
    assert baseline["kernel"]["identity_z_mass_artifact_payload"]["covariance"] == [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    assert baseline["verification_config"]["seed"] == list(runner.HMC_SEED)
    assert len(baseline["initial_state_bank"]) == runner.CHAIN_COUNT
    assert budget["launcher_attempt_limit"] == 2
    assert budget["gpu_initializing_attempt_limit"] == 1
    assert budget["retry_index"] == 1
    assert budget["failed_pre_tensorflow_launch"][
        "gpu_or_tensorflow_initialized"
    ] is False
    assert Path(budget["failed_pre_tensorflow_launch"]["root"]) == (
        runner.FAILED_LAUNCH_ROOT
    )
    assert runner.DEFAULT_OUTPUT != runner.FAILED_LAUNCH_ROOT
    assert budget["predictive_reserve_used"] is False


def test_route_classification_is_non_promotional_and_complete() -> None:
    runner = _module()
    audit = runner._route_policy_audit()
    assert audit["passed"] is True
    assert audit["discovered_route_count"] == audit["classified_route_count"]
    assert audit["current_route"]["classification"] == (
        "smoke_mechanics_or_reference"
    )
    assert audit["claim_bearing_route_unchanged"] is True


def test_trajectory_summary_reports_direction_without_extrapolation() -> None:
    runner = _module()
    rows = [
        _trajectory_row(500, 1.30),
        _trajectory_row(1000, 1.20),
        _trajectory_row(1500, 1.22),
        _trajectory_row(2000, 1.15),
        _trajectory_row(2500, 1.10),
        _trajectory_row(3000, 1.08),
        _trajectory_row(3500, 1.09),
        _trajectory_row(4000, 1.05),
    ]
    result = runner._trajectory_summary(rows, baseline_checkpoint=2000)
    assert result["observation_weight_rhat_change_baseline_to_endpoint"] == (
        pytest.approx(-0.10)
    )
    assert result["observation_weight_rhat_dropped_baseline_to_endpoint"] is True
    assert result["adjacent_decrease_count"] == 5
    assert result["adjacent_increase_count"] == 2
    assert result["all_adjacent_changes_nonpositive"] is False
    assert result["role"] == "descriptive_trajectory_without_extrapolation"


def test_mode_summary_rejects_pooled_sign_locked_chains() -> None:
    runner = _module()
    labels = tf.constant(
        (
            (0, 1, 0, 1),
            (0, 1, 0, 1),
            (0, 1, 0, 1),
            (0, 1, 0, 1),
        ),
        tf.int32,
    )
    report = runner._mode_summary(tf, labels)
    assert report["global_region_counts"] == [8, 8]
    assert report["chain_transition_counts"] == [0, 0, 0, 0]
    assert report["passed"] is False

    crossing = tf.constant(
        (
            (0, 1, 0, 1),
            (1, 0, 1, 0),
            (0, 1, 0, 1),
            (1, 0, 1, 0),
        ),
        tf.int32,
    )
    report = runner._mode_summary(tf, crossing)
    assert report["chain_transition_counts"] == [3, 3, 3, 3]
    assert report["passed"] is True


def test_checkpoint_diagnostics_include_cumulative_recent_and_sign_evidence() -> None:
    runner = _module()
    draws = 40
    chains = runner.CHAIN_COUNT
    parameters = runner.PARAMETER_COUNT
    physical = tf.random.stateless_normal(
        (draws, chains, parameters), seed=(71, 19), dtype=tf.float64
    )
    log_accept = -tf.abs(
        tf.random.stateless_normal((draws, chains), seed=(71, 20), dtype=tf.float64)
    )
    trace = {
        "log_accept_ratio": log_accept,
        "is_accepted": log_accept > -0.8,
    }
    payload = runner._checkpoint_diagnostics(
        tf,
        physical,
        trace,
        checkpoints=(20, 40),
        recent_window=20,
        baseline_checkpoint=20,
    )
    assert payload["checkpoint_schedule"] == [20, 40]
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["cumulative"]["draw_count_per_chain"] == 20
    assert payload["rows"][1]["recent_window"]["draw_count_per_chain"] == 20
    assert payload["rows"][1]["cumulative"]["mode_mixing"][
        "chain_transition_counts"
    ]
    assert payload["endpoint"]["sign_indicator_rhat"]["parameter_count"] == 1
    assert payload["posterior_admission"] is False


def test_replay_tieout_requires_exact_saved_summaries() -> None:
    runner = _module()
    modern = {
        "rank_normalized_split_rhat": [1.0, 1.1],
        "folded_rank_normalized_split_rhat": [1.0, 1.2],
    }
    acceptance = {
        "acceptance_rate": 0.75,
        "acceptance_probability_by_chain": [0.7, 0.8],
        "binary_acceptance_rate": 0.74,
        "binary_acceptance_by_chain": [0.7, 0.78],
    }
    baseline = {
        "modern_rhat": modern,
        "verification_diagnostics": acceptance,
    }
    checkpoint = {
        "rows": [
            {
                "checkpoint_draws": runner.BASELINE_CHECKPOINT,
                "cumulative": {
                    "physical_rhat": modern,
                    "acceptance": acceptance,
                },
            }
        ]
    }
    tied = runner._replay_tieout(baseline, checkpoint)
    assert tied["deterministic_summary_replay_passed"] is True
    assert tied["raw_prefix_identity_proved"] is False

    changed = dict(modern)
    changed["rank_normalized_split_rhat"] = [1.0, 1.1000000000000003]
    checkpoint["rows"][0]["cumulative"]["physical_rhat"] = changed
    untied = runner._replay_tieout(baseline, checkpoint)
    assert untied["deterministic_summary_replay_passed"] is False
    assert untied["rhat_maximum_absolute_float_residual"] > 0.0


def test_raw_archive_roundtrip_and_receipts(tmp_path: Path) -> None:
    runner = _module()
    latent = tf.reshape(
        tf.range(3 * 4 * 4, dtype=tf.float64),
        (3, 4, 4),
    )
    physical = latent / 10.0
    trace = {
        "is_accepted": tf.ones((3, 4), tf.bool),
        "log_accept_ratio": tf.zeros((3, 4), tf.float64),
        "target_status_telemetry": {
            "status_code": tf.zeros((3, 4), tf.int32),
            "valid_pre_regularized_score": tf.ones((3, 4), tf.bool),
        },
    }
    archive = runner._archive_raw(
        tf,
        tmp_path,
        latent=latent,
        physical=physical,
        trace=trace,
    )
    assert archive["verified_after_write"] is True
    assert archive["posterior_eligible"] is False
    assert archive["receipts"]["latent_samples"]["shape"] == [3, 4, 4]
    assert archive["receipts"]["observation_weight_sign_labels"]["shape"] == [
        3,
        4,
    ]
    assert (tmp_path / "raw-archive.json").is_file()


def test_runner_has_one_diagnostic_call_and_no_promotion_path() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    assert "NUM_RESULTS = 4000" in source
    assert "CHECKPOINTS = (500, 1000, 1500, 2000, 2500, 3000, 3500, 4000)" in source
    assert source.count("run_fixed_transport_full_chain_tfp_hmc(adapter, initial, config)") == 1
    assert "run_sequential_neutra_hmc(" not in source
    assert "HMC_ADMITTED_FOR_PREDICTIVE" not in source
    assert '"posterior_admitted": False' in source
    assert '"predictive_authorized": False' in source
    assert "import numpy" not in source
    assert "tf.io.serialize_tensor" in source
    assert "refusing to reuse diagnostic output root" in source
    env = source.index('os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"')
    validate_inputs = source.index("_validated_inputs()", source.index("def _execute"))
    tensorflow_import = source.index("import tensorflow as tf", source.index("def _execute"))
    assert env < validate_inputs < tensorflow_import
    root_path = source.index("sys.path.insert(0, str(ROOT))")
    route_import = source.index(
        "from bayesfilter.inference.neutra_hmc_policy import"
    )
    assert root_path < route_import
