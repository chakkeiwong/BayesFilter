from __future__ import annotations

import os
import json

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    _evaluate_retained_target_health,
    evaluate_hmc_acceptance_evidence,
    hmc_acceptance_evidence_from_payload,
    hmc_acceptance_evidence_v2_migration_view,
    hmc_acceptance_evidence_v3_migration_view,
    summarize_hmc_tuning_telemetry,
    target_status_telemetry_has_failure,
)


def _moving_samples(draw_count: int = 64) -> np.ndarray:
    draw = np.arange(draw_count, dtype=float)[:, None, None]
    chain = np.arange(4, dtype=float)[None, :, None]
    direction = np.array([1.0, -0.5], dtype=float)[None, None, :]
    return draw * direction + chain


def test_target_status_validator_accepts_core_only_and_rejects_partial_conditioning():
    payload = {
        "status_code": np.zeros(4, dtype=np.int32),
        "valid_pre_regularized_score": np.ones(4, dtype=bool),
        "floor_count_value": np.zeros(4, dtype=np.int32),
    }
    assert target_status_telemetry_has_failure(payload, expected_shape=(4,)) is False
    with pytest.raises(ValueError, match="both present or both absent"):
        target_status_telemetry_has_failure(
            {**payload, "min_innovation_eigenvalue": np.ones(4)},
            expected_shape=(4,),
        )


def _evidence(
    probability: np.ndarray | float,
    *,
    samples: np.ndarray | None = None,
    **kwargs,
):
    values = np.asarray(probability, dtype=float)
    if values.ndim == 0:
        values = np.full((64, 4), float(values))
    if samples is None:
        samples = _moving_samples(values.shape[0])
    return evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(values),
        is_accepted=np.ones(values.shape, dtype=bool),
        policy=HMCAcceptancePolicy(),
        **kwargs,
    )


def test_retained_target_health_separates_nonfinite_score_from_shape_invalidity() -> None:
    samples = np.arange(24, dtype=float).reshape(3, 4, 2)

    class _NonfiniteScoreAdapter:
        def log_prob_and_grad(self, theta):
            values = tf.convert_to_tensor(theta, dtype=tf.float64)
            return (
                -0.5 * tf.reduce_sum(tf.square(values), axis=-1),
                tf.fill(tf.shape(values), tf.constant(float("nan"), tf.float64)),
            )

    class _MalformedShapeAdapter:
        def log_prob_and_grad(self, theta):
            values = tf.convert_to_tensor(theta, dtype=tf.float64)
            return tf.constant(0.0, tf.float64), -values

    local = _evaluate_retained_target_health(
        adapter=_NonfiniteScoreAdapter(),
        samples=samples,
    )
    shared = _evaluate_retained_target_health(
        adapter=_MalformedShapeAdapter(),
        samples=samples,
    )

    assert local["shared_invalidity_reasons"] == ()
    assert local["candidate_data_invalidity_reasons"] == (
        "nonfinite_target_score",
    )
    assert local["target_value_finite"] is True
    assert local["target_score_finite"] is False
    assert local["evaluated_draw_count"] == 1
    assert shared["shared_invalidity_reasons"] == (
        "target_value_score_shape_invalid",
    )
    assert shared["candidate_data_invalidity_reasons"] == ()


def test_retained_target_health_types_unexpected_target_exception_as_shared() -> None:
    class _FailingAdapter:
        def log_prob_and_grad(self, _theta):
            raise RuntimeError("unexpected target callback failure")

    result = _evaluate_retained_target_health(
        adapter=_FailingAdapter(),
        samples=np.zeros((1, 4, 2), dtype=float),
    )

    assert result["shared_invalidity_reasons"] == ("shared_callback_invalid",)
    assert result["candidate_data_invalidity_reasons"] == ()
    assert result["target_value_finite"] is False
    assert result["target_score_finite"] is False
    assert result["evaluated_draw_count"] == 0


@pytest.mark.parametrize(
    ("telemetry_mode", "expected_scope", "expected_reason", "expected_count"),
    (
        ("missing", "shared", "required_target_status_telemetry_missing", None),
        ("malformed", "shared", "shared_schema_invalid", None),
        ("exception", "shared", "shared_callback_invalid", None),
        ("nonvalid", "candidate", "target_status_telemetry_failure", 1),
    ),
)
def test_retained_target_health_types_target_status_failures(
    telemetry_mode: str,
    expected_scope: str,
    expected_reason: str,
    expected_count: int | None,
) -> None:
    class _Adapter:
        def log_prob_and_grad(self, theta):
            values = tf.convert_to_tensor(theta, dtype=tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(values), axis=-1), -values

        if telemetry_mode != "missing":

            def target_status_telemetry(self, theta):
                if telemetry_mode == "exception":
                    raise RuntimeError("unexpected target-status callback failure")
                shape = tf.shape(theta)[:-1]
                payload = {
                    "status_code": tf.zeros(shape, tf.int32),
                    "valid_pre_regularized_score": tf.ones(shape, tf.bool),
                    "floor_count_value": tf.zeros(shape, tf.int32),
                    "min_innovation_eigenvalue": tf.ones(shape, tf.float64),
                    "innovation_condition_estimate": tf.ones(shape, tf.float64),
                }
                if telemetry_mode == "malformed":
                    payload.pop("floor_count_value")
                elif telemetry_mode == "nonvalid":
                    payload["status_code"] = tf.tensor_scatter_nd_update(
                        payload["status_code"],
                        indices=tf.zeros((1, tf.rank(payload["status_code"])), tf.int32),
                        updates=tf.ones((1,), tf.int32),
                    )
                return payload

    result = _evaluate_retained_target_health(
        adapter=_Adapter(),
        samples=np.zeros((1, 4, 2), dtype=float),
        target_status_trace_policy="per_chain_step",
    )

    reasons = result[
        "shared_invalidity_reasons"
        if expected_scope == "shared"
        else "candidate_data_invalidity_reasons"
    ]
    assert reasons == (expected_reason,)
    assert result["target_status_failure_count"] == expected_count


def test_tuning_telemetry_separates_mean_probability_from_binary_acceptance() -> None:
    samples = tf.constant(
        [
            [[0.0], [1.0]],
            [[1.0], [1.0]],
            [[2.0], [1.0]],
            [[3.0], [1.0]],
        ],
        dtype=tf.float64,
    )
    log_accept = tf.math.log(
        tf.constant([[0.8, 0.4], [0.6, 0.4], [1.0, 0.4], [0.2, 0.4]], tf.float64)
    )
    accepted = tf.constant(
        [[True, False], [False, True], [True, False], [False, False]]
    )

    result = summarize_hmc_tuning_telemetry(
        samples=samples,
        log_accept_ratio=log_accept,
        is_accepted=accepted,
    )

    np.testing.assert_allclose(result["mean_acceptance_probability_by_chain"], [0.65, 0.4])
    np.testing.assert_allclose(result["binary_acceptance_rate_by_chain"], [0.5, 0.25])
    np.testing.assert_allclose(result["movement_rate_by_chain"], [1.0, 0.0])
    np.testing.assert_allclose(result["repeated_state_fraction_by_chain"], [0.0, 1.0])
    assert result["schema"] == "bayesfilter.hmc_tuning_telemetry.v4"
    assert result["energy_proxy_role"].startswith("absolute_log_accept_ratio")


def test_tuning_telemetry_supports_single_chain_shape() -> None:
    result = summarize_hmc_tuning_telemetry(
        samples=tf.constant([[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]], tf.float64),
        log_accept_ratio=tf.constant([-0.1, -0.2, 0.0], tf.float64),
        is_accepted=tf.constant([True, True, False]),
    )

    assert int(result["chain_count"].numpy()) == 1
    assert int(result["draw_count"].numpy()) == 3
    assert result["mean_acceptance_probability_by_chain"].shape == (1,)


def test_tuning_telemetry_movement_matches_coordinate_wise_admission() -> None:
    samples = np.zeros((8, 2, 3), dtype=float)
    samples[:, :, 1:] = np.arange(8, dtype=float)[:, None, None]

    result = summarize_hmc_tuning_telemetry(
        samples=samples,
        log_accept_ratio=np.zeros((8, 2), dtype=float),
        is_accepted=np.ones((8, 2), dtype=bool),
    )

    np.testing.assert_allclose(result["movement_rate_by_chain"], [0.0, 0.0])
    np.testing.assert_allclose(
        result["repeated_state_fraction_by_chain"], [1.0, 1.0]
    )


def test_tuning_telemetry_marks_single_draw_movement_unavailable_without_nan() -> None:
    result = summarize_hmc_tuning_telemetry(
        samples=tf.constant([[0.0, 0.0]], tf.float64),
        log_accept_ratio=tf.constant([0.0], tf.float64),
        is_accepted=tf.constant([True]),
    )

    assert result["movement_rate_by_chain"].shape == (0,)
    assert result["repeated_state_fraction_by_chain"].shape == (0,)
    assert bool(result["movement_summary_available"].numpy()) is False


def test_tuning_telemetry_records_nonfinite_log_accept_without_hiding_it() -> None:
    result = summarize_hmc_tuning_telemetry(
        samples=tf.constant([[0.0], [0.0]], tf.float64),
        log_accept_ratio=tf.constant([0.0, float("nan")], tf.float64),
        is_accepted=tf.constant([True, False]),
    )

    assert int(result["log_accept_ratio_nonfinite_count"].numpy()) == 1
    assert np.isnan(float(result["mean_acceptance_probability"].numpy()))


def test_tuning_telemetry_rejects_mismatched_chain_shape() -> None:
    with pytest.raises(ValueError, match="chain counts must agree"):
        summarize_hmc_tuning_telemetry(
            samples=tf.zeros((4, 2, 1), tf.float64),
            log_accept_ratio=tf.zeros((4, 3), tf.float64),
            is_accepted=tf.zeros((4, 3), tf.bool),
        )


@pytest.mark.parametrize("period", (2, 3, 4, 16))
def test_tensorflow_and_numpy_path_return_summaries_match(period: int) -> None:
    chain_offsets = np.arange(4, dtype=float)[:, None]
    base = np.concatenate((chain_offsets, -chain_offsets), axis=1)
    states = tuple(
        base + np.array([float(index), -0.5 * index])
        for index in range(period)
    )
    samples = np.stack((states * (64 // period)) + states[: 64 % period], axis=0)
    telemetry = summarize_hmc_tuning_telemetry(
        samples=samples,
        log_accept_ratio=np.full((64, 4), np.log(0.70)),
        is_accepted=np.ones((64, 4), dtype=bool),
    )
    evidence = _evidence(0.70, samples=samples)

    np.testing.assert_allclose(
        telemetry["path_return_fraction_by_chain"].numpy(),
        evidence.path_return_fraction_by_chain,
    )
    assert telemetry["diagnostic_roles"]["path_return_fraction_by_chain"] == (
        "promotion_veto_and_resonance_repair_trigger"
    )


def test_tensorflow_and_numpy_nonperiodic_path_return_summaries_match() -> None:
    samples = _moving_samples()
    telemetry = summarize_hmc_tuning_telemetry(
        samples=samples,
        log_accept_ratio=np.full((64, 4), np.log(0.70)),
        is_accepted=np.ones((64, 4), dtype=bool),
    )
    evidence = _evidence(0.70, samples=samples)

    np.testing.assert_allclose(
        telemetry["path_return_fraction_by_chain"].numpy(),
        evidence.path_return_fraction_by_chain,
    )


def test_acceptance_policy_rejects_unsupported_interval_contracts() -> None:
    with pytest.raises(ValueError, match="exactly four chains"):
        HMCAcceptancePolicy(chain_count=3)
    with pytest.raises(ValueError, match="four blocks"):
        HMCAcceptancePolicy(block_count=3)
    with pytest.raises(ValueError, match="90% interval"):
        HMCAcceptancePolicy(confidence_level=0.95)
    with pytest.raises(ValueError, match="target must lie"):
        HMCAcceptancePolicy(target=0.8)
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        HMCAcceptancePolicy(min_movement_rate=1.1)
    with pytest.raises(ValueError, match=r"inside \[0, 1\]"):
        HMCAcceptancePolicy(max_repeated_state_fraction=-0.1)

    payload = HMCAcceptancePolicy().payload()
    assert payload["min_movement_rate"] == pytest.approx(0.05)
    assert payload["max_repeated_state_fraction"] == pytest.approx(0.95)
    assert payload["min_normalized_return_displacement"] == pytest.approx(1.0e-4)
    assert payload["max_abs_log_accept_energy_proxy"] == pytest.approx(1000.0)


@pytest.mark.parametrize(
    ("probability", "decision"),
    [(0.40, "repair_step_lower"), (0.70, "passed"), (0.90, "repair_step_higher")],
)
def test_acceptance_evidence_exact_directional_decisions(
    probability: float,
    decision: str,
) -> None:
    evidence = _evidence(probability)

    assert evidence.decision == decision
    assert evidence.usable_decisions_per_chain == 64
    assert evidence.excluded_remainder_per_chain == 0
    assert evidence.standard_error is None
    assert evidence.realized_acceptance_rate == pytest.approx(1.0)
    assert evidence.realized_acceptance_rate_by_chain == pytest.approx((1.0,) * 4)


def test_acceptance_evidence_uses_chain_mean_hoeffding_interval() -> None:
    block_means = np.array(
        [
            [0.60, 0.70, 0.65, 0.75],
            [0.62, 0.72, 0.67, 0.77],
            [0.58, 0.68, 0.63, 0.73],
            [0.61, 0.71, 0.66, 0.76],
        ]
    )
    probabilities = np.concatenate(
        [np.repeat(block_means[:, index][None, :], 16, axis=0) for index in range(4)],
        axis=0,
    )

    evidence = _evidence(probabilities)
    chain_means = np.mean(block_means, axis=1)
    half_width = np.sqrt(np.log(2.0 / 0.10) / (2.0 * 4))
    pooled = np.mean(chain_means)

    assert evidence.chain_mean_uncertainty_interval == pytest.approx(
        (max(0.0, pooled - half_width), min(1.0, pooled + half_width))
    )
    assert (
        evidence.chain_mean_uncertainty_method
        == "two_sided_hoeffding_independent_chains"
    )
    assert evidence.chain_mean_uncertainty_level == pytest.approx(0.90)
    assert evidence.standard_error is None
    assert evidence.decision == "inconclusive_evidence"


def test_acceptance_evidence_detects_heterogeneous_chain_conflict() -> None:
    probabilities = np.tile(np.array([0.50, 0.55, 0.85, 0.90]), (64, 1))

    evidence = _evidence(probabilities)

    assert evidence.decision == "inconclusive_conflict"


def test_chain_conflict_precedes_sticking_trajectory_repair() -> None:
    probabilities = np.tile(np.array([0.50, 0.55, 0.85, 0.90]), (64, 1))
    samples = _moving_samples()
    samples[:, 0, :] = 0.0

    evidence = _evidence(probabilities, samples=samples)

    assert evidence.acceptance_decision == "inconclusive_conflict"
    assert evidence.tuning_repair_triggers == ()
    assert "movement_gate_failed" in evidence.candidate_promotion_vetoes


def test_acceptance_evidence_four_results_are_inconclusive() -> None:
    evidence = _evidence(
        np.full((4, 4), 0.70),
        samples=_moving_samples(4),
    )

    assert evidence.decision == "inconclusive_evidence"
    assert evidence.interval is None
    assert evidence.explanatory_notes == ("minimum_chain_or_decision_evidence_missing",)


def test_acceptance_evidence_one_result_has_no_nan_movement_summary() -> None:
    evidence = _evidence(
        np.full((1, 4), 0.70),
        samples=_moving_samples(1),
    )

    assert evidence.decision == "inconclusive_evidence"
    assert evidence.movement_rate_by_chain == ()
    assert evidence.repeated_state_fraction_by_chain == ()
    assert "NaN" not in json.dumps(evidence.payload(), allow_nan=False)


def test_acceptance_evidence_sticking_overrides_high_step_repair() -> None:
    evidence = _evidence(0.90, samples=np.zeros((64, 4, 2), dtype=float))

    assert evidence.decision == "repair_trajectory"
    assert evidence.movement_rate_by_chain == pytest.approx((0.0,) * 4)
    assert evidence.tuning_repair_triggers == ("trajectory:repair_movement",)
    assert evidence.candidate_promotion_vetoes == ("movement_gate_failed",)


def test_supported_low_acceptance_precedes_rejection_induced_sticking() -> None:
    evidence = _evidence(0.40, samples=np.zeros((64, 4, 2), dtype=float))

    assert evidence.acceptance_decision == "repair_step_lower"
    assert evidence.repair_direction == "lower_epsilon"
    assert evidence.tuning_repair_triggers == ("step_size:lower_epsilon",)
    assert evidence.candidate_promotion_vetoes == ("movement_gate_failed",)
    assert evidence.promotion_eligible is False


def test_acceptance_evidence_detects_exact_period_two_return_path() -> None:
    chain_offsets = np.arange(4, dtype=float)[:, None]
    state_a = np.concatenate((chain_offsets, -chain_offsets), axis=1)
    state_b = state_a + np.array([1.0, -0.5])
    samples = np.stack((state_a, state_b) * 32, axis=0)

    evidence = _evidence(0.70, samples=samples)

    assert evidence.movement_rate_by_chain == pytest.approx((1.0,) * 4)
    assert evidence.repeated_state_fraction_by_chain == pytest.approx((0.0,) * 4)
    assert all(
        value > evidence.policy.min_normalized_return_displacement
        for value in evidence.normalized_return_displacement_by_chain
    )
    assert evidence.path_return_fraction_by_chain == pytest.approx((1.0,) * 4)
    assert evidence.acceptance_decision == "repair_trajectory"
    assert evidence.tuning_repair_triggers == ("trajectory:repair_resonance",)
    assert "path_return_resonance_detected" in evidence.candidate_promotion_vetoes
    assert evidence.promotion_eligible is False


def test_supported_low_acceptance_precedes_period_two_trajectory_alert() -> None:
    chain_offsets = np.arange(4, dtype=float)[:, None]
    state_a = np.concatenate((chain_offsets, -chain_offsets), axis=1)
    state_b = state_a + np.array([1.0, -0.5])
    samples = np.stack((state_a, state_b) * 32, axis=0)

    evidence = _evidence(0.40, samples=samples)

    assert evidence.acceptance_decision == "repair_step_lower"
    assert evidence.repair_direction == "lower_epsilon"
    assert evidence.tuning_repair_triggers == ("step_size:lower_epsilon",)
    assert evidence.candidate_promotion_vetoes == (
        "path_return_resonance_detected",
    )


@pytest.mark.parametrize("period", (3, 4, 16))
def test_acceptance_evidence_detects_short_period_return_path(period: int) -> None:
    chain_offsets = np.arange(4, dtype=float)[:, None]
    base = np.concatenate((chain_offsets, -chain_offsets), axis=1)
    states = tuple(base + np.array([float(index), -0.5 * index]) for index in range(period))
    samples = np.stack((states * (64 // period)) + states[: 64 % period], axis=0)

    evidence = _evidence(0.70, samples=samples)

    assert evidence.movement_rate_by_chain == pytest.approx((1.0,) * 4)
    assert evidence.path_return_fraction_by_chain == pytest.approx((1.0,) * 4)
    assert evidence.acceptance_decision == "repair_trajectory"
    assert evidence.tuning_repair_triggers == ("trajectory:repair_resonance",)
    assert "path_return_resonance_detected" in evidence.candidate_promotion_vetoes


def test_path_return_gate_does_not_flag_nonperiodic_moving_chain() -> None:
    evidence = _evidence(0.70, samples=_moving_samples())
    translated = _evidence(
        0.70,
        samples=_moving_samples() + np.array([1.0e12, -1.0e12]),
    )

    assert evidence.path_return_fraction_by_chain == pytest.approx((0.0,) * 4)
    assert translated.path_return_fraction_by_chain == pytest.approx((0.0,) * 4)
    assert evidence.acceptance_decision == "passed"
    assert translated.acceptance_decision == "passed"
    assert "path_return_resonance_detected" not in evidence.candidate_promotion_vetoes


def test_acceptance_evidence_does_not_treat_unavailable_divergence_as_zero() -> None:
    evidence = _evidence(
        0.70,
        native_divergence_status="not_exposed_by_kernel",
        native_divergence_count=None,
    )

    assert evidence.decision == "passed"
    assert evidence.native_divergence_count is None


def test_acceptance_evidence_positive_native_divergence_is_local_veto() -> None:
    evidence = _evidence(
        0.70,
        native_divergence_status="available",
        native_divergence_count=1,
    )

    assert evidence.evidence_validity == "valid"
    assert evidence.acceptance_decision == "passed"
    assert evidence.promotion_eligible is False
    assert "native_divergence_positive" in evidence.candidate_promotion_vetoes
    assert evidence.engineering_invalidity_reasons == ()


@pytest.mark.parametrize(
    ("status", "count"),
    [
        ("not_exposed_by_kernel", 0),
        ("available", None),
        ("available", -1),
        ("unknown_status", None),
        ("available", 1.5),
    ],
)
def test_acceptance_evidence_types_malformed_divergence_provenance(
    status: str,
    count: object,
) -> None:
    evidence = _evidence(
        0.70,
        native_divergence_status=status,
        native_divergence_count=count,
    )

    assert evidence.evidence_validity == "candidate_data_invalid"
    assert evidence.acceptance_decision == "unavailable"
    assert evidence.engineering_invalidity_reasons == (
        "native_divergence_provenance_inconsistent",
    )
    assert evidence.native_divergence_status == "not_collected"
    assert evidence.native_divergence_count is None


def test_acceptance_evidence_redacts_unknown_legacy_reason_at_boundary() -> None:
    evidence = _evidence(
        0.70,
        candidate_local_health_failures=("exception secret=/private/path",),
    )

    assert evidence.evidence_validity == "candidate_data_invalid"
    assert evidence.engineering_invalidity_reasons == (
        "unrecognized_health_failure",
    )
    serialized = repr(evidence.payload())
    assert "secret" not in serialized
    assert "/private/path" not in serialized


def test_candidate_local_nonfinite_state_cannot_be_promoted_to_shared_by_reason() -> None:
    evidence = _evidence(
        0.70,
        candidate_local_health_failures=("nonfinite_candidate_state",),
    )

    assert evidence.evidence_validity == "candidate_data_invalid"
    assert evidence.engineering_invalidity_reasons == (
        "nonfinite_candidate_state",
    )


@pytest.mark.parametrize(
    ("field", "failure"),
    [
        ("log_accept", "nonfinite_log_accept_ratio"),
        ("target", "nonfinite_target_log_prob"),
    ],
)
def test_acceptance_evidence_nonfinite_candidate_data_is_invalid(
    field: str,
    failure: str,
) -> None:
    log_accept = np.full((64, 4), np.log(0.70))
    target = np.zeros((64, 4), dtype=float)
    if field == "log_accept":
        log_accept[0, 0] = np.nan
    elif field == "target":
        target[0, 0] = np.inf
    evidence = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=log_accept,
        is_accepted=np.ones((64, 4), dtype=bool),
        target_log_prob=target,
        policy=HMCAcceptancePolicy(),
    )

    assert evidence.evidence_validity == "candidate_data_invalid"
    assert evidence.acceptance_decision == "unavailable"
    assert failure in evidence.engineering_invalidity_reasons
    assert evidence.pooled_mean is None
    assert "NaN" not in json.dumps(evidence.payload(), allow_nan=False)


def test_nonfinite_retained_state_is_shared_execution_invalidity() -> None:
    samples = _moving_samples()
    samples[0, 0, 0] = np.nan
    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.full((64, 4), np.log(0.70)),
        is_accepted=np.ones((64, 4), dtype=bool),
        target_log_prob=np.zeros((64, 4), dtype=float),
        policy=HMCAcceptancePolicy(),
    )

    assert evidence.evidence_validity == "shared_execution_invalid"
    assert evidence.engineering_invalidity_reasons == (
        "nonfinite_retained_samples",
    )
    assert evidence.pooled_mean is None


def test_acceptance_evidence_requires_boolean_realized_trace() -> None:
    with pytest.raises(TypeError, match="is_accepted must be boolean"):
        evaluate_hmc_acceptance_evidence(
            samples=_moving_samples(),
            log_accept_ratio=np.full((64, 4), np.log(0.70)),
            is_accepted=np.ones((64, 4), dtype=float),
            policy=HMCAcceptancePolicy(),
        )


@pytest.mark.parametrize(
    "samples",
    (
        np.empty((0, 4, 2)),
        np.empty((64, 4, 0)),
    ),
)
def test_empty_acceptance_evidence_is_typed_shared_schema_invalidity(samples) -> None:
    draw_count = samples.shape[0]
    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.empty((draw_count, 4)),
        is_accepted=np.empty((draw_count, 4), dtype=bool),
        target_log_prob=np.empty((draw_count, 4)),
        policy=HMCAcceptancePolicy(),
    )

    assert evidence.evidence_validity == "shared_execution_invalid"
    assert evidence.acceptance_decision == "unavailable"
    assert evidence.engineering_invalidity_reasons == ("shared_schema_invalid",)
    assert evidence.pooled_mean is None


def test_short_positive_trace_remains_inconclusive_not_schema_invalid() -> None:
    samples = np.arange(16, dtype=float).reshape(4, 4, 1)
    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.full((4, 4), np.log(0.70)),
        is_accepted=np.ones((4, 4), dtype=bool),
        target_log_prob=np.zeros((4, 4)),
        policy=HMCAcceptancePolicy(),
    )

    assert evidence.evidence_validity == "valid"
    assert evidence.acceptance_decision == "inconclusive_evidence"
    assert evidence.excluded_remainder_per_chain == 4


def test_cost_stop_requires_minimum_acceptance_evidence() -> None:
    policy = HMCAcceptancePolicy(
        allowed_cost_stop_reasons=("persistent_candidate_cost_stop",)
    )

    with pytest.raises(ValueError, match="minimum acceptance evidence"):
        evaluate_hmc_acceptance_evidence(
            samples=np.arange(16, dtype=float).reshape(4, 4, 1),
            log_accept_ratio=np.full((4, 4), np.log(0.70)),
            is_accepted=np.ones((4, 4), dtype=bool),
            target_log_prob=np.zeros((4, 4)),
            policy=policy,
            cost_stop_reasons=("persistent_candidate_cost_stop",),
        )


def test_proxy_alert_preserves_matrix_wide_low_acceptance_direction() -> None:
    log_accept = np.full((64, 4), -1001.0)
    evidence = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=log_accept,
        is_accepted=np.zeros((64, 4), dtype=bool),
        target_log_prob=np.zeros((64, 4), dtype=float),
        policy=HMCAcceptancePolicy(),
    )

    assert evidence.evidence_validity == "valid"
    assert evidence.acceptance_decision == "repair_step_lower"
    assert evidence.repair_direction == "lower_epsilon"
    assert evidence.pooled_mean == pytest.approx(0.0)
    assert evidence.engineering_invalidity_reasons == ()
    assert evidence.candidate_promotion_vetoes == ()
    assert "log_accept_energy_proxy_exceeded" in evidence.candidate_health_alerts
    assert evidence.negative_proxy_exceedance_count_by_chain == (64, 64, 64, 64)
    assert evidence.positive_proxy_exceedance_count_by_chain == (0, 0, 0, 0)


def test_isolated_proxy_alert_has_no_standalone_control_effect() -> None:
    log_accept = np.full((64, 4), np.log(0.70))
    log_accept[0, 0] = -1001.0
    evidence = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=log_accept,
        is_accepted=np.ones((64, 4), dtype=bool),
        target_log_prob=np.zeros((64, 4), dtype=float),
        policy=HMCAcceptancePolicy(),
    )

    assert evidence.evidence_validity == "valid"
    # One rejected proposal does not move this declared block interval outside
    # the practical region. The proxy alert must not override that arithmetic.
    assert evidence.acceptance_decision == "passed"
    assert evidence.promotion_eligible is True
    assert evidence.repair_direction is None
    assert evidence.candidate_promotion_vetoes == ()
    assert evidence.tuning_repair_triggers == ()
    assert evidence.cost_stop_reasons == ()
    assert "log_accept_energy_proxy_exceeded" in evidence.candidate_health_alerts
    assert evidence.negative_proxy_exceedance_count_by_chain == (1, 0, 0, 0)


def test_extreme_positive_proxy_is_signed_and_does_not_request_lower_step() -> None:
    evidence = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=np.full((64, 4), 1001.0),
        is_accepted=np.ones((64, 4), dtype=bool),
        target_log_prob=np.zeros((64, 4), dtype=float),
        policy=HMCAcceptancePolicy(),
    )

    assert evidence.acceptance_decision == "repair_step_higher"
    assert evidence.repair_direction == "higher_epsilon"
    assert evidence.candidate_promotion_vetoes == ()
    assert evidence.negative_proxy_exceedance_count_by_chain == (0, 0, 0, 0)
    assert evidence.positive_proxy_exceedance_count_by_chain == (64, 64, 64, 64)


def test_mixed_signed_proxy_tails_are_separate_and_permutation_invariant() -> None:
    log_accept = np.full((64, 4), np.log(0.70))
    for block_start in (0, 16, 32, 48):
        log_accept[block_start, :] = -1001.0
        log_accept[block_start + 1, :] = 1002.0
    permuted = log_accept.copy()
    for block_start in (0, 16, 32, 48):
        permuted[block_start : block_start + 16] = permuted[
            block_start : block_start + 16
        ][::-1]

    first = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=log_accept,
        is_accepted=np.ones((64, 4), dtype=bool),
        target_log_prob=np.zeros((64, 4), dtype=float),
        policy=HMCAcceptancePolicy(),
    )
    second = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=permuted,
        is_accepted=np.ones((64, 4), dtype=bool),
        target_log_prob=np.zeros((64, 4), dtype=float),
        policy=HMCAcceptancePolicy(),
    )

    assert first.acceptance_decision == second.acceptance_decision
    assert first.candidate_health_alerts == second.candidate_health_alerts
    assert first.tuning_repair_triggers == second.tuning_repair_triggers
    assert first.negative_proxy_exceedance_count_by_chain == (4, 4, 4, 4)
    assert first.positive_proxy_exceedance_count_by_chain == (4, 4, 4, 4)
    assert (
        first.negative_proxy_exceedance_count_by_chain
        == second.negative_proxy_exceedance_count_by_chain
    )
    assert (
        first.positive_proxy_exceedance_count_by_chain
        == second.positive_proxy_exceedance_count_by_chain
    )


def test_predeclared_cost_stop_preserves_valid_evidence_but_blocks_promotion() -> None:
    policy = HMCAcceptancePolicy(
        allowed_cost_stop_reasons=("persistent_candidate_cost_stop",)
    )
    evidence = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=np.full((64, 4), np.log(0.70)),
        is_accepted=np.tile([True, False], (64, 2)),
        target_log_prob=np.zeros((64, 4), dtype=float),
        policy=policy,
        cost_stop_reasons=("persistent_candidate_cost_stop",),
    )

    assert evidence.evidence_validity == "valid"
    assert evidence.acceptance_decision == "passed"
    assert evidence.pooled_mean == pytest.approx(0.70)
    assert evidence.realized_acceptance_rate == pytest.approx(0.5)
    assert evidence.cost_stop_reasons == ("persistent_candidate_cost_stop",)
    assert evidence.cost_stop_scope == "exact_candidate_replication"
    assert evidence.promotion_eligible is False


def test_v2_proxy_veto_migration_is_non_actionable() -> None:
    legacy = {
        "schema": "bayesfilter.hmc_acceptance_evidence.v2",
        "decision": "candidate_local_veto",
        "passed": False,
        "repair_direction": None,
        "pooled_mean": None,
        "interval": None,
        "standard_error": None,
        "chain_means": [],
        "block_means_by_chain": [],
        "movement_rate_by_chain": [],
        "repeated_state_fraction_by_chain": [],
        "normalized_return_displacement_by_chain": [],
        "usable_decisions_per_chain": 0,
        "excluded_remainder_per_chain": 0,
        "native_divergence_status": "not_exposed_by_kernel",
        "native_divergence_count": None,
        "max_abs_log_accept_energy_proxy": 1001.0,
        "policy": {
            **HMCAcceptancePolicy().payload(),
            "schema": "bayesfilter.hmc_acceptance_policy.v2",
        },
        "hard_health_failures": ["log_accept_energy_proxy_exceeded"],
        "explanatory_notes": [],
        "raw_traces_exposed": False,
        "reports_posterior_convergence": False,
    }

    view = hmc_acceptance_evidence_v2_migration_view(legacy)
    assert view["historical_contract_validity"] == "valid_under_historical_contract"
    assert view["v3_reanalysis_status"] == "impossible_missing_raw_acceptance_trace"
    assert view["acceptance_decision_under_v3"] == "unavailable_legacy_v2_early_return"
    assert view["repair_direction_under_v3"] == "unavailable"
    with pytest.raises(ValueError, match="schema mismatch"):
        hmc_acceptance_evidence_from_payload(legacy)


def test_v3_migration_view_cannot_promote_historical_evidence() -> None:
    current = _evidence(0.70).payload()
    legacy = {
        **current,
        "schema": "bayesfilter.hmc_acceptance_evidence.v3",
        "interval": current["chain_mean_uncertainty_interval"],
        "standard_error": 0.01,
    }
    for name in (
        "chain_mean_uncertainty_interval",
        "chain_mean_uncertainty_method",
        "chain_mean_uncertainty_level",
    ):
        legacy.pop(name)
    legacy["policy"] = {
        **legacy["policy"],
        "schema": "bayesfilter.hmc_acceptance_policy.v3",
        "student_critical_value": 2.3533634348,
    }
    legacy["policy"].pop("uncertainty_method")
    legacy["policy"].pop("tuning_decision_role")

    view = hmc_acceptance_evidence_v3_migration_view(legacy)

    assert view["acceptance_decision"] == "recompute_required"
    assert view["promotion_eligible"] is False
    assert view["promotion_eligible_under_v4"] is False
    assert view["historical_v3_acceptance_decision"] == "passed"
    assert view["historical_v3_promotion_eligible"] is True
    with pytest.raises(ValueError, match="schema mismatch"):
        hmc_acceptance_evidence_from_payload(legacy)


def test_acceptance_evidence_payload_replay_is_deterministic_and_validated() -> None:
    first = _evidence(0.70)
    second = _evidence(0.70)

    assert first.payload() == second.payload()
    assert hmc_acceptance_evidence_from_payload(first.payload()) == first

    corrupted = dict(first.payload())
    corrupted["passed"] = False
    with pytest.raises(ValueError, match="passed flag"):
        hmc_acceptance_evidence_from_payload(corrupted)

    corrupted_path_return = dict(first.payload())
    corrupted_path_return["path_return_fraction_by_chain"] = (1.0,) * 4
    with pytest.raises(ValueError, match="decision is inconsistent"):
        hmc_acceptance_evidence_from_payload(corrupted_path_return)

    invalid = evaluate_hmc_acceptance_evidence(
        samples=_moving_samples(),
        log_accept_ratio=np.full((64, 4), np.nan),
        is_accepted=np.zeros((64, 4), dtype=bool),
        policy=HMCAcceptancePolicy(),
    )
    corrupted = dict(invalid.payload())
    corrupted["engineering_invalidity_reasons"] = ("secret=/private/path",)
    with pytest.raises(ValueError, match="unsupported reason code"):
        hmc_acceptance_evidence_from_payload(corrupted)

    corrupted = dict(first.payload())
    corrupted["realized_acceptance_rate_by_chain"] = (0.99,) * 4
    corrupted["realized_acceptance_rate"] = 0.99
    with pytest.raises(ValueError, match="realized acceptance summary"):
        hmc_acceptance_evidence_from_payload(corrupted)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("usable_decisions_per_chain", 64.5),
        ("usable_decisions_per_chain", True),
        ("excluded_remainder_per_chain", np.array(0)),
        ("negative_proxy_exceedance_count_by_chain", (0.5, 0, 0, 0)),
        ("positive_proxy_exceedance_count_by_chain", (False, 0, 0, 0)),
    ],
)
def test_acceptance_evidence_payload_rejects_noninteger_count_scalars(
    field: str,
    value: object,
) -> None:
    payload = dict(_evidence(0.70).payload())
    payload[field] = value

    with pytest.raises(ValueError, match="integer scalar"):
        hmc_acceptance_evidence_from_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chain_count", 4.5),
        ("block_count", True),
        ("min_block_size", np.array(16)),
    ],
)
def test_acceptance_policy_rejects_noninteger_count_scalars(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="integer scalar"):
        HMCAcceptancePolicy(**{field: value})


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("pooled_mean", "pooled_mean does not match"),
        ("uncertainty_interval", "uncertainty interval is inconsistent"),
        ("uncertainty_method", "unsupported chain-mean uncertainty method"),
        ("decision", "decision is inconsistent"),
        ("policy", "decision is inconsistent"),
    ],
)
def test_acceptance_evidence_payload_rejects_arithmetic_and_policy_corruption(
    mutation: str,
    error: str,
) -> None:
    payload = dict(_evidence(0.70).payload())
    if mutation == "pooled_mean":
        payload["pooled_mean"] = 0.71
    elif mutation == "uncertainty_interval":
        payload["chain_mean_uncertainty_interval"] = (0.69, 0.71)
    elif mutation == "uncertainty_method":
        payload["chain_mean_uncertainty_method"] = "forged_method"
    elif mutation == "decision":
        payload["decision"] = "repair_step_higher"
        payload["passed"] = False
        payload["repair_direction"] = "higher_epsilon"
    else:
        payload["policy"] = {
            **payload["policy"],
            "target": 0.60,
            "practical_region": (0.55, 0.65),
            "repair_region": (0.50, 0.85),
        }

    with pytest.raises(ValueError, match=error):
        hmc_acceptance_evidence_from_payload(payload)
