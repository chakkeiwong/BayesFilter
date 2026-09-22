from __future__ import annotations

import itertools
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning
from bayesfilter.hmc_budget_contract import (
    HMCOperationalStatisticalWorkPolicy,
    build_public_hmc_work_manifest,
)
from bayesfilter.hmc_route_contract import OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID
from bayesfilter.inference.hmc_verification import (
    HMCAcceptancePolicy,
    _evaluate_retained_target_health,
    evaluate_hmc_acceptance_evidence,
)
from bayesfilter.inference.hmc_warmup import (
    assess_metric_covariance,
    build_private_start_bank,
)
from tests.test_hmc_kernel_tuning_windowed_mass import (
    _operational_budget,
    _operational_inputs,
    _stage_config,
)


def _moving_high_dimensional_samples(
    *,
    draw_count: int = 64,
    chain_count: int = 4,
    dimension: int = 314,
) -> np.ndarray:
    draw = np.arange(draw_count, dtype=float)[:, None, None]
    chain = np.arange(chain_count, dtype=float)[None, :, None]
    direction = np.linspace(0.5, 1.5, dimension)[None, None, :]
    return draw * direction + chain


def test_acceptance_admission_rejects_one_frozen_coordinate_in_314_dimensions() -> None:
    samples = _moving_high_dimensional_samples()
    samples[:, :, 0] = 0.0
    probability = np.full((64, 4), 0.70)

    evidence = evaluate_hmc_acceptance_evidence(
        samples=samples,
        log_accept_ratio=np.log(probability),
        is_accepted=np.ones_like(probability, dtype=bool),
        target_log_prob=np.zeros_like(probability),
        policy=HMCAcceptancePolicy(),
    )

    assert np.std(samples[:, :, 0]) == 0.0
    assert evidence.movement_rate_by_chain == pytest.approx((0.0,) * 4)
    assert evidence.promotion_eligible is False


def test_acceptance_interval_records_working_coverage_under_persistent_chain_effects() -> None:
    policy = HMCAcceptancePolicy()
    samples = _moving_high_dimensional_samples(dimension=2)
    covered = []

    # Exact finite enumeration: each chain is persistently at 0.60 or 0.80,
    # independently with equal probability. The population mean is 0.70. This
    # deliberately non-Gaussian four-unit fixture is explanatory only: the
    # working Student-t interval is not claimed to have exact nominal coverage.
    for chain_probabilities in itertools.product((0.60, 0.80), repeat=4):
        probability = np.tile(np.asarray(chain_probabilities, dtype=float), (64, 1))
        evidence = evaluate_hmc_acceptance_evidence(
            samples=samples,
            log_accept_ratio=np.log(probability),
            is_accepted=np.ones_like(probability, dtype=bool),
            target_log_prob=np.zeros_like(probability),
            policy=policy,
        )
        assert evidence.interval is not None
        covered.append(evidence.interval[0] <= policy.target <= evidence.interval[1])

    exact_coverage = float(np.mean(covered))
    assert exact_coverage == pytest.approx(0.875)
    assert exact_coverage < policy.confidence_level


def test_dense_metric_gate_rejects_highly_autocorrelated_full_rank_states() -> None:
    state_count = 400
    dimension = 4
    time = np.arange(state_count, dtype=float)
    states = np.stack(
        tuple(
            np.sqrt(2.0)
            * np.sin(2.0 * np.pi * frequency * time / state_count)
            for frequency in range(1, dimension + 1)
        ),
        axis=1,
    )

    lag_one_correlations = np.asarray(
        [
            np.corrcoef(states[:-1, index], states[1:, index])[0, 1]
            for index in range(dimension)
        ]
    )

    decision = assess_metric_covariance(states)

    assert np.min(lag_one_correlations) > 0.99
    assert decision.report["raw_numerical_rank"] == dimension
    assert decision.outcome != "dense_update"


def test_operational_warmup_survives_legacy_compatibility_projection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, geometry, bootstrap = _operational_inputs()
    events: list[str] = []

    def fail_compatibility_projection(*_args, **_kwargs):
        raise RuntimeError("compatibility projection failure sentinel")

    monkeypatch.setattr(
        hmc_kernel_tuning,
        "run_windowed_mass_adaptation_diagnostic",
        fail_compatibility_projection,
    )
    result = hmc_kernel_tuning.run_hmc_windowed_mass_stage(
        adapter=adapter,
        geometry=geometry,
        bootstrap=bootstrap,
        config=_stage_config(
            algorithm_id=OPERATIONAL_WINDOWED_WARMUP_ALGORITHM_ID,
            chain_execution_mode="tf_function",
        ),
        _attempt_budget_policy=_operational_budget(),
        _progress_callback=lambda stage, _payload: events.append(stage),
    )

    assert "windowed_mass_operational_warmup_complete" in events
    assert result.operational_warmup_result is not None
    assert result.operational_mass_artifact is not None
    assert result.windowed_mass_result is None
    assert result.passed is True
    assert "windowed_stage_hmc_error" not in result.hard_vetoes
    status = result.diagnostics["runtime_metadata"][
        "legacy_v1_compatibility_projection"
    ]
    assert status["status"] == "unavailable_error"
    assert status["authoritative"] is False
    assert status["error_type"] == "RuntimeError"


def test_retained_target_health_batches_value_and_score_rechecks() -> None:
    import tensorflow as tf

    class CountingAdapter:
        supports_retained_draw_batch = True

        def __init__(self) -> None:
            self.call_count = 0

        def log_prob_and_grad(self, theta):
            self.call_count += 1
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value

    adapter = CountingAdapter()
    draw_count = 32
    health = _evaluate_retained_target_health(
        adapter=adapter,
        samples=np.zeros((draw_count, 4, 2), dtype=float),
    )

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == ()
    assert health["evaluated_draw_count"] == draw_count
    assert adapter.call_count < draw_count


def test_retained_target_health_counts_first_invalid_draw_inside_batch() -> None:
    import tensorflow as tf

    class MidChunkInvalidAdapter:
        supports_retained_draw_batch = True

        def __init__(self) -> None:
            self.call_count = 0

        def log_prob_and_grad(self, theta):
            self.call_count += 1
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            log_prob = -0.5 * tf.reduce_sum(tf.square(value), axis=-1)
            invalid = tf.equal(value[..., 0], tf.constant(5.0, tf.float64))
            log_prob = tf.where(
                invalid,
                tf.constant(float("nan"), tf.float64),
                log_prob,
            )
            return log_prob, -value

    adapter = MidChunkInvalidAdapter()
    samples = np.zeros((20, 4, 2), dtype=float)
    samples[:, :, 0] = np.arange(20, dtype=float)[:, None]

    health = _evaluate_retained_target_health(adapter=adapter, samples=samples)

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == (
        "nonfinite_target_log_prob",
    )
    assert health["evaluated_draw_count"] == 6
    assert adapter.call_count == 7


def test_retained_target_health_shape_failure_before_valid_draw_counts_zero() -> None:
    import tensorflow as tf

    class MalformedBatchAdapter:
        supports_retained_draw_batch = True

        def log_prob_and_grad(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            return tf.constant(0.0, tf.float64), -value

    health = _evaluate_retained_target_health(
        adapter=MalformedBatchAdapter(),
        samples=np.zeros((20, 4, 2), dtype=float),
    )

    assert health["shared_invalidity_reasons"] == (
        "target_value_score_shape_invalid",
    )
    assert health["candidate_data_invalidity_reasons"] == ()
    assert health["evaluated_draw_count"] == 0


def test_retained_target_health_refinement_exception_counts_prior_valid_draws() -> None:
    import tensorflow as tf

    class RefinementExceptionAdapter:
        supports_retained_draw_batch = True

        def __init__(self) -> None:
            self.call_count = 0

        def log_prob_and_grad(self, theta):
            self.call_count += 1
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            draw_ids = np.asarray(value.numpy())[..., 0]
            if value.shape.rank == 2 and np.any(draw_ids == 2.0):
                raise RuntimeError("refined replay failure sentinel")
            log_prob = -0.5 * tf.reduce_sum(tf.square(value), axis=-1)
            if value.shape.rank == 3:
                invalid = tf.equal(value[..., 0], tf.constant(2.0, tf.float64))
                log_prob = tf.where(
                    invalid,
                    tf.constant(float("nan"), tf.float64),
                    log_prob,
                )
            return log_prob, -value

    adapter = RefinementExceptionAdapter()
    samples = np.zeros((20, 4, 2), dtype=float)
    samples[:, :, 0] = np.arange(20, dtype=float)[:, None]

    health = _evaluate_retained_target_health(adapter=adapter, samples=samples)

    assert health["shared_invalidity_reasons"] == ("shared_callback_invalid",)
    assert health["candidate_data_invalidity_reasons"] == ()
    assert health["evaluated_draw_count"] == 2
    assert adapter.call_count == 4


def test_retained_target_health_rejects_unreproducible_batch_nonfinite() -> None:
    import tensorflow as tf

    class BatchOnlyInvalidAdapter:
        supports_retained_draw_batch = True

        def log_prob_and_grad(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            log_prob = -0.5 * tf.reduce_sum(tf.square(value), axis=-1)
            if value.shape.rank == 3:
                log_prob = tf.fill(tf.shape(log_prob), tf.constant(float("nan")))
            return log_prob, -value

    health = _evaluate_retained_target_health(
        adapter=BatchOnlyInvalidAdapter(),
        samples=np.zeros((20, 4, 2), dtype=float),
    )

    assert health["shared_invalidity_reasons"] == ("shared_schema_invalid",)
    assert health["candidate_data_invalidity_reasons"] == ()
    assert health["evaluated_draw_count"] == 0


def test_retained_target_health_flat_batch_preserves_logical_draw_accounting() -> None:
    import tensorflow as tf

    class FlatBatchAdapter:
        supports_retained_flat_batch = True

        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def log_prob_and_grad(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            self.shapes.append(tuple(int(item) for item in value.shape))
            return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value

    adapter = FlatBatchAdapter()
    health = _evaluate_retained_target_health(
        adapter=adapter,
        samples=np.zeros((20, 4, 2), dtype=float),
    )

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == ()
    assert health["evaluated_draw_count"] == 20
    assert adapter.shapes == [(64, 2), (16, 2)]


def test_retained_target_health_counts_all_batched_telemetry_failures() -> None:
    import tensorflow as tf

    class BatchedTelemetryAdapter:
        supports_retained_draw_batch = True

        def log_prob_and_grad(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value

        def target_status_telemetry(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            leading_shape = tf.shape(value)[:-1]
            invalid = tf.logical_and(
                tf.equal(value[..., 0], tf.constant(5.0, tf.float64)),
                tf.equal(value[..., 1], tf.constant(2.0, tf.float64)),
            )
            return {
                "status_code": tf.cast(invalid, tf.int32),
                "valid_pre_regularized_score": tf.logical_not(invalid),
                "floor_count_value": tf.zeros(leading_shape, tf.int32),
                "min_innovation_eigenvalue": tf.ones(leading_shape, tf.float64),
                "innovation_condition_estimate": tf.ones(
                    leading_shape, tf.float64
                ),
            }

    samples = np.zeros((20, 4, 2), dtype=float)
    samples[:, :, 0] = np.arange(20, dtype=float)[:, None]
    samples[:, :, 1] = np.arange(4, dtype=float)[None, :]

    health = _evaluate_retained_target_health(
        adapter=BatchedTelemetryAdapter(),
        samples=samples,
        target_status_trace_policy="per_chain_step",
    )

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == (
        "target_status_telemetry_failure",
    )
    assert health["target_status_failure_count"] == 1
    assert health["evaluated_draw_count"] == 20


def test_retained_target_health_counts_all_flat_batch_telemetry_failures() -> None:
    import tensorflow as tf

    class FlatBatchedTelemetryAdapter:
        supports_retained_flat_batch = True

        def __init__(self) -> None:
            self.telemetry_shapes: list[tuple[int, ...]] = []

        def log_prob_and_grad(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            return -0.5 * tf.reduce_sum(tf.square(value), axis=-1), -value

        def target_status_telemetry(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            self.telemetry_shapes.append(tuple(int(item) for item in value.shape))
            leading_shape = tf.shape(value)[:-1]
            invalid = tf.logical_and(
                tf.equal(value[..., 0], tf.constant(5.0, tf.float64)),
                tf.equal(value[..., 1], tf.constant(2.0, tf.float64)),
            )
            return {
                "status_code": tf.cast(invalid, tf.int32),
                "valid_pre_regularized_score": tf.logical_not(invalid),
                "floor_count_value": tf.zeros(leading_shape, tf.int32),
                "min_innovation_eigenvalue": tf.ones(leading_shape, tf.float64),
                "innovation_condition_estimate": tf.ones(
                    leading_shape, tf.float64
                ),
            }

    samples = np.zeros((20, 4, 2), dtype=float)
    samples[:, :, 0] = np.arange(20, dtype=float)[:, None]
    samples[:, :, 1] = np.arange(4, dtype=float)[None, :]
    adapter = FlatBatchedTelemetryAdapter()

    health = _evaluate_retained_target_health(
        adapter=adapter,
        samples=samples,
        target_status_trace_policy="per_chain_step",
    )

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == (
        "target_status_telemetry_failure",
    )
    assert health["target_status_failure_count"] == 1
    assert health["evaluated_draw_count"] == 20
    assert adapter.telemetry_shapes == (
        [(64, 2)] + [(4, 2)] * 16 + [(16, 2)]
    )


def test_retained_target_health_uses_combined_value_score_status_once_per_batch() -> None:
    import tensorflow as tf

    class CombinedAdapter:
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True

        def __init__(self) -> None:
            self.combined_calls: list[tuple[int, ...]] = []
            self.legacy_calls = 0
            self.telemetry_calls = 0

        def log_prob_and_grad_status(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            self.combined_calls.append(tuple(int(item) for item in value.shape))
            leading_shape = tf.shape(value)[:-1]
            return (
                -0.5 * tf.reduce_sum(tf.square(value), axis=-1),
                -value,
                {
                    "status_code": tf.zeros(leading_shape, tf.int32),
                    "valid_pre_regularized_score": tf.ones(leading_shape, tf.bool),
                    "floor_count_value": tf.zeros(leading_shape, tf.int32),
                    "min_innovation_eigenvalue": tf.ones(leading_shape, tf.float64),
                    "innovation_condition_estimate": tf.ones(
                        leading_shape, tf.float64
                    ),
                },
            )

        def log_prob_and_grad(self, theta):
            self.legacy_calls += 1
            value, score, _status = self.log_prob_and_grad_status(theta)
            return value, score

        def target_status_telemetry(self, theta):
            self.telemetry_calls += 1
            raise AssertionError("combined retained protocol should avoid telemetry replay")

    adapter = CombinedAdapter()
    health = _evaluate_retained_target_health(
        adapter=adapter,
        samples=np.zeros((20, 4, 2), dtype=float),
        target_status_trace_policy="per_chain_step",
    )

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == ()
    assert health["target_status_failure_count"] == 0
    assert health["evaluated_draw_count"] == 20
    assert adapter.combined_calls == [(64, 2), (16, 2)]
    assert adapter.legacy_calls == 0
    assert adapter.telemetry_calls == 0


def test_combined_retained_status_failure_localization_does_not_call_legacy_telemetry() -> None:
    import tensorflow as tf

    class FailingCombinedAdapter:
        supports_retained_flat_batch = True
        supports_retained_value_score_status = True

        def __init__(self) -> None:
            self.combined_calls = 0
            self.telemetry_calls = 0

        def log_prob_and_grad_status(self, theta):
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            self.combined_calls += 1
            leading = tf.shape(value)[:-1]
            failed = tf.equal(value[..., 0], tf.constant(5.0, tf.float64))
            return (
                -0.5 * tf.reduce_sum(tf.square(value), axis=-1),
                -value,
                {
                    "status_code": tf.cast(failed, tf.int32),
                    "valid_pre_regularized_score": tf.logical_not(failed),
                    "floor_count_value": tf.zeros(leading, tf.int32),
                    "min_innovation_eigenvalue": tf.ones(leading, tf.float64),
                    "innovation_condition_estimate": tf.ones(leading, tf.float64),
                },
            )

        def log_prob_and_grad(self, theta):
            value, score, _status = self.log_prob_and_grad_status(theta)
            return value, score

        def target_status_telemetry(self, _theta):
            self.telemetry_calls += 1
            raise AssertionError("legacy telemetry must not be called")

    samples = np.zeros((20, 4, 2), dtype=float)
    samples[:, :, 0] = np.arange(20, dtype=float)[:, None]
    adapter = FailingCombinedAdapter()
    health = _evaluate_retained_target_health(
        adapter=adapter,
        samples=samples,
        target_status_trace_policy="per_chain_step",
    )

    assert health["candidate_data_invalidity_reasons"] == (
        "target_status_telemetry_failure",
    )
    assert health["target_status_failure_count"] == 4
    assert adapter.telemetry_calls == 0
    # One batched call, all sixteen logical-draw checks, and one final batch;
    # no legacy telemetry replay is permitted.
    assert adapter.combined_calls == 18


def test_private_start_bank_rejects_negligibly_dispersed_distinct_states() -> None:
    history = np.stack(
        (
            np.linspace(0.0, 7.0e-8, 8),
            np.linspace(0.0, -3.5e-8, 8),
        ),
        axis=1,
    )

    rejected = False
    try:
        build_private_start_bank(history)
    except ValueError:
        rejected = True

    assert rejected is True


def test_ccma_serious_budget_accounts_for_the_broad_per_l_search() -> None:
    attempt = hmc_kernel_tuning._default_attempt_budget_policy(314, 0)
    operational = HMCOperationalStatisticalWorkPolicy()
    manifest = build_public_hmc_work_manifest(
        target_dimension=314,
        metric_adaptation_steps=(attempt.phase4_warmup_steps,),
        selection_attempts_per_outer_attempt=(1,),
        max_leapfrog_steps=25,
        policy=operational,
    )
    work = manifest["maximum_work"]

    assert attempt.budget0_uncapped == 20 * 314
    assert attempt.phase4_warmup_steps == 5000
    assert attempt.budget0_after_floor_and_cap == 5000
    ladder = manifest["broad_grid_ladder_work_per_candidate"]
    expected = (
        attempt.phase4_warmup_steps
        + 13 * ladder["total_transitions_per_ladder"]
        + 2
        * (
            operational.fresh_verification_results
            + operational.fresh_verification_burnin_steps
        )
    )
    assert manifest["candidate_count_upper_bound"] == 13
    assert manifest["broad_primary_grid_width"] == 6
    assert manifest["broad_refinement_grid_width_upper_bound"] == 7
    assert work["total_batched_transitions"] == expected
    assert work["metric_adaptation_batched_transitions"] / expected < 0.05
    assert "not posterior convergence" in attempt.budget_claim


def test_operational_trajectory_anchor_tracks_final_adapted_geometry() -> None:
    inherited_leapfrog_count = 4
    final_step_size = 0.05
    target_trajectory_length = np.pi / 2.0

    anchor = hmc_kernel_tuning._joint_l_epsilon_anchor_l(
        selected_kernel={"num_leapfrog_steps": inherited_leapfrog_count},
        attempt_state=None,
        final_step_size=final_step_size,
        target_trajectory_length=target_trajectory_length,
        max_leapfrog_steps=64,
    )
    final_geometry_count = int(np.ceil(target_trajectory_length / final_step_size))

    assert anchor == final_geometry_count


def test_operational_trajectory_anchor_rejects_invalid_final_step() -> None:
    with pytest.raises(ValueError, match="final adapted step size"):
        hmc_kernel_tuning._joint_l_epsilon_anchor_l(
            selected_kernel={"num_leapfrog_steps": 4},
            attempt_state=None,
            final_step_size=0.0,
            target_trajectory_length=np.pi / 2.0,
        )


def test_operational_trajectory_anchor_preserves_explicit_retry_precedence() -> None:
    attempt_state = hmc_kernel_tuning._HMCPhaseAttemptState(
        selected_num_leapfrog_steps=7,
        phase6_retry_num_leapfrog_steps=11,
        phase6_retry_anchor_source="reviewed_repair",
    )

    anchor = hmc_kernel_tuning._joint_l_epsilon_anchor_l(
        selected_kernel={"num_leapfrog_steps": 4},
        attempt_state=attempt_state,
        final_step_size=0.05,
        target_trajectory_length=np.pi / 2.0,
        max_leapfrog_steps=64,
    )

    assert anchor == 11
