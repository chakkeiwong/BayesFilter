from __future__ import annotations

import json
import os
import types
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter.inference.hmc_warmup as hmc_warmup
from bayesfilter.inference.hmc_coordinates import (
    AffineCoordinateTransform,
    PositionCovarianceEstimate,
    WarmupTrajectoryPolicy,
)
from bayesfilter.inference.hmc_tuning import WindowedMassAdaptationConfig
from bayesfilter.inference.hmc_warmup import (
    _AffineWarmupAdapter,
    OperationalWindowedWarmupCloseout,
    ReasonableEpsilonAttempt,
    ReasonableEpsilonResult,
    assess_metric_covariance,
    build_private_start_bank,
    compose_operational_transform_in_base_coordinates,
    find_reasonable_epsilon,
    normalize_operational_warmup_config,
    run_operational_windowed_warmup,
)
from bayesfilter.inference.posterior_adapter import ValueScoreCapability


class _GaussianAdapter:
    def __init__(self, covariance: np.ndarray) -> None:
        self.covariance = np.asarray(covariance, dtype=float)
        self.precision = np.linalg.inv(self.covariance)
        self.parameter_dim = self.covariance.shape[0]

    def adapter_signature(self) -> str:
        return "hmc-warmup-gaussian-" + str(self.parameter_dim)

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=False,
            runtime_backend="tensorflow",
            evidence_path="tests/test_hmc_warmup.py",
            target_scope="hmc_warmup_gaussian",
            nonclaims=("analytical warmup target only",),
        )

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        precision = tf.convert_to_tensor(self.precision, dtype=value.dtype)
        score = -tf.linalg.matvec(precision, value)
        log_prob = -0.5 * tf.reduce_sum(value * -score, axis=-1)
        return log_prob, score


class _DeclaredDomainGaussianAdapter(_GaussianAdapter):
    def classify_target_exception(self, error: BaseException) -> bool:
        return isinstance(error, tf.errors.InvalidArgumentError)


class _TargetStatusGaussianAdapter(_GaussianAdapter):
    def __init__(self, covariance: np.ndarray, *, nonvalid: bool = False) -> None:
        super().__init__(covariance)
        self.nonvalid = nonvalid

    def target_status_telemetry(self, theta: tf.Tensor) -> dict[str, tf.Tensor]:
        leading_shape = tf.shape(theta)[:-1]
        status = tf.ones(leading_shape, tf.int32) if self.nonvalid else tf.zeros(
            leading_shape, tf.int32
        )
        valid = tf.zeros(leading_shape, tf.bool) if self.nonvalid else tf.ones(
            leading_shape, tf.bool
        )
        return {
            "status_code": status,
            "valid_pre_regularized_score": valid,
            "floor_count_value": tf.zeros(leading_shape, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading_shape, tf.float64),
            "innovation_condition_estimate": tf.ones(leading_shape, tf.float64),
        }


class _BatchCapabilityGaussianAdapter(_TargetStatusGaussianAdapter):
    def __init__(
        self,
        covariance: np.ndarray,
        *,
        draw_batch: bool = False,
        flat_batch: bool = False,
    ) -> None:
        super().__init__(covariance)
        self.supports_retained_draw_batch = bool(draw_batch)
        self.supports_retained_flat_batch = bool(flat_batch)
        self.value_score_shapes: list[tuple[int, ...]] = []
        self.telemetry_shapes: list[tuple[int, ...]] = []

    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        self.value_score_shapes.append(tuple(int(item) for item in value.shape))
        return super().log_prob_and_grad(value)

    def target_status_telemetry(self, theta: tf.Tensor) -> dict[str, tf.Tensor]:
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        self.telemetry_shapes.append(tuple(int(item) for item in value.shape))
        return super().target_status_telemetry(value)


class _TargetStatusOutsideRadiusAdapter(_GaussianAdapter):
    def target_status_telemetry(self, theta: tf.Tensor) -> dict[str, tf.Tensor]:
        nonvalid = tf.reduce_any(tf.abs(theta) > tf.constant(0.5, tf.float64), axis=-1)
        leading_shape = tf.shape(theta)[:-1]
        return {
            "status_code": tf.cast(nonvalid, tf.int32),
            "valid_pre_regularized_score": tf.logical_not(nonvalid),
            "floor_count_value": tf.zeros(leading_shape, tf.int32),
            "min_innovation_eigenvalue": tf.ones(leading_shape, tf.float64),
            "innovation_condition_estimate": tf.ones(leading_shape, tf.float64),
        }


class _NonfiniteInitialScoreAdapter(_GaussianAdapter):
    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score = super().log_prob_and_grad(theta)
        return value, tf.fill(tf.shape(score), tf.constant(float("nan"), tf.float64))


class _NonfiniteProposalOutsideRadiusAdapter(_GaussianAdapter):
    def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score = super().log_prob_and_grad(theta)
        outside = tf.reduce_any(tf.abs(theta) > tf.constant(0.5, tf.float64), axis=-1)
        score = tf.where(
            outside[..., tf.newaxis],
            tf.fill(tf.shape(score), tf.constant(float("nan"), tf.float64)),
            score,
        )
        return value, score


def _transform(covariance: np.ndarray, center: np.ndarray | None = None):
    dimension = covariance.shape[0]
    estimate = PositionCovarianceEstimate(
        center=np.zeros(dimension) if center is None else center,
        covariance=covariance,
        source_coordinate_signature="analytical-source",
        estimator_family="analytical",
        state_count=256,
        effective_rank=dimension,
        regularization_report={"method": "none"},
        adequacy_report={"passed": True},
    )
    return AffineCoordinateTransform.from_covariance_estimate(estimate)


@pytest.mark.parametrize(
    ("draw_batch", "flat_batch", "expected_shapes"),
    (
        (True, False, [(16, 4, 2), (4, 4, 2)]),
        (False, True, [(64, 2), (16, 2)]),
    ),
)
def test_affine_warmup_adapter_preserves_retained_batch_contract_and_semantics(
    draw_batch: bool,
    flat_batch: bool,
    expected_shapes: list[tuple[int, ...]],
) -> None:
    transform = _transform(
        np.array([[4.0, 1.0], [1.0, 2.0]]),
        center=np.array([0.3, -0.2]),
    )
    base = _BatchCapabilityGaussianAdapter(
        np.array([[1.5, 0.2], [0.2, 0.8]]),
        draw_batch=draw_batch,
        flat_batch=flat_batch,
    )
    wrapped = _AffineWarmupAdapter(
        base_adapter=base,
        transform=transform,
        target_scope="affine_batch_contract",
    )
    latent = np.arange(160, dtype=float).reshape(20, 4, 2) / 100.0
    flat = latent.reshape((-1, 2))

    wrapped_value, wrapped_score = wrapped.log_prob_and_grad(flat)
    theta = transform.latent_to_theta(flat)
    base_value, base_score = _GaussianAdapter.log_prob_and_grad(base, theta)
    np.testing.assert_allclose(wrapped_value.numpy(), base_value.numpy())
    np.testing.assert_allclose(
        wrapped_score.numpy(),
        transform.theta_score_to_latent_score(base_score).numpy(),
    )
    wrapped_status = wrapped.target_status_telemetry(flat)
    base_status = _TargetStatusGaussianAdapter.target_status_telemetry(base, theta)
    for key in wrapped_status:
        np.testing.assert_array_equal(wrapped_status[key].numpy(), base_status[key].numpy())

    base.value_score_shapes.clear()
    base.telemetry_shapes.clear()
    health = hmc_warmup._evaluate_retained_target_health(
        adapter=wrapped,
        samples=latent,
        target_status_trace_policy="per_chain_step",
    )

    assert wrapped.supports_retained_draw_batch is draw_batch
    assert wrapped.supports_retained_flat_batch is flat_batch
    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == ()
    assert health["target_status_failure_count"] == 0
    assert health["evaluated_draw_count"] == 20
    assert base.value_score_shapes == expected_shapes
    assert base.telemetry_shapes == expected_shapes


def test_affine_warmup_adapter_rejects_dual_batch_contract_and_preserves_none() -> None:
    transform = _transform(np.eye(2))
    plain = _AffineWarmupAdapter(
        base_adapter=_GaussianAdapter(np.eye(2)),
        transform=transform,
        target_scope="affine_no_batch_contract",
    )
    assert plain.supports_retained_draw_batch is False
    assert plain.supports_retained_flat_batch is False

    with pytest.raises(ValueError, match="two retained batching contracts"):
        _AffineWarmupAdapter(
            base_adapter=_BatchCapabilityGaussianAdapter(
                np.eye(2), draw_batch=True, flat_batch=True
            ),
            transform=transform,
            target_scope="affine_invalid_batch_contract",
        )


@pytest.mark.parametrize(("draw_batch", "flat_batch"), ((True, False), (False, True)))
def test_affine_warmup_batch_refinement_preserves_invalid_draw_accounting(
    draw_batch: bool,
    flat_batch: bool,
) -> None:
    class InvalidAdapter(_BatchCapabilityGaussianAdapter):
        def log_prob_and_grad(self, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            self.value_score_shapes.append(tuple(int(item) for item in value.shape))
            log_prob, score = _GaussianAdapter.log_prob_and_grad(self, value)
            invalid = tf.equal(value[..., 0], tf.constant(10.0, tf.float64))
            return tf.where(invalid, tf.constant(float("nan"), tf.float64), log_prob), score

    transform = _transform(np.array([[4.0, 1.0], [1.0, 2.0]]))
    base = InvalidAdapter(
        np.eye(2), draw_batch=draw_batch, flat_batch=flat_batch
    )
    wrapped = _AffineWarmupAdapter(
        base_adapter=base,
        transform=transform,
        target_scope="affine_invalid_draw_accounting",
    )
    samples = np.zeros((20, 4, 2), dtype=float)
    samples[:, :, 0] = np.arange(20, dtype=float)[:, None]
    samples[:, :, 1] = np.arange(4, dtype=float)[None, :]

    health = hmc_warmup._evaluate_retained_target_health(
        adapter=wrapped,
        samples=samples,
    )

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == (
        "nonfinite_target_log_prob",
    )
    assert health["evaluated_draw_count"] == 6
    assert len(base.value_score_shapes) == 7


@pytest.mark.parametrize(("draw_batch", "flat_batch"), ((True, False), (False, True)))
def test_affine_warmup_batch_telemetry_preserves_invalid_chain_count(
    draw_batch: bool,
    flat_batch: bool,
) -> None:
    class InvalidTelemetryAdapter(_BatchCapabilityGaussianAdapter):
        def target_status_telemetry(self, theta: tf.Tensor) -> dict[str, tf.Tensor]:
            value = tf.convert_to_tensor(theta, dtype=tf.float64)
            self.telemetry_shapes.append(tuple(int(item) for item in value.shape))
            leading_shape = tf.shape(value)[:-1]
            invalid_draw = tf.equal(value[..., 0], tf.constant(10.0, tf.float64))
            invalid_chain = value[..., 1] < tf.constant(2.0, tf.float64)
            invalid = tf.logical_and(invalid_draw, invalid_chain)
            return {
                "status_code": tf.cast(invalid, tf.int32),
                "valid_pre_regularized_score": tf.logical_not(invalid),
                "floor_count_value": tf.zeros(leading_shape, tf.int32),
                "min_innovation_eigenvalue": tf.ones(leading_shape, tf.float64),
                "innovation_condition_estimate": tf.ones(leading_shape, tf.float64),
            }

    transform = _transform(np.array([[4.0, 0.0], [0.0, 1.0]]))
    base = InvalidTelemetryAdapter(
        np.eye(2), draw_batch=draw_batch, flat_batch=flat_batch
    )
    wrapped = _AffineWarmupAdapter(
        base_adapter=base,
        transform=transform,
        target_scope="affine_invalid_telemetry_accounting",
    )
    samples = np.zeros((20, 4, 2), dtype=float)
    samples[:, :, 0] = np.arange(20, dtype=float)[:, None]
    samples[:, :, 1] = np.arange(4, dtype=float)[None, :]

    health = hmc_warmup._evaluate_retained_target_health(
        adapter=wrapped,
        samples=samples,
        target_status_trace_policy="per_chain_step",
    )

    assert health["shared_invalidity_reasons"] == ()
    assert health["candidate_data_invalidity_reasons"] == (
        "target_status_telemetry_failure",
    )
    assert health["target_status_failure_count"] == 2
    assert health["evaluated_draw_count"] == 6


def test_dense_metric_gate_requires_sample_adequacy() -> None:
    rng = np.random.default_rng(11)
    too_small = rng.normal(size=(20, 4))
    enough = rng.normal(size=(80, 4))

    rejected = assess_metric_covariance(too_small)
    accepted = assess_metric_covariance(enough)

    assert rejected.outcome == "no_update_insufficient_metric_evidence"
    assert rejected.report["shrinkage_spd_not_treated_as_adequacy"] is True
    assert accepted.outcome == "dense_update"
    assert accepted.report["dense_information_gate_passed"] is True
    assert accepted.report["minimum_effective_sample_size"] >= 8
    assert accepted.report["cross_chain_compatibility_method"] == (
        "not_applicable_single_chain"
    )


def test_dense_metric_gate_rejects_shifted_explicit_chains() -> None:
    rng = np.random.default_rng(20260716)
    states = rng.normal(size=(160, 4, 3))
    states[:, :, 0] += np.array([-4.0, -1.0, 1.0, 4.0])[None, :]

    decision = assess_metric_covariance(states)

    assert decision.outcome != "dense_update"
    assert decision.report["dense_checks"]["cross_chain_location_compatible"] is False
    assert decision.report["maximum_split_rhat"] > 1.10


def test_metric_gate_rejects_undefined_explicit_chain_compatibility() -> None:
    states = np.array(
        [
            [[-1.0], [0.0], [1.0], [2.0]],
            [[-0.5], [0.5], [1.5], [2.5]],
        ]
    )

    decision = assess_metric_covariance(
        states,
        dense_min_states=2,
        diagonal_min_states=2,
    )

    assert decision.outcome == "no_update_insufficient_metric_evidence"
    assert decision.report["maximum_split_rhat"] is None
    assert decision.report["cross_chain_compatibility_status"] == (
        "undefined_fail_closed"
    )
    assert decision.report["dense_checks"]["cross_chain_location_compatible"] is False
    assert decision.report["diagonal_checks"][
        "cross_chain_location_compatible"
    ] is False


def test_metric_gate_uses_diagonal_fallback_when_dense_rank_is_inadequate() -> None:
    rng = np.random.default_rng(12)
    states = rng.normal(size=(40, 4))

    decision = assess_metric_covariance(states)

    assert decision.outcome == "diagonal_fallback"
    assert decision.report["dense_checks"]["state_count_sufficient"] is False
    assert decision.report["diagonal_fallback_used"] is True
    assert np.allclose(decision.covariance, np.diag(np.diag(decision.covariance)))


def test_affine_compatibility_composition_preserves_center_and_covariance() -> None:
    base = _transform(np.array([[4.0, 0.6], [0.6, 1.5]]), np.array([0.5, -0.4]))
    final = _transform(np.array([[1.2, -0.2], [-0.2, 0.7]]), np.array([-0.3, 0.8]))

    nested = compose_operational_transform_in_base_coordinates(
        base_transform=base,
        final_transform=final,
        adapter_signature="nested-base-adapter",
    )
    nested_transform = nested.build_latent_transform()
    latent = np.array([[0.0, 0.0], [0.2, -0.5], [-1.0, 0.3]])

    np.testing.assert_allclose(
        base.latent_to_theta(nested_transform.latent_to_position(latent)),
        final.latent_to_theta(latent),
        atol=1.0e-12,
    )


def test_reasonable_epsilon_uses_real_mean_acceptance_probability() -> None:
    adapter = _GaussianAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    result = find_reasonable_epsilon(
        adapter=latent_adapter,
        current_state=np.array([0.3, -0.2]),
        initial_step_size=4.0,
        seed=(20260711, 30),
    )

    assert result.passed is True
    assert result.selected_step_size is not None
    assert len(result.attempts) >= 1
    assert 0.25 <= result.attempts[-1].mean_acceptance_probability <= 0.75
    assert len({attempt.seed for attempt in result.attempts}) == 1
    assert all(attempt.num_leapfrog_steps == 1 for attempt in result.attempts)


def test_reasonable_epsilon_uses_requested_fixed_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorflow_probability as tfp

    observed: list[int] = []
    original_kernel = tfp.mcmc.HamiltonianMonteCarlo

    def recording_kernel(**kwargs: object):
        observed.append(int(kwargs["num_leapfrog_steps"]))
        return original_kernel(**kwargs)

    monkeypatch.setattr(tfp.mcmc, "HamiltonianMonteCarlo", recording_kernel)
    adapter = _GaussianAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    result = find_reasonable_epsilon(
        adapter=latent_adapter,
        current_state=np.array([0.3, -0.2]),
        initial_step_size=1.0,
        seed=(20260719, 1),
        num_leapfrog_steps=7,
    )

    assert result.attempts
    assert observed and set(observed) == {7}
    assert all(attempt.num_leapfrog_steps == 7 for attempt in result.attempts)


def test_reasonable_epsilon_aggregates_distinct_momentum_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorflow_probability as tfp

    observed_seeds: list[tuple[int, int]] = []
    finite_result = types.SimpleNamespace(
        target_log_prob=tf.constant(0.0, tf.float64),
        grads_target_log_prob=(tf.zeros(2, tf.float64),),
        step_size=tf.constant(1.0, tf.float64),
    )

    class _ProbeKernel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def bootstrap_results(self, _state: tf.Tensor) -> object:
            return types.SimpleNamespace(accepted_results=finite_result)

        def one_step(
            self,
            state: tf.Tensor,
            _results: object,
            *,
            seed: tf.Tensor,
        ) -> tuple[tf.Tensor, object]:
            observed_seeds.append(tuple(int(item) for item in seed.numpy()))
            return state, types.SimpleNamespace(
                log_accept_ratio=tf.math.log(tf.constant(0.5, tf.float64)),
                accepted_results=finite_result,
                proposed_state=state,
                proposed_results=finite_result,
            )

    monkeypatch.setattr(tfp.mcmc, "HamiltonianMonteCarlo", _ProbeKernel)
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=_DeclaredDomainGaussianAdapter(np.eye(2)),
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    result = find_reasonable_epsilon(
        adapter=latent_adapter,
        current_state=np.array([0.3, -0.2]),
        initial_step_size=1.0,
        seed=(20260719, 3),
        num_leapfrog_steps=5,
        momentum_probe_count=4,
    )

    assert result.passed
    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert len(attempt.probe_seeds) == 4
    assert len(set(attempt.probe_seeds)) == 4
    assert observed_seeds == list(attempt.probe_seeds)
    assert attempt.mean_acceptance_probability == pytest.approx(0.5)
    assert attempt.minimum_acceptance_probability == pytest.approx(0.5)
    assert attempt.maximum_acceptance_probability == pytest.approx(0.5)


def test_reasonable_epsilon_shrinks_after_target_domain_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorflow_probability as tfp

    finite_result = types.SimpleNamespace(
        target_log_prob=tf.constant(0.0, tf.float64),
        grads_target_log_prob=(tf.zeros(2, tf.float64),),
        step_size=tf.constant(1.0, tf.float64),
    )

    class _DomainKernel:
        def __init__(self, **kwargs: object) -> None:
            self.step_size = float(tf.convert_to_tensor(kwargs["step_size"]).numpy())

        def bootstrap_results(self, _state: tf.Tensor) -> object:
            return types.SimpleNamespace(accepted_results=finite_result)

        def one_step(
            self,
            state: tf.Tensor,
            _results: object,
            *,
            seed: tf.Tensor,
        ) -> tuple[tf.Tensor, object]:
            del seed
            if self.step_size > 1.0:
                raise tf.errors.InvalidArgumentError(
                    None,
                    None,
                    "candidate left the exact target domain",
                )
            return state, types.SimpleNamespace(
                log_accept_ratio=tf.math.log(tf.constant(0.5, tf.float64)),
                accepted_results=finite_result,
                proposed_state=state,
                proposed_results=finite_result,
            )

    monkeypatch.setattr(tfp.mcmc, "HamiltonianMonteCarlo", _DomainKernel)
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=_DeclaredDomainGaussianAdapter(np.eye(2)),
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    result = find_reasonable_epsilon(
        adapter=latent_adapter,
        current_state=np.array([0.3, -0.2]),
        initial_step_size=4.0,
        seed=(20260719, 4),
        momentum_probe_count=4,
    )

    assert result.passed
    assert result.selected_step_size == pytest.approx(1.0)
    assert result.attempts[0].engineering_health_failures == (
        "target_domain_execution_failure",
    )
    assert result.attempts[0].usable is False
    assert result.attempts[-1].usable is True


def test_operational_warmup_passes_active_l_to_every_epsilon_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_find = hmc_warmup.find_reasonable_epsilon
    observed: list[int] = []

    def recording_find(**kwargs: object) -> ReasonableEpsilonResult:
        observed.append(int(kwargs["num_leapfrog_steps"]))
        return original_find(**kwargs)

    monkeypatch.setattr(hmc_warmup, "find_reasonable_epsilon", recording_find)
    run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.array([[1.2, -0.2], [-0.2, 0.5]])),
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.35,
        trajectory_policy=WarmupTrajectoryPolicy(5, 16),
        config=WindowedMassAdaptationConfig(
            warmup_steps=112,
            initial_buffer=16,
            final_buffer=32,
            first_window_size=64,
            min_window_samples=32,
            mass_shrinkage=0.25,
        ),
        target_accept_prob=0.70,
        seed=(20260719, 2),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
    )

    assert observed
    assert set(observed) == {5}


def test_operational_warmup_passes_xla_to_epsilon_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[bool] = []
    original_find = hmc_warmup.find_reasonable_epsilon

    def recording_find(**kwargs: object) -> ReasonableEpsilonResult:
        observed.append(bool(kwargs["jit_compile"]))
        return original_find(**kwargs)

    monkeypatch.setattr(hmc_warmup, "find_reasonable_epsilon", recording_find)
    run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.eye(2)),
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.35,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=112,
            initial_buffer=16,
            final_buffer=32,
            first_window_size=64,
            min_window_samples=32,
        ),
        target_accept_prob=0.70,
        seed=(20260719, 6),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="tf_function",
        jit_compile=True,
    )

    assert observed
    assert all(observed)


def test_fixed_identity_operational_warmup_never_assesses_or_changes_mass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_metric_assessment(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("fixed-identity warmup must not assess covariance")

    monkeypatch.setattr(hmc_warmup, "assess_metric_covariance", fail_metric_assessment)
    initial_transform = _transform(np.eye(2))
    result = run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.array([[1.2, -0.2], [-0.2, 0.5]])),
        initial_transform=initial_transform,
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.35,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=20,
            initial_buffer=2,
            final_buffer=8,
            first_window_size=10,
            min_window_samples=2,
            mass_policy="fixed_identity",
        ),
        target_accept_prob=0.70,
        seed=(20260721, 1),
        target_scope="hmc_warmup_fixed_identity",
        chain_execution_mode="eager",
    )

    assert result.operational_metric_update_count == 0
    assert all(window.metric_decision is None for window in result.windows)
    assert result.final_kernel_state.adaptation_generation == 0
    assert result.final_kernel_state.transform.signature == initial_transform.signature
    assert result.final_kernel_state.momentum_metric.signature == result.windows[0].metric_signature_used


def test_operational_warmup_external_bound_caps_consumed_step() -> None:
    upper_bound = 0.2
    result = run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.eye(2)),
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=upper_bound,
        initial_step_size_upper_bound=upper_bound,
        initial_step_qualification_source="unit_test_fixed_kernel_screen",
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=20,
            initial_buffer=2,
            final_buffer=8,
            first_window_size=10,
            min_window_samples=2,
        ),
        target_accept_prob=0.05,
        seed=(20260719, 5),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
    )

    assert result.reasonable_epsilon.status == "externally_qualified"
    assert result.reasonable_epsilon.attempts == ()
    assert result.reasonable_epsilon.qualification_source == (
        "unit_test_fixed_kernel_screen"
    )
    assert all(
        np.max(window.consumed_step_size_trace) <= upper_bound * (1.0 + 1.0e-12)
        for window in result.windows
    )
    assert all(
        np.max(window.step_size_trace) <= upper_bound * (1.0 + 1.0e-12)
        for window in result.windows
    )
    assert any(
        np.max(window.proposed_step_size_trace) > upper_bound
        for window in result.windows
    )
    assert sum(
        window.public_payload()["step_ceiling_hit_count"]
        for window in result.windows
    ) > 0


@pytest.mark.parametrize(
    ("upper_bound", "source", "message"),
    (
        (0.1, "screen", "bound the initial step"),
        (0.5, None, "requires qualification provenance"),
        (None, "screen", "requires an upper bound"),
    ),
)
def test_operational_warmup_rejects_invalid_external_bound_contract(
    upper_bound: float | None,
    source: str | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_operational_windowed_warmup(
            adapter=_GaussianAdapter(np.eye(2)),
            initial_transform=_transform(np.eye(2)),
            initial_canonical_theta=np.array([0.2, -0.1]),
            initial_step_size=0.2,
            initial_step_size_upper_bound=upper_bound,
            initial_step_qualification_source=source,
            trajectory_policy=WarmupTrajectoryPolicy(2, 8),
            config=WindowedMassAdaptationConfig(
                warmup_steps=20,
                initial_buffer=2,
                final_buffer=8,
                first_window_size=10,
                min_window_samples=2,
            ),
            target_accept_prob=0.70,
            seed=(20260719, 6),
            target_scope="hmc_warmup_gaussian",
            chain_execution_mode="eager",
        )


def test_reasonable_epsilon_failure_payload_is_standard_json() -> None:
    attempt = ReasonableEpsilonAttempt(
        step_size=1.0,
        mean_acceptance_probability=None,
        finite=False,
        seed=(20260711, 31),
    )

    assert json.loads(json.dumps(attempt.payload(), allow_nan=False))[
        "mean_acceptance_probability"
    ] is None
    with pytest.raises(ValueError, match="normalize to None"):
        ReasonableEpsilonAttempt(
            step_size=1.0,
            mean_acceptance_probability=float("nan"),
            finite=False,
            seed=(20260711, 31),
        )


def test_reasonable_epsilon_rejects_nonfinite_initial_score() -> None:
    adapter = _NonfiniteInitialScoreAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    with pytest.raises(ValueError, match="target value and score must be finite"):
        find_reasonable_epsilon(
            adapter=latent_adapter,
            current_state=np.array([0.3, -0.2]),
            initial_step_size=1.0,
            seed=(20260711, 32),
        )


def test_reasonable_epsilon_may_shrink_after_nonfinite_proposal() -> None:
    adapter = _NonfiniteProposalOutsideRadiusAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    result = find_reasonable_epsilon(
        adapter=latent_adapter,
        current_state=np.array([0.1, -0.1]),
        initial_step_size=4.0,
        seed=(20260711, 33),
    )

    assert result.passed is False
    assert result.status == "inconclusive_bracket"
    assert any(not attempt.finite for attempt in result.attempts)
    first_finite_after_failure = next(
        index
        for index, attempt in enumerate(result.attempts)
        if index > 0
        and not result.attempts[index - 1].finite
        and attempt.finite
    )
    assert result.attempts[first_finite_after_failure].step_size < (
        result.attempts[first_finite_after_failure - 1].step_size
    )


def test_reasonable_epsilon_does_not_treat_runner_exception_as_low_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorflow_probability as tfp

    finite_result = types.SimpleNamespace(
        target_log_prob=tf.constant(0.0, tf.float64),
        grads_target_log_prob=(tf.zeros(2, tf.float64),),
    )

    class _FailingKernel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def bootstrap_results(self, _state: tf.Tensor) -> object:
            return types.SimpleNamespace(accepted_results=finite_result)

        def one_step(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("unexpected kernel execution failure")

    monkeypatch.setattr(tfp.mcmc, "HamiltonianMonteCarlo", _FailingKernel)
    adapter = _GaussianAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    with pytest.raises(
        RuntimeError,
        match="reasonable-epsilon HMC proposal execution failed",
    ):
        find_reasonable_epsilon(
            adapter=latent_adapter,
            current_state=np.array([0.1, -0.1]),
            initial_step_size=1.0,
            seed=(20260711, 36),
        )


def _patch_reasonable_epsilon_kernel(
    monkeypatch: pytest.MonkeyPatch,
    *,
    retain_invalid_proposal: bool,
) -> None:
    import tensorflow_probability as tfp

    finite_result = types.SimpleNamespace(
        target_log_prob=tf.constant(0.0, tf.float64),
        grads_target_log_prob=(tf.zeros(2, tf.float64),),
    )

    class _FakeKernel:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def bootstrap_results(self, _state: tf.Tensor) -> object:
            return types.SimpleNamespace(accepted_results=finite_result)

        def one_step(
            self,
            state: tf.Tensor,
            _results: object,
            *,
            seed: tf.Tensor,
        ) -> tuple[tf.Tensor, object]:
            del seed
            proposal = tf.ones_like(state)
            return (
                proposal if retain_invalid_proposal else state,
                types.SimpleNamespace(
                    log_accept_ratio=tf.constant(-100.0, tf.float64),
                    accepted_results=finite_result,
                    proposed_state=proposal,
                    proposed_results=finite_result,
                ),
            )

    monkeypatch.setattr(tfp.mcmc, "HamiltonianMonteCarlo", _FakeKernel)


def test_reasonable_epsilon_keeps_rejected_target_status_failure_separate_from_finiteness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reasonable_epsilon_kernel(
        monkeypatch,
        retain_invalid_proposal=False,
    )
    adapter = _TargetStatusOutsideRadiusAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    result = find_reasonable_epsilon(
        adapter=latent_adapter,
        current_state=np.array([0.1, -0.1]),
        initial_step_size=4.0,
        seed=(20260711, 34),
        target_status_trace_policy="per_chain_step",
    )

    vetoed = next(
        attempt for attempt in result.attempts if attempt.engineering_health_failures
    )
    assert vetoed.finite is True
    assert vetoed.usable is False
    assert vetoed.engineering_health_failures == ("target_status_telemetry_failure",)


def test_reasonable_epsilon_rejects_retained_target_status_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_reasonable_epsilon_kernel(
        monkeypatch,
        retain_invalid_proposal=True,
    )
    adapter = _TargetStatusOutsideRadiusAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )

    with pytest.raises(
        ValueError,
        match="accepted or retained target status is nonvalid",
    ):
        find_reasonable_epsilon(
            adapter=latent_adapter,
            current_state=np.array([0.1, -0.1]),
            initial_step_size=4.0,
            seed=(20260711, 35),
            target_status_trace_policy="per_chain_step",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("max_attempts", 3.5, "integer scalar"),
        ("seed", (20260711, 30.5), "integer scalar"),
        ("current_state", np.empty(0), "non-empty and finite"),
    ),
)
def test_reasonable_epsilon_rejects_malformed_authority_inputs(
    field: str,
    value: object,
    message: str,
) -> None:
    adapter = _GaussianAdapter(np.eye(2))
    latent_adapter = _AffineWarmupAdapter(
        base_adapter=adapter,
        transform=_transform(np.eye(2)),
        target_scope="hmc_warmup_gaussian",
    )
    kwargs = {
        "adapter": latent_adapter,
        "current_state": np.array([0.3, -0.2]),
        "initial_step_size": 1.0,
        "seed": (20260711, 30),
        "max_attempts": 2,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        find_reasonable_epsilon(**kwargs)


def test_operational_schedule_reserves_four_final_coordinate_states() -> None:
    source = WindowedMassAdaptationConfig(
        warmup_steps=12,
        initial_buffer=2,
        final_buffer=2,
        first_window_size=3,
        min_window_samples=2,
    )

    normalized = normalize_operational_warmup_config(source)

    assert source.final_buffer == 2
    assert normalized.warmup_steps == source.warmup_steps
    assert normalized.initial_buffer == source.initial_buffer
    assert normalized.final_buffer == 4
    assert normalized.warmup_steps - normalized.initial_buffer - normalized.final_buffer >= 2


def test_operational_schedule_fails_when_final_bank_cannot_be_reserved() -> None:
    source = WindowedMassAdaptationConfig(
        warmup_steps=6,
        initial_buffer=1,
        final_buffer=1,
        first_window_size=2,
        min_window_samples=2,
    )

    with pytest.raises(ValueError, match="reserve four"):
        normalize_operational_warmup_config(source)


def test_real_operational_warmup_uses_updated_metric_in_later_transition() -> None:
    angle = np.pi / 5.0
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    target_covariance = rotation @ np.diag([1.0, 0.1]) @ rotation.T
    adapter = _GaussianAdapter(target_covariance)
    config = WindowedMassAdaptationConfig(
        warmup_steps=112,
        initial_buffer=16,
        final_buffer=32,
        first_window_size=64,
        min_window_samples=32,
        mass_shrinkage=0.25,
    )

    result = run_operational_windowed_warmup(
        adapter=adapter,
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.4, -0.3]),
        initial_step_size=0.5,
        trajectory_policy=WarmupTrajectoryPolicy(3, 16),
        config=config,
        target_accept_prob=0.70,
        seed=(20260711, 40),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="tf_function",
    )

    assert result.operational_metric_update_count >= 1
    assert result.every_update_used_by_later_transition is True
    update_index = next(
        index
        for index, window in enumerate(result.windows)
        if window.next_coordinate_signature is not None
    )
    assert result.windows[update_index + 1].coordinate_signature_used == (
        result.windows[update_index].next_coordinate_signature
    )
    assert result.windows[update_index + 1].metric_signature_used == (
        result.windows[update_index].next_metric_signature
    )
    assert result.windows[update_index].state_map_residual < 1.0e-10
    assert result.windows[update_index].target_value_map_residual < 1.0e-10
    assert result.windows[update_index].target_score_map_residual < 1.0e-10
    assert result.windows[update_index].next_reasonable_epsilon is not None
    assert all(window.runner_trace_count in {None, 1} for window in result.windows)
    assert result.final_kernel_state.epsilon is not None
    payload = result.public_payload()
    assert payload["private_start_bank"]["count"] == 4
    assert payload["private_start_bank"]["raw_values_exposed"] is False
    assert "private_start_bank_theta" not in str(payload)


def test_latent_metric_refinement_preserves_qualified_initial_correlations() -> None:
    initial_covariance = np.array([[2.0, 0.9], [0.9, 1.0]])
    initial_transform = _transform(initial_covariance)
    result = run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.array([[1.4, -0.3], [-0.3, 0.5]])),
        initial_transform=initial_transform,
        initial_canonical_theta=np.array([0.4, -0.3]),
        initial_step_size=0.35,
        trajectory_policy=WarmupTrajectoryPolicy(3, 16),
        config=WindowedMassAdaptationConfig(
            warmup_steps=112,
            initial_buffer=16,
            final_buffer=32,
            first_window_size=64,
            min_window_samples=32,
            mass_shrinkage=0.25,
        ),
        target_accept_prob=0.70,
        seed=(20260717, 10),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
    )

    update = next(
        window
        for window in result.windows
        if window.metric_decision is not None
        and window.metric_decision.update_applied
    )
    latent_covariance = np.asarray(update.metric_decision.covariance, dtype=float)
    expected_canonical = (
        initial_transform.factor
        @ latent_covariance
        @ initial_transform.factor.T
    )
    np.testing.assert_allclose(
        result.final_kernel_state.transform.covariance,
        expected_canonical,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert abs(expected_canonical[0, 1]) > 1.0e-6


def test_metric_boundary_epsilon_failure_rolls_back_to_qualified_initializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_transform = _transform(np.array([[1.8, 0.7], [0.7, 1.0]]))
    original_find = hmc_warmup.find_reasonable_epsilon
    calls = 0

    def reject_only_metric_boundary(**kwargs: object) -> ReasonableEpsilonResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original_find(**kwargs)
        return ReasonableEpsilonResult(
            "inconclusive_bracket",
            None,
            (
                ReasonableEpsilonAttempt(
                    step_size=0.1,
                    mean_acceptance_probability=None,
                    finite=False,
                    seed=(20260717, 12),
                ),
            ),
        )

    monkeypatch.setattr(
        hmc_warmup,
        "find_reasonable_epsilon",
        reject_only_metric_boundary,
    )

    result = run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.array([[1.2, -0.25], [-0.25, 0.45]])),
        initial_transform=initial_transform,
        initial_canonical_theta=np.array([0.4, -0.3]),
        initial_step_size=0.35,
        trajectory_policy=WarmupTrajectoryPolicy(3, 16),
        config=WindowedMassAdaptationConfig(
            warmup_steps=112,
            initial_buffer=16,
            final_buffer=32,
            first_window_size=64,
            min_window_samples=32,
            mass_shrinkage=0.25,
        ),
        target_accept_prob=0.70,
        seed=(20260717, 11),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
    )

    rejected = next(
        window
        for window in result.windows
        if window.metric_decision is not None
        and window.metric_decision.outcome == "candidate_metric_rejected"
    )
    assert calls == 2
    assert rejected.metric_decision.update_applied is False
    assert rejected.metric_decision.report["candidate_metric_evidence_passed"] is True
    assert rejected.metric_decision.report["candidate_rejection_stage"] == (
        "reasonable_epsilon"
    )
    assert rejected.next_coordinate_signature is None
    assert rejected.next_metric_signature is None
    assert rejected.next_reasonable_epsilon is None
    assert result.operational_metric_update_count == 0
    assert result.final_kernel_state.adaptation_generation == 0
    assert result.final_kernel_state.transform.signature == initial_transform.signature
    np.testing.assert_allclose(
        result.final_kernel_state.transform.covariance,
        initial_transform.covariance,
        rtol=0.0,
        atol=0.0,
    )


def test_operational_warmup_collects_and_vetoes_requested_target_status() -> None:
    config = WindowedMassAdaptationConfig(
        warmup_steps=20,
        initial_buffer=2,
        final_buffer=8,
        first_window_size=10,
        min_window_samples=2,
    )
    kwargs = {
        "initial_transform": _transform(np.eye(2)),
        "initial_canonical_theta": np.array([0.2, -0.1]),
        "initial_step_size": 0.5,
        "trajectory_policy": WarmupTrajectoryPolicy(2, 8),
        "config": config,
        "target_accept_prob": 0.70,
        "seed": (20260711, 49),
        "target_scope": "hmc_warmup_gaussian",
        "chain_execution_mode": "eager",
        "target_status_trace_policy": "per_chain_step",
    }

    result = run_operational_windowed_warmup(
        adapter=_TargetStatusGaussianAdapter(np.eye(2)),
        **kwargs,
    )

    assert result.target_status_trace_policy == "per_chain_step"
    assert all(window.target_status_failure_count == 0 for window in result.windows)
    assert result.public_payload()["target_status_trace_policy"] == "per_chain_step"

    with pytest.raises(ValueError, match="target-status telemetry"):
        run_operational_windowed_warmup(
            adapter=_TargetStatusGaussianAdapter(np.eye(2), nonvalid=True),
            **kwargs,
        )

    with pytest.raises(TypeError, match="requires adapter telemetry"):
        run_operational_windowed_warmup(
            adapter=_GaussianAdapter(np.eye(2)),
            **kwargs,
        )


def test_tiny_operational_warmup_does_not_claim_dense_metric() -> None:
    adapter = _GaussianAdapter(np.eye(2))
    config = WindowedMassAdaptationConfig(
        warmup_steps=20,
        initial_buffer=2,
        final_buffer=8,
        first_window_size=10,
        min_window_samples=2,
    )

    result = run_operational_windowed_warmup(
        adapter=adapter,
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.5,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=config,
        target_accept_prob=0.70,
        seed=(20260711, 50),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
    )

    decisions = [window.metric_decision for window in result.windows if window.metric_decision]
    assert decisions
    assert all(decision.outcome == "no_update_insufficient_metric_evidence" for decision in decisions)
    assert result.operational_metric_update_count == 0


def test_operational_stage_callback_is_additive_public_safe_and_observation_only() -> None:
    events: list[tuple[str, dict[str, object]]] = []

    def observe_stage(event: str, payload: Mapping[str, object]) -> None:
        events.append((event, dict(payload)))
        return None

    result = run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.eye(2)),
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.5,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=20,
            initial_buffer=2,
            final_buffer=8,
            first_window_size=10,
            min_window_samples=2,
        ),
        target_accept_prob=0.70,
        seed=(20260718, 701),
        target_scope="hmc_warmup_stage_callback",
        chain_execution_mode="eager",
        stage_callback=observe_stage,
    )

    assert result.status == "passed"
    assert events
    assert {event for event, _payload in events} == {"stage_start", "stage_complete"}
    stages = [payload["stage"] for _event, payload in events]
    assert stages.count("post_window_conversion") == 6
    assert stages.count("retained_target_health") == 6
    assert stages.count("metric_assessment") == 2
    for event, payload in events:
        assert payload["progress_only"] is True
        assert payload["states_exposed"] is False
        assert payload["scores_exposed"] is False
        assert payload["metric_exposed"] is False
        assert payload["epsilon_exposed"] is False
        assert payload["logical_draw_count"] > 0
        assert "stage_elapsed_s" in payload if event == "stage_complete" else True
        assert not any(
            forbidden in str(payload).lower()
            for forbidden in ("canonical_theta", "latent_state", "covariance", "step_size")
        )


def test_operational_stage_callback_cannot_return_a_closeout_payload() -> None:
    def invalid_stage_callback(
        _event: str, _payload: Mapping[str, object]
    ) -> dict[str, object]:
        return {"stop": True}

    with pytest.raises(ValueError, match="stage callback must return None"):
        run_operational_windowed_warmup(
            adapter=_GaussianAdapter(np.eye(2)),
            initial_transform=_transform(np.eye(2)),
            initial_canonical_theta=np.array([0.2, -0.1]),
            initial_step_size=0.5,
            trajectory_policy=WarmupTrajectoryPolicy(2, 8),
            config=WindowedMassAdaptationConfig(
                warmup_steps=20,
                initial_buffer=2,
                final_buffer=8,
                first_window_size=10,
                min_window_samples=2,
            ),
            target_accept_prob=0.70,
            seed=(20260718, 702),
            target_scope="hmc_warmup_invalid_stage_callback",
            chain_execution_mode="eager",
            stage_callback=invalid_stage_callback,
        )


def test_operational_timeout_closeout_occurs_after_completed_window_commit() -> None:
    observed: list[tuple[str, tuple[dict[str, object], ...]]] = []

    def close_after_one_window(
        boundary: str,
        completed_windows: tuple[dict[str, object], ...],
    ) -> dict[str, object] | None:
        observed.append((boundary, completed_windows))
        if boundary == "before_next_window":
            return {
                "remaining_s": 10.0,
                "reserve_s": 10.0,
                "within_closeout_window": True,
            }
        return None

    result = run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.eye(2)),
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.5,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=20,
            initial_buffer=2,
            final_buffer=8,
            first_window_size=10,
            min_window_samples=2,
        ),
        target_accept_prob=0.70,
        seed=(20260711, 501),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
        boundary_callback=close_after_one_window,
    )

    assert isinstance(result, OperationalWindowedWarmupCloseout)
    assert observed[0] == ("before_first_window", ())
    assert observed[1][0] == "before_next_window"
    assert len(observed[1][1]) == 1
    first_window = observed[1][1][0]
    assert first_window["transition_count_after_window"] == result.completed_transition_count
    assert first_window["raw_states_exposed"] is False
    public = result.public_payload()
    assert public["completed_window_count"] == 1
    assert public["completed_warmup_result"] is False
    assert public["private_start_bank_exposed"] is False
    assert public["candidate_selection_authorized"] is False
    assert public["legacy_fallback_used"] is False
    assert public["boundary_payload"]["remaining_s"] == public["boundary_payload"][
        "reserve_s"
    ]


def test_operational_segment_closeout_preserves_partial_transition_counts() -> None:
    completed_counts: list[int] = []

    def close_after_one_segment(
        event: str,
        segment: Mapping[str, object],
    ) -> dict[str, object] | None:
        if event == "segment_complete":
            completed_counts.append(int(segment["completed_transition_count"]))
            return None
        if int(segment["completed_transition_count"]) == 4:
            return {
                "stop_source": "bayesfilter_public_timeout_budget",
                "stop_reason": "test_resource_closeout",
                "supervision_counter_baseline": 4,
            }
        return None

    result = run_operational_windowed_warmup(
        adapter=_GaussianAdapter(np.eye(2)),
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.5,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=20,
            initial_buffer=8,
            final_buffer=4,
            first_window_size=8,
            min_window_samples=2,
        ),
        target_accept_prob=0.70,
        seed=(20260711, 502),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
        execution_segment_size=4,
        segment_callback=close_after_one_segment,
    )

    assert isinstance(result, OperationalWindowedWarmupCloseout)
    assert completed_counts == [4]
    assert result.completed_windows == ()
    assert result.completed_transition_count == 4
    assert result.planned_transition_count == 20
    assert result.completed_segment_count == 1
    assert result.completed_segment_count < result.planned_segment_count
    public = result.public_payload()
    assert public["remaining_transition_count"] == 16
    assert public["stop_source"] == "bayesfilter_public_timeout_budget"
    assert public["stop_reason"] == "test_resource_closeout"
    assert public["supervision_counter_baseline"] == 4
    assert public["candidate_selection_authorized"] is False


def test_operational_warmup_live_result_rejects_corrupt_window_ledger() -> None:
    adapter = _GaussianAdapter(np.eye(2))
    result = run_operational_windowed_warmup(
        adapter=adapter,
        initial_transform=_transform(np.eye(2)),
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.5,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=20,
            initial_buffer=2,
            final_buffer=8,
            first_window_size=10,
            min_window_samples=2,
        ),
        target_accept_prob=0.70,
        seed=(20260711, 51),
        target_scope="hmc_warmup_gaussian",
        chain_execution_mode="eager",
    )

    first = result.windows[0]
    with pytest.raises(ValueError, match="arrays are misaligned"):
        replace(first, log_accept_ratio=first.log_accept_ratio[:-1])
    with pytest.raises(ValueError, match="transition counts"):
        replace(first, transition_count_before_window=1)
    with pytest.raises(ValueError, match="acceptance summary"):
        replace(first, mean_acceptance_probability=0.123)
    with pytest.raises(ValueError, match="one finite transform vector"):
        run_operational_windowed_warmup(
            adapter=adapter,
            initial_transform=_transform(np.eye(2)),
            initial_canonical_theta=np.zeros((2, 2)),
            initial_step_size=0.5,
            trajectory_policy=WarmupTrajectoryPolicy(2, 8),
            config=result.config,
            target_accept_prob=0.70,
            seed=(20260711, 52),
            target_scope="hmc_warmup_gaussian",
            chain_execution_mode="eager",
        )


def test_operational_warmup_rejects_duplicate_start_bank() -> None:
    with pytest.raises(ValueError, match="start bank"):
        build_private_start_bank(np.zeros((8, 1)))
