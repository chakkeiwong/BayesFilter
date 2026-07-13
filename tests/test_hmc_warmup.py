from __future__ import annotations

import json
import os
import types
from dataclasses import replace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_coordinates import (
    AffineCoordinateTransform,
    PositionCovarianceEstimate,
    WarmupTrajectoryPolicy,
)
from bayesfilter.inference.hmc_tuning import WindowedMassAdaptationConfig
from bayesfilter.inference.hmc_warmup import (
    _AffineWarmupAdapter,
    ReasonableEpsilonAttempt,
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
