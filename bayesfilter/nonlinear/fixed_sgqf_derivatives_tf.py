"""Analytic first-order derivatives for the Fixed-SGQF lane."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Sequence

import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_tf import (
    FIXED_SGQF_RUNTIME_MODE,
    TFFixedSGQFBranchConfig,
    TFFixedSGQFBranchIdentity,
    TFFixedSGQFCloud,
    TFFixedSGQFNonlinearModel,
    TFFixedSGQFStepFailure,
    _as_observation_matrix,
    _as_points,
    _cholesky_factor_or_failure,
    _symmetrize,
    _weighted_covariance,
    _weighted_mean,
)


TFTransitionStateJacobianFn = Callable[[tf.Tensor], tf.Tensor]
TFTransitionParameterDerivativeFn = Callable[[tf.Tensor], tf.Tensor]
TFObservationStateJacobianFn = Callable[[tf.Tensor], tf.Tensor]
TFObservationParameterDerivativeFn = Callable[[tf.Tensor], tf.Tensor]
TFPointMapJVPFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class TFFixedSGQFDerivatives:
    """First-order derivatives required by the Fixed-SGQF score path."""

    d_initial_mean: tf.Tensor
    d_initial_covariance: tf.Tensor
    d_process_covariance: tf.Tensor
    d_observation_covariance: tf.Tensor
    transition_state_jacobian_fn: TFTransitionStateJacobianFn
    d_transition_fn: TFTransitionParameterDerivativeFn
    observation_state_jacobian_fn: TFObservationStateJacobianFn
    d_observation_fn: TFObservationParameterDerivativeFn
    name: str = "tf_fixed_sgqf_derivatives"
    transition_jvp_fn: TFPointMapJVPFn | None = None
    observation_jvp_fn: TFPointMapJVPFn | None = None

    def __post_init__(self) -> None:
        for name in (
            "d_initial_mean",
            "d_initial_covariance",
            "d_process_covariance",
            "d_observation_covariance",
        ):
            object.__setattr__(self, name, tf.convert_to_tensor(getattr(self, name), dtype=tf.float64))
        self.validate_static_shapes()

    @property
    def parameter_dim(self) -> int | None:
        return self.d_initial_mean.shape[0]

    @property
    def state_dim(self) -> int | None:
        return self.d_initial_mean.shape[-1]

    @property
    def observation_dim(self) -> int | None:
        return self.d_observation_covariance.shape[-1]

    def validate_static_shapes(self) -> None:
        p = self.parameter_dim
        n = self.state_dim
        m = self.observation_dim
        if p is None or n is None or m is None:
            return
        expected = {
            "d_initial_mean": (p, n),
            "d_initial_covariance": (p, n, n),
            "d_process_covariance": (p, n, n),
            "d_observation_covariance": (p, m, m),
        }
        for name, shape in expected.items():
            actual = tuple(getattr(self, name).shape.as_list())
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")


@dataclass(frozen=True)
class TFFixedSGQFScoreResult:
    """Analytic score result for the Fixed-SGQF lane."""

    log_likelihood: tf.Tensor | None
    score: tf.Tensor | None
    branch_identity: TFFixedSGQFBranchIdentity
    diagnostics: Mapping[str, object]
    filtered_mean: tf.Tensor | None = None
    filtered_covariance: tf.Tensor | None = None
    d_filtered_mean: tf.Tensor | None = None
    d_filtered_covariance: tf.Tensor | None = None
    failure: TFFixedSGQFStepFailure | None = None

    def __post_init__(self) -> None:
        if self.log_likelihood is not None:
            object.__setattr__(self, "log_likelihood", tf.convert_to_tensor(self.log_likelihood, dtype=tf.float64))
        if self.score is not None:
            object.__setattr__(self, "score", tf.convert_to_tensor(self.score, dtype=tf.float64))
        if self.filtered_mean is not None:
            object.__setattr__(self, "filtered_mean", tf.convert_to_tensor(self.filtered_mean, dtype=tf.float64))
        if self.filtered_covariance is not None:
            object.__setattr__(self, "filtered_covariance", tf.convert_to_tensor(self.filtered_covariance, dtype=tf.float64))
        if self.d_filtered_mean is not None:
            object.__setattr__(self, "d_filtered_mean", tf.convert_to_tensor(self.d_filtered_mean, dtype=tf.float64))
        if self.d_filtered_covariance is not None:
            object.__setattr__(self, "d_filtered_covariance", tf.convert_to_tensor(self.d_filtered_covariance, dtype=tf.float64))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if not isinstance(self.branch_identity, TFFixedSGQFBranchIdentity):
            raise TypeError("branch_identity must be a TFFixedSGQFBranchIdentity")
        if self.failure is not None and not isinstance(self.failure, TFFixedSGQFStepFailure):
            raise TypeError("failure must be a TFFixedSGQFStepFailure")


def _freeze_mapping(values: Mapping[str, object] | None) -> Mapping[str, object]:
    return MappingProxyType(dict(values or {}))


def _einsum_pointwise_jacobian(jacobians: tf.Tensor, point_derivatives: tf.Tensor) -> tf.Tensor:
    """Apply pointwise Jacobians to per-parameter point derivatives."""

    return tf.einsum("rmn,prn->prm", jacobians, point_derivatives)


def _point_map_first_order_values(
    points: tf.Tensor,
    point_derivatives: tf.Tensor,
    *,
    jacobian_fn: TFTransitionStateJacobianFn | TFObservationStateJacobianFn,
    parameter_derivative_fn: TFTransitionParameterDerivativeFn | TFObservationParameterDerivativeFn,
    jvp_fn: TFPointMapJVPFn | None,
) -> tf.Tensor:
    direct = parameter_derivative_fn(points)
    propagated = (
        jvp_fn(points, point_derivatives)
        if jvp_fn is not None
        else _einsum_pointwise_jacobian(jacobian_fn(points), point_derivatives)
    )
    return propagated + direct


def _covariance_first_derivative(centered: tf.Tensor, d_centered: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return _symmetrize(
        tf.einsum("r,prn,rm->pnm", weights, d_centered, centered)
        + tf.einsum("r,rn,prm->pnm", weights, centered, d_centered)
    )


def _cholesky_first_derivative(factor: tf.Tensor, d_covariance: tf.Tensor) -> tf.Tensor:
    """Differentiate a lower-triangular Cholesky factor analytically."""

    factor_inv = tf.linalg.inv(factor)
    inner = tf.einsum(
        "ab,pbc,dc->pad",
        factor_inv,
        d_covariance,
        factor_inv,
    )
    lower = tf.linalg.band_part(inner, -1, 0)
    diag = tf.linalg.diag(tf.linalg.diag_part(lower))
    phi = lower - 0.5 * diag
    return tf.einsum("ab,pbd->pad", factor, phi)


def _score_diagnostics(
    *,
    branch_identity: TFFixedSGQFBranchIdentity,
    branch_config: TFFixedSGQFBranchConfig,
    cloud: TFFixedSGQFCloud,
    accepted_steps: int,
    failure: TFFixedSGQFStepFailure | None,
    derivative_method: str | None = None,
    same_branch_signature: tuple[str, str, int] | None = None,
) -> Mapping[str, object]:
    diagnostics: dict[str, object] = {
        "branch_hash": branch_identity.hash.value,
        "accepted_steps": tf.convert_to_tensor(accepted_steps, dtype=tf.int32),
        "cloud_point_count": tf.convert_to_tensor(cloud.point_count, dtype=tf.int32),
        "weight_total": tf.convert_to_tensor(cloud.weight_total, dtype=tf.float64),
        "negative_weight_count": tf.convert_to_tensor(cloud.negative_weight_count, dtype=tf.int32),
        "rule_family": cloud.rule_family,
        "runtime_mode": FIXED_SGQF_RUNTIME_MODE,
        "observation_preprocessing": branch_config.observation_preprocessing,
        "initial_condition_policy": branch_config.initial_condition_policy,
        "failure_record_policy": branch_config.failure_record_policy,
        "factor_branch": branch_config.factor_branch,
        "additive_noise_policy": branch_config.additive_noise_policy,
        "veto_policy": branch_config.veto_policy,
        "predictive_epsilon": tf.convert_to_tensor(branch_config.predictive_epsilon, dtype=tf.float64),
        "innovation_epsilon": tf.convert_to_tensor(branch_config.innovation_epsilon, dtype=tf.float64),
    }
    if derivative_method is not None:
        diagnostics["derivative_method"] = derivative_method
    if same_branch_signature is not None:
        diagnostics["same_branch_signature"] = same_branch_signature
    if failure is not None:
        diagnostics["failure_stage"] = failure.stage
        diagnostics["failure_reason"] = failure.reason
        diagnostics["failure_time_index"] = tf.convert_to_tensor(failure.time_index, dtype=tf.int32)
    return _freeze_mapping(diagnostics)


def tf_fixed_sgqf_same_branch_signature(
    *,
    branch_identity: TFFixedSGQFBranchIdentity,
    failure: TFFixedSGQFStepFailure | None,
) -> tuple[str, str, int]:
    """Return a compact same-scalar signature for FD comparisons."""

    if failure is None:
        return branch_identity.hash.value, "accepted", -1
    return branch_identity.hash.value, str(failure.stage), int(failure.time_index)


def tf_fixed_sgqf_score(
    observations: tf.Tensor,
    model: TFFixedSGQFNonlinearModel,
    derivatives: TFFixedSGQFDerivatives,
    *,
    cloud: TFFixedSGQFCloud,
    branch_config: TFFixedSGQFBranchConfig | None = None,
    branch_identity: TFFixedSGQFBranchIdentity | None = None,
    expected_branch_identity: TFFixedSGQFBranchIdentity | None = None,
) -> TFFixedSGQFScoreResult:
    """Evaluate the analytic Fixed-SGQF score on the declared branch."""

    y = _as_observation_matrix(observations)
    branch_config = branch_config or TFFixedSGQFBranchConfig()
    branch_identity = branch_identity or branch_config.branch_identity(cloud)
    if expected_branch_identity is not None and expected_branch_identity != branch_identity:
        failure = TFFixedSGQFStepFailure(
            time_index=0,
            stage="branch_signature",
            reason="same_scalar_branch_mismatch",
            diagnostics={
                "expected_branch_hash": expected_branch_identity.hash.value,
                "actual_branch_hash": branch_identity.hash.value,
            },
        )
        return TFFixedSGQFScoreResult(
            log_likelihood=None,
            score=None,
            branch_identity=branch_identity,
            diagnostics=_score_diagnostics(
                branch_identity=branch_identity,
                branch_config=branch_config,
                cloud=cloud,
                accepted_steps=0,
                failure=failure,
                same_branch_signature=tf_fixed_sgqf_same_branch_signature(
                    branch_identity=branch_identity,
                    failure=failure,
                ),
            ),
            failure=failure,
        )

    mean = tf.convert_to_tensor(model.initial_mean, dtype=tf.float64)
    covariance = _symmetrize(model.initial_covariance)
    process_covariance = _symmetrize(model.process_covariance)
    observation_covariance = _symmetrize(model.observation_covariance)
    d_mean = tf.convert_to_tensor(derivatives.d_initial_mean, dtype=tf.float64)
    d_covariance = _symmetrize(derivatives.d_initial_covariance)
    d_process_covariance = _symmetrize(derivatives.d_process_covariance)
    d_observation_covariance = _symmetrize(derivatives.d_observation_covariance)

    parameter_dim = int(d_mean.shape[0])
    state_dim = int(mean.shape[0])
    observation_dim = int(observation_covariance.shape[0])
    weights = tf.convert_to_tensor(cloud.weights, dtype=tf.float64)
    log_likelihood = tf.constant(0.0, dtype=tf.float64)
    score = tf.zeros([parameter_dim], dtype=tf.float64)
    accepted_steps = 0

    for time_index in range(int(y.shape[0])):
        previous_factor, _previous_diag, previous_failure = _cholesky_factor_or_failure(
            covariance,
            epsilon=branch_config.predictive_epsilon,
            time_index=time_index,
            stage="previous_covariance",
        )
        if previous_failure is not None:
            return TFFixedSGQFScoreResult(
                log_likelihood=None,
                score=None,
                branch_identity=branch_identity,
                diagnostics=_score_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=previous_failure,
                    same_branch_signature=tf_fixed_sgqf_same_branch_signature(
                        branch_identity=branch_identity,
                        failure=previous_failure,
                    ),
                ),
                failure=previous_failure,
            )
        d_previous_factor = _cholesky_first_derivative(previous_factor, d_covariance)

        previous_points = mean[tf.newaxis, :] + cloud.points @ tf.transpose(previous_factor)
        d_previous_points = d_mean[:, tf.newaxis, :] + tf.einsum("rd,pnd->prn", cloud.points, d_previous_factor)

        transition_values = model.transition(previous_points)
        d_transition_values = _point_map_first_order_values(
            previous_points,
            d_previous_points,
            jacobian_fn=derivatives.transition_state_jacobian_fn,
            parameter_derivative_fn=derivatives.d_transition_fn,
            jvp_fn=derivatives.transition_jvp_fn,
        )

        predicted_mean = _weighted_mean(transition_values, weights)
        d_predicted_mean = tf.einsum("r,prn->pn", weights, d_transition_values)
        centered_predicted = transition_values - predicted_mean[tf.newaxis, :]
        d_centered_predicted = d_transition_values - d_predicted_mean[:, tf.newaxis, :]
        predicted_covariance = _symmetrize(process_covariance + _weighted_covariance(centered_predicted, weights))
        d_predicted_covariance = _symmetrize(
            d_process_covariance + _covariance_first_derivative(centered_predicted, d_centered_predicted, weights)
        )

        predicted_factor, _predictive_diag, predictive_failure = _cholesky_factor_or_failure(
            predicted_covariance,
            epsilon=branch_config.predictive_epsilon,
            time_index=time_index,
            stage="predictive_covariance",
        )
        if predictive_failure is not None:
            return TFFixedSGQFScoreResult(
                log_likelihood=None,
                score=None,
                branch_identity=branch_identity,
                diagnostics=_score_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=predictive_failure,
                    same_branch_signature=tf_fixed_sgqf_same_branch_signature(
                        branch_identity=branch_identity,
                        failure=predictive_failure,
                    ),
                ),
                failure=predictive_failure,
            )
        d_predicted_factor = _cholesky_first_derivative(predicted_factor, d_predicted_covariance)

        predictive_points = predicted_mean[tf.newaxis, :] + cloud.points @ tf.transpose(predicted_factor)
        d_predictive_points = d_predicted_mean[:, tf.newaxis, :] + tf.einsum("rd,pnd->prn", cloud.points, d_predicted_factor)

        observation_values = model.observe(predictive_points)
        d_observation_values = _point_map_first_order_values(
            predictive_points,
            d_predictive_points,
            jacobian_fn=derivatives.observation_state_jacobian_fn,
            parameter_derivative_fn=derivatives.d_observation_fn,
            jvp_fn=derivatives.observation_jvp_fn,
        )

        observation_mean = _weighted_mean(observation_values, weights)
        d_observation_mean = tf.einsum("r,prm->pm", weights, d_observation_values)
        centered_observation = observation_values - observation_mean[tf.newaxis, :]
        d_centered_observation = d_observation_values - d_observation_mean[:, tf.newaxis, :]

        innovation_covariance = _symmetrize(
            observation_covariance + _weighted_covariance(centered_observation, weights)
        )
        d_innovation_covariance = _symmetrize(
            d_observation_covariance
            + _covariance_first_derivative(centered_observation, d_centered_observation, weights)
        )
        cross_covariance = tf.transpose(predictive_points - predicted_mean[tf.newaxis, :]) @ (
            centered_observation * weights[:, tf.newaxis]
        )
        d_cross_covariance = (
            tf.einsum(
                "prn,rm->pnm",
                d_predictive_points - d_predicted_mean[:, tf.newaxis, :],
                centered_observation * weights[:, tf.newaxis],
            )
            + tf.einsum(
                "rn,prm->pnm",
                predictive_points - predicted_mean[tf.newaxis, :],
                d_centered_observation * weights[tf.newaxis, :, tf.newaxis],
            )
        )
        innovation = y[time_index] - observation_mean
        d_innovation = -d_observation_mean

        innovation_factor, _innovation_diag, innovation_failure = _cholesky_factor_or_failure(
            innovation_covariance,
            epsilon=branch_config.innovation_epsilon,
            time_index=time_index,
            stage="innovation_covariance",
        )
        if innovation_failure is not None:
            return TFFixedSGQFScoreResult(
                log_likelihood=None,
                score=None,
                branch_identity=branch_identity,
                diagnostics=_score_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=innovation_failure,
                    same_branch_signature=tf_fixed_sgqf_same_branch_signature(
                        branch_identity=branch_identity,
                        failure=innovation_failure,
                    ),
                ),
                failure=innovation_failure,
            )

        innovation_precision = tf.linalg.cholesky_solve(
            innovation_factor,
            tf.eye(observation_dim, dtype=tf.float64),
        )
        innovation_solve = tf.linalg.cholesky_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
        )[:, 0]

        trace_term = tf.einsum("mn,pnm->p", innovation_precision, d_innovation_covariance)
        dv_term = 2.0 * tf.einsum("pm,m->p", d_innovation, innovation_solve)
        quad_term = tf.einsum("m,pmn,n->p", innovation_solve, d_innovation_covariance, innovation_solve)
        score_increment = -0.5 * (trace_term + dv_term - quad_term)

        log_det = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)))
        log_likelihood_increment = -0.5 * (
            tf.cast(observation_dim, tf.float64) * tf.math.log(tf.constant(2.0 * math.pi, dtype=tf.float64))
            + log_det
            + tf.reduce_sum(innovation * innovation_solve)
        )

        gain = cross_covariance @ innovation_precision
        d_gain = (
            tf.einsum("pnm,mk->pnk", d_cross_covariance, innovation_precision)
            - tf.einsum("nm,pmk,kl->pnl", gain, d_innovation_covariance, innovation_precision)
        )
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        d_filtered_mean = d_predicted_mean + tf.einsum("pnm,m->pn", d_gain, innovation) + tf.einsum("nm,pm->pn", gain, d_innovation)
        filtered_covariance = _symmetrize(
            predicted_covariance - gain @ innovation_covariance @ tf.transpose(gain)
        )
        d_filtered_covariance = _symmetrize(
            d_predicted_covariance
            - tf.einsum("pnm,mk,lk->pnl", d_gain, innovation_covariance, gain)
            - tf.einsum("nm,pmk,lk->pnl", gain, d_innovation_covariance, gain)
            - tf.einsum("nm,mk,plk->pnl", gain, innovation_covariance, d_gain)
        )

        _carried_factor, _carried_diag, carried_failure = _cholesky_factor_or_failure(
            filtered_covariance,
            epsilon=branch_config.predictive_epsilon,
            time_index=time_index,
            stage="carried_covariance",
        )
        if carried_failure is not None:
            return TFFixedSGQFScoreResult(
                log_likelihood=None,
                score=None,
                branch_identity=branch_identity,
                diagnostics=_score_diagnostics(
                    branch_identity=branch_identity,
                    branch_config=branch_config,
                    cloud=cloud,
                    accepted_steps=accepted_steps,
                    failure=carried_failure,
                    same_branch_signature=tf_fixed_sgqf_same_branch_signature(
                        branch_identity=branch_identity,
                        failure=carried_failure,
                    ),
                ),
                failure=carried_failure,
            )

        log_likelihood = log_likelihood + log_likelihood_increment
        score = score + score_increment
        mean = filtered_mean
        covariance = filtered_covariance
        d_mean = d_filtered_mean
        d_covariance = d_filtered_covariance
        accepted_steps += 1

    return TFFixedSGQFScoreResult(
        log_likelihood=log_likelihood,
        score=score,
        branch_identity=branch_identity,
        diagnostics=_score_diagnostics(
            branch_identity=branch_identity,
            branch_config=branch_config,
            cloud=cloud,
            accepted_steps=accepted_steps,
            failure=None,
            derivative_method="analytic_first_order_fixed_branch",
            same_branch_signature=tf_fixed_sgqf_same_branch_signature(
                branch_identity=branch_identity,
                failure=None,
            ),
        ),
        filtered_mean=mean,
        filtered_covariance=covariance,
        d_filtered_mean=d_mean,
        d_filtered_covariance=d_covariance,
        failure=None,
    )
