"""Direct-factor, batched square-root unscented Kalman filter.

This module is the admitted factor route.  It carries lower factors through
time, uses QR on residual stacks, and uses sequential rank-one downdates for
the filtered factor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping

import tensorflow as tf

from bayesfilter.linear.lower_rank_downdate_tf import batched_lower_rank_downdate
from bayesfilter.linear.stack_qr_tf import batched_stack_qr_lower
from bayesfilter.nonlinear.srukf_backend_policy import (
    DEFAULT_SRUKF_BACKEND,
    srukf_backend_metadata,
)


TransitionFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
ObservationFn = Callable[[tf.Tensor], tf.Tensor]
TransitionJacobianFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
TransitionParameterDerivativeFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
ObservationJacobianFn = Callable[[tf.Tensor], tf.Tensor]
ObservationParameterDerivativeFn = Callable[[tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class TFFactorSRUKFModel:
    initial_mean: tf.Tensor
    initial_factor: tf.Tensor
    process_factor: tf.Tensor
    observation_factor: tf.Tensor
    transition_fn: TransitionFn
    observation_fn: ObservationFn
    name: str = DEFAULT_SRUKF_BACKEND

    def __post_init__(self) -> None:
        for field in ("initial_mean", "initial_factor", "process_factor", "observation_factor"):
            object.__setattr__(self, field, tf.convert_to_tensor(getattr(self, field), dtype=tf.float64))
        if self.initial_mean.shape.rank != 2:
            raise ValueError("initial_mean must be [B,N]")
        if self.initial_factor.shape.rank != 3 or self.process_factor.shape.rank != 3 or self.observation_factor.shape.rank != 3:
            raise ValueError("factors must be rank three")
        b, n = self.initial_mean.shape.as_list()
        if None in (b, n) or self.initial_factor.shape.as_list() != [b, n, n]:
            raise ValueError("initial_factor shape mismatch")
        q = self.process_factor.shape[1]
        m = self.observation_factor.shape[1]
        if q is None or m is None:
            raise ValueError("factor dimensions must be static")
        if self.process_factor.shape.as_list() != [b, q, q] or self.observation_factor.shape.as_list() != [b, m, m]:
            raise ValueError("process/observation factor shape mismatch")

    @property
    def batch_dim(self) -> int:
        return int(self.initial_mean.shape[0])

    @property
    def state_dim(self) -> int:
        return int(self.initial_mean.shape[1])

    @property
    def process_dim(self) -> int:
        return int(self.process_factor.shape[1])

    @property
    def observation_dim(self) -> int:
        return int(self.observation_factor.shape[1])


@dataclass(frozen=True)
class TFFactorSRUKFDerivatives:
    d_initial_mean: tf.Tensor
    d_initial_factor: tf.Tensor
    d_process_factor: tf.Tensor
    d_observation_factor: tf.Tensor
    transition_state_jacobian_fn: TransitionJacobianFn
    transition_process_jacobian_fn: TransitionJacobianFn
    d_transition_fn: TransitionParameterDerivativeFn
    observation_state_jacobian_fn: ObservationJacobianFn
    d_observation_fn: ObservationParameterDerivativeFn
    name: str = f"{DEFAULT_SRUKF_BACKEND}_derivatives"

    def __post_init__(self) -> None:
        for field in ("d_initial_mean", "d_initial_factor", "d_process_factor", "d_observation_factor"):
            object.__setattr__(self, field, tf.convert_to_tensor(getattr(self, field), dtype=tf.float64))
        if self.d_initial_mean.shape.rank != 3 or self.d_initial_factor.shape.rank != 4:
            raise ValueError("initial derivatives must be [B,P,N] and [B,P,N,N]")
        if self.d_process_factor.shape.rank != 4 or self.d_observation_factor.shape.rank != 4:
            raise ValueError("noise-factor derivatives must be rank four")
        b, p, n = self.d_initial_mean.shape.as_list()
        if None in (b, p, n) or self.d_initial_factor.shape.as_list() != [b, p, n, n]:
            raise ValueError("initial derivative shapes mismatch")
        if self.d_process_factor.shape[0] != b or self.d_observation_factor.shape[0] != b:
            raise ValueError("derivative batch shapes mismatch")

    @property
    def parameter_dim(self) -> int:
        return int(self.d_initial_mean.shape[1])


@dataclass(frozen=True)
class TFFactorSRUKFResult:
    log_likelihood: tf.Tensor
    score: tf.Tensor
    filtered_mean: tf.Tensor
    filtered_factor: tf.Tensor
    d_filtered_mean: tf.Tensor
    d_filtered_factor: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]


def tf_factor_srukf_dz5_rule(dim: int) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dim = int(dim)
    if dim <= 0:
        raise ValueError("dim must be positive")
    eye = tf.eye(dim, dtype=tf.float64)
    scale = tf.sqrt(tf.cast(dim, tf.float64))
    offsets = tf.concat([tf.zeros([1, dim], tf.float64), scale * eye, -scale * eye], axis=0)
    noncentral = tf.fill([2 * dim], tf.constant(1.0 / (2.0 * dim), tf.float64))
    return offsets, tf.concat([tf.zeros([1], tf.float64), noncentral], 0), tf.concat([tf.constant([2.0], tf.float64), noncentral], 0)


def _block_factor(state_factor: tf.Tensor, process_factor: tf.Tensor) -> tf.Tensor:
    b, n, _ = state_factor.shape.as_list()
    q = process_factor.shape[1]
    zeros_nq = tf.zeros([b, n, q], tf.float64)
    zeros_qn = tf.zeros([b, q, n], tf.float64)
    return tf.concat([tf.concat([state_factor, zeros_nq], 2), tf.concat([zeros_qn, process_factor], 2)], 1)


def _block_factor_derivative(d_state: tf.Tensor, d_process: tf.Tensor) -> tf.Tensor:
    b, p, n, _ = d_state.shape.as_list()
    q = d_process.shape[2]
    zeros_nq = tf.zeros([b, p, n, q], tf.float64)
    zeros_qn = tf.zeros([b, p, q, n], tf.float64)
    return tf.concat([tf.concat([d_state, zeros_nq], 3), tf.concat([zeros_qn, d_process], 3)], 2)


def _weighted_mean(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.einsum("r,brd->bd", weights, points)


def _weighted_derivative(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.einsum("r,bprd->bpd", weights, points)


def _stack(points: tf.Tensor, mean: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.transpose((points - mean[:, None, :]) * tf.sqrt(weights)[None, :, None], [0, 2, 1])


def _triangular_solve_batch(factor: tf.Tensor, rhs: tf.Tensor, lower: bool) -> tf.Tensor:
    if rhs.shape.rank == 3:
        return tf.linalg.triangular_solve(factor, rhs, lower=lower)
    b, p, n, k = rhs.shape.as_list()
    solved = tf.linalg.triangular_solve(
        tf.repeat(factor, p, axis=0),
        tf.reshape(rhs, [b * p, n, k]),
        lower=lower,
    )
    return tf.reshape(solved, [b, p, n, k])


def _right_solve(factor: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    """Solve ``X factor = matrix`` for a rank-three matrix."""
    if matrix.shape.rank != 3:
        raise ValueError("_right_solve expects [B,N,M] tensors")
    return tf.linalg.matrix_transpose(
        tf.linalg.triangular_solve(tf.linalg.matrix_transpose(factor), tf.linalg.matrix_transpose(matrix), lower=False)
    )


def _right_solve_batch(factor: tf.Tensor, matrix: tf.Tensor) -> tf.Tensor:
    """Solve ``X factor = matrix`` for derivative matrices ``[B,P,N,M]``."""
    if matrix.shape.rank != 4:
        raise ValueError("_right_solve_batch expects [B,P,N,M] tensors")
    b, p, n, m = matrix.shape.as_list()
    if None in (b, p, n, m) or factor.shape.as_list() != [b, m, m]:
        raise ValueError("factor and derivative matrix shapes are incompatible")
    repeated_factor = tf.repeat(factor, p, axis=0)
    rhs = tf.reshape(matrix, [b * p, n, m])
    solved = tf.linalg.triangular_solve(
        tf.linalg.matrix_transpose(repeated_factor),
        tf.linalg.matrix_transpose(rhs),
        lower=False,
    )
    return tf.reshape(tf.linalg.matrix_transpose(solved), [b, p, n, m])


def _one_step(
    observation: tf.Tensor,
    mean: tf.Tensor,
    state_factor: tf.Tensor,
    d_mean: tf.Tensor,
    d_state_factor: tf.Tensor,
    model: TFFactorSRUKFModel,
    derivatives: TFFactorSRUKFDerivatives,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, dict[str, tf.Tensor]]:
    b, n, q, m, p = model.batch_dim, model.state_dim, model.process_dim, model.observation_dim, derivatives.parameter_dim
    offsets, mean_weights, covariance_weights = tf_factor_srukf_dz5_rule(n + q)
    augmented_mean = tf.concat([mean, tf.zeros([b, q], tf.float64)], 1)
    augmented_factor = _block_factor(state_factor, model.process_factor)
    d_augmented_mean = tf.concat([d_mean, tf.zeros([b, p, q], tf.float64)], 2)
    d_augmented_factor = _block_factor_derivative(d_state_factor, derivatives.d_process_factor)
    # The carried factor is lower triangular L with P = L L^T. Sigma-point
    # row offsets therefore use the rows of L, i.e. offset @ L^T.
    points = augmented_mean[:, None, :] + tf.einsum(
        "rd,bkd->brk", offsets, augmented_factor
    )
    d_points = d_augmented_mean[:, :, None, :] + tf.einsum(
        "rd,bpkd->bprk", offsets, d_augmented_factor
    )
    previous_points, process_points = points[:, :, :n], points[:, :, n:]
    d_previous_points, d_process_points = d_points[:, :, :, :n], d_points[:, :, :, n:]
    predicted_points = tf.convert_to_tensor(model.transition_fn(previous_points, process_points), tf.float64)
    transition_state_j = tf.convert_to_tensor(derivatives.transition_state_jacobian_fn(previous_points, process_points), tf.float64)
    transition_process_j = tf.convert_to_tensor(derivatives.transition_process_jacobian_fn(previous_points, process_points), tf.float64)
    d_transition_direct = tf.convert_to_tensor(derivatives.d_transition_fn(previous_points, process_points), tf.float64)
    d_predicted_points = tf.einsum("brij,bprj->bpri", transition_state_j, d_previous_points) + tf.einsum("brij,bprj->bpri", transition_process_j, d_process_points) + d_transition_direct
    predicted_mean = _weighted_mean(predicted_points, mean_weights)
    d_predicted_mean = _weighted_derivative(d_predicted_points, mean_weights)
    state_stack = _stack(predicted_points, predicted_mean, covariance_weights)
    d_state_stack = tf.transpose((d_predicted_points - d_predicted_mean[:, :, None, :]) * tf.sqrt(covariance_weights)[None, None, :, None], [0, 1, 3, 2])
    predicted_factor, d_predicted_factor, qr_diag = batched_stack_qr_lower(state_stack, d_state_stack)

    observation_points = tf.convert_to_tensor(model.observation_fn(predicted_points), tf.float64)
    observation_state_j = tf.convert_to_tensor(derivatives.observation_state_jacobian_fn(predicted_points), tf.float64)
    d_observation_direct = tf.convert_to_tensor(derivatives.d_observation_fn(predicted_points), tf.float64)
    d_observation_points = tf.einsum("brij,bprj->bpri", observation_state_j, d_predicted_points) + d_observation_direct
    predicted_observation = _weighted_mean(observation_points, mean_weights)
    d_predicted_observation = _weighted_derivative(d_observation_points, mean_weights)
    y_stack = _stack(observation_points, predicted_observation, covariance_weights)
    dy_stack = tf.transpose((d_observation_points - d_predicted_observation[:, :, None, :]) * tf.sqrt(covariance_weights)[None, None, :, None], [0, 1, 3, 2])
    observation_stack = tf.concat([y_stack, model.observation_factor], axis=2)
    d_observation_stack = tf.concat([dy_stack, derivatives.d_observation_factor], axis=3)
    innovation_factor, d_innovation_factor, innovation_qr_diag = batched_stack_qr_lower(observation_stack, d_observation_stack)
    centered_x = predicted_points - predicted_mean[:, None, :]
    centered_y = observation_points - predicted_observation[:, None, :]
    d_centered_x = d_predicted_points - d_predicted_mean[:, :, None, :]
    d_centered_y = d_observation_points - d_predicted_observation[:, :, None, :]
    cross = tf.einsum("r,brn,brm->bnm", covariance_weights, centered_x, centered_y)
    d_cross = tf.einsum("r,bprn,brm->bpnm", covariance_weights, d_centered_x, centered_y) + tf.einsum("r,brn,bprm->bpnm", covariance_weights, centered_x, d_centered_y)
    innovation = observation - predicted_observation
    z = tf.linalg.triangular_solve(innovation_factor, innovation[:, :, None], lower=True)[:, :, 0]
    dz_rhs = -d_predicted_observation - tf.einsum("bpij,bj->bpi", d_innovation_factor, z)
    dz = _triangular_solve_batch(innovation_factor, dz_rhs[..., None], lower=True)[..., 0]
    log_likelihood = -0.5 * (m * math.log(2.0 * math.pi) + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(innovation_factor)), 1) + tf.reduce_sum(z * z, 1))
    score = -tf.reduce_sum(d_innovation_factor[:, :, :, :] * 0.0, axis=[2, 3])
    score = -0.5 * (2.0 * tf.reduce_sum(tf.linalg.diag_part(d_innovation_factor) / tf.linalg.diag_part(innovation_factor)[:, None, :], axis=2) + 2.0 * tf.reduce_sum(z[:, None, :] * dz, axis=2))
    u_t = _triangular_solve_batch(innovation_factor, tf.linalg.matrix_transpose(cross), lower=True)
    u = tf.linalg.matrix_transpose(u_t)
    gain = _right_solve(innovation_factor, u)
    d_u_rhs = tf.linalg.matrix_transpose(d_cross) - tf.einsum("bpij,bjn->bpin", d_innovation_factor, tf.linalg.matrix_transpose(u))
    d_u = tf.linalg.matrix_transpose(_triangular_solve_batch(innovation_factor, d_u_rhs, lower=True))
    d_gain = _right_solve_batch(innovation_factor, d_u)
    d_gain = d_gain - tf.stack(
        [
            _right_solve(innovation_factor, tf.einsum("bnm,bml->bnl", gain, d_innovation_factor[:, i]))
            for i in range(p)
        ],
        axis=1,
    )
    filtered_mean = predicted_mean + tf.einsum("bnm,bm->bn", gain, innovation)
    d_filtered_mean = d_predicted_mean + tf.einsum("bpnm,bm->bpn", d_gain, innovation) - tf.einsum("bnm,bpm->bpn", gain, d_predicted_observation)
    filtered_factor, d_filtered_factor, down_diag = batched_lower_rank_downdate(predicted_factor, u, d_predicted_factor, d_u)
    diagnostics = dict(qr_diag)
    diagnostics.update({"innovation_" + k: v for k, v in innovation_qr_diag.items()})
    diagnostics.update({
        "minimum_downdate_margin": down_diag["minimum_downdate_margin"],
        "relative_downdate_margin": down_diag["relative_downdate_margin"],
        "backend": tf.constant(DEFAULT_SRUKF_BACKEND),
        "backend_status": tf.constant("default"),
        "backend_contract": tf.constant("TFFactorSRUKFModel"),
        "factorization": tf.constant("direct_qr_and_lower_rank_downdate"),
        "rule": tf.constant("dz5_unscented_alpha1_beta2_kappa0"),
    })
    return log_likelihood, score, filtered_mean, filtered_factor, d_filtered_mean, d_filtered_factor, diagnostics


def tf_factor_srukf_value_and_score(observations: tf.Tensor, model: TFFactorSRUKFModel, derivatives: TFFactorSRUKFDerivatives, *, jit_compile: bool = True) -> TFFactorSRUKFResult:
    observations = tf.convert_to_tensor(observations, dtype=tf.float64)
    if observations.shape.rank != 3 or observations.shape[0] != model.batch_dim or observations.shape[2] != model.observation_dim:
        raise ValueError("observations must be [B,T,M]")
    t_count = observations.shape[1]
    if t_count is None:
        raise ValueError("time dimension must be static")

    def run(obs: tf.Tensor):
        tf.debugging.assert_all_finite(obs, "observations contain NaN or Inf")
        for factor_name, factor_value in (
            ("initial_factor", model.initial_factor),
            ("process_factor", model.process_factor),
            ("observation_factor", model.observation_factor),
        ):
            tf.debugging.assert_all_finite(factor_value, f"{factor_name} contains NaN or Inf")
            tf.debugging.assert_greater(
                tf.linalg.diag_part(factor_value),
                tf.zeros_like(tf.linalg.diag_part(factor_value)),
                message=f"{factor_name} diagonal must be positive",
            )
        p = derivatives.parameter_dim
        mean = model.initial_mean
        factor = model.initial_factor
        d_mean = derivatives.d_initial_mean
        d_factor = derivatives.d_initial_factor
        value = tf.zeros([model.batch_dim], tf.float64)
        score = tf.zeros([model.batch_dim, p], tf.float64)
        inf = tf.fill([model.batch_dim], tf.constant(float("inf"), tf.float64))
        zero = tf.zeros([model.batch_dim], tf.float64)

        def body(t, mean, factor, d_mean, d_factor, value, score, min_qr_pivot, rel_qr_pivot, min_down_margin, rel_down_margin, max_factor_residual, max_derivative_residual):
            inc, inc_score, new_mean, new_factor, new_d_mean, new_d_factor, diagnostics = _one_step(
                obs[:, t, :], mean, factor, d_mean, d_factor, model, derivatives
            )
            return (
                t + 1,
                new_mean,
                new_factor,
                new_d_mean,
                new_d_factor,
                value + inc,
                score + inc_score,
                tf.minimum(min_qr_pivot, tf.minimum(diagnostics["minimum_qr_pivot"], diagnostics["innovation_minimum_qr_pivot"])),
                tf.minimum(rel_qr_pivot, tf.minimum(diagnostics["relative_qr_pivot"], diagnostics["innovation_relative_qr_pivot"])),
                tf.minimum(min_down_margin, diagnostics["minimum_downdate_margin"]),
                tf.minimum(rel_down_margin, diagnostics["relative_downdate_margin"]),
                tf.maximum(max_factor_residual, tf.maximum(diagnostics["factor_reconstruction_residual"], diagnostics["innovation_factor_reconstruction_residual"])),
                tf.maximum(
                    max_derivative_residual,
                    tf.maximum(
                        tf.reduce_max(diagnostics["factor_derivative_reconstruction_residual"], axis=1),
                        tf.reduce_max(diagnostics["innovation_factor_derivative_reconstruction_residual"], axis=1),
                    ),
                ),
            )

        _, mean, factor, d_mean, d_factor, value, score, min_qr_pivot, rel_qr_pivot, min_down_margin, rel_down_margin, max_factor_residual, max_derivative_residual = tf.while_loop(
            lambda t, *_: t < t_count,
            body,
            (tf.constant(0), mean, factor, d_mean, d_factor, value, score, inf, inf, inf, inf, zero, zero),
            parallel_iterations=1,
        )
        return value, score, mean, factor, d_mean, d_factor, min_qr_pivot, rel_qr_pivot, min_down_margin, rel_down_margin, max_factor_residual, max_derivative_residual

    if jit_compile:
        value, score, mean, factor, d_mean, d_factor, min_qr_pivot, rel_qr_pivot, min_down_margin, rel_down_margin, max_factor_residual, max_derivative_residual = tf.function(run, jit_compile=True)(observations)
    else:
        value, score, mean, factor, d_mean, d_factor, min_qr_pivot, rel_qr_pivot, min_down_margin, rel_down_margin, max_factor_residual, max_derivative_residual = run(observations)
    return TFFactorSRUKFResult(
        value,
        score,
        mean,
        factor,
        d_mean,
        d_factor,
        {
            **{
                key: tf.constant(value)
                for key, value in srukf_backend_metadata(DEFAULT_SRUKF_BACKEND).items()
            },
            "jit_compile": tf.constant(bool(jit_compile)),
            "minimum_qr_pivot": min_qr_pivot,
            "relative_qr_pivot": rel_qr_pivot,
            "minimum_downdate_margin": min_down_margin,
            "relative_downdate_margin": rel_down_margin,
            "maximum_factor_reconstruction_residual": max_factor_residual,
            "maximum_derivative_reconstruction_residual": max_derivative_residual,
        },
    )


def tf_default_srukf_value_and_score(
    observations: tf.Tensor,
    model: TFFactorSRUKFModel,
    derivatives: TFFactorSRUKFDerivatives,
    *,
    jit_compile: bool = True,
) -> TFFactorSRUKFResult:
    """Evaluate the repository-default direct-factor SR-UKF route.

    This entry point deliberately accepts only the factor contract.  Legacy
    covariance models and specialized observation laws must use an explicit,
    contract-preserving adapter; they are never silently routed here.
    """

    if not isinstance(model, TFFactorSRUKFModel):
        raise TypeError(
            "the default SR-UKF route requires TFFactorSRUKFModel; "
            "legacy covariance models require an explicit compatibility adapter"
        )
    if not isinstance(derivatives, TFFactorSRUKFDerivatives):
        raise TypeError(
            "the default SR-UKF route requires TFFactorSRUKFDerivatives"
        )
    return tf_factor_srukf_value_and_score(
        observations,
        model,
        derivatives,
        jit_compile=jit_compile,
    )


__all__ = [
    "TFFactorSRUKFModel",
    "TFFactorSRUKFDerivatives",
    "TFFactorSRUKFResult",
    "tf_factor_srukf_dz5_rule",
    "tf_factor_srukf_value_and_score",
    "tf_default_srukf_value_and_score",
]
