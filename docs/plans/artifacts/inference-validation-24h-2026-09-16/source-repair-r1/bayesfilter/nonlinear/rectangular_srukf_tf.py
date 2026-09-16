"""Rectangular square-root UKF for singular covariance supports.

The discovery route applies SVD only to direct residual stacks and remains
value-only.  The score route uses repository-owned, fixed-pivot QR charts and
is differentiable only while ranks, charts, signs, and supports remain fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import hashlib
import json
from typing import Callable, Mapping

import tensorflow as tf

from bayesfilter.linear.rectangular_factor_tf import (
    batched_direct_stack_svd_factor,
    batched_direct_support_conditional,
    batched_fixed_pivot_rectangular_qr,
    batched_fixed_support_qr_update,
)
from bayesfilter.nonlinear.factor_srukf_tf import tf_factor_srukf_dz5_rule


TransitionFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
ObservationFn = Callable[[tf.Tensor], tf.Tensor]
TransitionJacobianFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
TransitionParameterDerivativeFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
ObservationJacobianFn = Callable[[tf.Tensor], tf.Tensor]
ObservationParameterDerivativeFn = Callable[[tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class TFRectangularSRUKFModel:
    initial_mean: tf.Tensor
    initial_factor: tf.Tensor
    process_factor: tf.Tensor
    observation_factor: tf.Tensor
    transition_fn: TransitionFn
    observation_fn: ObservationFn

    def __post_init__(self) -> None:
        for field in (
            "initial_mean",
            "initial_factor",
            "process_factor",
            "observation_factor",
        ):
            object.__setattr__(self, field, tf.convert_to_tensor(getattr(self, field), tf.float64))
        if self.initial_mean.shape.rank != 2:
            raise ValueError("initial_mean must be [B,N]")
        for field in ("initial_factor", "process_factor", "observation_factor"):
            if getattr(self, field).shape.rank != 3:
                raise ValueError(f"{field} must be [B,dimension,rank]")
        b, n = self.initial_mean.shape.as_list()
        if None in (b, n) or self.initial_factor.shape[0] != b or self.initial_factor.shape[1] != n:
            raise ValueError("initial factor shape mismatch")
        if self.process_factor.shape[0] != b or self.observation_factor.shape[0] != b:
            raise ValueError("noise factor batch shapes mismatch")
        if self.process_factor.shape[1] is None or self.observation_factor.shape[1] is None:
            raise ValueError("factor dimensions must be static")

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
class TFRectangularSRUKFResult:
    log_likelihood: tf.Tensor
    filtered_mean: tf.Tensor
    filtered_factor: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]


@dataclass(frozen=True)
class TFRectangularSRUKFDerivatives:
    d_initial_mean: tf.Tensor
    d_initial_factor: tf.Tensor
    d_process_factor: tf.Tensor
    d_observation_factor: tf.Tensor
    transition_state_jacobian_fn: TransitionJacobianFn
    transition_process_jacobian_fn: TransitionJacobianFn
    d_transition_fn: TransitionParameterDerivativeFn
    observation_state_jacobian_fn: ObservationJacobianFn
    d_observation_fn: ObservationParameterDerivativeFn

    def __post_init__(self) -> None:
        for field in ("d_initial_mean", "d_initial_factor", "d_process_factor", "d_observation_factor"):
            object.__setattr__(self, field, tf.convert_to_tensor(getattr(self, field), tf.float64))
        if self.d_initial_mean.shape.rank != 3 or self.d_initial_factor.shape.rank != 4:
            raise ValueError("initial derivatives must be [B,P,N] and [B,P,N,R]")
        if self.d_process_factor.shape.rank != 4 or self.d_observation_factor.shape.rank != 4:
            raise ValueError("noise factor derivatives must have rank four")

    @property
    def parameter_dim(self) -> int:
        return int(self.d_initial_mean.shape[1])


@dataclass(frozen=True)
class TFRectangularSRUKFFixedBranch:
    predicted_rank: int
    predicted_permutation: tuple[int, ...]
    innovation_rank: int
    innovation_permutation: tuple[int, ...]
    filtered_rank: int
    filtered_permutation: tuple[int, ...]
    pivot_tolerance: float = 1.0e-12
    chart_tolerance: float = 1.0e-10
    support_tolerance: float = 1.0e-10
    identity: str = field(init=False)

    def __post_init__(self) -> None:
        for rank in (self.predicted_rank, self.innovation_rank, self.filtered_rank):
            if int(rank) < 0:
                raise ValueError("fixed branch ranks must be nonnegative")
        if min(self.pivot_tolerance, self.chart_tolerance, self.support_tolerance) < 0.0:
            raise ValueError("fixed branch tolerances must be nonnegative")
        for permutation in (
            self.predicted_permutation,
            self.innovation_permutation,
            self.filtered_permutation,
        ):
            if sorted(permutation) != list(range(len(permutation))):
                raise ValueError("fixed branch permutations must be bijections")
        payload = {
            "schema": "fixed_rank_row_pivot_qr_v1",
            "predicted_rank": self.predicted_rank,
            "predicted_permutation": self.predicted_permutation,
            "innovation_rank": self.innovation_rank,
            "innovation_permutation": self.innovation_permutation,
            "filtered_rank": self.filtered_rank,
            "filtered_permutation": self.filtered_permutation,
            "pivot_tolerance": self.pivot_tolerance,
            "chart_tolerance": self.chart_tolerance,
            "support_tolerance": self.support_tolerance,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        object.__setattr__(self, "identity", f"fixed_rank_row_pivot_qr_v1:{digest}")


@dataclass(frozen=True)
class TFRectangularSRUKFScoreResult:
    log_likelihood: tf.Tensor
    score: tf.Tensor
    filtered_mean: tf.Tensor
    filtered_factor: tf.Tensor
    d_filtered_mean: tf.Tensor
    d_filtered_factor: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]


def _pad_factor(factor: tf.Tensor, width: int) -> tf.Tensor:
    current = factor.shape[-1]
    if current is None or current > width:
        raise ValueError("factor width exceeds requested padded width")
    if current == width:
        return factor
    return tf.concat([factor, tf.zeros([factor.shape[0], factor.shape[1], width - current], tf.float64)], axis=2)


def _block_factor(state_factor: tf.Tensor, process_factor: tf.Tensor) -> tf.Tensor:
    b, n, rs = state_factor.shape.as_list()
    q, rq = process_factor.shape[1], process_factor.shape[2]
    return tf.concat(
        [
            tf.concat([state_factor, tf.zeros([b, n, rq], tf.float64)], axis=2),
            tf.concat([tf.zeros([b, q, rs], tf.float64), process_factor], axis=2),
        ],
        axis=1,
    )


def _block_factor_derivative(d_state: tf.Tensor, d_process: tf.Tensor) -> tf.Tensor:
    b, p, n, rs = d_state.shape.as_list()
    q, rq = d_process.shape[2], d_process.shape[3]
    return tf.concat(
        [
            tf.concat([d_state, tf.zeros([b, p, n, rq], tf.float64)], axis=3),
            tf.concat([tf.zeros([b, p, q, rs], tf.float64), d_process], axis=3),
        ],
        axis=2,
    )


def _stack(points: tf.Tensor, mean: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.transpose(
        (points - mean[:, None, :]) * tf.sqrt(weights)[None, :, None],
        [0, 2, 1],
    )


def _derivative_stack(points: tf.Tensor, mean: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.transpose(
        (points - mean[:, :, None, :]) * tf.sqrt(weights)[None, None, :, None],
        [0, 1, 3, 2],
    )


def tf_rectangular_srukf_value(
    observations: tf.Tensor,
    model: TFRectangularSRUKFModel,
    *,
    relative_cutoff: float = 1.0e-12,
    support_tolerance: float = 1.0e-10,
    jit_compile: bool = True,
) -> TFRectangularSRUKFResult:
    """Run the rectangular value-only SR-UKF recursion."""

    observations = tf.convert_to_tensor(observations, tf.float64)
    if observations.shape.rank != 3 or observations.shape[0] != model.batch_dim or observations.shape[2] != model.observation_dim:
        raise ValueError("observations must be [B,T,M]")
    t_count = observations.shape[1]
    if t_count is None:
        raise ValueError("time dimension must be static")
    if relative_cutoff < 0.0 or support_tolerance < 0.0:
        raise ValueError("cutoffs and tolerances must be nonnegative")
    b, n, q, m = model.batch_dim, model.state_dim, model.process_dim, model.observation_dim

    def run(obs: tf.Tensor):
        tf.debugging.assert_all_finite(obs, "observations contain NaN or Inf")
        for name, value in (
            ("initial_factor", model.initial_factor),
            ("process_factor", model.process_factor),
            ("observation_factor", model.observation_factor),
        ):
            tf.debugging.assert_all_finite(value, f"{name} contains NaN or Inf")
        mean = model.initial_mean
        state_factor = _pad_factor(model.initial_factor, n)
        value = tf.zeros([b], tf.float64)
        on_support = tf.ones([b], tf.bool)
        min_rank = tf.fill([b], tf.constant(2**31 - 1, tf.int32))
        max_support_residual = tf.zeros([b], tf.float64)

        def body(t, mean, state_factor, value, on_support, min_rank, max_support_residual):
            process_factor = _pad_factor(model.process_factor, q)
            augmented_mean = tf.concat([mean, tf.zeros([b, q], tf.float64)], axis=1)
            augmented_factor = _block_factor(state_factor, process_factor)
            latent_dim = n + q
            offsets, mean_weights, covariance_weights = tf_factor_srukf_dz5_rule(latent_dim)
            points = augmented_mean[:, None, :] + tf.einsum("rd,bkd->brk", offsets, augmented_factor)
            state_points = points[:, :, :n]
            process_points = points[:, :, n:]
            predicted_points = tf.convert_to_tensor(model.transition_fn(state_points, process_points), tf.float64)
            predicted_mean = tf.einsum("r,brd->bd", mean_weights, predicted_points)
            state_stack = _stack(predicted_points, predicted_mean, covariance_weights)
            predicted_factor, _, _, state_diag = batched_direct_stack_svd_factor(state_stack, relative_cutoff)
            predicted_factor = _pad_factor(predicted_factor, n)

            observation_points = tf.convert_to_tensor(model.observation_fn(predicted_points), tf.float64)
            predicted_observation = tf.einsum("r,brd->bd", mean_weights, observation_points)
            observation_stack = tf.concat(
                [_stack(observation_points, predicted_observation, covariance_weights), model.observation_factor],
                axis=2,
            )
            state_joint_stack = tf.concat(
                [state_stack, tf.zeros([b, n, model.observation_factor.shape[2]], tf.float64)], axis=2
            )
            innovation = obs[:, t, :] - predicted_observation
            inc, gain, filtered_factor, rank, support_diag = batched_direct_support_conditional(
                observation_stack,
                state_joint_stack,
                innovation,
                relative_cutoff,
                support_tolerance,
            )
            filtered_factor = _pad_factor(filtered_factor, n)
            filtered_mean = predicted_mean + tf.einsum("bnm,bm->bn", gain, innovation)
            return (
                t + 1,
                filtered_mean,
                filtered_factor,
                value + inc,
                tf.logical_and(on_support, support_diag["on_support"]),
                tf.minimum(min_rank, tf.minimum(state_diag["rank"], rank)),
                tf.maximum(max_support_residual, support_diag["support_residual"]),
            )

        _, mean, state_factor, value, on_support, min_rank, max_support_residual = tf.while_loop(
            lambda t, *_: t < t_count,
            body,
            (tf.constant(0), mean, state_factor, value, on_support, min_rank, max_support_residual),
            parallel_iterations=1,
        )
        return value, mean, state_factor, on_support, min_rank, max_support_residual

    if jit_compile:
        run = tf.function(run, jit_compile=True)
    value, mean, factor, on_support, min_rank, max_support_residual = run(observations)
    return TFRectangularSRUKFResult(
        value,
        mean,
        factor,
        {
            "value_only": tf.constant(True),
            "rank_branch_status": tf.constant("value_only_rank_discovery"),
            "on_support": on_support,
            "minimum_observation_rank": min_rank,
            "maximum_support_residual": max_support_residual,
            "likelihood_measure": tf.constant("affine_support_gaussian"),
        },
    )


def tf_rectangular_srukf_value_and_score(
    observations: tf.Tensor,
    model: TFRectangularSRUKFModel,
    derivatives: TFRectangularSRUKFDerivatives,
    branch: TFRectangularSRUKFFixedBranch,
    *,
    jit_compile: bool = True,
) -> TFRectangularSRUKFScoreResult:
    """Run a fixed-rank, fixed-chart rectangular SR-UKF score recursion."""
    observations = tf.convert_to_tensor(observations, tf.float64)
    if observations.shape.rank != 3 or observations.shape[0] != model.batch_dim or observations.shape[2] != model.observation_dim:
        raise ValueError("observations must be [B,T,M]")
    t_count = observations.shape[1]
    if t_count is None:
        raise ValueError("time dimension must be static")
    b, n, q = model.batch_dim, model.state_dim, model.process_dim
    p = derivatives.parameter_dim
    rs, rq = model.initial_factor.shape[2], model.process_factor.shape[2]
    if branch.predicted_rank <= 0 or branch.innovation_rank <= 0 or branch.filtered_rank <= 0:
        raise ValueError("score route requires positive fixed ranks")
    if len(branch.predicted_permutation) != n or len(branch.innovation_permutation) != model.observation_dim or len(branch.filtered_permutation) != n:
        raise ValueError("branch permutations must match state/observation dimensions")
    if model.initial_factor.shape.as_list() != [b, n, branch.filtered_rank]:
        raise ValueError("initial factor width must equal the fixed filtered rank")
    if derivatives.d_initial_factor.shape.as_list() != [b, p, n, branch.filtered_rank]:
        raise ValueError("initial factor derivative width must equal the fixed filtered rank")

    predicted_permutation = tf.constant(branch.predicted_permutation, tf.int32)
    innovation_permutation = tf.constant(branch.innovation_permutation, tf.int32)
    filtered_permutation = tf.constant(branch.filtered_permutation, tf.int32)

    def run(obs: tf.Tensor):
        mean = model.initial_mean
        factor = model.initial_factor
        d_mean = derivatives.d_initial_mean
        d_factor = derivatives.d_initial_factor
        value = tf.zeros([b], tf.float64)
        score = tf.zeros([b, p], tf.float64)
        score_valid = tf.ones([b], tf.bool)
        minimum_pivot = tf.fill([b], tf.constant(float("inf"), tf.float64))
        maximum_chart_residual = tf.zeros([b], tf.float64)
        maximum_support_residual = tf.zeros([b], tf.float64)

        def body(t, mean, factor, d_mean, d_factor, value, score, score_valid, minimum_pivot, maximum_chart_residual, maximum_support_residual):
            augmented_mean = tf.concat([mean, tf.zeros([b, q], tf.float64)], axis=1)
            augmented_factor = _block_factor(factor, model.process_factor)
            d_augmented_mean = tf.concat([d_mean, tf.zeros([b, p, q], tf.float64)], axis=2)
            d_augmented_factor = _block_factor_derivative(d_factor, derivatives.d_process_factor)
            latent_rank = branch.filtered_rank + rq
            offsets, mean_weights, covariance_weights = tf_factor_srukf_dz5_rule(latent_rank)
            points = augmented_mean[:, None, :] + tf.einsum("rd,bkd->brk", offsets, augmented_factor)
            d_points = d_augmented_mean[:, :, None, :] + tf.einsum("rd,bpkd->bprk", offsets, d_augmented_factor)
            previous_points, process_points = points[:, :, :n], points[:, :, n:]
            d_previous_points, d_process_points = d_points[:, :, :, :n], d_points[:, :, :, n:]
            predicted_points = tf.convert_to_tensor(model.transition_fn(previous_points, process_points), tf.float64)
            d_predicted_points = tf.einsum(
                "brij,bprj->bpri",
                tf.convert_to_tensor(derivatives.transition_state_jacobian_fn(previous_points, process_points), tf.float64),
                d_previous_points,
            ) + tf.einsum(
                "brij,bprj->bpri",
                tf.convert_to_tensor(derivatives.transition_process_jacobian_fn(previous_points, process_points), tf.float64),
                d_process_points,
            ) + tf.convert_to_tensor(derivatives.d_transition_fn(previous_points, process_points), tf.float64)
            predicted_mean = tf.einsum("r,brd->bd", mean_weights, predicted_points)
            d_predicted_mean = tf.einsum("r,bprd->bpd", mean_weights, d_predicted_points)
            state_stack = _stack(predicted_points, predicted_mean, covariance_weights)
            d_state_stack = _derivative_stack(d_predicted_points, d_predicted_mean, covariance_weights)
            predicted_factor, d_predicted_factor, state_diag = batched_fixed_pivot_rectangular_qr(
                state_stack,
                predicted_permutation,
                branch.predicted_rank,
                d_state_stack,
                residual_tolerance=branch.chart_tolerance,
                pivot_tolerance=branch.pivot_tolerance,
            )

            observation_points = tf.convert_to_tensor(model.observation_fn(predicted_points), tf.float64)
            d_observation_points = tf.einsum(
                "brij,bprj->bpri",
                tf.convert_to_tensor(derivatives.observation_state_jacobian_fn(predicted_points), tf.float64),
                d_predicted_points,
            ) + tf.convert_to_tensor(derivatives.d_observation_fn(predicted_points), tf.float64)
            predicted_observation = tf.einsum("r,brd->bd", mean_weights, observation_points)
            d_predicted_observation = tf.einsum("r,bprd->bpd", mean_weights, d_observation_points)
            y_stack = _stack(observation_points, predicted_observation, covariance_weights)
            dy_stack = _derivative_stack(d_observation_points, d_predicted_observation, covariance_weights)
            observation_stack = tf.concat([y_stack, model.observation_factor], axis=2)
            d_observation_stack = tf.concat([dy_stack, derivatives.d_observation_factor], axis=3)
            state_joint_stack = tf.concat([state_stack, tf.zeros([b, n, model.observation_factor.shape[2]], tf.float64)], axis=2)
            d_state_joint_stack = tf.concat([d_state_stack, tf.zeros([b, p, n, model.observation_factor.shape[2]], tf.float64)], axis=3)
            innovation = obs[:, t, :] - predicted_observation
            d_innovation = -d_predicted_observation
            inc, inc_score, increment, d_increment, new_factor, new_d_factor, update_diag = batched_fixed_support_qr_update(
                observation_stack,
                state_joint_stack,
                innovation,
                innovation_permutation,
                branch.innovation_rank,
                filtered_permutation,
                branch.filtered_rank,
                d_observation_stack,
                d_state_joint_stack,
                d_innovation,
                chart_tolerance=branch.chart_tolerance,
                pivot_tolerance=branch.pivot_tolerance,
                support_tolerance=branch.support_tolerance,
            )
            new_mean = predicted_mean + increment
            new_d_mean = d_predicted_mean + d_increment
            valid = tf.logical_and(state_diag["chart_valid"], update_diag["chart_valid"])
            valid = tf.logical_and(valid, update_diag["conditional_chart_valid"])
            valid = tf.logical_and(valid, update_diag["on_support"])
            return (
                t + 1, new_mean, new_factor, new_d_mean, new_d_factor,
                value + inc, score + inc_score, tf.logical_and(score_valid, valid),
                tf.minimum(minimum_pivot, tf.minimum(state_diag["minimum_chart_pivot"], tf.minimum(update_diag["minimum_chart_pivot"], update_diag["conditional_minimum_chart_pivot"]))),
                tf.maximum(maximum_chart_residual, tf.maximum(state_diag["chart_residual"], tf.maximum(update_diag["chart_residual"], update_diag["conditional_chart_residual"]))),
                tf.maximum(maximum_support_residual, update_diag["support_residual"]),
            )

        result = tf.while_loop(
            lambda t, *_: t < t_count,
            body,
            (tf.constant(0), mean, factor, d_mean, d_factor, value, score, score_valid, minimum_pivot, maximum_chart_residual, maximum_support_residual),
            parallel_iterations=1,
        )
        return result[1:]

    if jit_compile:
        run = tf.function(run, jit_compile=True)
    mean, factor, d_mean, d_factor, value, score, score_valid, minimum_pivot, maximum_chart_residual, maximum_support_residual = run(observations)
    score = tf.where(score_valid[:, None], score, tf.fill(tf.shape(score), tf.constant(float("nan"), tf.float64)))
    return TFRectangularSRUKFScoreResult(
        value, score, mean, factor, d_mean, d_factor,
        {
            "value_only": tf.constant(False),
            "score_valid": score_valid,
            "branch_status": tf.where(score_valid, tf.constant("fixed_branch_valid"), tf.constant("fixed_branch_invalid")),
            "branch_identity": tf.constant(branch.identity),
            "factorization": tf.constant("direct_fixed_pivot_rectangular_qr"),
            "likelihood_measure": tf.constant("affine_support_gaussian_fixed_qr"),
            "minimum_chart_pivot": minimum_pivot,
            "maximum_chart_residual": maximum_chart_residual,
            "maximum_support_residual": maximum_support_residual,
            "jit_compile": tf.constant(bool(jit_compile)),
        },
    )


__all__ = [
    "TFRectangularSRUKFModel",
    "TFRectangularSRUKFDerivatives",
    "TFRectangularSRUKFFixedBranch",
    "TFRectangularSRUKFResult",
    "TFRectangularSRUKFScoreResult",
    "tf_rectangular_srukf_value",
    "tf_rectangular_srukf_value_and_score",
]
