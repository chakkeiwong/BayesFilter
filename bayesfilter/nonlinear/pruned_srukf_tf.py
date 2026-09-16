"""Batched second-order pruning with a direct, rank-tolerant QR filter.

The joint state is (first_order, second_order). Quadratic terms use only
first-order states and innovations. Scores differentiate the declared
Gaussian/unscented likelihood on a fixed structural rank branch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping
import math

import tensorflow as tf

from bayesfilter.linear.stack_qr_tf import batched_semidefinite_stack_qr_lower
from bayesfilter.nonlinear.factor_srukf_tf import tf_factor_srukf_dz5_rule


PRUNED_SRUKF_BACKEND = "pruned_direct_factor_srukf"


@dataclass(frozen=True)
class TFPrunedSRUKFModel:
    """Coefficient tensors have a leading, statically known batch dimension.

    For N economic states and Q innovations, transition_hessian is
    [B,N,N+Q,N+Q], observation_hessian is [B,M,N,N]. Hessians use the
    ordinary derivative convention: the recursion applies the factor 1/2.
    initial_factor is [B,2N,R], with R <= 2N, and may be singular.
    """

    initial_mean: tf.Tensor
    initial_factor: tf.Tensor
    transition_matrix: tf.Tensor
    innovation_loading: tf.Tensor
    transition_hessian: tf.Tensor
    second_order_constant: tf.Tensor
    observation_constant: tf.Tensor
    observation_matrix: tf.Tensor
    observation_hessian: tf.Tensor
    innovation_factor: tf.Tensor
    observation_factor: tf.Tensor

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, tf.convert_to_tensor(getattr(self, name), tf.float64))
        if self.transition_matrix.shape.rank != 3:
            raise ValueError("transition_matrix must be [B,N,N]")
        batch, dimension, width = self.transition_matrix.shape.as_list()
        if None in (batch, dimension, width) or min(batch, dimension) <= 0 or width != dimension:
            raise ValueError("transition_matrix must have static positive shape [B,N,N]")
        if self.innovation_loading.shape.rank != 3 or self.observation_matrix.shape.rank != 3:
            raise ValueError("innovation_loading and observation_matrix must have rank three")
        innovations = self.innovation_loading.shape[-1]
        observed = self.observation_matrix.shape[1]
        if innovations is None or observed is None or min(innovations, observed) <= 0:
            raise ValueError("innovation and observation dimensions must be static and positive")
        expected = {
            "initial_mean": [batch, 2 * dimension],
            "innovation_loading": [batch, dimension, innovations],
            "transition_hessian": [batch, dimension, dimension + innovations, dimension + innovations],
            "second_order_constant": [batch, dimension],
            "observation_constant": [batch, observed],
            "observation_matrix": [batch, observed, dimension],
            "observation_hessian": [batch, observed, dimension, dimension],
        }
        for name, shape in expected.items():
            if getattr(self, name).shape.as_list() != shape:
                raise ValueError(f"{name} must have shape {shape}")
        for name, rows in (("initial_factor", 2 * dimension), ("innovation_factor", innovations), ("observation_factor", observed)):
            shape = getattr(self, name).shape
            if shape.rank != 3 or shape[:2] != (batch, rows) or shape[2] is None or not 0 <= shape[2] <= rows:
                raise ValueError(f"{name} must be [B,{rows},R] with 0 <= R <= {rows}")

    def transition(self, state: tf.Tensor, innovation: tf.Tensor) -> tf.Tensor:
        dimension = self.transition_matrix.shape[1]
        first, second = state[..., :dimension], state[..., dimension:]
        augmented = tf.concat([first, innovation], axis=-1)
        first_next = tf.einsum("bij,brj->bri", self.transition_matrix, first)
        first_next += tf.einsum("bij,brj->bri", self.innovation_loading, innovation)
        second_next = tf.einsum("bij,brj->bri", self.transition_matrix, second)
        second_next += self.second_order_constant[:, None, :]
        second_next += 0.5 * tf.einsum("bijk,brj,brk->bri", self.transition_hessian, augmented, augmented)
        return tf.concat([first_next, second_next], axis=-1)

    def observe(self, state: tf.Tensor) -> tf.Tensor:
        dimension = self.transition_matrix.shape[1]
        first, second = state[..., :dimension], state[..., dimension:]
        result = self.observation_constant[:, None, :]
        result += tf.einsum("bij,brj->bri", self.observation_matrix, first + second)
        return result + 0.5 * tf.einsum("bijk,brj,brk->bri", self.observation_hessian, first, first)


@dataclass(frozen=True)
class TFPrunedSRUKFResult:
    log_likelihood: tf.Tensor
    filtered_mean: tf.Tensor
    filtered_factor: tf.Tensor
    diagnostics: Mapping[str, tf.Tensor]


def _pad_factor(factor: tf.Tensor) -> tf.Tensor:
    return tf.pad(factor, [[0, 0], [0, 0], [0, factor.shape[1] - factor.shape[2]]])


def tf_pruned_srukf_filter(
    observations: tf.Tensor, model: TFPrunedSRUKFModel, *, jit_compile: bool = True,
) -> TFPrunedSRUKFResult:
    """Filter [B,T,M] observations; return final factors and numeric row status.

    Invalid rows return -inf likelihood and valid=False, including under XLA.
    A zero state pivot is permitted; a zero innovation pivot is invalid.
    No state rank truncation, covariance refactorization or ridge is applied.
    """

    observations = tf.convert_to_tensor(observations, tf.float64)
    batch, dimension, _ = model.transition_matrix.shape.as_list()
    state_dim = 2 * dimension
    innovation_dim = model.innovation_loading.shape[2]
    observed = model.observation_matrix.shape[1]
    if observations.shape.rank != 3 or observations.shape[0] != batch or observations.shape[2] != observed or observations.shape[1] is None:
        raise ValueError("observations must have static shape [B,T,M]")
    time_steps = int(observations.shape[1])
    offsets, mean_weights, covariance_weights = tf_factor_srukf_dz5_rule(state_dim + innovation_dim)
    sqrt_weights = tf.sqrt(covariance_weights)[None, None, :]

    def run(panel: tf.Tensor):
        valid = tf.reduce_all(tf.math.is_finite(panel), axis=[1, 2])
        for name in model.__dataclass_fields__:
            tensor = getattr(model, name)
            valid &= tf.reduce_all(tf.math.is_finite(tensor), axis=list(range(1, tensor.shape.rank)))
        factor = _pad_factor(model.initial_factor)
        process = _pad_factor(model.innovation_factor)
        noise = _pad_factor(model.observation_factor)
        value = tf.zeros([batch], tf.float64)
        minimum_pivot = tf.fill([batch], tf.constant(float("inf"), tf.float64))

        def body(index, mean, factor, value, valid, minimum_pivot):
            augmented_factor = tf.concat([
                tf.concat([factor, tf.zeros([batch, state_dim, innovation_dim], tf.float64)], axis=2),
                tf.concat([tf.zeros([batch, innovation_dim, state_dim], tf.float64), process], axis=2),
            ], axis=1)
            center = tf.concat([mean, tf.zeros([batch, innovation_dim], tf.float64)], axis=1)
            points = center[:, None, :] + tf.einsum("rd,bkd->brk", offsets, augmented_factor)
            predicted = model.transition(points[..., :state_dim], points[..., state_dim:])
            predicted_mean = tf.einsum("r,bri->bi", mean_weights, predicted)
            measurements = model.observe(predicted)
            predicted_observation = tf.einsum("r,bri->bi", mean_weights, measurements)
            state_stack = tf.transpose(predicted - predicted_mean[:, None, :], [0, 2, 1]) * sqrt_weights
            observation_stack = tf.transpose(measurements - predicted_observation[:, None, :], [0, 2, 1]) * sqrt_weights
            joint = tf.concat([
                tf.concat([observation_stack, noise], axis=2),
                tf.concat([state_stack, tf.zeros([batch, state_dim, observed], tf.float64)], axis=2),
            ], axis=1)
            joint_factor = batched_semidefinite_stack_qr_lower(joint)
            innovation_factor = joint_factor[:, :observed, :observed]
            cross_factor = joint_factor[:, observed:, :observed]
            next_factor = joint_factor[:, observed:, observed:]
            pivots = tf.linalg.diag_part(innovation_factor)
            step_valid = tf.reduce_all(tf.math.is_finite(joint_factor), axis=[1, 2])
            step_valid &= tf.reduce_all(pivots > 0.0, axis=1)
            safe_factor = tf.where(step_valid[:, None, None], innovation_factor, tf.eye(observed, batch_shape=[batch], dtype=tf.float64))
            innovation = panel[:, index, :] - predicted_observation
            standardized = tf.linalg.triangular_solve(safe_factor, innovation[..., None], lower=True)[..., 0]
            increment = -0.5 * (observed * math.log(2.0 * math.pi) + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(safe_factor)), axis=1) + tf.reduce_sum(tf.square(standardized), axis=1))
            next_mean = predicted_mean + tf.einsum("bij,bj->bi", cross_factor, standardized)
            step_valid &= tf.math.is_finite(increment) & tf.reduce_all(tf.math.is_finite(next_mean), axis=1)
            return index + 1, next_mean, next_factor, value + increment, valid & step_valid, tf.minimum(minimum_pivot, tf.reduce_min(pivots, axis=1))

        _, mean, factor, value, valid, minimum_pivot = tf.while_loop(
            lambda index, *_: index < time_steps,
            body,
            (tf.constant(0), model.initial_mean, factor, value, valid, minimum_pivot),
            parallel_iterations=1, maximum_iterations=time_steps,
        )
        return tf.where(valid, value, tf.constant(float("-inf"), tf.float64)), mean, factor, valid, minimum_pivot

    if jit_compile:
        run = tf.function(run, input_signature=[tf.TensorSpec(observations.shape, tf.float64)], autograph=False, jit_compile=True)
    value, mean, factor, valid, minimum_pivot = run(observations)
    return TFPrunedSRUKFResult(value, mean, factor, {
        "valid": valid,
        "invalid_count": tf.cast(~valid, tf.int32),
        "minimum_innovation_pivot": minimum_pivot,
        "minimum_state_pivot": tf.reduce_min(tf.linalg.diag_part(factor), axis=1),
    })


def make_pruned_srukf_value_and_score(
    model_fn: Callable[[tf.Tensor], TFPrunedSRUKFModel],
    parameter_spec: tf.TensorSpec,
    observation_spec: tf.TensorSpec,
    *, jit_compile: bool = True,
) -> Callable:
    """Compile a row-independent model builder and its total likelihood score.

    model_fn must preserve row locality. Autodiff includes coefficients and
    initial/noise factors; it does not use pfor or native singular-QR gradients.
    """

    @tf.function(input_signature=[parameter_spec, observation_spec], autograph=False, jit_compile=jit_compile)
    def value_and_score(parameters: tf.Tensor, observations: tf.Tensor):
        with tf.GradientTape() as tape:
            tape.watch(parameters)
            result = tf_pruned_srukf_filter(observations, model_fn(parameters), jit_compile=False)
            total = tf.reduce_sum(result.log_likelihood)
        score = tape.gradient(total, parameters, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        valid = result.diagnostics["valid"] & tf.reduce_all(tf.math.is_finite(score), axis=1)
        diagnostics = dict(result.diagnostics)
        diagnostics["valid"] = valid
        diagnostics["invalid_count"] = tf.cast(~valid, tf.int32)
        return tf.where(valid, result.log_likelihood, tf.constant(float("-inf"), tf.float64)), tf.where(valid[:, None], score, tf.constant(float("nan"), tf.float64)), diagnostics

    return value_and_score
