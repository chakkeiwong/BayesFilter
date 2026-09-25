"""TensorFlow diagnostics for adaptive annealed SMC coordination."""

from __future__ import annotations

from typing import Any, Callable, Mapping

import tensorflow as tf
import tensorflow_probability as tfp


LogProbFn = Callable[[tf.Tensor], tf.Tensor]


def normalized_weight_diagnostics(log_weights: Any) -> Mapping[str, tf.Tensor]:
    """Normalize finite log weights and return ESS diagnostics."""

    values = tf.convert_to_tensor(log_weights)
    if not values.dtype.is_floating or values.shape.rank != 1:
        raise ValueError("log_weights must be a floating vector")
    tf.debugging.assert_all_finite(values, "SMC log weights")
    normalized_log_weights = tf.nn.log_softmax(values)
    normalized_weights = tf.exp(normalized_log_weights)
    ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights)))
    return {
        "normalized_log_weights": normalized_log_weights,
        "normalized_weights": normalized_weights,
        "effective_sample_size": ess,
        "effective_sample_size_fraction": ess / tf.cast(tf.size(values), values.dtype),
        "maximum_normalized_weight": tf.reduce_max(normalized_weights),
    }


def select_next_beta(
    beta: Any,
    log_ratio: Any,
    current_log_weights: Any,
    *,
    target_ess_fraction: float,
    bisection_iterations: int = 24,
    beta_tolerance: float = 1.0e-6,
) -> Mapping[str, tf.Tensor]:
    """Select the largest beta whose cumulative ESS meets the target."""

    ratio = tf.convert_to_tensor(log_ratio)
    weights = tf.convert_to_tensor(current_log_weights, ratio.dtype)
    current_beta = tf.convert_to_tensor(beta, ratio.dtype)
    if ratio.shape.rank != 1 or weights.shape != ratio.shape:
        raise ValueError("log_ratio and current_log_weights must be equal vectors")
    if not 0.0 < float(target_ess_fraction) <= 1.0:
        raise ValueError("target_ess_fraction must be in (0,1]")
    if isinstance(bisection_iterations, bool) or int(bisection_iterations) <= 0:
        raise ValueError("bisection_iterations must be positive")
    tf.debugging.assert_all_finite(ratio, "SMC log density ratio")
    tf.debugging.assert_all_finite(weights, "SMC current log weights")
    tf.debugging.assert_greater_equal(current_beta, tf.constant(0.0, ratio.dtype))
    tf.debugging.assert_less_equal(current_beta, tf.constant(1.0, ratio.dtype))
    target = tf.constant(float(target_ess_fraction), ratio.dtype)

    def ess(candidate: tf.Tensor) -> tf.Tensor:
        candidate_weights = weights + (candidate - current_beta) * ratio
        return normalized_weight_diagnostics(candidate_weights)[
            "effective_sample_size_fraction"
        ]

    terminal_ess = ess(tf.constant(1.0, ratio.dtype))

    def terminal() -> tuple[tf.Tensor, tf.Tensor]:
        return tf.constant(1.0, ratio.dtype), terminal_ess

    def bisect() -> tuple[tf.Tensor, tf.Tensor]:
        low = current_beta
        high = tf.constant(1.0, ratio.dtype)
        for _ in range(int(bisection_iterations)):
            middle = (low + high) / 2.0
            middle_ess = ess(middle)
            low = tf.where(middle_ess >= target, middle, low)
            high = tf.where(middle_ess >= target, high, middle)
        return low, ess(low)

    next_beta, selected_ess = tf.cond(terminal_ess >= target, terminal, bisect)
    reached = next_beta >= tf.constant(1.0 - float(beta_tolerance), ratio.dtype)
    next_beta = tf.where(reached, tf.constant(1.0, ratio.dtype), next_beta)
    return {
        "next_beta": next_beta,
        "delta_beta": next_beta - current_beta,
        "effective_sample_size_fraction": selected_ess,
        "terminal_ess_fraction": terminal_ess,
        "target_reached": reached,
    }


def systematic_resample_indices(
    normalized_log_weights: Any,
    *,
    seed: tuple[int, int] | tf.Tensor,
) -> tf.Tensor:
    """Draw one global systematic-resampling parent vector."""

    values = tf.convert_to_tensor(normalized_log_weights)
    if not values.dtype.is_floating or values.shape.rank != 1:
        raise ValueError("normalized_log_weights must be a floating vector")
    event_size = values.shape[0]
    if event_size is None:
        raise ValueError("systematic resampling requires a static particle count")
    tf.debugging.assert_all_finite(values, "normalized SMC log weights")
    tf.debugging.assert_near(
        tf.reduce_logsumexp(values),
        tf.constant(0.0, values.dtype),
        atol=tf.constant(1.0e-10, values.dtype),
    )
    indices = tfp.experimental.mcmc.resample_systematic(
        values,
        event_size=int(event_size),
        sample_shape=(),
        seed=tf.convert_to_tensor(seed, tf.int32),
    )
    return tf.ensure_shape(tf.convert_to_tensor(indices, tf.int32), values.shape)


def make_bridge_hmc_step(
    proposal_log_prob_fn: LogProbFn,
    target_log_prob_fn: LogProbFn,
    *,
    path_count: int,
    dimension: int,
    step_size: float,
    num_leapfrog_steps: int,
    jit_compile: bool = True,
) -> Callable[[tf.Tensor, tf.Tensor, tf.Tensor], Mapping[str, tf.Tensor]]:
    """Build a reusable, bridge-fresh fixed-HMC mutation step."""

    if int(path_count) <= 0 or int(dimension) <= 0:
        raise ValueError("path_count and dimension must be positive")
    if not float(step_size) > 0.0:
        raise ValueError("step_size must be positive")
    if int(num_leapfrog_steps) < 2:
        raise ValueError("num_leapfrog_steps must be at least two")

    @tf.function(
        input_signature=(
            tf.TensorSpec([int(path_count), int(dimension)], tf.float64),
            tf.TensorSpec([], tf.float64),
            tf.TensorSpec([2], tf.int32),
        ),
        jit_compile=jit_compile,
        reduce_retracing=False,
    )
    def step(state: tf.Tensor, beta: tf.Tensor, seed: tf.Tensor) -> Mapping[str, tf.Tensor]:
        def bridge(value: tf.Tensor) -> tf.Tensor:
            proposal = tf.convert_to_tensor(proposal_log_prob_fn(value), tf.float64)
            target = tf.convert_to_tensor(target_log_prob_fn(value), tf.float64)
            return beta * target + (tf.constant(1.0, tf.float64) - beta) * proposal

        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=bridge,
            step_size=tf.constant(float(step_size), tf.float64),
            num_leapfrog_steps=int(num_leapfrog_steps),
        )
        bootstrap = kernel.bootstrap_results(state)
        next_state, result = kernel.one_step(state, bootstrap, seed=seed)
        return {
            "state": next_state,
            "is_accepted": result.is_accepted,
            "log_accept_ratio": result.log_accept_ratio,
            "accepted_target_log_prob": result.accepted_results.target_log_prob,
            "proposed_target_log_prob": result.proposed_results.target_log_prob,
        }

    return step
