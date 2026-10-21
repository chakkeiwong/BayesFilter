"""Fixed-topology exact-likelihood Laplace proposal mechanics.

The factories in this module are model independent. Callers provide batched
TensorFlow functions for the observation log likelihood, its state score, and
its negative state Hessian. The resulting Gaussian bank is a proposal only;
an exact model target and complete proposal denominator remain the authority
for importance weighting.

The first claim-bearing use freezes every returned proposal quantity before
the analytical APF score is evaluated. This module does not claim a total
derivative through Newton iteration. Invalid covariance or curvature rows fail
closed and are never ridged, clipped, or routed to another proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import tensorflow as tf


DTYPE = tf.float64
ROUTE_ID = "exact_likelihood_fixed_laplace_bank_candidate_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"

StateLogLikelihoodFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
StateScoreFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
StateNegativeHessianFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class FixedLaplaceConfig:
    """Explicit fixed Newton schedule for one proposal factory."""

    tempering_schedule: tuple[float, ...]
    step_fractions: tuple[float, ...]
    start_scale: float
    stationarity_relative_tolerance: float

    def __post_init__(self) -> None:
        temperatures = tuple(float(value) for value in self.tempering_schedule)
        steps = tuple(float(value) for value in self.step_fractions)
        if not temperatures or len(temperatures) != len(steps):
            raise ValueError("tempering and step schedules must have equal nonzero length")
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in temperatures
        ):
            raise ValueError("tempering values must be finite and in (0, 1]")
        if any(right < left for left, right in zip(temperatures, temperatures[1:])):
            raise ValueError("tempering schedule must be nondecreasing")
        if temperatures[-1] != 1.0:
            raise ValueError("the final tempering value must equal one")
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0
            for value in steps
        ):
            raise ValueError("step fractions must be finite and in (0, 1]")
        if not math.isfinite(float(self.start_scale)) or float(self.start_scale) < 0.0:
            raise ValueError("start_scale must be finite and nonnegative")
        tolerance = float(self.stationarity_relative_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("stationarity tolerance must be finite and positive")
        object.__setattr__(self, "tempering_schedule", temperatures)
        object.__setattr__(self, "step_fractions", steps)
        object.__setattr__(self, "start_scale", float(self.start_scale))
        object.__setattr__(self, "stationarity_relative_tolerance", tolerance)


def regular_simplex_vertices(state_dim: int) -> tf.Tensor:
    """Return D+1 centered vertices with empirical covariance I_D."""

    dimension = int(state_dim)
    if dimension < 1:
        raise ValueError("state_dim must be positive")
    count = dimension + 1
    centering = tf.eye(count, dtype=DTYPE) - tf.ones(
        [count, count], dtype=DTYPE
    ) / tf.cast(count, DTYPE)
    basis = centering[:, :dimension]
    gram_cholesky = tf.linalg.cholesky(tf.linalg.matmul(basis, basis, transpose_a=True))
    whitened = tf.transpose(
        tf.linalg.triangular_solve(
            gram_cholesky,
            tf.transpose(basis),
            lower=True,
        )
    )
    return tf.sqrt(tf.cast(count, DTYPE)) * whitened


def single_start_offsets(state_dim: int) -> tf.Tensor:
    """Return the one-component start at the transition mean."""

    dimension = int(state_dim)
    if dimension < 1:
        raise ValueError("state_dim must be positive")
    return tf.zeros([1, dimension], DTYPE)


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _safe_spd_flat(
    matrix: tf.Tensor,
    *,
    dimension: int,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return an algebra-safe matrix and a separate fail-closed validity flag."""

    raw = _symmetrize(tf.convert_to_tensor(matrix, DTYPE))
    finite = tf.reduce_all(tf.math.is_finite(raw), axis=[1, 2])
    identity = tf.broadcast_to(
        tf.eye(dimension, dtype=DTYPE)[tf.newaxis, :, :], tf.shape(raw)
    )
    finite_matrix = tf.where(finite[:, tf.newaxis, tf.newaxis], raw, identity)
    eigenvalues = tf.linalg.eigvalsh(finite_matrix)
    minimum = tf.reduce_min(eigenvalues, axis=1)
    valid = finite & (minimum > tf.constant(0.0, DTYPE))
    safe = tf.where(valid[:, tf.newaxis, tf.newaxis], finite_matrix, identity)
    reported = tf.where(
        finite,
        minimum,
        tf.fill(tf.shape(minimum), tf.constant(float("-inf"), DTYPE)),
    )
    return safe, reported, valid


def make_fixed_laplace_bank_kernel(
    *,
    batch_size: int,
    state_dim: int,
    observation_dim: int,
    component_offsets: tf.Tensor,
    log_likelihood_fn: StateLogLikelihoodFn,
    state_score_fn: StateScoreFn,
    state_negative_hessian_fn: StateNegativeHessianFn,
    config: FixedLaplaceConfig,
    jit_compile: bool = True,
):
    """Create one fixed-shape exact-likelihood Laplace-bank kernel.

    Callback state rows have shape ``[B*K,D]`` and the observation has shape
    ``[M]``. The callbacks must return ``[B*K]``, ``[B*K,D]``, and
    ``[B*K,D,D]`` respectively.
    """

    count = int(batch_size)
    dimension = int(state_dim)
    observed_dimension = int(observation_dim)
    if count < 1 or dimension < 1 or observed_dimension < 1:
        raise ValueError("batch, state, and observation dimensions must be positive")
    if not isinstance(config, FixedLaplaceConfig):
        raise TypeError("config must be a FixedLaplaceConfig")
    offsets = tf.convert_to_tensor(component_offsets, DTYPE)
    if (
        offsets.shape.rank != 2
        or not offsets.shape.is_fully_defined()
        or offsets.shape[1] != dimension
        or int(offsets.shape[0]) < 1
    ):
        raise ValueError("component_offsets must have static shape [K,D]")
    if not bool(tf.reduce_all(tf.math.is_finite(offsets)).numpy()):
        raise ValueError("component_offsets must be finite")
    component_count = int(offsets.shape[0])
    flat_count = count * component_count
    schedule = tf.constant(config.tempering_schedule, DTYPE)
    step_fractions = tf.constant(config.step_fractions, DTYPE)
    iteration_count = len(config.tempering_schedule)
    identity = tf.eye(dimension, dtype=DTYPE)
    flat_identity = tf.broadcast_to(identity[tf.newaxis, :, :], [flat_count, dimension, dimension])

    @tf.function(
        input_signature=[
            tf.TensorSpec([count, dimension], DTYPE),
            tf.TensorSpec([count, dimension, dimension], DTYPE),
            tf.TensorSpec([observed_dimension], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        prior_means: tf.Tensor,
        prior_covariances: tf.Tensor,
        observation: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        prior_means = tf.ensure_shape(prior_means, [count, dimension])
        prior_covariances = tf.ensure_shape(
            prior_covariances, [count, dimension, dimension]
        )
        observation = tf.ensure_shape(observation, [observed_dimension])
        prior_safe, prior_minimum, prior_valid = _safe_spd_flat(
            prior_covariances, dimension=dimension
        )
        prior_cholesky = tf.linalg.cholesky(prior_safe)
        prior_precision = tf.linalg.cholesky_solve(
            prior_cholesky,
            tf.broadcast_to(identity[tf.newaxis, :, :], [count, dimension, dimension]),
        )
        placed_offsets = tf.einsum("bij,kj->bki", prior_cholesky, offsets)
        states = prior_means[:, tf.newaxis, :] + tf.constant(
            config.start_scale, DTYPE
        ) * placed_offsets
        flat_prior_means = tf.reshape(
            tf.broadcast_to(
                prior_means[:, tf.newaxis, :], [count, component_count, dimension]
            ),
            [flat_count, dimension],
        )
        flat_prior_precision = tf.reshape(
            tf.broadcast_to(
                prior_precision[:, tf.newaxis, :, :],
                [count, component_count, dimension, dimension],
            ),
            [flat_count, dimension, dimension],
        )
        component_valid = tf.reshape(
            tf.broadcast_to(prior_valid[:, tf.newaxis], [count, component_count]),
            [flat_count],
        )
        minimum_trace = tf.TensorArray(
            DTYPE, size=iteration_count, clear_after_read=False
        )
        step_trace = tf.TensorArray(
            DTYPE, size=iteration_count, clear_after_read=False
        )
        objective_trace = tf.TensorArray(
            DTYPE, size=iteration_count, clear_after_read=False
        )
        objective_after_trace = tf.TensorArray(
            DTYPE, size=iteration_count, clear_after_read=False
        )

        def condition(
            index: tf.Tensor,
            *_: object,
        ) -> tf.Tensor:
            return index < tf.constant(iteration_count, tf.int32)

        def body(
            index: tf.Tensor,
            current_states: tf.Tensor,
            current_valid: tf.Tensor,
            minimums: tf.TensorArray,
            steps: tf.TensorArray,
            objectives: tf.TensorArray,
            objectives_after: tf.TensorArray,
        ):
            flat_states = tf.reshape(current_states, [flat_count, dimension])
            temperature = schedule[index]
            fraction = step_fractions[index]
            raw_log_likelihood = tf.ensure_shape(
                tf.convert_to_tensor(
                    log_likelihood_fn(flat_states, observation), DTYPE
                ),
                [flat_count],
            )
            raw_score = tf.ensure_shape(
                tf.convert_to_tensor(state_score_fn(flat_states, observation), DTYPE),
                [flat_count, dimension],
            )
            raw_information = tf.ensure_shape(
                tf.convert_to_tensor(
                    state_negative_hessian_fn(flat_states, observation), DTYPE
                ),
                [flat_count, dimension, dimension],
            )
            callback_finite = (
                tf.math.is_finite(raw_log_likelihood)
                & tf.reduce_all(tf.math.is_finite(raw_score), axis=1)
                & tf.reduce_all(tf.math.is_finite(raw_information), axis=[1, 2])
            )
            safe_log_likelihood = tf.where(
                callback_finite, raw_log_likelihood, tf.zeros_like(raw_log_likelihood)
            )
            safe_score = tf.where(
                callback_finite[:, tf.newaxis], raw_score, tf.zeros_like(raw_score)
            )
            safe_information = tf.where(
                callback_finite[:, tf.newaxis, tf.newaxis],
                _symmetrize(raw_information),
                tf.zeros_like(raw_information),
            )
            delta = flat_states - flat_prior_means
            prior_gradient = tf.einsum("bij,bj->bi", flat_prior_precision, delta)
            precision_raw = flat_prior_precision + temperature * safe_information
            precision, minimum, precision_valid = _safe_spd_flat(
                precision_raw, dimension=dimension
            )
            precision_cholesky = tf.linalg.cholesky(precision)
            negative_gradient = prior_gradient - temperature * safe_score
            update = -tf.linalg.cholesky_solve(
                precision_cholesky, negative_gradient[:, :, tf.newaxis]
            )[:, :, 0]
            next_flat_states = flat_states + fraction * update
            next_finite = tf.reduce_all(tf.math.is_finite(next_flat_states), axis=1)
            next_valid = current_valid & callback_finite & precision_valid & next_finite
            next_flat_states = tf.where(
                next_valid[:, tf.newaxis], next_flat_states, flat_states
            )
            next_log_likelihood_raw = tf.ensure_shape(
                tf.convert_to_tensor(
                    log_likelihood_fn(next_flat_states, observation), DTYPE
                ),
                [flat_count],
            )
            next_log_likelihood_finite = tf.math.is_finite(next_log_likelihood_raw)
            next_valid = next_valid & next_log_likelihood_finite
            next_log_likelihood = tf.where(
                next_log_likelihood_finite,
                next_log_likelihood_raw,
                tf.zeros_like(next_log_likelihood_raw),
            )
            # This is the log-posterior objective L=-Phi.  The calibration
            # diagnostic therefore checks ascent after each fixed Newton step.
            objective_before = (
                -0.5 * tf.einsum("bi,bi->b", delta, prior_gradient)
                + temperature * safe_log_likelihood
            )
            next_delta = next_flat_states - flat_prior_means
            next_prior_gradient = tf.einsum(
                "bij,bj->bi", flat_prior_precision, next_delta
            )
            objective_after = (
                -0.5
                * tf.einsum("bi,bi->b", next_delta, next_prior_gradient)
                + temperature * next_log_likelihood
            )
            return (
                index + 1,
                tf.reshape(next_flat_states, [count, component_count, dimension]),
                next_valid,
                minimums.write(index, tf.reshape(minimum, [count, component_count])),
                steps.write(
                    index,
                    tf.reshape(
                        tf.reduce_max(tf.abs(fraction * update), axis=1),
                        [count, component_count],
                    ),
                ),
                objectives.write(
                    index, tf.reshape(objective_before, [count, component_count])
                ),
                objectives_after.write(
                    index, tf.reshape(objective_after, [count, component_count])
                ),
            )

        (
            _,
            final_states,
            component_valid,
            minimum_trace,
            step_trace,
            objective_trace,
            objective_after_trace,
        ) = tf.while_loop(
            condition,
            body,
            (
                tf.constant(0, tf.int32),
                states,
                component_valid,
                minimum_trace,
                step_trace,
                objective_trace,
                objective_after_trace,
            ),
            parallel_iterations=1,
        )

        flat_states = tf.reshape(final_states, [flat_count, dimension])
        final_log_likelihood = tf.ensure_shape(
            tf.convert_to_tensor(log_likelihood_fn(flat_states, observation), DTYPE),
            [flat_count],
        )
        final_score = tf.ensure_shape(
            tf.convert_to_tensor(state_score_fn(flat_states, observation), DTYPE),
            [flat_count, dimension],
        )
        final_information = tf.ensure_shape(
            tf.convert_to_tensor(
                state_negative_hessian_fn(flat_states, observation), DTYPE
            ),
            [flat_count, dimension, dimension],
        )
        final_callback_finite = (
            tf.math.is_finite(final_log_likelihood)
            & tf.reduce_all(tf.math.is_finite(final_score), axis=1)
            & tf.reduce_all(tf.math.is_finite(final_information), axis=[1, 2])
        )
        safe_final_information = tf.where(
            final_callback_finite[:, tf.newaxis, tf.newaxis],
            _symmetrize(final_information),
            tf.zeros_like(final_information),
        )
        final_precision, final_minimum, final_precision_valid = _safe_spd_flat(
            flat_prior_precision + safe_final_information,
            dimension=dimension,
        )
        final_precision_cholesky = tf.linalg.cholesky(final_precision)
        flat_covariance = tf.linalg.cholesky_solve(
            final_precision_cholesky, flat_identity
        )
        flat_covariance = _symmetrize(flat_covariance)
        covariance_safe, covariance_minimum, covariance_valid = _safe_spd_flat(
            flat_covariance, dimension=dimension
        )
        flat_covariance_cholesky = tf.linalg.cholesky(covariance_safe)

        delta = flat_states - flat_prior_means
        prior_gradient = tf.einsum("bij,bj->bi", flat_prior_precision, delta)
        residual = prior_gradient - final_score
        residual_scale = (
            tf.constant(1.0, DTYPE)
            + tf.reduce_max(tf.abs(prior_gradient), axis=1)
            + tf.reduce_max(tf.abs(final_score), axis=1)
        )
        relative_residual = tf.reduce_max(tf.abs(residual), axis=1) / residual_scale
        stationary = relative_residual <= tf.constant(
            config.stationarity_relative_tolerance, DTYPE
        )
        component_valid = (
            component_valid
            & final_callback_finite
            & final_precision_valid
            & covariance_valid
            & stationary
        )

        prior_logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(prior_cholesky)), axis=1
        )
        flat_prior_logdet = tf.reshape(
            tf.broadcast_to(
                prior_logdet[:, tf.newaxis], [count, component_count]
            ),
            [flat_count],
        )
        quadratic = tf.einsum("bi,bi->b", delta, prior_gradient)
        log_prior = -0.5 * (
            tf.cast(dimension, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
            + flat_prior_logdet
            + quadratic
        )
        covariance_logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(flat_covariance_cholesky)), axis=1
        )
        evidence = (
            log_prior
            + final_log_likelihood
            + 0.5 * tf.cast(dimension, DTYPE) * tf.constant(math.log(2.0 * math.pi), DTYPE)
            + 0.5 * covariance_logdet
        )
        evidence = tf.reshape(evidence, [count, component_count])
        equal_weights = tf.fill(
            [count, component_count],
            tf.constant(1.0 / float(component_count), DTYPE),
        )
        evidence_weights = tf.nn.softmax(evidence, axis=1)
        weight_valid = (
            tf.reduce_all(tf.math.is_finite(evidence), axis=1)
            & tf.reduce_all(evidence_weights > tf.constant(0.0, DTYPE), axis=1)
        )
        component_valid_matrix = tf.reshape(
            component_valid, [count, component_count]
        )
        row_valid = (
            tf.reduce_all(component_valid_matrix, axis=1)
            & prior_valid
            & weight_valid
            & tf.reduce_all(tf.math.is_finite(observation))
        )
        lookahead = tf.reduce_logsumexp(evidence, axis=1) - tf.math.log(
            tf.cast(component_count, DTYPE)
        )
        stacked_minimum_trace = minimum_trace.stack()
        stacked_step_trace = step_trace.stack()
        stacked_objective_trace = objective_trace.stack()
        stacked_objective_after_trace = objective_after_trace.stack()
        all_finite = tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(final_states, [-1]),
                        tf.reshape(covariance_safe, [-1]),
                        tf.reshape(flat_covariance_cholesky, [-1]),
                        tf.reshape(evidence, [-1]),
                        tf.reshape(lookahead, [-1]),
                        tf.reshape(relative_residual, [-1]),
                        tf.reshape(stacked_minimum_trace, [-1]),
                        tf.reshape(stacked_step_trace, [-1]),
                        tf.reshape(stacked_objective_trace, [-1]),
                        tf.reshape(stacked_objective_after_trace, [-1]),
                    ],
                    axis=0,
                )
            )
        )
        return {
            "component_means": final_states,
            "component_covariances": tf.reshape(
                covariance_safe,
                [count, component_count, dimension, dimension],
            ),
            "component_cholesky": tf.reshape(
                flat_covariance_cholesky,
                [count, component_count, dimension, dimension],
            ),
            "component_log_evidence": evidence,
            "equal_component_weights": equal_weights,
            "evidence_component_weights": evidence_weights,
            "lookahead_log_likelihood": lookahead,
            "relative_stationarity_residual": tf.reshape(
                relative_residual, [count, component_count]
            ),
            "minimum_precision_eigenvalue": tf.reshape(
                final_minimum, [count, component_count]
            ),
            "minimum_covariance_eigenvalue": tf.reshape(
                covariance_minimum, [count, component_count]
            ),
            "iteration_minimum_precision_eigenvalue": stacked_minimum_trace,
            "iteration_step_max_abs": stacked_step_trace,
            "iteration_objective_before_step": stacked_objective_trace,
            "iteration_objective_after_step": stacked_objective_after_trace,
            "component_valid": component_valid_matrix,
            "valid_rows": row_valid,
            "valid": tf.reduce_all(row_valid) & all_finite,
            "finite": all_finite,
            "component_count": tf.constant(component_count, tf.int32),
            "iteration_count": tf.constant(iteration_count, tf.int32),
            "prior_minimum_eigenvalue": prior_minimum,
        }

    return kernel


def require_valid_laplace_result(result: Mapping[str, tf.Tensor]) -> None:
    """Raise at the host boundary when any fixed Laplace row is invalid."""

    if "valid" not in result:
        raise ValueError("Laplace result does not expose a validity flag")
    if bool(tf.convert_to_tensor(result["valid"]).numpy()):
        return
    rows = result.get("valid_rows")
    components = result.get("component_valid")
    if rows is None or components is None:
        raise ValueError("fixed Laplace result is invalid")
    invalid_rows = int(
        tf.reduce_sum(tf.cast(~tf.convert_to_tensor(rows), tf.int32)).numpy()
    )
    invalid_components = int(
        tf.reduce_sum(tf.cast(~tf.convert_to_tensor(components), tf.int32)).numpy()
    )
    raise ValueError(
        "fixed Laplace result has "
        f"{invalid_rows} invalid row(s) and {invalid_components} invalid component(s)"
    )


def schedule_manifest(config: FixedLaplaceConfig) -> Mapping[str, object]:
    """Return JSON-compatible provenance for a fixed schedule."""

    return {
        "tempering_schedule": list(config.tempering_schedule),
        "step_fractions": list(config.step_fractions),
        "start_scale": float(config.start_scale),
        "stationarity_relative_tolerance": float(
            config.stationarity_relative_tolerance
        ),
        "ridge": 0.0,
        "eigenvalue_clipping": False,
        "invalid_curvature_policy": "fail_closed",
    }


__all__ = [
    "DTYPE",
    "FixedLaplaceConfig",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "make_fixed_laplace_bank_kernel",
    "regular_simplex_vertices",
    "require_valid_laplace_result",
    "schedule_manifest",
    "single_start_offsets",
]
