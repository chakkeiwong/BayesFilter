"""Generic batched UKF and K=1 APF proposal primitives for the C2 program.

This module is the repository-owned endpoint for the mixture-UKF campaign.  It
is model independent: callers provide TensorFlow transition and observation
maps on a fixed sigma-point tensor.  The kernel returns UKF moments and the
predictive innovation law; the APF compiler uses those moments only to select
ancestors and to construct a proposal.  It does not evaluate or replace an
exact model target, and it does not claim posterior or production readiness.

The repeated numerical kernel is a fixed-shape ``tf.function`` with XLA on by
default.  No NumPy, pfor, vectorized_map, or sample-wise Python loop is used.
Covariance checks fail closed instead of silently repairing an invalid
covariance.  A nonzero jitter is retained as an explicit Class-C option and is
recorded by the caller; the Phase 0 fixture uses zero jitter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Mapping, Sequence

import tensorflow as tf


DTYPE = tf.float64
ROUTE_ID = "c2_generic_batched_ukf_apf_candidate_v1"
ROUTE_CLASSIFICATION = "extension_or_invention_candidate_diagnostic_only"

TransitionSigmaFn = Callable[[tf.Tensor], tf.Tensor]
ObservationSigmaFn = Callable[[tf.Tensor], tf.Tensor]


@dataclass(frozen=True)
class BatchedUKFConfig:
    """Fixed sigma-point and covariance policy for one compiled kernel."""

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0
    jitter: float = 0.0
    rank_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(float(self.beta)):
            raise ValueError("beta must be finite")
        if not math.isfinite(float(self.kappa)):
            raise ValueError("kappa must be finite")
        if not math.isfinite(float(self.jitter)) or float(self.jitter) < 0.0:
            raise ValueError("jitter must be finite and nonnegative")
        if (
            not math.isfinite(float(self.rank_tolerance))
            or float(self.rank_tolerance) <= 0.0
        ):
            raise ValueError("rank_tolerance must be finite and positive")
        spread = float(self.alpha) ** 2 * (1.0 + float(self.kappa))
        if spread <= 0.0:
            # The dimension-dependent check is repeated by the factory.
            raise ValueError("alpha**2 * (1 + kappa) must be positive")


def _symmetrize(matrix: tf.Tensor) -> tf.Tensor:
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _row_finite(value: tf.Tensor) -> tf.Tensor:
    """Return one finite/invalid flag per leading batch row."""

    rank = value.shape.rank
    if rank is None or rank < 2:
        raise ValueError("batched values require a statically known rank >= 2")
    return tf.reduce_all(tf.math.is_finite(value), axis=list(range(1, rank)))


def _safe_spd(
    matrix: tf.Tensor,
    *,
    dimension: int,
    jitter: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return a finite SPD matrix, its raw minimum eigenvalue, and validity.

    XLA may erase assertion operators.  Invalid rows therefore select an
    identity matrix for the algebra while the returned validity flag remains
    false; the host-side proposal compiler rejects that result.  A valid row
    is never clipped or ridged unless the caller explicitly requested
    ``jitter``.
    """

    raw = _symmetrize(matrix)
    identity = tf.eye(dimension, dtype=DTYPE)[tf.newaxis, :, :]
    identity = tf.broadcast_to(identity, tf.shape(raw))
    raw = raw + jitter * identity
    finite = _row_finite(raw)
    finite_matrix = tf.where(finite[:, tf.newaxis, tf.newaxis], raw, identity)
    eigenvalues = tf.linalg.eigvalsh(finite_matrix)
    minimum = tf.reduce_min(eigenvalues, axis=1)
    valid = finite & (minimum > tf.constant(0.0, DTYPE))
    safe = tf.where(valid[:, tf.newaxis, tf.newaxis], finite_matrix, identity)
    invalid_value = tf.fill(tf.shape(minimum), tf.constant(float("-inf"), DTYPE))
    reported_minimum = tf.where(finite, minimum, invalid_value)
    return safe, reported_minimum, valid


def _sigma_rule(
    state_dim: int,
    config: BatchedUKFConfig,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    state_dim = int(state_dim)
    spread = float(config.alpha) ** 2 * (float(state_dim) + float(config.kappa))
    if spread <= 0.0:
        raise ValueError("alpha**2 * (state_dim + kappa) must be positive")
    lambda_value = spread - float(state_dim)
    eye = tf.eye(state_dim, dtype=DTYPE)
    offsets = tf.concat(
        [
            tf.zeros([1, state_dim], dtype=DTYPE),
            tf.sqrt(tf.constant(spread, DTYPE)) * eye,
            -tf.sqrt(tf.constant(spread, DTYPE)) * eye,
        ],
        axis=0,
    )
    axis_weight = tf.constant(1.0 / (2.0 * spread), dtype=DTYPE)
    mean_weights = tf.concat(
        [
            tf.constant([lambda_value / spread], dtype=DTYPE),
            tf.fill([2 * state_dim], axis_weight),
        ],
        axis=0,
    )
    covariance_weights = tf.concat(
        [
            tf.constant(
                [lambda_value / spread + 1.0 - config.alpha**2 + config.beta],
                dtype=DTYPE,
            ),
            tf.fill([2 * state_dim], axis_weight),
        ],
        axis=0,
    )
    return offsets, mean_weights, covariance_weights


def _place_sigma_points(
    means: tf.Tensor,
    covariances: tf.Tensor,
    offsets: tf.Tensor,
    jitter: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Place one fixed sigma rule for every leading batch row."""

    batch_size = int(means.shape[0])
    state_dim = int(means.shape[1])
    covariance, minimum, covariance_valid = _safe_spd(
        covariances,
        dimension=state_dim,
        jitter=jitter,
    )
    means_finite = _row_finite(means)
    safe_means = tf.where(means_finite[:, tf.newaxis], means, tf.zeros_like(means))
    chol = tf.linalg.cholesky(covariance)
    point_offsets = tf.einsum("pi,bji->bpj", offsets, chol)
    points = safe_means[:, tf.newaxis, :] + point_offsets
    return (
        tf.ensure_shape(points, [batch_size, int(offsets.shape[0]), state_dim]),
        chol,
        covariance_valid & means_finite,
        minimum,
    )


def _weighted_mean(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
    return tf.einsum("p,bpd->bd", weights, points)


def _weighted_covariance(
    points: tf.Tensor,
    means: tf.Tensor,
    weights: tf.Tensor,
) -> tf.Tensor:
    centered = points - means[:, tf.newaxis, :]
    return _symmetrize(tf.einsum("p,bpi,bpj->bij", weights, centered, centered))


def make_batched_ukf_kernel(
    *,
    batch_size: int,
    state_dim: int,
    observation_dim: int,
    transition_fn: TransitionSigmaFn,
    observation_fn: ObservationSigmaFn,
    config: BatchedUKFConfig | None = None,
    jit_compile: bool = True,
):
    """Create a fixed-shape batched UKF conditional kernel.

    ``transition_fn`` receives a tensor with shape ``[B, 2D+1, D]`` and must
    return the transitioned sigma points with the same shape.  ``observation_fn``
    receives ``[B, 2D+1, D]`` and returns ``[B, 2D+1, M]``.  Process and
    observation covariances are supplied per batch row, which permits the
    recursive per-ancestor covariance lifecycle without a lane-specific fork.
    """

    batch_size = int(batch_size)
    state_dim = int(state_dim)
    observation_dim = int(observation_dim)
    if batch_size < 1 or state_dim < 1 or observation_dim < 1:
        raise ValueError("batch_size, state_dim, and observation_dim must be positive")
    cfg = BatchedUKFConfig() if config is None else config
    offsets, mean_weights, covariance_weights = _sigma_rule(state_dim, cfg)
    point_count = 2 * state_dim + 1
    if offsets.shape != (point_count, state_dim):
        raise ValueError("internal sigma-point rule has an invalid shape")
    jitter = tf.constant(float(cfg.jitter), DTYPE)
    state_identity = tf.eye(state_dim, dtype=DTYPE)
    observation_identity = tf.eye(observation_dim, dtype=DTYPE)

    @tf.function(
        input_signature=[
            tf.TensorSpec([batch_size, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([observation_dim], DTYPE),
            tf.TensorSpec([batch_size, observation_dim, observation_dim], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def kernel(
        prior_means: tf.Tensor,
        prior_covariances: tf.Tensor,
        process_covariances: tf.Tensor,
        observation: tf.Tensor,
        observation_covariances: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        prior_means = tf.ensure_shape(prior_means, [batch_size, state_dim])
        prior_covariances = tf.ensure_shape(
            prior_covariances, [batch_size, state_dim, state_dim]
        )
        process_covariances = tf.ensure_shape(
            process_covariances, [batch_size, state_dim, state_dim]
        )
        observation = tf.ensure_shape(observation, [observation_dim])
        observation_covariances = tf.ensure_shape(
            observation_covariances,
            [batch_size, observation_dim, observation_dim],
        )
        input_valid = (
            tf.reduce_all(tf.math.is_finite(prior_means))
            & tf.reduce_all(tf.math.is_finite(prior_covariances))
            & tf.reduce_all(tf.math.is_finite(process_covariances))
            & tf.reduce_all(tf.math.is_finite(observation))
            & tf.reduce_all(tf.math.is_finite(observation_covariances))
        )
        prior_sigma_points, prior_cholesky, prior_valid, prior_minimum = _place_sigma_points(
            prior_means, prior_covariances, offsets, jitter
        )
        transitioned = tf.ensure_shape(
            tf.convert_to_tensor(transition_fn(prior_sigma_points), dtype=DTYPE),
            [batch_size, point_count, state_dim],
        )
        transition_valid = _row_finite(transitioned)
        transitioned = tf.where(
            transition_valid[:, tf.newaxis, tf.newaxis],
            transitioned,
            tf.zeros_like(transitioned),
        )
        predicted_means = _weighted_mean(transitioned, mean_weights)
        process_finite = _row_finite(process_covariances)
        safe_process_covariances = tf.where(
            process_finite[:, tf.newaxis, tf.newaxis],
            _symmetrize(process_covariances),
            tf.broadcast_to(
                tf.eye(state_dim, dtype=DTYPE)[tf.newaxis, :, :],
                [batch_size, state_dim, state_dim],
            ),
        )
        predicted_covariance_raw = _weighted_covariance(
            transitioned, predicted_means, covariance_weights
        ) + safe_process_covariances
        predicted_covariances, predicted_minimum, predicted_valid = _safe_spd(
            predicted_covariance_raw,
            dimension=state_dim,
            jitter=tf.constant(0.0, DTYPE),
        )

        predicted_sigma_points, predicted_cholesky, predicted_place_valid, _ = _place_sigma_points(
            predicted_means, predicted_covariances, offsets, jitter
        )
        observed_sigma_points = tf.ensure_shape(
            tf.convert_to_tensor(observation_fn(predicted_sigma_points), dtype=DTYPE),
            [batch_size, point_count, observation_dim],
        )
        observation_valid = _row_finite(observed_sigma_points)
        observed_sigma_points = tf.where(
            observation_valid[:, tf.newaxis, tf.newaxis],
            observed_sigma_points,
            tf.zeros_like(observed_sigma_points),
        )
        predicted_observation_means = _weighted_mean(
            observed_sigma_points, mean_weights
        )
        centered_state = predicted_sigma_points - predicted_means[:, tf.newaxis, :]
        centered_observation = (
            observed_sigma_points - predicted_observation_means[:, tf.newaxis, :]
        )
        observation_covariance_finite = _row_finite(observation_covariances)
        safe_observation_covariances = tf.where(
            observation_covariance_finite[:, tf.newaxis, tf.newaxis],
            _symmetrize(observation_covariances),
            tf.broadcast_to(
                tf.eye(observation_dim, dtype=DTYPE)[tf.newaxis, :, :],
                [batch_size, observation_dim, observation_dim],
            ),
        )
        innovation_covariance_raw = _symmetrize(
            tf.einsum(
                "p,bpi,bpj->bij",
                covariance_weights,
                centered_observation,
                centered_observation,
            )
            + safe_observation_covariances
        )
        cross_covariances = tf.einsum(
            "p,bpi,bpo->bio",
            covariance_weights,
            centered_state,
            centered_observation,
        )
        innovation_covariances, innovation_minimum, innovation_valid = _safe_spd(
            innovation_covariance_raw,
            dimension=observation_dim,
            jitter=tf.constant(0.0, DTYPE),
        )
        innovation_eigenvalues = tf.linalg.eigvalsh(innovation_covariances)
        innovation_cholesky = tf.linalg.cholesky(innovation_covariances)
        innovation = observation[tf.newaxis, :] - predicted_observation_means
        gain = tf.transpose(
            tf.linalg.cholesky_solve(
                innovation_cholesky,
                tf.linalg.matrix_transpose(cross_covariances),
            ),
            [0, 2, 1],
        )
        posterior_means = predicted_means + tf.einsum(
            "bio,bo->bi", gain, innovation
        )
        posterior_covariance_raw = _symmetrize(
            predicted_covariances
            - tf.matmul(
                tf.matmul(gain, innovation_covariances),
                gain,
                transpose_b=True,
            )
        )
        posterior_means_finite = _row_finite(posterior_means)
        posterior_means = tf.where(
            posterior_means_finite[:, tf.newaxis],
            posterior_means,
            tf.zeros_like(posterior_means),
        )
        posterior_covariances, posterior_minimum, posterior_valid = _safe_spd(
            posterior_covariance_raw,
            dimension=state_dim,
            jitter=tf.constant(0.0, DTYPE),
        )
        posterior_cholesky = tf.linalg.cholesky(posterior_covariances)
        solved_innovation = tf.linalg.cholesky_solve(
            innovation_cholesky, innovation[:, :, tf.newaxis]
        )[:, :, 0]
        quadratic = tf.reduce_sum(innovation * solved_innovation, axis=1)
        logdet = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_cholesky)), axis=1
        )
        innovation_log_likelihood = -0.5 * (
            tf.cast(observation_dim, DTYPE)
            * tf.constant(math.log(2.0 * math.pi), DTYPE)
            + logdet
            + quadratic
        )
        valid_rows = (
            prior_valid
            & transition_valid
            & process_finite
            & predicted_valid
            & predicted_place_valid
            & observation_valid
            & observation_covariance_finite
            & innovation_valid
            & posterior_means_finite
            & posterior_valid
        )
        finite = (
            tf.reduce_all(valid_rows)
            & input_valid
            & tf.reduce_all(
                tf.math.is_finite(
                    tf.concat(
                        [
                            tf.reshape(predicted_means, [-1]),
                            tf.reshape(predicted_covariances, [-1]),
                            tf.reshape(predicted_observation_means, [-1]),
                            tf.reshape(innovation_covariances, [-1]),
                            tf.reshape(posterior_means, [-1]),
                            tf.reshape(posterior_covariances, [-1]),
                            tf.reshape(innovation_log_likelihood, [-1]),
                        ],
                        axis=0,
                    )
                )
            )
        )
        return {
            "predicted_mean": predicted_means,
            "predicted_covariance": predicted_covariances,
            "predicted_cholesky": predicted_cholesky,
            "predicted_observation_mean": predicted_observation_means,
            "innovation": innovation,
            "innovation_covariance": innovation_covariances,
            "innovation_cholesky": innovation_cholesky,
            "cross_covariance": cross_covariances,
            "gain": gain,
            "posterior_mean": posterior_means,
            "posterior_covariance": posterior_covariances,
            "posterior_cholesky": posterior_cholesky,
            "innovation_log_likelihood": innovation_log_likelihood,
            "prior_min_eigenvalue": prior_minimum,
            "predicted_min_eigenvalue": predicted_minimum,
            "innovation_min_eigenvalue": innovation_minimum,
            "posterior_min_eigenvalue": posterior_minimum,
            "sigma_point_count": tf.constant(point_count, tf.int32),
            "valid_rows": valid_rows,
            "valid": finite,
            "finite": finite,
        }

    return kernel


def apf_log_ancestor_probabilities(
    log_parent_weights: tf.Tensor,
    lookahead_log_likelihood: tf.Tensor,
) -> tf.Tensor:
    """Return normalized APF ancestor log probabilities."""

    parent = tf.reshape(tf.convert_to_tensor(log_parent_weights, DTYPE), [-1])
    lookahead = tf.reshape(tf.convert_to_tensor(lookahead_log_likelihood, DTYPE), [-1])
    if parent.shape != lookahead.shape or parent.shape[0] is None:
        raise ValueError("APF parent and lookahead vectors must have equal static shape")
    tf.debugging.assert_all_finite(parent, "parent log weights must be finite")
    tf.debugging.assert_all_finite(lookahead, "lookahead likelihood must be finite")
    return tf.nn.log_softmax(parent + lookahead)


def require_valid_ukf_result(result: Mapping[str, tf.Tensor]) -> None:
    """Raise at the host boundary when a compiled UKF row was invalid."""

    valid = result.get("valid")
    if valid is None:
        raise ValueError("UKF result does not expose a validity flag")
    if not bool(tf.convert_to_tensor(valid).numpy()):
        rows = result.get("valid_rows")
        if rows is None:
            raise ValueError("batched UKF result is invalid")
        invalid_count = int(tf.reduce_sum(tf.cast(~tf.convert_to_tensor(rows), tf.int32)).numpy())
        raise ValueError(f"batched UKF result has {invalid_count} invalid row(s)")


def gaussian_log_density(
    points: tf.Tensor,
    means: tf.Tensor,
    cholesky: tf.Tensor,
) -> tf.Tensor:
    """Evaluate one Gaussian density per row without changing its measure."""

    points = tf.convert_to_tensor(points, DTYPE)
    means = tf.convert_to_tensor(means, DTYPE)
    cholesky = tf.convert_to_tensor(cholesky, DTYPE)
    if points.shape.rank != 2 or means.shape != points.shape:
        raise ValueError("points and means must have equal shape [N,D]")
    if cholesky.shape != (points.shape[0], points.shape[1], points.shape[1]):
        raise ValueError("cholesky must have shape [N,D,D]")
    solved = tf.linalg.triangular_solve(
        cholesky, (points - means)[:, :, tf.newaxis], lower=True
    )[:, :, 0]
    dim = tf.cast(tf.shape(points)[1], DTYPE)
    logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky)), axis=1)
    return -0.5 * (
        dim * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(solved), axis=1)
    ) - logdet


def complete_gaussian_mixture_log_density(
    points: tf.Tensor,
    means: tf.Tensor,
    cholesky: tf.Tensor,
    component_weights: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the complete K-component Gaussian mixture density.

    Inputs have shapes ``points=[N,D]``, ``means=[N,K,D]``,
    ``cholesky=[N,K,D,D]``, and ``component_weights=[N,K]``.  The returned
    density sums every component; it never uses a selected-component shortcut.
    """

    points = tf.convert_to_tensor(points, DTYPE)
    means = tf.convert_to_tensor(means, DTYPE)
    cholesky = tf.convert_to_tensor(cholesky, DTYPE)
    weights = tf.convert_to_tensor(component_weights, DTYPE)
    if points.shape.rank != 2 or means.shape.rank != 3:
        raise ValueError("points and means have invalid ranks")
    if (
        means.shape[0] != points.shape[0]
        or cholesky.shape != (points.shape[0], means.shape[1], points.shape[1], points.shape[1])
        or weights.shape != (points.shape[0], means.shape[1])
    ):
        raise ValueError("complete mixture shapes are inconsistent")
    tf.debugging.assert_all_finite(points, "mixture points must be finite")
    tf.debugging.assert_all_finite(means, "mixture means must be finite")
    tf.debugging.assert_all_finite(cholesky, "mixture factors must be finite")
    tf.debugging.assert_all_finite(weights, "mixture weights must be finite")
    tf.debugging.assert_positive(weights, "mixture weights must be positive")
    tf.debugging.assert_near(
        tf.reduce_sum(weights, axis=1),
        tf.ones([points.shape[0]], DTYPE),
        atol=tf.constant(1.0e-12, DTYPE),
        rtol=tf.constant(1.0e-12, DTYPE),
        message="mixture weights must sum to one",
    )
    difference = points[:, tf.newaxis, :] - means
    solved = tf.linalg.triangular_solve(
        cholesky, difference[:, :, :, tf.newaxis], lower=True
    )[:, :, :, 0]
    dim = tf.cast(tf.shape(points)[1], DTYPE)
    logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky)), axis=2)
    component_log_density = -0.5 * (
        dim * tf.constant(math.log(2.0 * math.pi), DTYPE)
        + tf.reduce_sum(tf.square(solved), axis=2)
    ) - logdet
    return tf.reduce_logsumexp(
        component_log_density + tf.math.log(weights), axis=1
    )


def compile_k1_apf_proposal(
    ukf_kernel,
    *,
    prior_means: tf.Tensor,
    prior_covariances: tf.Tensor,
    process_covariances: tf.Tensor,
    observation: tf.Tensor,
    observation_covariances: tf.Tensor,
    log_parent_weights: tf.Tensor,
    seed: Sequence[int] = (20260903, 1),
) -> Mapping[str, tf.Tensor]:
    """Compile one K=1 APF draw from a UKF-conditioned Gaussian bank."""

    prior_means = tf.convert_to_tensor(prior_means, DTYPE)
    if prior_means.shape.rank != 2 or prior_means.shape[0] is None:
        raise ValueError("prior_means must have static shape [N,D]")
    count = int(prior_means.shape[0])
    if len(tuple(seed)) != 2:
        raise ValueError("stateless APF seed must contain two integers")
    ukf = ukf_kernel(
        prior_means,
        tf.convert_to_tensor(prior_covariances, DTYPE),
        tf.convert_to_tensor(process_covariances, DTYPE),
        tf.convert_to_tensor(observation, DTYPE),
        tf.convert_to_tensor(observation_covariances, DTYPE),
    )
    require_valid_ukf_result(ukf)
    log_ancestor = apf_log_ancestor_probabilities(
        log_parent_weights, ukf["innovation_log_likelihood"]
    )
    ancestor = tf.random.stateless_categorical(
        log_ancestor[tf.newaxis, :],
        num_samples=count,
        seed=tf.convert_to_tensor(tuple(int(value) for value in seed), tf.int32),
        dtype=tf.int32,
    )[0]
    selected_mean = tf.gather(ukf["posterior_mean"], ancestor)
    selected_cholesky = tf.gather(ukf["posterior_cholesky"], ancestor)
    normal = tf.random.stateless_normal(
        [count, int(prior_means.shape[1])],
        seed=tf.convert_to_tensor(
            (int(seed[0]), int(seed[1]) + 1), tf.int32
        ),
        dtype=DTYPE,
    )
    samples = selected_mean + tf.einsum("bij,bj->bi", selected_cholesky, normal)
    log_q = gaussian_log_density(samples, selected_mean, selected_cholesky)
    return {
        "ancestor_indices": ancestor,
        "samples": samples,
        "selected_mean": selected_mean,
        "selected_cholesky": selected_cholesky,
        "selected_log_q": log_q,
        "log_ancestor_probabilities": log_ancestor,
        "lookahead_log_likelihood": ukf["innovation_log_likelihood"],
        "ukf": ukf,
        "finite": tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(samples, [-1]),
                        tf.reshape(log_q, [-1]),
                        tf.reshape(log_ancestor, [-1]),
                    ],
                    axis=0,
                )
            )
        ),
    }


def make_k1_apf_sampler(
    *,
    batch_size: int,
    state_dim: int,
    jit_compile: bool = True,
):
    """Create a fixed-shape compiled sampler for a K=1 UKF/APF step.

    The rows of ``posterior_means`` and ``posterior_covariances`` are indexed
    by the candidate ancestors.  The sampler first draws ancestors from the
    APF lookahead law and then draws one state from the selected Gaussian.  It
    returns the selected Gaussian density, which is the complete proposal
    density for K=1, together with the selected covariance for the next
    recursive UKF step.
    """

    batch_size = int(batch_size)
    state_dim = int(state_dim)
    if batch_size < 2 or state_dim < 1:
        raise ValueError("batch_size must be at least two and state_dim positive")

    @tf.function(
        input_signature=[
            tf.TensorSpec([batch_size, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([2], tf.int32),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def sampler(
        posterior_means: tf.Tensor,
        posterior_covariances: tf.Tensor,
        posterior_cholesky: tf.Tensor,
        lookahead_log_likelihood: tf.Tensor,
        log_parent_weights: tf.Tensor,
        seed: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        posterior_means = tf.ensure_shape(
            posterior_means, [batch_size, state_dim]
        )
        posterior_covariances = tf.ensure_shape(
            posterior_covariances, [batch_size, state_dim, state_dim]
        )
        posterior_cholesky = tf.ensure_shape(
            posterior_cholesky, [batch_size, state_dim, state_dim]
        )
        lookahead_log_likelihood = tf.ensure_shape(
            lookahead_log_likelihood, [batch_size]
        )
        log_parent_weights = tf.ensure_shape(log_parent_weights, [batch_size])
        seed = tf.ensure_shape(seed, [2])
        log_ancestor = tf.nn.log_softmax(
            log_parent_weights + lookahead_log_likelihood
        )
        ancestor = tf.random.stateless_categorical(
            log_ancestor[tf.newaxis, :],
            num_samples=batch_size,
            seed=seed,
            dtype=tf.int32,
        )[0]
        selected_mean = tf.gather(posterior_means, ancestor)
        selected_covariance = tf.gather(posterior_covariances, ancestor)
        selected_cholesky = tf.gather(posterior_cholesky, ancestor)
        normal = tf.random.stateless_normal(
            [batch_size, state_dim],
            seed=seed + tf.constant([0, 1], tf.int32),
            dtype=DTYPE,
        )
        samples = selected_mean + tf.einsum(
            "bij,bj->bi", selected_cholesky, normal
        )
        selected_log_q = gaussian_log_density(
            samples, selected_mean, selected_cholesky
        )
        finite = tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(log_ancestor, [-1]),
                        tf.reshape(samples, [-1]),
                        tf.reshape(selected_log_q, [-1]),
                        tf.reshape(selected_covariance, [-1]),
                    ],
                    axis=0,
                )
            )
        )
        return {
            "ancestor_indices": ancestor,
            "samples": samples,
            "selected_mean": selected_mean,
            "selected_covariance": selected_covariance,
            "next_covariances": selected_covariance,
            "selected_cholesky": selected_cholesky,
            "selected_log_q": selected_log_q,
            "complete_log_q": selected_log_q,
            "log_ancestor_probabilities": log_ancestor,
            "finite": finite,
        }

    return sampler


def make_k1_apf_sampler_from_random_inputs(
    *,
    batch_size: int,
    state_dim: int,
    jit_compile: bool = True,
):
    """Create a deterministic K=1 sampler with frozen random inputs.

    Supplying the categorical uniforms and standard-normal matrix explicitly
    makes eager, graph, and XLA executions evaluate the same finite random
    program.  This is the preferred route for a frozen analytical-gradient
    branch; backend-specific random kernels are kept out of the claim-bearing
    state construction.
    """

    batch_size = int(batch_size)
    state_dim = int(state_dim)
    if batch_size < 2 or state_dim < 1:
        raise ValueError("batch_size must be at least two and state_dim positive")

    @tf.function(
        input_signature=[
            tf.TensorSpec([batch_size, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size, state_dim], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def sampler(
        posterior_means: tf.Tensor,
        posterior_covariances: tf.Tensor,
        posterior_cholesky: tf.Tensor,
        lookahead_log_likelihood: tf.Tensor,
        log_parent_weights: tf.Tensor,
        categorical_uniforms: tf.Tensor,
        standard_normal: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        posterior_means = tf.ensure_shape(
            posterior_means, [batch_size, state_dim]
        )
        posterior_covariances = tf.ensure_shape(
            posterior_covariances, [batch_size, state_dim, state_dim]
        )
        posterior_cholesky = tf.ensure_shape(
            posterior_cholesky, [batch_size, state_dim, state_dim]
        )
        lookahead_log_likelihood = tf.ensure_shape(
            lookahead_log_likelihood, [batch_size]
        )
        log_parent_weights = tf.ensure_shape(log_parent_weights, [batch_size])
        categorical_uniforms = tf.ensure_shape(categorical_uniforms, [batch_size])
        standard_normal = tf.ensure_shape(
            standard_normal, [batch_size, state_dim]
        )
        log_ancestor = tf.nn.log_softmax(
            log_parent_weights + lookahead_log_likelihood
        )
        probabilities = tf.exp(log_ancestor)
        cdf = tf.math.cumsum(probabilities)
        cdf = tf.concat([cdf[:-1], tf.ones([1], DTYPE)], axis=0)
        uniforms = tf.clip_by_value(
            categorical_uniforms,
            tf.constant(0.0, DTYPE),
            tf.constant(1.0 - 1.0e-15, DTYPE),
        )
        ancestor = tf.searchsorted(
            cdf, uniforms, side="right", out_type=tf.int32
        )
        selected_mean = tf.gather(posterior_means, ancestor)
        selected_covariance = tf.gather(posterior_covariances, ancestor)
        selected_cholesky = tf.gather(posterior_cholesky, ancestor)
        samples = selected_mean + tf.einsum(
            "bij,bj->bi", selected_cholesky, standard_normal
        )
        selected_log_q = gaussian_log_density(
            samples, selected_mean, selected_cholesky
        )
        finite = tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(log_ancestor, [-1]),
                        tf.reshape(samples, [-1]),
                        tf.reshape(selected_log_q, [-1]),
                        tf.reshape(selected_covariance, [-1]),
                        tf.reshape(categorical_uniforms, [-1]),
                        tf.reshape(standard_normal, [-1]),
                    ],
                    axis=0,
                )
            )
        )
        return {
            "ancestor_indices": ancestor,
            "samples": samples,
            "selected_mean": selected_mean,
            "selected_covariance": selected_covariance,
            "next_covariances": selected_covariance,
            "selected_cholesky": selected_cholesky,
            "selected_log_q": selected_log_q,
            "complete_log_q": selected_log_q,
            "log_ancestor_probabilities": log_ancestor,
            "finite": finite,
        }

    return sampler


def make_k_mixture_apf_sampler_from_random_inputs(
    *,
    batch_size: int,
    state_dim: int,
    component_count: int,
    jit_compile: bool = True,
):
    """Create a fixed-shape APF sampler for a Gaussian UKF mixture.

    Ancestors are selected from the APF lookahead law, then a component is
    selected from that ancestor's frozen mixture.  The returned
    ``complete_log_q`` evaluates every component of the selected ancestor,
    including components that were not sampled.  Posterior covariance, rather
    than component covariance, is carried to the next UKF step so the split
    preserves the declared moment lifecycle.
    """

    batch_size = int(batch_size)
    state_dim = int(state_dim)
    component_count = int(component_count)
    if batch_size < 2 or state_dim < 1 or component_count < 2:
        raise ValueError(
            "batch_size must be at least two, state_dim positive, and component_count at least two"
        )

    @tf.function(
        input_signature=[
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size, component_count, state_dim], DTYPE),
            tf.TensorSpec(
                [batch_size, component_count, state_dim, state_dim], DTYPE
            ),
            tf.TensorSpec([batch_size, component_count], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size, state_dim], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def sampler(
        posterior_covariances: tf.Tensor,
        component_means: tf.Tensor,
        component_cholesky: tf.Tensor,
        component_weights: tf.Tensor,
        lookahead_log_likelihood: tf.Tensor,
        log_parent_weights: tf.Tensor,
        ancestor_uniforms: tf.Tensor,
        component_uniforms: tf.Tensor,
        standard_normal: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        posterior_covariances = tf.ensure_shape(
            posterior_covariances, [batch_size, state_dim, state_dim]
        )
        component_means = tf.ensure_shape(
            component_means, [batch_size, component_count, state_dim]
        )
        component_cholesky = tf.ensure_shape(
            component_cholesky,
            [batch_size, component_count, state_dim, state_dim],
        )
        component_weights = tf.ensure_shape(
            component_weights, [batch_size, component_count]
        )
        lookahead_log_likelihood = tf.ensure_shape(
            lookahead_log_likelihood, [batch_size]
        )
        log_parent_weights = tf.ensure_shape(log_parent_weights, [batch_size])
        ancestor_uniforms = tf.ensure_shape(ancestor_uniforms, [batch_size])
        component_uniforms = tf.ensure_shape(component_uniforms, [batch_size])
        standard_normal = tf.ensure_shape(
            standard_normal, [batch_size, state_dim]
        )
        tf.debugging.assert_all_finite(
            component_means, "mixture component means must be finite"
        )
        tf.debugging.assert_all_finite(
            component_cholesky, "mixture component factors must be finite"
        )
        tf.debugging.assert_all_finite(
            component_weights, "mixture component weights must be finite"
        )
        tf.debugging.assert_positive(
            component_weights, "mixture component weights must be positive"
        )
        tf.debugging.assert_near(
            tf.reduce_sum(component_weights, axis=1),
            tf.ones([batch_size], DTYPE),
            atol=tf.constant(1.0e-12, DTYPE),
            rtol=tf.constant(1.0e-12, DTYPE),
            message="mixture component weights must sum to one",
        )
        log_ancestor = tf.nn.log_softmax(
            log_parent_weights + lookahead_log_likelihood
        )
        ancestor_cdf = tf.math.cumsum(tf.exp(log_ancestor))
        ancestor_cdf = tf.concat(
            [ancestor_cdf[:-1], tf.ones([1], DTYPE)], axis=0
        )
        ancestor_uniforms = tf.clip_by_value(
            ancestor_uniforms,
            tf.constant(0.0, DTYPE),
            tf.constant(1.0 - 1.0e-15, DTYPE),
        )
        ancestor = tf.searchsorted(
            ancestor_cdf, ancestor_uniforms, side="right", out_type=tf.int32
        )
        selected_means = tf.gather(component_means, ancestor)
        selected_cholesky = tf.gather(component_cholesky, ancestor)
        selected_weights = tf.gather(component_weights, ancestor)
        selected_covariance = tf.gather(posterior_covariances, ancestor)
        component_cdf = tf.math.cumsum(selected_weights, axis=1)
        component_uniforms = tf.clip_by_value(
            component_uniforms,
            tf.constant(0.0, DTYPE),
            tf.constant(1.0 - 1.0e-15, DTYPE),
        )
        component_index = tf.reduce_sum(
            tf.cast(
                component_uniforms[:, tf.newaxis] >= component_cdf,
                tf.int32,
            ),
            axis=1,
        )
        component_index = tf.minimum(
            component_index, tf.constant(component_count - 1, tf.int32)
        )
        selected_mean = tf.gather(
            selected_means, component_index, axis=1, batch_dims=1
        )
        selected_chol = tf.gather(
            selected_cholesky, component_index, axis=1, batch_dims=1
        )
        samples = selected_mean + tf.einsum(
            "bij,bj->bi", selected_chol, standard_normal
        )
        complete_log_q = complete_gaussian_mixture_log_density(
            samples, selected_means, selected_cholesky, selected_weights
        )
        finite = tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(log_ancestor, [-1]),
                        tf.reshape(tf.cast(component_index, DTYPE), [-1]),
                        tf.reshape(samples, [-1]),
                        tf.reshape(complete_log_q, [-1]),
                        tf.reshape(selected_covariance, [-1]),
                    ],
                    axis=0,
                )
            )
        )
        return {
            "ancestor_indices": ancestor,
            "component_indices": component_index,
            "samples": samples,
            "selected_mean": selected_mean,
            "selected_cholesky": selected_chol,
            "selected_covariance": selected_covariance,
            "next_covariances": selected_covariance,
            "selected_log_q": complete_log_q,
            "complete_log_q": complete_log_q,
            "log_ancestor_probabilities": log_ancestor,
            "finite": finite,
        }

    return sampler


def student_log_density(
    points: tf.Tensor,
    means: tf.Tensor,
    cholesky: tf.Tensor,
    nu: float,
) -> tf.Tensor:
    """Evaluate a batched multivariate Student density from a scale factor.

    The scale factor is the scale, rather than the covariance.  Consequently
    a Student law with scale ``(nu - 2) / nu * P`` has covariance ``P``.
    Shapes are ``[N,D]``, ``[N,D]``, and ``[N,D,D]``.
    """

    points = tf.convert_to_tensor(points, DTYPE)
    means = tf.convert_to_tensor(means, DTYPE)
    cholesky = tf.convert_to_tensor(cholesky, DTYPE)
    if points.shape.rank != 2 or means.shape != points.shape:
        raise ValueError("Student points and means must have equal shape [N,D]")
    if cholesky.shape != (points.shape[0], points.shape[1], points.shape[1]):
        raise ValueError("Student cholesky must have shape [N,D,D]")
    nu = float(nu)
    if not math.isfinite(nu) or nu <= 2.0:
        raise ValueError("Student degrees of freedom must be finite and greater than two")
    difference = points - means
    solved = tf.linalg.triangular_solve(
        cholesky, difference[:, :, tf.newaxis], lower=True
    )[:, :, 0]
    dimension = tf.cast(tf.shape(points)[1], DTYPE)
    nu_tensor = tf.constant(nu, DTYPE)
    logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky)), axis=1)
    normalizer = (
        tf.math.lgamma(0.5 * (nu_tensor + dimension))
        - tf.math.lgamma(0.5 * nu_tensor)
        - 0.5 * dimension * tf.math.log(nu_tensor * tf.constant(math.pi, DTYPE))
        - logdet
    )
    quadratic = tf.reduce_sum(tf.square(solved), axis=1)
    return normalizer - 0.5 * (nu_tensor + dimension) * tf.math.log1p(
        quadratic / nu_tensor
    )


def make_gaussian_student_defensive_apf_sampler_from_random_inputs(
    *,
    batch_size: int,
    state_dim: int,
    local_component_count: int,
    nu: float,
    jit_compile: bool = True,
):
    """Create a fixed-shape APF sampler for a smooth Student defensive mix.

    The local proposal has ``local_component_count`` Gaussian components and
    is mixed with one full-support Student component.  ``epsilon`` is supplied
    as a per-ancestor tensor, so callers can construct a smooth sigmoid gate
    from UKF innovations while keeping the compiled sampler independent of the
    score parameter.  The returned density always sums all local and Student
    components, regardless of the sampled label.
    """

    batch_size = int(batch_size)
    state_dim = int(state_dim)
    local_component_count = int(local_component_count)
    nu = float(nu)
    if batch_size < 2 or state_dim < 1 or local_component_count < 1:
        raise ValueError(
            "batch_size must be at least two, state_dim positive, and local component count positive"
        )
    if not math.isfinite(nu) or nu <= 2.0:
        raise ValueError("Student degrees of freedom must be finite and greater than two")

    @tf.function(
        input_signature=[
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size, local_component_count, state_dim], DTYPE),
            tf.TensorSpec(
                [batch_size, local_component_count, state_dim, state_dim], DTYPE
            ),
            tf.TensorSpec([batch_size, local_component_count], DTYPE),
            tf.TensorSpec([batch_size, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim, state_dim], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
            tf.TensorSpec([batch_size, state_dim], DTYPE),
            tf.TensorSpec([batch_size, state_dim], DTYPE),
            tf.TensorSpec([batch_size], DTYPE),
        ],
        jit_compile=bool(jit_compile),
        autograph=False,
        reduce_retracing=True,
    )
    def sampler(
        posterior_covariances: tf.Tensor,
        local_means: tf.Tensor,
        local_cholesky: tf.Tensor,
        local_weights: tf.Tensor,
        defensive_means: tf.Tensor,
        defensive_cholesky: tf.Tensor,
        epsilon: tf.Tensor,
        lookahead_log_likelihood: tf.Tensor,
        log_parent_weights: tf.Tensor,
        ancestor_uniforms: tf.Tensor,
        component_uniforms: tf.Tensor,
        gaussian_normal: tf.Tensor,
        student_normal: tf.Tensor,
        student_chi_square: tf.Tensor,
    ) -> Mapping[str, tf.Tensor]:
        posterior_covariances = tf.ensure_shape(
            posterior_covariances, [batch_size, state_dim, state_dim]
        )
        local_means = tf.ensure_shape(
            local_means, [batch_size, local_component_count, state_dim]
        )
        local_cholesky = tf.ensure_shape(
            local_cholesky,
            [batch_size, local_component_count, state_dim, state_dim],
        )
        local_weights = tf.ensure_shape(
            local_weights, [batch_size, local_component_count]
        )
        defensive_means = tf.ensure_shape(defensive_means, [batch_size, state_dim])
        defensive_cholesky = tf.ensure_shape(
            defensive_cholesky, [batch_size, state_dim, state_dim]
        )
        epsilon = tf.ensure_shape(epsilon, [batch_size])
        lookahead_log_likelihood = tf.ensure_shape(
            lookahead_log_likelihood, [batch_size]
        )
        log_parent_weights = tf.ensure_shape(log_parent_weights, [batch_size])
        ancestor_uniforms = tf.ensure_shape(ancestor_uniforms, [batch_size])
        component_uniforms = tf.ensure_shape(component_uniforms, [batch_size])
        gaussian_normal = tf.ensure_shape(gaussian_normal, [batch_size, state_dim])
        student_normal = tf.ensure_shape(student_normal, [batch_size, state_dim])
        student_chi_square = tf.ensure_shape(student_chi_square, [batch_size])
        tf.debugging.assert_all_finite(
            local_means, "local defensive-mixture means must be finite"
        )
        tf.debugging.assert_all_finite(
            local_cholesky, "local defensive-mixture factors must be finite"
        )
        tf.debugging.assert_all_finite(
            defensive_means, "Student defensive means must be finite"
        )
        tf.debugging.assert_all_finite(
            defensive_cholesky, "Student defensive factors must be finite"
        )
        tf.debugging.assert_all_finite(epsilon, "defensive fractions must be finite")
        tf.debugging.assert_greater(
            epsilon, tf.zeros([batch_size], DTYPE), message="defensive fractions must be positive"
        )
        tf.debugging.assert_less(
            epsilon, tf.ones([batch_size], DTYPE), message="defensive fractions must be below one"
        )
        tf.debugging.assert_positive(
            local_weights, "local mixture weights must be positive"
        )
        tf.debugging.assert_near(
            tf.reduce_sum(local_weights, axis=1),
            tf.ones([batch_size], DTYPE),
            atol=tf.constant(1.0e-12, DTYPE),
            rtol=tf.constant(1.0e-12, DTYPE),
            message="local mixture weights must sum to one",
        )
        tf.debugging.assert_positive(
            student_chi_square, "Student chi-square variates must be positive"
        )

        log_ancestor = tf.nn.log_softmax(
            log_parent_weights + lookahead_log_likelihood
        )
        ancestor_cdf = tf.math.cumsum(tf.exp(log_ancestor))
        ancestor_cdf = tf.concat(
            [ancestor_cdf[:-1], tf.ones([1], DTYPE)], axis=0
        )
        ancestor_uniforms = tf.clip_by_value(
            ancestor_uniforms,
            tf.constant(0.0, DTYPE),
            tf.constant(1.0 - 1.0e-15, DTYPE),
        )
        ancestor = tf.searchsorted(
            ancestor_cdf, ancestor_uniforms, side="right", out_type=tf.int32
        )
        selected_local_means = tf.gather(local_means, ancestor)
        selected_local_cholesky = tf.gather(local_cholesky, ancestor)
        selected_local_weights = tf.gather(local_weights, ancestor)
        selected_defensive_mean = tf.gather(defensive_means, ancestor)
        selected_defensive_cholesky = tf.gather(defensive_cholesky, ancestor)
        selected_epsilon = tf.gather(epsilon, ancestor)
        selected_covariance = tf.gather(posterior_covariances, ancestor)
        combined_weights = tf.concat(
            [
                (1.0 - selected_epsilon)[:, tf.newaxis] * selected_local_weights,
                selected_epsilon[:, tf.newaxis],
            ],
            axis=1,
        )
        combined_cdf = tf.math.cumsum(combined_weights, axis=1)
        component_uniforms = tf.clip_by_value(
            component_uniforms,
            tf.constant(0.0, DTYPE),
            tf.constant(1.0 - 1.0e-15, DTYPE),
        )
        component_index = tf.reduce_sum(
            tf.cast(
                component_uniforms[:, tf.newaxis] >= combined_cdf,
                tf.int32,
            ),
            axis=1,
        )
        component_index = tf.minimum(
            component_index,
            tf.constant(local_component_count, tf.int32),
        )
        local_index = tf.minimum(
            component_index, tf.constant(local_component_count - 1, tf.int32)
        )
        selected_local_mean = tf.gather(
            selected_local_means, local_index, axis=1, batch_dims=1
        )
        selected_local_chol = tf.gather(
            selected_local_cholesky, local_index, axis=1, batch_dims=1
        )
        gaussian_sample = selected_local_mean + tf.einsum(
            "bij,bj->bi", selected_local_chol, gaussian_normal
        )
        student_whitened = student_normal / tf.sqrt(
            student_chi_square[:, tf.newaxis] / tf.constant(nu, DTYPE)
        )
        student_sample = selected_defensive_mean + tf.einsum(
            "bij,bj->bi", selected_defensive_cholesky, student_whitened
        )
        is_student = component_index == tf.constant(local_component_count, tf.int32)
        samples = tf.where(is_student[:, tf.newaxis], student_sample, gaussian_sample)

        difference = samples[:, tf.newaxis, :] - selected_local_means
        solved = tf.linalg.triangular_solve(
            selected_local_cholesky,
            difference[:, :, :, tf.newaxis],
            lower=True,
        )[:, :, :, 0]
        dimension = tf.cast(state_dim, DTYPE)
        local_logdet = tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(selected_local_cholesky)), axis=2
        )
        local_component_log_density = -0.5 * (
            dimension * tf.constant(math.log(2.0 * math.pi), DTYPE)
            + tf.reduce_sum(tf.square(solved), axis=2)
        ) - local_logdet
        student_log_q = student_log_density(
            samples, selected_defensive_mean, selected_defensive_cholesky, nu
        )
        complete_log_q = tf.reduce_logsumexp(
            tf.concat(
                [
                    local_component_log_density
                    + tf.math.log((1.0 - selected_epsilon)[:, tf.newaxis])
                    + tf.math.log(selected_local_weights),
                    (student_log_q + tf.math.log(selected_epsilon))[:, tf.newaxis],
                ],
                axis=1,
            ),
            axis=1,
        )
        finite = tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    [
                        tf.reshape(log_ancestor, [-1]),
                        tf.reshape(tf.cast(component_index, DTYPE), [-1]),
                        tf.reshape(samples, [-1]),
                        tf.reshape(complete_log_q, [-1]),
                        tf.reshape(student_log_q, [-1]),
                        tf.reshape(selected_covariance, [-1]),
                        tf.reshape(combined_weights, [-1]),
                    ],
                    axis=0,
                )
            )
        )
        return {
            "ancestor_indices": ancestor,
            "component_indices": component_index,
            "student_selected": is_student,
            "samples": samples,
            "selected_mean": tf.where(
                is_student[:, tf.newaxis], selected_defensive_mean, selected_local_mean
            ),
            "selected_cholesky": tf.where(
                is_student[:, tf.newaxis, tf.newaxis],
                selected_defensive_cholesky,
                selected_local_chol,
            ),
            "selected_covariance": selected_covariance,
            "next_covariances": selected_covariance,
            "selected_log_q": complete_log_q,
            "complete_log_q": complete_log_q,
            "student_log_q": student_log_q,
            "log_ancestor_probabilities": log_ancestor,
            "selected_epsilon": selected_epsilon,
            "combined_weights": combined_weights,
            "finite": finite,
        }

    return sampler


def sample_k1_apf_step(
    *,
    posterior_means: tf.Tensor,
    posterior_covariances: tf.Tensor,
    posterior_cholesky: tf.Tensor,
    lookahead_log_likelihood: tf.Tensor,
    log_parent_weights: tf.Tensor,
    seed: Sequence[int] = (20260903, 1),
    jit_compile: bool = True,
) -> Mapping[str, tf.Tensor]:
    """Draw one K=1 APF step using the generic compiled sampler."""

    means = tf.convert_to_tensor(posterior_means, DTYPE)
    if means.shape.rank != 2 or means.shape[0] is None:
        raise ValueError("posterior_means must have static shape [N,D]")
    seed_tuple = tuple(int(value) for value in seed)
    if len(seed_tuple) != 2:
        raise ValueError("stateless APF seed must contain two integers")
    sampler = make_k1_apf_sampler(
        batch_size=int(means.shape[0]),
        state_dim=int(means.shape[1]),
        jit_compile=bool(jit_compile),
    )
    result = sampler(
        means,
        tf.convert_to_tensor(posterior_covariances, DTYPE),
        tf.convert_to_tensor(posterior_cholesky, DTYPE),
        tf.convert_to_tensor(lookahead_log_likelihood, DTYPE),
        tf.convert_to_tensor(log_parent_weights, DTYPE),
        tf.convert_to_tensor(seed_tuple, tf.int32),
    )
    if not bool(tf.convert_to_tensor(result["finite"]).numpy()):
        raise ValueError("K=1 APF sampler produced a non-finite result")
    log_a = tf.convert_to_tensor(result["log_ancestor_probabilities"], DTYPE)
    normalization_error = tf.abs(tf.reduce_logsumexp(log_a))
    if float(normalization_error.numpy()) > 1.0e-10:
        raise ValueError("K=1 APF ancestor probabilities are not normalized")
    return result


__all__ = [
    "BatchedUKFConfig",
    "DTYPE",
    "ROUTE_CLASSIFICATION",
    "ROUTE_ID",
    "apf_log_ancestor_probabilities",
    "complete_gaussian_mixture_log_density",
    "compile_k1_apf_proposal",
    "gaussian_log_density",
    "make_k1_apf_sampler",
    "make_k1_apf_sampler_from_random_inputs",
    "make_k_mixture_apf_sampler_from_random_inputs",
    "make_gaussian_student_defensive_apf_sampler_from_random_inputs",
    "make_batched_ukf_kernel",
    "require_valid_ukf_result",
    "sample_k1_apf_step",
    "student_log_density",
]
