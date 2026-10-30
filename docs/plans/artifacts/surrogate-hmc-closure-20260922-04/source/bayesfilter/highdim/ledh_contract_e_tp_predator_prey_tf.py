"""Experimental Contract E--TP recursion for the predator--prey fixture."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from bayesfilter.highdim.models import PredatorPreySSM, p30_predator_prey_fixture_model


DTYPE = tf.float64
ALGORITHM_ID = tp.ALGORITHM_ID
FEATURE_NAMES = (
    "mass",
    "prey",
    "predator",
    "prey_square",
    "prey_predator",
    "predator_square",
    "stabilized_target_continuation_likelihood",
)
FEATURE_COUNT = len(FEATURE_NAMES)


def _product_rule(nodes: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    nodes = tf.reshape(tf.convert_to_tensor(nodes, DTYPE), [-1])
    weights = tf.reshape(tf.convert_to_tensor(weights, DTYPE), [-1])
    first, second = tf.meshgrid(nodes, nodes, indexing="ij")
    first_weight, second_weight = tf.meshgrid(weights, weights, indexing="ij")
    return (
        tf.stack([tf.reshape(first, [-1]), tf.reshape(second, [-1])], axis=1),
        tf.reshape(first_weight * second_weight, [-1]),
    )


def initial_rule(
    model: PredatorPreySSM, standard_nodes: tf.Tensor, standard_weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    standard_points, product_weights = _product_rule(standard_nodes, standard_weights)
    chol = tf.linalg.cholesky(model.initial_covariance)
    points = model.initial_mean[None, :] + tf.linalg.matmul(
        standard_points, chol, transpose_b=True
    )
    return points, tf.math.log(product_weights), standard_points, tf.math.log(product_weights)


def _gaussian_log_density(
    residual: tf.Tensor, covariance: tf.Tensor
) -> tf.Tensor:
    residual = tf.convert_to_tensor(residual, DTYPE)
    covariance = tf.convert_to_tensor(covariance, DTYPE)
    chol = tf.linalg.cholesky(covariance)
    solved = tf.linalg.triangular_solve(chol, tf.transpose(residual))
    quadratic = tf.reduce_sum(tf.square(solved), axis=0)
    logdet = tf.constant(2.0, DTYPE) * tf.reduce_sum(
        tf.math.log(tf.linalg.diag_part(chol))
    )
    dimension = tf.cast(tf.shape(residual)[1], DTYPE)
    return -0.5 * (
        dimension * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
        + logdet
        + quadratic
    )


def _affine_ledh_flow(
    pre_flow: tf.Tensor,
    prior_mean: tf.Tensor,
    prior_covariance: tf.Tensor,
    observation: tf.Tensor,
    observation_covariance: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    dimension = int(prior_covariance.shape[0])
    prior_precision = tf.linalg.inv(prior_covariance)
    observation_precision = tf.linalg.inv(observation_covariance)
    posterior_covariance = tf.linalg.inv(prior_precision + observation_precision)
    posterior_mean = tf.linalg.matmul(prior_mean, prior_precision, transpose_b=True)
    posterior_mean += tf.linalg.matvec(observation_precision, observation)[None, :]
    posterior_mean = tf.linalg.matmul(
        posterior_mean, posterior_covariance, transpose_b=True
    )
    prior_chol = tf.linalg.cholesky(prior_covariance)
    posterior_chol = tf.linalg.cholesky(posterior_covariance)
    affine = tf.linalg.matmul(
        posterior_chol,
        tf.linalg.inv(prior_chol),
    )
    post_flow = posterior_mean + tf.linalg.matmul(
        pre_flow - prior_mean, affine, transpose_b=True
    )
    proposal_log_density = _gaussian_log_density(
        pre_flow - prior_mean, prior_covariance
    )
    forward_log_det = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(posterior_chol)))
    forward_log_det -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(prior_chol)))
    return post_flow, proposal_log_density, forward_log_det


def _teacher_step(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    parents: tf.Tensor,
    parent_log_weights: tf.Tensor,
    standard_points: tf.Tensor,
    standard_log_weights: tf.Tensor,
    observation: tf.Tensor,
    time_index: int,
) -> dict[str, tf.Tensor]:
    if time_index == 0:
        prior_mean = tf.repeat(
            model.initial_mean[None, :], tf.shape(standard_points)[0], axis=0
        )
        prior_covariance = model.initial_covariance
        chol = tf.linalg.cholesky(prior_covariance)
        pre_flow = prior_mean + tf.linalg.matmul(
            standard_points, chol, transpose_b=True
        )
        base_log_weights = standard_log_weights
        previous = None
    else:
        parent_count = tf.shape(parents)[0]
        innovation_count = tf.shape(standard_points)[0]
        previous = tf.repeat(parents, innovation_count, axis=0)
        prior_mean = model.transition_mean(theta, previous)
        prior_covariance = model.process_covariance
        chol = tf.linalg.cholesky(prior_covariance)
        pre_flow = prior_mean + tf.linalg.matmul(
            tf.tile(standard_points, [parent_count, 1]), chol, transpose_b=True
        )
        base_log_weights = (
            tf.repeat(parent_log_weights, innovation_count)
            + tf.tile(standard_log_weights, [parent_count])
        )
    particles, proposal_log_density, forward_log_det = _affine_ledh_flow(
        pre_flow,
        prior_mean,
        prior_covariance,
        tf.reshape(tf.convert_to_tensor(observation, DTYPE), [2]),
        model.observation_covariance,
    )
    if time_index == 0:
        target_transition = model.initial_log_density(theta, particles)
    else:
        target_transition = model.transition_log_density(
            theta, previous, particles, t=time_index
        )
    target_observation = model.observation_log_density(
        theta, particles, observation, t=time_index
    )
    log_weights = (
        base_log_weights
        + target_transition
        + target_observation
        - proposal_log_density
        + forward_log_det
    )
    return {
        "particles": particles,
        "log_unnormalized_weights": log_weights,
        "increment": tf.reduce_logsumexp(log_weights),
    }


def _pairwise_transition(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    previous_points: tf.Tensor,
    next_points: tf.Tensor,
    time_index: int,
) -> tf.Tensor:
    previous = tf.convert_to_tensor(previous_points, DTYPE)
    following = tf.convert_to_tensor(next_points, DTYPE)
    del time_index
    means = model.transition_mean(theta, previous)
    residual = following[None, :, :] - means[:, None, :]
    flat = tf.reshape(residual, [-1, 2])
    return tf.reshape(
        _gaussian_log_density(flat, model.process_covariance),
        [tf.shape(previous)[0], tf.shape(following)[0]],
    )


def target_continuation_log_likelihood(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_observations: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: int,
) -> tf.Tensor:
    observations = tf.convert_to_tensor(future_observations, DTYPE)
    grid = tf.convert_to_tensor(grid_points, DTYPE)
    log_grid_weights = tf.math.log(tf.convert_to_tensor(grid_weights, DTYPE))
    future_count = observations.shape[0]
    if future_count is None or future_count < 1:
        return tf.zeros([tf.shape(points)[0]], DTYPE)
    child_log = tf.zeros([tf.shape(grid)[0]], DTYPE)
    for local_index in range(future_count - 1, 0, -1):
        absolute_time = first_future_time_index + local_index
        transition = _pairwise_transition(
            model, theta, grid, grid, absolute_time
        )
        observation_log = model.observation_log_density(
            theta, grid, observations[local_index], t=absolute_time
        )
        child_log = tf.reduce_logsumexp(
            transition
            + log_grid_weights[None, :]
            + observation_log[None, :]
            + child_log[None, :],
            axis=1,
        )
    transition = _pairwise_transition(
        model, theta, points, grid, first_future_time_index
    )
    observation_log = model.observation_log_density(
        theta, grid, observations[0], t=first_future_time_index
    )
    return tf.reduce_logsumexp(
        transition
        + log_grid_weights[None, :]
        + observation_log[None, :]
        + child_log[None, :],
        axis=1,
    )


def one_step_target_continuation_log_likelihood(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    points: tf.Tensor,
    next_observation: tf.Tensor,
) -> tf.Tensor:
    """Return the analytic one-step predictive observation likelihood."""

    means = model.transition_mean(theta, points)
    covariance = model.process_covariance + model.observation_covariance
    observation = tf.reshape(tf.convert_to_tensor(next_observation, DTYPE), [2])
    return _gaussian_log_density(observation[None, :] - means, covariance)


def gaussian_closure_continuation_log_likelihood(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_observations: tf.Tensor,
    standard_points: tf.Tensor,
    standard_weights: tf.Tensor,
) -> tf.Tensor:
    """Approximate a future likelihood by fixed Gaussian quadrature filtering."""

    means = tf.convert_to_tensor(points, DTYPE)
    observations = tf.convert_to_tensor(future_observations, DTYPE)
    standard_points = tf.convert_to_tensor(standard_points, DTYPE)
    weights = tf.convert_to_tensor(standard_weights, DTYPE)
    covariance = tf.zeros([tf.shape(means)[0], 2, 2], DTYPE)
    values = tf.zeros([tf.shape(means)[0]], DTYPE)
    for local_index, observation in enumerate(tf.unstack(observations, axis=0)):
        if local_index == 0:
            predicted_mean = model.transition_mean(theta, means)
            predicted_covariance = tf.broadcast_to(
                model.process_covariance[None, :, :],
                [tf.shape(means)[0], 2, 2],
            )
        else:
            chol = tf.linalg.cholesky(covariance)
            sigma_points = means[:, None, :] + tf.einsum(
                "md,ned->nme", standard_points, chol
            )
            transitioned = tf.reshape(
                model.transition_mean(theta, tf.reshape(sigma_points, [-1, 2])),
                [tf.shape(means)[0], tf.shape(standard_points)[0], 2],
            )
            predicted_mean = tf.einsum("m,nmd->nd", weights, transitioned)
            centered = transitioned - predicted_mean[:, None, :]
            predicted_covariance = (
                tf.einsum("m,nmi,nmj->nij", weights, centered, centered)
                + model.process_covariance[None, :, :]
            )
        innovation_covariance = (
            predicted_covariance + model.observation_covariance[None, :, :]
        )
        residual = observation[None, :] - predicted_mean
        innovation_chol = tf.linalg.cholesky(innovation_covariance)
        solved = tf.linalg.triangular_solve(
            innovation_chol, residual[:, :, None]
        )[:, :, 0]
        quadratic = tf.reduce_sum(tf.square(solved), axis=1)
        logdet = tf.constant(2.0, DTYPE) * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_chol)), axis=1
        )
        values += -0.5 * (
            tf.constant(2.0, DTYPE)
            * tf.math.log(tf.constant(2.0 * 3.141592653589793, DTYPE))
            + logdet
            + quadratic
        )
        gain = tf.linalg.matmul(
            predicted_covariance,
            tf.linalg.inv(innovation_covariance),
        )
        means = predicted_mean + tf.linalg.matvec(gain, residual)
        covariance = predicted_covariance - tf.linalg.matmul(
            tf.linalg.matmul(gain, innovation_covariance), gain, transpose_b=True
        )
        covariance = 0.5 * (covariance + tf.linalg.matrix_transpose(covariance))
    return values


def _features(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_observations: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: int,
) -> tf.Tensor:
    prey, predator = tf.unstack(points, axis=1)
    if int(future_observations.shape[0]) == 1:
        continuation_log = one_step_target_continuation_log_likelihood(
            model, theta, points, future_observations[0]
        )
        reference_log = one_step_target_continuation_log_likelihood(
            model, theta, model.initial_mean[None, :], future_observations[0]
        )[0]
    else:
        continuation_log = gaussian_closure_continuation_log_likelihood(
            model,
            theta,
            points,
            future_observations,
            grid_points,
            grid_weights,
        )
        reference_log = gaussian_closure_continuation_log_likelihood(
            model,
            theta,
            model.initial_mean[None, :],
            future_observations,
            grid_points,
            grid_weights,
        )[0]
    common_reference = tf.maximum(reference_log, tf.reduce_max(continuation_log))
    return tf.stack(
        [
            tf.ones_like(prey),
            prey,
            predator,
            tf.square(prey),
            prey * predator,
            tf.square(predator),
            tf.exp(continuation_log - common_reference),
        ],
        axis=0,
    )


def contract_e_tp_predator_prey_recursive_core(
    theta: tf.Tensor,
    observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    continuation_grid_points: tf.Tensor,
    continuation_grid_weights: tf.Tensor,
    *,
    lookahead_steps: int,
) -> dict[str, tf.Tensor]:
    model = p30_predator_prey_fixture_model()
    theta = tf.reshape(tf.convert_to_tensor(theta, DTYPE), [6])
    observations = tf.reshape(tf.convert_to_tensor(observations, DTYPE), [-1, 2])
    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 1:
        raise ValueError("predator--prey recursion requires a static positive horizon")
    if active_indices.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError("active_indices shape mismatch")
    if row_scales.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError("row_scales shape mismatch")
    parents, parent_log_weights, standard_points, standard_log_weights = initial_rule(
        model, standard_nodes, standard_weights
    )
    # Preserve an explicit zero derivative for horizons before theta enters.
    total = tf.constant(0.0, DTYPE) * tf.reduce_sum(theta)
    increments = []
    minimum_weights = []
    condition_numbers = []
    residuals = []
    valid = []
    for time_index in range(time_steps):
        teacher = _teacher_step(
            model,
            theta,
            parents,
            parent_log_weights,
            standard_points,
            standard_log_weights,
            observations[time_index],
            time_index,
        )
        total += teacher["increment"]
        increments.append(teacher["increment"])
        if time_index + 1 == time_steps:
            valid.append(tf.constant(True))
            continue
        stop = min(time_steps, time_index + 1 + lookahead_steps)
        features = _features(
            model,
            theta,
            teacher["particles"],
            observations[time_index + 1 : stop],
            continuation_grid_points,
            continuation_grid_weights,
            first_future_time_index=time_index + 1,
        )
        projection = tp._contract_e_tp_dense_square_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active_indices[time_index],
            row_scales[time_index],
        )
        parents = projection["student_points"]
        parent_log_weights = tf.math.log(projection["student_weights"])
        minimum_weights.append(projection["minimum_weight"])
        condition_numbers.append(projection["condition_number"])
        residuals.append(projection["feature_residual"])
        valid.append(projection["valid_chart"])
    return {
        "objective": total,
        "increment_history": tf.stack(increments),
        "minimum_weight_history": tf.stack(minimum_weights) if minimum_weights else tf.zeros([0], DTYPE),
        "condition_number_history": tf.stack(condition_numbers) if condition_numbers else tf.zeros([0], DTYPE),
        "feature_residual_history": tf.stack(residuals) if residuals else tf.zeros([0, FEATURE_COUNT], DTYPE),
        "valid_history": tf.stack(valid),
    }


__all__ = [
    "ALGORITHM_ID",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "contract_e_tp_predator_prey_recursive_core",
    "gaussian_closure_continuation_log_likelihood",
    "initial_rule",
    "one_step_target_continuation_log_likelihood",
    "target_continuation_log_likelihood",
]
