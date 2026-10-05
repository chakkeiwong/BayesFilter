"""Experimental Contract E--TP recursion for the predator--prey fixture."""

from __future__ import annotations

from functools import lru_cache

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
            model.initial_mean[None, :], int(standard_points.shape[0]), axis=0
        )
        prior_covariance = model.initial_covariance
        chol = tf.linalg.cholesky(prior_covariance)
        pre_flow = prior_mean + tf.linalg.matmul(
            standard_points, chol, transpose_b=True
        )
        base_log_weights = standard_log_weights
        previous = None
    else:
        # These are fixed quadrature/chart extents, including in the loop adjoint.
        parent_count = int(parents.shape[0])
        innovation_count = int(standard_points.shape[0])
        parent_indices = tf.range(parent_count * innovation_count) // innovation_count
        previous = tf.gather(parents, parent_indices)
        prior_mean = model.transition_mean(theta, previous)
        prior_covariance = model.process_covariance
        chol = tf.linalg.cholesky(prior_covariance)
        pre_flow = prior_mean + tf.linalg.matmul(
            tf.tile(standard_points, [parent_count, 1]), chol, transpose_b=True
        )
        base_log_weights = (
            tf.gather(parent_log_weights, parent_indices)
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
    def backward(local_index, child_log):
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
        return local_index - 1, child_log

    _, child_log = tf.while_loop(lambda local_index, _: local_index > 0, backward,
        (tf.constant(future_count - 1), child_log), maximum_iterations=future_count - 1,
        parallel_iterations=1)
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
    *,
    future_count: tf.Tensor | None = None,
) -> tf.Tensor:
    """Approximate a future likelihood by fixed Gaussian quadrature filtering."""

    means = tf.convert_to_tensor(points, DTYPE)
    observations = tf.convert_to_tensor(future_observations, DTYPE)
    standard_points = tf.convert_to_tensor(standard_points, DTYPE)
    weights = tf.convert_to_tensor(standard_weights, DTYPE)
    values = tf.zeros([tf.shape(means)[0]], DTYPE)
    if observations.shape[0] == 0:
        return values
    count = tf.shape(observations)[0] if future_count is None else future_count

    def update(predicted_mean, predicted_covariance, observation, values):
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
        return means, covariance, values

    means, covariance, values = update(model.transition_mean(theta, means),
        tf.broadcast_to(model.process_covariance[None, :, :], [tf.shape(means)[0], 2, 2]),
        observations[0], values)

    def step(local_index, means, covariance, values):
        chol = tf.linalg.cholesky(covariance)
        sigma_points = means[:, None, :] + tf.einsum("md,ned->nme", standard_points, chol)
        transitioned = tf.reshape(
            model.transition_mean(theta, tf.reshape(sigma_points, [-1, 2])),
            [tf.shape(means)[0], tf.shape(standard_points)[0], 2])
        predicted_mean = tf.einsum("m,nmd->nd", weights, transitioned)
        centered = transitioned - predicted_mean[:, None, :]
        predicted_covariance = (tf.einsum("m,nmi,nmj->nij", weights, centered, centered)
                                + model.process_covariance[None, :, :])
        return (local_index + 1, *update(predicted_mean, predicted_covariance, observations[local_index], values))

    start = 1
    if observations.shape[0] > 1:
        # A fixed initial recurrence preserves the original straight-line XLA
        # arithmetic for two-step continuations without unrolling the horizon.
        _, next_means, next_covariance, next_values = step(tf.constant(1), means, covariance, values)
        means = tf.where(count > 1, next_means, means)
        covariance = tf.where(count > 1, next_covariance, covariance)
        values = tf.where(count > 1, next_values, values)
        start = 2
    if observations.shape[0] > start:
        _, _, _, values = tf.while_loop(lambda local_index, *_: local_index < count, step,
            (tf.constant(start), means, covariance, values), maximum_iterations=observations.shape[0] - start,
            parallel_iterations=1)
    return tf.where(count > 0, values, tf.zeros_like(values))


def _features(
    model: PredatorPreySSM,
    theta: tf.Tensor,
    points: tf.Tensor,
    future_observations: tf.Tensor,
    grid_points: tf.Tensor,
    grid_weights: tf.Tensor,
    *,
    first_future_time_index: int,
    future_count: tf.Tensor | None = None,
) -> tf.Tensor:
    prey, predator = tf.unstack(points, axis=1)
    count = tf.shape(future_observations)[0] if future_count is None else future_count

    def one_step():
        continuation_log = one_step_target_continuation_log_likelihood(
            model, theta, points, future_observations[0]
        )
        reference_log = one_step_target_continuation_log_likelihood(
            model, theta, model.initial_mean[None, :], future_observations[0]
        )[0]
        return continuation_log, reference_log

    def multiple_steps():
        continuation_log = gaussian_closure_continuation_log_likelihood(
            model,
            theta,
            points,
            future_observations,
            grid_points,
            grid_weights,
            future_count=count,
        )
        reference_log = gaussian_closure_continuation_log_likelihood(
            model,
            theta,
            model.initial_mean[None, :],
            future_observations,
            grid_points,
            grid_weights,
            future_count=count,
        )[0]
        return continuation_log, reference_log

    if future_observations.shape[0] == 0:
        continuation_log, reference_log = multiple_steps()
    elif future_observations.shape[0] == 1:
        continuation_log, reference_log = one_step()
    else:
        single, single_reference = one_step()
        multiple, multiple_reference = multiple_steps()
        continuation_log = tf.where(count == 1, single, multiple)
        reference_log = tf.where(count == 1, single_reference, multiple_reference)
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


@lru_cache(maxsize=16)
def _recursive_program(specifications, lookahead_steps):
    return tf.function(lambda *values: contract_e_tp_predator_prey_recursive_core(
        *values, lookahead_steps=lookahead_steps), input_signature=specifications,
        jit_compile=True, autograph=False)


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
    if tf.executing_eagerly():
        values = (tf.convert_to_tensor(theta, DTYPE), tf.convert_to_tensor(observations, DTYPE),
            tf.convert_to_tensor(standard_nodes, DTYPE), tf.convert_to_tensor(standard_weights, DTYPE),
            tf.convert_to_tensor(active_indices, tf.int32), tf.convert_to_tensor(row_scales, DTYPE),
            tf.convert_to_tensor(continuation_grid_points, DTYPE), tf.convert_to_tensor(continuation_grid_weights, DTYPE))
        specifications = tuple(tf.TensorSpec(value.shape, value.dtype) for value in values)
        return _recursive_program(specifications, int(lookahead_steps))(*values)
    with tf.init_scope():
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
    teacher = _teacher_step(model, theta, parents, parent_log_weights,
        standard_points, standard_log_weights, observations[0], 0)
    # The initial observation does not depend on theta in this model.
    total = tf.constant(0.0, DTYPE) * tf.reduce_sum(theta) + teacher["increment"]
    increments = tf.TensorArray(DTYPE, time_steps, element_shape=[]).write(0, teacher["increment"])
    minimum_weights = tf.TensorArray(DTYPE, time_steps - 1, element_shape=[])
    condition_numbers = tf.TensorArray(DTYPE, time_steps - 1, element_shape=[])
    residuals = tf.TensorArray(DTYPE, time_steps - 1, element_shape=[FEATURE_COUNT])
    valid = tf.TensorArray(tf.bool, time_steps, element_shape=[]).write(time_steps - 1, True)

    def project(teacher, time_index):
        window_size = min(time_steps, lookahead_steps)
        future_indices = tf.minimum(time_index + 1 + tf.range(window_size), time_steps - 1)
        future_count = tf.minimum(time_steps - time_index - 1, lookahead_steps)
        features = _features(
            model,
            theta,
            teacher["particles"],
            tf.gather(observations, future_indices),
            continuation_grid_points,
            continuation_grid_weights,
            first_future_time_index=time_index + 1,
            future_count=future_count,
        )
        projection = tp._contract_e_tp_dense_square_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            features,
            active_indices[time_index],
            row_scales[time_index],
        )
        return projection

    if time_steps > 1:
        first = project(teacher, tf.constant(0))
        parents = first["student_points"]
        parent_log_weights = tf.math.log(first["student_weights"])
        minimum_weights = minimum_weights.write(0, first["minimum_weight"])
        condition_numbers = condition_numbers.write(0, first["condition_number"])
        residuals = residuals.write(0, first["feature_residual"])
        valid = valid.write(0, first["valid_chart"])

        def step(time_index, parents, parent_log_weights, total, increments,
                 minimum_weights, condition_numbers, residuals, valid):
            teacher = _teacher_step(model, theta, parents, parent_log_weights,
                standard_points, standard_log_weights, observations[time_index], 1)

            result = project(teacher, time_index)
            return (time_index + 1, result["student_points"], tf.math.log(result["student_weights"]),
                total + teacher["increment"], increments.write(time_index, teacher["increment"]),
                minimum_weights.write(time_index, result["minimum_weight"]),
                condition_numbers.write(time_index, result["condition_number"]),
                residuals.write(time_index, result["feature_residual"]),
                valid.write(time_index, result["valid_chart"]))

        _, parents, parent_log_weights, total, increments, minimum_weights, condition_numbers, residuals, valid = tf.while_loop(
            lambda time_index, *_: time_index < time_steps - 1, step,
            (tf.constant(1), parents, parent_log_weights, total, increments,
             minimum_weights, condition_numbers, residuals, valid),
            maximum_iterations=time_steps - 2, parallel_iterations=1)
        terminal = _teacher_step(model, theta, parents, parent_log_weights,
            standard_points, standard_log_weights, observations[-1], 1)
        total += terminal["increment"]
        increments = increments.write(time_steps - 1, terminal["increment"])
    return {
        "objective": total,
        "increment_history": increments.stack(),
        "minimum_weight_history": minimum_weights.stack(),
        "condition_number_history": condition_numbers.stack(),
        "feature_residual_history": residuals.stack(),
        "valid_history": valid.stack(),
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
