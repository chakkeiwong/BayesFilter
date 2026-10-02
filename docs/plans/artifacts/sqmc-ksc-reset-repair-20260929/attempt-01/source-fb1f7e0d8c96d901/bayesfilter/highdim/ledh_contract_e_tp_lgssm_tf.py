"""Experimental recursive LGSSM adapter for Contract E--TP.

This module shares the frozen LGSSM target and corrected LEDH flow mathematics
with the canonical Contract E--Chol graph, but it has independent reset
semantics and an experimental identity.
"""

from __future__ import annotations

from collections.abc import Mapping

import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as lgssm
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


ALGORITHM_ID = tp.ALGORITHM_ID
FEATURE_NAMES = (
    "mass",
    "x1",
    "x2",
    "x3",
    "x1_sq",
    "x1_x2",
    "x1_x3",
    "x2_sq",
    "x2_x3",
    "x3_sq",
    "next_corrected_ledh_predictive_contribution",
)
FEATURE_COUNT = len(FEATURE_NAMES)
PROGRESSIVE_FEATURE_NAMES = FEATURE_NAMES + tuple(
    f"next_predictive_x_centered_target_model_score_{name}"
    for name in lgssm.PARAMETER_NAMES
)
PROGRESSIVE_FEATURE_COUNT = len(PROGRESSIVE_FEATURE_NAMES)
CONTINUATION_FEATURE_NAMES = FEATURE_NAMES[:-1] + (
    "stabilized_exact_lgssm_remaining_horizon_likelihood",
)
CONTINUATION_FEATURE_COUNT = len(CONTINUATION_FEATURE_NAMES)


def _tensor_product_rule(
    nodes: tf.Tensor, weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Form a three-dimensional product rule using TensorFlow only."""

    nodes = tf.reshape(tf.convert_to_tensor(nodes), [-1])
    weights = tf.reshape(tf.convert_to_tensor(weights, nodes.dtype), [-1])
    order = tf.size(nodes)
    first, second, third = tf.meshgrid(nodes, nodes, nodes, indexing="ij")
    first_weight, second_weight, third_weight = tf.meshgrid(
        weights, weights, weights, indexing="ij"
    )
    points = tf.stack(
        [tf.reshape(first, [-1]), tf.reshape(second, [-1]), tf.reshape(third, [-1])],
        axis=1,
    )
    product_weights = tf.reshape(first_weight * second_weight * third_weight, [order**3])
    return points, product_weights


def _flow_correction(
    theta: tf.Tensor,
    parents: tf.Tensor,
    innovations: tf.Tensor,
    observation: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Execute the same corrected LGSSM LEDH finite proposal as canonical code."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    dtype = theta.dtype
    parents = tf.convert_to_tensor(parents, dtype)
    innovations = tf.convert_to_tensor(innovations, dtype)
    parent_count = tf.shape(parents)[0]
    innovation_count = tf.shape(innovations)[0]
    components = lgssm._lgssm_components(theta, 1)
    prior_by_parent = tf.linalg.matmul(
        parents, components["transition_matrix"][0], transpose_b=True
    )
    prior_mean_grid = prior_by_parent[:, None, :] + tf.zeros_like(innovations)[
        None, :, :
    ]
    pre_flow_grid = (
        prior_by_parent[:, None, :]
        + components["q_scale"] * innovations[None, :, :]
    )
    prior_mean = tf.reshape(prior_mean_grid, [-1, lgssm.STATE_DIMENSION])
    pre_flow = tf.reshape(pre_flow_grid, [-1, lgssm.STATE_DIMENSION])
    flow = lgssm._lgssm_flow_forward_core(
        prior_mean[None, :, :],
        pre_flow[None, :, :],
        observation,
        components["transition_covariance"],
        components["observation_covariance"],
    )
    particles = flow["particles"][0]
    transition_density = lgssm._gaussian_log_density_forward_core(
        (particles - prior_mean)[None, :, :], components["transition_covariance"]
    )["value"][0]
    proposal_density = lgssm._gaussian_log_density_forward_core(
        (pre_flow - prior_mean)[None, :, :], components["transition_covariance"]
    )["value"][0]
    predicted_observation = tf.linalg.matmul(
        particles, components["observation_matrix"], transpose_b=True
    )
    observation_density = lgssm._gaussian_log_density_forward_core(
        (predicted_observation - observation[None, :])[None, :, :],
        components["observation_covariance"],
    )["value"][0]
    correction = (
        transition_density
        + observation_density
        - proposal_density
        + flow["forward_log_abs_det"][0]
    )
    return {
        "particles": particles,
        "log_correction": correction,
        "flow_valid": tf.reduce_all(flow["valid_chart"]),
        "parent_count": parent_count,
        "innovation_count": innovation_count,
    }


def _combine_parent_innovation_log_weights(
    parent_log_weights: tf.Tensor,
    innovation_log_weights: tf.Tensor,
    log_correction: tf.Tensor,
) -> tf.Tensor:
    """Combine Cartesian-product weights without dynamic repeat/tile gradients."""

    correction_grid = tf.reshape(
        log_correction,
        [tf.shape(parent_log_weights)[0], tf.shape(innovation_log_weights)[0]],
    )
    return tf.reshape(
        parent_log_weights[:, None]
        + innovation_log_weights[None, :]
        + correction_grid,
        [-1],
    )


def _next_predictive_contribution(
    theta: tf.Tensor,
    parents: tf.Tensor,
    innovations: tf.Tensor,
    log_innovation_weights: tf.Tensor,
    next_observation: tf.Tensor,
) -> tf.Tensor:
    next_flow = _flow_correction(theta, parents, innovations, next_observation)
    parent_count = tf.shape(parents)[0]
    innovation_count = tf.shape(innovations)[0]
    log_terms = tf.reshape(
        next_flow["log_correction"], [parent_count, innovation_count]
    ) + log_innovation_weights[None, :]
    return tf.exp(tf.reduce_logsumexp(log_terms, axis=1))


def _features(
    theta: tf.Tensor,
    points: tf.Tensor,
    innovations: tf.Tensor,
    log_innovation_weights: tf.Tensor,
    next_observation: tf.Tensor,
) -> tf.Tensor:
    x1, x2, x3 = tf.unstack(points, axis=1)
    next_contribution = _next_predictive_contribution(
        theta,
        points,
        innovations,
        log_innovation_weights,
        next_observation,
    )
    return tf.concat(
        [_state_moment_features(points), next_contribution[None, :]], axis=0
    )


def _initial_target_model_score_marks(
    theta: tf.Tensor, points: tf.Tensor
) -> tf.Tensor:
    """Stationary initial-density score evaluated at fixed state locations."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    points = tf.convert_to_tensor(points, theta.dtype)
    phi = theta[: lgssm.STATE_DIMENSION]
    q_scale = theta[3]
    variance = tf.square(q_scale) / (1.0 - tf.square(phi))
    standardized_square = tf.square(points) / variance[None, :]
    phi_score = (standardized_square - 1.0) * (
        phi / (1.0 - tf.square(phi))
    )[None, :]
    q_score = tf.reduce_sum(standardized_square - 1.0, axis=1) / q_scale
    return tf.concat(
        [
            phi_score,
            q_score[:, None],
            tf.zeros([tf.shape(points)[0], 1], theta.dtype),
        ],
        axis=1,
    )


def _target_transition_log_density_and_score(
    theta: tf.Tensor, children: tf.Tensor, parents: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return target-model transition log density and fixed-state score."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    children = tf.convert_to_tensor(children, theta.dtype)
    parents = tf.convert_to_tensor(parents, theta.dtype)
    phi = theta[: lgssm.STATE_DIMENSION]
    q_scale = theta[3]
    residual = children[:, None, :] - phi[None, None, :] * parents[None, :, :]
    inverse_variance = tf.math.reciprocal(tf.square(q_scale))
    log_density = -0.5 * (
        tf.cast(lgssm.STATE_DIMENSION, theta.dtype)
        * tf.math.log(tf.constant(2.0 * 3.141592653589793, theta.dtype))
        + tf.cast(lgssm.STATE_DIMENSION, theta.dtype) * tf.math.log(tf.square(q_scale))
        + tf.reduce_sum(tf.square(residual), axis=2) * inverse_variance
    )
    phi_score = residual * parents[None, :, :] * inverse_variance
    q_score = (
        tf.reduce_sum(tf.square(residual), axis=2) / tf.pow(q_scale, 3)
        - tf.cast(lgssm.STATE_DIMENSION, theta.dtype) / q_scale
    )
    score = tf.concat(
        [
            phi_score,
            q_score[:, :, None],
            tf.zeros(
                [tf.shape(children)[0], tf.shape(parents)[0], 1], theta.dtype
            ),
        ],
        axis=2,
    )
    return log_density, score


def _target_observation_score(
    theta: tf.Tensor, points: tf.Tensor, observation: tf.Tensor
) -> tf.Tensor:
    """Return the fixed-state LGSSM observation-density score."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    points = tf.convert_to_tensor(points, theta.dtype)
    observation = tf.convert_to_tensor(observation, theta.dtype)
    predicted = tf.linalg.matmul(
        points, lgssm._observation_matrix(theta.dtype), transpose_b=True
    )
    residual = observation[None, :] - predicted
    r_scale = theta[4]
    r_score = (
        tf.reduce_sum(tf.square(residual), axis=1) / tf.pow(r_scale, 3)
        - tf.cast(lgssm.OBSERVATION_DIMENSION, theta.dtype) / r_scale
    )
    return tf.concat(
        [
            tf.zeros([tf.shape(points)[0], 4], theta.dtype),
            r_score[:, None],
        ],
        axis=1,
    )


def _target_model_progressive_score_marks(
    theta: tf.Tensor,
    parents: tf.Tensor,
    parent_log_weights: tf.Tensor,
    parent_score_marks: tf.Tensor,
    children: tf.Tensor,
    observation: tf.Tensor,
) -> tf.Tensor:
    """Apply the target-model pairwise backward score recursion."""

    transition_log_density, transition_score = (
        _target_transition_log_density_and_score(theta, children, parents)
    )
    log_backward = transition_log_density + parent_log_weights[None, :]
    backward_weights = tf.nn.softmax(log_backward, axis=1)
    inherited = parent_score_marks[None, :, :] + transition_score
    return _target_observation_score(theta, children, observation) + tf.einsum(
        "ki,kip->kp", backward_weights, inherited
    )


def _state_moment_features(points: tf.Tensor) -> tf.Tensor:
    x1, x2, x3 = tf.unstack(points, axis=1)
    return tf.stack(
        [
            tf.ones_like(x1),
            x1,
            x2,
            x3,
            tf.square(x1),
            x1 * x2,
            x1 * x3,
            tf.square(x2),
            x2 * x3,
            tf.square(x3),
        ],
        axis=0,
    )


def _progressive_features(
    theta: tf.Tensor,
    points: tf.Tensor,
    log_unnormalized_weights: tf.Tensor,
    score_marks: tf.Tensor,
    innovations: tf.Tensor,
    log_innovation_weights: tf.Tensor,
    next_observation: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    predictive = _next_predictive_contribution(
        theta, points, innovations, log_innovation_weights, next_observation
    )
    normalized_weights = tf.nn.softmax(log_unnormalized_weights)
    score_mean = tf.einsum("n,np->p", normalized_weights, score_marks)
    centered = score_marks - score_mean[None, :]
    features = tf.concat(
        [
            _state_moment_features(points),
            predictive[None, :],
            tf.transpose(predictive[:, None] * centered),
        ],
        axis=0,
    )
    return features, score_mean, centered


def _conditional_future_log_likelihood(
    theta: tf.Tensor, points: tf.Tensor, future_observations: tf.Tensor
) -> tf.Tensor:
    """Exact LGSSM future likelihood conditional on each current state."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    points = tf.convert_to_tensor(points, theta.dtype)
    future_observations = tf.convert_to_tensor(future_observations, theta.dtype)
    components = lgssm._lgssm_components(theta, 1)
    transition = components["transition_matrix"][0]
    transition_covariance = components["transition_covariance"][0]
    observation_matrix = components["observation_matrix"]
    observation_covariance = components["observation_covariance"][0]
    means = points
    covariance = tf.zeros(
        [lgssm.STATE_DIMENSION, lgssm.STATE_DIMENSION], theta.dtype
    )
    values = tf.zeros([tf.shape(points)[0]], theta.dtype)
    for observation in tf.unstack(future_observations, axis=0):
        means = tf.linalg.matmul(means, transition, transpose_b=True)
        covariance = transition @ covariance @ tf.transpose(transition) + transition_covariance
        predicted = tf.linalg.matmul(means, observation_matrix, transpose_b=True)
        innovation_covariance = (
            observation_matrix
            @ covariance
            @ tf.transpose(observation_matrix)
            + observation_covariance
        )
        residual = predicted - observation[None, :]
        values += lgssm._gaussian_log_density_forward_core(
            residual[None, :, :], innovation_covariance[None, :, :]
        )["value"][0]
        chol = tf.linalg.cholesky(innovation_covariance)
        gain = tf.transpose(
            tf.linalg.cholesky_solve(chol, observation_matrix @ covariance)
        )
        means = means + tf.linalg.matmul(
            observation[None, :] - predicted, gain, transpose_b=True
        )
        covariance = covariance - gain @ observation_matrix @ covariance
        covariance = 0.5 * (covariance + tf.transpose(covariance))
    return values


def _continuation_features(
    theta: tf.Tensor, points: tf.Tensor, future_observations: tf.Tensor
) -> tf.Tensor:
    log_likelihood = _conditional_future_log_likelihood(
        theta, points, future_observations
    )
    zero_log_likelihood = _conditional_future_log_likelihood(
        theta,
        tf.zeros([1, lgssm.STATE_DIMENSION], theta.dtype),
        future_observations,
    )[0]
    stabilized = tf.exp(log_likelihood - zero_log_likelihood)
    return tf.concat([_state_moment_features(points), stabilized[None, :]], axis=0)


def _backward_information_parameters(
    theta: tf.Tensor, observations: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return natural parameters of every exact future-likelihood feature."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    observations = tf.convert_to_tensor(observations, theta.dtype)
    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 2:
        return (
            tf.zeros([0, lgssm.STATE_DIMENSION, lgssm.STATE_DIMENSION], theta.dtype),
            tf.zeros([0, lgssm.STATE_DIMENSION], theta.dtype),
        )
    components = lgssm._lgssm_components(theta, 1)
    transition = components["transition_matrix"][0]
    transition_covariance = components["transition_covariance"][0]
    observation_matrix = components["observation_matrix"]
    observation_covariance = components["observation_covariance"][0]
    transition_precision = tf.linalg.inv(transition_covariance)
    observation_precision = tf.linalg.inv(observation_covariance)
    observation_information = (
        tf.transpose(observation_matrix)
        @ observation_precision
        @ observation_matrix
    )
    transition_information = (
        tf.transpose(transition) @ transition_precision @ transition
    )
    information_matrix = tf.zeros(
        [lgssm.STATE_DIMENSION, lgssm.STATE_DIMENSION], theta.dtype
    )
    information_vector = tf.zeros([lgssm.STATE_DIMENSION], theta.dtype)
    matrix_history = []
    vector_history = []
    for observation in reversed(tf.unstack(observations[1:], axis=0)):
        child_matrix = information_matrix + observation_information
        child_vector = information_vector + tf.linalg.matvec(
            tf.transpose(observation_matrix),
            tf.linalg.matvec(observation_precision, observation),
        )
        integrated_precision = transition_precision + child_matrix
        solved_transition = tf.linalg.solve(
            integrated_precision, transition_precision @ transition
        )
        solved_vector = tf.linalg.solve(
            integrated_precision, child_vector[:, None]
        )[:, 0]
        information_matrix = transition_information - (
            tf.transpose(transition)
            @ transition_precision
            @ solved_transition
        )
        information_matrix = 0.5 * (
            information_matrix + tf.transpose(information_matrix)
        )
        information_vector = tf.linalg.matvec(
            tf.transpose(transition) @ transition_precision, solved_vector
        )
        matrix_history.append(information_matrix)
        vector_history.append(information_vector)
    return (
        tf.stack(list(reversed(matrix_history))),
        tf.stack(list(reversed(vector_history))),
    )


def _finite_lookahead_information_parameters(
    theta: tf.Tensor, observations: tf.Tensor, lookahead_steps: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """Return exact information features for each bounded future window."""

    if lookahead_steps < 1:
        raise ValueError("lookahead_steps must be positive")
    observations = tf.convert_to_tensor(observations, theta.dtype)
    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 2:
        return (
            tf.zeros([0, lgssm.STATE_DIMENSION, lgssm.STATE_DIMENSION], theta.dtype),
            tf.zeros([0, lgssm.STATE_DIMENSION], theta.dtype),
        )
    matrices = []
    vectors = []
    for time_index in range(time_steps - 1):
        stop = min(time_steps, time_index + 1 + lookahead_steps)
        local_matrices, local_vectors = _backward_information_parameters(
            theta, observations[time_index:stop]
        )
        matrices.append(local_matrices[0])
        vectors.append(local_vectors[0])
    return tf.stack(matrices), tf.stack(vectors)


def _finite_lookahead_information_parameters_loop(
    theta: tf.Tensor, observations: tf.Tensor, lookahead_steps: int
) -> tuple[tf.Tensor, tf.Tensor]:
    """Build all bounded future-information features with one functional loop."""

    if lookahead_steps < 1:
        raise ValueError("lookahead_steps must be positive")
    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    observations = tf.convert_to_tensor(observations, theta.dtype)
    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 2:
        return (
            tf.zeros([0, lgssm.STATE_DIMENSION, lgssm.STATE_DIMENSION], theta.dtype),
            tf.zeros([0, lgssm.STATE_DIMENSION], theta.dtype),
        )

    components = lgssm._lgssm_components(theta, 1)
    transition = components["transition_matrix"][0]
    transition_covariance = components["transition_covariance"][0]
    observation_matrix = components["observation_matrix"]
    observation_covariance = components["observation_covariance"][0]
    transition_precision = tf.linalg.inv(transition_covariance)
    observation_precision = tf.linalg.inv(observation_covariance)
    observation_information = (
        tf.transpose(observation_matrix)
        @ observation_precision
        @ observation_matrix
    )
    observation_information_map = (
        tf.transpose(observation_matrix) @ observation_precision
    )
    transition_information = (
        tf.transpose(transition) @ transition_precision @ transition
    )
    left_transition_precision = tf.transpose(transition) @ transition_precision
    transition_rhs = transition_precision @ transition
    start_indices = tf.range(time_steps - 1, dtype=tf.int32)
    matrix0 = tf.zeros(
        [time_steps - 1, lgssm.STATE_DIMENSION, lgssm.STATE_DIMENSION],
        theta.dtype,
    )
    vector0 = tf.zeros([time_steps - 1, lgssm.STATE_DIMENSION], theta.dtype)
    offset0 = tf.constant(min(lookahead_steps, time_steps - 1), tf.int32)

    def cond(
        offset: tf.Tensor, _matrices: tf.Tensor, _vectors: tf.Tensor
    ) -> tf.Tensor:
        return offset > 0

    def body(
        offset: tf.Tensor,
        matrices: tf.Tensor,
        vectors: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        observation_indices = start_indices + offset
        active = observation_indices < time_steps
        safe_indices = tf.minimum(observation_indices, time_steps - 1)
        local_observations = tf.gather(observations, safe_indices)
        child_matrices = matrices + observation_information[None, :, :]
        child_vectors = vectors + tf.linalg.matmul(
            local_observations,
            observation_information_map,
            transpose_b=True,
        )
        integrated_precision = transition_precision[None, :, :] + child_matrices
        solved_transition = tf.linalg.solve(
            integrated_precision,
            tf.broadcast_to(
                transition_rhs,
                [time_steps - 1, lgssm.STATE_DIMENSION, lgssm.STATE_DIMENSION],
            ),
        )
        solved_vector = tf.linalg.solve(
            integrated_precision, child_vectors[:, :, None]
        )[:, :, 0]
        updated_matrices = transition_information[None, :, :] - tf.einsum(
            "ij,njk->nik", left_transition_precision, solved_transition
        )
        updated_matrices = 0.5 * (
            updated_matrices + tf.linalg.matrix_transpose(updated_matrices)
        )
        updated_vectors = tf.einsum(
            "ij,nj->ni", left_transition_precision, solved_vector
        )
        matrices = tf.where(active[:, None, None], updated_matrices, matrices)
        vectors = tf.where(active[:, None], updated_vectors, vectors)
        return offset - 1, matrices, vectors

    _, matrices, vectors = tf.while_loop(
        cond,
        body,
        (offset0, matrix0, vector0),
        maximum_iterations=min(lookahead_steps, time_steps - 1),
        parallel_iterations=1,
    )
    return matrices, vectors


def _continuation_features_from_information(
    points: tf.Tensor,
    information_matrix: tf.Tensor,
    information_vector: tf.Tensor,
) -> tf.Tensor:
    """Evaluate the stabilized exact future likelihood from natural parameters."""

    quadratic = tf.einsum(
        "ni,ij,nj->n", points, information_matrix, points
    )
    linear = tf.linalg.matvec(points, information_vector)
    stabilized = tf.exp(-0.5 * quadratic + linear)
    return tf.concat([_state_moment_features(points), stabilized[None, :]], axis=0)


def _teacher_step(
    theta: tf.Tensor,
    parents: tf.Tensor,
    parent_log_weights: tf.Tensor,
    innovations: tf.Tensor,
    log_innovation_weights: tf.Tensor,
    observation: tf.Tensor,
    next_observation: tf.Tensor | None,
) -> dict[str, tf.Tensor]:
    flow = _flow_correction(theta, parents, innovations, observation)
    parent_count = tf.shape(parents)[0]
    innovation_count = tf.shape(innovations)[0]
    log_weights = _combine_parent_innovation_log_weights(
        parent_log_weights,
        log_innovation_weights,
        flow["log_correction"],
    )
    result = {
        **flow,
        "log_unnormalized_weights": log_weights,
        "increment": tf.reduce_logsumexp(log_weights),
    }
    if next_observation is not None:
        result["features"] = _features(
            theta,
            flow["particles"],
            innovations,
            log_innovation_weights,
            next_observation,
        )
    return result


def initial_parents(
    theta: tf.Tensor, standard_nodes: tf.Tensor, standard_weights: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    """Return stationary-law parents and standard-normal innovations."""

    standard_points, product_weights = _tensor_product_rule(
        standard_nodes, standard_weights
    )
    components = lgssm._lgssm_components(theta, 1)
    parents = standard_points * components["initial_std"][None, :]
    return (
        parents,
        tf.math.log(product_weights),
        standard_points,
        tf.math.log(product_weights),
    )


def contract_e_tp_lgssm_recursive_core(
    theta: tf.Tensor,
    observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Execute the finite recursive corrected-LEDH Contract E--TP scalar."""

    observations = tf.convert_to_tensor(observations, theta.dtype)
    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 1:
        raise ValueError("LGSSM Contract E--TP requires a static positive horizon")
    if active_indices.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError(
            f"active_indices must have shape {(time_steps - 1, FEATURE_COUNT)}"
        )
    if row_scales.shape != (time_steps - 1, FEATURE_COUNT):
        raise ValueError(f"row_scales must have shape {(time_steps - 1, FEATURE_COUNT)}")
    parents, parent_log_weights, innovations, innovation_log_weights = initial_parents(
        theta, standard_nodes, standard_weights
    )
    total = tf.constant(0.0, theta.dtype)
    increments = []
    minimum_weights = []
    condition_numbers = []
    feature_residuals = []
    target_history = []
    matched_history = []
    valid_history = []
    incoming_weight_history = []
    for time_index in range(time_steps):
        incoming_weight_history.append(tf.exp(parent_log_weights))
        next_observation = (
            observations[time_index + 1] if time_index + 1 < time_steps else None
        )
        teacher = _teacher_step(
            theta,
            parents,
            parent_log_weights,
            innovations,
            innovation_log_weights,
            observations[time_index],
            next_observation,
        )
        total += teacher["increment"]
        increments.append(teacher["increment"])
        if next_observation is None:
            valid_history.append(teacher["flow_valid"])
            continue
        projection = tp._contract_e_tp_dense_square_forward_core(
            teacher["particles"],
            teacher["log_unnormalized_weights"],
            teacher["features"],
            active_indices[time_index],
            row_scales[time_index],
        )
        parents = projection["student_points"]
        parent_log_weights = tf.math.log(projection["student_weights"])
        minimum_weights.append(projection["minimum_weight"])
        condition_numbers.append(projection["condition_number"])
        feature_residuals.append(projection["feature_residual"])
        target_history.append(projection["target"])
        matched_history.append(projection["matched_target"])
        valid_history.append(teacher["flow_valid"] & projection["valid_chart"])
    return {
        "objective": total,
        "increment_history": tf.stack(increments),
        "minimum_weight_history": tf.stack(minimum_weights)
        if minimum_weights
        else tf.zeros([0], theta.dtype),
        "condition_number_history": tf.stack(condition_numbers)
        if condition_numbers
        else tf.zeros([0], theta.dtype),
        "feature_residual_history": tf.stack(feature_residuals)
        if feature_residuals
        else tf.zeros([0, FEATURE_COUNT], theta.dtype),
        "target_history": tf.stack(target_history)
        if target_history
        else tf.zeros([0, FEATURE_COUNT], theta.dtype),
        "matched_target_history": tf.stack(matched_history)
        if matched_history
        else tf.zeros([0, FEATURE_COUNT], theta.dtype),
        "valid_history": tf.stack(valid_history),
        "final_particles": teacher["particles"],
        "final_log_unnormalized_weights": teacher["log_unnormalized_weights"],
        "incoming_weight_history": incoming_weight_history,
    }


def contract_e_tp_lgssm_score_informed_recursive_core(
    theta: tf.Tensor,
    observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    *,
    feature_mode: str,
    lookahead_steps: int | None = None,
) -> dict[str, tf.Tensor]:
    """Execute a score-informed or continuation-oracle LGSSM recursion."""

    if feature_mode not in (
        "progressive_target_model_score",
        "exact_continuation",
        "finite_lookahead",
    ):
        raise ValueError(f"unknown score-informed feature mode: {feature_mode}")
    observations = tf.convert_to_tensor(observations, theta.dtype)
    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 1:
        raise ValueError("LGSSM Contract E--TP requires a static positive horizon")
    feature_count = (
        PROGRESSIVE_FEATURE_COUNT
        if feature_mode == "progressive_target_model_score"
        else CONTINUATION_FEATURE_COUNT
    )
    if active_indices.shape != (time_steps - 1, feature_count):
        raise ValueError(
            f"active_indices must have shape {(time_steps - 1, feature_count)}"
        )
    if row_scales.shape != (time_steps - 1, feature_count):
        raise ValueError(
            f"row_scales must have shape {(time_steps - 1, feature_count)}"
        )

    parents, parent_log_weights, innovations, innovation_log_weights = initial_parents(
        theta, standard_nodes, standard_weights
    )
    parent_score_marks = _initial_target_model_score_marks(theta, parents)
    if feature_mode == "exact_continuation":
        continuation_matrices, continuation_vectors = (
            _backward_information_parameters(theta, observations)
        )
    elif feature_mode == "finite_lookahead":
        if lookahead_steps is None:
            raise ValueError("finite_lookahead requires lookahead_steps")
        continuation_matrices, continuation_vectors = (
            _finite_lookahead_information_parameters(
                theta, observations, lookahead_steps
            )
        )
    total = tf.constant(0.0, theta.dtype)
    increments = []
    minimum_weights = []
    condition_numbers = []
    feature_residuals = []
    target_history = []
    matched_history = []
    valid_history = []
    score_mark_mean_history = []
    score_mark_center_residual_history = []
    incoming_weight_history = []
    for time_index in range(time_steps):
        incoming_weight_history.append(tf.exp(parent_log_weights))
        flow = _flow_correction(
            theta, parents, innovations, observations[time_index]
        )
        innovation_count = tf.shape(innovations)[0]
        log_weights = _combine_parent_innovation_log_weights(
            parent_log_weights,
            innovation_log_weights,
            flow["log_correction"],
        )
        increment = tf.reduce_logsumexp(log_weights)
        total += increment
        increments.append(increment)
        if feature_mode == "progressive_target_model_score":
            teacher_score_marks = _target_model_progressive_score_marks(
                theta,
                parents,
                parent_log_weights,
                parent_score_marks,
                flow["particles"],
                observations[time_index],
            )
            normalized_weights = tf.nn.softmax(log_weights)
            score_mean = tf.einsum(
                "n,np->p", normalized_weights, teacher_score_marks
            )
            centered_marks = teacher_score_marks - score_mean[None, :]
            score_mark_mean_history.append(score_mean)
            score_mark_center_residual_history.append(
                tf.einsum("n,np->p", normalized_weights, centered_marks)
            )
        if time_index + 1 == time_steps:
            valid_history.append(flow["flow_valid"])
            continue

        if feature_mode == "progressive_target_model_score":
            features, _, centered_marks = _progressive_features(
                theta,
                flow["particles"],
                log_weights,
                teacher_score_marks,
                innovations,
                innovation_log_weights,
                observations[time_index + 1],
            )
        else:
            features = _continuation_features_from_information(
                flow["particles"],
                continuation_matrices[time_index],
                continuation_vectors[time_index],
            )
        projection = tp._contract_e_tp_dense_square_forward_core(
            flow["particles"],
            log_weights,
            features,
            active_indices[time_index],
            row_scales[time_index],
        )
        parents = projection["student_points"]
        parent_log_weights = tf.math.log(projection["student_weights"])
        if feature_mode == "progressive_target_model_score":
            parent_score_marks = tf.gather(
                centered_marks, active_indices[time_index]
            )
            parent_score_marks -= tf.einsum(
                "n,np->p", projection["student_weights"], parent_score_marks
            )[None, :]
        minimum_weights.append(projection["minimum_weight"])
        condition_numbers.append(projection["condition_number"])
        feature_residuals.append(projection["feature_residual"])
        target_history.append(projection["target"])
        matched_history.append(projection["matched_target"])
        valid_history.append(flow["flow_valid"] & projection["valid_chart"])

    return {
        "objective": total,
        "increment_history": tf.stack(increments),
        "minimum_weight_history": tf.stack(minimum_weights)
        if minimum_weights
        else tf.zeros([0], theta.dtype),
        "condition_number_history": tf.stack(condition_numbers)
        if condition_numbers
        else tf.zeros([0], theta.dtype),
        "feature_residual_history": tf.stack(feature_residuals)
        if feature_residuals
        else tf.zeros([0, feature_count], theta.dtype),
        "target_history": tf.stack(target_history)
        if target_history
        else tf.zeros([0, feature_count], theta.dtype),
        "matched_target_history": tf.stack(matched_history)
        if matched_history
        else tf.zeros([0, feature_count], theta.dtype),
        "valid_history": tf.stack(valid_history),
        "score_mark_mean_history": tf.stack(score_mark_mean_history)
        if score_mark_mean_history
        else tf.zeros([0, lgssm.PARAMETER_COUNT], theta.dtype),
        "score_mark_center_residual_history": tf.stack(
            score_mark_center_residual_history
        )
        if score_mark_center_residual_history
        else tf.zeros([0, lgssm.PARAMETER_COUNT], theta.dtype),
        "final_particles": flow["particles"],
        "final_log_unnormalized_weights": log_weights,
        "incoming_weight_history": incoming_weight_history,
    }


def contract_e_tp_lgssm_finite_lookahead_loop_core(
    theta: tf.Tensor,
    observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    *,
    lookahead_steps: int,
) -> dict[str, tf.Tensor]:
    """Execute finite-lookahead Contract E--TP with a functional time loop."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    observations = tf.convert_to_tensor(observations, theta.dtype)
    time_steps = observations.shape[0]
    if time_steps is None or time_steps < 1:
        raise ValueError("LGSSM Contract E--TP requires a static positive horizon")
    if active_indices.shape != (time_steps - 1, CONTINUATION_FEATURE_COUNT):
        raise ValueError(
            "active_indices must have shape "
            f"{(time_steps - 1, CONTINUATION_FEATURE_COUNT)}"
        )
    if row_scales.shape != (time_steps - 1, CONTINUATION_FEATURE_COUNT):
        raise ValueError(
            "row_scales must have shape "
            f"{(time_steps - 1, CONTINUATION_FEATURE_COUNT)}"
        )

    parents, parent_log_weights, innovations, innovation_log_weights = initial_parents(
        theta, standard_nodes, standard_weights
    )
    initial_incoming_weights = tf.exp(parent_log_weights)
    if time_steps == 1:
        terminal = _flow_correction(
            theta, parents, innovations, observations[0]
        )
        innovation_count = tf.shape(innovations)[0]
        terminal_log_weights = _combine_parent_innovation_log_weights(
            parent_log_weights,
            innovation_log_weights,
            terminal["log_correction"],
        )
        terminal_increment = tf.reduce_logsumexp(terminal_log_weights)
        return {
            "objective": terminal_increment,
            "increment_history": terminal_increment[None],
            "minimum_weight_history": tf.zeros([0], theta.dtype),
            "condition_number_history": tf.zeros([0], theta.dtype),
            "feature_residual_history": tf.zeros(
                [0, CONTINUATION_FEATURE_COUNT], theta.dtype
            ),
            "target_history": tf.zeros(
                [0, CONTINUATION_FEATURE_COUNT], theta.dtype
            ),
            "matched_target_history": tf.zeros(
                [0, CONTINUATION_FEATURE_COUNT], theta.dtype
            ),
            "valid_history": terminal["flow_valid"][None],
            "score_mark_mean_history": tf.zeros(
                [0, lgssm.PARAMETER_COUNT], theta.dtype
            ),
            "score_mark_center_residual_history": tf.zeros(
                [0, lgssm.PARAMETER_COUNT], theta.dtype
            ),
            "final_particles": terminal["particles"],
            "final_log_unnormalized_weights": terminal_log_weights,
            "initial_incoming_weights": initial_incoming_weights,
            "post_reset_incoming_weight_history": tf.zeros(
                [0, CONTINUATION_FEATURE_COUNT], theta.dtype
            ),
        }

    continuation_matrices, continuation_vectors = (
        _finite_lookahead_information_parameters_loop(
            theta, observations, lookahead_steps
        )
    )
    increment_history = tf.TensorArray(
        theta.dtype, size=time_steps, clear_after_read=False
    )
    minimum_weight_history = tf.TensorArray(
        theta.dtype, size=time_steps - 1, clear_after_read=False
    )
    condition_number_history = tf.TensorArray(
        theta.dtype, size=time_steps - 1, clear_after_read=False
    )
    feature_residual_history = tf.TensorArray(
        theta.dtype,
        size=time_steps - 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([CONTINUATION_FEATURE_COUNT]),
    )
    target_history = tf.TensorArray(
        theta.dtype,
        size=time_steps - 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([CONTINUATION_FEATURE_COUNT]),
    )
    matched_target_history = tf.TensorArray(
        theta.dtype,
        size=time_steps - 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([CONTINUATION_FEATURE_COUNT]),
    )
    valid_history = tf.TensorArray(
        tf.bool, size=time_steps, clear_after_read=False
    )
    post_reset_incoming_history = tf.TensorArray(
        theta.dtype,
        size=time_steps - 1,
        clear_after_read=False,
        element_shape=tf.TensorShape([CONTINUATION_FEATURE_COUNT]),
    )

    first_flow = _flow_correction(theta, parents, innovations, observations[0])
    innovation_count = tf.shape(innovations)[0]
    first_log_weights = _combine_parent_innovation_log_weights(
        parent_log_weights,
        innovation_log_weights,
        first_flow["log_correction"],
    )
    first_increment = tf.reduce_logsumexp(first_log_weights)
    first_features = _continuation_features_from_information(
        first_flow["particles"],
        continuation_matrices[0],
        continuation_vectors[0],
    )
    first_projection = tp._contract_e_tp_dense_square_forward_core(
        first_flow["particles"],
        first_log_weights,
        first_features,
        active_indices[0],
        row_scales[0],
    )
    parents = first_projection["student_points"]
    parent_log_weights = tf.math.log(first_projection["student_weights"])
    increment_history = increment_history.write(0, first_increment)
    minimum_weight_history = minimum_weight_history.write(
        0, first_projection["minimum_weight"]
    )
    condition_number_history = condition_number_history.write(
        0, first_projection["condition_number"]
    )
    feature_residual_history = feature_residual_history.write(
        0, first_projection["feature_residual"]
    )
    target_history = target_history.write(0, first_projection["target"])
    matched_target_history = matched_target_history.write(
        0, first_projection["matched_target"]
    )
    valid_history = valid_history.write(
        0, first_flow["flow_valid"] & first_projection["valid_chart"]
    )
    post_reset_incoming_history = post_reset_incoming_history.write(
        0, first_projection["student_weights"]
    )
    total = first_increment

    def cond(
        time_index: tf.Tensor,
        _parents: tf.Tensor,
        _parent_log_weights: tf.Tensor,
        _total: tf.Tensor,
        *_histories: tf.TensorArray,
    ) -> tf.Tensor:
        return time_index < time_steps - 1

    def body(
        time_index: tf.Tensor,
        parents: tf.Tensor,
        parent_log_weights: tf.Tensor,
        total: tf.Tensor,
        increment_history: tf.TensorArray,
        minimum_weight_history: tf.TensorArray,
        condition_number_history: tf.TensorArray,
        feature_residual_history: tf.TensorArray,
        target_history: tf.TensorArray,
        matched_target_history: tf.TensorArray,
        valid_history: tf.TensorArray,
        post_reset_incoming_history: tf.TensorArray,
    ):
        flow = _flow_correction(
            theta, parents, innovations, observations[time_index]
        )
        log_weights = _combine_parent_innovation_log_weights(
            parent_log_weights,
            innovation_log_weights,
            flow["log_correction"],
        )
        increment = tf.reduce_logsumexp(log_weights)
        features = _continuation_features_from_information(
            flow["particles"],
            continuation_matrices[time_index],
            continuation_vectors[time_index],
        )
        projection = tp._contract_e_tp_dense_square_forward_core(
            flow["particles"],
            log_weights,
            features,
            active_indices[time_index],
            row_scales[time_index],
        )
        parents = projection["student_points"]
        parent_log_weights = tf.math.log(projection["student_weights"])
        increment_history = increment_history.write(time_index, increment)
        minimum_weight_history = minimum_weight_history.write(
            time_index, projection["minimum_weight"]
        )
        condition_number_history = condition_number_history.write(
            time_index, projection["condition_number"]
        )
        feature_residual_history = feature_residual_history.write(
            time_index, projection["feature_residual"]
        )
        target_history = target_history.write(time_index, projection["target"])
        matched_target_history = matched_target_history.write(
            time_index, projection["matched_target"]
        )
        valid_history = valid_history.write(
            time_index, flow["flow_valid"] & projection["valid_chart"]
        )
        post_reset_incoming_history = post_reset_incoming_history.write(
            time_index, projection["student_weights"]
        )
        return (
            time_index + 1,
            parents,
            parent_log_weights,
            total + increment,
            increment_history,
            minimum_weight_history,
            condition_number_history,
            feature_residual_history,
            target_history,
            matched_target_history,
            valid_history,
            post_reset_incoming_history,
        )

    if time_steps > 2:
        (
            _,
            parents,
            parent_log_weights,
            total,
            increment_history,
            minimum_weight_history,
            condition_number_history,
            feature_residual_history,
            target_history,
            matched_target_history,
            valid_history,
            post_reset_incoming_history,
        ) = tf.while_loop(
            cond,
            body,
            (
                tf.constant(1, tf.int32),
                parents,
                parent_log_weights,
                total,
                increment_history,
                minimum_weight_history,
                condition_number_history,
                feature_residual_history,
                target_history,
                matched_target_history,
                valid_history,
                post_reset_incoming_history,
            ),
            maximum_iterations=time_steps - 2,
            parallel_iterations=1,
        )

    terminal_flow = _flow_correction(
        theta, parents, innovations, observations[time_steps - 1]
    )
    terminal_log_weights = _combine_parent_innovation_log_weights(
        parent_log_weights,
        innovation_log_weights,
        terminal_flow["log_correction"],
    )
    terminal_increment = tf.reduce_logsumexp(terminal_log_weights)
    increment_history = increment_history.write(
        time_steps - 1, terminal_increment
    )
    valid_history = valid_history.write(
        time_steps - 1, terminal_flow["flow_valid"]
    )
    return {
        "objective": total + terminal_increment,
        "increment_history": increment_history.stack(),
        "minimum_weight_history": minimum_weight_history.stack(),
        "condition_number_history": condition_number_history.stack(),
        "feature_residual_history": feature_residual_history.stack(),
        "target_history": target_history.stack(),
        "matched_target_history": matched_target_history.stack(),
        "valid_history": valid_history.stack(),
        "score_mark_mean_history": tf.zeros(
            [0, lgssm.PARAMETER_COUNT], theta.dtype
        ),
        "score_mark_center_residual_history": tf.zeros(
            [0, lgssm.PARAMETER_COUNT], theta.dtype
        ),
        "final_particles": terminal_flow["particles"],
        "final_log_unnormalized_weights": terminal_log_weights,
        "initial_incoming_weights": initial_incoming_weights,
        "post_reset_incoming_weight_history": (
            post_reset_incoming_history.stack()
        ),
    }


def exact_kalman_value(theta: tf.Tensor, observations: tf.Tensor) -> tf.Tensor:
    """Differentiable transition-first Kalman likelihood for the same target."""

    theta = tf.reshape(tf.convert_to_tensor(theta), [lgssm.PARAMETER_COUNT])
    observations = tf.convert_to_tensor(observations, theta.dtype)
    components = lgssm._lgssm_components(theta, 1)
    mean = tf.zeros([lgssm.STATE_DIMENSION], theta.dtype)
    covariance = tf.linalg.diag(tf.square(components["initial_std"]))
    transition = components["transition_matrix"][0]
    transition_covariance = components["transition_covariance"][0]
    observation_matrix = components["observation_matrix"]
    observation_covariance = components["observation_covariance"][0]
    total = tf.constant(0.0, theta.dtype)
    for observation in tf.unstack(observations, axis=0):
        mean = tf.linalg.matvec(transition, mean)
        covariance = transition @ covariance @ tf.transpose(transition) + transition_covariance
        predicted = tf.linalg.matvec(observation_matrix, mean)
        innovation_covariance = (
            observation_matrix
            @ covariance
            @ tf.transpose(observation_matrix)
            + observation_covariance
        )
        residual = observation - predicted
        total += lgssm._gaussian_log_density_forward_core(
            residual[None, None, :], innovation_covariance[None, :, :]
        )["value"][0, 0]
        chol = tf.linalg.cholesky(innovation_covariance)
        gain = tf.transpose(
            tf.linalg.cholesky_solve(
                chol, observation_matrix @ covariance
            )
        )
        mean = mean + tf.linalg.matvec(gain, residual)
        covariance = covariance - gain @ observation_matrix @ covariance
        covariance = 0.5 * (covariance + tf.transpose(covariance))
    return total


def make_contract_e_tp_lgssm_recursive_tf(
    observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    *,
    jit_compile: bool = True,
):
    """Bind one prepared recursive finite target into an XLA-default graph."""

    observations = tf.convert_to_tensor(observations, lgssm.DTYPE)
    standard_nodes = tf.convert_to_tensor(standard_nodes, lgssm.DTYPE)
    standard_weights = tf.convert_to_tensor(standard_weights, lgssm.DTYPE)
    active_indices = tf.convert_to_tensor(active_indices, tf.int32)
    row_scales = tf.convert_to_tensor(row_scales, lgssm.DTYPE)

    @tf.function(
        input_signature=[tf.TensorSpec([lgssm.PARAMETER_COUNT], lgssm.DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = contract_e_tp_lgssm_recursive_core(
                theta,
                observations,
                standard_nodes,
                standard_weights,
                active_indices,
                row_scales,
            )
        score = tape.gradient(result["objective"], theta)
        return {**result, "score": score}

    return evaluate


def make_contract_e_tp_lgssm_score_informed_recursive_tf(
    observations: tf.Tensor,
    standard_nodes: tf.Tensor,
    standard_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    *,
    feature_mode: str,
    lookahead_steps: int | None = None,
    jit_compile: bool = True,
):
    """Bind a prepared score-informed recursion into an XLA-default graph."""

    observations = tf.convert_to_tensor(observations, lgssm.DTYPE)
    standard_nodes = tf.convert_to_tensor(standard_nodes, lgssm.DTYPE)
    standard_weights = tf.convert_to_tensor(standard_weights, lgssm.DTYPE)
    active_indices = tf.convert_to_tensor(active_indices, tf.int32)
    row_scales = tf.convert_to_tensor(row_scales, lgssm.DTYPE)
    if feature_mode not in (
        "progressive_target_model_score",
        "exact_continuation",
        "finite_lookahead",
    ):
        raise ValueError(f"unknown score-informed feature mode: {feature_mode}")
    if feature_mode == "finite_lookahead" and lookahead_steps is None:
        raise ValueError("finite_lookahead requires lookahead_steps")

    @tf.function(
        input_signature=[tf.TensorSpec([lgssm.PARAMETER_COUNT], lgssm.DTYPE)],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            if feature_mode == "finite_lookahead":
                result = contract_e_tp_lgssm_finite_lookahead_loop_core(
                    theta,
                    observations,
                    standard_nodes,
                    standard_weights,
                    active_indices,
                    row_scales,
                    lookahead_steps=lookahead_steps,
                )
            else:
                result = contract_e_tp_lgssm_score_informed_recursive_core(
                    theta,
                    observations,
                    standard_nodes,
                    standard_weights,
                    active_indices,
                    row_scales,
                    feature_mode=feature_mode,
                    lookahead_steps=lookahead_steps,
                )
        score = tape.gradient(result["objective"], theta)
        return {**result, "score": score}

    return evaluate


__all__ = [
    "ALGORITHM_ID",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "CONTINUATION_FEATURE_COUNT",
    "CONTINUATION_FEATURE_NAMES",
    "PROGRESSIVE_FEATURE_COUNT",
    "PROGRESSIVE_FEATURE_NAMES",
    "contract_e_tp_lgssm_recursive_core",
    "contract_e_tp_lgssm_finite_lookahead_loop_core",
    "contract_e_tp_lgssm_score_informed_recursive_core",
    "exact_kalman_value",
    "initial_parents",
    "make_contract_e_tp_lgssm_recursive_tf",
    "make_contract_e_tp_lgssm_score_informed_recursive_tf",
]
