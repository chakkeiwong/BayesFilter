"""XLA value and analytical score recurrence for a fixed SGQF cloud.

The numerical factor branches match the existing Fixed-SGQF contract: Cholesky
at/above epsilon, the original eigenfactor below epsilon, and a veto for a
negative eigenvalue. No ridge, eigenvalue floor, or alternative derivative is
introduced. Python step records are assembled by the reporting APIs afterwards.
"""

from __future__ import annotations

import math

import tensorflow as tf


def _symmetrize(matrix):
    # Avoid Grappler's scalar transpose rewrite inside nested recurrences.
    if matrix.shape[-2:] == (1, 1):
        return 0.5 * (matrix + matrix)
    return 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))


def _weighted_mean(points, weights):
    return tf.einsum("r,rn->n", weights, points)


def _weighted_covariance(points, weights):
    return tf.einsum("r,rn,rm->nm", weights, points, points)


def _factor(matrix, epsilon):
    matrix = _symmetrize(matrix)
    finite = tf.reduce_all(tf.math.is_finite(matrix))
    identity = tf.eye(matrix.shape[-1], dtype=tf.float64)
    eigenvalues, eigenvectors = tf.linalg.eigh(tf.where(finite, matrix, identity))
    minimum = tf.reduce_min(eigenvalues)
    valid = tf.logical_and(finite, minimum >= 0.0)
    safe = tf.where(valid, matrix, identity)
    factor = tf.cond(
        minimum >= tf.constant(epsilon, tf.float64),
        lambda: tf.linalg.cholesky(safe),
        lambda: eigenvectors @ tf.linalg.diag(tf.sqrt(tf.maximum(eigenvalues, 0.0))),
    )
    return tf.where(valid, factor, identity), minimum, valid


def fixed_sgqf_tensor_result(
    observations, model, cloud, branch_config, derivatives=None, *, jit_compile=True
):
    """Return tensor histories and the first failure code (0 means accepted).

    Callers can enclose this API in one stable parameter-to-result tf.function.
    ``jit_compile=False`` is an explicit independent reference/debug exception.
    """
    from bayesfilter.nonlinear.fixed_sgqf_derivatives_tf import (
        _cholesky_first_derivative,
        _covariance_first_derivative,
        _point_map_first_order_values,
    )

    y = tf.convert_to_tensor(observations, tf.float64)
    with_score = derivatives is not None
    mean = tf.convert_to_tensor(model.initial_mean, tf.float64)
    covariance = _symmetrize(model.initial_covariance)
    process_covariance = _symmetrize(model.process_covariance)
    observation_covariance = _symmetrize(model.observation_covariance)
    dates, state_dim, observation_dim = (
        int(y.shape[0]),
        int(mean.shape[0]),
        int(observation_covariance.shape[0]),
    )
    weights = tf.convert_to_tensor(cloud.weights, tf.float64)
    parameter_dim = int(derivatives.parameter_dim) if with_score else 1
    if with_score:
        d_mean = derivatives.d_initial_mean
        d_covariance = _symmetrize(derivatives.d_initial_covariance)
        d_process_covariance = _symmetrize(derivatives.d_process_covariance)
        d_observation_covariance = _symmetrize(derivatives.d_observation_covariance)
    else:
        d_mean = tf.zeros([parameter_dim, state_dim], tf.float64)
        d_covariance = tf.zeros([parameter_dim, state_dim, state_dim], tf.float64)
    history_shapes = {
        "previous_covariance": [state_dim, state_dim],
        "predicted_mean": [state_dim],
        "predicted_covariance": [state_dim, state_dim],
        "predicted_factor": [state_dim, state_dim],
        "observation_mean": [observation_dim],
        "innovation_covariance": [observation_dim, observation_dim],
        "innovation_factor": [observation_dim, observation_dim],
        "cross_covariance": [state_dim, observation_dim],
        "innovation": [observation_dim],
        "innovation_solve": [observation_dim],
        "gain": [state_dim, observation_dim],
        "filtered_mean": [state_dim],
        "filtered_covariance": [state_dim, state_dim],
        "log_likelihood_increment": [],
        "minimum_eigenvalues": [4],
    }
    histories = {
        name: tf.zeros([dates, *shape], tf.float64)
        for name, shape in history_shapes.items()
    }

    def body(
        time_index,
        mean,
        covariance,
        d_mean,
        d_covariance,
        value,
        score,
        failure,
        histories,
    ):
        d_filtered_mean, d_filtered_covariance = d_mean, d_covariance
        score_increment = tf.zeros([parameter_dim], tf.float64)
        previous_factor, previous_covariance_min, previous_covariance_valid = _factor(
            covariance, branch_config.predictive_epsilon
        )
        if with_score:
            d_previous_factor = _cholesky_first_derivative(
                previous_factor, d_covariance
            )
        previous_points = mean[tf.newaxis, :] + cloud.points @ tf.transpose(
            previous_factor
        )
        if with_score:
            d_previous_points = d_mean[:, tf.newaxis, :] + tf.einsum(
                "rd,pnd->prn", cloud.points, d_previous_factor
            )
        transition_values = model.transition(previous_points)
        if with_score:
            d_transition_values = _point_map_first_order_values(
                previous_points,
                d_previous_points,
                jacobian_fn=derivatives.transition_state_jacobian_fn,
                parameter_derivative_fn=derivatives.d_transition_fn,
                jvp_fn=derivatives.transition_jvp_fn,
            )
        predicted_mean = _weighted_mean(transition_values, weights)
        if with_score:
            d_predicted_mean = tf.einsum("r,prn->pn", weights, d_transition_values)
        centered_predicted = transition_values - predicted_mean[tf.newaxis, :]
        if with_score:
            d_centered_predicted = (
                d_transition_values - d_predicted_mean[:, tf.newaxis, :]
            )
        predicted_covariance = _symmetrize(
            process_covariance + _weighted_covariance(centered_predicted, weights)
        )
        if with_score:
            d_predicted_covariance = _symmetrize(
                d_process_covariance
                + _covariance_first_derivative(
                    centered_predicted, d_centered_predicted, weights
                )
            )
        predicted_factor, predictive_covariance_min, predictive_covariance_valid = (
            _factor(predicted_covariance, branch_config.predictive_epsilon)
        )
        if with_score:
            d_predicted_factor = _cholesky_first_derivative(
                predicted_factor, d_predicted_covariance
            )
        predictive_points = predicted_mean[tf.newaxis, :] + cloud.points @ tf.transpose(
            predicted_factor
        )
        if with_score:
            d_predictive_points = d_predicted_mean[:, tf.newaxis, :] + tf.einsum(
                "rd,pnd->prn", cloud.points, d_predicted_factor
            )
        observation_values = model.observe(predictive_points)
        if with_score:
            d_observation_values = _point_map_first_order_values(
                predictive_points,
                d_predictive_points,
                jacobian_fn=derivatives.observation_state_jacobian_fn,
                parameter_derivative_fn=derivatives.d_observation_fn,
                jvp_fn=derivatives.observation_jvp_fn,
            )
        observation_mean = _weighted_mean(observation_values, weights)
        if with_score:
            d_observation_mean = tf.einsum("r,prm->pm", weights, d_observation_values)
        centered_observation = observation_values - observation_mean[tf.newaxis, :]
        if with_score:
            d_centered_observation = (
                d_observation_values - d_observation_mean[:, tf.newaxis, :]
            )
        innovation_covariance = _symmetrize(
            observation_covariance + _weighted_covariance(centered_observation, weights)
        )
        if with_score:
            d_innovation_covariance = _symmetrize(
                d_observation_covariance
                + _covariance_first_derivative(
                    centered_observation, d_centered_observation, weights
                )
            )
        cross_covariance = tf.transpose(
            predictive_points - predicted_mean[tf.newaxis, :]
        ) @ (centered_observation * weights[:, tf.newaxis])
        if with_score:
            d_cross_covariance = tf.einsum(
                "prn,rm->pnm",
                d_predictive_points - d_predicted_mean[:, tf.newaxis, :],
                centered_observation * weights[:, tf.newaxis],
            ) + tf.einsum(
                "rn,prm->pnm",
                predictive_points - predicted_mean[tf.newaxis, :],
                d_centered_observation * weights[tf.newaxis, :, tf.newaxis],
            )
        innovation = y[time_index] - observation_mean
        if with_score:
            d_innovation = -d_observation_mean
        innovation_factor, innovation_covariance_min, innovation_covariance_valid = (
            _factor(innovation_covariance, branch_config.innovation_epsilon)
        )
        innovation_precision = tf.linalg.cholesky_solve(
            innovation_factor,
            tf.eye(observation_dim, dtype=tf.float64),
        )
        innovation_solve = tf.linalg.cholesky_solve(
            innovation_factor,
            innovation[:, tf.newaxis],
        )[:, 0]
        if with_score:
            trace_term = tf.einsum(
                "mn,pnm->p", innovation_precision, d_innovation_covariance
            )
            dv_term = 2.0 * tf.einsum("pm,m->p", d_innovation, innovation_solve)
            quad_term = tf.einsum(
                "m,pmn,n->p",
                innovation_solve,
                d_innovation_covariance,
                innovation_solve,
            )
            score_increment = -0.5 * (trace_term + dv_term - quad_term)
        log_det = 2.0 * tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(innovation_factor))
        )
        log_likelihood_increment = -0.5 * (
            tf.cast(observation_dim, tf.float64)
            * tf.math.log(tf.constant(2.0 * math.pi, dtype=tf.float64))
            + log_det
            + tf.reduce_sum(innovation * innovation_solve)
        )
        gain = cross_covariance @ innovation_precision
        if with_score:
            d_gain = tf.einsum(
                "pnm,mk->pnk", d_cross_covariance, innovation_precision
            ) - tf.einsum(
                "nm,pmk,kl->pnl", gain, d_innovation_covariance, innovation_precision
            )
        filtered_mean = predicted_mean + tf.linalg.matvec(gain, innovation)
        if with_score:
            d_filtered_mean = (
                d_predicted_mean
                + tf.einsum("pnm,m->pn", d_gain, innovation)
                + tf.einsum("nm,pm->pn", gain, d_innovation)
            )
        filtered_covariance = _symmetrize(
            predicted_covariance - gain @ innovation_covariance @ tf.transpose(gain)
        )
        if with_score:
            d_filtered_covariance = _symmetrize(
                d_predicted_covariance
                - tf.einsum("pnm,mk,lk->pnl", d_gain, innovation_covariance, gain)
                - tf.einsum("nm,pmk,lk->pnl", gain, d_innovation_covariance, gain)
                - tf.einsum("nm,mk,plk->pnl", gain, innovation_covariance, d_gain)
            )
        _carried_factor, carried_covariance_min, carried_covariance_valid = _factor(
            filtered_covariance, branch_config.predictive_epsilon
        )

        stage = tf.where(
            previous_covariance_valid,
            tf.where(
                predictive_covariance_valid,
                tf.where(
                    innovation_covariance_valid,
                    tf.where(carried_covariance_valid, 0, 4),
                    3,
                ),
                2,
            ),
            1,
        )
        finite = tf.logical_and(
            tf.math.is_finite(log_likelihood_increment),
            tf.reduce_all(tf.math.is_finite(score_increment)),
        )
        stage = tf.where(tf.logical_and(stage == 0, tf.logical_not(finite)), 5, stage)
        snapshot = {
            "previous_covariance": covariance,
            "predicted_mean": predicted_mean,
            "predicted_covariance": predicted_covariance,
            "predicted_factor": predicted_factor,
            "observation_mean": observation_mean,
            "innovation_covariance": innovation_covariance,
            "innovation_factor": innovation_factor,
            "cross_covariance": cross_covariance,
            "innovation": innovation,
            "innovation_solve": innovation_solve,
            "gain": gain,
            "filtered_mean": filtered_mean,
            "filtered_covariance": filtered_covariance,
            "log_likelihood_increment": log_likelihood_increment,
            "minimum_eigenvalues": tf.stack(
                [
                    previous_covariance_min,
                    predictive_covariance_min,
                    innovation_covariance_min,
                    carried_covariance_min,
                ]
            ),
        }
        histories = {
            name: tf.tensor_scatter_nd_update(
                histories[name], tf.reshape(time_index, [1, 1]), data[None]
            )
            for name, data in snapshot.items()
        }
        return (
            time_index + 1,
            filtered_mean,
            filtered_covariance,
            d_filtered_mean,
            d_filtered_covariance,
            value + log_likelihood_increment,
            score + score_increment,
            stage,
            histories,
        )

    def run():
        return tf.while_loop(
            lambda time_index, _mean, _cov, _dm, _dc, _v, _s, status, _h: (
                tf.logical_and(time_index < dates, status == 0)
            ),
            body,
            (
                tf.constant(0),
                mean,
                covariance,
                d_mean,
                d_covariance,
                tf.constant(0.0, tf.float64),
                tf.zeros([parameter_dim], tf.float64),
                tf.constant(0),
                histories,
            ),
            parallel_iterations=1,
            maximum_iterations=dates,
        )

    result = (
        tf.function(run, input_signature=[], jit_compile=True)()
        if jit_compile
        else run()
    )
    valid = result[7] == 0
    return {
        "attempted_steps": result[0],
        "accepted_steps": result[0] - tf.cast(tf.logical_not(valid), tf.int32),
        "status_code": result[7],
        "valid": valid,
        "log_likelihood": tf.where(
            valid, result[5], tf.constant(float("nan"), tf.float64)
        ),
        "score": tf.where(valid, result[6], tf.constant(float("nan"), tf.float64)),
        "filtered_mean": result[1],
        "filtered_covariance": result[2],
        "d_filtered_mean": result[3],
        "d_filtered_covariance": result[4],
        "history": result[8],
    }
