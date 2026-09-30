"""TensorFlow steps for fixed and defensive C2 UKF preparation.

These preserve the existing extension-or-invention proposal formulas. The
shared preparation owner encloses their time and exact-prefix recurrence.
"""

import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_tf import (
    DTYPE,
    complete_gaussian_mixture_log_density,
    student_log_density,
)
from bayesfilter.ops.stateless_gamma_tf import philox_gamma_float64
from bayesfilter.ops.stateless_random_tf import (
    philox_normal_float64,
    philox_uniform_float64,
)


def mixture_step(
    *,
    ukf,
    transformed,
    sampler,
    count,
    dimension,
    component_count,
    offset,
    sign_tensor,
    component_weights,
    log_parent_weights,
    seed,
    time_index,
    nu,
    epsilon_min,
    epsilon_max,
    center,
    temperature,
):
    def words(offset):
        return tf.stack([seed, tf.cast(offset, tf.int64)])

    posterior_cholesky = ukf["posterior_cholesky"]
    direction_one = posterior_cholesky[:, :, 0]
    direction_two = (
        posterior_cholesky[:, :, 1] if dimension >= 2 else tf.zeros_like(direction_one)
    )
    between_component_covariance = offset**2 * tf.einsum(
        "bi,bj->bij", direction_one, direction_one
    )
    if component_count == 4:
        between_component_covariance = (
            between_component_covariance
            + offset** 2 * tf.einsum("bi,bj->bij", direction_two, direction_two)
        )
    component_covariance = 0.5 * (
        ukf["posterior_covariance"]
        - between_component_covariance
        + tf.linalg.matrix_transpose(
            ukf["posterior_covariance"] - between_component_covariance
        )
    )
    component_minimum = tf.reduce_min(tf.linalg.eigvalsh(component_covariance), axis=1)
    component_chol_single = tf.linalg.cholesky(component_covariance)
    component_means = ukf["posterior_mean"][:, tf.newaxis, :] + offset * (
        sign_tensor[tf.newaxis, :, 0, tf.newaxis] * direction_one[:, tf.newaxis, :]
        + sign_tensor[tf.newaxis, :, 1, tf.newaxis] * direction_two[:, tf.newaxis, :]
    )
    component_cholesky = tf.broadcast_to(
        component_chol_single[:, tf.newaxis, :, :],
        [count, component_count, dimension, dimension],
    )
    ancestor_uniforms = philox_uniform_float64([count], words(6100 + 41 * time_index))
    component_uniforms = philox_uniform_float64([count], words(6200 + 43 * time_index))
    standard_normal = philox_normal_float64(
        [count, dimension], words(6300 + 47 * time_index)
    )
    step = sampler(
        ukf["posterior_covariance"],
        component_means,
        component_cholesky,
        component_weights,
        ukf["innovation_log_likelihood"],
        log_parent_weights,
        ancestor_uniforms,
        component_uniforms,
        standard_normal,
    )
    selected_component_means = tf.gather(component_means, step["ancestor_indices"])
    selected_component_cholesky = tf.gather(
        component_cholesky, step["ancestor_indices"]
    )
    selected_weights = tf.gather(component_weights, step["ancestor_indices"])
    recomposed_mean = tf.reduce_sum(
        selected_weights[:, :, tf.newaxis] * selected_component_means, axis=1
    )
    centered = selected_component_means - recomposed_mean[:, tf.newaxis, :]
    recomposed_covariance = tf.reduce_sum(
        selected_weights[:, :, tf.newaxis, tf.newaxis]
        * (
            tf.matmul(
                selected_component_cholesky,
                selected_component_cholesky,
                transpose_b=True,
            )
            + tf.matmul(centered[:, :, :, tf.newaxis], centered[:, :, tf.newaxis, :])
        ),
        axis=1,
    )
    moment_error = tf.reduce_max(
        tf.abs(
            recomposed_covariance
            - tf.gather(ukf["posterior_covariance"], step["ancestor_indices"])
        )
    )
    reverse = tf.range(component_count - 1, -1, -1)
    permuted_log_q = complete_gaussian_mixture_log_density(
        step["samples"],
        tf.gather(selected_component_means, reverse, axis=1),
        tf.gather(selected_component_cholesky, reverse, axis=1),
        tf.gather(selected_weights, reverse, axis=1),
    )
    label_permutation_error = tf.reduce_max(
        tf.abs(step["complete_log_q"] - permuted_log_q)
    )
    row = {
        "time_index": time_index,
        "component_count": component_count,
        "offset": tf.constant(offset, DTYPE),
        "transformed_observation": transformed,
        "lookahead_log_likelihood": ukf["innovation_log_likelihood"],
        "ancestor_log_probabilities": step["log_ancestor_probabilities"],
        "component_indices": step["component_indices"],
        "component_minimum_eigenvalue": component_minimum,
        "moment_recomposition_max_abs": moment_error,
        "label_permutation_max_abs": label_permutation_error,
        "proposal_density_recomposition_max_abs": tf.reduce_max(
            tf.abs(
                step["complete_log_q"]
                - complete_gaussian_mixture_log_density(
                    step["samples"],
                    selected_component_means,
                    selected_component_cholesky,
                    selected_weights,
                )
            )
        ),
        "posterior_mean": ukf["posterior_mean"],
        "posterior_min_eigenvalue": ukf["posterior_min_eigenvalue"],
        "proposal_finite": step["finite"],
    }
    return step, row, tf.reduce_all(component_minimum > 0.0)


def defensive_step(
    *,
    ukf,
    transformed,
    sampler,
    count,
    dimension,
    component_count,
    offset,
    sign_tensor,
    component_weights,
    log_parent_weights,
    seed,
    time_index,
    nu,
    epsilon_min,
    epsilon_max,
    center,
    temperature,
):
    def words(offset):
        return tf.stack([seed, tf.cast(offset, tf.int64)])

    local_component_count = component_count
    local_weights = component_weights
    posterior_cholesky = ukf["posterior_cholesky"]
    direction_one = posterior_cholesky[:, :, 0]
    direction_two = (
        posterior_cholesky[:, :, 1] if dimension >= 2 else tf.zeros_like(direction_one)
    )
    between_component_covariance = tf.zeros_like(ukf["posterior_covariance"])
    if local_component_count in (2, 4):
        between_component_covariance = offset**2 * tf.einsum(
            "bi,bj->bij", direction_one, direction_one
        )
    if local_component_count == 4:
        between_component_covariance = (
            between_component_covariance
            + offset** 2 * tf.einsum("bi,bj->bij", direction_two, direction_two)
        )
    component_covariance = 0.5 * (
        ukf["posterior_covariance"]
        - between_component_covariance
        + tf.linalg.matrix_transpose(
            ukf["posterior_covariance"] - between_component_covariance
        )
    )
    component_minimum = tf.reduce_min(tf.linalg.eigvalsh(component_covariance), axis=1)
    component_chol_single = tf.linalg.cholesky(component_covariance)
    local_means = ukf["posterior_mean"][:, tf.newaxis, :] + offset * (
        sign_tensor[tf.newaxis, :, 0, tf.newaxis] * direction_one[:, tf.newaxis, :]
        + sign_tensor[tf.newaxis, :, 1, tf.newaxis] * direction_two[:, tf.newaxis, :]
    )
    local_cholesky = tf.broadcast_to(
        component_chol_single[:, tf.newaxis, :, :],
        [count, local_component_count, dimension, dimension],
    )
    if local_component_count == 1:
        local_means = ukf["posterior_mean"][:, tf.newaxis, :]
        local_cholesky = posterior_cholesky[:, tf.newaxis, :, :]
    innovation_solution = tf.linalg.cholesky_solve(
        ukf["innovation_cholesky"], ukf["innovation"][:, :, tf.newaxis]
    )[:, :, 0]
    innovation_quadratic = tf.reduce_sum(
        ukf["innovation"] * innovation_solution, axis=1
    )
    epsilon = epsilon_min + (epsilon_max - epsilon_min) * tf.math.sigmoid(
        (innovation_quadratic - tf.constant(center, DTYPE))
        / tf.constant(temperature, DTYPE)
    )
    defensive_scale = (nu - 2.0) / nu * ukf["posterior_covariance"]
    defensive_cholesky = tf.linalg.cholesky(defensive_scale)
    ancestor_uniforms = philox_uniform_float64([count], words(7100 + 41 * time_index))
    component_uniforms = philox_uniform_float64([count], words(7200 + 43 * time_index))
    gaussian_normal = philox_normal_float64(
        [count, dimension], words(7300 + 47 * time_index)
    )
    student_normal = philox_normal_float64(
        [count, dimension], words(7400 + 53 * time_index)
    )
    student_chi_square = philox_gamma_float64(
        [count],
        words(7500 + 59 * time_index),
        alpha=tf.constant(nu / 2.0, DTYPE),
        beta=tf.constant(0.5, DTYPE),
    )
    step = sampler(
        ukf["posterior_covariance"],
        local_means,
        local_cholesky,
        local_weights,
        ukf["posterior_mean"],
        defensive_cholesky,
        epsilon,
        ukf["innovation_log_likelihood"],
        log_parent_weights,
        ancestor_uniforms,
        component_uniforms,
        gaussian_normal,
        student_normal,
        student_chi_square,
    )
    selected_local_means = tf.gather(local_means, step["ancestor_indices"])
    selected_local_cholesky = tf.gather(local_cholesky, step["ancestor_indices"])
    selected_local_weights = tf.gather(local_weights, step["ancestor_indices"])
    selected_defensive_mean = tf.gather(ukf["posterior_mean"], step["ancestor_indices"])
    selected_defensive_cholesky = tf.gather(
        defensive_cholesky, step["ancestor_indices"]
    )
    selected_epsilon = tf.gather(epsilon, step["ancestor_indices"])
    local_log_q = tf.math.log1p(
        -selected_epsilon
    ) + complete_gaussian_mixture_log_density(
        step["samples"],
        selected_local_means,
        selected_local_cholesky,
        selected_local_weights,
    )
    independent_log_q = tf.reduce_logsumexp(
        tf.stack(
            [
                local_log_q,
                tf.math.log(selected_epsilon)
                + student_log_density(
                    step["samples"],
                    selected_defensive_mean,
                    selected_defensive_cholesky,
                    nu,
                ),
            ],
            axis=1,
        ),
        axis=1,
    )
    reversed_local_log_q = tf.math.log1p(
        -selected_epsilon
    ) + complete_gaussian_mixture_log_density(
        step["samples"],
        tf.reverse(selected_local_means, axis=[1]),
        tf.reverse(selected_local_cholesky, axis=[1]),
        tf.reverse(selected_local_weights, axis=[1]),
    )
    permuted_log_q = tf.reduce_logsumexp(
        tf.stack(
            [
                reversed_local_log_q,
                tf.math.log(selected_epsilon)
                + student_log_density(
                    step["samples"],
                    selected_defensive_mean,
                    selected_defensive_cholesky,
                    nu,
                ),
            ],
            axis=1,
        ),
        axis=1,
    )
    local_mean_recomposed = tf.reduce_sum(
        local_weights[:, :, tf.newaxis] * local_means, axis=1
    )
    local_centered = local_means - local_mean_recomposed[:, tf.newaxis, :]
    local_covariance_recomposed = tf.reduce_sum(
        local_weights[:, :, tf.newaxis, tf.newaxis]
        * (
            tf.matmul(local_cholesky, local_cholesky, transpose_b=True)
            + tf.matmul(
                local_centered[:, :, :, tf.newaxis], local_centered[:, :, tf.newaxis, :]
            )
        ),
        axis=1,
    )
    local_moment_error = tf.reduce_max(
        tf.abs(local_covariance_recomposed - ukf["posterior_covariance"])
    )
    row = {
        "time_index": time_index,
        "local_component_count": local_component_count,
        "offset": tf.constant(offset, DTYPE),
        "nu": tf.constant(nu, DTYPE),
        "epsilon_min": tf.constant(epsilon_min, DTYPE),
        "epsilon_max": tf.constant(epsilon_max, DTYPE),
        "gate_center": tf.constant(center, DTYPE),
        "gate_temperature": tf.constant(temperature, DTYPE),
        "epsilon": epsilon,
        "epsilon_spread": tf.reduce_max(epsilon) - tf.reduce_min(epsilon),
        "epsilon_mean": tf.reduce_mean(epsilon),
        "innovation_quadratic": innovation_quadratic,
        "student_selected_fraction": tf.reduce_mean(
            tf.cast(step["student_selected"], DTYPE)
        ),
        "lookahead_log_likelihood": ukf["innovation_log_likelihood"],
        "ancestor_log_probabilities": step["log_ancestor_probabilities"],
        "posterior_mean": ukf["posterior_mean"],
        "posterior_min_eigenvalue": ukf["posterior_min_eigenvalue"],
        "component_minimum_eigenvalue": component_minimum,
        "local_moment_recomposition_max_abs": local_moment_error,
        "label_permutation_max_abs": tf.reduce_max(
            tf.abs(step["complete_log_q"] - permuted_log_q)
        ),
        "proposal_density_recomposition_max_abs": tf.reduce_max(
            tf.abs(step["complete_log_q"] - independent_log_q)
        ),
        "proposal_finite": step["finite"],
    }
    return step, row, tf.reduce_all(component_minimum > 0.0)
