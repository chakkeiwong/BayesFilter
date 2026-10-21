"""Independent matrix Kalman reference for the declared linear fixture scopes.

FP64 reference exception. This computes the model likelihood, not a DCTR value
or an analytical DCTR score. Automatic differentiation here differentiates the
complete Kalman endpoint, including its stationary initial covariance.
"""
from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim.ledh_program_fixtures_tf import linear_fixture_spec


def stationary_covariance(transition, process):
    """Solve P_ij - F_ik F_jl P_kl = Q_ij using row-major vectorization."""
    dimension = int(transition.shape[-1])
    product = tf.einsum("bik,bjl->bijkl", transition, transition)
    operator = tf.eye(dimension * dimension, dtype=tf.float64)[None] - tf.reshape(
        product, [-1, dimension * dimension, dimension * dimension])
    covariance = tf.reshape(tf.linalg.solve(operator, tf.reshape(
        process, [-1, dimension * dimension, 1])), [-1, dimension, dimension])
    # The exact solution is symmetric; remove solve roundoff asymmetry only.
    return .5 * (covariance + tf.linalg.matrix_transpose(covariance))


@dataclass(frozen=True)
class LinearReference:
    physical_log_likelihood: object
    physical_statistics: object
    unconstrained_value_score: object
    specification: dict
    model_id: str


def make_linear_reference(model_id):
    spec = linear_fixture_spec(model_id, tf.float64)
    dimension, horizon = spec["d"], spec["t"]
    parameter_count = 2 if spec["infer_process_scale"] else 1
    observation_dimension = int(spec["h"].shape[0])
    lower, upper = tf.constant(spec["lower"], tf.float64), tf.constant(spec["upper"], tf.float64)
    observation_matrix, observation_covariance = spec["h"], spec["r"]
    signature = [tf.TensorSpec([None, parameter_count], tf.float64),
                 tf.TensorSpec([horizon, observation_dimension], tf.float64)]

    def likelihood(theta, observations):
        count = tf.shape(theta)[0]
        transition = theta[:, 0, None, None] * spec["base_f"][None]
        scale = tf.exp(2. * theta[:, 1]) if parameter_count == 2 else tf.ones([count], tf.float64)
        process = scale[:, None, None] * spec["base_q"][None]
        covariance = stationary_covariance(transition, process)
        mean = tf.zeros([count, dimension, 1], tf.float64)
        total = tf.zeros([count], tf.float64)
        identity = tf.eye(dimension, dtype=tf.float64)[None]
        log_two_pi = tf.math.log(tf.constant(2. * 3.141592653589793, tf.float64))
        # One traced native loop; time is not unrolled and rows are batch native.
        def step(index, mean, covariance, total):
            predicted_mean = transition @ mean
            predicted_covariance = transition @ covariance @ tf.linalg.matrix_transpose(transition) + process
            cross = predicted_covariance @ tf.transpose(observation_matrix)
            innovation_covariance = observation_matrix @ cross + observation_covariance
            chol = tf.linalg.cholesky(innovation_covariance)
            innovation = observations[index][None, :, None] - observation_matrix @ predicted_mean
            whitened = tf.linalg.triangular_solve(chol, innovation)
            total -= .5 * (observation_dimension * log_two_pi
                           + 2. * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)), axis=-1)
                           + tf.reduce_sum(tf.square(whitened), axis=[1, 2]))
            gain = tf.linalg.matrix_transpose(tf.linalg.cholesky_solve(chol, tf.linalg.matrix_transpose(cross)))
            mean = predicted_mean + gain @ innovation
            residual = identity - gain @ observation_matrix
            # Joseph's algebraically equivalent form avoids a covariance subtraction.
            covariance = residual @ predicted_covariance @ tf.linalg.matrix_transpose(residual)
            covariance += gain @ observation_covariance @ tf.linalg.matrix_transpose(gain)
            return index + 1, mean, covariance, total
        _, mean, covariance, total = tf.while_loop(
            lambda index, *_: index < horizon, step,
            (tf.constant(0), mean, covariance, total),
            parallel_iterations=1, maximum_iterations=horizon)
        next_mean = observation_matrix @ transition @ mean
        next_covariance = transition @ covariance @ tf.linalg.matrix_transpose(transition) + process
        next_covariance = observation_matrix @ next_covariance @ tf.transpose(observation_matrix) + observation_covariance
        predictive = .5 * tf.math.erfc(-next_mean[:, :, 0] / tf.sqrt(
            2. * tf.linalg.diag_part(next_covariance)))
        return total, predictive

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def physical(theta, observations):
        return likelihood(theta, observations)[0]

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def statistics(theta, observations):
        return likelihood(theta, observations)

    @tf.function(input_signature=signature, jit_compile=True, autograph=False)
    def unconstrained(z, observations):
        with tf.GradientTape() as tape:
            tape.watch(z)
            theta = lower + (upper - lower) * tf.sigmoid(z)
            # Uniform-prior widths cancel the chart's width factors exactly.
            log_density = likelihood(theta, observations)[0] - tf.reduce_sum(
                tf.nn.softplus(z) + tf.nn.softplus(-z), axis=-1)
        return log_density, tape.gradient(log_density, z)

    return LinearReference(physical, statistics, unconstrained, spec, str(model_id))
