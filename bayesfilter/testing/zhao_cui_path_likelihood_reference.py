"""Independent CPU/reference path likelihoods using canonical TF value callbacks.

No gradient, score, autodiff, filter, tuning or runtime admission is implemented
here. The quadratic-reference harness integrates these values over saved paths.
"""
from __future__ import annotations
import math
import tensorflow as tf
from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec


def make_path_log_joint(model_name: str, horizon: int, *, jit_compile: bool = True):
    """Return a stable-signature float64 graph for log p_theta(x_0:T,y_1:T).

    Both supported transitions are time homogeneous; flattening paths and time
    calls the actual canonical transition endpoint without changing its law.
    The model factory runs at trace time with symbolic theta, retaining covariance
    parameter dependence. No density is differentiated.
    """
    spec = NonlinearSQMCSpec(model_name)
    if horizon < 1:
        raise ValueError('positive horizon required')
    dimension, observed = spec.dimension, spec.observation_dimension
    mean = spec.initial_mean(tf.float64)
    log_two_pi = tf.constant(math.log(2.0 * math.pi), tf.float64)

    def gaussian(residual, covariance):
        cholesky = tf.linalg.cholesky(covariance)
        standardized = tf.linalg.triangular_solve(cholesky, tf.transpose(residual))
        return -0.5 * (tf.cast(tf.shape(residual)[1], tf.float64) * log_two_pi
                       + 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(cholesky)))
                       + tf.reduce_sum(tf.square(standardized), axis=0))

    @tf.function(input_signature=[
        tf.TensorSpec([spec.parameter_count], tf.float64),
        tf.TensorSpec([None, horizon + 1, dimension], tf.float64),
        tf.TensorSpec([horizon, observed], tf.float64),
    ], jit_compile=jit_compile, autograph=False)
    def log_joint(theta, paths, observations):
        model, _ = spec.model(theta, tf.zeros_like(theta))
        previous = tf.reshape(paths[:, :-1, :], [-1, dimension])
        current = tf.reshape(paths[:, 1:, :], [-1, dimension])
        initial = -0.5 * (dimension * log_two_pi
                          + tf.reduce_sum(tf.square(paths[:, 0, :] - mean), axis=1))
        transition = gaussian(current - model.transition_mean_fn(theta, previous),
                              model.process_covariance)
        predicted_observations = tf.reshape(model.observation_fn(current), [-1, horizon, observed])
        residual = tf.reshape(observations[None, :, :] - predicted_observations, [-1, observed])
        likelihood = gaussian(residual, model.observation_covariance)
        return initial + tf.reduce_sum(tf.reshape(transition + likelihood, [-1, horizon]), axis=1)

    return log_joint
