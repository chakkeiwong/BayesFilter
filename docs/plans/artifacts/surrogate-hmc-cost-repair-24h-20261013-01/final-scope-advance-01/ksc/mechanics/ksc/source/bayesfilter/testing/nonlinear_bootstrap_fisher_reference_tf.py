"""Independent diagnostic Gaussian bootstrap PF and Fisher-identity score.

This is a finite-particle reference, never the canonical LEDH score or an exact
oracle. Initial N(mean,I) is parameter independent; each observation follows a
transition. The observation mean has no direct theta dependence unless supplied
through the model's explicit log-density/tangent callbacks. See the derivation in
docs/plans/ledh-nonlinear-execution-20261002.md. No autodiff through resampling.
"""
from __future__ import annotations

import math
import tensorflow as tf


def gaussian_log_density(residual, covariance):
    factor = tf.linalg.cholesky(covariance)
    white = tf.linalg.triangular_solve(factor, tf.transpose(residual))
    logdet = 2 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(factor)))
    dimension = tf.cast(tf.shape(residual)[1], residual.dtype)
    return -.5 * (tf.reduce_sum(tf.square(white), axis=0) + logdet
                   + dimension * tf.constant(math.log(2*math.pi), residual.dtype))


def gaussian_parameter_tangent(residual, covariance, d_mean, d_covariance):
    precision_residual = tf.transpose(tf.linalg.solve(covariance, tf.transpose(residual)))
    result = tf.reduce_sum(precision_residual*d_mean, axis=1)
    if d_covariance is not None:
        result += .5 * (tf.einsum('ni,ij,nj->n', precision_residual,
                                  d_covariance, precision_residual)
                         - tf.linalg.trace(tf.linalg.solve(covariance, d_covariance)))
    return result


def complete_increment(model_factory, theta, parents, states, observation):
    """Partial derivative of log f_theta(x|parent)+log g_theta(y|x).

    Both states are held fixed. Transition and observation covariance derivatives
    are retained. The per-coordinate loop has one traced body; no pfor is used.
    """
    p = theta.shape[0]
    values = tf.TensorArray(theta.dtype, size=p)

    def coordinate(k, values):
        model, _ = model_factory(theta, tf.one_hot(k, p, dtype=theta.dtype))
        means = model.transition_mean_fn(theta, parents)
        dmeans = model.transition_mean_tangent_fn(theta, parents, tf.zeros_like(parents))
        if model.transition_log_density_tangent_fn is not None:
            transition_score = model.transition_log_density_tangent_fn(
                theta, states, means, tf.zeros_like(states), dmeans)
        else:
            dq = (model.process_covariance_tangent_fn(theta)
                  if model.process_covariance_tangent_fn is not None else None)
            transition_score = gaussian_parameter_tangent(
                states-means, model.process_covariance, dmeans, dq)
        if model.observation_log_density_tangent_fn is not None:
            observation_score = model.observation_log_density_tangent_fn(
                theta, states, observation, tf.zeros_like(states))
        else:
            predicted = model.observation_fn(states)
            dr = (model.observation_covariance_tangent_fn(theta)
                  if model.observation_covariance_tangent_fn is not None else None)
            observation_score = gaussian_parameter_tangent(
                observation[None, :]-predicted, model.observation_covariance,
                tf.zeros_like(predicted), dr)
        return k+1, values.write(k, transition_score+observation_score)

    _, values = tf.while_loop(lambda k, _: k < p, coordinate, (0, values))
    return tf.transpose(values.stack())


def make_bootstrap_fisher_kernel(model_factory, initial_mean, parameter_count,
                                 observation_dimension, horizon, particles,
                                 *, jit_compile=True):
    """Compile one sequential bootstrap filter with all analytical score columns.

    The estimate is a posterior average of accumulated complete-data scores,
    not the derivative of the random, discontinuous finite PF program. Resampling
    uses iid inverse-CDF uniforms, with O(N) storage rather than an NxN draw.
    """
    mean = tf.convert_to_tensor(initial_mean)
    dtype, d, p, n = mean.dtype, mean.shape[0], parameter_count, particles
    if min(horizon, n, d, p, observation_dimension) < 1:
        raise ValueError('all dimensions/counts must be positive')

    @tf.function(input_signature=[tf.TensorSpec([p], dtype),
                                  tf.TensorSpec([horizon, observation_dimension], dtype),
                                  tf.TensorSpec([], tf.int32)],
                 jit_compile=jit_compile, autograph=False)
    def evaluate(theta, observations, seed):
        model, _ = model_factory(theta, tf.zeros_like(theta))
        qfactor = tf.linalg.cholesky(model.process_covariance)
        states = mean + tf.random.stateless_normal([n,d], [seed,0], dtype=dtype)
        paths = tf.zeros([n,p], dtype)
        trace = tf.TensorArray(dtype, size=horizon)
        score = tf.zeros([p], dtype)
        ancestors = tf.range(n)

        def step(t, parents, paths, ancestors, ell, score, trace):
            means = model.transition_mean_fn(theta, parents)
            noise = tf.random.stateless_normal([n,d], [seed,3*t+1], dtype=dtype)
            states = means + tf.matmul(noise, qfactor, transpose_b=True)
            if model.observation_log_density_fn is None:
                logw = gaussian_log_density(observations[t][None,:]-model.observation_fn(states),
                                             model.observation_covariance)
            else:
                logw = model.observation_log_density_fn(theta, states, observations[t])
            logsum = tf.reduce_logsumexp(logw)
            weights = tf.exp(logw-logsum)
            increment = logsum-tf.math.log(tf.cast(n,dtype))
            paths = paths + complete_increment(model_factory, theta, parents, states, observations[t])
            score = tf.einsum('n,np->p', weights, paths)
            ess = 1/tf.reduce_sum(tf.square(weights))
            ordered = tf.sort(ancestors)
            distinct = 1+tf.reduce_sum(tf.cast(ordered[1:] != ordered[:-1],tf.int32))
            trace = trace.write(t,tf.stack([increment,ess,tf.reduce_max(weights),tf.cast(distinct,dtype)]))
            cdf = tf.concat([tf.cumsum(weights)[:-1],tf.ones([1],dtype)],axis=0)
            uniforms = tf.random.stateless_uniform([n], [seed,3*t+2], dtype=dtype)
            index = tf.searchsorted(cdf, uniforms, side='right')
            return (t+1, tf.gather(states,index), tf.gather(paths,index),
                    tf.gather(ancestors,index), ell+increment, score, trace)

        _, _, _, _, ell, score, trace = tf.while_loop(
            lambda t, *_: t < horizon, step,
            (0, states, paths, ancestors, tf.zeros([],dtype), score, trace))
        return ell, score, trace.stack()

    return evaluate
