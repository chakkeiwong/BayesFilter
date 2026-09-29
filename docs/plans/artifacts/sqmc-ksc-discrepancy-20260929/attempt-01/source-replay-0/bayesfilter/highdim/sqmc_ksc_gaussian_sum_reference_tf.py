"""Independent diagnostic full-seven-mixture Kalman reference.

All seven observation components branch at every update. Posterior Gaussian
sums are projected onto a fixed quadrature grid before prediction. Finite-grid
results require refinement; this is not an exact fixed-component filter and
is never called by a particle/candidate runtime. Analytic sensitivities include
all Gaussian parameters, branch weights, and projection normalization.
"""
from __future__ import annotations

import functools
import math
import tensorflow as tf
from bayesfilter.highdim.sqmc_ksc_tf import WEIGHTS, MEANS, VARIANCES, DTYPE


@functools.lru_cache(maxsize=16)
def gaussian_sum_reference(nodes=1201, bound=48., jit_compile=True):
    if nodes < 3 or bound <= 0:
        raise ValueError('A positive extent and at least three nodes are required')

    @tf.function(input_signature=[tf.TensorSpec([2], DTYPE),
                                  tf.TensorSpec([None, 1], DTYPE)],
                 jit_compile=jit_compile, autograph=False)
    def compute(theta, observations):
        x = tf.linspace(tf.constant(-bound, DTYPE), tf.constant(bound, DTYPE), nodes)
        dx = tf.concat([tf.constant([.5], DTYPE), tf.ones([nodes-2], DTYPE),
                        tf.constant([.5], DTYPE)], 0) * (2.*bound/(nodes-1))
        gamma = .5 * (1. + tf.math.erf(theta[0] / tf.sqrt(tf.constant(2., DTYPE))))
        phi = tf.exp(-.5*tf.square(theta[0])) / math.sqrt(2.*math.pi)
        p = tf.constant(WEIGHTS, DTYPE)
        offset = tf.constant(MEANS, DTYPE)
        noise_var = tf.constant(VARIANCES, DTYPE)
        beta_direction = tf.constant([0., 2.], DTYPE)

        def branch(alpha, dalpha, mean, dmean, variance, dvariance, z):
            residual = z - 2.*theta[1] - mean[:, None] - offset[None, :]
            innovation_var = variance[:, None] + noise_var[None, :]
            dresidual = -dmean[:, None, :] - beta_direction
            dinnovation_var = dvariance[:, None, :]
            log_l = -.5*(tf.square(residual)/innovation_var +
                          tf.math.log(innovation_var) + math.log(2.*math.pi))
            # Common scale cancels in normalized weights and their derivatives.
            shift = tf.reduce_max(log_l)
            l = tf.exp(log_l-shift) * p[None, :]
            dlog_l = -residual[:, :, None]/innovation_var[:, :, None]*dresidual
            dlog_l += .5*(tf.square(residual)/tf.square(innovation_var)-1./innovation_var)[:, :, None]*dinnovation_var
            q = alpha[:, None]*l
            dq = dalpha[:, None, :]*l[:, :, None] + q[:, :, None]*dlog_l
            normalizer = tf.reduce_sum(q)
            dlogz = tf.reduce_sum(dq, axis=[0, 1])/normalizer
            weights = q/normalizer
            dweights = dq/normalizer - weights[:, :, None]*dlogz
            gain = variance[:, None]/innovation_var
            dgain = dvariance[:, None, :]/innovation_var[:, :, None] - variance[:, None, None]*dinnovation_var/tf.square(innovation_var[:, :, None])
            post_mean = mean[:, None] + gain*residual
            dpost_mean = dmean[:, None, :] + dgain*residual[:, :, None] + gain[:, :, None]*dresidual
            post_var = variance[:, None]*noise_var[None, :]/innovation_var
            dpost_var = tf.square(noise_var[None, :]/innovation_var)[:, :, None]*dvariance[:, None, :]
            valid = tf.logical_and(normalizer > 0., tf.reduce_all(post_var > 0.))
            return (tf.reshape(weights, [-1]), tf.reshape(dweights, [-1, 2]),
                    tf.reshape(post_mean, [-1]), tf.reshape(dpost_mean, [-1, 2]),
                    tf.reshape(post_var, [-1]), tf.reshape(dpost_var, [-1, 2]),
                    tf.math.log(normalizer)+shift, dlogz, valid)

        def project(weights, dweights, mean, dmean, variance, dvariance):
            delta = x[:, None]-mean[None, :]
            inverse_var = 1./variance[None, :]
            density = tf.exp(-.5*tf.square(delta)*inverse_var) / tf.sqrt(2.*math.pi*variance[None, :])
            posterior = tf.linalg.matvec(density, weights)
            derivative = tf.matmul(density, dweights)
            derivative += tf.matmul(density*delta*inverse_var, weights[:, None]*dmean)
            derivative += tf.matmul(.5*density*(tf.square(delta*inverse_var)-inverse_var), weights[:, None]*dvariance)
            masses = posterior*dx
            dmasses = derivative*dx[:, None]
            mass = tf.reduce_sum(masses)
            dmass = tf.reduce_sum(dmasses, axis=0)
            normalized = masses/mass
            dnormalized = dmasses/mass-normalized[:, None]*dmass/mass
            valid = tf.logical_and(mass > 0., tf.reduce_all(tf.math.is_finite(dnormalized)))
            return normalized, dnormalized, tf.abs(mass-1.), valid

        first = branch(tf.ones([1], DTYPE), tf.zeros([1, 2], DTYPE),
                       tf.zeros([1], DTYPE), tf.zeros([1, 2], DTYPE),
                       (1.+gamma*gamma)[None], tf.reshape(tf.stack([2.*gamma*phi, 0.]), [1, 2]), observations[0, 0])
        fw, fdw, fm, fdm, fv, fdv, first_value, first_score, first_valid = first
        initial_alpha, initial_dalpha, initial_mass_error, initial_valid = project(fw, fdw, fm, fdm, fv, fdv)
        first_mean = tf.reduce_sum(fw*fm)
        first_var = tf.reduce_sum(fw*(fv+tf.square(fm)))-first_mean**2

        def body(t, alpha, dalpha, value, score, max_mass_error, moments, valid):
            w, dw, m, dm, v, dv, increment, d_increment, branch_valid = branch(
                alpha, dalpha, gamma*x, tf.stack([phi*x, tf.zeros_like(x)], 1),
                tf.ones_like(x), tf.zeros([nodes, 2], DTYPE), observations[t, 0])
            posterior_mean = tf.reduce_sum(w*m)
            posterior_var = tf.reduce_sum(w*(v+tf.square(m)))-posterior_mean**2
            new_moments = tf.stack([posterior_mean, posterior_var])
            new_alpha, new_dalpha, mass_error, projection_valid = project(w, dw, m, dm, v, dv)
            value += increment
            score += d_increment
            valid = valid & branch_valid & projection_valid & tf.math.is_finite(value)
            valid = valid & tf.reduce_all(tf.math.is_finite(score)) & tf.reduce_all(tf.math.is_finite(new_moments))
            return (t+1, new_alpha, new_dalpha, value, score,
                    tf.maximum(max_mass_error, mass_error), new_moments, valid)

        result = tf.while_loop(lambda t, *_: t < tf.shape(observations)[0], body,
                               (tf.constant(1), initial_alpha, initial_dalpha, first_value,
                                first_score, initial_mass_error, tf.stack([first_mean, first_var]),
                                first_valid & initial_valid & tf.math.is_finite(first_value)
                                & tf.reduce_all(tf.math.is_finite(first_score))),
                               parallel_iterations=1)
        return dict(value=result[3], score=result[4], maximum_projection_mass_error=result[5],
                    posterior_mean=result[6][0], posterior_variance=result[6][1],
                    valid=result[7], observation_components=tf.constant(7),
                    maximum_gaussian_branches=tf.where(tf.shape(observations)[0] > 1, 7*nodes, 7))

    return compute
