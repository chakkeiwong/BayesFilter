"""Analytic TF/XLA reference controls for the two known Gaussian mixtures.

Oracle samples/maps here are diagnostic controls, not learned transports and
not q20 implementations. Exact expressions refer to real arithmetic; numerical
inversion is independently checked before use.
"""
from __future__ import annotations

import hashlib
import math
import tensorflow as tf
import tensorflow_probability as tfp

F64 = tf.float64
NORMAL = tfp.distributions.Normal(tf.constant(0., F64), tf.constant(1., F64))
LOG_WEIGHTS = tf.math.log(tf.constant([1/3, 2/3], F64))
CENTERS = tf.constant([-5., 5.], F64)


def marginal_log_prob(x):
    return tf.reduce_logsumexp(NORMAL.log_prob(x[..., None]-CENTERS)+LOG_WEIGHTS, -1)


def marginal_cdf(x):
    return tf.reduce_sum(NORMAL.cdf(x[..., None]-CENTERS)*tf.exp(LOG_WEIGHTS), -1)


@tf.custom_gradient
def exact_marginal(z):
    """Bisection with log-tail comparisons and implicit first derivative."""
    log_cdf = NORMAL.log_cdf(z)
    log_sf = NORMAL.log_survival_function(z)
    def body(i, lo, hi):
        mid = (lo+hi)/2
        cdf = tf.reduce_logsumexp(NORMAL.log_cdf(mid[..., None]-CENTERS)+LOG_WEIGHTS, -1)
        sf = tf.reduce_logsumexp(NORMAL.log_survival_function(mid[..., None]-CENTERS)+LOG_WEIGHTS, -1)
        below = tf.where(z < 0., cdf < log_cdf, sf > log_sf)
        return i+1, tf.where(below, mid, lo), tf.where(below, hi, mid)
    _, lo, hi = tf.while_loop(lambda i, *_: i < 64, body,
                             (tf.constant(0), z-5., z+5.))
    x = (lo+hi)/2
    def gradient(upstream):
        # dF(T(z))/dz = phi(z). This control validates the first derivative.
        # Nested derivatives through this root-finder wrapper are not certified;
        # higher-order consumers need the separate implicit identities.
        return upstream*tf.exp(NORMAL.log_prob(z)-marginal_log_prob(x))
    return x, gradient


class ExactMixtureTransport:
    parameter_dim = 2

    def __init__(self, warped=False):
        self.warped = bool(warped)
        spec = [tf.TensorSpec([None, 2], F64)]
        self.forward_batch = tf.function(self._forward, input_signature=spec, jit_compile=True, autograph=False)
        self.check = tf.function(self._check, input_signature=spec, jit_compile=True, autograph=False)

    def _forward(self, z):
        first = exact_marginal(z[:, 0])
        second = z[:, 1]+(.1*(first**2-26.) if self.warped else 0.)
        return tf.stack((first, second), 1)

    def decode(self, samples):
        shape = tf.shape(samples)
        return tf.reshape(self.forward_batch(tf.reshape(samples, [-1, 2])), shape)

    def _check(self, z):
        from bayesfilter.testing.neutra_warm_start_targets_tf import WarmStartTarget
        # The numerical expression includes both density and log determinant.
        with tf.GradientTape() as tape:
            tape.watch(z)
            x = self._forward(z)
            ld = NORMAL.log_prob(z[:, 0])-marginal_log_prob(x[:, 0])
            target = WarmStartTarget('warped_mixture' if self.warped else 'mixture')
            lp = target.log_prob_kernel(x)+ld
        score = tape.gradient(lp, z)
        base = tf.reduce_sum(NORMAL.log_prob(z), 1)
        return x, lp-base, score+z, marginal_cdf(x[:, 0])-NORMAL.cdf(z[:, 0])


class StandardGaussianReference:
    """Exact latent target for a separately labeled Gaussian HMC control."""
    parameter_dim = 2
    name = 'standard_gaussian_reference'
    specification = {'schema': 'bayesfilter.standard_gaussian_reference.v1', 'dimension': 2}
    signature = hashlib.sha256(b'bayesfilter.standard_gaussian_reference.v1/2').hexdigest()

    @tf.function(input_signature=[tf.TensorSpec([None, 2], F64)], jit_compile=True, autograph=False)
    def value_score(self, x):
        values = tf.reduce_sum(NORMAL.log_prob(x), 1)
        return values, -x, tf.math.is_finite(values)


def oracle_bank(target, count_per_stratum, seed):
    """Five exact disjoint conditional banks; probability weights preserved."""
    bounds = tf.constant([-math.inf, -2., -.1, .1, 2., math.inf], F64)
    cdf = marginal_cdf(bounds)
    mass = cdf[1:]-cdf[:-1]
    transport = ExactMixtureTransport(target.name == 'warped_mixture')
    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
    def run(s):
        u = tf.random.stateless_uniform([5, count_per_stratum], s, dtype=F64)
        prob = cdf[:-1, None]+mass[:, None]*u
        z1 = NORMAL.quantile(tf.reshape(prob, [-1]))
        z2 = tf.random.stateless_normal([5*count_per_stratum], tf.random.experimental.stateless_fold_in(s, 1), dtype=F64)
        rows = transport.forward_batch(tf.stack((z1, z2), 1))
        weights = tf.repeat(tf.math.log(mass)-tf.math.log(tf.cast(count_per_stratum, F64)), count_per_stratum)
        return rows, weights
    return run(tf.constant(seed, tf.int32))
