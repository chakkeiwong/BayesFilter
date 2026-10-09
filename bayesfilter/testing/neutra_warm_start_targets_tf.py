"""Analytic benchmark/reference targets for the warm-start research campaign.

These fixtures are not a q20 evaluator or a promoted training default. The
wiggle is the exact pinned flonaco test_wiggle.py potential, whose normalizer
is determined separately. All target evaluations preserve the leading batch.
"""
from __future__ import annotations

import hashlib
import json
import math

import tensorflow as tf
import tensorflow_probability as tfp


TARGET_NAMES = ("gaussian", "mixture", "warped_mixture", "wiggle", "funnel")
F64 = tf.float64


class WarmStartTarget:
    def __init__(self, name, *, jit_compile=True):
        if name not in TARGET_NAMES:
            raise ValueError(f"unknown benchmark {name}")
        self.name = name
        self.parameter_dim = 10 if name == "funnel" else 2
        self.specification = {
            "schema": "bayesfilter.warm_start_target.v1", "name": name,
            "dimension": self.parameter_dim, "dtype": "float64",
            "gaussian_mean": [1., -1.], "gaussian_eigenvalues": [1., 25.],
            "mixture_means": [[-5., 0.], [5., 0.]], "mixture_weights": [1/3, 2/3],
            "warp_curvature": .1, "warp_center": 26.,
            "wiggle_source": "flonaco@6b9286b4e58194aa65373200d7bfacde06a2d180",
            "wiggle_mean": [6., 0.], "wiggle_ring_mean": 5., "wiggle_ring_var": 8.,
            "funnel_v_sd": 1., "funnel_child_log_sd": "v",
        }
        self.signature = hashlib.sha256(json.dumps(
            self.specification, sort_keys=True).encode()).hexdigest()
        # Pure tensor body for composition under a caller's tf.function. This
        # is the identical density, not an eager public evaluation route.
        self.log_prob_kernel = self._log_prob
        spec = [tf.TensorSpec([None, self.parameter_dim], F64)]
        self.log_prob = tf.function(self._log_prob, input_signature=spec,
                                    jit_compile=jit_compile, autograph=False)
        self.value_score = tf.function(self._value_score, input_signature=spec,
                                       jit_compile=jit_compile, autograph=False)
        self.hessian = tf.function(self._hessian, input_signature=spec,
                                   jit_compile=jit_compile, autograph=False)

    def _log_prob(self, x):
        c = tf.constant(math.log(2*math.pi), F64)
        if self.name == "gaussian":
            centered = x - tf.constant([1., -1.], F64)
            a = (centered[:, 0] + centered[:, 1])/tf.sqrt(tf.constant(2., F64))
            b = (centered[:, 0] - centered[:, 1])/tf.sqrt(tf.constant(2., F64))
            return -.5*(a*a + b*b/25. + 2*c + tf.math.log(tf.constant(25., F64)))
        if self.name in ("mixture", "warped_mixture"):
            if self.name == "warped_mixture":
                x = tf.stack((x[:, 0], x[:, 1] - .1*(x[:, 0]**2 - 26.)), axis=1)
            d = x[:, None, :] - tf.constant([[-5., 0.], [5., 0.]], F64)[None, :, :]
            return tf.reduce_logsumexp(-.5*tf.reduce_sum(d*d, axis=-1) - c
                + tf.math.log(tf.constant([1/3, 2/3], F64))[None, :], axis=1)
        if self.name == "wiggle":
            # Preserve upstream broadcasting: both coordinates use x_0-sin(x_1).
            r = x[:, 0] - tf.sin(x[:, 1])
            return -.5*((r-6.)**2+r*r) - c - (tf.linalg.norm(x, axis=1)-5.)**2/16.
        v, children = x[:, 0], x[:, 1:]
        return -.5*(v*v+c) - tf.reduce_sum(.5*(children*tf.exp(-v[:, None]))**2
                                          + v[:, None] + .5*c, axis=1)

    def _value_score(self, x):
        with tf.GradientTape() as tape:
            tape.watch(x)
            values = self._log_prob(x)
        score = tape.gradient(values, x)
        valid = tf.math.is_finite(values) & tf.reduce_all(tf.math.is_finite(score), axis=1)
        return values, score, valid

    def _hessian(self, x):
        # A static coordinate loop, not a sample loop or pfor. Differentiating
        # sums is batch-native because the density has no cross-row coupling.
        with tf.GradientTape(persistent=True) as outer:
            outer.watch(x)
            _, score, _ = self._value_score(x)
            totals = [tf.reduce_sum(score[:, j]) for j in range(self.parameter_dim)]
        return tf.stack([outer.gradient(t, x) for t in totals], axis=1)

    def value_score_status(self, x, beta):
        values, score, valid = self.value_score(x)
        return values, score, {"bridge_valid": valid & tf.equal(beta,tf.constant(1.,F64))}

    def reference_sample(self, count, seed):
        if self.name == "wiggle":
            raise ValueError("wiggle has a quadrature reference, not an exact sampler")
        z = tf.random.stateless_normal([count, self.parameter_dim], seed, dtype=F64)
        if self.name == "gaussian":
            return tf.stack(((z[:, 0]+5*z[:, 1])/math.sqrt(2.)+1.,
                             (z[:, 0]-5*z[:, 1])/math.sqrt(2.)-1.), axis=1)
        if self.name in ("mixture", "warped_mixture"):
            uniforms = tf.random.stateless_uniform([count], tf.random.experimental.stateless_fold_in(seed, 1), dtype=F64)
            x = tf.stack((z[:, 0]+tf.where(uniforms < 1/3,
                tf.constant(-5.,F64),tf.constant(5.,F64)), z[:, 1]), axis=1)
            return self.warp(x) if self.name == "warped_mixture" else x
        return tf.concat((z[:, :1], tf.exp(z[:, :1])*z[:, 1:]), axis=1)

    def warp(self, x):
        return tf.stack((x[:, 0], x[:, 1]+.1*(x[:, 0]**2-26.)), axis=1)

    def region_features(self, x):
        if self.name in ("mixture", "warped_mixture"):
            u = x if self.name == "mixture" else tf.stack(
                (x[:, 0], x[:, 1]-.1*(x[:, 0]**2-26.)), axis=1)
            d = u[:, None, :] - tf.constant([[-5., 0.], [5., 0.]], F64)[None, :, :]
            responsibility = tf.nn.softmax(-.5*tf.reduce_sum(d*d, axis=-1)
                        + tf.math.log(tf.constant([1/3, 2/3], F64)), axis=1)
            return tf.concat((responsibility, tf.cast(x[:, :1]>0, F64)), axis=1)
        if self.name == "funnel":
            return tf.cast(tf.stack((x[:, 0]<-1., tf.abs(x[:, 0])<=1., x[:, 0]>1.), axis=1), F64)
        if self.name == "wiggle":
            # Fixed Voronoi reporting regions around the three stationary
            # maxima measured in prepare-wiggle-r1; these are not known masses.
            centers=self.known_representatives()
            labels=tf.argmin(tf.reduce_sum((x[:,None,:]-centers[None,:,:])**2,axis=-1),axis=1)
            return tf.one_hot(labels,3,dtype=F64)
        return tf.cast(tf.stack((x[:, 0]>0., x[:, 1]>0.), axis=1), F64)

    def known_representatives(self):
        if self.name == "gaussian":
            return tf.constant([[1., -1.]], F64)
        if self.name in ("mixture", "warped_mixture"):
            x = tf.constant([[-5., 0.], [5., 0.]], F64)
            return self.warp(x) if self.name == "warped_mixture" else x
        if self.name == "funnel":
            # Representative scale regions, not three density modes or oracle draws.
            return tf.concat((tf.constant([[-1.], [0.], [1.]], F64), tf.zeros([3,9], F64)), axis=1)
        return tf.constant([[3.455148589261057,-3.614131737672104],
            [3.8586905947989782,2.1671483394364097],
            [2.008927176072839,4.578669196636026]],F64)


class BroadStudentProposal:
    """Declared warm-start hypothesis: multivariate t5 with scale 4, full support."""
    def __init__(self, dimension):
        self.dimension = dimension
        self.distribution = tfp.distributions.MultivariateStudentTLinearOperator(
            df=tf.constant(5., F64), loc=tf.zeros([dimension], F64),
            scale=tf.linalg.LinearOperatorScaledIdentity(dimension, multiplier=tf.constant(4., F64)))

    def log_prob(self, x):
        return self.distribution.log_prob(x)

    def sample(self, count, seed):
        return self.distribution.sample(count, seed=seed)


def wiggle_quadrature(target, *, extent, resolution):
    """Midpoint integration reference; caller must compare domain and resolution."""
    dx = 2*extent/resolution
    axis = tf.cast(tf.range(resolution), F64)*dx - extent + dx/2
    a, b = tf.meshgrid(axis, axis, indexing="ij")
    x = tf.stack((tf.reshape(a, [-1]), tf.reshape(b, [-1])), axis=1)
    log_values = target.log_prob(x)
    log_z = tf.reduce_logsumexp(log_values) + 2*tf.math.log(tf.constant(dx, F64))
    log_weights = tf.nn.log_softmax(log_values)
    return x, log_weights, log_z
