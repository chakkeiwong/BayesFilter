"""Normalized scalar Gaussian-proposal particle filters.

EKF and positive three-node Gaussian quadrature give proposal moments only.
The filter always weights with the original transition and observation density
divided by that normalized proposal. Native repetition batches share one traced
time loop. This is a classical comparator, not a controlled-SMC author route.
"""
from __future__ import annotations
import math
import tensorflow as tf
from bayesfilter.replication.controlled_smc_tf import systematic_ancestors


def normal_log_density(x, mean, variance):
    return -.5*(math.log(2*math.pi)+tf.math.log(variance)+(x-mean)**2/variance)


def binomial_moments(trials):
    def mean(x):
        return trials*tf.math.sigmoid(x)
    def variance(x):
        # Multiplying sigmoid(x) by sigmoid(-x) preserves the positive tail.
        return trials*tf.math.sigmoid(x)*tf.math.sigmoid(-x)
    return mean, variance, variance


class ScalarGaussianGuidedPF:
    def __init__(self, *, observations, particles, replicas, alpha,
                 initial_mean, initial_variance, observation_log_density,
                 observation_mean, observation_variance, observation_derivative,
                 guide, jit_compile=True):
        if guide not in ('ekf', 'quadrature3'):
            raise ValueError('Unknown guide')
        if min(observations, particles, replicas) < 1 or initial_variance <= 0:
            raise ValueError('Invalid dimensions or prior variance')
        self.h, self.n, self.b = int(observations), int(particles), int(replicas)
        self.alpha, self.m0, self.v0 = float(alpha), float(initial_mean), float(initial_variance)
        self.logg = observation_log_density
        self.hm, self.rv, self.dh = observation_mean, observation_variance, observation_derivative
        self.guide, self.dtype = guide, tf.float64
        d, h, n, b = self.dtype, self.h, self.n, self.b
        self.forward = tf.function(self._forward, input_signature=[
            tf.TensorSpec([h, b], d), tf.TensorSpec([h, b, n], d),
            tf.TensorSpec([h, b], d), tf.TensorSpec([], d)],
            jit_compile=jit_compile, autograph=False)
        self.proposal = tf.function(self._proposal, input_signature=[
            tf.TensorSpec([b, n], d), tf.TensorSpec([], d), tf.TensorSpec([b], d)],
            jit_compile=jit_compile, autograph=False)

    def _proposal(self, mean, variance, y):
        if self.guide == 'ekf':
            predicted_y, r, slope = self.hm(mean), self.rv(mean), self.dh(mean)
            innovation_variance = variance*slope*slope+r
            cross = variance*slope
            conditional_variance = variance*r/innovation_variance
        else:
            nodes = mean[..., None]+tf.sqrt(variance)*tf.constant(
                [-math.sqrt(3), 0., math.sqrt(3)], self.dtype)
            weights = tf.constant([1/6, 2/3, 1/6], self.dtype)
            responses = self.hm(nodes)
            predicted_y = tf.reduce_sum(weights*responses, axis=-1)
            centered = responses-predicted_y[..., None]
            innovation_variance = tf.reduce_sum(weights*(centered**2+self.rv(nodes)), axis=-1)
            cross = tf.reduce_sum(weights*(nodes-mean[..., None])*centered, axis=-1)
            conditional_variance = variance-cross*cross/innovation_variance
        conditional_mean = mean+cross/innovation_variance*(y[:, None]-predicted_y)
        relative_margin = conditional_variance/variance
        valid = tf.reduce_all(tf.logical_and(
            tf.math.is_finite(conditional_mean), tf.math.is_finite(conditional_variance)))
        valid &= tf.reduce_all(innovation_variance > 0)
        valid &= tf.reduce_all(relative_margin > 64*2.220446049250313e-16)
        valid &= variance > 0
        return dict(mean=conditional_mean, variance=conditional_variance,
                    relative_margin=relative_margin, valid=valid)

    def _forward(self, y, normal, uniform, process_variance):
        h, n, b, d = self.h, self.n, self.b, self.dtype
        names = ('logZ', 'mean', 'variance', 'ess', 'proposal_variance_min', 'relative_margin_min')
        arrays = tuple(tf.TensorArray(d, size=h, element_shape=[b]) for _ in names)
        clouds = tf.TensorArray(d, size=h, element_shape=[b, n])
        ancestors = tf.TensorArray(tf.int32, size=h, element_shape=[b, n])
        potentials = tf.TensorArray(d, size=h, element_shape=[b, n])
        valid = (tf.reduce_all(tf.math.is_finite(y)) &
                 tf.reduce_all(tf.math.is_finite(normal)) &
                 tf.reduce_all((uniform >= 0) & (uniform < 1)) &
                 tf.math.is_finite(process_variance) & (process_variance > 0))

        def step(t, previous, total, valid, arrays, clouds, ancestors, potentials):
            mean = tf.cond(t == 0, lambda: tf.fill([b, n], tf.constant(self.m0, d)),
                           lambda: self.alpha*previous)
            variance = tf.where(t == 0, tf.constant(self.v0, d), process_variance)
            guide = self._proposal(mean, variance, y[t])
            x = guide['mean']+tf.sqrt(guide['variance'])*normal[t]
            logw = (self.logg(y[t, :, None], x)+normal_log_density(x, mean, variance)
                    -normal_log_density(x, guide['mean'], guide['variance']))
            weights = tf.nn.softmax(logw, axis=-1)
            total += tf.reduce_logsumexp(logw, axis=-1)-tf.math.log(tf.cast(n, d))
            filtered_mean = tf.reduce_sum(weights*x, axis=-1)
            filtered_variance = tf.reduce_sum(weights*(x-filtered_mean[:, None])**2, axis=-1)
            values = (total, filtered_mean, filtered_variance,
                      1/tf.reduce_sum(weights**2, axis=-1),
                      tf.reduce_min(guide['variance'], axis=-1),
                      tf.reduce_min(guide['relative_margin'], axis=-1))
            arrays = tuple(a.write(t, v) for a, v in zip(arrays, values))
            indices = systematic_ancestors(weights, uniform[t])
            valid &= guide['valid'] & tf.reduce_all(tf.math.is_finite(logw))
            return (t+1, tf.gather(x, indices, batch_dims=1), total, valid, arrays,
                    clouds.write(t, x), ancestors.write(t, indices), potentials.write(t, logw))

        _, _, _, valid, arrays, clouds, ancestors, potentials = tf.while_loop(
            lambda t, *_: t < h, step,
            (tf.constant(0), tf.zeros([b, n], d), tf.zeros([b], d), valid,
             arrays, clouds, ancestors, potentials), parallel_iterations=1)
        result = dict(zip(names, (a.stack() for a in arrays)))
        result.update(valid=valid, x=clouds.stack(), ancestors=ancestors.stack(),
                      log_incremental_weight=potentials.stack())
        return result
