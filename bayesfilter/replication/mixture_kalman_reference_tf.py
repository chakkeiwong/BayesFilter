"""Exact reference for Younis--Sudderth Appendix B equations (20)--(22).

Diagnostic/reference only: enumerating 2**H paths is restricted to H <= 10.
The supplied Gaussian initial state and transition-before-observation timing
are explicit model assumptions, not recovered numerical author settings.
TensorFlow implements batched conditional Kalman recursions and exact terminal
mixture density. This module must not be represented as the learned KDM filter.
"""
import math
import tensorflow as tf


class BimodalLinearMixtureKalmanReference:
    def __init__(self, *, batch, horizon, sigma, gamma, initial_mean,
                 initial_variance, jit_compile=True):
        if int(horizon) != horizon or int(batch) != batch or not (1 <= horizon <= 10) or batch < 1:
            raise ValueError('Exact reference requires positive batch and 1 <= H <= 10')
        if not all(math.isfinite(v) for v in (sigma, gamma, initial_mean, initial_variance)):
            raise ValueError('Nonfinite model setting')
        if sigma <= 0 or gamma <= 0 or initial_variance <= 0:
            raise ValueError('Reference requires positive variances')
        self.batch, self.horizon = int(batch), int(horizon)
        self.paths = 2**self.horizon
        self.sigma, self.gamma = float(sigma), float(gamma)
        self.mean, self.variance = float(initial_mean), float(initial_variance)
        signature = [tf.TensorSpec([7], tf.float64),
                     tf.TensorSpec([self.batch, self.horizon], tf.float64),
                     tf.TensorSpec([self.batch, self.horizon], tf.float64),
                     tf.TensorSpec([self.batch], tf.float64)]
        self.evaluate = tf.function(self._evaluate, input_signature=signature,
                                    jit_compile=jit_compile, autograph=False)
        self.value_and_gradient = tf.function(self._value_and_gradient,
                                              input_signature=signature,
                                              jit_compile=jit_compile, autograph=False)

    def _evaluate(self, theta, actions, observations, terminal_states):
        a, b, c1, c2, offset1, offset2, v = tf.unstack(theta)
        d = theta.dtype
        shape = [self.batch, self.paths]
        means = tf.fill(shape, tf.constant(self.mean, d))
        variances = tf.fill(shape, tf.constant(self.variance, d))
        logweights = tf.zeros(shape, d)
        indices = tf.range(self.paths, dtype=tf.int32)
        slopes = tf.stack([c1, c2])
        offsets = tf.stack([offset1, offset2])
        component_logweights = tf.stack([-tf.nn.softplus(v), -tf.nn.softplus(-v)])
        log2pi = tf.constant(math.log(2*math.pi), d)
        q = tf.constant(self.sigma**2, d)
        r = tf.constant(self.gamma**2, d)

        def step(t, means, variances, logweights):
            component = tf.bitwise.bitwise_and(tf.bitwise.right_shift(indices, t), 1)
            c = tf.gather(slopes, component)[None, :]
            offset = tf.gather(offsets, component)[None, :]
            predicted_mean = a*means+b*actions[:, t, None]
            predicted_variance = a*a*variances+q
            residual = observations[:, t, None]-c*predicted_mean-offset
            innovation_variance = c*c*predicted_variance+r
            means = predicted_mean+predicted_variance*c*residual/innovation_variance
            variances = predicted_variance*r/innovation_variance
            logweights += (tf.gather(component_logweights, component)[None, :]
                           -.5*(log2pi+tf.math.log(innovation_variance)
                                +residual*residual/innovation_variance))
            return t+1, means, variances, logweights

        _, means, variances, logweights = tf.while_loop(
            lambda t, *_:t < self.horizon, step,
            (tf.constant(0), means, variances, logweights), parallel_iterations=1, maximum_iterations=self.horizon)
        logz = tf.reduce_logsumexp(logweights, axis=-1)
        normalized = logweights-logz[:, None]
        delta = terminal_states[:, None]-means
        logdensity = tf.reduce_logsumexp(
            normalized-.5*(log2pi+tf.math.log(variances)+delta*delta/variances), axis=-1)
        valid = tf.logical_and(tf.reduce_all(variances > 0), tf.reduce_all(tf.math.is_finite(logdensity)))
        valid = tf.logical_and(valid, tf.reduce_all(tf.math.is_finite(normalized)))
        return dict(means=means, variances=variances, logweights=normalized,
                    logZ=logz, terminal_logdensity=logdensity,
                    loss=-tf.reduce_mean(logdensity), valid=valid)

    def _value_and_gradient(self, theta, actions, observations, terminal_states):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = self._evaluate(theta, actions, observations, terminal_states)
        gradient = tape.gradient(result['loss'], theta)
        result['gradient'] = gradient
        result['valid'] = tf.logical_and(result['valid'], tf.reduce_all(tf.math.is_finite(gradient)))
        return result
