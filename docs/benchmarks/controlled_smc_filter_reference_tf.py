"""Independent diagnostic dense-quadrature filter; not a runtime candidate.

The reference integrates ordinary filtering masses by a trapezoid rule. Its
outputs require refinement and support checks before scientific interpretation.
"""
import math
import tensorflow as tf


class GridFilterReference:
    def __init__(self, *, observations, points, bound, alpha, initial_mean,
                 initial_variance, observation_log_density, jit_compile=True):
        self.h, self.g, self.bound = observations, points, bound
        self.alpha, self.m0, self.v0 = alpha, initial_mean, initial_variance
        self.logg, self.dtype = observation_log_density, tf.float64
        self.forward = tf.function(self._forward, input_signature=[
            tf.TensorSpec([observations], self.dtype), tf.TensorSpec([], self.dtype)],
            jit_compile=jit_compile, autograph=False)

    def _forward(self, observations, q):
        d, g, h = self.dtype, self.g, self.h
        grid = tf.linspace(tf.constant(-self.bound, d), tf.constant(self.bound, d), g)
        step = 2*self.bound/(g-1)
        quadrature = tf.concat([tf.constant([.5], d), tf.ones([g-2], d), tf.constant([.5], d)], 0)*step
        transition = tf.exp(-.5*(grid[:, None]-self.alpha*grid[None, :])**2/q)/tf.sqrt(2*math.pi*q)
        transition *= quadrature[:, None]
        prior = tf.exp(-.5*(grid-self.m0)**2/self.v0)/math.sqrt(2*math.pi*self.v0)*quadrature
        names = ('logZ', 'mean', 'variance', 'predictive_mass_loss')
        arrays = tuple(tf.TensorArray(d, size=h, element_shape=[]) for _ in names)
        def step_fn(t, weights, total, arrays):
            predicted = tf.cond(t == 0, lambda: prior, lambda: tf.linalg.matvec(transition, weights))
            loss = tf.abs(1-tf.reduce_sum(predicted))
            logweights = tf.math.log(predicted)+self.logg(observations[t], grid)
            increment = tf.reduce_logsumexp(logweights)
            weights = tf.nn.softmax(logweights)
            mean = tf.reduce_sum(weights*grid)
            variance = tf.reduce_sum(weights*(grid-mean)**2)
            total += increment
            arrays = tuple(a.write(t, v) for a, v in zip(arrays, (total, mean, variance, loss)))
            return t+1, weights, total, arrays
        _, _, _, arrays = tf.while_loop(lambda t, *_: t < h, step_fn,
            (tf.constant(0), prior, tf.constant(0., d), arrays), parallel_iterations=1)
        result = dict(zip(names, (a.stack() for a in arrays)))
        result['valid'] = tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(x)) for x in result.values()]))
        return result
