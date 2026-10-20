"""Batched analytic KDM consumer for Appendix B equations (20)--(22).

The prior, transition timing and numerical coefficients are explicit caller
choices. This does not implement or certify the unrecovered author training
configuration, neural bearings models or a general multidimensional filter.
"""
import math
import tensorflow as tf
from bayesfilter.replication.kdm_iwsg_tf import GaussianMixtureIWSG


class BimodalLinearKDM:
    def __init__(self, *, batch, horizon, particles, sigma, gamma,
                 posterior_bandwidth, resampling_bandwidth,
                 density_epsilon, injection_epsilon, gradient_mode,
                 jit_compile=True):
        if any(type(v) is not int or v < 1 for v in (batch, horizon, particles)):
            raise ValueError("Positive integer sizes required")
        if batch < 2:
            raise ValueError("The update consumer requires a native batch > 1")
        if not all(math.isfinite(v) and v > 0 for v in
                   (sigma, gamma, posterior_bandwidth, resampling_bandwidth)):
            raise ValueError("Finite positive scales required")
        if gradient_mode not in ("iwsg", "truncated"):
            raise ValueError("Unknown gradient mode")
        self.batch, self.horizon, self.particles = batch, horizon, particles
        self.sigma, self.gamma = float(sigma), float(gamma)
        self.posterior_bandwidth = float(posterior_bandwidth)
        self.resampling_bandwidth = float(resampling_bandwidth)
        self.gradient_mode = gradient_mode
        self.mixture = GaussianMixtureIWSG(
            batch_size=batch, particles=particles, samples=particles, dimensions=1,
            density_weight_epsilon=density_epsilon,
            injection_epsilon=injection_epsilon, jit_compile=jit_compile)
        self.signature = [
            tf.TensorSpec([7], tf.float64),
            tf.TensorSpec([batch, particles], tf.float64),
            tf.TensorSpec([batch, horizon], tf.float64),
            tf.TensorSpec([batch, horizon], tf.float64),
            tf.TensorSpec([batch], tf.float64),
            tf.TensorSpec([horizon, batch, particles], tf.float64),
            tf.TensorSpec([horizon, batch, particles], tf.float64),
            tf.TensorSpec([horizon, batch, particles], tf.float64)]
        kwargs = dict(input_signature=self.signature, jit_compile=jit_compile,
                      autograph=False)
        self.evaluate = tf.function(self._evaluate, **kwargs)
        self.value_and_gradient = tf.function(self._value_and_gradient, **kwargs)
        self.step = tf.function(self._step, input_signature=self.signature + [
            tf.TensorSpec([], tf.float64)], jit_compile=jit_compile, autograph=False)

    def _evaluate(self, theta, initial_particles, actions, observations,
                  terminal_states, uniforms, kernel_noise, dynamics_noise):
        a, b, c1, c2, offset1, offset2, v = tf.unstack(theta)
        h, batch, n = self.horizon, self.batch, self.particles
        logmix = tf.stack([-tf.nn.softplus(v), -tf.nn.softplus(-v)])
        weights = tf.ones([batch, n], tf.float64) / n
        bandwidth = tf.fill([batch, 1], tf.constant(self.resampling_bandwidth, tf.float64))
        arrays = [tf.TensorArray(tf.float64, size=h, element_shape=[batch, n])
                  for _ in range(3)]
        ancestry = tf.TensorArray(tf.int32, size=h, element_shape=[batch, n])
        valid = tf.reduce_all(tf.stack([
            tf.reduce_all(tf.math.is_finite(x)) for x in
            (theta, initial_particles, actions, observations, terminal_states,
             uniforms, kernel_noise, dynamics_noise)]))
        valid = valid & tf.reduce_all(uniforms >= 0) & tf.reduce_all(uniforms < 1)
        minimum_raw_weight_sum = tf.constant(float("inf"), tf.float64)

        def body(t, locations, weights, valid, min_sum, resampled, propagated, weighted, ancestors):
            # Categorical law uses the weights themselves, as in author KDE.sample.
            probability = weights / tf.reduce_sum(weights, axis=1, keepdims=True)
            cdf = tf.cumsum(probability, axis=1)
            cdf = tf.concat([cdf[:, :-1], tf.ones([batch, 1], tf.float64)], axis=1)
            parent = tf.searchsorted(cdf, uniforms[t], side="right", out_type=tf.int32)
            sample = tf.stop_gradient(
                tf.gather(locations, parent, batch_dims=1) +
                self.resampling_bandwidth * kernel_noise[t])
            if self.gradient_mode == "iwsg":
                injection = self.mixture._evaluate(
                    locations[:, :, None], weights, bandwidth, sample[:, :, None])
                resampling_weights = injection["iwsg"]
                valid = valid & injection["valid"]
            else:
                resampling_weights = tf.ones_like(weights) / n
            cloud = a * sample + b * actions[:, t, None] + self.sigma * dynamics_noise[t]
            means = tf.stack([c1 * cloud + offset1, c2 * cloud + offset2], axis=-1)
            residual = (observations[:, t, None, None] - means) / self.gamma
            log_component = -.5 * residual * residual - math.log(self.gamma) - .5 * math.log(2 * math.pi)
            likelihood = tf.exp(tf.reduce_logsumexp(log_component + logmix, axis=-1))
            raw = likelihood * resampling_weights
            total = tf.reduce_sum(raw, axis=1, keepdims=True)
            # Match author normalize(p=1,eps=1e-8); reject affected densities.
            updated = raw / tf.maximum(total, tf.constant(1e-8, tf.float64))
            valid = valid & tf.reduce_all(total >= 1e-8)
            valid = valid & tf.reduce_all(tf.math.is_finite(cloud))
            valid = valid & tf.reduce_all(tf.math.is_finite(updated))
            valid = valid & tf.reduce_all(updated >= 0)
            valid = valid & tf.reduce_all(tf.abs(tf.reduce_sum(updated, axis=1) - 1) <= 1e-12)
            return (t + 1, cloud, updated, valid, tf.minimum(min_sum, tf.reduce_min(total)),
                    resampled.write(t, sample), propagated.write(t, cloud),
                    weighted.write(t, updated), ancestors.write(t, parent))

        result = tf.while_loop(
            lambda t, *_: t < h, body,
            (tf.constant(0), initial_particles, weights, valid,
             minimum_raw_weight_sum, *arrays, ancestry),
            maximum_iterations=h, parallel_iterations=1)
        _, locations, weights, valid, minimum_sum, resampled, propagated, weighted, ancestry = result
        posterior_bandwidth = tf.fill(
            [batch, 1], tf.constant(self.posterior_bandwidth, tf.float64))
        logdensity = self.mixture._density(
            locations[:, :, None], weights, posterior_bandwidth,
            terminal_states[:, None, None])[:, 0]
        loss = -tf.reduce_mean(logdensity)
        valid = valid & tf.reduce_all(tf.math.is_finite(logdensity))
        return dict(resampled=resampled.stack(), particles=propagated.stack(),
                    weights=weighted.stack(), ancestors=ancestry.stack(),
                    terminal_logdensity=logdensity, loss=loss, valid=valid,
                    minimum_raw_weight_sum=minimum_sum)

    def _value_and_gradient(self, theta, *args):
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = self._evaluate(theta, *args)
        gradient = tape.gradient(result["loss"], theta)
        return dict(**{k: v for k, v in result.items() if k != "valid"},
                    gradient=gradient,
                    valid=result["valid"] & tf.reduce_all(tf.math.is_finite(gradient)))

    def _step(self, theta, initial_particles, actions, observations, terminal_states,
              uniforms, kernel_noise, dynamics_noise, learning_rate):
        result = self._value_and_gradient(
            theta, initial_particles, actions, observations, terminal_states,
            uniforms, kernel_noise, dynamics_noise)
        next_theta = theta - learning_rate * result["gradient"]
        return dict(**{k: v for k, v in result.items() if k != "valid"},
                    next_theta=next_theta,
                    valid=result["valid"] & (learning_rate > 0) &
                    tf.math.is_finite(learning_rate) &
                    tf.reduce_all(tf.math.is_finite(next_theta)))
