"""Normal/von-Mises KDM source consumers; replay only, not a trained filter.

See controlled-smc-kdm-circular-consumer-20261006.md for source anchors.
The posterior loss and resampling KDE preserve distinct author conventions.
"""
import math
import tensorflow as tf


class NormalVonMisesKDM:
    def __init__(self, *, batch_size, particles, circular_dimensions,
                 dtype=tf.float32, jit_compile=True):
        if batch_size < 1 or particles < 1 or not circular_dimensions:
            raise ValueError("Positive batch/particle/dimension sizes required")
        self.dtype = tf.as_dtype(dtype)
        if self.dtype not in (tf.float32, tf.float64):
            raise ValueError("float32 or float64 required")
        self.circular = tf.constant(circular_dimensions, tf.bool)
        self.epsilon = self._constant(1e-8)
        b, n, d = batch_size, particles, len(circular_dimensions)
        cloud = tf.TensorSpec([b, n, d], self.dtype)
        weight = tf.TensorSpec([b, n], self.dtype)
        band = tf.TensorSpec([b, d], self.dtype)
        state = tf.TensorSpec([b, d], self.dtype)
        self.resampling_score = tf.function(
            self._resampling_score, input_signature=[cloud, weight, band, cloud],
            jit_compile=jit_compile, autograph=False)
        self.posterior_score = tf.function(
            self._posterior_score, input_signature=[cloud, weight, band, state],
            jit_compile=jit_compile, autograph=False)
        self.consumer_score = tf.function(
            self._consumer_score,
            input_signature=[cloud, weight, band, cloud, cloud, weight, band, state],
            jit_compile=jit_compile, autograph=False)

    def _constant(self, value):
        return tf.constant(value, self.dtype)

    def _poly(self, argument, coefficients):
        # Fixed 6/8 Horner iterations with one traced body; no per-sample loop.
        coefficients = self._constant(coefficients)
        last = tf.size(coefficients) - 1
        result = tf.fill(tf.shape(argument), coefficients[last])
        _, result = tf.while_loop(
            lambda i, result: i >= 0,
            lambda i, result: (i - 1, coefficients[i] + argument * result),
            (last - 1, result), maximum_iterations=8)
        return result

    def _log_i0(self, concentration):
        # Standard I0 polynomial used by the installed PyTorch VonMises authority.
        small = self._poly(tf.square(concentration / self._constant(3.75)),
                           [1., 3.5156229, 3.0899424, 1.2067492,
                            .2659732, .0360768, .0045813])
        large = self._poly(self._constant(3.75) / concentration,
                           [.39894228, .01328592, .00225319, -.00157565,
                            .00916281, -.02057706, .02635537, -.01647633,
                            .00392377])
        small_branch = concentration < self._constant(3.75)
        value = tf.where(small_branch, tf.math.log(small),
                         concentration - .5 * tf.math.log(concentration)
                         + tf.math.log(large))
        valid = tf.reduce_all((concentration > 0) & tf.math.is_finite(concentration))
        valid &= tf.reduce_all(tf.where(small_branch, small, large) > 0)
        valid &= tf.reduce_all(tf.math.is_finite(value))
        return value, valid

    def _components(self, locations, bandwidth, points, *, resampling):
        residual = points[:, :, None, :] - locations[:, None, :, :]
        band = bandwidth[:, None, None, :]
        normal = -.5 * tf.square(residual / band) - tf.math.log(band)
        normal -= self._constant(.5 * math.log(2 * math.pi))
        concentration = 1. / (band + self.epsilon if resampling else band)
        # torch.distributions.broadcast_all expands concentration to [B,N]
        # before computing log I0. Preserve that graph: reducing over particles
        # before differentiating log I0 changes FP32 cancellation materially.
        concentration = tf.broadcast_to(concentration[:, 0, :, :], tf.shape(locations))
        log_i0, special_valid = self._log_i0(concentration)
        circular = concentration[:, None, :, :] * tf.cos(residual)
        circular -= self._constant(math.log(2 * math.pi))
        circular -= log_i0[:, None, :, :]
        component = tf.reduce_sum(tf.where(self.circular, circular, normal), -1)
        valid = special_valid & tf.reduce_all(bandwidth > 0)
        valid &= tf.reduce_all(tf.math.is_finite(locations))
        valid &= tf.reduce_all(tf.math.is_finite(points))
        valid &= tf.reduce_all(tf.math.is_finite(component))
        return component, valid

    def _weight_valid(self, weights):
        return (tf.reduce_all(tf.math.is_finite(weights))
                & tf.reduce_all(weights >= 0)
                & tf.reduce_all(tf.reduce_sum(weights, -1) > 0))

    def _resampling(self, locations, weights, bandwidth, samples):
        samples = tf.stop_gradient(samples)
        component, valid = self._components(locations, bandwidth, samples, resampling=True)
        log_weight = tf.math.log(weights + self.epsilon)
        log_weight -= tf.reduce_logsumexp(log_weight, -1, keepdims=True)
        density = tf.reduce_logsumexp(component + log_weight[:, None, :], -1)
        iwsg = tf.exp(density - tf.stop_gradient(density)) + self.epsilon
        valid &= self._weight_valid(weights)
        valid &= tf.reduce_all(tf.math.is_finite(density))
        return dict(log_density=density, iwsg=iwsg, valid=valid)

    def _posterior(self, locations, weights, bandwidth, states):
        component, valid = self._components(
            locations, bandwidth, states[:, None, :], resampling=False)
        # Preserve exp -> weighted sum -> log floor in the actual source loss.
        density = tf.reduce_sum(tf.exp(component[:, 0, :]) * weights, -1)
        nll = -tf.math.log(density + self.epsilon)
        valid &= self._weight_valid(weights)
        valid &= tf.reduce_all(tf.math.is_finite(nll))
        return dict(density=density, nll=nll, loss=tf.reduce_mean(nll), valid=valid)

    @staticmethod
    def _gradients_valid(gradients):
        # Fixed parameter-group list, not a loop over numerical samples.
        return tf.reduce_all(tf.stack([tf.reduce_all(tf.math.is_finite(g))
                                       for g in gradients]))

    def _resampling_score(self, locations, weights, bandwidth, samples):
        parameters = [locations, weights, bandwidth]
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(parameters + [samples])
            result = self._resampling(locations, weights, bandwidth, samples)
            stopped = tf.stop_gradient(samples)
            integrand = tf.reduce_sum(tf.square(stopped) + .3 * stopped, -1)
            raw_loss = tf.reduce_sum(result['iwsg'] * integrand)
            normalized = result['iwsg'] / tf.reduce_sum(result['iwsg'], -1, keepdims=True)
            normalized_loss = tf.reduce_sum(normalized * integrand)
        raw_gradient = tape.gradient(raw_loss, parameters)
        normalized_gradient = tape.gradient(normalized_loss, parameters)
        sample_gradient = tape.gradient(raw_loss, samples)
        result['valid'] &= self._gradients_valid(raw_gradient + normalized_gradient)
        return dict(**result, raw_loss=raw_loss, normalized_loss=normalized_loss,
                    raw_gradient=raw_gradient, normalized_gradient=normalized_gradient,
                    sample_gradient_stopped=tf.constant(sample_gradient is None))

    def _posterior_score(self, locations, weights, bandwidth, states):
        parameters = [locations, weights, bandwidth]
        with tf.GradientTape() as tape:
            tape.watch(parameters)
            result = self._posterior(locations, weights, bandwidth, states)
        gradient = tape.gradient(result['loss'], parameters)
        result['valid'] &= self._gradients_valid(gradient)
        return dict(**result, gradient=gradient)

    def _consumer_score(self, locations, weights, bandwidth, samples,
                        posterior_locations, model_weights, posterior_bandwidth, states):
        parameters = [locations, weights, bandwidth, posterior_locations,
                      model_weights, posterior_bandwidth]
        with tf.GradientTape() as tape:
            tape.watch(parameters + [samples])
            resampling = self._resampling(locations, weights, bandwidth, samples)
            raw_weights = model_weights * resampling['iwsg']
            norm = tf.reduce_sum(tf.abs(raw_weights), -1, keepdims=True)
            normalized = raw_weights / tf.maximum(norm, self.epsilon)
            result = self._posterior(posterior_locations, normalized,
                                     posterior_bandwidth, states)
        gradient = tape.gradient(result['loss'], parameters + [samples])
        result['valid'] &= resampling['valid'] & self._weight_valid(model_weights)
        result['valid'] &= self._gradients_valid(gradient[:-1])
        return dict(**result, normalized_weights=normalized, gradient=gradient[:-1],
                    sample_gradient_stopped=tf.constant(gradient[-1] is None))
