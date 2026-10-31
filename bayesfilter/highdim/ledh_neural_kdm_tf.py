"""Conditional Gaussian KDM proposal adaptation with fixed component weights.

This optional family is not the source bearings filter or an IWSG learner.
Neural outputs depend on observations only, which are fixed for model scores.
"""

from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.highdim.ledh_backward_guide_tf import _checked_cholesky


@dataclass(frozen=True)
class FrozenConditionalKDMProposal:
    offsets: object
    log_scales: object
    conditioning: str = "neural_conditional_kdm"

    def validate(self, horizon, dimension, dtype):
        if self.offsets.shape.rank != 3 or self.offsets.shape[0] != horizon:
            raise ValueError("KDM offsets require [horizon, components, dimension]")
        if self.offsets.shape[-1] != dimension or self.offsets.shape[1] < 1:
            raise ValueError("KDM proposal dimension mismatch")
        if self.log_scales.shape != self.offsets.shape:
            raise ValueError("KDM log-scale shape mismatch")
        if self.offsets.dtype != dtype or self.log_scales.dtype != dtype:
            raise ValueError("KDM proposal dtype mismatch")

    def component_moments(self, time, mean, covariance, d_mean, d_covariance):
        factor, valid = _checked_cholesky(covariance)
        left = tf.linalg.triangular_solve(factor, d_covariance)
        congruence = tf.linalg.matrix_transpose(tf.linalg.triangular_solve(
            factor, tf.linalg.matrix_transpose(left)))
        phi = (tf.linalg.band_part(congruence, -1, 0)
               - .5 * tf.linalg.diag(tf.linalg.diag_part(congruence)))
        d_factor = factor @ phi
        offsets = self.offsets[time]
        scales = tf.exp(self.log_scales[time])
        offsets_finite = tf.math.is_finite(offsets)
        scales_valid = tf.math.is_finite(scales) & (scales > 0.)
        valid &= tf.reduce_all(offsets_finite) & tf.reduce_all(scales_valid)
        offsets = tf.where(offsets_finite, offsets, tf.zeros_like(offsets))
        scales = tf.where(scales_valid, scales, tf.ones_like(scales))
        means = mean[:, None, :] + tf.linalg.matvec(factor[:, None], offsets)
        d_means = d_mean[:, None, :] + tf.linalg.matvec(d_factor[:, None], offsets)
        factors = factor[:, None] * scales[None, :, None, :]
        d_factors = d_factor[:, None] * scales[None, :, None, :]
        covariances = factors @ tf.linalg.matrix_transpose(factors)
        d_covariances = (d_factors @ tf.linalg.matrix_transpose(factors)
                         + factors @ tf.linalg.matrix_transpose(d_factors))
        return means, covariances, d_means, d_covariances, valid


class ObservationKDMNetwork(tf.Module):
    """All-parameter neural location/bandwidth map for a fixed observation scope."""

    def __init__(self, horizon, dimension, components, width, observation_center,
                 observation_scale, *, seed, dtype=tf.float64, jit_compile=True):
        super().__init__()
        if min(horizon, dimension, components, width) < 1:
            raise ValueError("Positive neural KDM dimensions required")
        self.horizon, self.dimension, self.components = horizon, dimension, components
        self.observation_center = tf.convert_to_tensor(observation_center, dtype)
        self.observation_scale = tf.convert_to_tensor(observation_scale, dtype)
        if self.observation_center.shape != self.observation_scale.shape:
            raise ValueError("Observation scaling shape mismatch")
        if not bool(tf.reduce_all(tf.math.is_finite(self.observation_scale)
                                  & (self.observation_scale > 0.))):
            raise ValueError("Observation scales must be finite and positive")
        input_size = self.observation_center.shape.num_elements()
        output_size = horizon * components * dimension * 2
        self.parameters = []
        for layer, (inputs, outputs) in enumerate(((input_size, width), (width, width),
                                                   (width, output_size))):
            amplitude = (2. / (inputs + outputs)) ** .5
            if layer == 2:
                amplitude *= .05
            weights = tf.random.stateless_normal([inputs, outputs], [seed, layer], dtype=dtype)
            self.parameters.extend([
                tf.Variable(weights * amplitude, name=f"layer_{layer}_weights"),
                tf.Variable(tf.zeros([outputs], dtype), name=f"layer_{layer}_bias")])
        self.predict = tf.function(self._predict,
            input_signature=[tf.TensorSpec(self.observation_center.shape, dtype)],
            jit_compile=jit_compile, autograph=False)

    def _predict(self, observations):
        hidden = tf.reshape((observations - self.observation_center) / self.observation_scale,
                            [1, -1])
        for layer in range(3):
            hidden = hidden @ self.parameters[2 * layer] + self.parameters[2 * layer + 1]
            if layer < 2:
                hidden = tf.math.tanh(hidden)
        outputs = tf.reshape(hidden, [self.horizon, self.components, 2, self.dimension])
        return outputs[:, :, 0], outputs[:, :, 1]
