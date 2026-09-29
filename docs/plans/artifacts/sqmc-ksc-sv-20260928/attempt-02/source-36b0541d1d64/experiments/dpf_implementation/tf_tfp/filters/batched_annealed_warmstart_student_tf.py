"""Batched retained-teacher warm-start helpers for annealed transport."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf import (
    AnnealedTransportWarmstartStateTF,
    build_annealed_transport_warmstart_state_tf,
)


DTYPE = tf.float32


@dataclass(frozen=True)
class BatchedAnnealedWarmstartConfigTF:
    particle_hidden_dim: int = 32
    particle_hidden_layers: int = 2
    pooled_hidden_dim: int = 32
    pooled_hidden_layers: int = 1
    epsilon_feature_scale: float = 1.0


class BatchedAnnealedWarmstartStudentTF(tf.keras.Model):
    """Minimal batched predictor for annealed transport potential warm starts."""

    def __init__(
        self,
        config: BatchedAnnealedWarmstartConfigTF | None = None,
        *,
        name: str = "batched_annealed_warmstart_student_tf",
    ) -> None:
        super().__init__(name=name, dtype=DTYPE)
        self.config = config or BatchedAnnealedWarmstartConfigTF()
        self._particle_layers = [
            tf.keras.layers.Dense(
                self.config.particle_hidden_dim,
                activation="tanh",
                dtype=DTYPE,
                name=f"particle_dense_{idx}",
            )
            for idx in range(self.config.particle_hidden_layers)
        ]
        self._global_layers = [
            tf.keras.layers.Dense(
                self.config.pooled_hidden_dim,
                activation="tanh",
                dtype=DTYPE,
                name=f"global_dense_{idx}",
            )
            for idx in range(self.config.pooled_hidden_layers)
        ]
        self._output_layer = tf.keras.layers.Dense(
            4,
            activation=None,
            dtype=DTYPE,
            name="annealed_latent_output",
        )

    def call(
        self,
        scaled_particles: tf.Tensor,
        log_weights: tf.Tensor,
        epsilon: float | tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        del training
        features = _batched_particle_features_tf(
            scaled_particles,
            log_weights,
            epsilon,
            epsilon_feature_scale=self.config.epsilon_feature_scale,
        )
        hidden = features
        for layer in self._particle_layers:
            hidden = layer(hidden)
        pooled = tf.reduce_mean(hidden, axis=1, keepdims=True)
        repeated_pooled = tf.repeat(pooled, tf.shape(hidden)[1], axis=1)
        combined = tf.concat([hidden, repeated_pooled], axis=2)
        for layer in self._global_layers:
            combined = layer(combined)
        return self._output_layer(combined)

    def predict_warmstart_state(
        self,
        scaled_particles: tf.Tensor,
        log_weights: tf.Tensor,
        epsilon: float | tf.Tensor,
        *,
        valid_mask: tf.Tensor,
    ) -> AnnealedTransportWarmstartStateTF:
        raw = self(scaled_particles, log_weights, epsilon=epsilon, training=False)
        return build_annealed_transport_warmstart_state_tf(
            raw[:, :, 0],
            raw[:, :, 1],
            raw[:, :, 2],
            raw[:, :, 3],
            valid_mask,
        )


def predict_batched_annealed_warmstart_state_tf(
    model: BatchedAnnealedWarmstartStudentTF,
    scaled_particles: tf.Tensor,
    log_weights: tf.Tensor,
    epsilon: float | tf.Tensor,
    *,
    valid_mask: tf.Tensor,
) -> AnnealedTransportWarmstartStateTF:
    return model.predict_warmstart_state(
        scaled_particles,
        log_weights,
        epsilon,
        valid_mask=valid_mask,
    )


def _batched_particle_features_tf(
    scaled_particles: tf.Tensor,
    log_weights: tf.Tensor,
    epsilon: float | tf.Tensor,
    *,
    epsilon_feature_scale: float,
) -> tf.Tensor:
    x = tf.cast(scaled_particles, DTYPE)
    w = tf.cast(log_weights, DTYPE)
    if len(x.shape) != 3 or len(w.shape) != 2:
        raise ValueError("scaled_particles must be [B,N,D] and log_weights must be [B,N]")
    if x.shape[0] != w.shape[0] or x.shape[1] != w.shape[1]:
        raise ValueError("scaled_particles and log_weights must agree on batch/particle dimensions")
    epsilon_value = tf.cast(epsilon, DTYPE) / tf.cast(epsilon_feature_scale, DTYPE)
    epsilon_column = tf.fill([tf.shape(x)[0], tf.shape(x)[1], 1], epsilon_value)
    return tf.concat([x, w[:, :, None], epsilon_column], axis=2)
