"""Minimal retained-teacher Sinkhorn warm-start student helpers."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import (
    DTYPE,
    SinkhornLogStateTF,
    build_sinkhorn_log_state_tf,
    canonicalize_sinkhorn_log_state_tf,
    pairwise_squared_euclidean_tf,
)


@dataclass(frozen=True)
class RetainedTeacherWarmStartConfigTF:
    particle_hidden_dim: int = 32
    particle_hidden_layers: int = 2
    pooled_hidden_dim: int = 32
    pooled_hidden_layers: int = 1
    epsilon_feature_scale: float = 1.0
    prediction_head: str = "dual_pair"


class SinkhornWarmStartStudentTF(tf.keras.Model):
    """Minimal DeepSets-style predictor for retained Sinkhorn warm starts."""

    def __init__(
        self,
        config: RetainedTeacherWarmStartConfigTF | None = None,
        *,
        name: str = "sinkhorn_warmstart_student_tf",
    ) -> None:
        super().__init__(name=name, dtype=DTYPE)
        self.config = config or RetainedTeacherWarmStartConfigTF()
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
            2 if self.config.prediction_head == "dual_pair" else 1,
            activation=None,
            dtype=DTYPE,
            name="latent_output",
        )

    def call(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        epsilon: float | tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        del training
        features = _particle_features_tf(
            particles,
            weights,
            epsilon,
            epsilon_feature_scale=self.config.epsilon_feature_scale,
        )
        hidden = features
        for layer in self._particle_layers:
            hidden = layer(hidden)
        pooled = tf.reduce_mean(hidden, axis=0, keepdims=True)
        repeated_pooled = tf.repeat(pooled, tf.shape(hidden)[0], axis=0)
        combined = tf.concat([hidden, repeated_pooled], axis=1)
        for layer in self._global_layers:
            combined = layer(combined)
        return self._output_layer(combined)

    def predict_log_state(
        self,
        particles: tf.Tensor,
        weights: tf.Tensor,
        epsilon: float | tf.Tensor,
    ) -> SinkhornLogStateTF:
        raw = self(particles, weights, epsilon=epsilon, training=False)
        if self.config.prediction_head == "dual_pair":
            return canonicalize_sinkhorn_log_state_tf(raw[:, 0], raw[:, 1])
        return recover_sinkhorn_log_state_from_log_u_tf(
            particles,
            weights,
            epsilon,
            canonical_log_u=raw[:, 0],
        )


def predict_sinkhorn_initial_state_tf(
    model: SinkhornWarmStartStudentTF,
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float | tf.Tensor,
) -> SinkhornLogStateTF:
    return model.predict_log_state(particles, weights, epsilon)


def predict_canonical_log_u_tf(
    model: SinkhornWarmStartStudentTF,
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float | tf.Tensor,
) -> tf.Tensor:
    raw = model(particles, weights, epsilon=epsilon, training=False)
    if model.config.prediction_head == "dual_pair":
        return canonicalize_sinkhorn_log_state_tf(raw[:, 0], raw[:, 1]).log_u
    return tf.cast(raw[:, 0], DTYPE)


def recover_sinkhorn_log_state_from_log_u_tf(
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float | tf.Tensor,
    *,
    canonical_log_u: tf.Tensor,
) -> SinkhornLogStateTF:
    x = tf.cast(particles, DTYPE)
    if len(x.shape) == 1:
        x = x[:, None]
    normalized_weights = tf.reshape(tf.cast(weights, DTYPE), [-1])
    normalized_weights = normalized_weights / tf.reduce_sum(normalized_weights)
    if int(x.shape[0] or 0) and int(normalized_weights.shape[0] or 0) and x.shape[0] != normalized_weights.shape[0]:
        raise ValueError("particles and weights must agree on particle count")
    epsilon_value = tf.cast(epsilon, DTYPE)
    if bool((epsilon_value <= 0).numpy()):
        raise ValueError("epsilon must be positive")
    cost = pairwise_squared_euclidean_tf(x)
    kernel_log = -cost / epsilon_value
    target = tf.ones_like(normalized_weights) / tf.cast(tf.size(normalized_weights), DTYPE)
    log_target = tf.math.log(target)
    canonical_log_u = tf.reshape(tf.cast(canonical_log_u, DTYPE), [-1])
    if bool((tf.size(canonical_log_u) != tf.size(normalized_weights)).numpy()):
        raise ValueError("canonical_log_u must match particle count")
    recovered_log_v = log_target - tf.reduce_logsumexp(
        kernel_log + canonical_log_u[:, None],
        axis=0,
    )
    return build_sinkhorn_log_state_tf(
        canonical_log_u,
        recovered_log_v,
        gauge_policy="mean_log_u_zero",
    )


def teacher_state_loss_tf(
    predicted_state: SinkhornLogStateTF,
    teacher_state: SinkhornLogStateTF,
) -> tf.Tensor:
    pred = canonicalize_sinkhorn_log_state_tf(predicted_state)
    teacher = canonicalize_sinkhorn_log_state_tf(teacher_state)
    diff_u = pred.log_u - teacher.log_u
    diff_v = pred.log_v - teacher.log_v
    return tf.reduce_mean(diff_u * diff_u + diff_v * diff_v)


def teacher_log_u_loss_tf(
    predicted_log_u: tf.Tensor,
    teacher_state: SinkhornLogStateTF,
) -> tf.Tensor:
    teacher = canonicalize_sinkhorn_log_state_tf(teacher_state)
    diff_u = tf.reshape(tf.cast(predicted_log_u, DTYPE), [-1]) - teacher.log_u
    return tf.reduce_mean(diff_u * diff_u)


def meta_ot_dual_objective_loss_tf(
    predicted_log_u: tf.Tensor,
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float | tf.Tensor,
) -> tf.Tensor:
    state = recover_sinkhorn_log_state_from_log_u_tf(
        particles,
        weights,
        epsilon,
        canonical_log_u=predicted_log_u,
    )
    source = tf.reshape(tf.cast(weights, DTYPE), [-1])
    source = source / tf.reduce_sum(source)
    target = tf.ones_like(source) / tf.cast(tf.size(source), DTYPE)
    log_source = tf.math.log(tf.maximum(source, tf.constant(1e-300, dtype=DTYPE)))
    log_target = tf.math.log(target)
    cost = pairwise_squared_euclidean_tf(particles)
    epsilon_value = tf.cast(epsilon, DTYPE)
    kernel_log = -cost / epsilon_value
    div_a = tf.reduce_sum(source * (state.log_u - log_source))
    div_b = tf.reduce_sum(target * (state.log_v - log_target))
    total_sum = tf.reduce_sum(tf.exp(state.log_u[:, None] + kernel_log + state.log_v[None, :]))
    dual_obj = div_a + div_b + epsilon_value * (1.0 - total_sum)
    return -dual_obj


def _particle_features_tf(
    particles: tf.Tensor,
    weights: tf.Tensor,
    epsilon: float | tf.Tensor,
    *,
    epsilon_feature_scale: float,
) -> tf.Tensor:
    x = tf.cast(particles, DTYPE)
    if len(x.shape) == 1:
        x = x[:, None]
    normalized_weights = tf.reshape(tf.cast(weights, DTYPE), [-1])
    normalized_weights = normalized_weights / tf.reduce_sum(normalized_weights)
    if int(x.shape[0] or 0) and int(normalized_weights.shape[0] or 0) and x.shape[0] != normalized_weights.shape[0]:
        raise ValueError("particles and weights must agree on particle count")
    log_weights = tf.math.log(tf.maximum(normalized_weights, tf.constant(1e-300, dtype=DTYPE)))
    epsilon_value = tf.cast(epsilon, DTYPE) / tf.cast(epsilon_feature_scale, DTYPE)
    epsilon_column = tf.fill([tf.shape(x)[0], 1], epsilon_value)
    return tf.concat([x, log_weights[:, None], epsilon_column], axis=1)
