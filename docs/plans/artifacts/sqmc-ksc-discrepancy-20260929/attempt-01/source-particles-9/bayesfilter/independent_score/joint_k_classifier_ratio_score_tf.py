"""Joint perturbation conditional classifier-ratio score estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import tensorflow as tf

from bayesfilter.independent_score.classifier_ratio_score_tf import (
    DTYPE,
    binary_auc,
    binary_log_loss,
    expected_calibration_error,
    validate_balanced_observation_dataset,
    _observations,
)


DELTA_SCALE = 0.04
ARCHITECTURES = ("joint_linear_quadratic_odd5", "joint_mlp_quadratic_odd5")


def odd_delta_basis(delta: tf.Tensor, *, delta_scale: float = DELTA_SCALE) -> tf.Tensor:
    values = tf.cast(tf.reshape(tf.convert_to_tensor(delta), [-1]), DTYPE)
    scale = tf.cast(float(delta_scale), DTYPE)
    if not math.isfinite(float(delta_scale)) or float(delta_scale) <= 0.0:
        raise ValueError("delta_scale must be finite and positive")
    radius = values / scale
    return tf.stack([radius, tf.pow(radius, 3), tf.pow(radius, 5)], axis=1)


def validate_conditional_balanced_dataset(
    observations: tf.Tensor,
    deltas: tf.Tensor,
    labels: tf.Tensor,
    *,
    expected_deltas: Sequence[float],
) -> None:
    values = _observations(observations)
    delta_values = tf.cast(tf.reshape(tf.convert_to_tensor(deltas), [-1]), DTYPE)
    label_values = tf.cast(tf.reshape(tf.convert_to_tensor(labels), [-1]), DTYPE)
    validate_balanced_observation_dataset(values, label_values)
    if delta_values.shape[0] != label_values.shape[0]:
        raise ValueError("delta and label counts differ")
    if not bool(tf.reduce_all(tf.math.is_finite(delta_values)).numpy()):
        raise ValueError("deltas contain non-finite values")
    expected = tuple(float(value) for value in expected_deltas)
    for delta in expected:
        mask = tf.abs(delta_values - tf.cast(delta, DTYPE)) < tf.constant(1.0e-6, DTYPE)
        count = int(tf.reduce_sum(tf.cast(mask, tf.int32)).numpy())
        positive = int(tf.reduce_sum(tf.cast(tf.logical_and(mask, label_values > 0.5), tf.int32)).numpy())
        negative = int(tf.reduce_sum(tf.cast(tf.logical_and(mask, label_values < 0.5), tf.int32)).numpy())
        if count == 0 or positive != negative:
            raise ValueError(f"conditional class balance failed at delta={delta}")
    allowed = tf.reduce_any(
        tf.stack(
            [tf.abs(delta_values - tf.cast(delta, DTYPE)) < tf.constant(1.0e-6, DTYPE) for delta in expected],
            axis=1,
        ),
        axis=1,
    )
    if not bool(tf.reduce_all(allowed).numpy()):
        raise ValueError("dataset contains an undeclared delta")


def _features(values: tf.Tensor, center: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    standardized = (_observations(values) - center[None, :, :]) / scale[None, :, :]
    flat = tf.reshape(standardized, [tf.shape(standardized)[0], -1])
    return tf.concat([flat, tf.square(flat) - tf.constant(1.0, DTYPE)], axis=1)


def _make_model(architecture: str, input_dimension: int, seed: int) -> tf.keras.Model:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    if architecture == "joint_linear_quadratic_odd5":
        layers = [
            tf.keras.layers.InputLayer(shape=(input_dimension,)),
            tf.keras.layers.Dense(
                3, kernel_initializer="zeros", bias_initializer="zeros"
            ),
        ]
    else:
        layers = [
            tf.keras.layers.InputLayer(shape=(input_dimension,)),
            tf.keras.layers.Dense(
                128,
                activation="tanh",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(seed)),
                bias_initializer="zeros",
            ),
            tf.keras.layers.Dense(
                64,
                activation="tanh",
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(seed) + 1),
                bias_initializer="zeros",
            ),
            tf.keras.layers.Dense(
                3,
                kernel_initializer="zeros",
                bias_initializer="zeros",
            ),
        ]
    return tf.keras.Sequential(layers)


def _raw_logits(model: tf.keras.Model, features: tf.Tensor, deltas: tf.Tensor) -> tf.Tensor:
    coefficients = tf.cast(model(features, training=False), DTYPE)
    return tf.reduce_sum(coefficients * odd_delta_basis(deltas), axis=1)


@dataclass(frozen=True)
class JointKFit:
    architecture: str
    center: tf.Tensor
    scale: tf.Tensor
    model_weights: tuple[tf.Tensor, ...]
    calibration_temperature: tf.Tensor
    best_epoch: int
    epochs_run: int
    final_ten_epoch_improvement: float
    train_log_loss: tf.Tensor
    validation_log_loss: tf.Tensor
    validation_log_loss_standard_error: tf.Tensor
    calibration_log_loss_before: tf.Tensor
    calibration_log_loss_after: tf.Tensor
    test_log_loss: tf.Tensor
    test_log_loss_standard_error: tf.Tensor
    test_auc_by_delta: Mapping[str, tf.Tensor]
    test_ece_by_delta: Mapping[str, tf.Tensor]
    test_logit_minimum_by_delta: Mapping[str, tf.Tensor]
    test_logit_maximum_by_delta: Mapping[str, tf.Tensor]
    finite: tf.Tensor
    delta_scale: float

    def _model(self, input_dimension: int) -> tf.keras.Model:
        model = _make_model(self.architecture, input_dimension, 0)
        for variable, value in zip(model.weights, self.model_weights):
            variable.assign(value)
        return model

    def coefficient_values(self, values: tf.Tensor) -> tf.Tensor:
        features = _features(values, self.center, self.scale)
        model = self._model(int(features.shape[1]))
        return tf.cast(model(features, training=False), DTYPE) * self.calibration_temperature

    def calibrated_logit(self, values: tf.Tensor, deltas: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(
            self.coefficient_values(values) * odd_delta_basis(deltas, delta_scale=self.delta_scale),
            axis=1,
        )

    def score_at_observation(self, observation: tf.Tensor) -> tf.Tensor:
        coefficient = self.coefficient_values(_observations(observation))[:, 0]
        return tf.cast(coefficient, tf.float64) / tf.cast(2.0 * self.delta_scale, tf.float64)


def fit_joint_k_classifier(
    train_observations: tf.Tensor,
    train_deltas: tf.Tensor,
    train_labels: tf.Tensor,
    *,
    validation_observations: tf.Tensor,
    validation_deltas: tf.Tensor,
    validation_labels: tf.Tensor,
    calibration_observations: tf.Tensor,
    calibration_deltas: tf.Tensor,
    calibration_labels: tf.Tensor,
    test_observations: tf.Tensor,
    test_deltas: tf.Tensor,
    test_labels: tf.Tensor,
    expected_deltas: Sequence[float],
    architecture: str,
    seed: int,
    learning_rate: float = 1.0e-3,
    epochs: int = 80,
    minimum_epochs: int = 15,
    patience: int = 10,
    batch_size: int = 2048,
    l2: float = 0.0,
    jit_compile: bool = True,
    delta_scale: float = DELTA_SCALE,
) -> JointKFit:
    splits = (
        (train_observations, train_deltas, train_labels),
        (validation_observations, validation_deltas, validation_labels),
        (calibration_observations, calibration_deltas, calibration_labels),
        (test_observations, test_deltas, test_labels),
    )
    for observations, deltas, labels in splits:
        validate_conditional_balanced_dataset(
            observations, deltas, labels, expected_deltas=expected_deltas
        )
    train = _observations(train_observations)
    center = tf.reduce_mean(train, axis=0)
    scale = tf.maximum(tf.math.reduce_std(train, axis=0), tf.constant(1.0e-4, DTYPE))
    x_train = _features(train, center, scale)
    y_train = tf.cast(tf.reshape(train_labels, [-1]), DTYPE)
    x_validation = _features(validation_observations, center, scale)
    y_validation = tf.cast(tf.reshape(validation_labels, [-1]), DTYPE)
    x_calibration = _features(calibration_observations, center, scale)
    y_calibration = tf.cast(tf.reshape(calibration_labels, [-1]), DTYPE)
    x_test = _features(test_observations, center, scale)
    y_test = tf.cast(tf.reshape(test_labels, [-1]), DTYPE)
    d_train = tf.cast(tf.reshape(train_deltas, [-1]), DTYPE)
    d_validation = tf.cast(tf.reshape(validation_deltas, [-1]), DTYPE)
    d_calibration = tf.cast(tf.reshape(calibration_deltas, [-1]), DTYPE)
    d_test = tf.cast(tf.reshape(test_deltas, [-1]), DTYPE)
    if int(x_train.shape[0]) % int(batch_size) != 0:
        raise ValueError("training rows must be divisible by batch_size")
    model = _make_model(architecture, int(x_train.shape[1]), int(seed))
    optimizer = tf.keras.optimizers.Adam(float(learning_rate))

    @tf.function(jit_compile=bool(jit_compile))
    def train_step(batch_x: tf.Tensor, batch_delta: tf.Tensor, batch_y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            coefficient = tf.cast(model(batch_x, training=True), DTYPE)
            logits = tf.reduce_sum(coefficient * odd_delta_basis(batch_delta, delta_scale=delta_scale), axis=1)
            loss = binary_log_loss(logits, batch_y)
            if float(l2) > 0.0:
                loss += tf.cast(l2, DTYPE) * tf.add_n(
                    [tf.reduce_sum(tf.square(variable)) for variable in model.trainable_variables if "kernel" in variable.name]
                )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_loss = math.inf
    best_epoch = -1
    best_weights: tuple[tf.Tensor, ...] | None = None
    validation_history: list[float] = []
    stale = 0
    row_count = int(x_train.shape[0])
    for epoch in range(int(epochs)):
        permutation = tf.random.experimental.stateless_shuffle(tf.range(row_count), seed=[int(seed), 3000 + epoch])
        for start in range(0, row_count, int(batch_size)):
            indices = permutation[start : start + int(batch_size)]
            loss = train_step(tf.gather(x_train, indices), tf.gather(d_train, indices), tf.gather(y_train, indices))
        if not bool(tf.math.is_finite(loss).numpy()):
            raise ValueError("joint classifier training produced a non-finite loss")
        validation_logits = _raw_logits(model, x_validation, d_validation)
        validation_loss = binary_log_loss(validation_logits, y_validation)
        validation_value = float(validation_loss.numpy())
        validation_history.append(validation_value)
        if validation_value < best_loss - 1.0e-7:
            best_loss = validation_value
            best_epoch = epoch
            best_weights = tuple(tf.identity(variable) for variable in model.weights)
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= int(minimum_epochs) and stale >= int(patience):
            break
    if best_weights is None:
        raise ValueError("joint classifier produced no validation checkpoint")
    for variable, value in zip(model.weights, best_weights):
        variable.assign(value)
    final_ten_improvement = (
        max(validation_history[-10:]) - min(validation_history[-10:])
        if len(validation_history) >= 2
        else math.inf
    )
    validation_logits = _raw_logits(model, x_validation, d_validation)
    validation_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_validation, logits=validation_logits)
    validation_se = tf.math.reduce_std(validation_losses) / tf.sqrt(tf.cast(tf.size(validation_losses), DTYPE))

    raw_calibration = _raw_logits(model, x_calibration, d_calibration)
    log_temperature = tf.Variable(tf.constant(0.0, DTYPE))
    calibration_optimizer = tf.keras.optimizers.Adam(1.0e-2)

    @tf.function(jit_compile=bool(jit_compile))
    def calibration_step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            temperature = tf.exp(log_temperature)
            loss = binary_log_loss(temperature * raw_calibration, y_calibration)
        gradient = tape.gradient(loss, [log_temperature])
        calibration_optimizer.apply_gradients(zip(gradient, [log_temperature]))
        return loss

    calibration_before = binary_log_loss(raw_calibration, y_calibration)
    for _ in range(200):
        calibration_loss = calibration_step()
    if not bool(tf.math.is_finite(calibration_loss).numpy()):
        raise ValueError("joint classifier calibration produced a non-finite loss")
    raw_test = _raw_logits(model, x_test, d_test)
    temperature = tf.exp(log_temperature)
    calibrated_test = temperature * raw_test
    test_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test, logits=calibrated_test)
    test_se = tf.math.reduce_std(test_losses) / tf.sqrt(tf.cast(tf.size(test_losses), DTYPE))
    auc_by_delta: dict[str, tf.Tensor] = {}
    ece_by_delta: dict[str, tf.Tensor] = {}
    minimum_by_delta: dict[str, tf.Tensor] = {}
    maximum_by_delta: dict[str, tf.Tensor] = {}
    for delta in expected_deltas:
        mask = tf.abs(d_test - tf.cast(delta, DTYPE)) < tf.constant(1.0e-6, DTYPE)
        logits = tf.boolean_mask(calibrated_test, mask)
        labels = tf.boolean_mask(y_test, mask)
        key = str(float(delta))
        auc_by_delta[key] = binary_auc(logits, labels)
        ece_by_delta[key] = expected_calibration_error(logits, labels)
        minimum_by_delta[key] = tf.reduce_min(logits)
        maximum_by_delta[key] = tf.reduce_max(logits)
    finite = tf.reduce_all(tf.math.is_finite(calibrated_test)) & tf.math.is_finite(temperature)
    return JointKFit(
        architecture=architecture,
        center=tf.identity(center),
        scale=tf.identity(scale),
        model_weights=tuple(tf.identity(variable) for variable in model.weights),
        calibration_temperature=tf.identity(temperature),
        best_epoch=best_epoch,
        epochs_run=len(validation_history),
        final_ten_epoch_improvement=float(final_ten_improvement),
        train_log_loss=binary_log_loss(_raw_logits(model, x_train, d_train), y_train),
        validation_log_loss=tf.constant(best_loss, DTYPE),
        validation_log_loss_standard_error=validation_se,
        calibration_log_loss_before=calibration_before,
        calibration_log_loss_after=binary_log_loss(temperature * raw_calibration, y_calibration),
        test_log_loss=binary_log_loss(calibrated_test, y_test),
        test_log_loss_standard_error=test_se,
        test_auc_by_delta=auc_by_delta,
        test_ece_by_delta=ece_by_delta,
        test_logit_minimum_by_delta=minimum_by_delta,
        test_logit_maximum_by_delta=maximum_by_delta,
        finite=finite,
        delta_scale=float(delta_scale),
    )


__all__ = [
    "ARCHITECTURES",
    "DELTA_SCALE",
    "JointKFit",
    "fit_joint_k_classifier",
    "odd_delta_basis",
    "validate_conditional_balanced_dataset",
]
