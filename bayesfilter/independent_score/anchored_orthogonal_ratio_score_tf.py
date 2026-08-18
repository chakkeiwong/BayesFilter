"""Anchored discrete-orthogonal joint perturbation ratio-score estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence

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
DELTAS = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04)
ARCHITECTURES = ("anchored_linear_quadratic", "anchored_mlp_quadratic")


def basis_alpha(deltas: Sequence[float] = DELTAS, *, delta_scale: float = DELTA_SCALE) -> float:
    radius = [float(delta) / float(delta_scale) for delta in deltas]
    numerator = sum(value**4 for value in radius)
    denominator = sum(value**6 for value in radius)
    if denominator <= 0.0:
        raise ValueError("basis design has zero sixth moment")
    return numerator / denominator


def anchored_basis(
    deltas: tf.Tensor,
    *,
    delta_scale: float = DELTA_SCALE,
    design_deltas: Sequence[float] = DELTAS,
) -> tf.Tensor:
    values = tf.cast(tf.reshape(tf.convert_to_tensor(deltas), [-1]), DTYPE)
    radius = values / tf.cast(float(delta_scale), DTYPE)
    alpha = tf.cast(basis_alpha(design_deltas, delta_scale=delta_scale), DTYPE)
    return tf.stack([radius, tf.pow(radius, 3) - alpha * tf.pow(radius, 5)], axis=1)


def basis_diagnostics(
    deltas: Sequence[float] = DELTAS, *, delta_scale: float = DELTA_SCALE
) -> dict[str, float]:
    radius = [float(delta) / float(delta_scale) for delta in deltas]
    alpha = basis_alpha(deltas, delta_scale=delta_scale)
    first = [value for value in radius]
    second = [value**3 - alpha * value**5 for value in radius]
    inner = sum(a * b for a, b in zip(first, second))
    gram = [[sum(a * b for a, b in zip(first, first)), sum(a * b for a, b in zip(first, second))], [inner, sum(a * b for a, b in zip(second, second))]]
    trace = gram[0][0] + gram[1][1]
    determinant = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    discriminant = max(0.0, trace * trace - 4.0 * determinant)
    eig_max = 0.5 * (trace + math.sqrt(discriminant))
    eig_min = 0.5 * (trace - math.sqrt(discriminant))
    return {"alpha": alpha, "inner_product": inner, "phi0_derivative_at_zero": 1.0, "phi1_derivative_at_zero": 0.0, "condition_number": math.sqrt(eig_max / eig_min) if eig_min > 0.0 else math.inf}


def validate_conditional_dataset(observations: tf.Tensor, deltas: tf.Tensor, labels: tf.Tensor, *, expected_deltas: Sequence[float] = DELTAS) -> None:
    values = _observations(observations)
    d = tf.cast(tf.reshape(tf.convert_to_tensor(deltas), [-1]), DTYPE)
    y = tf.cast(tf.reshape(tf.convert_to_tensor(labels), [-1]), DTYPE)
    validate_balanced_observation_dataset(values, y)
    if d.shape[0] != y.shape[0]:
        raise ValueError("delta and label counts differ")
    allowed = tf.reduce_any(tf.stack([tf.abs(d - tf.cast(delta, DTYPE)) < tf.constant(1.0e-6, DTYPE) for delta in expected_deltas], axis=1), axis=1)
    if not bool(tf.reduce_all(allowed).numpy()):
        raise ValueError("undeclared delta")
    for delta in expected_deltas:
        mask = tf.abs(d - tf.cast(delta, DTYPE)) < tf.constant(1.0e-6, DTYPE)
        positive = int(tf.reduce_sum(tf.cast(tf.logical_and(mask, y > 0.5), tf.int32)).numpy())
        negative = int(tf.reduce_sum(tf.cast(tf.logical_and(mask, y < 0.5), tf.int32)).numpy())
        if positive == 0 or positive != negative:
            raise ValueError(f"conditional class balance failed at delta={delta}")


def _features(values: tf.Tensor, center: tf.Tensor, scale: tf.Tensor) -> tf.Tensor:
    standardized = (_observations(values) - center[None, :, :]) / scale[None, :, :]
    flat = tf.reshape(standardized, [tf.shape(standardized)[0], -1])
    return tf.concat([flat, tf.square(flat) - tf.constant(1.0, DTYPE)], axis=1)


def _make_model(architecture: str, input_dimension: int, seed: int) -> tf.keras.Model:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    if architecture == "anchored_linear_quadratic":
        layers = [tf.keras.layers.InputLayer(shape=(input_dimension,)), tf.keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros")]
    else:
        layers = [
            tf.keras.layers.InputLayer(shape=(input_dimension,)),
            tf.keras.layers.Dense(128, activation="tanh", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(seed)), bias_initializer="zeros"),
            tf.keras.layers.Dense(64, activation="tanh", kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(seed) + 1), bias_initializer="zeros"),
            tf.keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros"),
        ]
    return tf.keras.Sequential(layers)


def _raw_logits(model: tf.keras.Model, features: tf.Tensor, deltas: tf.Tensor) -> tf.Tensor:
    return tf.reduce_sum(tf.cast(model(features, training=False), DTYPE) * anchored_basis(deltas), axis=1)


def _training_pair_rows(pair_ids: tf.Tensor, labels: tf.Tensor, deltas: tf.Tensor) -> tf.Tensor:
    ids = tf.reshape(tf.convert_to_tensor(pair_ids), [-1])
    y = tf.cast(tf.reshape(tf.convert_to_tensor(labels), [-1]), DTYPE)
    d = tf.cast(tf.reshape(tf.convert_to_tensor(deltas), [-1]), DTYPE)
    if ids.shape[0] != y.shape[0] or ids.shape[0] != d.shape[0]:
        raise ValueError("train_pair_ids count differs from training rows")
    _, _, counts = tf.unique_with_counts(ids)
    if not bool(tf.reduce_all(counts == 2).numpy()):
        raise ValueError("every training pair ID must occur exactly twice")
    order = tf.argsort(ids, stable=True)
    rows = tf.reshape(order, [-1, 2])
    pair_labels = tf.gather(y, rows)
    pair_deltas = tf.gather(d, rows)
    if not bool(tf.reduce_all(tf.reduce_sum(pair_labels, axis=1) == 1.0).numpy()):
        raise ValueError("every training pair must contain one row per class")
    if not bool(tf.reduce_all(tf.abs(pair_deltas[:, 0] - pair_deltas[:, 1]) < 1.0e-6).numpy()):
        raise ValueError("training pair members must use the same delta")
    return rows


@dataclass(frozen=True)
class AnchoredFit:
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
    test_auc_by_delta: dict[str, tf.Tensor]
    test_ece_by_delta: dict[str, tf.Tensor]
    test_logit_minimum_by_delta: dict[str, tf.Tensor]
    test_logit_maximum_by_delta: dict[str, tf.Tensor]
    finite: tf.Tensor
    delta_scale: float

    def _model(self, input_dimension: int) -> tf.keras.Model:
        model = _make_model(self.architecture, input_dimension, 0)
        for variable, value in zip(model.weights, self.model_weights):
            variable.assign(value)
        return model

    def coefficient_values(self, values: tf.Tensor) -> tf.Tensor:
        features = _features(values, self.center, self.scale)
        return tf.cast(self._model(int(features.shape[1]))(features, training=False), DTYPE) * self.calibration_temperature

    def calibrated_logit(self, values: tf.Tensor, deltas: tf.Tensor) -> tf.Tensor:
        return tf.reduce_sum(self.coefficient_values(values) * anchored_basis(deltas, delta_scale=self.delta_scale), axis=1)

    def score_at_observation(self, observation: tf.Tensor) -> tf.Tensor:
        c0 = self.coefficient_values(_observations(observation))[:, 0]
        return tf.cast(c0, tf.float64) / tf.cast(2.0 * self.delta_scale, tf.float64)


def fit_anchored_classifier(train_observations: tf.Tensor, train_deltas: tf.Tensor, train_labels: tf.Tensor, *, validation_observations: tf.Tensor, validation_deltas: tf.Tensor, validation_labels: tf.Tensor, calibration_observations: tf.Tensor, calibration_deltas: tf.Tensor, calibration_labels: tf.Tensor, test_observations: tf.Tensor, test_deltas: tf.Tensor, test_labels: tf.Tensor, architecture: str, seed: int, expected_deltas: Sequence[float] = DELTAS, learning_rate: float = 1.0e-3, epochs: int = 80, minimum_epochs: int = 15, patience: int = 10, batch_size: int = 2048, l2: float = 0.0, jit_compile: bool = True, delta_scale: float = DELTA_SCALE, train_pair_ids: tf.Tensor | None = None) -> AnchoredFit:
    splits = ((train_observations, train_deltas, train_labels), (validation_observations, validation_deltas, validation_labels), (calibration_observations, calibration_deltas, calibration_labels), (test_observations, test_deltas, test_labels))
    for observations, deltas, labels in splits:
        validate_conditional_dataset(observations, deltas, labels, expected_deltas=expected_deltas)
    train = _observations(train_observations)
    center = tf.reduce_mean(train, axis=0)
    scale = tf.maximum(tf.math.reduce_std(train, axis=0), tf.constant(1.0e-4, DTYPE))
    x_train, x_validation, x_calibration, x_test = (_features(values, center, scale) for values in (train, validation_observations, calibration_observations, test_observations))
    y_train, y_validation, y_calibration, y_test = (tf.cast(tf.reshape(values, [-1]), DTYPE) for values in (train_labels, validation_labels, calibration_labels, test_labels))
    d_train, d_validation, d_calibration, d_test = (tf.cast(tf.reshape(values, [-1]), DTYPE) for values in (train_deltas, validation_deltas, calibration_deltas, test_deltas))
    if int(x_train.shape[0]) % int(batch_size) != 0:
        raise ValueError("training rows must be divisible by batch_size")
    pair_rows = None
    if train_pair_ids is not None:
        if int(batch_size) % 2 != 0:
            raise ValueError("paired training requires an even batch_size")
        pair_rows = _training_pair_rows(train_pair_ids, y_train, d_train)
    model = _make_model(architecture, int(x_train.shape[1]), int(seed))
    optimizer = tf.keras.optimizers.Adam(float(learning_rate))

    @tf.function(jit_compile=bool(jit_compile))
    def train_step(batch_x: tf.Tensor, batch_d: tf.Tensor, batch_y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = tf.reduce_sum(tf.cast(model(batch_x, training=True), DTYPE) * anchored_basis(batch_d, delta_scale=delta_scale), axis=1)
            loss = binary_log_loss(logits, batch_y)
            if float(l2) > 0.0:
                loss += tf.cast(l2, DTYPE) * tf.add_n([tf.reduce_sum(tf.square(variable)) for variable in model.trainable_variables if "kernel" in variable.name])
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_loss = math.inf
    best_epoch = -1
    best_weights = None
    history: list[float] = []
    stale = 0
    rows = int(x_train.shape[0])
    for epoch in range(int(epochs)):
        if pair_rows is None:
            permutation = tf.random.experimental.stateless_shuffle(tf.range(rows), seed=[int(seed), 5000 + epoch])
        else:
            pair_permutation = tf.random.experimental.stateless_shuffle(tf.range(tf.shape(pair_rows)[0]), seed=[int(seed), 5000 + epoch])
            permutation = tf.reshape(tf.gather(pair_rows, pair_permutation), [-1])
        for start in range(0, rows, int(batch_size)):
            indices = permutation[start:start + int(batch_size)]
            loss = train_step(tf.gather(x_train, indices), tf.gather(d_train, indices), tf.gather(y_train, indices))
        validation_loss = binary_log_loss(_raw_logits(model, x_validation, d_validation), y_validation)
        value = float(validation_loss.numpy())
        history.append(value)
        if value < best_loss - 1.0e-7:
            best_loss = value
            best_epoch = epoch
            best_weights = tuple(tf.identity(variable) for variable in model.weights)
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= int(minimum_epochs) and stale >= int(patience):
            break
    if best_weights is None:
        raise ValueError("no validation checkpoint")
    for variable, value in zip(model.weights, best_weights):
        variable.assign(value)
    late = max(history[-10:]) - min(history[-10:]) if len(history) >= 2 else math.inf
    raw_validation = _raw_logits(model, x_validation, d_validation)
    validation_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_validation, logits=raw_validation)
    validation_se = tf.math.reduce_std(validation_losses) / tf.sqrt(tf.cast(tf.size(validation_losses), DTYPE))
    raw_calibration = _raw_logits(model, x_calibration, d_calibration)
    log_temperature = tf.Variable(tf.constant(0.0, DTYPE))
    calibration_optimizer = tf.keras.optimizers.Adam(1.0e-2)

    @tf.function(jit_compile=bool(jit_compile))
    def calibration_step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            temperature = tf.exp(log_temperature)
            loss = binary_log_loss(temperature * raw_calibration, y_calibration)
        gradients = tape.gradient(loss, [log_temperature])
        calibration_optimizer.apply_gradients(zip(gradients, [log_temperature]))
        return loss

    calibration_before = binary_log_loss(raw_calibration, y_calibration)
    for _ in range(200):
        calibration_loss = calibration_step()
    temperature = tf.exp(log_temperature)
    raw_test = _raw_logits(model, x_test, d_test)
    calibrated_test = temperature * raw_test
    test_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_test, logits=calibrated_test)
    test_se = tf.math.reduce_std(test_losses) / tf.sqrt(tf.cast(tf.size(test_losses), DTYPE))
    aucs: dict[str, tf.Tensor] = {}
    eces: dict[str, tf.Tensor] = {}
    minimums: dict[str, tf.Tensor] = {}
    maximums: dict[str, tf.Tensor] = {}
    for delta in expected_deltas:
        mask = tf.abs(d_test - tf.cast(delta, DTYPE)) < tf.constant(1.0e-6, DTYPE)
        logits = tf.boolean_mask(calibrated_test, mask)
        labels = tf.boolean_mask(y_test, mask)
        key = str(float(delta))
        aucs[key] = binary_auc(logits, labels)
        eces[key] = expected_calibration_error(logits, labels)
        minimums[key] = tf.reduce_min(logits)
        maximums[key] = tf.reduce_max(logits)
    return AnchoredFit(architecture=architecture, center=tf.identity(center), scale=tf.identity(scale), model_weights=tuple(tf.identity(variable) for variable in model.weights), calibration_temperature=tf.identity(temperature), best_epoch=best_epoch, epochs_run=len(history), final_ten_epoch_improvement=float(late), train_log_loss=binary_log_loss(_raw_logits(model, x_train, d_train), y_train), validation_log_loss=tf.constant(best_loss, DTYPE), validation_log_loss_standard_error=validation_se, calibration_log_loss_before=calibration_before, calibration_log_loss_after=binary_log_loss(temperature * raw_calibration, y_calibration), test_log_loss=binary_log_loss(calibrated_test, y_test), test_log_loss_standard_error=test_se, test_auc_by_delta=aucs, test_ece_by_delta=eces, test_logit_minimum_by_delta=minimums, test_logit_maximum_by_delta=maximums, finite=tf.reduce_all(tf.math.is_finite(calibrated_test)) & tf.math.is_finite(temperature), delta_scale=float(delta_scale))


__all__ = ["ARCHITECTURES", "DELTA_SCALE", "DELTAS", "AnchoredFit", "anchored_basis", "basis_alpha", "basis_diagnostics", "fit_anchored_classifier", "validate_conditional_dataset"]
