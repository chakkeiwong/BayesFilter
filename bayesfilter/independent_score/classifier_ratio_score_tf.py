"""Observation-only classifier likelihood-ratio score estimation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence

import tensorflow as tf


DTYPE = tf.float32
ARCHITECTURES = (
    "linear_full_path",
    "linear_full_path_quadratic",
    "mlp_full_path_quadratic",
)


def _observations(values: tf.Tensor) -> tf.Tensor:
    tensor = tf.cast(tf.convert_to_tensor(values), DTYPE)
    if tensor.shape.rank == 2:
        tensor = tensor[None, ...]
    if tensor.shape.rank != 3 or tensor.shape[1] is None or tensor.shape[2] is None:
        raise ValueError("observations must have shape [batch,time,observation_dim]")
    return tensor


def _labels(values: tf.Tensor) -> tf.Tensor:
    return tf.cast(tf.reshape(tf.convert_to_tensor(values), [-1]), DTYPE)


def validate_balanced_observation_dataset(values: tf.Tensor, labels: tf.Tensor) -> None:
    x = _observations(values)
    y = _labels(labels)
    if x.shape[0] is not None and y.shape[0] is not None and x.shape[0] != y.shape[0]:
        raise ValueError("observation and label counts differ")
    if y.shape[0] is not None and y.shape[0] < 2:
        raise ValueError("at least two labeled paths are required")
    if not bool(tf.reduce_all(tf.math.is_finite(x)).numpy()):
        raise ValueError("observations contain non-finite values")
    if not bool(tf.reduce_all((y == 0.0) | (y == 1.0)).numpy()):
        raise ValueError("labels must be binary")
    positives = int(tf.reduce_sum(tf.cast(y, tf.int32)).numpy())
    if positives * 2 != int(tf.size(y).numpy()):
        raise ValueError("classifier classes must be exactly balanced")


def binary_log_loss(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=_labels(labels), logits=tf.cast(tf.reshape(logits, [-1]), DTYPE)
        )
    )


def binary_auc(logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor:
    values = tf.cast(tf.reshape(logits, [-1]), DTYPE)
    y = _labels(labels)
    positive = tf.boolean_mask(values, y > 0.5)
    negative = tf.boolean_mask(values, y < 0.5)
    greater = tf.cast(positive[:, None] > negative[None, :], DTYPE)
    equal = tf.cast(positive[:, None] == negative[None, :], DTYPE)
    return tf.reduce_mean(greater + 0.5 * equal)


def expected_calibration_error(
    logits: tf.Tensor, labels: tf.Tensor, *, bins: int = 10
) -> tf.Tensor:
    probabilities = tf.math.sigmoid(tf.cast(tf.reshape(logits, [-1]), DTYPE))
    y = _labels(labels)
    total = tf.cast(tf.size(y), DTYPE)
    terms = []
    for index in range(int(bins)):
        lower = tf.cast(index / bins, DTYPE)
        upper = tf.cast((index + 1) / bins, DTYPE)
        mask = (probabilities >= lower) & (
            probabilities < upper if index + 1 < bins else probabilities <= upper
        )
        mask_float = tf.cast(mask, DTYPE)
        count = tf.reduce_sum(mask_float)
        denominator = tf.maximum(count, 1.0)
        observed = tf.reduce_sum(mask_float * y) / denominator
        predicted = tf.reduce_sum(mask_float * probabilities) / denominator
        terms.append(count / total * tf.abs(observed - predicted))
    return tf.add_n(terms)


def central_score_from_calibrated_logit(
    calibrated_logit: tf.Tensor, epsilon: float
) -> tf.Tensor:
    """Return the central classifier-ratio score and no other quantity."""

    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or epsilon_value <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    return tf.cast(calibrated_logit, tf.float64) / tf.cast(
        2.0 * epsilon_value, tf.float64
    )


def _weighted_line(
    epsilon_rows: Sequence[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Fit value = intercept + slope * epsilon**2 with known-variance weights."""

    if len(epsilon_rows) < 2:
        raise ValueError("at least two epsilon rows are required")
    weighted = []
    for epsilon, mean, standard_error in epsilon_rows:
        variance = max(float(standard_error) ** 2, 1.0e-6)
        weighted.append((float(epsilon) ** 2, float(mean), 1.0 / variance))
    s0 = sum(weight for _, _, weight in weighted)
    sx = sum(weight * x for x, _, weight in weighted)
    sxx = sum(weight * x * x for x, _, weight in weighted)
    sy = sum(weight * y for _, y, weight in weighted)
    sxy = sum(weight * x * y for x, y, weight in weighted)
    determinant = s0 * sxx - sx * sx
    if not math.isfinite(determinant) or determinant <= 0.0:
        raise ValueError("epsilon-squared design is singular")
    intercept = (sxx * sy - sx * sxy) / determinant
    slope = (s0 * sxy - sx * sy) / determinant
    residual_scale = 1.0
    if len(weighted) > 2:
        chi_squared = sum(
            weight * (y - intercept - slope * x) ** 2 for x, y, weight in weighted
        )
        residual_scale = max(1.0, math.sqrt(chi_squared / (len(weighted) - 2)))
    intercept_standard_error = math.sqrt(sxx / determinant) * residual_scale
    return intercept, slope, intercept_standard_error


def epsilon_squared_extrapolation(
    estimates_by_epsilon: Mapping[float, Sequence[float]],
    *,
    required_replicates: int = 3,
) -> dict[str, object]:
    """Summarize independent classifier fits and apply the frozen admission rules."""

    epsilon_rows: list[tuple[float, float, float]] = []
    per_epsilon: dict[str, object] = {}
    for epsilon in sorted(float(value) for value in estimates_by_epsilon):
        values = [float(value) for value in estimates_by_epsilon[epsilon]]
        finite = all(math.isfinite(value) for value in values)
        complete = finite and len(values) == int(required_replicates)
        if complete:
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                standard_error = math.sqrt(variance / len(values))
            else:
                standard_error = 0.0
            epsilon_rows.append((epsilon, mean, standard_error))
        else:
            mean = None
            standard_error = None
        per_epsilon[str(epsilon)] = {
            "estimates": values,
            "complete_replicates": complete,
            "mean": mean,
            "standard_error": standard_error,
        }

    enough_epsilons = len(epsilon_rows) >= 3
    if not enough_epsilons:
        return {
            "per_epsilon": per_epsilon,
            "admitted_epsilon_count": len(epsilon_rows),
            "gates": {"at_least_three_epsilons": False},
            "reference_admitted": False,
            "status": "no_classifier_ratio_reference",
        }

    intercept, slope, intercept_standard_error = _weighted_line(epsilon_rows)
    leave_one_out = []
    for omitted in range(len(epsilon_rows)):
        reduced = [row for index, row in enumerate(epsilon_rows) if index != omitted]
        if len(reduced) >= 2:
            leave_one_out.append(_weighted_line(reduced)[0])
    leave_one_out_range = max(leave_one_out) - min(leave_one_out)
    smallest_first, smallest_second = epsilon_rows[:2]
    smallest_difference = abs(smallest_first[1] - smallest_second[1])
    smallest_combined_standard_error = math.sqrt(
        smallest_first[2] ** 2 + smallest_second[2] ** 2
    )
    finite = all(
        math.isfinite(value)
        for value in (intercept, slope, intercept_standard_error, leave_one_out_range)
    )
    loo_pass = leave_one_out_range <= max(1.0, 2.0 * intercept_standard_error)
    smallest_pass = smallest_difference <= max(
        1.0, 3.0 * smallest_combined_standard_error
    )
    gates = {
        "at_least_three_epsilons": True,
        "finite_extrapolation": finite,
        "leave_one_epsilon_out_stability": loo_pass,
        "smallest_two_epsilon_agreement": smallest_pass,
    }
    admitted = all(gates.values())
    return {
        "per_epsilon": per_epsilon,
        "admitted_epsilon_count": len(epsilon_rows),
        "intercept": intercept,
        "slope_epsilon_squared": slope,
        "intercept_standard_error": intercept_standard_error,
        "leave_one_out_intercepts": leave_one_out,
        "leave_one_out_range": leave_one_out_range,
        "smallest_two_difference": smallest_difference,
        "smallest_two_combined_standard_error": smallest_combined_standard_error,
        "gates": gates,
        "reference_admitted": admitted,
        "status": "admitted" if admitted else "no_classifier_ratio_reference",
    }


def _standardize(
    values: tf.Tensor, center: tf.Tensor, scale: tf.Tensor, architecture: str
) -> tf.Tensor:
    standardized = (_observations(values) - center[None, :, :]) / scale[None, :, :]
    flat = tf.reshape(standardized, [tf.shape(standardized)[0], -1])
    if architecture == "linear_full_path":
        return flat
    if architecture in ("linear_full_path_quadratic", "mlp_full_path_quadratic"):
        # Each standardized training coordinate has unit second moment.  The
        # plan requires centered quadratic features; omitting this subtraction
        # makes scale-ratio logits carry an avoidable O(path_dimension) offset
        # into tanh layers and can saturate the classifier.
        return tf.concat([flat, tf.square(flat) - tf.constant(1.0, DTYPE)], axis=1)
    raise ValueError(f"unknown architecture: {architecture}")


def _make_model(architecture: str, input_dimension: int, seed: int) -> tf.keras.Model:
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {architecture}")
    if architecture in ("linear_full_path", "linear_full_path_quadratic"):
        layers = [
            tf.keras.layers.InputLayer(shape=(input_dimension,)),
            tf.keras.layers.Dense(
                1,
                kernel_initializer="zeros",
                bias_initializer="zeros",
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
                1,
                kernel_initializer=tf.keras.initializers.GlorotUniform(seed=int(seed) + 2),
                bias_initializer="zeros",
            ),
        ]
    return tf.keras.Sequential(layers)


@dataclass(frozen=True)
class RatioFit:
    architecture: str
    center: tf.Tensor
    scale: tf.Tensor
    model_weights: tuple[tf.Tensor, ...]
    calibration_slope: tf.Tensor
    calibration_intercept: tf.Tensor
    best_epoch: int
    train_log_loss: tf.Tensor
    validation_log_loss: tf.Tensor
    validation_log_loss_standard_error: tf.Tensor
    calibration_log_loss_before: tf.Tensor
    calibration_log_loss_after: tf.Tensor
    test_log_loss: tf.Tensor
    test_log_loss_standard_error: tf.Tensor
    test_auc: tf.Tensor
    expected_calibration_error: tf.Tensor
    test_logit_minimum: tf.Tensor
    test_logit_maximum: tf.Tensor
    finite: tf.Tensor

    def raw_logit(self, values: tf.Tensor) -> tf.Tensor:
        features = _standardize(values, self.center, self.scale, self.architecture)
        weights = self.model_weights
        if self.architecture in ("linear_full_path", "linear_full_path_quadratic"):
            return tf.reshape(tf.linalg.matmul(features, weights[0]) + weights[1], [-1])
        hidden1 = tf.math.tanh(tf.linalg.matmul(features, weights[0]) + weights[1])
        hidden2 = tf.math.tanh(tf.linalg.matmul(hidden1, weights[2]) + weights[3])
        return tf.reshape(tf.linalg.matmul(hidden2, weights[4]) + weights[5], [-1])

    def calibrated_logit(self, values: tf.Tensor) -> tf.Tensor:
        return self.calibration_slope * self.raw_logit(values) + self.calibration_intercept


def fit_ratio_classifier(
    train_observations: tf.Tensor,
    train_labels: tf.Tensor,
    *,
    validation_observations: tf.Tensor,
    validation_labels: tf.Tensor,
    calibration_observations: tf.Tensor,
    calibration_labels: tf.Tensor,
    test_observations: tf.Tensor,
    test_labels: tf.Tensor,
    architecture: str,
    seed: int,
    learning_rate: float = 3.0e-4,
    epochs: int = 80,
    minimum_epochs: int = 12,
    patience: int = 8,
    batch_size: int = 256,
    l2: float = 0.0,
    jit_compile: bool = True,
) -> RatioFit:
    """Fit one balanced full-observation likelihood-ratio classifier."""

    split_pairs = (
        (train_observations, train_labels),
        (validation_observations, validation_labels),
        (calibration_observations, calibration_labels),
        (test_observations, test_labels),
    )
    for values, labels in split_pairs:
        validate_balanced_observation_dataset(values, labels)
    train = _observations(train_observations)
    center = tf.reduce_mean(train, axis=0)
    scale = tf.maximum(tf.math.reduce_std(train, axis=0), tf.constant(1.0e-4, DTYPE))
    x_train = _standardize(train, center, scale, architecture)
    y_train = _labels(train_labels)
    x_validation = _standardize(validation_observations, center, scale, architecture)
    y_validation = _labels(validation_labels)
    x_calibration = _standardize(calibration_observations, center, scale, architecture)
    y_calibration = _labels(calibration_labels)
    x_test = _standardize(test_observations, center, scale, architecture)
    y_test = _labels(test_labels)
    if int(x_train.shape[0]) % int(batch_size) != 0:
        raise ValueError("training rows must be divisible by batch_size")
    model = _make_model(architecture, int(x_train.shape[1]), int(seed))
    optimizer = tf.keras.optimizers.Adam(float(learning_rate))

    @tf.function(jit_compile=bool(jit_compile))
    def train_step(batch_x: tf.Tensor, batch_y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = tf.reshape(model(batch_x, training=True), [-1])
            loss = binary_log_loss(logits, batch_y)
            if float(l2) > 0.0:
                loss += tf.cast(l2, DTYPE) * tf.add_n(
                    [
                        tf.reduce_sum(tf.square(variable))
                        for variable in model.trainable_variables
                        if "kernel" in variable.name
                    ]
                )
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    best_loss = math.inf
    best_epoch = -1
    best_weights: tuple[tf.Tensor, ...] | None = None
    stale = 0
    row_count = int(x_train.shape[0])
    for epoch in range(int(epochs)):
        permutation = tf.random.experimental.stateless_shuffle(
            tf.range(row_count), seed=[int(seed), 1000 + epoch]
        )
        shuffled_x = tf.gather(x_train, permutation)
        shuffled_y = tf.gather(y_train, permutation)
        for start in range(0, row_count, int(batch_size)):
            loss = train_step(
                shuffled_x[start : start + int(batch_size)],
                shuffled_y[start : start + int(batch_size)],
            )
        if not bool(tf.math.is_finite(loss).numpy()):
            raise ValueError("classifier training produced a non-finite loss")
        validation_loss = binary_log_loss(
            tf.reshape(model(x_validation, training=False), [-1]), y_validation
        )
        validation_value = float(validation_loss.numpy())
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
        raise ValueError("classifier produced no finite validation checkpoint")
    for variable, value in zip(model.weights, best_weights):
        variable.assign(value)

    validation_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_validation,
        logits=tf.reshape(model(x_validation, training=False), [-1]),
    )
    validation_standard_error = tf.math.reduce_std(validation_losses) / tf.sqrt(
        tf.cast(tf.size(validation_losses), DTYPE)
    )

    raw_calibration = tf.reshape(model(x_calibration, training=False), [-1])
    calibration_slope = tf.Variable(tf.constant(1.0, DTYPE))
    calibration_intercept = tf.Variable(tf.constant(0.0, DTYPE))
    calibration_optimizer = tf.keras.optimizers.Adam(1.0e-2)

    @tf.function(jit_compile=bool(jit_compile))
    def calibration_step() -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = binary_log_loss(
                calibration_slope * raw_calibration + calibration_intercept,
                y_calibration,
            )
        gradients = tape.gradient(loss, [calibration_slope, calibration_intercept])
        calibration_optimizer.apply_gradients(
            zip(gradients, [calibration_slope, calibration_intercept])
        )
        return loss

    calibration_before = binary_log_loss(raw_calibration, y_calibration)
    for _ in range(200):
        calibration_loss = calibration_step()
    if not bool(tf.math.is_finite(calibration_loss).numpy()):
        raise ValueError("Platt calibration produced a non-finite loss")
    raw_test = tf.reshape(model(x_test, training=False), [-1])
    calibrated_test = calibration_slope * raw_test + calibration_intercept
    test_losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=y_test, logits=calibrated_test
    )
    test_loss_standard_error = tf.math.reduce_std(test_losses) / tf.sqrt(
        tf.cast(tf.size(test_losses), DTYPE)
    )
    all_finite = tf.reduce_all(tf.math.is_finite(calibrated_test)) & tf.reduce_all(
        tf.math.is_finite(tf.stack([calibration_slope, calibration_intercept]))
    )
    return RatioFit(
        architecture=architecture,
        center=tf.identity(center),
        scale=tf.identity(scale),
        model_weights=tuple(tf.identity(variable) for variable in model.weights),
        calibration_slope=tf.identity(calibration_slope),
        calibration_intercept=tf.identity(calibration_intercept),
        best_epoch=best_epoch,
        train_log_loss=binary_log_loss(
            tf.reshape(model(x_train, training=False), [-1]), y_train
        ),
        validation_log_loss=tf.constant(best_loss, DTYPE),
        validation_log_loss_standard_error=validation_standard_error,
        calibration_log_loss_before=calibration_before,
        calibration_log_loss_after=binary_log_loss(
            calibration_slope * raw_calibration + calibration_intercept,
            y_calibration,
        ),
        test_log_loss=binary_log_loss(calibrated_test, y_test),
        test_log_loss_standard_error=test_loss_standard_error,
        test_auc=binary_auc(calibrated_test, y_test),
        expected_calibration_error=expected_calibration_error(calibrated_test, y_test),
        test_logit_minimum=tf.reduce_min(calibrated_test),
        test_logit_maximum=tf.reduce_max(calibrated_test),
        finite=all_finite,
    )


__all__ = [
    "ARCHITECTURES",
    "RatioFit",
    "binary_auc",
    "binary_log_loss",
    "central_score_from_calibrated_logit",
    "epsilon_squared_extrapolation",
    "expected_calibration_error",
    "fit_ratio_classifier",
    "validate_balanced_observation_dataset",
]
