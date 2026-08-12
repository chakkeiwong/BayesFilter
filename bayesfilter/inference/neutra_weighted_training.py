"""TensorFlow/XLA weighted forward-KL training for NeuTra transports.

This module trains an explicit transport density on externally generated,
importance-weighted physical particles.  It is separate from reverse-KL NeuTra:
the training rows are fixed target-covering evidence rather than samples from
the current transport.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import tensorflow as tf


WEIGHTED_NEUTRA_NONCLAIMS = (
    "weighted particles are not an unweighted posterior archive",
    "training loss alone is not a transport promotion criterion",
    "known-mode coverage does not prove exhaustive mode discovery",
    "no HMC, posterior-correctness, predictive, or default-readiness claim",
)


class WeightedNeuTraTrainingError(RuntimeError):
    """Raised when a weighted transport update violates its finite contract."""


@dataclass(frozen=True)
class WeightedNeuTraConfig:
    """Configuration for an invertible dense-autoregressive transport density."""

    dimension: int
    hidden_layers: tuple[int, ...] = (32, 32)
    stages: int = 3
    activation: str = "elu"
    s_max: float = 2.0
    initialization_scale: float = 0.02
    initialization_seed: tuple[int, int] = (20260811, 9101)
    learning_rate: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-7
    gradient_clip_norm: float = 10.0
    jit_compile: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or int(self.dimension) <= 0:
            raise ValueError("dimension must be positive")
        if not self.hidden_layers or any(int(width) <= 0 for width in self.hidden_layers):
            raise ValueError("hidden_layers must be nonempty and positive")
        if isinstance(self.stages, bool) or int(self.stages) <= 0:
            raise ValueError("stages must be positive")
        if self.activation not in {"elu", "tanh", "relu"}:
            raise ValueError("unsupported activation")
        for name in (
            "s_max",
            "learning_rate",
            "epsilon",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(float(self.initialization_scale)) or self.initialization_scale < 0.0:
            raise ValueError("initialization_scale must be finite and nonnegative")
        if len(self.initialization_seed) != 2:
            raise ValueError("initialization_seed must contain two integers")
        for name in ("beta1", "beta2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")

    def manifest_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["hidden_layers"] = list(self.hidden_layers)
        payload["initialization_seed"] = list(self.initialization_seed)
        payload["schema"] = "bayesfilter.neutra.weighted_forward_kl_config.v1"
        return payload


@dataclass(frozen=True)
class WeightedNeuTraStep:
    loss: tf.Tensor
    effective_sample_size: tf.Tensor
    effective_sample_size_fraction: tf.Tensor
    maximum_normalized_weight: tf.Tensor
    gradient_norm: tf.Tensor
    clipped_gradient_norm: tf.Tensor
    clipping_applied: tf.Tensor
    step: tf.Tensor


@dataclass(frozen=True)
class WeightedNeuTraValidation:
    loss: tf.Tensor
    per_sample_negative_log_prob: tf.Tensor
    normalized_weights: tf.Tensor
    latent: tf.Tensor
    latent_weighted_mean: tf.Tensor
    latent_weighted_covariance: tf.Tensor
    effective_sample_size: tf.Tensor
    effective_sample_size_fraction: tf.Tensor
    maximum_normalized_weight: tf.Tensor


def _activation(values: tf.Tensor, name: str) -> tf.Tensor:
    if name == "elu":
        return tf.nn.elu(values)
    if name == "tanh":
        return tf.math.tanh(values)
    if name == "relu":
        return tf.nn.relu(values)
    raise WeightedNeuTraTrainingError(f"unsupported activation: {name}")


def _dense_masks(dimension: int, hidden_layers: tuple[int, ...]) -> tuple[tf.Tensor, ...]:
    degrees: list[list[int]] = [list(range(1, int(dimension) + 1))]
    maximum = max(1, int(dimension) - 1)
    for width in hidden_layers:
        degrees.append([1 + index % maximum for index in range(int(width))])
    degrees.append(
        list(range(1, int(dimension) + 1))
        + list(range(1, int(dimension) + 1))
    )
    masks = []
    for layer, (source, target) in enumerate(zip(degrees[:-1], degrees[1:])):
        output_layer = layer == len(degrees) - 2
        masks.append(
            tf.constant(
                [
                    [
                        1.0
                        if ((left < right) if output_layer else (left <= right))
                        else 0.0
                        for right in target
                    ]
                    for left in source
                ],
                tf.float64,
            )
        )
    return tuple(masks)


class _DenseAutoregressiveStage:
    def __init__(self, config: WeightedNeuTraConfig, stage: int) -> None:
        self.dimension = int(config.dimension)
        self.activation = str(config.activation)
        self.s_max = float(config.s_max)
        self.masks = _dense_masks(self.dimension, tuple(config.hidden_layers))
        sizes = (self.dimension, *config.hidden_layers, 2 * self.dimension)
        root = tf.random.experimental.stateless_fold_in(
            tf.constant(config.initialization_seed, tf.int32), int(stage)
        )
        weights = []
        biases = []
        for index, (input_width, output_width) in enumerate(zip(sizes[:-1], sizes[1:])):
            seed = tf.random.experimental.stateless_fold_in(root, index)
            scale = 0.0 if index == len(sizes) - 2 else float(config.initialization_scale)
            weights.append(
                tf.Variable(
                    tf.random.stateless_normal(
                        (int(input_width), int(output_width)),
                        seed=seed,
                        dtype=tf.float64,
                    )
                    * tf.constant(scale, tf.float64),
                    name=f"weighted_neutra_stage_{stage}_weight_{index}",
                )
            )
            biases.append(
                tf.Variable(
                    tf.zeros((int(output_width),), tf.float64),
                    name=f"weighted_neutra_stage_{stage}_bias_{index}",
                )
            )
        self.weights = tuple(weights)
        self.biases = tuple(biases)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        output = []
        for weight, bias in zip(self.weights, self.biases):
            output.extend((weight, bias))
        return tuple(output)

    def _network(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        hidden = values
        for weight, bias, mask in zip(
            self.weights[:-1], self.biases[:-1], self.masks[:-1]
        ):
            hidden = _activation(tf.matmul(hidden, weight * mask) + bias, self.activation)
        raw = tf.matmul(hidden, self.weights[-1] * self.masks[-1]) + self.biases[-1]
        scale_logits = raw[..., : self.dimension]
        scale_log = self.s_max * tf.math.tanh(scale_logits / self.s_max)
        return scale_log, raw[..., self.dimension :]

    def forward_and_logdet(self, latent: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale_log, shift = self._network(latent)
        return (
            latent * tf.exp(scale_log) + shift,
            tf.reduce_sum(scale_log, axis=-1),
        )

    def inverse_and_forward_logdet(self, output: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        latent = tf.zeros_like(output)
        for index in range(self.dimension):
            scale_log, shift = self._network(latent)
            solved = (output[..., index] - shift[..., index]) * tf.exp(
                -scale_log[..., index]
            )
            latent = latent + (solved - latent[..., index])[..., tf.newaxis] * tf.one_hot(
                index, self.dimension, dtype=latent.dtype
            )
        scale_log, _ = self._network(latent)
        return latent, tf.reduce_sum(scale_log, axis=-1)

    def _network_with_cache(
        self, values: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tuple[tf.Tensor, ...]]:
        hidden = values
        preactivations = []
        for weight, bias, mask in zip(
            self.weights[:-1], self.biases[:-1], self.masks[:-1]
        ):
            preactivation = tf.matmul(hidden, weight * mask) + bias
            preactivations.append(preactivation)
            hidden = _activation(preactivation, self.activation)
        raw = tf.matmul(hidden, self.weights[-1] * self.masks[-1]) + self.biases[-1]
        scale_logits = raw[..., : self.dimension]
        scaled = scale_logits / self.s_max
        tanh_scaled = tf.math.tanh(scaled)
        return (
            self.s_max * tanh_scaled,
            raw[..., self.dimension :],
            1.0 - tf.square(tanh_scaled),
            tuple(preactivations),
        )

    def _network_pullback(
        self,
        raw_cotangent: tf.Tensor,
        preactivations: tuple[tf.Tensor, ...],
    ) -> tf.Tensor:
        cotangent = tf.matmul(
            raw_cotangent, self.weights[-1] * self.masks[-1], transpose_b=True
        )
        for layer_index in reversed(range(len(preactivations))):
            values = preactivations[layer_index]
            if self.activation == "elu":
                derivative = tf.where(values > 0.0, 1.0, tf.exp(values))
            elif self.activation == "tanh":
                derivative = 1.0 - tf.square(tf.math.tanh(values))
            elif self.activation == "relu":
                derivative = tf.cast(values > 0.0, values.dtype)
            else:
                raise WeightedNeuTraTrainingError(
                    f"unsupported activation: {self.activation}"
                )
            cotangent = cotangent * derivative
            cotangent = tf.matmul(
                cotangent,
                self.weights[layer_index] * self.masks[layer_index],
                transpose_b=True,
            )
        return cotangent

    def pullback_score(
        self, values: tf.Tensor, output_score: tf.Tensor
    ) -> tf.Tensor:
        scale_log, _shift, scale_derivative, cache = self._network_with_cache(values)
        direct = output_score * tf.exp(scale_log)
        scale_cotangent = output_score * values * tf.exp(scale_log)
        raw_cotangent = tf.concat(
            (scale_cotangent * scale_derivative, output_score), axis=-1
        )
        return direct + self._network_pullback(raw_cotangent, cache)

    def logdet_score(self, values: tf.Tensor) -> tf.Tensor:
        _scale_log, _shift, scale_derivative, cache = self._network_with_cache(values)
        raw_cotangent = tf.concat(
            (scale_derivative, tf.zeros_like(scale_derivative)), axis=-1
        )
        return self._network_pullback(raw_cotangent, cache)


class WeightedDenseIAFTransport:
    """Trainable composed IAF with differentiable forward and inverse density."""

    def __init__(self, config: WeightedNeuTraConfig) -> None:
        self.config = config
        self._frozen_identity: Mapping[str, Any] | None = None
        self.stages = tuple(
            _DenseAutoregressiveStage(config, stage)
            for stage in range(int(config.stages))
        )

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(
            variable for stage in self.stages for variable in stage.trainable_variables
        )

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2(latent, self.config.dimension, "latent")
        logdet = tf.zeros(tf.shape(values)[0], tf.float64)
        for index, stage in enumerate(self.stages):
            values, increment = stage.forward_and_logdet(values)
            logdet = logdet + increment
            if index + 1 < len(self.stages):
                values = tf.reverse(values, axis=(-1,))
        return values, logdet

    @property
    def parameter_dim(self) -> int:
        return int(self.config.dimension)

    def manifest_payload(self) -> Mapping[str, Any]:
        if self._frozen_identity is None:
            raise WeightedNeuTraTrainingError(
                "weighted IAF must be bound to a verified frozen state before HMC use"
            )
        return {
            "schema": "bayesfilter.neutra.weighted_dense_iaf_frozen.v1",
            "transport_id": "weighted_dense_iaf_frozen",
            "parameter_dim": self.parameter_dim,
            "config": self.config.manifest_payload(),
            "stages": int(self.config.stages),
            "hidden_layers": list(self.config.hidden_layers),
            "activation": self.config.activation,
            "s_max": float(self.config.s_max),
            "frozen_identity": dict(self._frozen_identity),
        }

    def bind_frozen_identity(self, identity: Mapping[str, Any]) -> None:
        required = {
            "checkpoint_sha256",
            "training_state_hash",
            "transport_tensor_hash",
        }
        normalized = {str(key): value for key, value in identity.items()}
        missing = sorted(required - normalized.keys())
        if missing:
            raise WeightedNeuTraTrainingError(
                "frozen identity missing: " + ", ".join(missing)
            )
        if self._frozen_identity is not None and dict(self._frozen_identity) != normalized:
            raise WeightedNeuTraTrainingError("weighted IAF frozen identity is immutable")
        self._frozen_identity = normalized

    def forward(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.forward_and_logdet(values[tf.newaxis, :])[0][0]
        return self.forward_and_logdet(values)[0]

    def forward_batch(self, latent: Any) -> tf.Tensor:
        return self.forward(latent)

    def log_abs_det_jacobian(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.forward_and_logdet(values[tf.newaxis, :])[1][0]
        return self.forward_and_logdet(values)[1]

    def log_abs_det_jacobian_batch(self, latent: Any) -> tf.Tensor:
        return self.log_abs_det_jacobian(latent)

    def pullback_score(self, latent: Any, output_score: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        score = tf.convert_to_tensor(output_score, tf.float64)
        if values.shape.rank == 1 and score.shape.rank == 1:
            return self.pullback_score_batch(
                values[tf.newaxis, :], score[tf.newaxis, :]
            )[0]
        return self.pullback_score_batch(values, score)

    def pullback_score_batch(
        self, latent: Any, output_score: Any
    ) -> tf.Tensor:
        values = _rank2(latent, self.parameter_dim, "latent")
        score = _rank2(output_score, self.parameter_dim, "output_score")
        inputs = []
        current = values
        for index, stage in enumerate(self.stages):
            inputs.append(current)
            current, _ = stage.forward_and_logdet(current)
            if index + 1 < len(self.stages):
                current = tf.reverse(current, axis=(-1,))
        for index in reversed(range(len(self.stages))):
            score = self.stages[index].pullback_score(inputs[index], score)
            if index > 0:
                score = tf.reverse(score, axis=(-1,))
        return score

    def log_abs_det_jacobian_score(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.log_abs_det_jacobian_score_batch(values[tf.newaxis, :])[0]
        return self.log_abs_det_jacobian_score_batch(values)

    def log_abs_det_jacobian_score_batch(self, latent: Any) -> tf.Tensor:
        values = _rank2(latent, self.parameter_dim, "latent")
        inputs = []
        current = values
        for index, stage in enumerate(self.stages):
            inputs.append(current)
            current, _ = stage.forward_and_logdet(current)
            if index + 1 < len(self.stages):
                current = tf.reverse(current, axis=(-1,))
        score = tf.zeros_like(current)
        for index in reversed(range(len(self.stages))):
            score = self.stages[index].pullback_score(inputs[index], score)
            score = score + self.stages[index].logdet_score(inputs[index])
            if index > 0:
                score = tf.reverse(score, axis=(-1,))
        return score

    def inverse_and_forward_logdet(self, physical: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2(physical, self.config.dimension, "physical")
        logdet = tf.zeros(tf.shape(values)[0], tf.float64)
        for index in reversed(range(len(self.stages))):
            values, increment = self.stages[index].inverse_and_forward_logdet(values)
            logdet = logdet + increment
            if index > 0:
                values = tf.reverse(values, axis=(-1,))
        return values, logdet

    def log_prob(self, physical: Any) -> tf.Tensor:
        latent, forward_logdet = self.inverse_and_forward_logdet(physical)
        dimension = tf.cast(self.config.dimension, tf.float64)
        base = -0.5 * (
            tf.reduce_sum(tf.square(latent), axis=-1)
            + dimension * tf.math.log(tf.constant(2.0 * math.pi, tf.float64))
        )
        return base - forward_logdet


class WeightedForwardKLNeuTraTrainer:
    """Optimize ``-sum(normalized_weight * log q_phi(theta))``."""

    def __init__(self, config: WeightedNeuTraConfig) -> None:
        self.config = config
        self.transport = WeightedDenseIAFTransport(config)
        self.variables = self.transport.trainable_variables
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name="weighted_neutra_step")
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(config.learning_rate),
            beta_1=float(config.beta1),
            beta_2=float(config.beta2),
            epsilon=float(config.epsilon),
        )
        self.optimizer.build(self.variables)
        self._compiled_train_step = tf.function(
            self._train_step_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )
        self._compiled_validation = tf.function(
            self._validation_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.transport.forward_and_logdet(latent)

    def inverse_and_forward_logdet(self, physical: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.transport.inverse_and_forward_logdet(physical)

    def log_prob(self, physical: Any) -> tf.Tensor:
        return self.transport.log_prob(physical)

    def train_step(self, physical: Any, log_weights: Any) -> WeightedNeuTraStep:
        rows = _rank2(physical, self.config.dimension, "physical")
        weights = _weights(log_weights, rows.shape[0])
        values = self._compiled_train_step(rows, weights)
        if not bool(values[-1].numpy()):
            raise WeightedNeuTraTrainingError(
                "weighted forward-KL update rejected nonfinite loss or gradient"
            )
        return WeightedNeuTraStep(*values[:-1])

    def validation_batch(self, physical: Any, log_weights: Any) -> WeightedNeuTraValidation:
        rows = _rank2(physical, self.config.dimension, "physical")
        weights = _weights(log_weights, rows.shape[0])
        return WeightedNeuTraValidation(*self._compiled_validation(rows, weights))

    def state_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": "bayesfilter.neutra.weighted_forward_kl_state.v1",
            "config": self.config.manifest_payload(),
            "step": int(self.step.numpy()),
            "variables": [variable.numpy().tolist() for variable in self.variables],
            "optimizer_variables": [value.numpy().tolist() for value in self.optimizer.variables],
            "nonclaims": list(WEIGHTED_NEUTRA_NONCLAIMS),
        }
        return {**payload, "state_hash": _stable_hash(payload)}

    def _train_step_impl(
        self, physical: tf.Tensor, log_weights: tf.Tensor
    ) -> tuple[tf.Tensor, ...]:
        normalized_log_weights = tf.nn.log_softmax(log_weights)
        normalized_weights = tf.stop_gradient(tf.exp(normalized_log_weights))
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            negative_log_prob = -self.transport.log_prob(physical)
            loss = tf.reduce_sum(normalized_weights * negative_log_prob)
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(gradient is None for gradient in gradients):
            raise WeightedNeuTraTrainingError("weighted forward-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        gradient_norm = tf.linalg.global_norm(gradients)
        clipped, _ = tf.clip_by_global_norm(
            gradients,
            tf.constant(float(self.config.gradient_clip_norm), tf.float64),
            use_norm=gradient_norm,
        )
        clipped_norm = tf.linalg.global_norm(clipped)
        finite = tf.reduce_all(
            tf.stack(
                (
                    tf.reduce_all(tf.math.is_finite(loss)),
                    tf.reduce_all(tf.math.is_finite(gradient_norm)),
                    tf.reduce_all(tf.math.is_finite(clipped_norm)),
                    *(tf.reduce_all(tf.math.is_finite(value)) for value in clipped),
                )
            )
        )

        def update() -> tf.Tensor:
            self.optimizer.apply_gradients(zip(clipped, self.variables))
            return tf.cast(self.optimizer.iterations, tf.int64)

        next_step = tf.cond(finite, update, lambda: tf.identity(self.step))
        self.step.assign(next_step)
        ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights)))
        count = tf.cast(tf.size(normalized_weights), tf.float64)
        return (
            loss,
            ess,
            ess / count,
            tf.reduce_max(normalized_weights),
            gradient_norm,
            clipped_norm,
            gradient_norm > tf.constant(float(self.config.gradient_clip_norm), tf.float64),
            tf.identity(self.step),
            finite,
        )

    def _validation_impl(
        self, physical: tf.Tensor, log_weights: tf.Tensor
    ) -> tuple[tf.Tensor, ...]:
        normalized_weights = tf.exp(tf.nn.log_softmax(log_weights))
        negative_log_prob = -self.transport.log_prob(physical)
        latent, _ = self.transport.inverse_and_forward_logdet(physical)
        mean = tf.reduce_sum(normalized_weights[:, tf.newaxis] * latent, axis=0)
        centered = latent - mean
        covariance = tf.matmul(
            centered,
            normalized_weights[:, tf.newaxis] * centered,
            transpose_a=True,
        )
        ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights)))
        count = tf.cast(tf.size(normalized_weights), tf.float64)
        return (
            tf.reduce_sum(normalized_weights * negative_log_prob),
            negative_log_prob,
            normalized_weights,
            latent,
            mean,
            covariance,
            ess,
            ess / count,
            tf.reduce_max(normalized_weights),
        )


class MatchedReverseKLNeuTraTrainer:
    """Reverse-KL comparator using the identical weighted-campaign transport."""

    def __init__(self, config: WeightedNeuTraConfig, target_log_prob_fn: Any) -> None:
        if not callable(target_log_prob_fn):
            raise ValueError("target_log_prob_fn must be callable")
        self.config = config
        self.target_log_prob_fn = target_log_prob_fn
        self.transport = WeightedDenseIAFTransport(config)
        self.variables = self.transport.trainable_variables
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name="matched_rkl_step")
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(config.learning_rate),
            beta_1=float(config.beta1),
            beta_2=float(config.beta2),
            epsilon=float(config.epsilon),
        )
        self.optimizer.build(self.variables)
        self._compiled_train_step = tf.function(
            self._train_step_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )

    def train_step(self, latent: Any) -> WeightedNeuTraStep:
        rows = _rank2(latent, self.config.dimension, "latent")
        values = self._compiled_train_step(rows)
        if not bool(values[-1].numpy()):
            raise WeightedNeuTraTrainingError(
                "matched reverse-KL update rejected nonfinite loss or gradient"
            )
        return WeightedNeuTraStep(*values[:-1])

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.transport.forward_and_logdet(latent)

    def log_prob(self, physical: Any) -> tf.Tensor:
        return self.transport.log_prob(physical)

    def _train_step_impl(self, latent: tf.Tensor) -> tuple[tf.Tensor, ...]:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            physical, logdet = self.transport.forward_and_logdet(latent)
            target = tf.convert_to_tensor(self.target_log_prob_fn(physical), tf.float64)
            loss = tf.reduce_mean(-target - logdet)
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(gradient is None for gradient in gradients):
            raise WeightedNeuTraTrainingError("matched reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        gradient_norm = tf.linalg.global_norm(gradients)
        clipped, _ = tf.clip_by_global_norm(
            gradients,
            tf.constant(float(self.config.gradient_clip_norm), tf.float64),
            use_norm=gradient_norm,
        )
        clipped_norm = tf.linalg.global_norm(clipped)
        finite = tf.reduce_all(
            tf.stack(
                (
                    tf.reduce_all(tf.math.is_finite(loss)),
                    tf.reduce_all(tf.math.is_finite(target)),
                    tf.reduce_all(tf.math.is_finite(logdet)),
                    tf.reduce_all(tf.math.is_finite(gradient_norm)),
                    tf.reduce_all(tf.math.is_finite(clipped_norm)),
                    *(tf.reduce_all(tf.math.is_finite(value)) for value in clipped),
                )
            )
        )

        def update() -> tf.Tensor:
            self.optimizer.apply_gradients(zip(clipped, self.variables))
            return tf.cast(self.optimizer.iterations, tf.int64)

        next_step = tf.cond(finite, update, lambda: tf.identity(self.step))
        self.step.assign(next_step)
        batch_size = tf.cast(tf.shape(latent)[0], tf.float64)
        return (
            loss,
            batch_size,
            tf.constant(1.0, tf.float64),
            tf.math.reciprocal(batch_size),
            gradient_norm,
            clipped_norm,
            gradient_norm > tf.constant(float(self.config.gradient_clip_norm), tf.float64),
            tf.identity(self.step),
            finite,
        )


def _rank2(value: Any, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if tensor.shape.rank != 2 or tensor.shape[-1] != int(dimension):
        raise ValueError(f"{name} must have shape [row, {int(dimension)}]")
    if tensor.shape[0] is None:
        raise ValueError(f"{name} row count must be static")
    tf.debugging.assert_all_finite(tensor, name)
    return tensor


def _weights(value: Any, row_count: int | None) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, tf.float64)
    if row_count is None or tensor.shape != (int(row_count),):
        raise ValueError("log_weights must match the static physical row count")
    tf.debugging.assert_all_finite(tensor, "log_weights")
    return tensor


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
