"""TensorFlow/XLA weighted forward-KL training for NeuTra transports.

The built-in WeightedDenseIAFTransport/default stages are HISTORICAL —
UNFAITHFUL TO THE AUTHOR'S CODE (owner directive 2026-09-25). Sharing the core
does not upgrade their architecture. An explicitly configured canonical IAF
may use this consumer, but the weighted objective remains a documented training
alternative. See docs/reference/neutra-implementation.md and the policy notice.

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

from bayesfilter.inference import neutra_transport_core as _transport_core
from bayesfilter.inference.neutra_training_graphs import FixedShapeTrainingProgram


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
    stage_s_max: tuple[float, ...] = ()
    stage_scale_linear_skip: tuple[bool, ...] = ()
    stage_unbounded_scale_linear: tuple[bool, ...] = ()
    permutation_policy: str = "full_reverse"
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
        if self.permutation_policy not in {"full_reverse", "root_preserving_reverse"}:
            raise ValueError("unsupported permutation_policy")
        for name in (
            "s_max",
            "learning_rate",
            "epsilon",
            "gradient_clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        stage_s_max = tuple(float(value) for value in self.stage_s_max)
        if stage_s_max:
            if len(stage_s_max) != int(self.stages):
                raise ValueError("stage_s_max must be empty or match stages")
            if any(not math.isfinite(value) or value <= 0.0 for value in stage_s_max):
                raise ValueError("stage_s_max values must be finite and positive")
        object.__setattr__(self, "stage_s_max", stage_s_max)
        stage_scale_linear_skip = tuple(self.stage_scale_linear_skip)
        if stage_scale_linear_skip:
            if len(stage_scale_linear_skip) != int(self.stages):
                raise ValueError("stage_scale_linear_skip must be empty or match stages")
            if any(not isinstance(value, bool) for value in stage_scale_linear_skip):
                raise ValueError("stage_scale_linear_skip values must be booleans")
        object.__setattr__(self, "stage_scale_linear_skip", stage_scale_linear_skip)
        stage_unbounded_scale_linear = tuple(self.stage_unbounded_scale_linear)
        if stage_unbounded_scale_linear:
            if len(stage_unbounded_scale_linear) != int(self.stages):
                raise ValueError("stage_unbounded_scale_linear must be empty or match stages")
            if any(not isinstance(value, bool) for value in stage_unbounded_scale_linear):
                raise ValueError("stage_unbounded_scale_linear values must be booleans")
        object.__setattr__(
            self, "stage_unbounded_scale_linear", stage_unbounded_scale_linear
        )
        if stage_scale_linear_skip and stage_unbounded_scale_linear and any(
            pre_cap and unbounded
            for pre_cap, unbounded in zip(
                stage_scale_linear_skip, stage_unbounded_scale_linear, strict=True
            )
        ):
            raise ValueError(
                "pre-cap and unbounded scale-linear paths are mutually exclusive per stage"
            )
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
        payload["stage_s_max"] = list(self.stage_s_max)
        payload["stage_scale_linear_skip"] = list(self.stage_scale_linear_skip)
        payload["stage_unbounded_scale_linear"] = list(
            self.stage_unbounded_scale_linear
        )
        payload["schema"] = "bayesfilter.neutra.weighted_forward_kl_config.v1"
        return payload

    def scale_cap_for_stage(self, stage: int) -> float:
        if not 0 <= int(stage) < int(self.stages):
            raise ValueError("stage index is out of range")
        return (
            float(self.stage_s_max[int(stage)])
            if self.stage_s_max
            else float(self.s_max)
        )

    def scale_linear_skip_for_stage(self, stage: int) -> bool:
        if not 0 <= int(stage) < int(self.stages):
            raise ValueError("stage index is out of range")
        return (
            bool(self.stage_scale_linear_skip[int(stage)])
            if self.stage_scale_linear_skip
            else False
        )

    def unbounded_scale_linear_for_stage(self, stage: int) -> bool:
        if not 0 <= int(stage) < int(self.stages):
            raise ValueError("stage index is out of range")
        return (
            bool(self.stage_unbounded_scale_linear[int(stage)])
            if self.stage_unbounded_scale_linear
            else False
        )


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
    return _transport_core.activation(values, name)


def _dense_masks(dimension: int, hidden_layers: tuple[int, ...]) -> tuple[tf.Tensor, ...]:
    return _transport_core.dense_masks(dimension, hidden_layers)


def _strict_autoregressive_mask(dimension: int) -> tf.Tensor:
    return _transport_core.strict_autoregressive_mask(dimension)


class _DenseAutoregressiveStage:
    autoregressive = True

    def __init__(self, config: WeightedNeuTraConfig, stage: int) -> None:
        self.dimension = int(config.dimension)
        self.activation = str(config.activation)
        self.s_max = config.scale_cap_for_stage(stage)
        self.scale_linear_skip_enabled = config.scale_linear_skip_for_stage(stage)
        self.unbounded_scale_linear_enabled = config.unbounded_scale_linear_for_stage(stage)
        self.masks = _dense_masks(self.dimension, tuple(config.hidden_layers))
        self.scale_linear_skip_mask = _strict_autoregressive_mask(self.dimension)
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
        self.scale_linear_skip_weight = (
            tf.Variable(
                tf.zeros((self.dimension, self.dimension), tf.float64),
                name=f"weighted_neutra_stage_{stage}_scale_linear_skip_weight",
            )
            if self.scale_linear_skip_enabled
            else None
        )
        self.unbounded_scale_linear_weight = (
            tf.Variable(
                tf.zeros((self.dimension, self.dimension), tf.float64),
                name=f"weighted_neutra_stage_{stage}_unbounded_scale_linear_weight",
            )
            if self.unbounded_scale_linear_enabled
            else None
        )

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        output = []
        for weight, bias in zip(self.weights, self.biases):
            output.extend((weight, bias))
        if self.scale_linear_skip_weight is not None:
            output.append(self.scale_linear_skip_weight)
        if self.unbounded_scale_linear_weight is not None:
            output.append(self.unbounded_scale_linear_weight)
        return tuple(output)

    def _scale_linear_skip(self, values: tf.Tensor) -> tf.Tensor:
        return _transport_core._linear(self, "scale_linear_skip_weight", values)

    def _unbounded_scale_linear(self, values: tf.Tensor) -> tf.Tensor:
        return _transport_core._linear(self, "unbounded_scale_linear_weight", values)

    def _unbounded_scale_linear_pullback(self, cotangent: tf.Tensor) -> tf.Tensor:
        return _transport_core._linear_pullback(self, "unbounded_scale_linear_weight", cotangent)

    def _network(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _transport_core.iaf_network(self, values)

    def forward_and_logdet(self, latent: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _transport_core.iaf_forward(self, latent)

    def inverse_and_forward_logdet(self, output: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return _transport_core.iaf_inverse_logdet(self, output)

    def _network_with_cache(
        self, values: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tuple[tf.Tensor, ...]]:
        return _transport_core.iaf_network_cache(self, values)

    def _network_pullback(
        self,
        raw_cotangent: tf.Tensor,
        preactivations: tuple[tf.Tensor, ...],
    ) -> tf.Tensor:
        return _transport_core.iaf_network_pullback(self, raw_cotangent, preactivations)

    def pullback_score(
        self, values: tf.Tensor, output_score: tf.Tensor
    ) -> tf.Tensor:
        return _transport_core.iaf_pullback(self, values, output_score)

    def logdet_score(self, values: tf.Tensor) -> tf.Tensor:
        return _transport_core.iaf_logdet_score(self, values)


class WeightedDenseIAFTransport:
    """Trainable composed IAF with differentiable forward and inverse density."""

    def __init__(self, config: WeightedNeuTraConfig) -> None:
        self.config = config
        self._frozen_identity: Mapping[str, Any] | None = None
        self.stages = tuple(
            _DenseAutoregressiveStage(config, stage)
            for stage in range(int(config.stages))
        )

    def _between_stage_permutation(self, values: tf.Tensor) -> tf.Tensor:
        return _transport_core.Permutation(self.parameter_dim, self.config.permutation_policy).inverse(values)

    @property
    def score_layers(self):
        return _transport_core.interleaved_stages(self.stages, self.parameter_dim,
                                                   self.config.permutation_policy)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(
            variable for stage in self.stages for variable in stage.trainable_variables
        )

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2(latent, self.config.dimension, "latent")
        return _transport_core.compose_forward(self.score_layers, values)

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
            "stage_s_max": list(self.config.stage_s_max),
            "resolved_stage_s_max": [stage.s_max for stage in self.stages],
            "stage_scale_linear_skip": list(self.config.stage_scale_linear_skip),
            "resolved_stage_scale_linear_skip": [
                stage.scale_linear_skip_enabled for stage in self.stages
            ],
            "stage_unbounded_scale_linear": list(
                self.config.stage_unbounded_scale_linear
            ),
            "resolved_stage_unbounded_scale_linear": [
                stage.unbounded_scale_linear_enabled for stage in self.stages
            ],
            "permutation_policy": self.config.permutation_policy,
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
        return _transport_core.compose_pullback(self.score_layers, values, score)

    def log_abs_det_jacobian_score(self, latent: Any) -> tf.Tensor:
        values = tf.convert_to_tensor(latent, tf.float64)
        if values.shape.rank == 1:
            return self.log_abs_det_jacobian_score_batch(values[tf.newaxis, :])[0]
        return self.log_abs_det_jacobian_score_batch(values)

    def log_abs_det_jacobian_score_batch(self, latent: Any) -> tf.Tensor:
        values = _rank2(latent, self.parameter_dim, "latent")
        return _transport_core.compose_pullback(self.score_layers, values)

    def inverse_and_forward_logdet(self, physical: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2(physical, self.config.dimension, "physical")
        latent = _transport_core.compose_inverse(self.score_layers, values)
        return latent, _transport_core.compose_forward(self.score_layers, latent)[1]

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

    def __init__(self, config: WeightedNeuTraConfig, *, transport=None) -> None:
        self.config = config
        self.transport = WeightedDenseIAFTransport(config) if transport is None else transport
        if self.transport.parameter_dim != config.dimension:
            raise ValueError("transport dimension does not match training config")
        self.dtype = getattr(self.transport, "dtype", tf.float64)
        self.variables = self.transport.trainable_variables
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name="weighted_neutra_step")
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=float(config.learning_rate),
            beta_1=float(config.beta1),
            beta_2=float(config.beta2),
            epsilon=float(config.epsilon),
        )
        self.optimizer.build(self.variables)
        self._compiled_train_step = FixedShapeTrainingProgram(self._train_step_impl, jit_compile=bool(config.jit_compile))
        self._compiled_validation = FixedShapeTrainingProgram(self._validation_impl, jit_compile=bool(config.jit_compile))

    def forward_and_logdet(self, latent: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.transport.forward_and_logdet(latent)

    def inverse_and_forward_logdet(self, physical: Any) -> tuple[tf.Tensor, tf.Tensor]:
        return self.transport.inverse_and_forward_logdet(physical)

    def log_prob(self, physical: Any) -> tf.Tensor:
        return self.transport.log_prob(physical)

    def train_step(self, physical: Any, log_weights: Any) -> WeightedNeuTraStep:
        rows = _rank2(physical, self.config.dimension, "physical", dtype=self.dtype)
        weights = _weights(log_weights, rows.shape[0], dtype=self.dtype)
        if int(rows.shape[0]) < 2:
            raise ValueError("training batch size must exceed one")
        values = self._compiled_train_step(rows, weights)
        if not bool(values[-1].numpy()):
            raise WeightedNeuTraTrainingError(
                "weighted forward-KL update rejected nonfinite loss or gradient"
            )
        return WeightedNeuTraStep(*values[:-1])

    def validation_batch(self, physical: Any, log_weights: Any) -> WeightedNeuTraValidation:
        rows = _rank2(physical, self.config.dimension, "physical", dtype=self.dtype)
        weights = _weights(log_weights, rows.shape[0], dtype=self.dtype)
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
        if not isinstance(self.transport, WeightedDenseIAFTransport):
            payload["transport_config"] = self.transport.config.manifest_payload()
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
            tf.constant(float(self.config.gradient_clip_norm), self.dtype),
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
        count = tf.cast(tf.size(normalized_weights), self.dtype)
        return (
            loss,
            ess,
            ess / count,
            tf.reduce_max(normalized_weights),
            gradient_norm,
            clipped_norm,
            gradient_norm > tf.constant(float(self.config.gradient_clip_norm), self.dtype),
            tf.identity(self.step),
            finite,
        )

    def _validation_impl(
        self, physical: tf.Tensor, log_weights: tf.Tensor
    ) -> tuple[tf.Tensor, ...]:
        normalized_weights = tf.exp(tf.nn.log_softmax(log_weights))
        # Reuse the inverse solve for both q_phi(theta) and latent diagnostics.
        latent, forward_logdet = self.transport.inverse_and_forward_logdet(physical)
        dimension = tf.cast(self.config.dimension, self.dtype)
        negative_log_prob = (
            tf.constant(0.5, self.dtype) * tf.reduce_sum(tf.square(latent), axis=-1)
            + tf.constant(0.5, self.dtype)
            * dimension
            * tf.math.log(tf.constant(2.0 * math.pi, self.dtype))
            + forward_logdet
        )
        mean = tf.reduce_sum(normalized_weights[:, tf.newaxis] * latent, axis=0)
        centered = latent - mean
        covariance = tf.matmul(
            centered,
            normalized_weights[:, tf.newaxis] * centered,
            transpose_a=True,
        )
        ess = tf.math.reciprocal(tf.reduce_sum(tf.square(normalized_weights)))
        count = tf.cast(tf.size(normalized_weights), self.dtype)
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
        self._compiled_train_step = FixedShapeTrainingProgram(self._train_step_impl, jit_compile=bool(config.jit_compile))

    def train_step(self, latent: Any) -> WeightedNeuTraStep:
        rows = _rank2(latent, self.config.dimension, "latent")
        if int(rows.shape[0]) < 2:
            raise ValueError("training batch size must exceed one")
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


def _rank2(value: Any, dimension: int, name: str, *, dtype=tf.float64) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype)
    if tensor.shape.rank != 2 or tensor.shape[-1] != int(dimension):
        raise ValueError(f"{name} must have shape [row, {int(dimension)}]")
    if tensor.shape[0] is None:
        raise ValueError(f"{name} row count must be static")
    tf.debugging.assert_all_finite(tensor, name)
    return tensor


def _weights(value: Any, row_count: int | None, *, dtype=tf.float64) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype)
    if row_count is None or tensor.shape != (int(row_count),):
        raise ValueError("log_weights must match the static physical row count")
    tf.debugging.assert_all_finite(tensor, "log_weights")
    return tensor


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
