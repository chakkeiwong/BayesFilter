"""GPU/XLA-oriented reverse-KL training for BayesFilter NeuTra transports.

The target supplies graph-native values and scores. GradientTape is restricted
to the trainable transport; it never differentiates through the target/filter.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import (
    finalize_dense_iaf_neutra_artifact_payload,
)

# The plain dense-IAF campaign API is retained in a separate compatibility
# module for existing end-to-end and PP-UKF lanes.  The current trainer above
# remains the default named-family training implementation.
from bayesfilter.inference.neutra_training_legacy import (
    NeuTraTrainingError,
    PlainDenseIAFSegmentedTrainingResult,
    PlainDenseIAFTrainingConfig,
    PlainDenseIAFTrainingResult,
    PlainDenseIAFTransport,
    restore_plain_dense_iaf_flow,
    train_plain_dense_iaf,
    train_plain_dense_iaf_infrastructure_segments,
)


NEUTRA_TRAINING_NONCLAIMS = (
    "reverse-KL trainer engineering surface only",
    "training loss is not a transport promotion criterion",
    "no HMC or sampler-validity claim",
    "no posterior-correctness claim",
    "no predictive or scientific-validity claim",
    "no default or production-readiness claim",
)

DSGE_PAPER_NEUTRA_FAMILY = "dsge_paper_dense_iaf"
SSL_LSTM_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_capacity_dense_iaf"
SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_tuned_capacity_dense_iaf"
SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_deep_capacity_dense_iaf"
SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY = "ssl_lstm_wide_capacity_dense_iaf"
COMPOSED_NEUTRA_FAMILIES = frozenset(
    (
        DSGE_PAPER_NEUTRA_FAMILY,
        SSL_LSTM_CAPACITY_NEUTRA_FAMILY,
        SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
        SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
        SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
    )
)
DSGE_PAPER_TRAINING_STEPS = 5000
DSGE_PAPER_TRAINING_BATCH_SIZE = 480
DSGE_PAPER_LR_BOUNDARIES = (999, 3999)


@dataclass(frozen=True)
class NeuTraTrainerConfig:
    """Configuration for a trainable diagonal-affine or dense-IAF transport."""

    dimension: int
    family: str = "dense_iaf"
    hidden_layers: tuple[int, ...] = (8, 8)
    activation: str = "tanh"
    s_max: float = 1.0
    initialization_scale: float = 0.02
    initialization_seed: tuple[int, int] = (20260714, 2101)
    learning_rate: float = 1.0e-3
    learning_rate_schedule: str = "constant"
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    gradient_clip_norm: float = 10.0
    gradient_clip_mode: str = "global"
    stages: int = 1
    fixed_translation: tuple[float, ...] = ()
    target_parameter_names: tuple[str, ...] = ()
    target_chart: str = "unspecified"
    target_signature: str | None = None
    target_adapter_signature: str | None = None
    jit_compile: bool = True

    def __post_init__(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError("dimension must be positive")
        if self.family not in {"affine_diag", "dense_iaf", *COMPOSED_NEUTRA_FAMILIES}:
            raise ValueError(
                "unsupported NeuTra training family"
            )
        if self.family in {"dense_iaf", *COMPOSED_NEUTRA_FAMILIES} and (
            not self.hidden_layers or any(int(width) <= 0 for width in self.hidden_layers)
        ):
            raise ValueError("dense_iaf hidden layers must be positive")
        if self.activation not in {"elu", "tanh", "relu"}:
            raise ValueError("unsupported activation")
        if not math.isfinite(self.s_max) or self.s_max <= 0.0:
            raise ValueError("s_max must be finite and positive")
        if not math.isfinite(self.initialization_scale) or self.initialization_scale < 0.0:
            raise ValueError("initialization_scale must be finite and nonnegative")
        if len(self.initialization_seed) != 2:
            raise ValueError("initialization_seed must contain two integers")
        for name in ("learning_rate", "epsilon", "gradient_clip_norm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("beta1", "beta2"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between zero and one")
        if self.learning_rate_schedule not in {
            "constant",
            "paper_piecewise",
            "adaptive_constant",
        }:
            raise ValueError("unsupported learning_rate_schedule")
        if self.gradient_clip_mode not in {"global", "per_variable"}:
            raise ValueError("unsupported gradient_clip_mode")
        if int(self.stages) <= 0:
            raise ValueError("stages must be positive")
        translation = tuple(float(value) for value in self.fixed_translation)
        if any(not math.isfinite(value) for value in translation):
            raise ValueError("fixed_translation must be finite")
        names = tuple(str(value) for value in self.target_parameter_names)
        if len(set(names)) != len(names):
            raise ValueError("target_parameter_names must be unique")
        if self.family in COMPOSED_NEUTRA_FAMILIES:
            if self.family == DSGE_PAPER_NEUTRA_FAMILY:
                hidden_layers = (int(self.dimension), int(self.dimension))
            elif self.family == SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY:
                hidden_layers = (32, 32, 32)
            elif self.family == SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY:
                hidden_layers = (64, 64)
            else:
                hidden_layers = (32, 32)
            required = {
                "hidden_layers": hidden_layers,
                "activation": "elu",
                "s_max": 1.0,
                "epsilon": 1.0e-7,
                "beta1": 0.9,
                "beta2": 0.999,
                "gradient_clip_mode": "per_variable",
                "stages": 3,
                "target_chart": "identity",
            }
            if self.family in {
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
            }:
                required["learning_rate_schedule"] = "adaptive_constant"
            else:
                required.update(
                    {
                        "initialization_scale": 0.02,
                        "learning_rate": 0.01,
                        "learning_rate_schedule": "paper_piecewise",
                        "gradient_clip_norm": 10.0,
                    }
                )
            actual = {
                "hidden_layers": tuple(self.hidden_layers),
                "activation": self.activation,
                "s_max": float(self.s_max),
                "initialization_scale": float(self.initialization_scale),
                "learning_rate": float(self.learning_rate),
                "learning_rate_schedule": self.learning_rate_schedule,
                "epsilon": float(self.epsilon),
                "beta1": float(self.beta1),
                "beta2": float(self.beta2),
                "gradient_clip_norm": float(self.gradient_clip_norm),
                "gradient_clip_mode": self.gradient_clip_mode,
                "stages": int(self.stages),
                "target_chart": self.target_chart,
            }
            mismatches = [key for key, value in required.items() if actual[key] != value]
            if mismatches:
                raise ValueError(
                    f"{self.family} preset mismatch: " + ", ".join(mismatches)
                )
            if self.family in {
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
            }:
                if not 1.0e-4 <= float(self.learning_rate) <= 2.0e-3:
                    raise ValueError("tuned capacity learning_rate outside search contract")
                if float(self.initialization_scale) not in {0.005, 0.01, 0.02}:
                    raise ValueError(
                        "tuned capacity initialization_scale outside search contract"
                    )
                if float(self.gradient_clip_norm) not in {5.0, 10.0}:
                    raise ValueError(
                        "tuned capacity gradient_clip_norm outside search contract"
                    )
            if len(translation) != int(self.dimension):
                raise ValueError(f"{self.family} requires fixed_translation")
            if len(names) != int(self.dimension):
                raise ValueError(f"{self.family} requires target_parameter_names")
            for field_name in ("target_signature", "target_adapter_signature"):
                value = getattr(self, field_name)
                if value is None or len(str(value)) != 64:
                    raise ValueError(f"{self.family} requires {field_name}")
        else:
            if self.learning_rate_schedule != "constant":
                raise ValueError(
                    "paper_piecewise is reserved for named composed IAF presets"
                )
            if int(self.stages) != 1 or translation:
                raise ValueError("stages and fixed_translation are reserved for composed IAF")

    def manifest_payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["hidden_layers"] = list(self.hidden_layers)
        payload["initialization_seed"] = list(self.initialization_seed)
        payload["fixed_translation"] = list(self.fixed_translation)
        payload["target_parameter_names"] = list(self.target_parameter_names)
        payload["schema"] = "bayesfilter.neutra.trainer_config.v1"
        return payload


def dsge_paper_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the frozen Rotemberg/SGU plain-NeuTra procedure preset."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=DSGE_PAPER_NEUTRA_FAMILY,
        hidden_layers=(int(dimension), int(dimension)),
        activation="elu",
        s_max=1.0,
        initialization_scale=0.02,
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=0.01,
        learning_rate_schedule="paper_piecewise",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=10.0,
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the three-stage 32x32 SSL-LSTM capacity-repair preset."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(32, 32),
        activation="elu",
        s_max=1.0,
        initialization_scale=0.02,
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=0.01,
        learning_rate_schedule="paper_piecewise",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=10.0,
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_tuned_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    learning_rate: float,
    initialization_scale: float,
    gradient_clip_norm: float,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the bounded `(32,32)` SSL-LSTM tuning preset."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(32, 32),
        activation="elu",
        s_max=1.0,
        initialization_scale=float(initialization_scale),
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=float(learning_rate),
        learning_rate_schedule="adaptive_constant",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=float(gradient_clip_norm),
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_deep_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    learning_rate: float,
    initialization_scale: float,
    gradient_clip_norm: float,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the explicitly labeled three-hidden-layer SSL-LSTM diagnostic."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(32, 32, 32),
        activation="elu",
        s_max=1.0,
        initialization_scale=float(initialization_scale),
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=float(learning_rate),
        learning_rate_schedule="adaptive_constant",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=float(gradient_clip_norm),
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def ssl_lstm_wide_capacity_neutra_config(
    *,
    dimension: int,
    fixed_translation: Sequence[float],
    target_parameter_names: Sequence[str],
    target_signature: str,
    target_adapter_signature: str,
    learning_rate: float,
    initialization_scale: float,
    gradient_clip_norm: float,
    initialization_seed: tuple[int, int] = (20260715, 4101),
    jit_compile: bool = True,
) -> NeuTraTrainerConfig:
    """Return the explicitly labeled two-hidden-layer 64x64 diagnostic."""

    return NeuTraTrainerConfig(
        dimension=int(dimension),
        family=SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
        hidden_layers=(64, 64),
        activation="elu",
        s_max=1.0,
        initialization_scale=float(initialization_scale),
        initialization_seed=tuple(int(value) for value in initialization_seed),
        learning_rate=float(learning_rate),
        learning_rate_schedule="adaptive_constant",
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-7,
        gradient_clip_norm=float(gradient_clip_norm),
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=tuple(float(value) for value in fixed_translation),
        target_parameter_names=tuple(str(value) for value in target_parameter_names),
        target_chart="identity",
        target_signature=str(target_signature),
        target_adapter_signature=str(target_adapter_signature),
        jit_compile=bool(jit_compile),
    )


def dsge_paper_learning_rate(
    learning_rate: float = 0.01,
) -> tf.keras.optimizers.schedules.PiecewiseConstantDecay:
    """Build the exact zero-based schedule used by the DSGE NeuTra runner."""

    rate = float(learning_rate)
    return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=list(DSGE_PAPER_LR_BOUNDARIES),
        values=[rate, rate * 0.1, rate * 0.01],
    )


@dataclass(frozen=True)
class NeuTraTrainStep:
    """Tensor outputs from one reverse-KL gradient or update evaluation."""

    loss: tf.Tensor
    surrogate: tf.Tensor
    target_value_mean: tf.Tensor
    logdet_mean: tf.Tensor
    gradient_norm: tf.Tensor
    clipped_gradient_norm: tf.Tensor
    clipping_applied: tf.Tensor
    step: tf.Tensor


@dataclass(frozen=True)
class NeuTraValidation:
    """Non-updating reverse-KL diagnostics on an independent base batch."""

    per_sample_loss: tf.Tensor
    target_value: tf.Tensor
    theta: tf.Tensor
    logdet: tf.Tensor
    scale_log: tf.Tensor
    scale_logits: tf.Tensor
    hidden_preactivations: tf.Tensor


class _TrainableTransport:
    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        raise NotImplementedError

    @property
    def variable_keys(self) -> tuple[str, ...]:
        raise NotImplementedError

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        """Return raw scale logits and hidden preactivations for diagnostics."""

        scale_log = self.scale_log(z)
        batch = tf.shape(z)[0]
        return scale_log, tf.zeros((batch, 0, 0), dtype=z.dtype)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        raise NotImplementedError


class _TrainableAffineDiagonal(_TrainableTransport):
    def __init__(self, config: NeuTraTrainerConfig) -> None:
        self.dimension = int(config.dimension)
        self.shift = tf.Variable(
            tf.zeros((self.dimension,), dtype=tf.float64),
            name="neutra_affine_shift",
        )
        self.raw_scale = tf.Variable(
            tf.zeros((self.dimension,), dtype=tf.float64),
            name="neutra_affine_raw_scale",
        )

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return (self.shift, self.raw_scale)

    @property
    def variable_keys(self) -> tuple[str, ...]:
        return ("shift", "raw_scale")

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale = tf.exp(self.raw_scale)
        theta = self.shift + z * scale
        logdet = tf.zeros(tf.shape(z)[:-1], dtype=z.dtype) + tf.reduce_sum(
            self.raw_scale
        )
        return theta, logdet

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z) + self.raw_scale

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self.scale_log(z), tf.zeros((tf.shape(z)[0], 0, 0), dtype=z.dtype)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        return {
            "component_id": component_id,
            "kind": "affine",
            "dim": self.dimension,
            "dtype": "float64",
            "offset": _tensor_values(self.shift),
            "scale": _tensor_values(tf.exp(self.raw_scale)),
        }


class _TrainableDenseIAF(_TrainableTransport):
    def __init__(self, config: NeuTraTrainerConfig, *, stage_index: int = 0) -> None:
        self.dimension = int(config.dimension)
        self.stage_index = int(stage_index)
        self.hidden_layers = tuple(int(width) for width in config.hidden_layers)
        self.activation = str(config.activation)
        self.s_max = float(config.s_max)
        self.masks = _dense_iaf_masks(self.dimension, self.hidden_layers)
        layer_sizes = (self.dimension, *self.hidden_layers, 2 * self.dimension)
        weights = []
        biases = []
        seed = tf.random.experimental.stateless_fold_in(
            tf.constant(config.initialization_seed, dtype=tf.int32),
            self.stage_index,
        )
        for index, (input_width, output_width) in enumerate(
            zip(layer_sizes[:-1], layer_sizes[1:])
        ):
            layer_seed = tf.random.experimental.stateless_fold_in(seed, index)
            scale = 0.0 if index == len(layer_sizes) - 2 else float(
                config.initialization_scale
            )
            initial_weight = tf.random.stateless_normal(
                (input_width, output_width),
                seed=layer_seed,
                dtype=tf.float64,
            ) * tf.cast(scale, tf.float64)
            weights.append(
                tf.Variable(
                    initial_weight,
                    name=f"neutra_dense_iaf_{self.stage_index}_weight_{index}",
                )
            )
            biases.append(
                tf.Variable(
                    tf.zeros((output_width,), dtype=tf.float64),
                    name=f"neutra_dense_iaf_{self.stage_index}_bias_{index}",
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

    @property
    def variable_keys(self) -> tuple[str, ...]:
        output = []
        for index in range(len(self.weights)):
            output.extend((f"weight[{index}]", f"bias[{index}]"))
        return tuple(output)

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale_log, shift = self._network(z)
        theta = z * tf.exp(scale_log) + shift
        return theta, tf.reduce_sum(scale_log, axis=-1)

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        scale_log, _ = self._network(z)
        return scale_log

    def _network(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        scale_log, shift, _, _ = self._network_with_diagnostics(z)
        return scale_log, shift

    def _network_with_diagnostics(
        self, z: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        h = z
        preactivations = []
        for weight, bias, mask in zip(
            self.weights[:-1], self.biases[:-1], self.masks[:-1]
        ):
            h = tf.matmul(h, weight * mask) + bias
            preactivations.append(h)
            h = _activation(h, self.activation)
        raw = tf.matmul(h, self.weights[-1] * self.masks[-1]) + self.biases[-1]
        scale_logits = raw[..., : self.dimension]
        shift = raw[..., self.dimension :]
        scale_log = self.s_max * tf.math.tanh(scale_logits / self.s_max)
        max_width = max(self.hidden_layers, default=0)
        padded = [
            tf.pad(values, [[0, 0], [0, max_width - int(values.shape[-1])]])
            for values in preactivations
        ]
        hidden = (
            tf.stack(padded, axis=1)
            if padded
            else tf.zeros((tf.shape(z)[0], 0, max_width), dtype=z.dtype)
        )
        return scale_log, shift, scale_logits, hidden

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        _, _, scale_logits, hidden = self._network_with_diagnostics(z)
        return scale_logits, hidden

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        return {
            "component_id": component_id,
            "kind": "dense_autoregressive_iaf",
            "dim": self.dimension,
            "hidden_layers": list(self.hidden_layers),
            "activation": self.activation,
            "s_max": self.s_max,
            "masks_policy": "legacy_degree_masks_v1",
            "dtype": "float64",
            "weights": [_tensor_values(weight) for weight in self.weights],
            "biases": [_tensor_values(bias) for bias in self.biases],
        }


class _FixedMixingReverse(_TrainableTransport):
    def __init__(self, dimension: int) -> None:
        self.dimension = int(dimension)
        self.matrix = tf.reverse(tf.eye(self.dimension, dtype=tf.float64), axis=(0,))

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return ()

    @property
    def variable_keys(self) -> tuple[str, ...]:
        return ()

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.matmul(z, self.matrix), tf.zeros(tf.shape(z)[:-1], z.dtype)

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        return {
            "component_id": component_id,
            "kind": "mixing_linear",
            "dim": self.dimension,
            "dtype": "float64",
            "matrix": _tensor_values(self.matrix),
        }


class _FixedTranslation(_TrainableTransport):
    def __init__(self, values: Sequence[float]) -> None:
        self.offset = tf.constant(tuple(float(value) for value in values), tf.float64)
        self.dimension = int(self.offset.shape[0])

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return ()

    @property
    def variable_keys(self) -> tuple[str, ...]:
        return ()

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return z + self.offset, tf.zeros(tf.shape(z)[:-1], z.dtype)

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        return tf.zeros_like(z)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        return {
            "component_id": component_id,
            "kind": "affine",
            "dim": self.dimension,
            "dtype": "float64",
            "offset": _tensor_values(self.offset),
            "scale": _tensor_values(tf.ones_like(self.offset)),
        }


class _TrainableComposedIAF(_TrainableTransport):
    def __init__(self, config: NeuTraTrainerConfig) -> None:
        components: list[_TrainableTransport] = []
        for stage in range(int(config.stages)):
            components.append(_TrainableDenseIAF(config, stage_index=stage))
            if stage + 1 < int(config.stages):
                components.append(_FixedMixingReverse(config.dimension))
        components.append(_FixedTranslation(config.fixed_translation))
        self.components = tuple(components)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(
            variable
            for component in self.components
            for variable in component.trainable_variables
        )

    @property
    def variable_keys(self) -> tuple[str, ...]:
        keys = []
        for component_index, component in enumerate(self.components):
            keys.extend(
                f"component[{component_index}].{key}" for key in component.variable_keys
            )
        return tuple(keys)

    def forward_and_logdet(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = z
        logdet = tf.zeros(tf.shape(z)[:-1], z.dtype)
        for component in self.components:
            values, increment = component.forward_and_logdet(values)
            logdet = logdet + increment
        return values, logdet

    def scale_log(self, z: tf.Tensor) -> tf.Tensor:
        values = z
        stage_scales = []
        for component in self.components:
            if isinstance(component, _TrainableDenseIAF):
                stage_scales.append(component.scale_log(values))
            values, _ = component.forward_and_logdet(values)
        return tf.concat(stage_scales, axis=-1)

    def diagnostics(self, z: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        values = z
        stage_logits = []
        stage_hidden = []
        for component in self.components:
            if isinstance(component, _TrainableDenseIAF):
                scale_log, shift, scale_logits, hidden = (
                    component._network_with_diagnostics(values)
                )
                stage_logits.append(scale_logits)
                stage_hidden.append(hidden)
                values = values * tf.exp(scale_log) + shift
            else:
                values, _ = component.forward_and_logdet(values)
        return tf.stack(stage_logits, axis=1), tf.stack(stage_hidden, axis=1)

    def frozen_component_payload(self, *, component_id: str) -> Mapping[str, Any]:
        raise NeuTraTrainingError("composed transport serializes its ordered children")

    def frozen_components(self) -> tuple[Mapping[str, Any], ...]:
        output = []
        iaf_index = 0
        mix_index = 0
        for component in self.components:
            if isinstance(component, _TrainableDenseIAF):
                component_id = f"dense_iaf_{iaf_index:02d}"
                iaf_index += 1
            elif isinstance(component, _FixedMixingReverse):
                component_id = f"mixing_reverse_{mix_index:02d}"
                mix_index += 1
            else:
                component_id = "fixed_translation_00"
            output.append(component.frozen_component_payload(component_id=component_id))
        return tuple(output)


class NeuTraReverseKLTrainer:
    """Reverse-KL optimizer with an explicit target-score boundary."""

    def __init__(self, target: Any, config: NeuTraTrainerConfig) -> None:
        self.target = target
        self.config = config
        self.transport: _TrainableTransport
        if config.family == "affine_diag":
            self.transport = _TrainableAffineDiagonal(config)
        elif config.family == "dense_iaf":
            self.transport = _TrainableDenseIAF(config)
        else:
            self.transport = _TrainableComposedIAF(config)
            _validate_named_composed_target(target, config)
        self.variables = self.transport.trainable_variables
        if not self.variables:
            raise NeuTraTrainingError("trainer requires trainable variables")
        self.step = tf.Variable(0, dtype=tf.int64, trainable=False, name="neutra_step")
        self.optimizer: tf.keras.optimizers.Adam | None = None
        if config.family in COMPOSED_NEUTRA_FAMILIES:
            self.first_moments = ()
            self.second_moments = ()
            optimizer_learning_rate: Any = (
                float(config.learning_rate)
                if config.family in {
                    SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
                    SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
                    SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
                }
                else dsge_paper_learning_rate(config.learning_rate)
            )
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=optimizer_learning_rate,
                beta_1=config.beta1,
                beta_2=config.beta2,
                epsilon=config.epsilon,
            )
            self.optimizer.build(self.variables)
        else:
            self.first_moments = tuple(
                tf.Variable(
                    tf.zeros_like(variable),
                    trainable=False,
                    name=f"neutra_m_{index}",
                )
                for index, variable in enumerate(self.variables)
            )
            self._generic_learning_rate = tf.Variable(
                float(config.learning_rate),
                dtype=tf.float64,
                trainable=False,
                name="neutra_generic_learning_rate",
            )
            self.second_moments = tuple(
                tf.Variable(
                    tf.zeros_like(variable),
                    trainable=False,
                    name=f"neutra_v_{index}",
                )
                for index, variable in enumerate(self.variables)
            )
        batch_program = self._train_step_impl
        self._compiled_train_step = tf.function(
            batch_program,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )
        self._compiled_validation = tf.function(
            self._validation_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )
        # The process-parallel target route evaluates values/scores outside
        # this process.  This parent-only program keeps the transport update
        # on the selected GPU while accepting detached worker outputs.
        self._compiled_external_train_step = tf.function(
            self._external_train_step_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )
        # The chunked bridge keeps target evaluation and reverse-KL gradient
        # graphs at a bounded static shape. The Python caller aggregates raw
        # gradients before one optimizer update, preserving full-batch means.
        self._compiled_external_gradients = tf.function(
            self._external_gradients_impl,
            jit_compile=bool(config.jit_compile),
            reduce_retracing=True,
        )

    def forward_and_logdet(self, z: Any) -> tuple[tf.Tensor, tf.Tensor]:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        return self.transport.forward_and_logdet(values)

    def loss_and_gradients(
        self,
        z: Any,
    ) -> tuple[NeuTraTrainStep, tuple[tf.Tensor, ...]]:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        outputs = self._loss_and_gradients_impl(values)
        return outputs[0], outputs[1]

    def train_step(self, z: Any) -> NeuTraTrainStep:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        rows = self._compiled_train_step(values)
        if not bool(rows[-1].numpy()):
            raise NeuTraTrainingError(
                "compiled NeuTra step rejected nonfinite loss or gradient before update"
            )
        return NeuTraTrainStep(*rows[:-1])

    def train_step_with_external_value_score(
        self,
        z: Any,
        target_value: Any,
        target_score: Any,
    ) -> NeuTraTrainStep:
        """Update the transport from values/scores returned by CPU workers.

        ``target_value`` and ``target_score`` must correspond, in order, to
        ``transport.forward_and_logdet(z)``.  The target is deliberately not
        called in this method; its analytic score is treated as a detached
        custom-gradient payload, matching the DSGE-HMC worker bridge.
        """

        values = _rank2(z, dimension=self.config.dimension, name="z")
        value_tensor = tf.convert_to_tensor(target_value, tf.float64)
        score_tensor = tf.convert_to_tensor(target_score, tf.float64)
        if value_tensor.shape != (values.shape[0],):
            raise ValueError("external target value shape mismatch")
        if score_tensor.shape != values.shape:
            raise ValueError("external target score shape mismatch")
        rows = self._compiled_external_train_step(values, value_tensor, score_tensor)
        if not bool(rows[-1].numpy()):
            raise NeuTraTrainingError(
                "external NeuTra step rejected nonfinite loss or gradient before update"
            )
        return NeuTraTrainStep(*rows[:-1])

    def train_step_with_external_value_score_chunks(
        self,
        z_chunks: Sequence[Any],
        target_value_chunks: Sequence[Any],
        target_score_chunks: Sequence[Any],
        row_counts: Sequence[int],
    ) -> NeuTraTrainStep:
        """Apply one full-batch update from fixed-shape detached-score chunks.

        Each chunk is evaluated with the same transport state and contributes
        ``row_count / total_rows`` of the un-clipped reverse-KL gradient. The
        aggregate is clipped once and applied once, exactly matching the
        full-batch external-score contract up to floating-point summation
        order. Chunks may contain deterministic padding; ``row_counts``
        excludes those rows from every aggregate statistic.
        """

        if not z_chunks or not (
            len(z_chunks) == len(target_value_chunks)
            == len(target_score_chunks)
            == len(row_counts)
        ):
            raise ValueError("chunk sequences must be nonempty and have equal length")
        counts = tuple(int(count) for count in row_counts)
        if any(count <= 0 for count in counts):
            raise ValueError("row_counts must be positive")
        total_rows = sum(counts)
        raw_outputs = []
        for z, target_value, target_score, count in zip(
            z_chunks,
            target_value_chunks,
            target_score_chunks,
            counts,
            strict=True,
        ):
            values = _rank2(z, dimension=self.config.dimension, name="z chunk")
            value_tensor = tf.convert_to_tensor(target_value, tf.float64)
            score_tensor = tf.convert_to_tensor(target_score, tf.float64)
            if value_tensor.shape != (values.shape[0],):
                raise ValueError("external chunk target value shape mismatch")
            if score_tensor.shape != values.shape:
                raise ValueError("external chunk target score shape mismatch")
            if count > int(values.shape[0]):
                raise ValueError("row_count exceeds chunk row count")
            raw_outputs.append(
                self._compiled_external_gradients(
                    values,
                    value_tensor,
                    score_tensor,
                    tf.concat(
                        (
                            tf.ones((count,), tf.float64),
                            tf.zeros((int(values.shape[0]) - count,), tf.float64),
                        ),
                        axis=0,
                    ),
                )
            )

        weight = tf.cast(1.0 / float(total_rows), tf.float64)
        loss = weight * tf.add_n([output[0] for output in raw_outputs])
        surrogate = tf.add_n(
            [weight * output[1] for output in raw_outputs]
        )
        target_value_mean = tf.add_n(
            [weight * output[2] for output in raw_outputs]
        )
        logdet_mean = tf.add_n(
            [weight * output[3] for output in raw_outputs]
        )
        gradients = tuple(
            weight * tf.add_n([output[4 + index] for output in raw_outputs])
            for index in range(len(self.variables))
        )
        for index, gradient in enumerate(gradients):
            _assert_finite(gradient, f"chunked gradient[{index}]")
        gradient_norm = tf.linalg.global_norm(gradients)
        if self.config.gradient_clip_mode == "per_variable":
            clipped = tuple(
                tf.clip_by_norm(gradient, self.config.gradient_clip_norm)
                for gradient in gradients
            )
            clipping_applied = tf.reduce_any(
                tf.stack(
                    [
                        tf.linalg.norm(gradient) > self.config.gradient_clip_norm
                        for gradient in gradients
                    ]
                )
            )
        else:
            clipped_rows, _ = tf.clip_by_global_norm(
                gradients,
                tf.cast(self.config.gradient_clip_norm, tf.float64),
                use_norm=gradient_norm,
            )
            clipped = tuple(clipped_rows)
            clipping_applied = gradient_norm > tf.cast(
                self.config.gradient_clip_norm, gradient_norm.dtype
            )
        clipped_norm = tf.linalg.global_norm(clipped)
        finite_step = bool(
            tf.reduce_all(
                tf.stack(
                    (
                        tf.reduce_all(tf.math.is_finite(loss)),
                        tf.reduce_all(tf.math.is_finite(surrogate)),
                        tf.reduce_all(tf.math.is_finite(target_value_mean)),
                        tf.reduce_all(tf.math.is_finite(logdet_mean)),
                        tf.reduce_all(tf.math.is_finite(gradient_norm)),
                        tf.reduce_all(tf.math.is_finite(clipped_norm)),
                        *(tf.reduce_all(tf.math.is_finite(gradient)) for gradient in clipped),
                    )
                )
            ).numpy()
        )
        if not finite_step:
            raise NeuTraTrainingError(
                "chunked external NeuTra step rejected nonfinite loss or gradient"
            )
        if self.optimizer is not None:
            self.optimizer.apply_gradients(zip(clipped, self.variables, strict=True))
            next_step = tf.cast(self.optimizer.iterations, tf.int64)
        else:
            next_step = self.step + tf.constant(1, dtype=tf.int64)
            beta1 = tf.cast(self.config.beta1, tf.float64)
            beta2 = tf.cast(self.config.beta2, tf.float64)
            learning_rate = tf.cast(self._generic_learning_rate, tf.float64)
            epsilon = tf.cast(self.config.epsilon, tf.float64)
            step_float = tf.cast(next_step, tf.float64)
            for variable, gradient, first, second in zip(
                self.variables, clipped, self.first_moments, self.second_moments, strict=True
            ):
                first.assign(beta1 * first + (1.0 - beta1) * gradient)
                second.assign(beta2 * second + (1.0 - beta2) * tf.square(gradient))
                variable.assign_sub(
                    learning_rate * (first / (1.0 - tf.pow(beta1, step_float))) /
                    (tf.sqrt(second / (1.0 - tf.pow(beta2, step_float))) + epsilon)
                )
                _assert_finite(variable, "updated transport variable")
        self.step.assign(next_step)
        return NeuTraTrainStep(
            loss=loss,
            surrogate=surrogate,
            target_value_mean=target_value_mean,
            logdet_mean=logdet_mean,
            gradient_norm=gradient_norm,
            clipped_gradient_norm=clipped_norm,
            clipping_applied=clipping_applied,
            step=tf.identity(self.step),
        )

    def validation_batch(self, z: Any) -> NeuTraValidation:
        values = _rank2(z, dimension=self.config.dimension, name="z")
        rows = self._compiled_validation(values)
        return NeuTraValidation(*rows)

    def validation_batch_with_external_value(
        self,
        z: Any,
        target_value: Any,
    ) -> NeuTraValidation:
        """Evaluate validation diagnostics from detached worker values."""

        values = _rank2(z, dimension=self.config.dimension, name="z")
        value_tensor = tf.convert_to_tensor(target_value, tf.float64)
        if value_tensor.shape != (values.shape[0],):
            raise ValueError("external validation value shape mismatch")
        theta, logdet = self.transport.forward_and_logdet(values)
        scale_log = self.transport.scale_log(values)
        scale_logits, hidden_preactivations = self.transport.diagnostics(values)
        per_sample_loss = -tf.stop_gradient(value_tensor) - logdet
        _assert_finite(per_sample_loss, "external validation loss")
        _assert_finite(theta, "external validation theta")
        _assert_finite(scale_log, "external validation scale_log")
        _assert_finite(scale_logits, "external validation scale_logits")
        _assert_finite(
            hidden_preactivations,
            "external validation hidden_preactivations",
        )
        return NeuTraValidation(
            per_sample_loss=per_sample_loss,
            target_value=tf.stop_gradient(value_tensor),
            theta=theta,
            logdet=logdet,
            scale_log=scale_log,
            scale_logits=scale_logits,
            hidden_preactivations=hidden_preactivations,
        )

    def sample_base(self, *, batch_size: int, seed: Sequence[int]) -> tf.Tensor:
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if len(tuple(seed)) != 2:
            raise ValueError("seed must contain two integers")
        return tf.random.stateless_normal(
            (int(batch_size), int(self.config.dimension)),
            seed=tf.constant(tuple(int(item) for item in seed), dtype=tf.int32),
            dtype=tf.float64,
        )

    def learning_rate_at(self, iteration: int) -> tf.Tensor:
        if int(iteration) < 0:
            raise ValueError("iteration must be nonnegative")
        if self.config.family in {
            SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
        }:
            if self.optimizer is None:
                raise NeuTraTrainingError("tuned capacity optimizer is unavailable")
            return tf.cast(self.optimizer.learning_rate, tf.float64)
        if self.config.learning_rate_schedule == "paper_piecewise":
            return tf.cast(
                dsge_paper_learning_rate(self.config.learning_rate)(
                    tf.constant(int(iteration), tf.int64)
                ),
                tf.float64,
            )
        if self.optimizer is None:
            return tf.identity(self._generic_learning_rate)
        return tf.constant(self.config.learning_rate, tf.float64)

    def set_learning_rate(self, learning_rate: float) -> None:
        """Assign the effective LR for the tuned family without resetting Adam."""

        value = float(learning_rate)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("learning_rate must be finite and positive")
        if value > float(self.config.learning_rate):
            raise ValueError("learning_rate cannot exceed configured initial rate")
        if self.optimizer is None:
            self._generic_learning_rate.assign(value)
        else:
            if self.config.family not in {
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
            }:
                raise NeuTraTrainingError(
                    "mutable learning rate is restricted to generic or tuned capacity families"
                )
            self.optimizer.learning_rate.assign(value)

    def state_payload(self) -> Mapping[str, Any]:
        payload = {
            "schema": "bayesfilter.neutra.reverse_kl_trainer_state.v1",
            "config": self.config.manifest_payload(),
            "step": int(self.step.numpy()),
            "variable_keys": list(self.transport.variable_keys),
            "variables": [_tensor_values(variable) for variable in self.variables],
            "first_moments": [_tensor_values(value) for value in self.first_moments],
            "second_moments": [_tensor_values(value) for value in self.second_moments],
            "effective_learning_rate": float(self.learning_rate_at(int(self.step.numpy())).numpy()),
            "nonclaims": list(NEUTRA_TRAINING_NONCLAIMS),
        }
        if self.optimizer is not None:
            payload["optimizer_variables"] = [
                _native_tensor_values(value) for value in self.optimizer.variables
            ]
            payload["optimizer_variable_specs"] = [
                {
                    "name": value.name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for value in self.optimizer.variables
            ]
        return {**payload, "state_hash": _stable_hash(payload)}

    def restore_state(self, payload: Mapping[str, Any]) -> None:
        state = dict(payload)
        supplied_hash = str(state.pop("state_hash", ""))
        if supplied_hash != _stable_hash(state):
            raise NeuTraTrainingError("trainer state_hash mismatch")
        if state.get("schema") != "bayesfilter.neutra.reverse_kl_trainer_state.v1":
            raise NeuTraTrainingError("unsupported trainer state schema")
        if state.get("config") != self.config.manifest_payload():
            raise NeuTraTrainingError("trainer state config mismatch")
        keys = tuple(str(item) for item in state.get("variable_keys", ()))
        if keys != self.transport.variable_keys:
            raise NeuTraTrainingError("trainer variable keys mismatch")
        step = int(state.get("step", -1))
        if step < 0:
            raise NeuTraTrainingError("trainer step must be nonnegative")
        effective_learning_rate = float(
            state.get("effective_learning_rate", self.config.learning_rate)
        )
        if (
            not math.isfinite(effective_learning_rate)
            or effective_learning_rate <= 0.0
            or effective_learning_rate > float(self.config.learning_rate)
        ):
            raise NeuTraTrainingError("trainer effective_learning_rate is invalid")
        validated_variables = _validated_rows(
            self.variables, state.get("variables"), "variables", dtype=tf.float64
        )
        validated_first = _validated_rows(
            self.first_moments, state.get("first_moments"), "first_moments", dtype=tf.float64
        )
        validated_second = _validated_rows(
            self.second_moments, state.get("second_moments"), "second_moments", dtype=tf.float64
        )
        validated_optimizer = None
        if self.optimizer is not None:
            expected_specs = [
                {
                    "name": value.name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
                for value in self.optimizer.variables
            ]
            if state.get("optimizer_variable_specs") != expected_specs:
                raise NeuTraTrainingError("trainer optimizer variable specs mismatch")
            validated_optimizer = _validated_rows(
                self.optimizer.variables,
                state.get("optimizer_variables"),
                "optimizer_variables",
                dtype=None,
            )
            if int(validated_optimizer[0].numpy()) != step:
                raise NeuTraTrainingError("trainer optimizer iteration mismatch")
        for variable, value in zip(self.variables, validated_variables):
            variable.assign(value)
        for variable, value in zip(self.first_moments, validated_first):
            variable.assign(value)
        for variable, value in zip(self.second_moments, validated_second):
            variable.assign(value)
        if self.optimizer is not None and validated_optimizer is not None:
            for variable, value in zip(self.optimizer.variables, validated_optimizer):
                variable.assign(value)
        if self.optimizer is None:
            self._generic_learning_rate.assign(effective_learning_rate)
        elif self.config.family in {
            SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY,
            SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY,
        }:
            self.optimizer.learning_rate.assign(effective_learning_rate)
        self.step.assign(step)

    def frozen_transport_payload(
        self,
        *,
        transport_id: str,
        target_signature: str,
    ) -> Mapping[str, Any]:
        if not transport_id:
            raise ValueError("transport_id must be nonempty")
        if len(target_signature) != 64:
            raise ValueError("target_signature must be a sha256 hex digest")
        if (
            self.config.family in COMPOSED_NEUTRA_FAMILIES
            and target_signature != self.config.target_signature
        ):
            raise NeuTraTrainingError("frozen target_signature does not match trainer target")
        state = self.state_payload()
        if isinstance(self.transport, _TrainableComposedIAF):
            components = self.transport.frozen_components()
        else:
            component_id = f"{self.config.family}_00"
            components = (
                self.transport.frozen_component_payload(component_id=component_id),
            )
        raw = {
            "schema": "bayesfilter.neutra.dense_iaf_frozen_transport.v1",
            "transport_id": transport_id,
            "dimension": int(self.config.dimension),
            "target_signature": target_signature,
            "log_jacobian_available": True,
            "component_order": [component["component_id"] for component in components],
            "components": list(components),
            "training_state_hash": state["state_hash"],
            "nonclaims": list(NEUTRA_TRAINING_NONCLAIMS),
        }
        if self.config.family in COMPOSED_NEUTRA_FAMILIES:
            procedure = {
                DSGE_PAPER_NEUTRA_FAMILY: "dsge_hmc_rotemberg_sgu_plain_neutra_v1",
                SSL_LSTM_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_capacity_32x32_neutra_v1"
                ),
                SSL_LSTM_TUNED_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_tuned_capacity_32x32_neutra_v1"
                ),
                SSL_LSTM_DEEP_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_deep_capacity_32x32x32_neutra_v1"
                ),
                SSL_LSTM_WIDE_CAPACITY_NEUTRA_FAMILY: (
                    "bayesfilter_ssl_lstm_wide_capacity_64x64_neutra_v1"
                ),
            }[self.config.family]
            raw.update(
                {
                    "target_adapter_signature": self.config.target_adapter_signature,
                    "target_chart": self.config.target_chart,
                    "target_parameter_names": list(self.config.target_parameter_names),
                    "fixed_translation": list(self.config.fixed_translation),
                    "procedure": procedure,
                }
            )
        return finalize_dense_iaf_neutra_artifact_payload(raw)

    def _loss_and_gradients_impl(
        self,
        z: tf.Tensor,
    ) -> tuple[NeuTraTrainStep, tuple[tf.Tensor, ...]]:
        theta_for_target, _ = self.transport.forward_and_logdet(z)
        target_value, target_score = _target_value_and_score(
            self.target,
            tf.stop_gradient(theta_for_target),
        )
        target_value = tf.stop_gradient(target_value)
        target_score = tf.stop_gradient(target_score)
        _assert_finite(target_value, "target value")
        _assert_finite(target_score, "target score")
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            theta, logdet = self.transport.forward_and_logdet(z)
            surrogate = tf.reduce_mean(
                -tf.reduce_sum(target_score * theta, axis=-1) - logdet
            )
        gradients = tuple(tape.gradient(surrogate, self.variables))
        if any(gradient is None for gradient in gradients):
            raise NeuTraTrainingError("reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        if self.config.family in COMPOSED_NEUTRA_FAMILIES:
            for index, gradient in enumerate(gradients):
                _assert_finite(gradient, f"gradient[{index}]")
        loss = tf.reduce_mean(-target_value - logdet)
        if self.config.gradient_clip_mode == "per_variable":
            clip_inputs = gradients
            gradient_norm = tf.linalg.global_norm(clip_inputs)
            clipped = tuple(
                tf.clip_by_norm(gradient, self.config.gradient_clip_norm)
                for gradient in clip_inputs
            )
            clipping_applied = tf.reduce_any(
                tf.stack(
                    [
                        tf.linalg.norm(gradient) > self.config.gradient_clip_norm
                        for gradient in clip_inputs
                    ]
                )
            )
        else:
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped_rows, _ = tf.clip_by_global_norm(
                gradients,
                tf.cast(self.config.gradient_clip_norm, tf.float64),
                use_norm=gradient_norm,
            )
            clipped = tuple(clipped_rows)
            clipping_applied = gradient_norm > tf.cast(
                self.config.gradient_clip_norm, gradient_norm.dtype
            )
        clipped_norm = tf.linalg.global_norm(clipped)
        _assert_finite(loss, "reverse-KL loss")
        _assert_finite(surrogate, "reverse-KL surrogate")
        for index, gradient in enumerate(clipped):
            _assert_finite(gradient, f"gradient[{index}]")
        result = NeuTraTrainStep(
            loss=loss,
            surrogate=surrogate,
            target_value_mean=tf.reduce_mean(target_value),
            logdet_mean=tf.reduce_mean(logdet),
            gradient_norm=gradient_norm,
            clipped_gradient_norm=clipped_norm,
            clipping_applied=clipping_applied,
            step=tf.identity(self.step),
        )
        return result, clipped

    def _validation_impl(self, z: tf.Tensor) -> tuple[tf.Tensor, ...]:
        theta, logdet = self.transport.forward_and_logdet(z)
        target_value, _ = _target_value_and_score(self.target, theta)
        scale_log = self.transport.scale_log(z)
        scale_logits, hidden_preactivations = self.transport.diagnostics(z)
        per_sample_loss = -target_value - logdet
        _assert_finite(per_sample_loss, "validation loss")
        _assert_finite(theta, "validation theta")
        _assert_finite(scale_log, "validation scale_log")
        _assert_finite(scale_logits, "validation scale_logits")
        _assert_finite(hidden_preactivations, "validation hidden_preactivations")
        return (
            per_sample_loss,
            target_value,
            theta,
            logdet,
            scale_log,
            scale_logits,
            hidden_preactivations,
        )

    def _train_step_impl(self, z: tf.Tensor) -> tuple[tf.Tensor, ...]:
        result, gradients = self._loss_and_gradients_impl(z)
        if self.optimizer is not None:
            finite_step = tf.reduce_all(
                tf.stack(
                    (
                        tf.reduce_all(tf.math.is_finite(result.loss)),
                        tf.reduce_all(tf.math.is_finite(result.surrogate)),
                        tf.reduce_all(tf.math.is_finite(result.target_value_mean)),
                        tf.reduce_all(tf.math.is_finite(result.logdet_mean)),
                        tf.reduce_all(tf.math.is_finite(result.gradient_norm)),
                        tf.reduce_all(tf.math.is_finite(result.clipped_gradient_norm)),
                        *(
                            tf.reduce_all(tf.math.is_finite(gradient))
                            for gradient in gradients
                        ),
                    )
                )
            )

            def apply_update() -> tf.Tensor:
                self.optimizer.apply_gradients(zip(gradients, self.variables))
                return tf.cast(self.optimizer.iterations, tf.int64)

            next_step = tf.cond(
                finite_step,
                apply_update,
                lambda: tf.identity(self.step),
            )
            self.step.assign(next_step)
            return (
                result.loss,
                result.surrogate,
                result.target_value_mean,
                result.logdet_mean,
                result.gradient_norm,
                result.clipped_gradient_norm,
                result.clipping_applied,
                tf.identity(self.step),
                finite_step,
            )
        next_step = self.step + tf.constant(1, dtype=tf.int64)
        beta1 = tf.cast(self.config.beta1, tf.float64)
        beta2 = tf.cast(self.config.beta2, tf.float64)
        learning_rate = tf.cast(self._generic_learning_rate, tf.float64)
        epsilon = tf.cast(self.config.epsilon, tf.float64)
        step_float = tf.cast(next_step, tf.float64)
        candidate_rows = []
        for variable, gradient, first, second in zip(
            self.variables,
            gradients,
            self.first_moments,
            self.second_moments,
        ):
            next_first = beta1 * first + (1.0 - beta1) * gradient
            next_second = beta2 * second + (1.0 - beta2) * tf.square(gradient)
            first_hat = next_first / (1.0 - tf.pow(beta1, step_float))
            second_hat = next_second / (1.0 - tf.pow(beta2, step_float))
            next_variable = variable - (
                learning_rate * first_hat / (tf.sqrt(second_hat) + epsilon)
            )
            candidate_rows.append((next_variable, next_first, next_second))
        finite_step = tf.reduce_all(
            tf.stack(
                (
                    tf.reduce_all(tf.math.is_finite(result.loss)),
                    tf.reduce_all(tf.math.is_finite(result.surrogate)),
                    tf.reduce_all(tf.math.is_finite(result.target_value_mean)),
                    tf.reduce_all(tf.math.is_finite(result.logdet_mean)),
                    tf.reduce_all(tf.math.is_finite(result.gradient_norm)),
                    tf.reduce_all(tf.math.is_finite(result.clipped_gradient_norm)),
                    *(
                        tf.reduce_all(tf.math.is_finite(value))
                        for row in candidate_rows
                        for value in row
                    ),
                )
            )
        )

        def apply_generic_update() -> tf.Tensor:
            for (variable, first, second), (
                next_variable,
                next_first,
                next_second,
            ) in zip(
                zip(self.variables, self.first_moments, self.second_moments),
                candidate_rows,
            ):
                variable.assign(next_variable)
                first.assign(next_first)
                second.assign(next_second)
            self.step.assign(next_step)
            return tf.identity(self.step)

        applied_step = tf.cond(
            finite_step,
            apply_generic_update,
            lambda: tf.identity(self.step),
        )
        return (
            result.loss,
            result.surrogate,
            result.target_value_mean,
            result.logdet_mean,
            result.gradient_norm,
            result.clipped_gradient_norm,
            result.clipping_applied,
            applied_step,
            finite_step,
        )

    def _external_train_step_impl(
        self,
        z: tf.Tensor,
        target_value: tf.Tensor,
        target_score: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Parent-side optimizer program for detached worker scores."""

        target_value = tf.stop_gradient(tf.convert_to_tensor(target_value, tf.float64))
        target_score = tf.stop_gradient(tf.convert_to_tensor(target_score, tf.float64))
        _assert_finite(target_value, "external target value")
        _assert_finite(target_score, "external target score")

        @tf.custom_gradient
        def target_values_with_worker_score(
            theta_live: tf.Tensor,
        ) -> tuple[tf.Tensor, Any]:
            del theta_live

            def grad(upstream: tf.Tensor) -> tf.Tensor:
                return tf.reshape(
                    tf.cast(upstream, tf.float64), (-1, 1)
                ) * target_score

            return target_value, grad

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            theta, logdet = self.transport.forward_and_logdet(z)
            bridged_target_value = target_values_with_worker_score(theta)
            loss = tf.reduce_mean(-bridged_target_value - logdet)
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(gradient is None for gradient in gradients):
            raise NeuTraTrainingError("external reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        surrogate = tf.reduce_mean(
            -tf.reduce_sum(target_score * tf.stop_gradient(theta), axis=-1)
            - tf.stop_gradient(logdet)
        )
        if self.config.gradient_clip_mode == "per_variable":
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped = tuple(
                tf.clip_by_norm(gradient, self.config.gradient_clip_norm)
                for gradient in gradients
            )
            clipping_applied = tf.reduce_any(tf.stack([
                tf.linalg.norm(gradient) > self.config.gradient_clip_norm
                for gradient in gradients
            ]))
        else:
            gradient_norm = tf.linalg.global_norm(gradients)
            clipped_rows, _ = tf.clip_by_global_norm(
                gradients,
                tf.cast(self.config.gradient_clip_norm, tf.float64),
                use_norm=gradient_norm,
            )
            clipped = tuple(clipped_rows)
            clipping_applied = gradient_norm > tf.cast(
                self.config.gradient_clip_norm, gradient_norm.dtype
            )
        clipped_norm = tf.linalg.global_norm(clipped)
        finite_step = tf.reduce_all(tf.stack((
            tf.reduce_all(tf.math.is_finite(loss)),
            tf.reduce_all(tf.math.is_finite(surrogate)),
            tf.reduce_all(tf.math.is_finite(gradient_norm)),
            tf.reduce_all(tf.math.is_finite(clipped_norm)),
            *(tf.reduce_all(tf.math.is_finite(gradient)) for gradient in clipped),
        )))

        def apply_update() -> tf.Tensor:
            if self.optimizer is not None:
                self.optimizer.apply_gradients(zip(clipped, self.variables))
                return tf.cast(self.optimizer.iterations, tf.int64)
            next_step = self.step + tf.constant(1, dtype=tf.int64)
            beta1 = tf.cast(self.config.beta1, tf.float64)
            beta2 = tf.cast(self.config.beta2, tf.float64)
            learning_rate = tf.cast(self._generic_learning_rate, tf.float64)
            epsilon = tf.cast(self.config.epsilon, tf.float64)
            step_float = tf.cast(next_step, tf.float64)
            for variable, gradient, first, second in zip(
                self.variables, clipped, self.first_moments, self.second_moments
            ):
                first.assign(beta1 * first + (1.0 - beta1) * gradient)
                second.assign(beta2 * second + (1.0 - beta2) * tf.square(gradient))
                variable.assign_sub(
                    learning_rate * (first / (1.0 - tf.pow(beta1, step_float))) /
                    (tf.sqrt(second / (1.0 - tf.pow(beta2, step_float))) + epsilon)
                )
            return next_step

        next_step = tf.cond(finite_step, apply_update, lambda: tf.identity(self.step))
        self.step.assign(next_step)
        return (
            loss,
            surrogate,
            tf.reduce_mean(target_value),
            tf.reduce_mean(logdet),
            gradient_norm,
            clipped_norm,
            clipping_applied,
            tf.identity(self.step),
            finite_step,
        )

    def _external_gradients_impl(
        self,
        z: tf.Tensor,
        target_value: tf.Tensor,
        target_score: tf.Tensor,
        valid_mask: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        """Return un-clipped detached-score gradients for one static chunk."""

        target_value = tf.stop_gradient(tf.convert_to_tensor(target_value, tf.float64))
        target_score = tf.stop_gradient(tf.convert_to_tensor(target_score, tf.float64))
        valid_mask = tf.stop_gradient(tf.convert_to_tensor(valid_mask, tf.float64))
        _assert_finite(target_value, "external chunk target value")
        _assert_finite(target_score, "external chunk target score")
        _assert_finite(valid_mask, "external chunk valid mask")

        @tf.custom_gradient
        def target_values_with_worker_score(theta_live: tf.Tensor) -> tuple[tf.Tensor, Any]:
            del theta_live

            def grad(upstream: tf.Tensor) -> tf.Tensor:
                return tf.reshape(tf.cast(upstream, tf.float64), (-1, 1)) * target_score

            return target_value, grad

        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.variables)
            theta, logdet = self.transport.forward_and_logdet(z)
            bridged_target_value = target_values_with_worker_score(theta)
            loss = tf.reduce_sum(valid_mask * (-bridged_target_value - logdet))
        gradients = tuple(tape.gradient(loss, self.variables))
        if any(gradient is None for gradient in gradients):
            raise NeuTraTrainingError("external chunk reverse-KL gradient is missing")
        gradients = tuple(tf.convert_to_tensor(gradient) for gradient in gradients)
        surrogate = tf.reduce_sum(
            valid_mask
            * (
                -tf.reduce_sum(target_score * tf.stop_gradient(theta), axis=-1)
                - tf.stop_gradient(logdet)
            )
        )
        return (
            loss,
            surrogate,
            tf.reduce_sum(valid_mask * target_value),
            tf.reduce_sum(valid_mask * logdet),
            *gradients,
        )


def _target_value_and_score(target: Any, theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    batch_method = getattr(target, "batch_value_and_score", None)
    if callable(batch_method):
        value, score = batch_method(theta)
    else:
        method = getattr(target, "log_prob_and_grad", None)
        if not callable(method):
            raise TypeError("target must expose batch_value_and_score or log_prob_and_grad")
        value, score = method(theta)
    value = tf.convert_to_tensor(value, dtype=tf.float64)
    score = tf.convert_to_tensor(score, dtype=tf.float64)
    if value.shape != theta.shape[:-1]:
        raise NeuTraTrainingError("target value shape mismatch")
    if score.shape != theta.shape:
        raise NeuTraTrainingError("target score shape mismatch")
    return value, score


def _validate_named_composed_target(target: Any, config: NeuTraTrainerConfig) -> None:
    label = config.family
    dimension = getattr(target, "parameter_dim", None)
    names = getattr(target, "parameter_names", None)
    if int(dimension) != int(config.dimension):
        raise NeuTraTrainingError(f"{label} target dimension mismatch")
    if tuple(str(value) for value in names) != tuple(config.target_parameter_names):
        raise NeuTraTrainingError(f"{label} target parameter names/order mismatch")
    target_signature = getattr(target, "target_signature", None)
    adapter_signature = getattr(target, "adapter_signature", None)
    if not callable(target_signature) or target_signature() != config.target_signature:
        raise NeuTraTrainingError(f"{label} target signature mismatch")
    if not callable(adapter_signature) or adapter_signature() != config.target_adapter_signature:
        raise NeuTraTrainingError(f"{label} target adapter signature mismatch")
    target_config = getattr(target, "config", None)
    signature_payload = getattr(target_config, "signature_payload", None)
    if not callable(signature_payload):
        raise NeuTraTrainingError(f"{label} target chart manifest unavailable")
    manifest = signature_payload()
    transform = manifest.get("parameter_transform", {})
    if transform.get("orientation") != "identity" or transform.get(
        "inverse_orientation"
    ) != "identity":
        raise NeuTraTrainingError(f"{label} target chart is not identity-oriented")
    prior_center = getattr(target_config, "prior_center", None)
    if prior_center is None:
        raise NeuTraTrainingError(f"{label} target fixed translation unavailable")
    actual_center = tuple(
        float(value) for value in tf.reshape(tf.convert_to_tensor(prior_center), (-1,))
    )
    if actual_center != tuple(config.fixed_translation):
        raise NeuTraTrainingError(f"{label} target fixed translation mismatch")


def _rank2(value: Any, *, dimension: int, name: str) -> tf.Tensor:
    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    if tensor.shape.rank != 2:
        raise ValueError(f"{name} must have rank 2")
    if tensor.shape[-1] != int(dimension):
        raise ValueError(f"{name} trailing dimension mismatch")
    return tensor


def _activation(values: tf.Tensor, activation: str) -> tf.Tensor:
    if activation == "elu":
        return tf.nn.elu(values)
    if activation == "tanh":
        return tf.math.tanh(values)
    if activation == "relu":
        return tf.nn.relu(values)
    raise NeuTraTrainingError(f"unsupported activation: {activation}")


def _dense_iaf_masks(dim: int, hidden_layers: tuple[int, ...]) -> tuple[tf.Tensor, ...]:
    degrees: list[list[int]] = [list(range(1, dim + 1))]
    maximum = max(1, dim - 1)
    for width in hidden_layers:
        degrees.append([1 + (index % maximum) for index in range(width)])
    degrees.append(list(range(1, dim + 1)) + list(range(1, dim + 1)))
    masks = []
    for index, (source_degrees, target_degrees) in enumerate(
        zip(degrees[:-1], degrees[1:])
    ):
        output_layer = index == len(degrees) - 2
        masks.append(
            tf.constant(
                [
                    [
                        1.0
                        if ((source < target) if output_layer else (source <= target))
                        else 0.0
                        for target in target_degrees
                    ]
                    for source in source_degrees
                ],
                dtype=tf.float64,
            )
        )
    return tuple(masks)


def _assign_rows(
    variables: Sequence[tf.Variable],
    rows: Any,
    name: str,
) -> None:
    if not isinstance(rows, (tuple, list)) or len(rows) != len(variables):
        raise NeuTraTrainingError(f"trainer {name} length mismatch")
    for index, (variable, row) in enumerate(zip(variables, rows)):
        value = tf.convert_to_tensor(row, dtype=tf.float64)
        if value.shape != variable.shape:
            raise NeuTraTrainingError(f"trainer {name}[{index}] shape mismatch")
        _assert_finite(value, f"trainer {name}[{index}]")
        variable.assign(value)


def _validated_rows(
    variables: Sequence[tf.Variable],
    rows: Any,
    name: str,
    *,
    dtype: tf.dtypes.DType | None,
) -> tuple[tf.Tensor, ...]:
    if not isinstance(rows, (tuple, list)) or len(rows) != len(variables):
        raise NeuTraTrainingError(f"trainer {name} length mismatch")
    validated = []
    for index, (variable, row) in enumerate(zip(variables, rows)):
        value = tf.convert_to_tensor(
            row,
            dtype=variable.dtype if dtype is None else dtype,
        )
        if value.shape != variable.shape:
            raise NeuTraTrainingError(f"trainer {name}[{index}] shape mismatch")
        _assert_finite(tf.cast(value, tf.float64), f"trainer {name}[{index}]")
        validated.append(value)
    return tuple(validated)


def _tensor_values(value: tf.Tensor) -> Any:
    return tf.convert_to_tensor(value, dtype=tf.float64).numpy().tolist()


def _native_tensor_values(value: tf.Tensor) -> Any:
    return tf.convert_to_tensor(value).numpy().tolist()


def _assert_finite(value: tf.Tensor, name: str) -> None:
    tf.debugging.assert_all_finite(value, f"{name} must be finite")


def _stable_hash(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
