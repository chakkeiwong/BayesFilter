"""Graph-native target-specific scalar residual-force training.

The learned object is a scalar residual potential in fixed transport
coordinates.  Its exported force is obtained by differentiation and therefore
remains conservative by construction.  Training loss is nomination evidence;
only corrected downstream HMC can admit a force for sampling.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import tensorflow as tf

from bayesfilter.inference.neural_force_hmc import FrozenPositionOnlyForce


NEURAL_FORCE_TRAINING_SCHEMA = "bayesfilter.neural_force_training.v1"
FROZEN_SCALAR_FORCE_SCHEMA = "bayesfilter.frozen_scalar_residual_force.v1"


class NeuralForceTrainingError(RuntimeError):
    """Raised when scalar-force training or artifact validation fails closed."""


def _sha256_text(payload: Any) -> str:
    blob = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _bare_sha256(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy().tolist())
    return value


def _write_new_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")


@dataclass(frozen=True)
class ScalarResidualForceTrainingConfig:
    """Frozen target-specific training recipe."""

    target_signature: str
    transport_signature: str
    dimension: int
    hidden_layers: tuple[int, ...]
    output_dir: Path
    seed: tuple[int, int]
    steps: int = 500
    batch_size: int = 128
    learning_rate: float = 1.0e-3
    activation: str = "tanh"
    force_loss_weight: float = 1.0
    centered_value_loss_weight: float = 0.1
    clip_norm: float = 10.0
    heartbeat_every: int = 50
    jit_compile: bool = True
    device: str = "/GPU:0"
    require_gpu: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "target_signature", _bare_sha256(self.target_signature, "target_signature")
        )
        object.__setattr__(
            self,
            "transport_signature",
            _bare_sha256(self.transport_signature, "transport_signature"),
        )
        dimension = int(self.dimension)
        layers = tuple(int(value) for value in self.hidden_layers)
        seed = tuple(int(value) for value in self.seed)
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if not layers or any(value <= 0 for value in layers):
            raise ValueError("hidden_layers must contain positive widths")
        if len(seed) != 2:
            raise ValueError("seed must contain two integers")
        if self.activation != "tanh":
            raise ValueError("P2 scalar-force training supports only frozen tanh activation")
        for name in (
            "learning_rate",
            "force_loss_weight",
            "centered_value_loss_weight",
            "clip_norm",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("steps", "batch_size", "heartbeat_every"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if int(self.batch_size) <= 1:
            raise ValueError("batch_size must exceed one")
        if self.jit_compile is not True:
            raise ValueError("scalar-force training requires XLA JIT")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "hidden_layers", layers)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": NEURAL_FORCE_TRAINING_SCHEMA,
            "target_signature": self.target_signature,
            "transport_signature": self.transport_signature,
            "dimension": self.dimension,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "seed": self.seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "force_loss_weight": self.force_loss_weight,
            "centered_value_loss_weight": self.centered_value_loss_weight,
            "clip_norm": self.clip_norm,
            "heartbeat_every": self.heartbeat_every,
            "jit_compile": self.jit_compile,
            "device": self.device,
            "require_gpu": self.require_gpu,
            "objective": "standardized_force_mse_plus_centered_potential_mse",
            "baseline_potential": "one_half_z_squared",
            "promotion_role": "nomination_or_veto_only",
            "output_dir": str(self.output_dir),
        }

    @property
    def config_hash(self) -> str:
        payload = dict(self.payload())
        payload.pop("output_dir")
        return _sha256_text(payload)


class ScalarForceEvaluation(NamedTuple):
    potential: tf.Tensor
    force: tf.Tensor


class ScalarResidualPotentialNetwork:
    """Small dense scalar residual potential with explicit TensorFlow tensors."""

    def __init__(
        self,
        *,
        dimension: int,
        hidden_layers: Sequence[int],
        position_mean: tf.Tensor,
        position_scale: tf.Tensor,
        seed: tuple[int, int],
        tensors: Sequence[Sequence[tf.Tensor]] | None = None,
        trainable: bool = True,
    ) -> None:
        self.dimension = int(dimension)
        self.hidden_layers = tuple(int(value) for value in hidden_layers)
        self.position_mean = tf.convert_to_tensor(position_mean, tf.float64)
        self.position_scale = tf.convert_to_tensor(position_scale, tf.float64)
        sizes = (self.dimension, *self.hidden_layers, 1)
        weights: list[tf.Tensor] = []
        biases: list[tf.Tensor] = []
        if tensors is not None and len(tensors) != 2:
            raise ValueError("tensors must contain weights and biases")
        supplied_weights = () if tensors is None else tuple(tensors[0])
        supplied_biases = () if tensors is None else tuple(tensors[1])
        for index, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
            if tensors is None:
                fan_scale = math.sqrt(2.0 / float(input_size + output_size))
                value = tf.random.stateless_normal(
                    [input_size, output_size],
                    seed=(seed[0], seed[1] + index),
                    dtype=tf.float64,
                ) * tf.constant(fan_scale, tf.float64)
                bias = tf.zeros([output_size], tf.float64)
            else:
                value = tf.convert_to_tensor(supplied_weights[index], tf.float64)
                bias = tf.convert_to_tensor(supplied_biases[index], tf.float64)
            if value.shape != (input_size, output_size) or bias.shape != (output_size,):
                raise ValueError("scalar residual network tensor shape mismatch")
            if trainable:
                weights.append(tf.Variable(value, name=f"residual_W{index}"))
                biases.append(tf.Variable(bias, name=f"residual_b{index}"))
            else:
                weights.append(tf.constant(value))
                biases.append(tf.constant(bias))
        self.weights = tuple(weights)
        self.biases = tuple(biases)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(
            value
            for pair in zip(self.weights, self.biases)
            for value in pair
            if isinstance(value, tf.Variable)
        )

    def residual_potential(self, position: tf.Tensor) -> tf.Tensor:
        value = (
            tf.convert_to_tensor(position, tf.float64) - self.position_mean
        ) / self.position_scale
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            value = tf.math.tanh(tf.matmul(value, weight) + bias)
        value = tf.matmul(value, self.weights[-1]) + self.biases[-1]
        return tf.squeeze(value, axis=-1)

    def potential_and_force(self, position: tf.Tensor) -> ScalarForceEvaluation:
        value = tf.convert_to_tensor(position, tf.float64)
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(value)
            residual = self.residual_potential(value)
            potential = 0.5 * tf.reduce_sum(tf.square(value), axis=-1) + residual
        force = tape.gradient(
            potential, value, output_gradients=tf.ones_like(potential)
        )
        if force is None:
            raise NeuralForceTrainingError("scalar potential input gradient is missing")
        return ScalarForceEvaluation(potential=potential, force=force)


@dataclass(frozen=True)
class FrozenScalarResidualForce:
    """Immutable loaded scalar potential and its conservative force."""

    network: ScalarResidualPotentialNetwork = field(repr=False, compare=False)
    target_signature: str
    transport_signature: str
    artifact_signature: str
    config_hash: str

    def potential(self, position: tf.Tensor) -> tf.Tensor:
        return self.network.potential_and_force(position).potential

    def force(self, position: tf.Tensor) -> tf.Tensor:
        return self.network.potential_and_force(position).force

    def hmc_force(self) -> FrozenPositionOnlyForce:
        return FrozenPositionOnlyForce(
            function=self.force,
            identity=self.artifact_signature,
        )


@dataclass(frozen=True)
class ScalarResidualForceTrainingResult:
    frozen: FrozenScalarResidualForce
    config_path: Path
    artifact_path: Path
    result_path: Path
    metrics: Mapping[str, Any]
    runtime_metadata: Mapping[str, Any]


def train_scalar_residual_force(
    *,
    train_positions: Any,
    train_potentials: Any,
    train_forces: Any,
    heldout_positions: Any,
    heldout_potentials: Any,
    heldout_forces: Any,
    config: ScalarResidualForceTrainingConfig,
) -> ScalarResidualForceTrainingResult:
    """Train once in one compiled graph, freeze, reload, and evaluate heldout data."""

    output_dir = config.output_dir
    if output_dir.exists():
        raise FileExistsError(f"training output directory must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    train_x, train_u, train_g = _validated_training_arrays(
        train_positions,
        train_potentials,
        train_forces,
        dimension=config.dimension,
        label="training",
    )
    heldout_x, heldout_u, heldout_g = _validated_training_arrays(
        heldout_positions,
        heldout_potentials,
        heldout_forces,
        dimension=config.dimension,
        label="heldout",
    )
    if int(train_x.shape[0]) < config.batch_size:
        raise ValueError("training row count must be at least batch_size")
    position_mean = tf.reduce_mean(train_x, axis=0)
    position_scale = tf.math.reduce_std(train_x, axis=0)
    position_scale = tf.maximum(position_scale, tf.constant(1.0e-6, tf.float64))
    force_scale = tf.math.reduce_std(train_g, axis=0)
    force_scale = tf.maximum(force_scale, tf.constant(1.0e-6, tf.float64))
    potential_scale = tf.math.reduce_std(train_u)
    potential_scale = tf.maximum(potential_scale, tf.constant(1.0e-6, tf.float64))

    with tf.device(config.device):
        network = ScalarResidualPotentialNetwork(
            dimension=config.dimension,
            hidden_layers=config.hidden_layers,
            position_mean=position_mean,
            position_scale=position_scale,
            seed=config.seed,
        )
        variables = network.trainable_variables
        first_moments = tuple(tf.Variable(tf.zeros_like(value), trainable=False) for value in variables)
        second_moments = tuple(tf.Variable(tf.zeros_like(value), trainable=False) for value in variables)
    variable_devices = tuple(value.device for value in (*variables, *first_moments, *second_moments))
    if config.require_gpu and not all("GPU" in value.upper() for value in variable_devices):
        raise NeuralForceTrainingError("training variables and optimizer state must be on GPU")

    def one_step(step_index: tf.Tensor) -> tuple[tf.Tensor, ...]:
        sample_seed = tf.stack(
            [tf.cast(config.seed[0], tf.int32), tf.cast(config.seed[1], tf.int32) + step_index + 1000]
        )
        indices = tf.random.stateless_uniform(
            [config.batch_size],
            sample_seed,
            minval=0,
            maxval=tf.shape(train_x)[0],
            dtype=tf.int32,
        )
        x = tf.gather(train_x, indices)
        target_u = tf.gather(train_u, indices)
        target_g = tf.gather(train_g, indices)
        with tf.GradientTape() as parameter_tape:
            prediction = network.potential_and_force(x)
            standardized_force_error = (prediction.force - target_g) / force_scale
            force_loss = tf.reduce_mean(tf.square(standardized_force_error))
            value_error = (prediction.potential - target_u) / potential_scale
            centered_value_error = value_error - tf.reduce_mean(value_error)
            value_loss = tf.reduce_mean(tf.square(centered_value_error))
            loss = (
                tf.cast(config.force_loss_weight, tf.float64) * force_loss
                + tf.cast(config.centered_value_loss_weight, tf.float64) * value_loss
            )
        gradients = parameter_tape.gradient(loss, variables)
        if any(value is None for value in gradients):
            raise NeuralForceTrainingError("force-loss parameter gradient is missing")
        raw_norm = tf.linalg.global_norm(gradients)
        gradients, _ = tf.clip_by_global_norm(gradients, config.clip_norm)
        step = tf.cast(step_index + 1, tf.float64)
        beta1 = tf.constant(0.9, tf.float64)
        beta2 = tf.constant(0.999, tf.float64)
        learning_rate = tf.constant(config.learning_rate, tf.float64)
        for variable, gradient, first, second in zip(
            variables, gradients, first_moments, second_moments
        ):
            first.assign(beta1 * first + (1.0 - beta1) * gradient)
            second.assign(beta2 * second + (1.0 - beta2) * tf.square(gradient))
            first_hat = first / (1.0 - tf.pow(beta1, step))
            second_hat = second / (1.0 - tf.pow(beta2, step))
            variable.assign_sub(
                learning_rate * first_hat / (tf.sqrt(second_hat) + tf.constant(1.0e-8, tf.float64))
            )
        return loss, force_loss, value_loss, raw_norm

    def training_program() -> tuple[tf.Tensor, ...]:
        losses = tf.TensorArray(tf.float64, size=config.steps, clear_after_read=False)
        force_losses = tf.TensorArray(tf.float64, size=config.steps, clear_after_read=False)
        value_losses = tf.TensorArray(tf.float64, size=config.steps, clear_after_read=False)
        gradient_norms = tf.TensorArray(tf.float64, size=config.steps, clear_after_read=False)

        def cond(index: tf.Tensor, *_: Any) -> tf.Tensor:
            return index < config.steps

        def body(
            index: tf.Tensor,
            loss_array: tf.TensorArray,
            force_array: tf.TensorArray,
            value_array: tf.TensorArray,
            norm_array: tf.TensorArray,
        ) -> tuple[Any, ...]:
            loss, force_loss, value_loss, norm = one_step(index)
            return (
                index + 1,
                loss_array.write(index, loss),
                force_array.write(index, force_loss),
                value_array.write(index, value_loss),
                norm_array.write(index, norm),
            )

        result = tf.while_loop(
            cond,
            body,
            (
                tf.constant(0, tf.int32),
                losses,
                force_losses,
                value_losses,
                gradient_norms,
            ),
            parallel_iterations=1,
        )
        step_numbers = tf.range(1, config.steps + 1, dtype=tf.int32)
        record_mask = tf.logical_or(
            tf.equal(step_numbers, 1),
            tf.logical_or(
                tf.equal(tf.math.floormod(step_numbers, config.heartbeat_every), 0),
                tf.equal(step_numbers, config.steps),
            ),
        )
        return (
            tf.boolean_mask(step_numbers, record_mask),
            *(tf.boolean_mask(array.stack(), record_mask) for array in result[1:]),
        )

    compiled_program = tf.function(training_program, jit_compile=True, reduce_retracing=True)
    concrete = compiled_program.get_concrete_function()
    operations = tuple(sorted({operation.type for operation in concrete.graph.get_operations()}))
    if not any("While" in operation for operation in operations):
        raise NeuralForceTrainingError("compiled training program lacks TensorFlow control flow")
    started = time.monotonic()
    with tf.device(config.device):
        records = compiled_program()
    # Materialize a scalar depending on every returned tensor before stopping
    # the GPU timer; otherwise asynchronous execution can understate fit time.
    tf.add_n(
        [tf.reduce_sum(tf.cast(value, tf.float64)) for value in records]
    ).numpy()
    elapsed = time.monotonic() - started
    if not all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in records[1:]):
        raise NeuralForceTrainingError("training diagnostics must be finite")

    payload = _frozen_payload(
        network=network,
        config=config,
        position_mean=position_mean,
        position_scale=position_scale,
    )
    artifact_path = output_dir / "frozen_force.json"
    config_path = output_dir / "training_config.json"
    _write_new_json(config_path, {**config.payload(), "config_hash": config.config_hash})
    _write_new_json(artifact_path, payload)
    frozen = load_frozen_scalar_residual_force(
        payload,
        expected_target_signature=config.target_signature,
        expected_transport_signature=config.transport_signature,
    )
    heldout_metrics = evaluate_scalar_force(
        frozen=frozen,
        positions=heldout_x,
        target_potentials=heldout_u,
        target_forces=heldout_g,
        force_scale=force_scale,
        potential_scale=potential_scale,
    )
    train_metrics = evaluate_scalar_force(
        frozen=frozen,
        positions=train_x,
        target_potentials=train_u,
        target_forces=train_g,
        force_scale=force_scale,
        potential_scale=potential_scale,
    )
    result_payload = {
        "schema": "bayesfilter.scalar_residual_force_training_result.v1",
        "passed": True,
        "decision": "NOMINATE_FROZEN_SCALAR_FORCE_FOR_DOWNSTREAM_SCREEN",
        "target_signature": config.target_signature,
        "transport_signature": config.transport_signature,
        "artifact_signature": frozen.artifact_signature,
        "config_hash": config.config_hash,
        "train_metrics": train_metrics,
        "heldout_metrics": heldout_metrics,
        "training_records": {
            "step": records[0],
            "loss": records[1],
            "force_loss": records[2],
            "centered_value_loss": records[3],
            "raw_gradient_norm": records[4],
        },
        "runtime_metadata": {
            "elapsed_seconds": elapsed,
            "jit_compile": True,
            "training_control_flow": "tf_while_loop",
            "graph_operation_types": operations,
            "variable_devices": variable_devices,
            "sample_axis_python_loop_used": False,
            "numpy_active_path_used": False,
        },
        "nonclaims": (
            "training and heldout diagnostics are nomination-only",
            "no HMC convergence or posterior validity claim",
            "no performance or superiority claim",
        ),
    }
    result_path = output_dir / "result.json"
    _write_new_json(result_path, result_payload)
    return ScalarResidualForceTrainingResult(
        frozen=frozen,
        config_path=config_path,
        artifact_path=artifact_path,
        result_path=result_path,
        metrics={"train": train_metrics, "heldout": heldout_metrics},
        runtime_metadata=result_payload["runtime_metadata"],
    )


def evaluate_scalar_force(
    *,
    frozen: FrozenScalarResidualForce,
    positions: Any,
    target_potentials: Any,
    target_forces: Any,
    force_scale: Any,
    potential_scale: Any,
) -> Mapping[str, Any]:
    x = tf.convert_to_tensor(positions, tf.float64)
    target_u = tf.convert_to_tensor(target_potentials, tf.float64)
    target_g = tf.convert_to_tensor(target_forces, tf.float64)
    prediction = frozen.network.potential_and_force(x)
    force_error = (prediction.force - target_g) / tf.convert_to_tensor(force_scale, tf.float64)
    value_error = (prediction.potential - target_u) / tf.convert_to_tensor(potential_scale, tf.float64)
    centered_value_error = value_error - tf.reduce_mean(value_error)
    force_norm = tf.linalg.norm(target_g, axis=-1)
    prediction_norm = tf.linalg.norm(prediction.force, axis=-1)
    cosine = tf.reduce_sum(prediction.force * target_g, axis=-1) / tf.maximum(
        force_norm * prediction_norm, tf.constant(1.0e-12, tf.float64)
    )
    all_finite = tf.reduce_all(
        tf.math.is_finite(
            tf.concat(
                [
                    tf.reshape(prediction.potential, [-1]),
                    tf.reshape(prediction.force, [-1]),
                ],
                axis=0,
            )
        )
    )
    return {
        "row_count": int(x.shape[0]),
        "standardized_force_rmse": float(tf.sqrt(tf.reduce_mean(tf.square(force_error))).numpy()),
        "centered_standardized_potential_rmse": float(
            tf.sqrt(tf.reduce_mean(tf.square(centered_value_error))).numpy()
        ),
        "mean_force_cosine": float(tf.reduce_mean(cosine).numpy()),
        "predictions_all_finite": bool(all_finite.numpy()),
        "role": "nomination_or_veto_only",
    }


def load_frozen_scalar_residual_force(
    payload: Mapping[str, Any],
    *,
    expected_target_signature: str,
    expected_transport_signature: str,
) -> FrozenScalarResidualForce:
    """Load and verify a frozen scalar-force artifact against target and chart."""

    if payload.get("schema") != FROZEN_SCALAR_FORCE_SCHEMA:
        raise NeuralForceTrainingError("frozen force schema mismatch")
    expected_target = _bare_sha256(expected_target_signature, "expected_target_signature")
    expected_transport = _bare_sha256(
        expected_transport_signature, "expected_transport_signature"
    )
    if payload.get("target_signature") != expected_target:
        raise NeuralForceTrainingError("frozen force target signature mismatch")
    if payload.get("transport_signature") != expected_transport:
        raise NeuralForceTrainingError("frozen force transport signature mismatch")
    supplied = str(payload.get("artifact_signature", ""))
    unsigned = dict(payload)
    unsigned.pop("artifact_signature", None)
    actual = _sha256_text(unsigned)
    if supplied != actual:
        raise NeuralForceTrainingError("frozen force artifact signature mismatch")
    network_payload = payload.get("network")
    if not isinstance(network_payload, Mapping):
        raise NeuralForceTrainingError("frozen force network payload is missing")
    network = ScalarResidualPotentialNetwork(
        dimension=int(payload["dimension"]),
        hidden_layers=tuple(int(value) for value in network_payload["hidden_layers"]),
        position_mean=tf.constant(network_payload["position_mean"], tf.float64),
        position_scale=tf.constant(network_payload["position_scale"], tf.float64),
        seed=(0, 0),
        tensors=(network_payload["weights"], network_payload["biases"]),
        trainable=False,
    )
    return FrozenScalarResidualForce(
        network=network,
        target_signature=expected_target,
        transport_signature=expected_transport,
        artifact_signature=actual,
        config_hash=str(payload["config_hash"]),
    )


def _validated_training_arrays(
    positions: Any,
    potentials: Any,
    forces: Any,
    *,
    dimension: int,
    label: str,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x = tf.convert_to_tensor(positions, tf.float64)
    u = tf.convert_to_tensor(potentials, tf.float64)
    g = tf.convert_to_tensor(forces, tf.float64)
    if x.shape.rank != 2 or x.shape[1] != dimension:
        raise ValueError(f"{label} positions must have shape [row, {dimension}]")
    if u.shape != (x.shape[0],) or g.shape != x.shape:
        raise ValueError(f"{label} potential/force shapes must match positions")
    if x.shape[0] is None or x.shape[0] < 2:
        raise ValueError(f"{label} needs at least two static rows")
    for value, name in ((x, "positions"), (u, "potentials"), (g, "forces")):
        if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
            raise ValueError(f"{label} {name} must be finite")
    return x, u, g


def _frozen_payload(
    *,
    network: ScalarResidualPotentialNetwork,
    config: ScalarResidualForceTrainingConfig,
    position_mean: tf.Tensor,
    position_scale: tf.Tensor,
) -> Mapping[str, Any]:
    payload = {
        "schema": FROZEN_SCALAR_FORCE_SCHEMA,
        "target_signature": config.target_signature,
        "transport_signature": config.transport_signature,
        "dimension": config.dimension,
        "config_hash": config.config_hash,
        "force_semantics": "gradient_of_scalar_baseline_plus_residual_potential",
        "baseline_potential": "one_half_z_squared",
        "network": {
            "hidden_layers": config.hidden_layers,
            "activation": config.activation,
            "position_mean": position_mean,
            "position_scale": position_scale,
            "weights": tuple(value for value in network.weights),
            "biases": tuple(value for value in network.biases),
            "dtype": "float64",
        },
        "frozen": True,
        "nonclaims": (
            "force artifact only",
            "no downstream HMC admission",
            "no cross-target reuse",
        ),
    }
    payload = _json_ready(payload)
    payload["artifact_signature"] = _sha256_text(payload)
    return payload


__all__ = [
    "FROZEN_SCALAR_FORCE_SCHEMA",
    "FrozenScalarResidualForce",
    "NEURAL_FORCE_TRAINING_SCHEMA",
    "NeuralForceTrainingError",
    "ScalarForceEvaluation",
    "ScalarResidualForceTrainingConfig",
    "ScalarResidualForceTrainingResult",
    "ScalarResidualPotentialNetwork",
    "evaluate_scalar_force",
    "load_frozen_scalar_residual_force",
    "train_scalar_residual_force",
]
