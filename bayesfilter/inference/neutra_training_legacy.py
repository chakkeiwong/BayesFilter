"""Focused TensorFlow training utilities for plain dense-IAF NeuTra."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import tensorflow as tf

from bayesfilter.inference.neutra_batching import (
    InvalidNeuTraBatchTarget,
    batch_native_value_status_target_fn,
    require_batch_native_neutra_target,
)
from bayesfilter.inference.neutra_artifacts import (
    finalize_dense_iaf_neutra_artifact_payload,
)


def _stable_config_hash(config: Any) -> str:
    normalized = _json_ready(config)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _atomic_write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise NeuTraTrainingError(f"temporary artifact already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(_json_ready(payload), handle, sort_keys=True, indent=2)
            handle.write("\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _append_jsonl(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":")))
        handle.write("\n")


DENSE_IAF_SCHEMA = "bayesfilter.neutra.dense_iaf_frozen_transport.v1"
TRAINING_CONFIG_SCHEMA = "bayesfilter.neutra.plain_dense_iaf_training_config.v1"
TRAINING_STATE_SCHEMA = "bayesfilter.neutra.plain_dense_iaf_training_state.v1"

NEUTRA_TRAINING_NONCLAIMS = (
    "plain reverse-KL NeuTra transport training only",
    "training loss is explanatory only",
    "no HMC tuning or convergence claim",
    "no posterior correctness or sampler superiority claim",
    "no production or default readiness claim",
)


class NeuTraTrainingError(RuntimeError):
    """Raised when training configuration, state, or numerics fail closed."""


@dataclass(frozen=True)
class PlainDenseIAFTrainingConfig:
    """Deterministic plain dense-IAF reverse-KL training configuration."""

    target_signature: str
    dimension: int
    affine_center: Sequence[float]
    affine_factor: Sequence[Sequence[float]]
    output_dir: Path
    seed: tuple[int, int] = (20260713, 1201)
    hidden_layers: tuple[int, ...] = (18, 18)
    stage_count: int = 3
    activation: str = "elu"
    s_max: float = 1.0
    init_scale: float = 0.02
    steps: int = 1000
    batch_size: int = 256
    learning_rate: float = 1.0e-3
    final_learning_rate_fraction: float = 0.1
    clip_norm: float = 10.0
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1.0e-8
    checkpoint_every: int = 50
    heartbeat_every: int = 10
    jit_compile: bool = True
    device: str = "/GPU:0"
    require_gpu: bool = True

    def __post_init__(self) -> None:
        signature = _bare_sha256(self.target_signature, "target_signature")
        dimension = int(self.dimension)
        center = tf.convert_to_tensor(self.affine_center, dtype=tf.float64)
        factor = tf.convert_to_tensor(self.affine_factor, dtype=tf.float64)
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if center.shape != (dimension,):
            raise ValueError("affine_center shape mismatch")
        if factor.shape != (dimension, dimension):
            raise ValueError("affine_factor shape mismatch")
        if not bool(tf.reduce_all(tf.math.is_finite(center)).numpy()):
            raise ValueError("affine_center must be finite")
        if not bool(tf.reduce_all(tf.math.is_finite(factor)).numpy()):
            raise ValueError("affine_factor must be finite")
        if abs(float(tf.linalg.det(factor).numpy())) == 0.0:
            raise ValueError("affine_factor must be nonsingular")
        hidden = tuple(int(item) for item in self.hidden_layers)
        if not hidden or any(item <= 0 for item in hidden):
            raise ValueError("hidden_layers must contain positive integers")
        if int(self.stage_count) <= 0:
            raise ValueError("stage_count must be positive")
        if self.activation not in {"elu", "tanh", "relu"}:
            raise ValueError("unsupported activation")
        if float(self.s_max) != 1.0:
            raise ValueError("frozen legacy-degree schema training requires s_max == 1")
        for name in (
            "init_scale",
            "learning_rate",
            "final_learning_rate_fraction",
            "clip_norm",
            "beta1",
            "beta2",
            "epsilon",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < float(self.beta1) < 1.0 or not 0.0 < float(self.beta2) < 1.0:
            raise ValueError("Adam beta values must be in (0, 1)")
        if int(self.steps) <= 0:
            raise ValueError("steps must be positive")
        if int(self.batch_size) <= 1:
            raise ValueError("NeuTra training batch_size must be greater than one")
        if int(self.checkpoint_every) <= 0 or int(self.heartbeat_every) <= 0:
            raise ValueError("checkpoint and heartbeat intervals must be positive")
        seed = tuple(int(item) for item in self.seed)
        if len(seed) != 2:
            raise ValueError("seed must contain two integers")
        if bool(self.jit_compile) is not True:
            raise ValueError("plain NeuTra training requires jit_compile=True")
        object.__setattr__(self, "target_signature", signature)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "affine_center", tuple(float(x) for x in center.numpy()))
        object.__setattr__(
            self,
            "affine_factor",
            tuple(tuple(float(x) for x in row) for row in factor.numpy()),
        )
        object.__setattr__(self, "hidden_layers", hidden)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": TRAINING_CONFIG_SCHEMA,
            "target_signature": self.target_signature,
            "dimension": self.dimension,
            "affine_center": self.affine_center,
            "affine_factor": self.affine_factor,
            "composition": "T_phi(z)=affine(dense_iaf_stack(z))",
            "affine_trainable": False,
            "seed": self.seed,
            "hidden_layers": self.hidden_layers,
            "stage_count": self.stage_count,
            "activation": self.activation,
            "s_max": self.s_max,
            "init_scale": self.init_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "final_learning_rate_fraction": self.final_learning_rate_fraction,
            "learning_rate_schedule": "linear_decay",
            "clip_norm": self.clip_norm,
            "optimizer": "manual_adam",
            "beta1": self.beta1,
            "beta2": self.beta2,
            "epsilon": self.epsilon,
            "checkpoint_every": self.checkpoint_every,
            "heartbeat_every": self.heartbeat_every,
            "jit_compile": self.jit_compile,
            "device": self.device,
            "require_gpu": self.require_gpu,
            "output_dir": str(self.output_dir),
            "objective": "exact_target_reverse_kl",
            "nonclaims": NEUTRA_TRAINING_NONCLAIMS,
        }

    @property
    def config_hash(self) -> str:
        payload = dict(self.payload())
        payload.pop("output_dir", None)
        return _stable_config_hash(payload)


class TrainableDenseAutoregressiveIAF:
    """MADE-style trainable IAF using the frozen schema's exact convention."""

    def __init__(
        self,
        dimension: int,
        *,
        hidden_layers: Sequence[int],
        activation: str,
        s_max: float,
        seed: tuple[int, int],
        init_scale: float,
        name: str,
    ) -> None:
        self.dimension = int(dimension)
        self.hidden_layers = tuple(int(item) for item in hidden_layers)
        self.activation = str(activation)
        self.s_max = float(s_max)
        self.name = str(name)
        sizes = (self.dimension, *self.hidden_layers, 2 * self.dimension)
        self.masks = _dense_iaf_masks(self.dimension, self.hidden_layers)
        self.weights: list[tf.Variable] = []
        self.biases: list[tf.Variable] = []
        for index, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            output_layer = index == len(sizes) - 2
            scale = 0.0 if output_layer else float(init_scale)
            values = tf.random.stateless_normal(
                (n_in, n_out),
                seed=tf.constant((seed[0], seed[1] + index), dtype=tf.int32),
                dtype=tf.float64,
            ) * tf.constant(scale, tf.float64)
            self.weights.append(tf.Variable(values, name=f"{self.name}_W{index}"))
            self.biases.append(
                tf.Variable(tf.zeros((n_out,), tf.float64), name=f"{self.name}_b{index}")
            )

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        result: list[tf.Variable] = []
        for weight, bias in zip(self.weights, self.biases):
            result.extend((weight, bias))
        return tuple(result)

    def forward_and_logdet(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        z = tf.convert_to_tensor(values, dtype=tf.float64)
        h = z
        for weight, bias, mask in zip(
            self.weights[:-1], self.biases[:-1], self.masks[:-1]
        ):
            h = tf.matmul(h, weight * mask) + bias
            h = _activation(h, self.activation)
        raw = tf.matmul(h, self.weights[-1] * self.masks[-1]) + self.biases[-1]
        scale_logits, shift = tf.split(raw, 2, axis=-1)
        scale_log = self.s_max * tf.math.tanh(scale_logits / self.s_max)
        return z * tf.exp(scale_log) + shift, tf.reduce_sum(scale_log, axis=-1)

    def component_payload(self, component_id: str) -> Mapping[str, Any]:
        return {
            "component_id": component_id,
            "kind": "dense_autoregressive_iaf",
            "dim": self.dimension,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "s_max": self.s_max,
            "masks_policy": "legacy_degree_masks_v1",
            "dtype": "float64",
            "weights": tuple(weight.numpy().tolist() for weight in self.weights),
            "biases": tuple(bias.numpy().tolist() for bias in self.biases),
        }


class PlainDenseIAFTransport:
    """Trainable residual IAF stack followed by a fixed full-affine map."""

    def __init__(self, config: PlainDenseIAFTrainingConfig) -> None:
        self.config = config
        self.dimension = config.dimension
        self.layers = tuple(
            TrainableDenseAutoregressiveIAF(
                config.dimension,
                hidden_layers=config.hidden_layers,
                activation=config.activation,
                s_max=config.s_max,
                seed=(config.seed[0], config.seed[1] + 101 + stage * 100),
                init_scale=config.init_scale,
                name=f"dense_iaf_{stage}",
            )
            for stage in range(config.stage_count)
        )
        self.reverse_matrix = tf.reverse(tf.eye(config.dimension, dtype=tf.float64), axis=(0,))
        self.affine_center = tf.constant(config.affine_center, dtype=tf.float64)
        self.affine_factor = tf.constant(config.affine_factor, dtype=tf.float64)
        sign, logdet = tf.linalg.slogdet(self.affine_factor)
        if bool(tf.equal(sign, 0.0).numpy()):
            raise ValueError("affine factor must be nonsingular")
        self.affine_logdet = tf.convert_to_tensor(logdet, dtype=tf.float64)

    @property
    def trainable_variables(self) -> tuple[tf.Variable, ...]:
        return tuple(variable for layer in self.layers for variable in layer.trainable_variables)

    def residual_forward_and_logdet(
        self, values: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor]:
        output = tf.convert_to_tensor(values, dtype=tf.float64)
        logdet = tf.zeros(tf.shape(output)[:-1], dtype=tf.float64)
        for index, layer in enumerate(self.layers):
            output, layer_logdet = layer.forward_and_logdet(output)
            logdet = logdet + layer_logdet
            if index + 1 < len(self.layers):
                output = tf.matmul(output, self.reverse_matrix)
        return output, logdet

    def forward_and_logdet(self, values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        residual, logdet = self.residual_forward_and_logdet(values)
        output = self.affine_center + tf.matmul(
            residual, self.affine_factor, transpose_b=True
        )
        return output, logdet + self.affine_logdet

    def forward_batch(self, values: Any) -> tf.Tensor:
        output, _ = self.forward_and_logdet(tf.convert_to_tensor(values, tf.float64))
        return output

    def log_abs_det_jacobian_batch(self, values: Any) -> tf.Tensor:
        _, logdet = self.forward_and_logdet(tf.convert_to_tensor(values, tf.float64))
        return logdet

    def frozen_payload(
        self,
        *,
        transport_id: str,
        target_signature: str,
        training_state_hash: str,
    ) -> Mapping[str, Any]:
        components: list[Mapping[str, Any]] = []
        order: list[str] = []
        for index, layer in enumerate(self.layers):
            layer_id = f"dense_iaf_{index:02d}"
            components.append(layer.component_payload(layer_id))
            order.append(layer_id)
            if index + 1 < len(self.layers):
                mix_id = f"reverse_permutation_{index:02d}"
                components.append(
                    {
                        "component_id": mix_id,
                        "kind": "mixing_linear",
                        "dim": self.dimension,
                        "dtype": "float64",
                        "matrix": self.reverse_matrix.numpy().tolist(),
                    }
                )
                order.append(mix_id)
        affine_id = "fixed_full_affine"
        components.append(
            {
                "component_id": affine_id,
                "kind": "affine",
                "dim": self.dimension,
                "dtype": "float64",
                "offset": self.affine_center.numpy().tolist(),
                "L_np": self.affine_factor.numpy().tolist(),
            }
        )
        order.append(affine_id)
        return finalize_dense_iaf_neutra_artifact_payload(
            {
                "schema": DENSE_IAF_SCHEMA,
                "transport_id": str(transport_id),
                "dimension": self.dimension,
                "target_signature": _bare_sha256(target_signature, "target_signature"),
                "log_jacobian_available": True,
                "component_order": tuple(order),
                "components": tuple(components),
                "training_state_hash": f"sha256:{_bare_sha256(training_state_hash, 'training_state_hash')}",
                "nonclaims": NEUTRA_TRAINING_NONCLAIMS,
            }
        )


@dataclass(frozen=True)
class PlainDenseIAFTrainingResult:
    completed_steps: int
    state_path: Path
    latest_path: Path
    progress_path: Path
    frozen_payload_path: Path | None
    state_hash: str
    records: tuple[Mapping[str, Any], ...]
    resumed: bool
    runtime_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class PlainDenseIAFSegmentedTrainingResult:
    """Terminal result and immutable lineage for fresh-directory segments."""

    final_result: PlainDenseIAFTrainingResult
    segment_rows: tuple[Mapping[str, Any], ...]
    progress_path: Path
    result_path: Path


def train_plain_dense_iaf_infrastructure_segments(
    *,
    adapter: Any,
    config: PlainDenseIAFTrainingConfig,
    segment_steps: int,
    freeze_transport_id: str,
) -> PlainDenseIAFSegmentedTrainingResult:
    """Run one unchanged training configuration through recoverable segments.

    Every numeric segment still executes as one batched XLA program inside
    :func:`train_plain_dense_iaf`. Only infrastructure boundaries are iterated
    in Python. The global step, stateless-noise sequence, learning-rate
    schedule, trainable state, and Adam moments are inherited exactly.
    """

    segment_size = int(segment_steps)
    if segment_size <= 0:
        raise ValueError("segment_steps must be positive")
    if not freeze_transport_id:
        raise ValueError("freeze_transport_id must be nonempty")
    output_root = Path(config.output_dir)
    if output_root.exists():
        raise NeuTraTrainingError(
            f"segmented training output must be fresh: {output_root}"
        )
    output_root.mkdir(parents=True)
    progress_path = output_root / "segmented_training_progress.json"
    result_path = output_root / "segmented_training_result.json"
    segment_rows: list[Mapping[str, Any]] = []
    parent_state_path: Path | None = None
    final_result: PlainDenseIAFTrainingResult | None = None

    for start_step in range(0, config.steps, segment_size):
        terminal_step = min(start_step + segment_size, config.steps)
        segment_id = f"steps-{start_step + 1:06d}-{terminal_step:06d}"
        segment_dir = output_root / segment_id
        segment_config = replace(config, output_dir=segment_dir)
        terminal = terminal_step == config.steps
        trained = train_plain_dense_iaf(
            adapter=adapter,
            config=segment_config,
            resume_infrastructure_from=parent_state_path,
            stop_after_steps=terminal_step,
            freeze_transport_id=freeze_transport_id if terminal else None,
        )
        if trained.completed_steps != terminal_step:
            raise NeuTraTrainingError(
                f"segment {segment_id} stopped at {trained.completed_steps}"
            )
        state = _read_training_state(trained.state_path, config=segment_config)
        lineage = state.get("repair_lineage")
        if start_step == 0:
            if lineage is not None or trained.resumed:
                raise NeuTraTrainingError("initial segment unexpectedly has resume lineage")
        else:
            if not isinstance(lineage, Mapping):
                raise NeuTraTrainingError("resumed segment lacks infrastructure lineage")
            if lineage.get("parent_state_path") != str(parent_state_path.resolve()):
                raise NeuTraTrainingError("segment parent checkpoint lineage mismatch")
            if int(lineage.get("parent_completed_steps", -1)) != start_step:
                raise NeuTraTrainingError("segment parent step lineage mismatch")
            if lineage.get("scientific_configuration_changed") is not False:
                raise NeuTraTrainingError("segment changed scientific configuration")
        if terminal != (trained.frozen_payload_path is not None):
            raise NeuTraTrainingError("transport freeze did not occur terminal-only")
        row = {
            "segment_id": segment_id,
            "start_step": start_step,
            "completed_steps": terminal_step,
            "program_step_count": terminal_step - start_step,
            "config_hash": segment_config.config_hash,
            "state_path": str(trained.state_path),
            "state_hash": trained.state_hash,
            "parent_state_path": (
                None if parent_state_path is None else str(parent_state_path.resolve())
            ),
            "resumed": trained.resumed,
            "terminal": terminal,
            "frozen_payload_path": (
                None
                if trained.frozen_payload_path is None
                else str(trained.frozen_payload_path)
            ),
            "runtime_metadata": trained.runtime_metadata,
        }
        segment_rows.append(row)
        _atomic_write_json(
            progress_path,
            {
                "schema": "bayesfilter.neutra.segmented_training_progress.v1",
                "config_hash": config.config_hash,
                "total_steps": config.steps,
                "segment_steps": segment_size,
                "completed_steps": terminal_step,
                "segment_rows": segment_rows,
                "terminal": terminal,
                "sample_axis_python_loop_used": False,
                "optimization_step_python_loop_used": False,
                "segment_orchestration_python_loop_used": True,
            },
        )
        parent_state_path = trained.state_path
        final_result = trained

    if final_result is None or final_result.frozen_payload_path is None:
        raise NeuTraTrainingError("segmented training did not reach a frozen terminal state")
    config_hashes = {str(row["config_hash"]) for row in segment_rows}
    if config_hashes != {config.config_hash}:
        raise NeuTraTrainingError("segment configuration hashes are inconsistent")
    _write_new_json(
        result_path,
        {
            "schema": "bayesfilter.neutra.segmented_training_result.v1",
            "config_hash": config.config_hash,
            "total_steps": config.steps,
            "segment_steps": segment_size,
            "segment_count": len(segment_rows),
            "segment_rows": segment_rows,
            "final_state_path": str(final_result.state_path),
            "final_state_hash": final_result.state_hash,
            "frozen_payload_path": str(final_result.frozen_payload_path),
            "terminal_only_freeze": True,
            "scientific_configuration_changed": False,
            "sample_axis_python_loop_used": False,
            "optimization_step_python_loop_used": False,
            "segment_orchestration_python_loop_used": True,
        },
    )
    return PlainDenseIAFSegmentedTrainingResult(
        final_result=final_result,
        segment_rows=tuple(segment_rows),
        progress_path=progress_path,
        result_path=result_path,
    )


def train_plain_dense_iaf(
    *,
    adapter: Any,
    config: PlainDenseIAFTrainingConfig,
    resume_from: str | Path | None = None,
    resume_repair_from: str | Path | None = None,
    resume_infrastructure_from: str | Path | None = None,
    stop_after_steps: int | None = None,
    freeze_transport_id: str | None = None,
) -> PlainDenseIAFTrainingResult:
    """Train or resume the exact-target plain flow and emit immutable state."""

    resume_sources = tuple(
        value
        for value in (
            resume_from,
            resume_repair_from,
            resume_infrastructure_from,
        )
        if value is not None
    )
    if len(resume_sources) > 1:
        raise NeuTraTrainingError(
            "resume modes are mutually exclusive"
        )
    try:
        batch_target = require_batch_native_neutra_target(
            adapter,
            target_signature=config.target_signature,
            batch_size=config.batch_size,
        )
    except InvalidNeuTraBatchTarget as exc:
        raise NeuTraTrainingError(str(exc)) from exc
    _validate_training_runtime(config)
    output_dir = config.output_dir
    if resume_infrastructure_from is not None and output_dir.exists():
        raise NeuTraTrainingError(
            "infrastructure resume requires a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_guard_path = output_dir / "frozen_transport.json"
    if frozen_guard_path.exists():
        raise NeuTraTrainingError(
            "frozen transport already exists; training after freeze is forbidden"
        )
    progress_path = output_dir / "training_progress.jsonl"
    latest_path = output_dir / "training_latest.json"
    config_path = output_dir / "training_config.json"
    config_payload = dict(config.payload())
    config_payload["config_hash"] = config.config_hash
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != _json_ready(config_payload):
            raise NeuTraTrainingError("existing training config mismatch")
    else:
        _write_new_json(config_path, config_payload)

    with tf.device(config.device):
        flow = PlainDenseIAFTransport(config)
        variables = flow.trainable_variables
        first_moments = [
            tf.Variable(tf.zeros_like(value), trainable=False) for value in variables
        ]
        second_moments = [
            tf.Variable(tf.zeros_like(value), trainable=False) for value in variables
        ]
    variable_devices = tuple(str(value.device) for value in variables)
    moment_devices = tuple(
        str(value.device) for value in (*first_moments, *second_moments)
    )
    if config.require_gpu and not all(
        "GPU" in device.upper() for device in (*variable_devices, *moment_devices)
    ):
        raise NeuTraTrainingError("training and Adam variables must be placed on GPU")
    runtime_metadata: dict[str, Any] = {
        "requested_device": config.device,
        "physical_gpus": tuple(
            str(device) for device in tf.config.list_physical_devices("GPU")
        ),
        "logical_gpus": tuple(
            str(device) for device in tf.config.list_logical_devices("GPU")
        ),
        "trainable_variable_devices": variable_devices,
        "adam_moment_devices": moment_devices,
        "compiled_output_devices": (),
        "jit_compile": True,
        "require_gpu": config.require_gpu,
        "batch_native_target": batch_target.payload(),
        "training_batch_size": config.batch_size,
        "scalar_fallback_used": False,
        "sample_axis_python_loop_used": False,
        "row_mapped_scalar_target_used": False,
    }
    completed_steps = 0
    records: list[Mapping[str, Any]] = []
    resumed = bool(resume_sources)
    repair_lineage: Mapping[str, Any] | None = None
    if resume_sources:
        if resume_repair_from is not None:
            state, repair_lineage = _read_lower_rate_repair_state(
                Path(resume_repair_from), config=config
            )
        elif resume_infrastructure_from is not None:
            state, repair_lineage = _read_infrastructure_resume_state(
                Path(resume_infrastructure_from), config=config
            )
        else:
            state = _read_training_state(Path(resume_from), config=config)
        completed_steps = int(state["completed_steps"])
        records = list(state.get("records", ()))
        _assign_state(variables, state["trainable_variables"], "trainable")
        _assign_state(first_moments, state["adam_first_moments"], "Adam first moment")
        _assign_state(second_moments, state["adam_second_moments"], "Adam second moment")

    terminal_step = config.steps if stop_after_steps is None else int(stop_after_steps)
    if terminal_step < completed_steps or terminal_step > config.steps:
        raise NeuTraTrainingError("stop_after_steps is outside the valid range")
    target_value_status = batch_native_value_status_target_fn(batch_target)

    def step_fn(step_index: tf.Tensor):
        step_i32 = tf.cast(step_index, tf.int32)
        seed = tf.stack(
            [
                tf.cast(config.seed[0], tf.int32),
                tf.cast(config.seed[1], tf.int32) + step_i32,
            ]
        )
        z = tf.random.stateless_normal(
            (config.batch_size, config.dimension), seed=seed, dtype=tf.float64
        )
        with tf.GradientTape() as tape:
            theta, logdet = flow.forward_and_logdet(z)
            log_prob, status = target_value_status(theta)
            status_code = tf.convert_to_tensor(status["status_code"], tf.int32)
            status_valid = tf.convert_to_tensor(
                status["valid_pre_regularized_score"], tf.bool
            )
            status_nonvalid = tf.logical_or(
                tf.not_equal(status_code, tf.zeros_like(status_code)),
                tf.logical_not(status_valid),
            )
            target_status_available = tf.constant(True)
            target_status_valid = tf.reduce_all(tf.logical_not(status_nonvalid))
            target_status_nonvalid_count = tf.reduce_sum(
                tf.cast(status_nonvalid, tf.int32)
            )
            target_floor_count_total = tf.reduce_sum(
                tf.convert_to_tensor(status["floor_count_value"], tf.int32)
            )
            target_min_innovation_eigenvalue = tf.reduce_min(
                tf.convert_to_tensor(
                    status["min_innovation_eigenvalue"], tf.float64
                )
            )
            target_condition_estimate_available = tf.reduce_all(
                tf.convert_to_tensor(
                    status["innovation_condition_estimate_available"], tf.bool
                )
            )
            target_max_innovation_condition = tf.reduce_max(
                tf.convert_to_tensor(
                    status["innovation_condition_estimate"], tf.float64
                )
            )
            loss = -tf.reduce_mean(log_prob + logdet)
        gradients = tape.gradient(loss, variables)
        if any(gradient is None for gradient in gradients):
            raise NeuTraTrainingError("training gradient is missing")
        raw_norm = tf.linalg.global_norm(gradients)
        clipped, _ = tf.clip_by_global_norm(gradients, config.clip_norm)
        clipped_norm = tf.linalg.global_norm(clipped)
        progress = tf.cast(step_i32 + 1, tf.float64) / tf.cast(config.steps, tf.float64)
        lr = tf.cast(config.learning_rate, tf.float64) * (
            1.0
            - progress
            * (1.0 - tf.cast(config.final_learning_rate_fraction, tf.float64))
        )
        t = tf.cast(step_i32 + 1, tf.float64)
        beta1 = tf.cast(config.beta1, tf.float64)
        beta2 = tf.cast(config.beta2, tf.float64)
        for variable, gradient, first, second in zip(
            variables, clipped, first_moments, second_moments
        ):
            first.assign(beta1 * first + (1.0 - beta1) * gradient)
            second.assign(beta2 * second + (1.0 - beta2) * tf.square(gradient))
            first_hat = first / (1.0 - tf.pow(beta1, t))
            second_hat = second / (1.0 - tf.pow(beta2, t))
            variable.assign_sub(
                lr * first_hat / (tf.sqrt(second_hat) + tf.cast(config.epsilon, tf.float64))
            )
        target_finite = tf.reduce_all(tf.math.is_finite(log_prob))
        return (
            loss,
            raw_norm,
            clipped_norm,
            lr,
            tf.reduce_mean(logdet),
            target_finite,
            target_status_available,
            target_status_valid,
            target_status_nonvalid_count,
            target_floor_count_total,
            target_min_innovation_eigenvalue,
            target_condition_estimate_available,
            target_max_innovation_condition,
        )

    diagnostic_specs = (
        ("loss", tf.float64),
        ("raw_gradient_norm", tf.float64),
        ("clipped_gradient_norm", tf.float64),
        ("learning_rate", tf.float64),
        ("mean_log_abs_det_jacobian", tf.float64),
        ("target_values_finite", tf.bool),
        ("target_status_available", tf.bool),
        ("target_status_all_valid", tf.bool),
        ("target_status_nonvalid_count", tf.int32),
        ("target_floor_count_total", tf.int32),
        ("target_min_innovation_eigenvalue", tf.float64),
        ("target_condition_estimate_available", tf.bool),
        ("target_max_innovation_condition_estimate", tf.float64),
    )
    program_step_count = terminal_step - completed_steps

    def training_program():
        arrays = tuple(
            tf.TensorArray(
                dtype=dtype,
                size=program_step_count,
                clear_after_read=False,
                element_shape=(),
            )
            for _name, dtype in diagnostic_specs
        )

        def condition(step_index, *_arrays):
            return step_index < tf.constant(terminal_step, tf.int32)

        def body(step_index, *current_arrays):
            outputs = step_fn(step_index)
            output_index = step_index - tf.constant(completed_steps, tf.int32)
            updated = tuple(
                array.write(output_index, value)
                for array, value in zip(current_arrays, outputs)
            )
            return (step_index + tf.constant(1, tf.int32), *updated)

        final = tf.while_loop(
            condition,
            body,
            (tf.constant(completed_steps, tf.int32), *arrays),
            parallel_iterations=1,
        )
        step_numbers = tf.range(
            completed_steps + 1, terminal_step + 1, dtype=tf.int32
        )
        record_mask = tf.logical_or(
            tf.logical_or(
                tf.equal(step_numbers, completed_steps + 1),
                tf.equal(tf.math.floormod(step_numbers, config.heartbeat_every), 0),
            ),
            tf.equal(step_numbers, terminal_step),
        )
        return (
            tf.boolean_mask(step_numbers, record_mask),
            *(
                tf.boolean_mask(array.stack(), record_mask)
                for array in final[1:]
            ),
        )

    compiled_program = tf.function(
        training_program, jit_compile=True, reduce_retracing=True
    )
    concrete_program = compiled_program.get_concrete_function()
    graph_operation_types = tuple(
        sorted({operation.type for operation in concrete_program.graph.get_operations()})
    )
    if not any("While" in operation_type for operation_type in graph_operation_types):
        raise NeuTraTrainingError("compiled multi-step training graph lacks control flow")
    start = time.monotonic()
    with tf.device(config.device):
        program_outputs = compiled_program()
    elapsed_seconds = time.monotonic() - start
    output_devices = tuple(str(item.device) for item in program_outputs)
    if config.require_gpu and not all(
        "GPU" in device.upper() for device in output_devices
    ):
        raise NeuTraTrainingError("compiled training outputs must remain on GPU")
    runtime_metadata.update(
        {
            "compiled_output_devices": output_devices,
            "compiled_training_program_invocations": 1,
            "compiled_training_control_flow": "tf_while_loop",
            "checkpoint_policy": "terminal_only_graph_native_v1",
            "graph_operation_types": graph_operation_types,
            "program_step_count": program_step_count,
        }
    )
    step_numbers = tuple(int(value) for value in program_outputs[0].numpy().tolist())
    diagnostic_values = {
        name: tuple(tensor.numpy().tolist())
        for (name, _dtype), tensor in zip(diagnostic_specs, program_outputs[1:])
    }
    numeric_names = (
        "loss",
        "raw_gradient_norm",
        "clipped_gradient_norm",
        "learning_rate",
        "mean_log_abs_det_jacobian",
    )
    for name in numeric_names:
        if not all(math.isfinite(float(value)) for value in diagnostic_values[name]):
            raise NeuTraTrainingError(f"nonfinite training diagnostic: {name}")
    if not all(bool(value) for value in diagnostic_values["target_values_finite"]):
        raise NeuTraTrainingError("nonfinite exact target value")
    if any(
        bool(available) and not bool(valid)
        for available, valid in zip(
            diagnostic_values["target_status_available"],
            diagnostic_values["target_status_all_valid"],
        )
    ):
        raise NeuTraTrainingError("invalid exact target status")
    new_records = []
    for index, step_number in enumerate(step_numbers):
        status_available = bool(
            diagnostic_values["target_status_available"][index]
        )
        raw_norm = float(diagnostic_values["raw_gradient_norm"][index])
        new_records.append(
            {
                "step": step_number,
                "loss": float(diagnostic_values["loss"][index]),
                "raw_gradient_norm": raw_norm,
                "clipped_gradient_norm": float(
                    diagnostic_values["clipped_gradient_norm"][index]
                ),
                "gradient_was_clipped": bool(raw_norm > config.clip_norm),
                "learning_rate": float(
                    diagnostic_values["learning_rate"][index]
                ),
                "mean_log_abs_det_jacobian": float(
                    diagnostic_values["mean_log_abs_det_jacobian"][index]
                ),
                "target_values_finite": True,
                "target_status_available": status_available,
                "target_status_all_valid": bool(
                    diagnostic_values["target_status_all_valid"][index]
                ),
                "target_status_nonvalid_count": int(
                    diagnostic_values["target_status_nonvalid_count"][index]
                ),
                "target_floor_count_total": int(
                    diagnostic_values["target_floor_count_total"][index]
                ),
                "target_min_innovation_eigenvalue": (
                    float(
                        diagnostic_values["target_min_innovation_eigenvalue"][
                            index
                        ]
                    )
                    if status_available
                    else None
                ),
                "target_max_innovation_condition_estimate": (
                    float(
                        diagnostic_values[
                            "target_max_innovation_condition_estimate"
                        ][index]
                    )
                    if status_available
                    and bool(
                        diagnostic_values[
                            "target_condition_estimate_available"
                        ][index]
                    )
                    else None
                ),
                "target_condition_estimate_available": bool(
                    diagnostic_values[
                        "target_condition_estimate_available"
                    ][index]
                ),
                "program_elapsed_seconds": elapsed_seconds,
            }
        )
    records.extend(new_records)
    for record in new_records:
        _append_jsonl(
            progress_path,
            {
                "schema": "bayesfilter.neutra.training_progress.v1",
                "event": "training_heartbeat",
                "config_hash": config.config_hash,
                **record,
                "nonclaims": NEUTRA_TRAINING_NONCLAIMS,
            },
        )
    _atomic_write_json(latest_path, records[-1])
    _write_checkpoint(
        output_dir=output_dir,
        config=config,
        flow=flow,
        first_moments=first_moments,
        second_moments=second_moments,
        completed_steps=terminal_step,
        records=records,
        runtime_metadata=runtime_metadata,
        repair_lineage=repair_lineage,
    )

    state_path = output_dir / f"checkpoint_step_{terminal_step:06d}.json"
    state = _read_training_state(state_path, config=config)
    state_hash = _training_state_hash(state)
    frozen_path: Path | None = None
    if freeze_transport_id is not None:
        if terminal_step != config.steps:
            raise NeuTraTrainingError("only a completed training run may be frozen")
        payload = flow.frozen_payload(
            transport_id=freeze_transport_id,
            target_signature=config.target_signature,
            training_state_hash=state_hash,
        )
        frozen_path = output_dir / "frozen_transport.json"
        _write_new_json(frozen_path, payload)
    return PlainDenseIAFTrainingResult(
        completed_steps=terminal_step,
        state_path=state_path,
        latest_path=latest_path,
        progress_path=progress_path,
        frozen_payload_path=frozen_path,
        state_hash=state_hash,
        records=tuple(records),
        resumed=resumed,
        runtime_metadata=runtime_metadata,
    )


def restore_plain_dense_iaf_flow(
    *,
    config: PlainDenseIAFTrainingConfig,
    state_path: str | Path,
) -> PlainDenseIAFTransport:
    """Restore trainable transport tensors for parity checks, not continued training."""

    with tf.device(config.device):
        flow = PlainDenseIAFTransport(config)
    state = _read_training_state(Path(state_path), config=config)
    _assign_state(flow.trainable_variables, state["trainable_variables"], "trainable")
    return flow


def _write_checkpoint(
    *,
    output_dir: Path,
    config: PlainDenseIAFTrainingConfig,
    flow: PlainDenseIAFTransport,
    first_moments: Sequence[tf.Variable],
    second_moments: Sequence[tf.Variable],
    completed_steps: int,
    records: Sequence[Mapping[str, Any]],
    runtime_metadata: Mapping[str, Any],
    repair_lineage: Mapping[str, Any] | None,
) -> Path:
    state = {
        "schema": TRAINING_STATE_SCHEMA,
        "config_hash": config.config_hash,
        "target_signature": config.target_signature,
        "completed_steps": int(completed_steps),
        "total_steps": config.steps,
        "trainable_variables": tuple(value.numpy().tolist() for value in flow.trainable_variables),
        "adam_first_moments": tuple(value.numpy().tolist() for value in first_moments),
        "adam_second_moments": tuple(value.numpy().tolist() for value in second_moments),
        "records": tuple(records),
        "runtime_metadata": dict(runtime_metadata),
        "objective": "exact_target_reverse_kl",
        "composition": "T_phi(z)=affine(dense_iaf_stack(z))",
        "repair_lineage": repair_lineage,
        "nonclaims": NEUTRA_TRAINING_NONCLAIMS,
    }
    state["state_hash"] = _training_state_hash(state)
    path = output_dir / f"checkpoint_step_{completed_steps:06d}.json"
    _write_new_json(path, state)
    _atomic_write_json(output_dir / "checkpoint_latest.json", state)
    return path


def _read_training_state(
    path: Path, *, config: PlainDenseIAFTrainingConfig
) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != TRAINING_STATE_SCHEMA:
        raise NeuTraTrainingError("training state schema mismatch")
    if value.get("config_hash") != config.config_hash:
        raise NeuTraTrainingError("training state config hash mismatch")
    if value.get("target_signature") != config.target_signature:
        raise NeuTraTrainingError("training state target signature mismatch")
    supplied = str(value.get("state_hash", ""))
    if supplied != _training_state_hash(value):
        raise NeuTraTrainingError("training state hash mismatch")
    return value


def _read_lower_rate_repair_state(
    path: Path, *, config: PlainDenseIAFTrainingConfig
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema") != TRAINING_STATE_SCHEMA:
        raise NeuTraTrainingError("repair parent training state schema mismatch")
    if value.get("target_signature") != config.target_signature:
        raise NeuTraTrainingError("repair parent target signature mismatch")
    if str(value.get("state_hash", "")) != _training_state_hash(value):
        raise NeuTraTrainingError("repair parent training state hash mismatch")
    parent_config_path = path.parent / "training_config.json"
    parent_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    if not isinstance(parent_config, Mapping):
        raise NeuTraTrainingError("repair parent training config must be a mapping")
    if parent_config.get("config_hash") != value.get("config_hash"):
        raise NeuTraTrainingError("repair parent config/state hash mismatch")
    child_config = dict(config.payload())
    child_config["config_hash"] = config.config_hash
    parent_comparable = dict(parent_config)
    child_comparable = _json_ready(child_config)
    for key in ("config_hash", "output_dir", "learning_rate"):
        parent_comparable.pop(key, None)
        child_comparable.pop(key, None)
    if parent_comparable != child_comparable:
        raise NeuTraTrainingError(
            "lower-rate repair may change only learning_rate and output_dir"
        )
    parent_rate = float(parent_config.get("learning_rate", float("nan")))
    child_rate = float(config.learning_rate)
    if not math.isfinite(parent_rate) or not child_rate < parent_rate:
        raise NeuTraTrainingError(
            "lower-rate repair learning_rate must be below the parent rate"
        )
    return value, {
        "repair_type": "single_lower_learning_rate_retry",
        "parent_state_path": str(path),
        "parent_state_hash": value["state_hash"],
        "parent_config_hash": value["config_hash"],
        "parent_learning_rate": parent_rate,
        "child_learning_rate": child_rate,
    }


def _read_infrastructure_resume_state(
    path: Path, *, config: PlainDenseIAFTrainingConfig
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    value = _read_training_state(path, config=config)
    if config.output_dir.resolve() == path.parent.resolve():
        raise NeuTraTrainingError(
            "infrastructure resume must use a different output directory"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return value, {
        "repair_type": "infrastructure_resume_same_config_fresh_output_v1",
        "parent_state_path": str(path.resolve()),
        "parent_state_file_sha256": digest,
        "parent_state_hash": value["state_hash"],
        "parent_config_hash": value["config_hash"],
        "parent_completed_steps": int(value["completed_steps"]),
        "scientific_configuration_changed": False,
    }


def _assign_state(
    variables: Sequence[tf.Variable], values: Sequence[Any], label: str
) -> None:
    if len(variables) != len(values):
        raise NeuTraTrainingError(f"{label} state length mismatch")
    for variable, value in zip(variables, values):
        tensor = tf.convert_to_tensor(value, dtype=tf.float64)
        if tensor.shape != variable.shape:
            raise NeuTraTrainingError(f"{label} state shape mismatch")
        if not bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy()):
            raise NeuTraTrainingError(f"{label} state must be finite")
        variable.assign(tensor)


def _validate_training_runtime(config: PlainDenseIAFTrainingConfig) -> None:
    if config.require_gpu:
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise NeuTraTrainingError("trusted TensorFlow GPU visibility is required")
        if "GPU" not in config.device.upper():
            raise NeuTraTrainingError("GPU training requires a GPU device")


def _dense_iaf_masks(
    dimension: int, hidden_layers: Sequence[int]
) -> tuple[tf.Tensor, ...]:
    degrees: list[tuple[int, ...]] = [tuple(range(1, dimension + 1))]
    max_degree = max(1, dimension - 1)
    for width in hidden_layers:
        degrees.append(tuple(1 + index % max_degree for index in range(int(width))))
    degrees.append(tuple(range(1, dimension + 1)) * 2)
    result = []
    for index, (left, right) in enumerate(zip(degrees[:-1], degrees[1:])):
        output_layer = index == len(degrees) - 2
        result.append(
            tf.constant(
                [
                    [
                        1.0
                        if ((source < target) if output_layer else (source <= target))
                        else 0.0
                        for target in right
                    ]
                    for source in left
                ],
                dtype=tf.float64,
            )
        )
    return tuple(result)


def _activation(values: tf.Tensor, name: str) -> tf.Tensor:
    if name == "elu":
        return tf.nn.elu(values)
    if name == "tanh":
        return tf.math.tanh(values)
    if name == "relu":
        return tf.nn.relu(values)
    raise ValueError(f"unsupported activation: {name}")


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")


def _training_state_hash(payload: Mapping[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("state_hash", None)
    return _stable_config_hash(unhashed)


def _bare_sha256(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{label} must be a bare lowercase sha256 digest")
    return text


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        raise NeuTraTrainingError("training artifacts must contain finite numbers")
    return value
