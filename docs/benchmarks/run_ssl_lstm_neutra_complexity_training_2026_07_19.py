#!/usr/bin/env python3
"""Q-general Optuna nomination and plateau NeuTra training harness.

Material study/final modes are fail-closed behind ``--authorize-material-run``.
The default contract-smoke mode performs no target evaluation or training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


def _configure_visibility_before_tensorflow_import() -> str:
    """Hide GPUs for contract smokes; otherwise prefer physical GPU 1."""

    if os.environ.get("BAYESFILTER_CPU_VALUE_SCORE_WORKER") == "1":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "-1"
    mode = None
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if index + 1 < len(sys.argv):
            mode = sys.argv[index + 1]
    if mode == "contract-smoke":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return "cpu-hidden-contract-smoke"
    if mode not in {"study", "final", "confirmation", "single-diagnostic"} and os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ) is not None:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    probe = subprocess.run(
        ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    available = {
        int(line.strip())
        for line in probe.stdout.splitlines()
        if line.strip().isdigit()
    }
    selected = "1" if 1 in available else ("0" if 0 in available else "")
    if not selected:
        raise RuntimeError("no physical GPU 1 or GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_GPU = _configure_visibility_before_tensorflow_import()
import tensorflow as tf


def _enable_memory_growth_before_project_imports() -> None:
    """Configure the TensorFlow allocator before importing TFP/project modules."""

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            raise RuntimeError(
                "GPU memory growth must be established immediately after TensorFlow import"
            ) from exc
        if tf.config.experimental.get_memory_growth(gpu) is not True:
            raise RuntimeError("GPU memory growth verification failed")


_enable_memory_growth_before_project_imports()


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.cpu_value_score_pool import (  # noqa: E402
    CPUValueScorePool,
    CPUValueScorePoolConfig,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_deep_capacity_neutra_config,
    ssl_lstm_wide_capacity_neutra_config,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.neutra_training_control import (  # noqa: E402
    NeuTraPlateauConfig,
    NeuTraPlateauController,
    joint_training_checkpoint_payload,
    validate_joint_training_checkpoint,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_training.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
SINGLE_DIAGNOSTIC_PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-single-seed-neutra-training-diagnostic-plan-2026-07-20.md"
)
DEEP_DIAGNOSTIC_PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-deep-32x32x32-saturation-test-plan-2026-07-20.md"
)
WIDE_DIAGNOSTIC_PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-wide-64x64-saturation-test-plan-2026-07-20.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-neutra-hmc-state-complexity-2026-07-19/phase-3-training"
)
Q_VALUES = (1, 2, 5, 10, 20)
WORKERS_BY_Q = {1: 32, 2: 32, 5: 32, 10: 32, 20: 16}
OPTUNA_RUNGS = (50, 100, 200, 400)
VALIDATION_BATCH_SIZE = 64
BATCH_SIZE = 480
DEFAULT_HIDDEN_LAYERS = (32, 32)
MAX_STEPS = 2000
SHELL_RADIUS = 4.0
SHELL_RADIUS_MAX = 4.30
HOST_RAM_CAP_BYTES = 64 * 1024**3


class ComplexityTrainingError(RuntimeError):
    """Raised when the q-general training contract is invalid."""


class ResourceStop(ComplexityTrainingError):
    """Raised after preserving a resource-stop checkpoint."""


class HostMemoryVeto(ComplexityTrainingError):
    """Raised when the selected topology breaches the 64 GiB host cap."""


@dataclass(frozen=True)
class Stream:
    label: str
    initialization_seed: tuple[int, int]
    training_seed: tuple[int, int]
    validation_seed: tuple[int, int]


@dataclass(frozen=True)
class TrialParameters:
    learning_rate: float
    initialization_scale: float
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        if not 1.0e-4 <= float(self.learning_rate) <= 2.0e-3:
            raise ValueError("learning_rate outside study search contract")
        if float(self.initialization_scale) not in {0.005, 0.01, 0.02}:
            raise ValueError("initialization_scale outside study search contract")
        if float(self.gradient_clip_norm) not in {5.0, 10.0}:
            raise ValueError("gradient_clip_norm outside study search contract")


STREAMS = (
    Stream("seed-a", (20260719, 12101), (20260719, 13101), (20260719, 14101)),
    Stream("seed-b", (20260719, 12102), (20260719, 13102), (20260719, 14102)),
)
FRESH_CONFIRMATION = Stream(
    "seed-c", (20260719, 12103), (20260719, 13103), (20260719, 14103)
)


class Budget:
    def __init__(self, seconds: float, *, prior_seconds: float = 0.0) -> None:
        self.seconds = float(seconds)
        self.prior_seconds = float(prior_seconds)
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self.prior_seconds + time.perf_counter() - self.started

    def require(self, reserve_seconds: float = 0.0) -> None:
        if self.elapsed + float(reserve_seconds) >= self.seconds:
            raise ResourceStop("declared training GPU-time cap exhausted")


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise ComplexityTrainingError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ComplexityTrainingError(f"{label} must remain inside the repository")
    return resolved


def pool_config(q: int, worker_count: int | None = None) -> CPUValueScorePoolConfig:
    q = int(q)
    if q not in Q_VALUES:
        raise ValueError("q is outside the complexity ladder")
    selected_workers = WORKERS_BY_Q[q] if worker_count is None else int(worker_count)
    if selected_workers != WORKERS_BY_Q[q]:
        raise ValueError("worker_count must match the Phase 2 selected topology")
    return CPUValueScorePoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_target_tf:"
            "complexity_target_worker_factory"
        ),
        worker_config={"q": q},
        dimension=4,
        worker_count=selected_workers,
        cores_per_worker=1,
    )


def stream_batch(stream: Stream, step: int, batch_size: int = BATCH_SIZE) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant(stream.training_seed, tf.int32), int(step)
    )
    return tf.random.stateless_normal((int(batch_size), 4), seed, dtype=tf.float64)


def validation_batch(stream: Stream) -> tf.Tensor:
    return tf.random.stateless_normal(
        (VALIDATION_BATCH_SIZE, 4),
        tf.constant(stream.validation_seed, tf.int32),
        dtype=tf.float64,
    )


def trainer_config(
    target: Any,
    stream: Stream,
    params: TrialParameters,
    hidden_layers: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
) -> Any:
    hidden_layers = tuple(int(value) for value in hidden_layers)
    if hidden_layers == (64, 64):
        config_factory = ssl_lstm_wide_capacity_neutra_config
    elif hidden_layers == (32, 32, 32):
        config_factory = ssl_lstm_deep_capacity_neutra_config
    elif hidden_layers == DEFAULT_HIDDEN_LAYERS:
        config_factory = ssl_lstm_tuned_capacity_neutra_config
    else:
        raise ComplexityTrainingError(
            "hidden_layers must be exactly (32,32), (32,32,32), or (64,64)"
        )
    return config_factory(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy()),
        target_parameter_names=target.parameter_names,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=float(params.learning_rate),
        initialization_scale=float(params.initialization_scale),
        gradient_clip_norm=float(params.gradient_clip_norm),
        initialization_seed=stream.initialization_seed,
        jit_compile=True,
    )


def plateau_config(params: TrialParameters) -> NeuTraPlateauConfig:
    return NeuTraPlateauConfig(
        validation_check_every=250,
        patience_steps=250,
        max_steps=MAX_STEPS,
        initial_learning_rate=float(params.learning_rate),
        learning_rate_factor=0.5,
        post_repair_no_improvement_cycles=2,
        minimum_learning_rate_fraction=1.0 / 16.0,
        absolute_min_delta=0.0,
        saturation_max=0.05,
        roundtrip_max_abs=1.0e-9,
        moderate_shell_max_inverse_radius=SHELL_RADIUS_MAX,
    )


def trial_parameters(trial: Any) -> TrialParameters:
    return TrialParameters(
        learning_rate=trial.suggest_float("learning_rate", 1.0e-4, 2.0e-3, log=True),
        initialization_scale=trial.suggest_categorical(
            "initialization_scale", [0.005, 0.01, 0.02]
        ),
        gradient_clip_norm=trial.suggest_categorical(
            "gradient_clip_norm", [5.0, 10.0]
        ),
    )


def fixed_smoke_parameters() -> TrialParameters:
    return TrialParameters(4.0e-4, 0.01, 10.0)


def _host_step(result: Any) -> dict[str, Any]:
    return {
        "step": int(result.step.numpy()),
        "loss": float(result.loss.numpy()),
        "surrogate": float(result.surrogate.numpy()),
        "target_value_mean": float(result.target_value_mean.numpy()),
        "logdet_mean": float(result.logdet_mean.numpy()),
        "gradient_norm": float(result.gradient_norm.numpy()),
        "clipped_gradient_norm": float(result.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(result.clipping_applied.numpy()),
    }


def _host_validation(validation: Any, *, step: int, learning_rate: float) -> dict[str, Any]:
    losses = np.asarray(validation.per_sample_loss.numpy(), dtype=np.float64)
    target_values = np.asarray(validation.target_value.numpy(), dtype=np.float64)
    theta = np.asarray(validation.theta.numpy(), dtype=np.float64)
    logdet = np.asarray(validation.logdet.numpy(), dtype=np.float64)
    scale_log = np.asarray(validation.scale_log.numpy(), dtype=np.float64)
    scale_logits = np.asarray(validation.scale_logits.numpy(), dtype=np.float64)
    hidden_preactivations = np.asarray(
        validation.hidden_preactivations.numpy(), dtype=np.float64
    )
    if not all(
        np.all(np.isfinite(value))
        for value in (
            losses,
            target_values,
            theta,
            logdet,
            scale_log,
            scale_logits,
            hidden_preactivations,
        )
    ):
        raise FloatingPointError("validation returned nonfinite values")
    if scale_log.shape[-1] % theta.shape[-1] != 0:
        raise ValueError("scale_log width must be divisible by target dimension")
    stage_scale_log = np.reshape(
        scale_log,
        (scale_log.shape[0], scale_log.shape[-1] // theta.shape[-1], theta.shape[-1]),
    )
    if scale_logits.ndim != 3 or scale_logits.shape[:2] != stage_scale_log.shape[:2]:
        raise ValueError("scale_logits must have shape [batch, stages, dimension]")
    if hidden_preactivations.ndim != 4:
        raise ValueError(
            "hidden_preactivations must have shape [batch, stages, layers, width]"
        )
    raw_scale_threshold = float(np.arctanh(0.95))
    hidden_abs = np.abs(hidden_preactivations)
    return {
        "step": int(step),
        "learning_rate": float(learning_rate),
        "per_sample_loss": losses.tolist(),
        "mean_loss": float(np.mean(losses)),
        "target_value_mean": float(np.mean(target_values)),
        "logdet_mean": float(np.mean(logdet)),
        "scale_log_min": float(np.min(scale_log)),
        "scale_log_max": float(np.max(scale_log)),
        "saturation_fraction": float(np.mean(np.abs(scale_log) >= 0.95)),
        "saturation_fraction_by_stage": np.mean(
            np.abs(stage_scale_log) >= 0.95, axis=(0, 2)
        ).tolist(),
        "scale_logit_min": float(np.min(scale_logits)),
        "scale_logit_max": float(np.max(scale_logits)),
        "scale_logit_tail_fraction_by_stage": np.mean(
            np.abs(scale_logits) >= raw_scale_threshold, axis=(0, 2)
        ).tolist(),
        "scale_logit_tail_threshold": raw_scale_threshold,
        "hidden_preactivation_min_by_stage": np.min(
            hidden_preactivations, axis=(0, 2, 3)
        ).tolist(),
        "hidden_preactivation_max_by_stage": np.max(
            hidden_preactivations, axis=(0, 2, 3)
        ).tolist(),
        "hidden_abs_tail_fraction_by_stage": np.mean(
            hidden_abs >= 5.0, axis=(0, 2, 3)
        ).tolist(),
        "hidden_negative_tail_fraction_by_stage": np.mean(
            hidden_preactivations <= -5.0, axis=(0, 2, 3)
        ).tolist(),
        "hidden_positive_tail_fraction_by_stage": np.mean(
            hidden_preactivations >= 5.0, axis=(0, 2, 3)
        ).tolist(),
        "hidden_preactivation_abs_threshold": 5.0,
        "theta_min_by_coordinate": np.min(theta, axis=0).tolist(),
        "theta_max_by_coordinate": np.max(theta, axis=0).tolist(),
    }


def _external_training_step(
    trainer: NeuTraReverseKLTrainer,
    pool: CPUValueScorePool,
    z: tf.Tensor,
    *,
    request_id: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    theta, _ = trainer.forward_and_logdet(z)
    values, scores, metadata = pool.evaluate(theta.numpy(), request_id=request_id)
    _enforce_host_memory(metadata)
    result = trainer.train_step_with_external_value_score(z, values, scores)
    return _host_step(result), metadata


def _external_validation(
    trainer: NeuTraReverseKLTrainer,
    pool: CPUValueScorePool,
    z: tf.Tensor,
    *,
    step: int,
    request_id: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    theta, _ = trainer.forward_and_logdet(z)
    values, metadata = pool.evaluate_values(theta.numpy(), request_id=request_id)
    _enforce_host_memory(metadata)
    validation = trainer.validation_batch_with_external_value(z, values)
    return (
        _host_validation(
            validation,
            step=step,
            learning_rate=float(trainer.learning_rate_at(step).numpy()),
        ),
        metadata,
    )


def _enforce_host_memory(metadata: Mapping[str, Any]) -> int:
    worker_bytes = max(
        int(metadata.get("active_worker_ru_maxrss_sum_bytes", 0)),
        int(metadata.get("startup_worker_ru_maxrss_sum_bytes", 0)),
    )
    parent_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    combined = worker_bytes + parent_bytes
    if combined > HOST_RAM_CAP_BYTES:
        raise HostMemoryVeto("combined parent/worker RSS exceeded 64 GiB")
    return combined


def _latent_shell() -> tf.Tensor:
    rows = [tf.zeros((4,), tf.float64)]
    for coordinate in range(4):
        direction = np.zeros(4, dtype=np.float64)
        direction[coordinate] = SHELL_RADIUS
        rows.extend((tf.constant(direction), tf.constant(-direction)))
    return tf.stack(rows)


def support_probe(
    transport: Any,
    pool: CPUValueScorePool,
    *,
    request_id: str,
) -> dict[str, Any]:
    z = _latent_shell()
    theta = transport.forward_batch(z)
    values, scores, metadata = pool.evaluate(theta.numpy(), request_id=request_id)
    _enforce_host_memory(metadata)
    replay_z = transport.inverse_theta_to_z_batch(theta)
    replay_theta = transport.forward_batch(replay_z)
    transformed_score = transport.pullback_score_batch(
        z, tf.constant(scores, tf.float64)
    ) + transport.log_abs_det_jacobian_score_batch(z)
    logdet = transport.log_abs_det_jacobian_batch(z)
    tensors = (theta, replay_z, replay_theta, transformed_score, logdet)
    all_finite = bool(all(tf.reduce_all(tf.math.is_finite(row)).numpy() for row in tensors))
    roundtrip = float(
        tf.reduce_max(
            tf.concat(
                (
                    tf.reshape(tf.abs(replay_z - z), [-1]),
                    tf.reshape(tf.abs(replay_theta - theta), [-1]),
                ),
                axis=0,
            )
        ).numpy()
    )
    return {
        "all_finite": all_finite and bool(np.all(np.isfinite(values))),
        "roundtrip_max_abs": roundtrip,
        "moderate_shell_max_inverse_radius": float(
            tf.reduce_max(tf.linalg.norm(replay_z, axis=-1)).numpy()
        ),
        "transformed_score_max_abs": float(
            tf.reduce_max(tf.abs(transformed_score)).numpy()
        ),
        "worker_backend": metadata,
        "probe_definition": "origin_plus_coordinate_shell_radius_4_in_neutra_z_chart",
    }


def trainer_support_probe(
    trainer: NeuTraReverseKLTrainer,
    target: Any,
    pool: CPUValueScorePool,
    *,
    request_id: str,
) -> dict[str, Any]:
    payload = trainer.frozen_transport_payload(
        transport_id=f"{request_id}-checkpoint-probe",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=target.target_signature()
    )
    return support_probe(loaded.transport, pool, request_id=request_id)


def externalize_payload(
    row: dict[str, Any],
    *,
    key: str,
    path: Path,
) -> None:
    payload = row.pop(key)
    write_json(path, payload)
    row[f"{key}_path"] = path.relative_to(ROOT).as_posix()
    row[f"{key}_sha256"] = sha256(path)


def _rung_vetoes(row: Mapping[str, Any]) -> list[str]:
    vetoes = []
    if not math.isfinite(float(row["mean_loss"])):
        vetoes.append("nonfinite_validation_loss")
    if float(row["saturation_fraction"]) > 0.05:
        vetoes.append("dense_scale_saturation_above_cap")
    return vetoes


def _paired_upper(initial: list[float], final: list[float]) -> dict[str, float]:
    differences = np.asarray(final, np.float64) - np.asarray(initial, np.float64)
    mean = float(np.mean(differences))
    se = float(np.std(differences, ddof=1) / math.sqrt(len(differences)))
    return {
        "mean_difference": mean,
        "standard_error": se,
        "one_sided_95_upper": mean + 1.6694022215079607 * se,
    }


def run_nomination_stream(
    *,
    target: Any,
    pool: CPUValueScorePool,
    stream: Stream,
    params: TrialParameters,
    trial_number: int,
    budget: Budget,
    report: Callable[[int, Mapping[str, Any]], bool] | None,
    batch_size: int = BATCH_SIZE,
    hidden_layers: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
) -> dict[str, Any]:
    trainer = NeuTraReverseKLTrainer(
        target, trainer_config(target, stream, params, hidden_layers)
    )
    validation_z = validation_batch(stream)
    initial, initial_pool = _external_validation(
        trainer,
        pool,
        validation_z,
        step=0,
        request_id=f"q{target.q}-trial{trial_number}-{stream.label}-validation-0",
    )
    history = [initial]
    train_rows = []
    pool_rows = [initial_pool]
    rung_set = set(OPTUNA_RUNGS)
    for step in range(1, OPTUNA_RUNGS[-1] + 1):
        budget.require(30.0)
        train_row, metadata = _external_training_step(
            trainer,
            pool,
            stream_batch(stream, step, batch_size=batch_size),
            request_id=f"q{target.q}-trial{trial_number}-{stream.label}-train-{step}",
        )
        if step in rung_set:
            train_rows.append(train_row)
            pool_rows.append(metadata)
            validation, validation_metadata = _external_validation(
                trainer,
                pool,
                validation_z,
                step=step,
                request_id=(
                    f"q{target.q}-trial{trial_number}-{stream.label}-validation-{step}"
                ),
            )
            history.append(validation)
            pool_rows.append(validation_metadata)
            vetoes = _rung_vetoes(validation)
            prune = bool(vetoes)
            if report is not None and not prune:
                prune = bool(report(OPTUNA_RUNGS.index(step), validation))
                if prune:
                    vetoes.append("optuna_pruner")
            if prune:
                return {
                    "status": "PRUNED",
                    "stream": asdict(stream),
                    "history": history,
                    "training_rows": train_rows,
                    "vetoes": vetoes,
                    "pool_receipts": pool_rows,
                }
    frozen = trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-complexity-q{target.q}-trial{trial_number}-{stream.label}",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    probes = support_probe(
        loaded.transport,
        pool,
        request_id=f"q{target.q}-trial{trial_number}-{stream.label}-support",
    )
    interval = _paired_upper(initial["per_sample_loss"], history[-1]["per_sample_loss"])
    vetoes = _rung_vetoes(history[-1])
    if not probes["all_finite"]:
        vetoes.append("support_probe_nonfinite")
    if probes["roundtrip_max_abs"] > 1.0e-9:
        vetoes.append("roundtrip_residual_above_threshold")
    if interval["one_sided_95_upper"] >= 0.0:
        vetoes.append("heldout_loss_improvement_not_established")
    return {
        "status": "SURVIVED" if not vetoes else "VETOED",
        "stream": asdict(stream),
        "history": history,
        "training_rows": train_rows,
        "vetoes": sorted(set(vetoes)),
        "paired_final_minus_initial": interval,
        "support_probe": probes,
        "objective": float(history[-1]["mean_loss"]),
        "frozen_payload": frozen,
        "pool_receipts": pool_rows,
    }


def run_final_stream(
    *,
    target: Any,
    pool: CPUValueScorePool,
    stream: Stream,
    params: TrialParameters,
    budget: Budget,
    output_dir: Path,
    resume: bool = False,
    batch_size: int = BATCH_SIZE,
    hidden_layers: tuple[int, ...] = DEFAULT_HIDDEN_LAYERS,
) -> dict[str, Any]:
    trainer = NeuTraReverseKLTrainer(
        target, trainer_config(target, stream, params, hidden_layers)
    )
    controller = NeuTraPlateauController(plateau_config(params))
    validation_z = validation_batch(stream)
    progress_path = output_dir / "progress.json"
    resource_path = output_dir / "resource-stop.json"
    if resume:
        if not progress_path.is_file() or not resource_path.is_file():
            raise ComplexityTrainingError("resume requires progress and resource-stop files")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        resource_stop = json.loads(resource_path.read_text(encoding="utf-8"))
        if progress.get("schema") != SCHEMA or resource_stop.get("schema") != SCHEMA:
            raise ComplexityTrainingError("resume artifact schema mismatch")
        if progress.get("stream") != asdict(stream):
            raise ComplexityTrainingError("resume stream mismatch")
        joint = resource_stop.get("joint_checkpoint", {})
        validate_joint_training_checkpoint(joint)
        trainer.restore_state(joint["trainer_state"])
        controller.restore_state(joint["controller_state"])
        best_state = joint["best_trainer_state"]
        history = list(progress.get("history", []))
        checkpoints = list(progress.get("checkpoints", []))
        pool_receipts = list(progress.get("pool_receipts", []))
        if not history or best_state is None:
            raise ComplexityTrainingError("resume progress is incomplete")
        initial = history[0]
        start_step = int(resource_stop["next_program_step"])
        resource_path.unlink()
    else:
        initial, initial_pool = _external_validation(
            trainer,
            pool,
            validation_z,
            step=0,
            request_id=f"q{target.q}-final-{stream.label}-validation-0",
        )
        initial_probe = trainer_support_probe(
            trainer,
            target,
            pool,
            request_id=f"q{target.q}-final-{stream.label}-support-0",
        )
        initial_state = trainer.state_payload()
        initial_action = controller.observe(
            step=0,
            per_sample_loss=initial["per_sample_loss"],
            saturation_fraction=initial["saturation_fraction"],
            all_finite=initial_probe["all_finite"],
            roundtrip_max_abs=initial_probe["roundtrip_max_abs"],
            moderate_shell_max_inverse_radius=initial_probe[
                "moderate_shell_max_inverse_radius"
            ],
            trainer_state_hash=initial_state["state_hash"],
        )
        if initial_action.kind != "initialize_best":
            raise ComplexityTrainingError("initial validation did not initialize best state")
        best_state = initial_state
        history = [
            {
                **initial,
                "support_probe": initial_probe,
                "controller_action": initial_action.payload(),
            }
        ]
        pool_receipts = [initial_pool]
        checkpoints = []
        start_step = 1
        write_final_progress(
            progress_path,
            stream=stream,
            history=history,
            checkpoints=checkpoints,
            pool_receipts=pool_receipts,
            last_program_step=0,
        )
    stop_reason = None
    terminal_program_step = None
    for step in range(start_step, MAX_STEPS + 1):
        try:
            budget.require(60.0)
        except ResourceStop:
            emergency = joint_training_checkpoint_payload(
                trainer_state=trainer.state_payload(),
                controller_state=controller.state_payload(),
                best_trainer_state=best_state,
            )
            write_json(
                resource_path,
                {
                    "schema": SCHEMA,
                    "status": "RESOURCE_STOP",
                    "stream": asdict(stream),
                    "next_program_step": step,
                    "joint_checkpoint": emergency,
                    "candidate_veto": False,
                    "scientific_interpretation": "none",
                },
                replace=True,
            )
            raise
        _row, metadata = _external_training_step(
            trainer,
            pool,
            stream_batch(stream, step, batch_size=batch_size),
            request_id=f"q{target.q}-final-{stream.label}-train-{step}",
        )
        if step % controller.config.validation_check_every != 0:
            continue
        validation, validation_metadata = _external_validation(
            trainer,
            pool,
            validation_z,
            step=step,
            request_id=f"q{target.q}-final-{stream.label}-validation-{step}",
        )
        pool_receipts.extend((metadata, validation_metadata))
        checkpoint_probe = trainer_support_probe(
            trainer,
            target,
            pool,
            request_id=f"q{target.q}-final-{stream.label}-support-{step}",
        )
        current_state = trainer.state_payload()
        action = controller.observe(
            step=step,
            per_sample_loss=validation["per_sample_loss"],
            saturation_fraction=validation["saturation_fraction"],
            all_finite=checkpoint_probe["all_finite"],
            roundtrip_max_abs=checkpoint_probe["roundtrip_max_abs"],
            moderate_shell_max_inverse_radius=checkpoint_probe[
                "moderate_shell_max_inverse_radius"
            ],
            trainer_state_hash=current_state["state_hash"],
        )
        if action.meaningful_improvement:
            best_state = current_state
        if action.should_reduce_learning_rate:
            trainer.restore_state(best_state)
            trainer.set_learning_rate(action.current_learning_rate)
            current_state = trainer.state_payload()
        joint = joint_training_checkpoint_payload(
            trainer_state=current_state,
            controller_state=controller.state_payload(),
            best_trainer_state=best_state,
        )
        checkpoint_path = output_dir / f"checkpoint-{step:04d}.json"
        write_json(checkpoint_path, joint)
        checkpoints.append(
            {
                "step": step,
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(checkpoint_path),
                "checkpoint_hash": joint["checkpoint_hash"],
            }
        )
        history.append(
            {
                **validation,
                "support_probe": checkpoint_probe,
                "controller_action": action.payload(),
            }
        )
        write_final_progress(
            progress_path,
            stream=stream,
            history=history,
            checkpoints=checkpoints,
            pool_receipts=pool_receipts,
            last_program_step=step,
        )
        if action.should_stop:
            stop_reason = action.stop_reason
            terminal_program_step = step
            break
    if stop_reason is None:
        raise ComplexityTrainingError("final stream ended without a stop reason")
    best_trainer = NeuTraReverseKLTrainer(target, trainer.config)
    best_trainer.restore_state(best_state)
    frozen = best_trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-complexity-q{target.q}-final-{stream.label}",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    probes = support_probe(
        loaded.transport,
        pool,
        request_id=f"q{target.q}-final-{stream.label}-support",
    )
    paired = _paired_upper(initial["per_sample_loss"], list(controller.best_per_sample_loss))
    vetoes = []
    if stop_reason == "scale_saturation_above_cap":
        vetoes.append("dense_scale_saturation_above_cap")
    if not probes["all_finite"]:
        vetoes.append("support_probe_nonfinite")
    if probes["roundtrip_max_abs"] > 1.0e-9:
        vetoes.append("roundtrip_residual_above_threshold")
    if paired["one_sided_95_upper"] >= 0.0:
        vetoes.append("heldout_loss_improvement_not_established")
    return {
        "schema": SCHEMA,
        "q": target.q,
        "status": "ADMITTED" if not vetoes else "VETOED",
        "stream": asdict(stream),
        "params": asdict(params),
        "stop_reason": stop_reason,
        "best_step": controller.best_step,
        "terminal_program_step": terminal_program_step,
        "terminal_optimizer_step": int(trainer.step.numpy()),
        "learning_rate_reductions": controller.learning_rate_reductions,
        "history": history,
        "checkpoints": checkpoints,
        "paired_best_minus_initial": paired,
        "support_probe": probes,
        "vetoes": vetoes,
        "best_trainer_state": best_state,
        "best_frozen_payload": frozen,
        "pool_receipts": pool_receipts,
    }


def write_final_progress(
    path: Path,
    *,
    stream: Stream,
    history: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    pool_receipts: list[Mapping[str, Any]],
    last_program_step: int,
) -> None:
    write_json(
        path,
        {
            "schema": SCHEMA,
            "status": "RUNNING",
            "stream": asdict(stream),
            "last_program_step": int(last_program_step),
            "history": history,
            "checkpoints": checkpoints,
            "pool_receipts": pool_receipts,
        },
        replace=True,
    )


def source_bindings(*, plan: Path = PLAN) -> dict[str, Any]:
    paths = {
        "runner": SCRIPT,
        "plan": plan,
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        "pool": Path("bayesfilter/inference/cpu_value_score_pool.py"),
        "trainer": Path("bayesfilter/inference/neutra_training.py"),
        "controller": Path("bayesfilter/inference/neutra_training_control.py"),
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_paths": {key: path.as_posix() for key, path in paths.items()},
        "source_sha256": {key: sha256(ROOT / path) for key, path in paths.items()},
    }


def run_manifest(args: argparse.Namespace, charged_seconds: float) -> dict[str, Any]:
    if args.mode == "single-diagnostic" and args.hidden_layers == (64, 64):
        plan = WIDE_DIAGNOSTIC_PLAN
    elif args.mode == "single-diagnostic" and args.hidden_layers == (32, 32, 32):
        plan = DEEP_DIAGNOSTIC_PLAN
    elif args.mode == "single-diagnostic":
        plan = SINGLE_DIAGNOSTIC_PLAN
    else:
        plan = PLAN
    layer_sizes = (4, *args.hidden_layers, 8)
    parameters_per_stage = sum(
        input_width * output_width + output_width
        for input_width, output_width in zip(layer_sizes[:-1], layer_sizes[1:])
    )
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "optuna": _optuna_version(),
        "selected_physical_gpu": SELECTED_GPU,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
        "dtype": "float64",
        "jit_compile_parent": True,
        "jit_compile_workers": False,
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "soft_device_placement": False,
        "worker_count": WORKERS_BY_Q[args.q],
        "cores_per_worker": 1,
        "charged_seconds": float(charged_seconds),
        "gpu_cap_seconds": args.gpu_cap_seconds,
        "host_ru_maxrss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "plan": plan.as_posix(),
        "output_root": args.output_root.as_posix(),
        "batch_size": int(args.batch_size),
        "hidden_layers": list(args.hidden_layers),
        "trainable_parameter_count": 3 * parameters_per_stage,
    }


def _optuna_version() -> str:
    try:
        import optuna
    except ModuleNotFoundError:
        return "unavailable"
    return str(optuna.__version__)


def configure_gpu() -> list[Any]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise ComplexityTrainingError("material NeuTra training requires a visible GPU")
    for gpu in gpus:
        if tf.config.experimental.get_memory_growth(gpu) is not True:
            raise ComplexityTrainingError("GPU memory growth verification failed")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    return gpus


def validate_material_args(args: argparse.Namespace) -> None:
    if args.mode != "contract-smoke" and not args.authorize_material_run:
        raise ComplexityTrainingError(
            "material modes require --authorize-material-run and explicit budget"
        )
    if args.mode != "contract-smoke":
        if args.gpu_cap_seconds is None or args.gpu_cap_seconds <= 0.0:
            raise ComplexityTrainingError("material modes require a positive GPU cap")
        if args.output_root == DEFAULT_OUTPUT_ROOT:
            raise ComplexityTrainingError("material modes require an explicit output root")
        repo_path(args.output_root, label="output root")
    if args.mode in {"final", "confirmation", "single-diagnostic"} and args.params_json is None:
        raise ComplexityTrainingError(f"{args.mode} mode requires --params-json")
    if args.mode == "confirmation" and (
        args.failed_result is None or args.final_summary is None
    ):
        raise ComplexityTrainingError(
            "confirmation mode requires --final-summary and --failed-result"
        )


def contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    params = fixed_smoke_parameters()
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": args.q,
        "hidden_layers": list(args.hidden_layers),
        "selected_worker_count": WORKERS_BY_Q[args.q],
        "pool_config": asdict(pool_config(args.q)),
        "search_space": {
            "learning_rate": [1.0e-4, 2.0e-3, "log"],
            "initialization_scale": [0.005, 0.01, 0.02],
            "gradient_clip_norm": [5.0, 10.0],
            "rungs": list(OPTUNA_RUNGS),
            "n_trials": 6,
        },
        "plateau_config": plateau_config(params).manifest_payload(),
        "external_boundary": {
            "training": "CPU pool value_score -> GPU external custom-gradient update",
            "validation": "CPU pool value_only -> GPU transport validation",
            "pool_lifetime": "one persistent pool across trials and streams",
        },
        "repair_order": [
            "observe paired validation plateau",
            "restore best trainer and Adam state",
            "halve learning rate without resetting controller patience",
            "stop after two additional validation cycles without improvement",
        ],
        "fresh_confirmation_contract": {
            "stream": asdict(FRESH_CONFIRMATION),
            "trigger": "exactly one q-matched VETOED seed-a or seed-b result",
            "params": "same nominated hyperparameters as the failed final run",
            "separate_receipt_required": True,
        },
        "material_execution_authorized": False,
        "source_bindings": source_bindings(),
        "nonclaims": [
            "contract and import smoke only",
            "no target evaluation",
            "no NeuTra training",
            "no hyperparameter nomination",
            "no HMC or posterior claim",
        ],
    }


def run_study(args: argparse.Namespace) -> dict[str, Any]:
    import optuna

    target = complexity_posterior_target(args.q, jit_compile=True)
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    summary_path = output / "study-summary.json"
    prior_seconds = 0.0
    if args.resume:
        if not summary_path.is_file():
            raise ComplexityTrainingError("study resume requires study-summary.json")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("schema") != SCHEMA or int(previous.get("q", -1)) != args.q:
            raise ComplexityTrainingError("study resume summary mismatch")
        prior_seconds = float(previous.get("charged_seconds", 0.0))
        previous_trial_records = list(previous.get("trial_records", []))
    else:
        previous_trial_records = []
    budget = Budget(args.gpu_cap_seconds, prior_seconds=prior_seconds)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{output / 'study.sqlite3'}",
        load_if_exists=args.resume,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=args.sampler_seed,
            n_startup_trials=min(2, args.n_trials),
        ),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=2, min_early_stopping_rate=0
        ),
    )
    trial_records = previous_trial_records
    with CPUValueScorePool(pool_config(args.q)) as pool:
        def objective(trial: Any) -> float:
            params = trial_parameters(trial)
            rows = []
            histories: list[list[Mapping[str, Any]]] = []

            def report_a(index: int, row: Mapping[str, Any]) -> bool:
                trial.report(float(row["mean_loss"]), step=index + 1)
                return trial.should_prune()

            try:
                a = run_nomination_stream(
                    target=target,
                    pool=pool,
                    stream=STREAMS[0],
                    params=params,
                    trial_number=trial.number,
                    budget=budget,
                    report=report_a,
                    batch_size=args.batch_size,
                    hidden_layers=args.hidden_layers,
                )
            except (ResourceStop, HostMemoryVeto) as exc:
                record = {
                    "trial": trial.number,
                    "params": asdict(params),
                    "status": (
                        "RESOURCE_STOP" if isinstance(exc, ResourceStop) else "HARD_VETO"
                    ),
                    "active_stream": STREAMS[0].label,
                    "error": str(exc),
                    "candidate_veto": False,
                    "scientific_interpretation": "none",
                }
                trial_records.append(record)
                write_trial_record(output, trial.number, record)
                raise
            stream_dir = output / "trials" / f"trial-{trial.number:04d}" / STREAMS[0].label
            stream_dir.mkdir(parents=True, exist_ok=False)
            if "frozen_payload" in a:
                externalize_payload(
                    a,
                    key="frozen_payload",
                    path=stream_dir / "frozen-payload.json",
                )
            rows.append(a)
            histories.append(a["history"])
            if a["status"] != "SURVIVED":
                record = {"trial": trial.number, "params": asdict(params), "streams": rows}
                trial_records.append(record)
                write_trial_record(output, trial.number, record)
                raise optuna.TrialPruned("seed-a pruned or vetoed")

            def report_b(index: int, row: Mapping[str, Any]) -> bool:
                common_worst = max(
                    float(histories[0][index + 1]["mean_loss"]),
                    float(row["mean_loss"]),
                )
                trial.report(common_worst, step=len(OPTUNA_RUNGS) + index + 1)
                return trial.should_prune()

            try:
                b = run_nomination_stream(
                    target=target,
                    pool=pool,
                    stream=STREAMS[1],
                    params=params,
                    trial_number=trial.number,
                    budget=budget,
                    report=report_b,
                    batch_size=args.batch_size,
                    hidden_layers=args.hidden_layers,
                )
            except (ResourceStop, HostMemoryVeto) as exc:
                record = {
                    "trial": trial.number,
                    "params": asdict(params),
                    "status": (
                        "RESOURCE_STOP" if isinstance(exc, ResourceStop) else "HARD_VETO"
                    ),
                    "active_stream": STREAMS[1].label,
                    "completed_streams": rows,
                    "error": str(exc),
                    "candidate_veto": False,
                    "scientific_interpretation": "none",
                }
                trial_records.append(record)
                write_trial_record(output, trial.number, record)
                raise
            stream_dir = output / "trials" / f"trial-{trial.number:04d}" / STREAMS[1].label
            stream_dir.mkdir(parents=True, exist_ok=False)
            if "frozen_payload" in b:
                externalize_payload(
                    b,
                    key="frozen_payload",
                    path=stream_dir / "frozen-payload.json",
                )
            rows.append(b)
            record = {"trial": trial.number, "params": asdict(params), "streams": rows}
            trial_records.append(record)
            write_trial_record(output, trial.number, record)
            if b["status"] != "SURVIVED":
                raise optuna.TrialPruned("seed-b pruned or vetoed")
            return max(float(a["objective"]), float(b["objective"]))

        resource_stop = None
        hard_veto = None
        try:
            remaining_trials = max(0, int(args.n_trials) - len(study.trials))
            if remaining_trials > 0:
                study.optimize(
                    objective,
                    n_trials=remaining_trials,
                    timeout=args.timeout_seconds,
                )
        except ResourceStop as exc:
            resource_stop = str(exc)
        except HostMemoryVeto as exc:
            hard_veto = str(exc)
    viable = [
        trial for trial in study.trials if trial.state.name == "COMPLETE" and trial.value is not None
    ]
    payload = {
        "schema": SCHEMA,
        "mode": "study",
        "status": (
            "HARD_VETO"
            if hard_veto is not None
            else "RESOURCE_STOP"
            if resource_stop is not None
            else ("COMPLETED" if viable else "NO_VIABLE_TRIAL")
        ),
        "q": args.q,
        "trial_records": trial_records,
        "nominated_params": None if not viable else study.best_trial.params,
        "objective_role": "nomination_proxy_not_promotion_criterion",
        "charged_seconds": budget.elapsed,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "candidate_veto": False if resource_stop is not None else None,
        "scientific_interpretation": "none" if resource_stop is not None else None,
        "run_manifest": run_manifest(args, budget.elapsed),
        "source_bindings": source_bindings(),
        "nonclaims": [
            "Optuna nomination only",
            "does not statistically rank viable trials",
            "no HMC or posterior correctness claim",
        ],
    }
    write_json(summary_path, payload, replace=args.resume)
    return payload


def write_trial_record(output: Path, trial_number: int, record: Mapping[str, Any]) -> None:
    trial_dir = output / "trials" / f"trial-{int(trial_number):04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_json(trial_dir / "trial-result.json", record)


def run_final(args: argparse.Namespace) -> dict[str, Any]:
    if args.params_json is None:
        raise ComplexityTrainingError("final mode requires --params-json")
    params_path = repo_path(args.params_json, label="params json")
    params_payload = json.loads(params_path.read_text(encoding="utf-8"))
    if "nominated_params" in params_payload:
        params_payload = params_payload["nominated_params"]
    params = TrialParameters(**params_payload)
    target = complexity_posterior_target(args.q, jit_compile=True)
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    summary_path = output / "final-summary.json"
    prior_seconds = 0.0
    previous_results: dict[str, Mapping[str, Any]] = {}
    if args.resume:
        if not summary_path.is_file():
            raise ComplexityTrainingError("final resume requires final-summary.json")
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("schema") != SCHEMA or int(previous.get("q", -1)) != args.q:
            raise ComplexityTrainingError("final resume summary mismatch")
        if previous.get("params") != asdict(params):
            raise ComplexityTrainingError("final resume parameter mismatch")
        prior_seconds = float(previous.get("charged_seconds", 0.0))
        previous_results = {
            str(row["label"]): row for row in previous.get("results", [])
        }
    budget = Budget(args.gpu_cap_seconds, prior_seconds=prior_seconds)
    results = []
    resource_stop = None
    hard_veto = None
    with CPUValueScorePool(pool_config(args.q)) as pool:
        for stream in STREAMS:
            if stream.label in previous_results:
                row = previous_results[stream.label]
                path = repo_path(Path(str(row["path"])), label="completed stream result")
                if not path.is_file() or sha256(path) != row["sha256"]:
                    raise ComplexityTrainingError("completed stream result hash mismatch")
                results.append(dict(row))
                continue
            stream_dir = output / stream.label
            stream_dir.mkdir(parents=True, exist_ok=args.resume)
            try:
                result = run_final_stream(
                    target=target,
                    pool=pool,
                    stream=stream,
                    params=params,
                    budget=budget,
                    output_dir=stream_dir,
                    resume=args.resume and (stream_dir / "resource-stop.json").is_file(),
                    batch_size=args.batch_size,
                    hidden_layers=args.hidden_layers,
                )
            except ResourceStop as exc:
                resource_stop = str(exc)
                break
            except HostMemoryVeto as exc:
                hard_veto = str(exc)
                write_json(
                    stream_dir / "host-memory-veto.json",
                    {
                        "schema": SCHEMA,
                        "status": "HARD_VETO",
                        "stream": asdict(stream),
                        "reason": hard_veto,
                        "candidate_veto": False,
                        "continuation_veto": True,
                        "scientific_interpretation": "none",
                    },
                )
                break
            externalize_payload(
                result,
                key="best_trainer_state",
                path=stream_dir / "best-state.json",
            )
            externalize_payload(
                result,
                key="best_frozen_payload",
                path=stream_dir / "best-frozen-payload.json",
            )
            result_path = stream_dir / "result.json"
            write_json(result_path, result)
            results.append(
                {
                    "label": stream.label,
                    "path": result_path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(result_path),
                    "status": result["status"],
                }
            )
    payload = {
        "schema": SCHEMA,
        "mode": "final",
        "status": (
            "HARD_VETO"
            if hard_veto is not None
            else "RESOURCE_STOP"
            if resource_stop is not None
            else "COMPLETED"
        ),
        "q": args.q,
        "params": asdict(params),
        "results": results,
        "fresh_confirmation_eligible": (
            resource_stop is None
            and hard_veto is None
            and len(results) == 2
            and {row["label"] for row in results} == {stream.label for stream in STREAMS}
            and sorted(row["status"] for row in results) == ["ADMITTED", "VETOED"]
        ),
        "fresh_confirmation_authorized": False,
        "charged_seconds": budget.elapsed,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "candidate_veto": False if resource_stop is not None else None,
        "scientific_interpretation": "none" if resource_stop is not None else None,
        "run_manifest": run_manifest(args, budget.elapsed),
        "source_bindings": source_bindings(),
        "nonclaims": [
            "transport training/admission only",
            "fresh confirmation requires a separate recorded launch",
            "no HMC or posterior correctness claim",
        ],
    }
    write_json(summary_path, payload, replace=args.resume)
    return payload


def run_single_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    """Run one seed without weakening the two-seed Phase 3 admission contract."""

    if args.q != 20:
        raise ComplexityTrainingError("single-diagnostic mode is bound to q=20")
    assert args.params_json is not None
    params_path = repo_path(args.params_json, label="diagnostic params json")
    params = TrialParameters(**json.loads(params_path.read_text(encoding="utf-8")))
    stream = STREAMS[0]
    target = complexity_posterior_target(args.q, jit_compile=True)
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    summary_path = output / "single-diagnostic-summary.json"
    prior_seconds = 0.0
    if args.resume:
        if not summary_path.is_file():
            raise ComplexityTrainingError(
                "single-diagnostic resume requires single-diagnostic-summary.json"
            )
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            previous.get("schema") != SCHEMA
            or previous.get("mode") != "single-diagnostic"
            or int(previous.get("q", -1)) != 20
            or previous.get("params") != asdict(params)
            or previous.get("stream") != asdict(stream)
            or previous.get("hidden_layers") != list(args.hidden_layers)
            or previous.get("status") != "RESOURCE_STOP"
        ):
            raise ComplexityTrainingError("single-diagnostic resume summary mismatch")
        prior_seconds = float(previous.get("charged_seconds", 0.0))
    budget = Budget(args.gpu_cap_seconds, prior_seconds=prior_seconds)
    stream_dir = output / stream.label
    stream_dir.mkdir(parents=True, exist_ok=args.resume)
    result_row = None
    resource_stop = None
    hard_veto = None
    with CPUValueScorePool(pool_config(args.q)) as pool:
        try:
            result = run_final_stream(
                target=target,
                pool=pool,
                stream=stream,
                params=params,
                budget=budget,
                output_dir=stream_dir,
                resume=args.resume and (stream_dir / "resource-stop.json").is_file(),
                batch_size=args.batch_size,
                hidden_layers=args.hidden_layers,
            )
        except ResourceStop as exc:
            resource_stop = str(exc)
        except HostMemoryVeto as exc:
            hard_veto = str(exc)
            write_json(
                stream_dir / "host-memory-veto.json",
                {
                    "schema": SCHEMA,
                    "status": "HARD_VETO",
                    "stream": asdict(stream),
                    "reason": hard_veto,
                    "continuation_veto": True,
                    "scientific_interpretation": "none",
                },
                replace=args.resume,
            )
        else:
            per_seed_gate_status = str(result["status"])
            result["per_seed_gate_status"] = per_seed_gate_status
            result["phase3_admission_status"] = "NOT_EVALUATED_ONE_SEED"
            result["status"] = (
                "DIAGNOSTIC_SURVIVED"
                if per_seed_gate_status == "ADMITTED"
                else "DIAGNOSTIC_VETOED"
            )
            result["evidence_role"] = "single_seed_mechanism_diagnostic_only"
            result["nonclaims"] = [
                "fixed-smoke hyperparameters are not q=20 nominated",
                "one seed cannot establish robustness or Phase 3 admission",
                "no HMC, posterior-correctness, or scientific-validity claim",
            ]
            externalize_payload(
                result,
                key="best_trainer_state",
                path=stream_dir / "best-state.json",
            )
            externalize_payload(
                result,
                key="best_frozen_payload",
                path=stream_dir / "best-frozen-payload.json",
            )
            result_path = stream_dir / "result.json"
            write_json(result_path, result, replace=args.resume)
            result_row = {
                "label": stream.label,
                "path": result_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(result_path),
                "status": result["status"],
                "per_seed_gate_status": per_seed_gate_status,
            }
    payload = {
        "schema": SCHEMA,
        "mode": "single-diagnostic",
        "status": (
            "HARD_VETO"
            if hard_veto is not None
            else "RESOURCE_STOP"
            if resource_stop is not None
            else "COMPLETED"
        ),
        "q": 20,
        "stream": asdict(stream),
        "params": asdict(params),
        "hidden_layers": list(args.hidden_layers),
        "params_provenance": "existing_fixed_smoke_parameters_unpromoted_q20_hypothesis",
        "phase3_admission_status": "NOT_EVALUATED_ONE_SEED",
        "result": result_row,
        "charged_seconds": budget.elapsed,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "run_manifest": run_manifest(args, budget.elapsed),
        "source_bindings": source_bindings(
            plan=(
                WIDE_DIAGNOSTIC_PLAN
                if args.hidden_layers == (64, 64)
                else DEEP_DIAGNOSTIC_PLAN
                if args.hidden_layers == (32, 32, 32)
                else SINGLE_DIAGNOSTIC_PLAN
            )
        ),
        "params_path": params_path.relative_to(ROOT).as_posix(),
        "params_sha256": sha256(params_path),
        "nonclaims": [
            "single-seed mechanism diagnostic only",
            "no q=20 hyperparameter nomination or Phase 3 admission",
            "no HMC, posterior-correctness, default-readiness, or scientific claim",
        ],
    }
    write_json(summary_path, payload, replace=args.resume)
    return payload


def load_confirmation_trigger(
    q: int,
    *,
    final_summary_path: Path,
    failed_result_path: Path,
) -> dict[str, Any]:
    summary_path = repo_path(final_summary_path, label="final Phase 3 summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        not isinstance(summary, dict)
        or summary.get("schema") != SCHEMA
        or summary.get("mode") != "final"
        or summary.get("status") != "COMPLETED"
    ):
        raise ComplexityTrainingError("fresh confirmation requires a completed final summary")
    if int(summary.get("q", -1)) != int(q):
        raise ComplexityTrainingError("final Phase 3 summary q mismatch")
    results = summary.get("results")
    if (
        not isinstance(results, list)
        or len(results) != 2
        or not all(isinstance(row, Mapping) for row in results)
    ):
        raise ComplexityTrainingError("final Phase 3 summary must contain two results")
    if summary.get("fresh_confirmation_eligible") is not True:
        raise ComplexityTrainingError("final Phase 3 summary is not confirmation-eligible")
    if {row.get("label") for row in results} != {stream.label for stream in STREAMS}:
        raise ComplexityTrainingError("final Phase 3 summary stream set mismatch")
    status_counts = {
        status: sum(row.get("status") == status for row in results)
        for status in ("ADMITTED", "VETOED")
    }
    if status_counts != {"ADMITTED": 1, "VETOED": 1}:
        raise ComplexityTrainingError(
            "fresh confirmation requires exactly one ADMITTED and one VETOED result"
        )
    bound_results = []
    for summary_row in results:
        result_path = repo_path(
            Path(str(summary_row.get("path", ""))),
            label="result from final summary",
        )
        expected_hash = str(summary_row.get("sha256", ""))
        if not result_path.is_file() or sha256(result_path) != expected_hash:
            raise ComplexityTrainingError("result binding in final summary is invalid")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            not isinstance(result, dict)
            or result.get("schema") != SCHEMA
            or int(result.get("q", -1)) != int(q)
            or result.get("status") != summary_row.get("status")
            or result.get("params") != summary.get("params")
        ):
            raise ComplexityTrainingError("bound final result contract mismatch")
        stream = result.get("stream")
        expected_stream = json.loads(
            canonical(
                next(
                    asdict(row)
                    for row in STREAMS
                    if row.label == summary_row.get("label")
                )
            )
        )
        if (
            not isinstance(stream, Mapping)
            or dict(stream) != expected_stream
        ):
            raise ComplexityTrainingError("bound final result stream mismatch")
        bound_results.append((summary_row, result_path, result))
    _failed_summary_row, expected_failed_path, _failed_result = next(
        row for row in bound_results if row[0].get("status") == "VETOED"
    )
    receipt_path = repo_path(failed_result_path, label="failed Phase 3 result")
    if receipt_path != expected_failed_path:
        raise ComplexityTrainingError("failed result does not match the final summary")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ComplexityTrainingError("failed Phase 3 result schema mismatch")
    if int(receipt.get("q", -1)) != int(q):
        raise ComplexityTrainingError("failed Phase 3 result q mismatch")
    if receipt.get("status") != "VETOED":
        raise ComplexityTrainingError("fresh confirmation requires a VETOED result")
    stream = receipt.get("stream")
    if not isinstance(stream, Mapping) or stream.get("label") not in {
        STREAMS[0].label,
        STREAMS[1].label,
    }:
        raise ComplexityTrainingError(
            "fresh confirmation trigger must be failed seed-a or seed-b"
        )
    return {
        "final_summary_path": summary_path.relative_to(ROOT).as_posix(),
        "final_summary_sha256": sha256(summary_path),
        "final_params": summary.get("params"),
        "failed_result_path": receipt_path.relative_to(ROOT).as_posix(),
        "failed_result_sha256": sha256(receipt_path),
        "failed_stream": dict(stream),
        "failed_vetoes": list(receipt.get("vetoes", [])),
    }


def run_confirmation(args: argparse.Namespace) -> dict[str, Any]:
    assert args.params_json is not None
    assert args.failed_result is not None
    assert args.final_summary is not None
    params_path = repo_path(args.params_json, label="params json")
    params_payload = json.loads(params_path.read_text(encoding="utf-8"))
    if "nominated_params" in params_payload:
        params_payload = params_payload["nominated_params"]
    params = TrialParameters(**params_payload)
    trigger = load_confirmation_trigger(
        args.q,
        final_summary_path=args.final_summary,
        failed_result_path=args.failed_result,
    )
    if trigger["final_params"] != asdict(params):
        raise ComplexityTrainingError(
            "confirmation parameters do not match the completed final run"
        )
    target = complexity_posterior_target(args.q, jit_compile=True)
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    summary_path = output / "confirmation-summary.json"
    prior_seconds = 0.0
    if args.resume:
        if not summary_path.is_file():
            raise ComplexityTrainingError(
                "confirmation resume requires confirmation-summary.json"
            )
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if previous.get("schema") != SCHEMA or int(previous.get("q", -1)) != args.q:
            raise ComplexityTrainingError("confirmation resume summary mismatch")
        if previous.get("params") != asdict(params):
            raise ComplexityTrainingError("confirmation resume parameter mismatch")
        if previous.get("hidden_layers") != list(args.hidden_layers):
            raise ComplexityTrainingError("confirmation resume hidden-layer mismatch")
        if previous.get("confirmation_trigger") != trigger:
            raise ComplexityTrainingError("confirmation resume trigger mismatch")
        if previous.get("status") != "RESOURCE_STOP":
            raise ComplexityTrainingError(
                "only a resource-stopped confirmation can be resumed"
            )
        prior_seconds = float(previous.get("charged_seconds", 0.0))
    budget = Budget(args.gpu_cap_seconds, prior_seconds=prior_seconds)
    stream_dir = output / FRESH_CONFIRMATION.label
    stream_dir.mkdir(parents=True, exist_ok=args.resume)
    result_row = None
    resource_stop = None
    hard_veto = None
    with CPUValueScorePool(pool_config(args.q)) as pool:
        try:
            result = run_final_stream(
                target=target,
                pool=pool,
                stream=FRESH_CONFIRMATION,
                params=params,
                budget=budget,
                output_dir=stream_dir,
                resume=args.resume and (stream_dir / "resource-stop.json").is_file(),
                batch_size=args.batch_size,
                hidden_layers=args.hidden_layers,
            )
        except ResourceStop as exc:
            resource_stop = str(exc)
        except HostMemoryVeto as exc:
            hard_veto = str(exc)
            write_json(
                stream_dir / "host-memory-veto.json",
                {
                    "schema": SCHEMA,
                    "status": "HARD_VETO",
                    "stream": asdict(FRESH_CONFIRMATION),
                    "confirmation_trigger": trigger,
                    "reason": hard_veto,
                    "candidate_veto": False,
                    "continuation_veto": True,
                    "scientific_interpretation": "none",
                },
                replace=args.resume,
            )
        else:
            result["confirmation_trigger"] = trigger
            externalize_payload(
                result,
                key="best_trainer_state",
                path=stream_dir / "best-state.json",
            )
            externalize_payload(
                result,
                key="best_frozen_payload",
                path=stream_dir / "best-frozen-payload.json",
            )
            result_path = stream_dir / "result.json"
            write_json(result_path, result)
            result_row = {
                "label": FRESH_CONFIRMATION.label,
                "path": result_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(result_path),
                "status": result["status"],
            }
    payload = {
        "schema": SCHEMA,
        "mode": "confirmation",
        "status": (
            "HARD_VETO"
            if hard_veto is not None
            else "RESOURCE_STOP"
            if resource_stop is not None
            else "COMPLETED"
        ),
        "q": args.q,
        "params": asdict(params),
        "hidden_layers": list(args.hidden_layers),
        "confirmation_trigger": trigger,
        "result": result_row,
        "charged_seconds": budget.elapsed,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "run_manifest": run_manifest(args, budget.elapsed),
        "source_bindings": source_bindings(),
        "nonclaims": [
            "single prospectively allowed fresh-seed confirmation only",
            "does not authorize another hyperparameter or architecture search",
            "no HMC or posterior correctness claim",
        ],
    }
    write_json(summary_path, payload, replace=args.resume)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "contract-smoke",
            "study",
            "final",
            "confirmation",
            "single-diagnostic",
        ),
        required=True,
    )
    parser.add_argument("--q", type=int, choices=Q_VALUES, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu-cap-seconds", type=float)
    parser.add_argument("--authorize-material-run", action="store_true")
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--sampler-seed", type=int, default=20260719)
    parser.add_argument("--study-name", default="ssl_lstm_neutra_complexity_q_general_v1")
    parser.add_argument("--params-json", type=Path)
    parser.add_argument("--final-summary", type=Path)
    parser.add_argument("--failed-result", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="training draws per optimizer step (default: 480)",
    )
    parser.add_argument(
        "--hidden-layers",
        type=str,
        default="32,32",
        help="comma-separated IAF hidden widths: 32,32; 32,32,32; or 64,64",
    )
    args = parser.parse_args(argv)
    if args.n_trials <= 0:
        parser.error("--n-trials must be positive")
    if args.gpu_cap_seconds is not None and (
        not math.isfinite(args.gpu_cap_seconds) or args.gpu_cap_seconds <= 0.0
    ):
        parser.error("--gpu-cap-seconds must be finite and positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        args.hidden_layers = tuple(
            int(value.strip()) for value in str(args.hidden_layers).split(",")
        )
    except ValueError as exc:
        parser.error("--hidden-layers must be a comma-separated integer tuple")
    if args.hidden_layers not in {(32, 32), (32, 32, 32), (64, 64)}:
        parser.error(
            "--hidden-layers must be exactly 32,32; 32,32,32; or 64,64"
        )
    if args.mode == "contract-smoke" and args.resume:
        parser.error("contract smoke cannot resume")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_material_args(args)
    if args.mode == "contract-smoke":
        payload = contract_payload(args)
    else:
        configure_gpu()
        if args.mode == "study":
            payload = run_study(args)
        elif args.mode == "final":
            payload = run_final(args)
        elif args.mode == "single-diagnostic":
            payload = run_single_diagnostic(args)
        else:
            payload = run_confirmation(args)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "mode": payload["mode"],
                "status": payload["status"],
                "q": payload["q"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
