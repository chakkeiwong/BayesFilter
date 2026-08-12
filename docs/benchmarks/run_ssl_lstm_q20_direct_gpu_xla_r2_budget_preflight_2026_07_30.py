#!/usr/bin/env python3
"""Hardware-identity and receipt-heavy timing preflight for q=20 NeuTra."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _mode_from_argv() -> str:
    if "--mode" not in sys.argv:
        return "cpu-identity"
    index = sys.argv.index("--mode")
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else "cpu-identity"


MODE_AT_IMPORT = _mode_from_argv()
GPU_MODES = {"gpu-identity", "timing"}
if MODE_AT_IMPORT not in GPU_MODES:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise RuntimeError("GPU preflight modes require TF_FORCE_GPU_ALLOW_GROWTH=true")
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {None, "", "-1"}:
        raise RuntimeError("GPU preflight modes require an explicit visible GPU")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)


GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf,
    require_gpu=MODE_AT_IMPORT in GPU_MODES,
)
if MODE_AT_IMPORT in GPU_MODES:
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)


from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_batching import (  # noqa: E402
    bound_batch_native_neutra_training_target,
    require_batch_native_neutra_target,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
    ssl_lstm_wide_capacity_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (  # noqa: E402
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    FREE_NAMES,
    PRIOR_CENTER,
)


SCHEMA = "bayesfilter.ssl_lstm.q20_direct_gpu_xla_r2_budget_preflight.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-plan-2026-07-30.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
SOURCE_PATHS = {
    "runner": SCRIPT,
    "plan": PLAN,
    "batch_target": Path(
        "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
    ),
    "scalar_target": Path(
        "bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"
    ),
    "binding": Path("bayesfilter/inference/neutra_batching.py"),
    "trainer": Path("bayesfilter/inference/neutra_training.py"),
    "artifacts": Path("bayesfilter/inference/neutra_artifacts.py"),
    "memory_policy": Path("bayesfilter/runtime/gpu_memory_policy.py"),
}
SOURCE_SHA256 = {
    key: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    for key, path in SOURCE_PATHS.items()
}
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-direct-gpu-xla-r2-budget-preflight-2026-07-30/r1"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
MATERIAL_CAP_SECONDS = 12000.0
AUTHORIZED_REMAINING_SECONDS = 13184.690720226998
RESERVED_NONMATERIAL_SECONDS = AUTHORIZED_REMAINING_SECONDS - MATERIAL_CAP_SECONDS
BATCH_SIZE = 100
VALIDATION_SIZE = 64
AUDIT_SIZE = 256
WARM_UPDATE_COUNT = 5
MINIMUM_WARM_UPDATE_COUNT = 3
ROUNDTRIP_MAX = 1.0e-9
SHELL_RADIUS_MAX = 4.30
CONTINGENCY_FACTOR = 1.25
IDENTITY_FIELDS = (
    "target_signature",
    "adapter_signature",
    "signature_payload_sha256",
    "signature_payload",
)


class PreflightError(RuntimeError):
    pass


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
        raise PreflightError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, Mapping):
        raise PreflightError(f"artifact is not a mapping: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def device_payload() -> Mapping[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    growth = {
        device.name: bool(tf.config.experimental.get_memory_growth(device))
        for device in physical
    }
    if MODE_AT_IMPORT in GPU_MODES:
        if not logical:
            raise PreflightError("trusted GPU preflight has no logical GPU")
        if not growth or not all(growth.values()):
            raise PreflightError("GPU memory growth was not verified")
    allocator = {
        device.name: {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info(
                f"GPU:{index}"
            ).items()
        }
        for index, device in enumerate(logical)
    }
    return {
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "memory_growth": growth,
        "memory_policy": GPU_MEMORY_POLICY,
        "allocator_memory_bytes": allocator,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        "trust_basis": TRUST_BASIS,
    }


def run_manifest(args: argparse.Namespace) -> Mapping[str, Any]:
    return {
        "schema": f"{SCHEMA}.run_manifest",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "device": device_payload(),
        "dtype": "float64",
        "jit_compile": True,
        "batch_size": BATCH_SIZE,
        "source_paths": {key: value.as_posix() for key, value in SOURCE_PATHS.items()},
        "source_sha256": SOURCE_SHA256,
        "plan": PLAN.as_posix(),
        "output_root": args.output_root.as_posix(),
        "mode": args.mode,
        "architecture": args.architecture,
        "authorized_remaining_seconds": AUTHORIZED_REMAINING_SECONDS,
        "material_cap_seconds": MATERIAL_CAP_SECONDS,
        "reserved_nonmaterial_seconds": RESERVED_NONMATERIAL_SECONDS,
    }


class MaterialBudget:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root
        self.started = time.perf_counter()
        ledger_path = ROOT / output_root / "material-budget-ledger.json"
        if ledger_path.exists():
            ledger = read_json(ledger_path)
            if ledger.get("schema") != f"{SCHEMA}.material_budget_ledger":
                raise PreflightError("material budget ledger schema mismatch")
            self.prior_seconds = float(ledger.get("charged_seconds", -1.0))
            self.attempts = list(ledger.get("attempts", ()))
        else:
            self.prior_seconds = 0.0
            self.attempts = []
        if not 0.0 <= self.prior_seconds < MATERIAL_CAP_SECONDS:
            raise PreflightError("preflight material budget is exhausted or invalid")

    @property
    def elapsed(self) -> float:
        return self.prior_seconds + time.perf_counter() - self.started

    def require(self, reserve_seconds: float = 600.0) -> None:
        if self.elapsed + float(reserve_seconds) >= MATERIAL_CAP_SECONDS:
            self.persist("RESOURCE_STOP")
            raise PreflightError("preflight material cap exhausted")

    def persist(self, status: str) -> None:
        process_seconds = time.perf_counter() - self.started
        payload = {
            "schema": f"{SCHEMA}.material_budget_ledger",
            "status": "RUNNING" if status == "RUNNING" else str(status),
            "cap_seconds": MATERIAL_CAP_SECONDS,
            "charged_seconds": self.prior_seconds + process_seconds,
            "remaining_seconds": MATERIAL_CAP_SECONDS
            - self.prior_seconds
            - process_seconds,
            "attempts": [
                *self.attempts,
                {
                    "mode": MODE_AT_IMPORT,
                    "command": " ".join(sys.argv),
                    "process_seconds": process_seconds,
                    "status": str(status),
                },
            ],
        }
        write_json(
            ROOT / self.output_root / "material-budget-ledger.json",
            payload,
            replace=True,
        )


def target_identity() -> Mapping[str, Any]:
    target = batch_native_complexity_posterior_target(
        20,
        jit_compile=True,
        principal_sqrt_backend="compiled_custom_op",
    )
    signature_payload = target.config.signature_payload()
    return {
        "schema": f"{SCHEMA}.target_identity",
        "q": 20,
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
        "signature_payload": signature_payload,
        "signature_payload_sha256": hashlib.sha256(
            canonical(signature_payload)
        ).hexdigest(),
        "batch_target_signature_payload": target.signature_payload(),
        "fixture_device": target.config.fixture.device,
        "observations_device": target.config.observations.device,
        "source_sha256": SOURCE_SHA256,
        "tensorflow": tf.__version__,
        "device": device_payload(),
        "nonclaims": [
            "target construction and identity only",
            "no filter, optimizer, training-quality, HMC, or posterior claim",
        ],
    }


def issue_identity(args: argparse.Namespace) -> Mapping[str, Any]:
    payload = {
        **target_identity(),
        "status": "TARGET_IDENTITY_ISSUED",
        "mode": args.mode,
        "run_manifest": run_manifest(args),
    }
    filename = "cpu-identity.json" if args.mode == "cpu-identity" else "gpu-identity.json"
    write_json(ROOT / args.output_root / filename, payload)
    return payload


def compare_identity(args: argparse.Namespace) -> Mapping[str, Any]:
    cpu_path = ROOT / args.output_root / "cpu-identity.json"
    gpu_path = ROOT / args.output_root / "gpu-identity.json"
    cpu = read_json(cpu_path)
    gpu = read_json(gpu_path)
    comparisons = {field: cpu.get(field) == gpu.get(field) for field in IDENTITY_FIELDS}
    source_match = cpu.get("source_sha256") == gpu.get("source_sha256") == SOURCE_SHA256
    devices_are_cpu = all(
        "CPU:0" in str(payload.get(field))
        for payload in (cpu, gpu)
        for field in ("fixture_device", "observations_device")
    )
    passed = all(comparisons.values()) and source_match and devices_are_cpu
    payload = {
        "schema": f"{SCHEMA}.identity_comparison",
        "status": "TARGET_IDENTITY_PARITY_PASSED" if passed else "TARGET_IDENTITY_PARITY_VETOED",
        "comparisons": comparisons,
        "source_match": source_match,
        "static_data_devices_are_cpu": devices_are_cpu,
        "cpu_path": cpu_path.relative_to(ROOT).as_posix(),
        "cpu_sha256": sha256(cpu_path),
        "gpu_path": gpu_path.relative_to(ROOT).as_posix(),
        "gpu_sha256": sha256(gpu_path),
        "target_signature": cpu.get("target_signature") if passed else None,
        "adapter_signature": cpu.get("adapter_signature") if passed else None,
        "nonclaims": [
            "identity parity only",
            "no filter, optimizer, training-quality, HMC, or posterior claim",
        ],
    }
    write_json(ROOT / args.output_root / "identity-comparison.json", payload)
    if not passed:
        raise PreflightError("CPU/GPU target identity parity failed")
    return payload


def stateless_batch(seed: tuple[int, int], fold: int, size: int) -> tf.Tensor:
    folded = tf.random.experimental.stateless_fold_in(
        tf.constant(seed, tf.int32), int(fold)
    )
    return tf.random.stateless_normal([int(size), 4], folded, dtype=tf.float64)


def bound_target() -> tuple[Any, Any, Any]:
    owner = batch_native_complexity_posterior_target(
        20,
        jit_compile=True,
        principal_sqrt_backend="compiled_custom_op",
    )
    binding = require_batch_native_neutra_target(
        owner,
        target_signature=owner.target_signature(),
        batch_size=BATCH_SIZE,
    )
    return owner, binding, bound_batch_native_neutra_training_target(binding)


def make_trainer(target: Any, architecture: tuple[int, int]) -> NeuTraReverseKLTrainer:
    factory = (
        ssl_lstm_tuned_capacity_neutra_config
        if architecture == (32, 32)
        else ssl_lstm_wide_capacity_neutra_config
    )
    config = factory(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=2.0e-4,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260730, 8101 if architecture == (32, 32) else 8102),
        jit_compile=True,
    )
    return NeuTraReverseKLTrainer(target, config)


def allocator_payload() -> Mapping[str, Mapping[str, int]]:
    return {
        device.name: {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info(
                f"GPU:{index}"
            ).items()
        }
        for index, device in enumerate(tf.config.list_logical_devices("GPU"))
    }


def tensor_finite(value: Any) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy())


def step_payload(step: Any) -> Mapping[str, Any]:
    return {
        "step": int(step.step.numpy()),
        "loss": float(step.loss.numpy()),
        "gradient_norm": float(step.gradient_norm.numpy()),
        "clipped_gradient_norm": float(step.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(step.clipping_applied.numpy()),
    }


def validation_result(trainer: NeuTraReverseKLTrainer, target: Any, z: tf.Tensor) -> Mapping[str, Any]:
    validation = trainer.validation_batch(z)
    if not tensor_finite(validation.per_sample_loss):
        raise PreflightError("validation loss is nonfinite")
    _value, _score, status = target.batch_value_score_status(validation.theta)
    hard_valid = bool(tf.reduce_all(status["hard_valid_for_training"]).numpy())
    if not hard_valid:
        raise PreflightError("validation target status failed")
    return {
        "row_count": int(z.shape[0]),
        "mean_loss": float(tf.reduce_mean(validation.per_sample_loss).numpy()),
        "all_finite": True,
        "all_target_status_valid": hard_valid,
        "floor_count_total": int(tf.reduce_sum(status["floor_count_value"]).numpy()),
        "min_innovation_eigenvalue": float(
            tf.reduce_min(status["min_innovation_eigenvalue"]).numpy()
        ),
    }


def support_result(trainer: NeuTraReverseKLTrainer, target: Any, label: str) -> Mapping[str, Any]:
    frozen = trainer.frozen_transport_payload(
        transport_id=f"q20-r2-preflight-{label}",
        target_signature=target.target_signature(),
    )
    loaded = load_frozen_neutra_artifact(
        frozen,
        expected_target_signature=target.target_signature(),
    )
    rows = [tf.zeros([4], tf.float64)]
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=tf.float64) * 4.0
        rows.extend((direction, -direction))
    z = tf.stack(rows)
    theta = loaded.transport.forward_batch(z)
    value, score, status = target.batch_value_score_status(theta)
    replay_z = loaded.transport.inverse_theta_to_z_batch(theta)
    replay_theta = loaded.transport.forward_batch(replay_z)
    transformed_score = (
        loaded.transport.pullback_score_batch(z, score)
        + loaded.transport.log_abs_det_jacobian_score_batch(z)
    )
    residual = float(
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
    radius = float(tf.reduce_max(tf.linalg.norm(replay_z, axis=-1)).numpy())
    finite = all(
        tensor_finite(item)
        for item in (theta, value, score, replay_z, replay_theta, transformed_score)
    )
    hard_valid = bool(tf.reduce_all(status["hard_valid_for_training"]).numpy())
    if not finite or not hard_valid or residual > ROUNDTRIP_MAX or radius > SHELL_RADIUS_MAX:
        raise PreflightError("support probe failed")
    return {
        "all_finite": finite,
        "all_target_status_valid": hard_valid,
        "roundtrip_max_abs": residual,
        "moderate_shell_max_inverse_radius": radius,
        "frozen_payload_sha256": hashlib.sha256(canonical(frozen)).hexdigest(),
    }


class ProgressRecorder:
    def __init__(self, path: Path, base: Mapping[str, Any], budget: MaterialBudget) -> None:
        self.path = path
        self.payload = {
            **dict(base),
            "schema": f"{SCHEMA}.timing_progress",
            "status": "RUNNING",
            "active_operation": None,
            "operations": [],
        }
        self.budget = budget
        self._write()

    def _write(self) -> None:
        self.payload["process_elapsed_seconds"] = time.perf_counter() - self.budget.started
        self.payload["allocator_memory_bytes"] = allocator_payload()
        write_json(self.path, self.payload, replace=True)
        self.budget.persist("RUNNING")

    def run(self, name: str, operation: Callable[[], Any]) -> Any:
        self.budget.require()
        started_wall = time.time()
        started = time.perf_counter()
        self.payload["active_operation"] = {
            "name": name,
            "started_unix_seconds": started_wall,
        }
        self._write()
        result = operation()
        duration = time.perf_counter() - started
        self.payload["operations"].append(
            {
                "name": name,
                "duration_seconds": duration,
                "completed_unix_seconds": time.time(),
                "result": result,
            }
        )
        self.payload["active_operation"] = None
        self._write()
        return result

    def complete(self) -> Mapping[str, Any]:
        self.payload["status"] = "TIMING_DIAGNOSTIC_COMPLETED"
        self.payload["active_operation"] = None
        self._write()
        return self.payload


def run_timing(args: argparse.Namespace, budget: MaterialBudget) -> Mapping[str, Any]:
    architecture = (32, 32) if args.architecture == "32x32" else (64, 64)
    identity = read_json(ROOT / args.output_root / "identity-comparison.json")
    if identity.get("status") != "TARGET_IDENTITY_PARITY_PASSED":
        raise PreflightError("timing requires passing identity parity")
    arm_root = ROOT / args.output_root / "timing" / args.architecture
    progress = ProgressRecorder(
        arm_root / "progress.json",
        {
            "architecture": list(architecture),
            "batch_size": BATCH_SIZE,
            "learning_rate": 2.0e-4,
            "source_sha256": SOURCE_SHA256,
            "identity_comparison_sha256": sha256(
                ROOT / args.output_root / "identity-comparison.json"
            ),
            "run_manifest": run_manifest(args),
        },
        budget,
    )
    state: dict[str, Any] = {}

    def construct_target() -> Mapping[str, Any]:
        owner, binding, target = bound_target()
        if owner.target_signature() != identity.get("target_signature"):
            raise PreflightError("timing target signature differs from identity gate")
        state.update(owner=owner, binding=binding, target=target)
        return {
            "target_signature": owner.target_signature(),
            "adapter_signature": owner.adapter_signature(),
            "binding": binding.payload(),
        }

    progress.run("target_and_binding_construction", construct_target)

    def construct_trainer() -> Mapping[str, Any]:
        trainer = make_trainer(state["target"], architecture)
        state["trainer"] = trainer
        return {
            "variable_devices": sorted({variable.device for variable in trainer.variables}),
            "trainable_variable_count": len(trainer.variables),
            "trainer_state_hash": trainer.state_payload()["state_hash"],
        }

    progress.run("trainer_construction", construct_trainer)
    trainer = state["trainer"]
    target = state["target"]
    validation_z = stateless_batch((20260730, 8201), 0, VALIDATION_SIZE)
    audit_z = stateless_batch((20260730, 8301), 0, AUDIT_SIZE)
    progress.run(
        "validation_64_first",
        lambda: validation_result(trainer, target, validation_z),
    )
    for step_index in range(1, WARM_UPDATE_COUNT + 2):
        z = stateless_batch((20260730, 8401), step_index, BATCH_SIZE)
        progress.run(
            f"optimizer_update_{step_index}",
            lambda z=z: step_payload(trainer.train_step(z)),
        )
    progress.run(
        "validation_64_warm",
        lambda: validation_result(trainer, target, validation_z),
    )
    status_z = stateless_batch((20260730, 8501), 0, 2)

    def status_probe() -> Mapping[str, Any]:
        theta, _logdet = trainer.forward_and_logdet(status_z)
        _value, _score, status = target.batch_value_score_status(theta)
        valid = bool(tf.reduce_all(status["hard_valid_for_training"]).numpy())
        if not valid:
            raise PreflightError("two-row status probe failed")
        return {
            "row_count": 2,
            "all_hard_valid": valid,
            "floor_count_total": int(tf.reduce_sum(status["floor_count_value"]).numpy()),
        }

    progress.run("status_probe_2", status_probe)
    progress.run("support_export_first", lambda: support_result(trainer, target, "first"))
    progress.run("support_export_warm", lambda: support_result(trainer, target, "warm"))
    progress.run(
        "audit_shape_256_first",
        lambda: validation_result(trainer, target, audit_z),
    )
    progress.run(
        "audit_shape_256_warm",
        lambda: validation_result(trainer, target, audit_z),
    )

    def hlo_receipt() -> Mapping[str, Any]:
        z = stateless_batch((20260730, 8601), 0, BATCH_SIZE)
        hlo = trainer._compiled_train_step.experimental_get_compiler_ir(z)(stage="hlo")
        encoded = hlo if isinstance(hlo, bytes) else str(hlo).encode("utf-8")
        return {"hlo_sha256": hashlib.sha256(encoded).hexdigest()}

    progress.run("hlo_extraction", hlo_receipt)
    complete = progress.complete()
    operation_path = arm_root / "result.json"
    result = {
        **complete,
        "status": "TIMING_DIAGNOSTIC_COMPLETED",
        "progress_path": progress.path.relative_to(ROOT).as_posix(),
        "progress_sha256": sha256(progress.path),
        "nonclaims": [
            "timing and mechanics only",
            "no tuning selection, candidate rejection, HMC, posterior, or default claim",
        ],
    }
    write_json(operation_path, result)
    return result


def operation_durations(result: Mapping[str, Any]) -> Mapping[str, float]:
    return {
        str(row["name"]): float(row["duration_seconds"])
        for row in result.get("operations", ())
    }


def timing_summary(result: Mapping[str, Any]) -> Mapping[str, Any]:
    durations = operation_durations(result)
    warm_updates = [
        durations[f"optimizer_update_{index}"]
        for index in range(2, WARM_UPDATE_COUNT + 2)
        if f"optimizer_update_{index}" in durations
    ]
    if len(warm_updates) < MINIMUM_WARM_UPDATE_COUNT:
        raise PreflightError("insufficient warm optimizer timings")
    return {
        "construction_seconds": durations["target_and_binding_construction"]
        + durations["trainer_construction"],
        "validation_64_first_seconds": durations["validation_64_first"],
        "validation_64_warm_seconds": durations["validation_64_warm"],
        "optimizer_update_first_seconds": durations["optimizer_update_1"],
        "optimizer_update_warm_seconds": warm_updates,
        "optimizer_update_warm_median_seconds": statistics.median(warm_updates),
        "optimizer_update_warm_max_seconds": max(warm_updates),
        "optimizer_update_warm_min_seconds": min(warm_updates),
        "status_probe_2_seconds": durations["status_probe_2"],
        "support_export_first_seconds": durations["support_export_first"],
        "support_export_warm_seconds": durations["support_export_warm"],
        "audit_shape_256_first_seconds": durations["audit_shape_256_first"],
        "audit_shape_256_warm_seconds": durations["audit_shape_256_warm"],
        "hlo_extraction_seconds": durations["hlo_extraction"],
        "process_elapsed_seconds": float(result["process_elapsed_seconds"]),
    }


def projected_process_cost(
    timing: Mapping[str, Any],
    *,
    updates: int,
    validation_calls: int,
    support_calls: int,
    audit_calls: int,
    use_max: bool,
) -> float:
    warm_update = float(
        timing[
            "optimizer_update_warm_max_seconds"
            if use_max
            else "optimizer_update_warm_median_seconds"
        ]
    )
    return (
        float(timing["construction_seconds"])
        + float(timing["validation_64_first_seconds"])
        + float(timing["optimizer_update_first_seconds"])
        + max(0, int(updates) - 1) * warm_update
        + max(0, int(validation_calls) - 1)
        * float(timing["validation_64_warm_seconds"])
        + (
            float(timing["support_export_first_seconds"])
            if support_calls
            else 0.0
        )
        + max(0, int(support_calls) - 1)
        * float(timing["support_export_warm_seconds"])
        + (
            float(timing["audit_shape_256_first_seconds"])
            if audit_calls
            else 0.0
        )
        + max(0, int(audit_calls) - 1)
        * float(timing["audit_shape_256_warm_seconds"])
        + float(timing["hlo_extraction_seconds"])
    )


def project_budget(args: argparse.Namespace) -> Mapping[str, Any]:
    identity = read_json(ROOT / args.output_root / "identity-comparison.json")
    if identity.get("status") != "TARGET_IDENTITY_PARITY_PASSED":
        raise PreflightError("projection requires passing identity parity")
    results = {
        label: read_json(ROOT / args.output_root / "timing" / label / "result.json")
        for label in ("32x32", "64x64")
    }
    summaries = {label: timing_summary(result) for label, result in results.items()}

    def scenario(selected_final: str, use_max: bool) -> float:
        tuning = sum(
            projected_process_cost(
                summaries[label],
                updates=100,
                validation_calls=3,
                support_calls=1,
                audit_calls=0,
                use_max=use_max,
            )
            * 2.0
            for label in ("32x32", "64x64")
        )
        finals = projected_process_cost(
            summaries[selected_final],
            updates=1000,
            validation_calls=11,
            support_calls=12,
            audit_calls=2,
            use_max=use_max,
        ) * 2.0
        return tuning + finals

    scenarios = {}
    for selected in ("32x32", "64x64"):
        median_total = scenario(selected, False)
        max_total = scenario(selected, True)
        scenarios[selected] = {
            "unbuffered_warm_median_seconds": median_total,
            "unbuffered_warm_max_seconds": max_total,
            "buffered_warm_median_seconds": median_total * CONTINGENCY_FACTOR,
            "buffered_warm_max_seconds": max_total * CONTINGENCY_FACTOR,
        }
    recommended = max(
        row["buffered_warm_max_seconds"] for row in scenarios.values()
    )
    payload = {
        "schema": f"{SCHEMA}.projection",
        "status": "BUDGET_PREFLIGHT_COMPLETED",
        "identity": identity,
        "timing": summaries,
        "projection_protocol": {
            "tuning_arms": 4,
            "tuning_updates_per_arm": 100,
            "tuning_validation_calls_per_arm": 3,
            "final_streams": 2,
            "final_updates_per_stream": 1000,
            "final_validation_calls_per_stream": 11,
            "final_support_calls_per_stream": 12,
            "final_audit_calls_per_stream": 2,
            "contingency_factor": CONTINGENCY_FACTOR,
        },
        "scenarios_by_selected_final_architecture": scenarios,
        "conservative_requested_campaign_seconds": recommended,
        "conservative_requested_campaign_hours": recommended / 3600.0,
        "numeric_provenance": {
            "timing_values": "measured in r2 preflight",
            "protocol_counts": "inherited r1 full protocol for pricing only",
            "contingency_factor": "convenience planning margin",
        },
        "nonclaims": [
            "compute estimate only; no campaign authorization",
            "no tuning selection, training-quality, convergence, HMC, posterior, or default claim",
        ],
    }
    write_json(ROOT / args.output_root / "projection.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("cpu-identity", "gpu-identity", "compare-identity", "timing", "project"),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--architecture", choices=("32x32", "64x64"), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.is_absolute():
        raise PreflightError("output root must be repository-relative")
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise PreflightError("output root escapes repository")
    if args.mode == "timing" and args.architecture is None:
        raise PreflightError("timing mode requires --architecture")
    budget = MaterialBudget(args.output_root) if args.mode in GPU_MODES else None
    try:
        if args.mode in {"cpu-identity", "gpu-identity"}:
            result = issue_identity(args)
        elif args.mode == "compare-identity":
            result = compare_identity(args)
        elif args.mode == "timing":
            assert budget is not None
            result = run_timing(args, budget)
        else:
            result = project_budget(args)
    except Exception:
        if budget is not None:
            budget.persist("FAILED_ATTEMPT")
        raise
    if budget is not None:
        budget.persist(str(result["status"]))
    print(json.dumps({"mode": args.mode, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
