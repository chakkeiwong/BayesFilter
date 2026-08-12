#!/usr/bin/env python3
"""Continue one q=20 NeuTra seed with GPU updates and CPU/XLA targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _is_contract_mode() -> bool:
    return "--contract-only" in sys.argv


CONTRACT_MODE = _is_contract_mode()
MATERIAL_PARENT = (
    not CONTRACT_MODE and multiprocessing.current_process().name == "MainProcess"
)
if CONTRACT_MODE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif MATERIAL_PARENT and os.environ.get(
    "TF_FORCE_GPU_ALLOW_GROWTH", ""
).strip().lower() != "true":
    raise RuntimeError("material continuation requires TF_FORCE_GPU_ALLOW_GROWTH=true")
elif MATERIAL_PARENT and os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
    raise RuntimeError("material continuation requires CUDA_VISIBLE_DEVICES=1")
elif not MATERIAL_PARENT and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("spawned target workers require CUDA_VISIBLE_DEVICES=-1")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(
    tf, require_gpu=MATERIAL_PARENT
)
if MATERIAL_PARENT:
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)


from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.neutra_training_control import (  # noqa: E402
    validate_joint_training_checkpoint,
)
from bayesfilter.inference.neutra_target_validity_recovery import (  # noqa: E402
    TargetValidityAttempt,
    bounded_target_validity_recovery,
)
from bayesfilter.inference.tf_batch_value_score_pool import (  # noqa: E402
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (  # noqa: E402
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    FREE_NAMES,
    PRIOR_CENTER,
)


LEGACY_SCHEMA = "bayesfilter.ssl_lstm.q20_neutra_budgeted_continuation.v1"
SCHEMA = "bayesfilter.ssl_lstm.q20_neutra_budgeted_continuation.v2"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-budgeted-continuation-plan-2026-08-06.md"
)
RECOVERY_PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-neutra-target-validity-recovery-plan-2026-08-06.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
DEFAULT_OUTPUT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-neutra-budgeted-continuation-2026-08-06/r1"
)
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"
EXPECTED_CHECKPOINTS = {
    "seed-a": {
        "path": Path(
            "docs/plans/artifacts/"
            "ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/"
            "seed-a/seed-a/checkpoint-1500.json"
        ),
        "sha256": "c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff",
        "embedded_step": 1500,
    },
    "seed-b": {
        "path": Path(
            "docs/plans/artifacts/"
            "ssl-lstm-q20-cpu-xla-parallel-training-2026-08-01/r1/"
            "seed-b/seed-b/checkpoint-2500.json"
        ),
        "sha256": "849e33855d87dc34644e15757942bf872937d9f4d4b00a4f03855661827d761d",
        "embedded_step": 2250,
    },
}
SOURCE_PATHS = {
    "runner": SCRIPT,
    "plan": PLAN,
    "recovery_plan": RECOVERY_PLAN,
    "trainer": Path("bayesfilter/inference/neutra_training.py"),
    "target_validity_recovery": Path(
        "bayesfilter/inference/neutra_target_validity_recovery.py"
    ),
    "pool": Path("bayesfilter/inference/tf_batch_value_score_pool.py"),
    "target": Path(
        "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"
    ),
    "memory_policy": Path("bayesfilter/runtime/gpu_memory_policy.py"),
}
TOTAL_UPDATES = 4000
CHECKPOINT_EVERY = 500
TRAIN_BATCH_SIZE = 100
MONITOR_SIZE = 500
SELECTION_SIZE = 500
AUDIT_SIZE = 500
MAX_CAP_SECONDS = 43200.0
HOST_RAM_CAP_BYTES = 64 * 1024**3
MAX_TARGET_VALIDITY_RETRIES = 3
LR_SCHEDULE = ((1, 2000, 2.0e-4), (2001, 3000, 1.0e-4), (3001, 4000, 5.0e-5))
SEEDS = {
    "seed-a": {
        "training": (20260806, 12101),
        "monitor": (20260806, 12102),
        "selection": (20260806, 12103),
        "audit": (20260806, 12104),
    },
    "seed-b": {
        "training": (20260806, 12201),
        "monitor": (20260806, 12202),
        "selection": (20260806, 12203),
        "audit": (20260806, 12204),
    },
}


class CampaignError(RuntimeError):
    pass


class TargetValidityStop(CampaignError):
    def __init__(
        self,
        *,
        request_id: str,
        invalid_rows: Sequence[int],
        z: Any | None,
        theta: Any,
        values: Any,
        scores: Any,
        status: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> None:
        super().__init__(
            f"{request_id} target validity failed for rows {list(invalid_rows)}"
        )
        self.request_id = str(request_id)
        self.invalid_rows = tuple(int(item) for item in invalid_rows)
        self.z = z
        self.theta = theta
        self.values = values
        self.scores = scores
        self.status = status
        self.metadata = metadata

    def artifact_payload(self, *, stream: str) -> Mapping[str, Any]:
        theta = tf.convert_to_tensor(self.theta, tf.float64)
        z = None if self.z is None else tf.convert_to_tensor(self.z, tf.float64)
        rows = []
        for index in range(int(theta.shape[0])):
            row = {
                "row_index": index,
                "z": None if z is None else tensor_json(z[index]),
                "theta": tensor_json(theta[index]),
                "value": tensor_json(tf.convert_to_tensor(self.values)[index])[0],
                "score": tensor_json(tf.convert_to_tensor(self.scores)[index]),
            }
            for key, tensor in self.status.items():
                row[key] = tensor_json(tf.convert_to_tensor(tensor)[index])[0]
            rows.append(row)
        return {
            "schema": f"{SCHEMA}.target_validity_terminal",
            "status": "TARGET_VALIDITY_VETO_CONTROLLED",
            "stream": stream,
            "request_id": self.request_id,
            "invalid_row_indices": list(self.invalid_rows),
            "rows": rows,
            "worker_backend": self.metadata,
            "promotion_vetoes": ["target_validity_failure_observed"],
            "continuation_vetoes": ["nontraining_target_validity_failure"],
            "nonclaims": [
                "controlled terminal artifact only",
                "no candidate nomination, convergence, HMC, posterior, or scientific claim",
            ],
        }


class ResourceStop(CampaignError):
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


def trainer_state_hash(payload: Any) -> str:
    blob = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def migrate_legacy_empty_output_scale_state(
    state: Mapping[str, Any], current_config: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
    """Migrate only the post-checkpoint empty output-scale schema addition."""

    original = dict(state)
    supplied = str(original.pop("state_hash", ""))
    if supplied != trainer_state_hash(original):
        raise CampaignError("legacy trainer state_hash mismatch before migration")
    archived_config = original.get("config")
    if not isinstance(archived_config, Mapping):
        raise CampaignError("legacy trainer state has no config mapping")
    if archived_config == current_config:
        return state, None
    expected = dict(current_config)
    if expected.pop("fixed_output_scale", None) != [] or dict(archived_config) != expected:
        raise CampaignError(
            "trainer config mismatch is not the reviewed empty-output-scale migration"
        )
    migrated_config = dict(archived_config)
    migrated_config["fixed_output_scale"] = []
    migrated = {**original, "config": migrated_config}
    migrated_hash = trainer_state_hash(migrated)
    return {**migrated, "state_hash": migrated_hash}, {
        "schema": "bayesfilter.neutra.trainer_state_compatibility_migration.v1",
        "source_state_hash": supplied,
        "migrated_state_hash": migrated_hash,
        "added_field": "config.fixed_output_scale",
        "added_value": [],
        "numerical_transform_changed": False,
        "historical_checkpoint_modified": False,
    }


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise CampaignError(f"refusing to overwrite artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def read_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, Mapping):
        raise CampaignError(f"artifact is not a mapping: {path}")
    return payload


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def batch(seed: Sequence[int], fold: int, size: int) -> tf.Tensor:
    folded = tf.random.experimental.stateless_fold_in(
        tf.constant(tuple(int(item) for item in seed), tf.int32), int(fold)
    )
    return tf.random.stateless_normal((int(size), 4), folded, dtype=tf.float64)


def tensor_json(value: Any) -> list[Any]:
    """Materialize a tensor for diagnostics without emitting invalid JSON."""

    tensor = tf.convert_to_tensor(value)
    flat = tf.reshape(tensor, [-1]).numpy().tolist()
    if tensor.dtype == tf.bool:
        converted = [bool(item) for item in flat]
    elif tensor.dtype.is_integer:
        converted = [int(item) for item in flat]
    else:
        converted = [float(item) if math.isfinite(float(item)) else None for item in flat]
    return converted


def target_batch_admission(
    values: Any, scores: Any, status: Mapping[str, Any]
) -> tuple[bool, list[int], Mapping[str, int]]:
    """Return full-batch admission and invalid rows; never masks a row."""

    value_tensor = tf.convert_to_tensor(values, tf.float64)
    score_tensor = tf.convert_to_tensor(scores, tf.float64)
    finite = tf.logical_and(
        tf.math.is_finite(value_tensor),
        tf.reduce_all(tf.math.is_finite(score_tensor), axis=1),
    )
    valid = tf.logical_and(
        finite,
        tf.logical_and(
            tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
            tf.convert_to_tensor(status["valid_pre_regularized_score"], tf.bool),
        ),
    )
    invalid_rows = [
        int(index)
        for index, item in enumerate(tf.logical_not(valid).numpy().tolist())
        if bool(item)
    ]
    summary = {
        "row_count": int(value_tensor.shape[0]),
        "invalid_row_count": len(invalid_rows),
        "finite_value_row_count": int(
            tf.reduce_sum(tf.cast(tf.math.is_finite(value_tensor), tf.int32)).numpy()
        ),
        "finite_score_row_count": int(
            tf.reduce_sum(
                tf.cast(
                    tf.reduce_all(tf.math.is_finite(score_tensor), axis=1), tf.int32
                )
            ).numpy()
        ),
    }
    return not invalid_rows, invalid_rows, summary


def target_failure_payload(
    *,
    stream: str,
    continuation_update: int,
    attempt: int,
    seed_fold: int,
    z: Any,
    theta: Any,
    values: Any,
    scores: Any,
    status: Mapping[str, Any],
    metadata: Mapping[str, Any],
    trainer_state_hash: str,
    optimizer_step: int,
    invalid_row_indices: Sequence[int],
) -> Mapping[str, Any]:
    row_count = int(tf.convert_to_tensor(z).shape[0])
    tasks = tuple(metadata.get("worker_tasks", ()))
    rows = []
    for index in range(row_count):
        task = next(
            (
                item
                for item in tasks
                if int(item["item_start"]) <= index < int(item["item_stop"])
            ),
            None,
        )
        row = {"row_index": index, "worker_task": task}
        row["z"] = tensor_json(tf.convert_to_tensor(z)[index])
        row["theta"] = tensor_json(tf.convert_to_tensor(theta)[index])
        row["value"] = tensor_json(tf.convert_to_tensor(values)[index])[0]
        row["score"] = tensor_json(tf.convert_to_tensor(scores)[index])
        for key, tensor in status.items():
            row[key] = tensor_json(tf.convert_to_tensor(tensor)[index])[0]
        rows.append(row)
    return {
        "schema": f"{SCHEMA}.target_validity_failure",
        "stream": stream,
        "continuation_update": int(continuation_update),
        "attempt": int(attempt),
        "seed_fold": int(seed_fold),
        "optimizer_step_before_attempt": int(optimizer_step),
        "trainer_state_hash_before_attempt": str(trainer_state_hash),
        "row_count": row_count,
        "invalid_row_indices": [int(index) for index in invalid_row_indices],
        "rows": rows,
        "worker_backend": metadata,
        "nonclaims": [
            "diagnostic target-validity event only",
            "does not establish the exact UKF time step without upstream trace instrumentation",
        ],
    }


def training_seed_fold(continuation_update: int, attempt: int) -> int:
    update = int(continuation_update)
    retry = int(attempt)
    if not 1 <= update <= TOTAL_UPDATES or not 0 <= retry <= MAX_TARGET_VALIDITY_RETRIES:
        raise ValueError("training update or target-validity attempt is out of range")
    return update + retry * TOTAL_UPDATES


def failure_receipt(path: Path, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "continuation_update": int(payload["continuation_update"]),
        "attempt": int(payload["attempt"]),
        "seed_fold": int(payload["seed_fold"]),
        "invalid_row_indices": list(payload["invalid_row_indices"]),
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
    }


def controlled_target_exhaustion_result(
    *,
    stream: str,
    continuation_update: int,
    trainer: Any,
    failures: Sequence[Mapping[str, Any]],
    trace: Sequence[Mapping[str, Any]],
    checkpoints: Sequence[Mapping[str, Any]],
    placement: Mapping[str, Any],
    wall_seconds: float,
) -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "TARGET_VALIDITY_RECOVERY_EXHAUSTED",
        "stream": stream,
        "failed_continuation_update": int(continuation_update),
        "terminal_optimizer_step": int(trainer.step.numpy()),
        "target_validity_failures": list(failures),
        "training_trace": list(trace),
        "checkpoints": list(checkpoints),
        "promotion_vetoes": ["target_validity_failure_observed"],
        "continuation_vetoes": ["target_validity_recovery_exhausted"],
        "device": device_manifest(),
        "placement": placement,
        "wall_seconds": float(wall_seconds),
        "decision": {
            "primary_criterion_status": "not_met",
            "veto_diagnostic_status": "target_validity_and_continuation_veto_fired",
            "main_uncertainty": "exact upstream UKF failure time step is not yet traced",
            "next_justified_action": "replay the archived exact proposal with UKF time-step instrumentation",
            "not_concluded": [
                "failure of NeuTra as a method",
                "target mathematical invalidity",
                "posterior or HMC validity",
            ],
        },
        "nonclaims": [
            "controlled terminal error-handling result only",
            "no candidate nomination, convergence, HMC, posterior, or scientific claim",
        ],
    }


class Budget:
    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def require(self, reserve: float = 0.0) -> None:
        if self.elapsed + float(reserve) >= self.seconds:
            raise ResourceStop("declared continuation wall cap exhausted")


def learning_rate_for_update(update: int) -> float:
    step = int(update)
    if not 1 <= step <= TOTAL_UPDATES:
        raise ValueError("continuation update is outside the reviewed schedule")
    for first, last, rate in LR_SCHEDULE:
        if first <= step <= last:
            return rate
    raise AssertionError("learning-rate schedule is incomplete")


def allocator_memory() -> Mapping[str, Mapping[str, int]]:
    return {
        device.name: {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info(
                f"GPU:{index}"
            ).items()
        }
        for index, device in enumerate(tf.config.list_logical_devices("GPU"))
    }


def device_manifest() -> Mapping[str, Any]:
    physical = tf.config.list_physical_devices("GPU")
    logical = tf.config.list_logical_devices("GPU")
    growth = {
        device.name: bool(tf.config.experimental.get_memory_growth(device))
        for device in physical
    }
    if MATERIAL_PARENT:
        if len(physical) != 1 or len(logical) != 1:
            raise CampaignError("continuation requires exactly one visible GPU")
        if not growth or not all(growth.values()):
            raise CampaignError("GPU memory growth was not verified")
    return {
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "memory_growth": growth,
        "memory_policy": GPU_MEMORY_POLICY,
        "allocator_memory_bytes": allocator_memory(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "soft_device_placement": bool(tf.config.get_soft_device_placement()),
        "trust_basis": TRUST_BASIS,
    }


def make_pool(worker_count: int, batch_per_worker: int) -> TFBatchValueScorePool:
    affinity = tuple(sorted(os.sched_getaffinity(0)))
    if len(affinity) != int(worker_count):
        raise CampaignError(
            "child affinity must expose exactly one CPU per configured worker"
        )
    return TFBatchValueScorePool(
        TFBatchValueScorePoolConfig(
            factory_path=(
                "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
                "batch_native_complexity_target_worker_factory"
            ),
            factory_config={
                "q": 20,
                "principal_sqrt_backend": "tensorflow_eigh",
                "jit_compile": True,
            },
            dimension=4,
            worker_count=int(worker_count),
            cores_per_worker=1,
            batch_sizes=tuple(range(1, int(batch_per_worker) + 1)),
            batch_per_worker=int(batch_per_worker),
            worker_cpu_ids=affinity,
            timeout_seconds=900.0,
        )
    )


def require_pool_metadata(
    pool: TFBatchValueScorePool,
    metadata: Mapping[str, Any],
    *,
    row_count: int,
) -> Mapping[str, Any]:
    workers = int(pool.config.worker_count)
    startup = tuple(metadata.get("startup_worker_metadata", ()))
    if len(startup) != workers:
        raise CampaignError("worker startup telemetry is incomplete")
    policies = {row.get("evaluation_policy") for row in startup}
    if policies != {"batch_native_tensorflow_status_no_row_mapping_v2"}:
        raise CampaignError("worker evaluation policy is not the status-bearing route")
    assigned = tuple(row.get("assigned_cpu") for row in startup)
    if set(assigned) != set(pool.config.worker_cpu_ids):
        raise CampaignError("worker affinity telemetry is incomplete")
    if int(row_count) == TRAIN_BATCH_SIZE:
        if len(set(metadata.get("worker_result_pids", ()))) != workers:
            raise CampaignError("training batch did not use every persistent worker")
        if metadata.get("worker_shard_sizes") != [
            int(pool.config.batch_per_worker)
        ] * workers:
            raise CampaignError("training batch shard sizes changed")
    worker_pids = sorted({int(value) for value in metadata.get("worker_result_pids", ())})
    if not worker_pids:
        raise CampaignError("worker result PID telemetry is empty")

    def process_hwm_bytes(pid: int) -> int:
        status = Path(f"/proc/{int(pid)}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmHWM:"):
                fields = line.split()
                if len(fields) != 3 or fields[2] != "kB":
                    break
                return int(fields[1]) * 1024
        raise CampaignError(f"worker {pid} has no readable VmHWM telemetry")

    parent_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    unique_worker_rss = {str(pid): process_hwm_bytes(pid) for pid in worker_pids}
    unique_worker_rss_sum = sum(unique_worker_rss.values())
    raw_task_summed_rss = int(metadata.get("active_worker_ru_maxrss_sum_bytes", 0))
    combined_rss = parent_rss + unique_worker_rss_sum
    if combined_rss > HOST_RAM_CAP_BYTES:
        raise CampaignError(
            "parent and worker RSS exceeded 64 GiB: "
            f"combined={combined_rss} parent={parent_rss} "
            f"unique_workers={unique_worker_rss_sum}"
        )
    return {
        **metadata,
        "parent_ru_maxrss_bytes": parent_rss,
        "unique_worker_vm_hwm_bytes_by_pid": unique_worker_rss,
        "unique_worker_vm_hwm_sum_bytes": unique_worker_rss_sum,
        "raw_task_summed_worker_ru_maxrss_bytes": raw_task_summed_rss,
        "raw_task_sum_is_not_process_memory_total": True,
        "combined_ru_maxrss_bytes": combined_rss,
    }


def evaluate(
    pool: TFBatchValueScorePool,
    rows: tf.Tensor,
    *,
    request_id: str,
    diagnostic_z: Any | None = None,
) -> tuple[tf.Tensor, tf.Tensor, Mapping[str, Any]]:
    values, scores, status, metadata = pool.evaluate_with_status(
        rows, request_id=request_id
    )
    metadata = require_pool_metadata(
        pool, metadata, row_count=int(tf.convert_to_tensor(rows).shape[0])
    )
    admitted, invalid_rows, _summary = target_batch_admission(
        values, scores, status
    )
    if not admitted:
        raise TargetValidityStop(
            request_id=request_id,
            invalid_rows=invalid_rows,
            z=diagnostic_z,
            theta=rows,
            values=values,
            scores=scores,
            status=status,
            metadata=metadata,
        )
    tf.debugging.assert_all_finite(values, "external target values")
    tf.debugging.assert_all_finite(scores, "external target scores")
    return values, scores, require_pool_metadata(
        pool, metadata, row_count=int(rows.shape[0])
    )


def config_from_state(target: Any, state: Mapping[str, Any]) -> Any:
    config = state.get("config")
    if not isinstance(config, Mapping):
        raise CampaignError("resume state has no trainer config")
    return ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy().tolist()),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=float(config["learning_rate"]),
        initialization_scale=float(config["initialization_scale"]),
        gradient_clip_norm=float(config["gradient_clip_norm"]),
        initialization_seed=tuple(int(value) for value in config["initialization_seed"]),
        jit_compile=True,
    )


def load_resume(
    stream: str,
    supplied_path: Path,
    target: Any,
) -> tuple[NeuTraReverseKLTrainer, Mapping[str, Any]]:
    expected = EXPECTED_CHECKPOINTS[stream]
    path = supplied_path.resolve()
    if path != (ROOT / expected["path"]).resolve():
        raise CampaignError("resume checkpoint path differs from the reviewed baseline")
    if sha256(path) != expected["sha256"]:
        raise CampaignError("resume checkpoint SHA-256 mismatch")
    checkpoint = read_json(path)
    validate_joint_training_checkpoint(checkpoint)
    state = checkpoint.get("best_trainer_state")
    if not isinstance(state, Mapping):
        raise CampaignError("resume checkpoint has no best_trainer_state")
    if int(state.get("step", -1)) != int(expected["embedded_step"]):
        raise CampaignError("embedded best trainer step differs from reviewed provenance")
    trainer = NeuTraReverseKLTrainer(target, config_from_state(target, state))
    migrated_state, migration = migrate_legacy_empty_output_scale_state(
        state, trainer.config.manifest_payload()
    )
    trainer.restore_state(migrated_state)
    if int(trainer.step.numpy()) != int(expected["embedded_step"]):
        raise CampaignError("restored optimizer iteration mismatch")
    return trainer, {
        "path": expected["path"].as_posix(),
        "sha256": expected["sha256"],
        "joint_checkpoint_hash": checkpoint["checkpoint_hash"],
        "embedded_best_state_hash": state["state_hash"],
        "embedded_optimizer_step": int(state["step"]),
        "historical_container_label_step": int(
            checkpoint.get("controller_state", {}).get("best_step", -1)
        ),
        "compatibility_migration": migration,
    }


def require_gpu_placement(trainer: NeuTraReverseKLTrainer) -> Mapping[str, Any]:
    variable_devices = sorted({str(variable.device) for variable in trainer.variables})
    probe_z = tf.zeros((TRAIN_BATCH_SIZE, 4), tf.float64)
    theta, logdet = trainer.forward_and_logdet(probe_z)
    output_devices = sorted({str(theta.device), str(logdet.device)})
    if not variable_devices or any("GPU:0" not in value.upper() for value in variable_devices):
        raise CampaignError("trainer variables are not all placed on logical GPU 0")
    if any("GPU:0" not in value.upper() for value in output_devices):
        raise CampaignError("transport outputs are not all placed on logical GPU 0")
    return {
        "trainer_variable_devices": variable_devices,
        "representative_output_devices": output_devices,
        "logical_gpu_0_maps_from_cuda_visible_devices": "1",
    }


def host_step(result: Any, *, continuation_update: int) -> Mapping[str, Any]:
    row = {
        "continuation_update": int(continuation_update),
        "optimizer_step": int(result.step.numpy()),
        "loss": float(result.loss.numpy()),
        "surrogate": float(result.surrogate.numpy()),
        "target_value_mean": float(result.target_value_mean.numpy()),
        "logdet_mean": float(result.logdet_mean.numpy()),
        "gradient_norm": float(result.gradient_norm.numpy()),
        "clipped_gradient_norm": float(result.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(result.clipping_applied.numpy()),
    }
    if not all(
        math.isfinite(value)
        for key, value in row.items()
        if key not in {"continuation_update", "optimizer_step", "clipping_applied"}
    ):
        raise CampaignError("nonfinite host training telemetry")
    return row


def validation_payload(
    trainer: NeuTraReverseKLTrainer,
    pool: TFBatchValueScorePool,
    z: tf.Tensor,
    *,
    purpose: str,
    continuation_update: int,
) -> Mapping[str, Any]:
    theta, _ = trainer.forward_and_logdet(z)
    values, _scores, metadata = evaluate(
        pool,
        theta,
        request_id=f"{purpose}-{continuation_update:04d}",
        diagnostic_z=z,
    )
    validation = trainer.validation_batch_with_external_value(z, values)
    losses = [float(value) for value in validation.per_sample_loss.numpy().tolist()]
    scale = [
        float(value)
        for row in validation.scale_log.numpy().tolist()
        for value in row
    ]
    hidden = [
        float(value)
        for row in validation.hidden_preactivations.numpy().tolist()
        for stage in row
        for layer in stage
        for value in layer
    ]
    if not losses or not all(math.isfinite(value) for value in (*losses, *scale, *hidden)):
        raise CampaignError(f"{purpose} diagnostics are nonfinite")
    mean = sum(losses) / len(losses)
    variance = (
        sum((value - mean) ** 2 for value in losses) / (len(losses) - 1)
        if len(losses) > 1
        else 0.0
    )
    return {
        "purpose": purpose,
        "continuation_update": int(continuation_update),
        "optimizer_step": int(trainer.step.numpy()),
        "batch_size": len(losses),
        "mean_loss": mean,
        "loss_standard_error": math.sqrt(variance / len(losses)),
        "per_sample_loss": losses,
        "scale_log_min": min(scale),
        "scale_log_max": max(scale),
        "scale_saturation_fraction": sum(abs(value) >= 0.95 for value in scale)
        / len(scale),
        "hidden_min": min(hidden),
        "hidden_max": max(hidden),
        "worker_backend": metadata,
        "role": (
            "explanatory_telemetry_only_no_training_control"
            if purpose == "monitor"
            else purpose
        ),
    }


def checkpoint_payload(
    trainer: NeuTraReverseKLTrainer,
    *,
    stream: str,
    continuation_update: int,
    resume: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = {
        "schema": f"{SCHEMA}.checkpoint",
        "stream": stream,
        "continuation_update": int(continuation_update),
        "optimizer_step": int(trainer.step.numpy()),
        "learning_rate": float(
            trainer.learning_rate_at(int(trainer.step.numpy())).numpy()
        ),
        "resume": resume,
        "trainer_state": trainer.state_payload(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return {**payload, "checkpoint_hash": hashlib.sha256(canonical(payload)).hexdigest()}


def save_checkpoint(
    trainer: NeuTraReverseKLTrainer,
    output: Path,
    *,
    stream: str,
    continuation_update: int,
    resume: Mapping[str, Any],
) -> Mapping[str, Any]:
    payload = checkpoint_payload(
        trainer,
        stream=stream,
        continuation_update=continuation_update,
        resume=resume,
    )
    path = output / f"checkpoint-{continuation_update:04d}.json"
    write_json(path, payload)
    return {
        "continuation_update": int(continuation_update),
        "optimizer_step": int(trainer.step.numpy()),
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "checkpoint_hash": payload["checkpoint_hash"],
    }


def restore_continuation_checkpoint(
    target: Any, receipt: Mapping[str, Any]
) -> NeuTraReverseKLTrainer:
    path = (ROOT / str(receipt["path"])).resolve()
    if sha256(path) != receipt["sha256"]:
        raise CampaignError("continuation checkpoint SHA-256 mismatch")
    payload = read_json(path)
    raw = dict(payload)
    supplied = str(raw.pop("checkpoint_hash", ""))
    if supplied != hashlib.sha256(canonical(raw)).hexdigest():
        raise CampaignError("continuation checkpoint payload hash mismatch")
    state = payload.get("trainer_state")
    if not isinstance(state, Mapping):
        raise CampaignError("continuation checkpoint has no trainer state")
    trainer = NeuTraReverseKLTrainer(target, config_from_state(target, state))
    trainer.restore_state(state)
    return trainer


def load_continuation_resume(
    target: Any,
    path: Path,
    *,
    stream: str,
    expected_continuation_update: int,
) -> tuple[NeuTraReverseKLTrainer, Mapping[str, Any]]:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT):
        raise CampaignError("continuation checkpoint must be inside the repository")
    payload = read_json(resolved)
    if payload.get("schema") not in {
        f"{LEGACY_SCHEMA}.checkpoint",
        f"{SCHEMA}.checkpoint",
    }:
        raise CampaignError("unsupported continuation checkpoint schema")
    raw = dict(payload)
    supplied = str(raw.pop("checkpoint_hash", ""))
    if supplied != hashlib.sha256(canonical(raw)).hexdigest():
        raise CampaignError("continuation checkpoint payload hash mismatch")
    if payload.get("stream") != stream:
        raise CampaignError("continuation checkpoint stream mismatch")
    if int(payload.get("continuation_update", -1)) != int(expected_continuation_update):
        raise CampaignError("continuation checkpoint update mismatch")
    state = payload.get("trainer_state")
    resume = payload.get("resume")
    if not isinstance(state, Mapping) or not isinstance(resume, Mapping):
        raise CampaignError("continuation checkpoint state or provenance is missing")
    trainer = NeuTraReverseKLTrainer(target, config_from_state(target, state))
    migrated_state, migration = migrate_legacy_empty_output_scale_state(
        state, trainer.config.manifest_payload()
    )
    trainer.restore_state(migrated_state)
    expected_step = int(resume["embedded_optimizer_step"]) + int(
        expected_continuation_update
    )
    if int(trainer.step.numpy()) != expected_step:
        raise CampaignError("continuation checkpoint optimizer step mismatch")
    return trainer, {
        **resume,
        "diagnostic_continuation_checkpoint": {
            "path": resolved.relative_to(ROOT).as_posix(),
            "sha256": sha256(resolved),
            "checkpoint_hash": supplied,
            "continuation_update": int(expected_continuation_update),
            "optimizer_step": int(trainer.step.numpy()),
            "compatibility_migration": migration,
        },
    }


def support_probe(
    trainer: NeuTraReverseKLTrainer,
    target: Any,
    pool: TFBatchValueScorePool,
    *,
    request_id: str,
) -> Mapping[str, Any]:
    frozen = trainer.frozen_transport_payload(
        transport_id=request_id, target_signature=target.target_signature()
    )
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    rows = [tf.zeros((4,), tf.float64)]
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=tf.float64) * 4.0
        rows.extend((direction, -direction))
    z = tf.stack(rows)
    theta = loaded.transport.forward_batch(z)
    values, scores, metadata = evaluate(
        pool, theta, request_id=request_id, diagnostic_z=z
    )
    replay_z = loaded.transport.inverse_theta_to_z_batch(theta)
    replay_theta = loaded.transport.forward_batch(replay_z)
    transformed_score = (
        loaded.transport.pullback_score_batch(z, scores)
        + loaded.transport.log_abs_det_jacobian_score_batch(z)
    )
    tensors = (theta, values, scores, replay_z, replay_theta, transformed_score)
    finite = all(
        bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in tensors
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
    return {
        "all_finite": finite,
        "roundtrip_max_abs": residual,
        "moderate_shell_max_inverse_radius": float(
            tf.reduce_max(tf.linalg.norm(replay_z, axis=-1)).numpy()
        ),
        "transformed_score_max_abs": float(
            tf.reduce_max(tf.abs(transformed_score)).numpy()
        ),
        "worker_backend": metadata,
        "probe_definition": "origin_plus_coordinate_shell_radius_4_in_neutra_z_chart",
    }


def source_manifest() -> Mapping[str, Any]:
    return {
        key: {
            "path": path.as_posix(),
            "sha256": sha256(ROOT / path),
        }
        for key, path in SOURCE_PATHS.items()
    }


def run_manifest(
    args: argparse.Namespace,
    *,
    output: Path,
    resume: Mapping[str, Any],
    placement: Mapping[str, Any],
    started: float,
) -> Mapping[str, Any]:
    return {
        "schema": f"{SCHEMA}.run_manifest",
        "status": "RUNNING",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": [sys.executable, *sys.argv],
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "device": device_manifest(),
        "placement": placement,
        "dtype": "float64",
        "jit_compile": True,
        "xla_required": True,
        "tf32_enabled_but_fp64_numerics_unaffected": True,
        "training_batch_size": TRAIN_BATCH_SIZE,
        "batch_native_target_backend": "persistent_cpu_xla_25x4",
        "target_evaluation_policy": "batch_native_tensorflow_status_no_row_mapping_v2",
        "sample_wise_loop_used": False,
        "scalar_fallback_used": False,
        "row_mapped_scalar_target_used": False,
        "cpu_processes": int(args.cpu_processes),
        "batch_per_process": int(args.batch_per_process),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "stream": args.stream,
        "seeds": SEEDS[args.stream],
        "resume": resume,
        "source_manifest": source_manifest(),
        "plan": PLAN.as_posix(),
        "recovery_plan": RECOVERY_PLAN.as_posix(),
        "output_root": output.relative_to(ROOT).as_posix(),
        "result_path": output.joinpath("result.json").relative_to(ROOT).as_posix(),
        "cap_seconds": float(args.cap_seconds),
        "total_continuation_updates": (
            int(args.canary_updates)
            if args.canary_updates is not None
            else TOTAL_UPDATES
        ),
        "monitor_size": MONITOR_SIZE,
        "selection_size": SELECTION_SIZE,
        "audit_size": AUDIT_SIZE,
        "max_target_validity_retries": MAX_TARGET_VALIDITY_RETRIES,
        "target_invalid_batch_policy": (
            "reject_whole_batch_no_optimizer_update_deterministic_fresh_retry"
        ),
        "monitor_controls_training": False,
        "selection_occurs_after_training": True,
        "audit_is_untouched_until_after_selection": True,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_wall_seconds": time.perf_counter() - started,
        "nonclaims": [
            "training continuation and checkpoint nomination only",
            "no NeuTra convergence, HMC, posterior, superiority, default, or scientific claim",
        ],
    }


def contract_payload() -> Mapping[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "CONTRACT_ONLY",
        "total_updates": TOTAL_UPDATES,
        "checkpoint_every": CHECKPOINT_EVERY,
        "training_batch_size": TRAIN_BATCH_SIZE,
        "monitor_size": MONITOR_SIZE,
        "selection_size": SELECTION_SIZE,
        "audit_size": AUDIT_SIZE,
        "max_target_validity_retries": MAX_TARGET_VALIDITY_RETRIES,
        "monitor_controls_training": False,
        "learning_rate_schedule": [list(row) for row in LR_SCHEDULE],
        "expected_checkpoints": {
            label: {
                **row,
                "path": row["path"].as_posix(),
            }
            for label, row in EXPECTED_CHECKPOINTS.items()
        },
        "plan": PLAN.as_posix(),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.contract_only:
        return contract_payload()
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise CampaignError("output root must be inside the repository")
    if output.exists() and any(output.iterdir()):
        raise CampaignError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    budget = Budget(args.cap_seconds)
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    if args.continuation_checkpoint is None:
        trainer, resume = load_resume(args.stream, args.resume_checkpoint, target)
        start_continuation_update = 0
    else:
        trainer, resume = load_continuation_resume(
            target,
            args.continuation_checkpoint,
            stream=args.stream,
            expected_continuation_update=args.start_continuation_update,
        )
        start_continuation_update = int(args.start_continuation_update)
    trainer.set_learning_rate(learning_rate_for_update(start_continuation_update + 1))
    placement = require_gpu_placement(trainer)
    manifest = run_manifest(
        args,
        output=output,
        resume=resume,
        placement=placement,
        started=started,
    )
    write_json(output / "run-manifest.json", manifest)
    max_updates = (
        start_continuation_update + int(args.canary_updates)
        if args.canary_updates is not None
        else TOTAL_UPDATES
    )
    monitor_z = batch(SEEDS[args.stream]["monitor"], 0, MONITOR_SIZE)
    checkpoints = [
        save_checkpoint(
            trainer,
            output,
            stream=args.stream,
            continuation_update=start_continuation_update,
            resume=resume,
        )
    ]
    monitors: list[Mapping[str, Any]] = []
    trace: list[Mapping[str, Any]] = []
    target_validity_failures: list[Mapping[str, Any]] = []
    last_metadata: Mapping[str, Any] | None = None
    with make_pool(args.cpu_processes, args.batch_per_process) as pool:
        for continuation_update in range(1, max_updates + 1):
            if continuation_update <= start_continuation_update:
                continue
            budget.require(600.0 if args.canary_updates is None else 60.0)
            rate = learning_rate_for_update(continuation_update)
            current = float(
                trainer.learning_rate_at(int(trainer.step.numpy())).numpy()
            )
            if not math.isclose(current, rate, rel_tol=0.0, abs_tol=1.0e-12):
                trainer.set_learning_rate(rate)
            attempt_payloads: dict[int, Mapping[str, Any]] = {}

            def evaluate_training_attempt(attempt: int) -> TargetValidityAttempt:
                nonlocal last_metadata
                seed_fold = training_seed_fold(continuation_update, attempt)
                z = batch(
                    SEEDS[args.stream]["training"], seed_fold, TRAIN_BATCH_SIZE
                )
                theta, _ = trainer.forward_and_logdet(z)
                state_before = trainer.state_payload()
                values, scores, status, raw_metadata = pool.evaluate_with_status(
                    theta,
                    request_id=(
                        f"{args.stream}-train-{continuation_update:04d}"
                        f"-attempt-{attempt}"
                    ),
                )
                last_metadata = require_pool_metadata(
                    pool, raw_metadata, row_count=TRAIN_BATCH_SIZE
                )
                admitted, invalid_rows, admission_summary = target_batch_admission(
                    values, scores, status
                )
                payload = {
                    "seed_fold": seed_fold,
                    "z": z,
                    "theta": theta,
                    "values": values,
                    "scores": scores,
                    "status": status,
                    "metadata": last_metadata,
                    "invalid_rows": invalid_rows,
                    "admission_summary": admission_summary,
                }
                attempt_payloads[attempt] = payload
                return TargetValidityAttempt(
                    admitted=admitted,
                    payload=payload,
                    diagnostic={
                        "invalid_rows": invalid_rows,
                        "admission_summary": admission_summary,
                    },
                )

            def archive_training_rejection(
                attempt: int, attempt_result: TargetValidityAttempt
            ) -> Mapping[str, Any]:
                payload = attempt_payloads[attempt]
                state_before = trainer.state_payload()
                failure = target_failure_payload(
                    stream=args.stream,
                    continuation_update=continuation_update,
                    attempt=attempt,
                    seed_fold=int(payload["seed_fold"]),
                    z=payload["z"],
                    theta=payload["theta"],
                    values=payload["values"],
                    scores=payload["scores"],
                    status=payload["status"],
                    metadata=payload["metadata"],
                    trainer_state_hash=str(state_before["state_hash"]),
                    optimizer_step=int(state_before["step"]),
                    invalid_row_indices=payload["invalid_rows"],
                )
                failure = {
                    **failure,
                    "admission_summary": payload["admission_summary"],
                }
                failure_path = output / (
                    f"target-validity-failure-{continuation_update:04d}"
                    f"-attempt-{attempt}.json"
                )
                write_json(failure_path, failure)
                return failure_receipt(failure_path, failure)

            recovery = bounded_target_validity_recovery(
                max_retries=MAX_TARGET_VALIDITY_RETRIES,
                state_snapshot=trainer.state_payload,
                evaluate_attempt=evaluate_training_attempt,
                archive_rejection=archive_training_rejection,
            )
            target_validity_failures.extend(recovery.rejection_receipts)
            if not recovery.admitted:
                result = controlled_target_exhaustion_result(
                    stream=args.stream,
                    continuation_update=continuation_update,
                    trainer=trainer,
                    failures=target_validity_failures,
                    trace=trace,
                    checkpoints=checkpoints,
                    placement=placement,
                    wall_seconds=budget.elapsed,
                )
                write_json(output / "result.json", result)
                write_json(
                    output / "progress.json",
                    {
                        "schema": SCHEMA,
                        "status": result["status"],
                        "stream": args.stream,
                        "last_completed_continuation_update": continuation_update - 1,
                        "failed_continuation_update": continuation_update,
                        "last_optimizer_step": int(trainer.step.numpy()),
                        "target_validity_failures": target_validity_failures,
                        "elapsed_seconds": budget.elapsed,
                    },
                    replace=(output / "progress.json").exists(),
                )
                final_manifest = {
                    **manifest,
                    "status": result["status"],
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "process_wall_seconds": budget.elapsed,
                    "device_at_finish": device_manifest(),
                    "result_sha256": sha256(output / "result.json"),
                    "target_validity_failures": target_validity_failures,
                }
                write_json(
                    output / "run-manifest.json", final_manifest, replace=True
                )
                return result
            accepted = recovery.payload
            if not isinstance(accepted, Mapping):
                raise CampaignError("target-validity recovery returned no accepted payload")
            z = accepted["z"]
            values = accepted["values"]
            scores = accepted["scores"]
            last_metadata = accepted["metadata"]
            step = trainer.train_step_with_external_value_score(z, values, scores)
            trace.append(host_step(step, continuation_update=continuation_update))
            state = trainer.state_payload()
            if int(state["step"]) != int(resume["embedded_optimizer_step"]) + continuation_update:
                raise CampaignError("optimizer iteration did not advance exactly once")
            if continuation_update % CHECKPOINT_EVERY == 0:
                checkpoints.append(
                    save_checkpoint(
                        trainer,
                        output,
                        stream=args.stream,
                        continuation_update=continuation_update,
                        resume=resume,
                    )
                )
                monitors.append(
                    validation_payload(
                        trainer,
                        pool,
                        monitor_z,
                        purpose="monitor",
                        continuation_update=continuation_update,
                    )
                )
                write_json(
                    output / "progress.json",
                    {
                        "schema": SCHEMA,
                        "status": "RUNNING",
                        "stream": args.stream,
                        "last_continuation_update": continuation_update,
                        "last_optimizer_step": int(trainer.step.numpy()),
                        "elapsed_seconds": budget.elapsed,
                        "checkpoints": checkpoints,
                        "monitors": monitors,
                        "last_training_step": trace[-1],
                        "monitor_controls_training": False,
                        "target_validity_failures": target_validity_failures,
                        "promotion_vetoes": (
                            ["target_validity_failure_observed"]
                            if target_validity_failures
                            else []
                        ),
                    },
                    replace=(output / "progress.json").exists(),
                )

        if args.canary_updates is not None:
            canary_monitor = validation_payload(
                trainer,
                pool,
                monitor_z,
                purpose="monitor",
                continuation_update=max_updates,
            )
            support = support_probe(
                trainer,
                target,
                pool,
                request_id=f"{args.stream}-canary-support",
            )
            result = {
                "schema": SCHEMA,
                "status": "GPU_CONTINUATION_CANARY_COMPLETED",
                "stream": args.stream,
                "continuation_updates": max_updates,
                "terminal_optimizer_step": int(trainer.step.numpy()),
                "trace": trace,
                "monitor": canary_monitor,
                "monitor_controls_training": False,
                "support_probe": support,
                "last_worker_backend": last_metadata,
                "target_validity_failures": target_validity_failures,
                "promotion_vetoes": (
                    ["target_validity_failure_observed"]
                    if target_validity_failures
                    else []
                ),
                "device": device_manifest(),
                "placement": placement,
                "wall_seconds": budget.elapsed,
                "training_quality_eligible": False,
                "nonclaims": [
                    "two-update mechanics canary only",
                    "no training-quality, selection, audit, HMC, or convergence claim",
                ],
            }
            write_json(output / "result.json", result)
        else:
            if len(checkpoints) != 1 + TOTAL_UPDATES // CHECKPOINT_EVERY:
                raise CampaignError("required checkpoint set is incomplete")
            selection_z = batch(
                SEEDS[args.stream]["selection"], 0, SELECTION_SIZE
            )
            selection_rows = []
            for receipt in checkpoints:
                candidate = restore_continuation_checkpoint(target, receipt)
                selection_rows.append(
                    {
                        **validation_payload(
                            candidate,
                            pool,
                            selection_z,
                            purpose="selection",
                            continuation_update=int(receipt["continuation_update"]),
                        ),
                        "checkpoint": receipt,
                    }
                )
            if not all(math.isfinite(float(row["mean_loss"])) for row in selection_rows):
                raise CampaignError("checkpoint selection contains nonfinite loss")
            selected = min(
                selection_rows,
                key=lambda row: (float(row["mean_loss"]), int(row["continuation_update"])),
            )
            selected_trainer = restore_continuation_checkpoint(
                target, selected["checkpoint"]
            )
            audit_z = batch(SEEDS[args.stream]["audit"], 0, AUDIT_SIZE)
            audit = validation_payload(
                selected_trainer,
                pool,
                audit_z,
                purpose="audit",
                continuation_update=int(selected["continuation_update"]),
            )
            support = support_probe(
                selected_trainer,
                target,
                pool,
                request_id=f"{args.stream}-selected-support",
            )
            vetoes = []
            if target_validity_failures:
                vetoes.append("target_validity_failure_observed")
            if not support["all_finite"]:
                vetoes.append("selected_support_nonfinite")
            if float(support["roundtrip_max_abs"]) > 1.0e-9:
                vetoes.append("selected_roundtrip_exceeds_1e-9")
            result = {
                "schema": SCHEMA,
                "status": (
                    "GPU_CONTINUATION_COMPLETED_CANDIDATE_NOMINATED"
                    if not vetoes
                    else "GPU_CONTINUATION_COMPLETED_SELECTION_VETOED"
                ),
                "stream": args.stream,
                "resume": resume,
                "continuation_updates": TOTAL_UPDATES,
                "terminal_optimizer_step": int(trainer.step.numpy()),
                "learning_rate_schedule": [list(row) for row in LR_SCHEDULE],
                "monitor_controls_training": False,
                "training_trace": trace,
                "monitors": monitors,
                "checkpoints": checkpoints,
                "selection": {
                    "batch_size": SELECTION_SIZE,
                    "rows": selection_rows,
                    "selected_continuation_update": int(
                        selected["continuation_update"]
                    ),
                    "selected_optimizer_step": int(selected["optimizer_step"]),
                    "selected_mean_loss": float(selected["mean_loss"]),
                    "selection_rule": "lowest_finite_mean_loss_tie_earlier_checkpoint",
                },
                "audit": audit,
                "support_probe": support,
                "vetoes": vetoes,
                "target_validity_failures": target_validity_failures,
                "last_worker_backend": last_metadata,
                "device": device_manifest(),
                "placement": placement,
                "wall_seconds": budget.elapsed,
                "decision": {
                    "primary_criterion_status": "completed" if not vetoes else "vetoed",
                    "veto_diagnostic_status": "clear" if not vetoes else "fired",
                    "main_uncertainty": "downstream fixed-HMC geometry is not evaluated",
                    "next_justified_action": "fresh per-seed HMC retuning",
                    "not_concluded": [
                        "NeuTra convergence",
                        "posterior correctness",
                        "HMC readiness",
                        "statistical superiority",
                    ],
                },
                "inference_status": {
                    "hard_veto_screen": "support and numerical validity only",
                    "statistically_supported_ranking": "none",
                    "descriptive_only_differences": [
                        "monitor loss",
                        "selection loss",
                        "audit loss",
                        "gradient and runtime telemetry",
                    ],
                    "default_readiness": "not evaluated",
                    "next_evidence_needed": "fresh sequential fixed-HMC validation",
                },
                "nonclaims": [
                    "checkpoint nomination only",
                    "no convergence, HMC, posterior, superiority, default, or scientific claim",
                ],
            }
            write_json(output / "result.json", result)

    final_manifest = {
        **manifest,
        "status": "FINISHED",
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "process_wall_seconds": budget.elapsed,
        "device_at_finish": device_manifest(),
        "result_sha256": sha256(output / "result.json"),
    }
    write_json(output / "run-manifest.json", final_manifest, replace=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stream", choices=tuple(EXPECTED_CHECKPOINTS), required=False)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--continuation-checkpoint", type=Path)
    parser.add_argument("--start-continuation-update", type=int, default=0)
    parser.add_argument("--cpu-processes", type=int, default=25)
    parser.add_argument("--batch-per-process", type=int, default=4)
    parser.add_argument("--cap-seconds", type=float, default=MAX_CAP_SECONDS)
    parser.add_argument("--canary-updates", type=int)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args(argv)
    if args.contract_only:
        payload = run(args)
        print(json.dumps(payload, sort_keys=True))
        return 0
    if args.stream is None or args.resume_checkpoint is None:
        parser.error("material continuation requires --stream and --resume-checkpoint")
    if (args.continuation_checkpoint is None) != (args.start_continuation_update == 0):
        parser.error(
            "--continuation-checkpoint requires a positive "
            "--start-continuation-update and vice versa"
        )
    if not 0 <= args.start_continuation_update < TOTAL_UPDATES:
        parser.error(f"--start-continuation-update must be in [0,{TOTAL_UPDATES - 1}]")
    if not math.isfinite(args.cap_seconds) or not 0.0 < args.cap_seconds <= MAX_CAP_SECONDS:
        parser.error(f"--cap-seconds must be in (0,{MAX_CAP_SECONDS:g}]")
    if args.cpu_processes != 25 or args.batch_per_process != 4:
        parser.error("reviewed material topology is exactly 25 workers x 4 rows")
    if args.cpu_processes * args.batch_per_process != TRAIN_BATCH_SIZE:
        parser.error("worker topology must equal the training batch size")
    if args.canary_updates is not None and not 1 <= args.canary_updates < CHECKPOINT_EVERY:
        parser.error(f"--canary-updates must be in [1,{CHECKPOINT_EVERY - 1}]")
    if (
        args.canary_updates is not None
        and args.start_continuation_update + args.canary_updates > TOTAL_UPDATES
    ):
        parser.error("canary endpoint exceeds the reviewed continuation budget")
    try:
        result = run(args)
    except TargetValidityStop as exc:
        output = (ROOT / args.output_root).resolve()
        event = exc.artifact_payload(stream=args.stream)
        event_path = output / f"target-validity-terminal-{exc.request_id}.json"
        write_json(event_path, event)
        result = {
            **event,
            "terminal_optimizer_step": None,
            "wall_seconds": None,
            "target_validity_event": {
                "path": event_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(event_path),
            },
        }
        write_json(output / "result.json", result)
        manifest_path = output / "run-manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            write_json(
                manifest_path,
                {
                    **manifest,
                    "status": result["status"],
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "result_sha256": sha256(output / "result.json"),
                    "target_validity_event": result["target_validity_event"],
                },
                replace=True,
            )
    print(
        json.dumps(
            {
                "status": result["status"],
                "stream": result["stream"],
                "terminal_optimizer_step": result["terminal_optimizer_step"],
                "wall_seconds": result["wall_seconds"],
            },
            sort_keys=True,
        )
    )
    terminal = str(result["status"])
    return (
        0
        if all(token not in terminal for token in ("VETOED", "EXHAUSTED", "VETO"))
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
