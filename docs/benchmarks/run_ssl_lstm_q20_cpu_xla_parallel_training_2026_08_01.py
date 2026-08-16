#!/usr/bin/env python3
"""Run one q=20 SSL-LSTM NeuTra seed on a pinned CPU/XLA worker lane."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("CPU campaign found a visible GPU")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_training import (
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.inference.neutra_training_control import (
    NeuTraPlateauConfig,
    NeuTraPlateauController,
    joint_training_checkpoint_payload,
    validate_joint_training_checkpoint,
)
from bayesfilter.inference.tf_batch_value_score_pool import (
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import FREE_NAMES, PRIOR_CENTER


SCHEMA = "bayesfilter.ssl_lstm.q20_cpu_xla_parallel_training.v1"
PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-parallel-training-plan-2026-08-01.md")
MAX_CAMPAIGN_SECONDS = 40000.0
MAX_STEPS = 4000
CHECK_EVERY = 250
BATCH_SIZE = 100
VALIDATION_SIZE = 64
AUDIT_SIZE = 256
COMPUTE_CORE_LIMIT = 50
HOST_RAM_CAP_BYTES = 64 * 1024**3
PARAMETERS = {"learning_rate": 4.0e-4, "initialization_scale": 0.01, "gradient_clip_norm": 10.0}


@dataclass(frozen=True)
class Stream:
    label: str
    initialization_seed: tuple[int, int]
    training_seed: tuple[int, int]
    validation_seed: tuple[int, int]


STREAMS = (
    Stream("seed-a", (20260719, 12101), (20260719, 13101), (20260719, 14101)),
    Stream("seed-b", (20260719, 12102), (20260719, 13102), (20260719, 14102)),
)


class CampaignError(RuntimeError):
    pass


class ResourceStop(CampaignError):
    pass


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise CampaignError(f"artifact already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Budget:
    def __init__(self, seconds: float, *, prior_seconds: float = 0.0) -> None:
        self.seconds = float(seconds)
        self.prior_seconds = float(prior_seconds)
        if (
            not math.isfinite(self.prior_seconds)
            or self.prior_seconds < 0.0
            or self.prior_seconds >= self.seconds
        ):
            raise ValueError("prior_seconds must be finite and in [0, seconds)")
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self.prior_seconds + time.perf_counter() - self.started

    def require(self, reserve: float = 0.0) -> None:
        if self.elapsed + float(reserve) >= self.seconds:
            raise ResourceStop("declared CPU campaign cap exhausted")


class ThreadAudit:
    def __init__(self) -> None:
        self.checks = 0
        self.maximum = 0
        self.snapshot: list[Mapping[str, Any]] = []

    def check(
        self,
        worker_pids: Sequence[int],
        assigned_cpu_ids: Sequence[int | None],
        configured_worker_count: int,
        cores_per_worker: int,
    ) -> Mapping[str, Any]:
        rows = []
        for pid in (os.getpid(), *tuple(int(value) for value in worker_pids)):
            status = Path(f"/proc/{pid}/status")
            text = status.read_text(encoding="utf-8")
            threads = next(int(line.split(":", 1)[1]) for line in text.splitlines() if line.startswith("Threads:"))
            rows.append({"pid": pid, "threads": threads, "role": "parent" if pid == os.getpid() else "target_worker"})
        total = sum(int(row["threads"]) for row in rows)
        self.checks += 1
        if total > self.maximum:
            self.maximum = total
            self.snapshot = rows
        pinned_cores = {int(value) for value in assigned_cpu_ids if value is not None}
        configured_cores = (
            len(pinned_cores)
            if pinned_cores
            else int(configured_worker_count) * int(cores_per_worker)
        )
        if configured_cores > COMPUTE_CORE_LIMIT:
            raise CampaignError(
                f"configured compute-core count {configured_cores} exceeds {COMPUTE_CORE_LIMIT}"
            )
        return {
            "total_native_threads": total,
            "configured_compute_cores": configured_cores,
            "rows": rows,
        }

    def payload(self) -> Mapping[str, Any]:
        return {
            "compute_core_limit": COMPUTE_CORE_LIMIT,
            "check_count": self.checks,
            "maximum_process_tree_native_threads": self.maximum,
            "maximum_snapshot": self.snapshot,
            "native_threads_are_recorded_not_counted_as_compute_cores": True,
            "passed": self.checks > 0,
        }


THREAD_AUDIT = ThreadAudit()


def _pool(cpu_processes: int, batch_per_process: int | None) -> TFBatchValueScorePool:
    cpu_ids = ()
    batch_sizes = (2, 3, 16, 25, 64)
    if batch_per_process is not None:
        cpu_ids = tuple(sorted(os.sched_getaffinity(0))[: int(cpu_processes)])
        batch_sizes = tuple(range(1, int(batch_per_process) + 1))
    return TFBatchValueScorePool(
        TFBatchValueScorePoolConfig(
            factory_path=("bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:batch_native_complexity_target_worker_factory"),
            factory_config={"q": 20, "principal_sqrt_backend": "tensorflow_eigh", "jit_compile": True},
            dimension=4,
            worker_count=int(cpu_processes),
            cores_per_worker=1,
            batch_sizes=batch_sizes,
            batch_per_worker=batch_per_process,
            worker_cpu_ids=cpu_ids,
        )
    )


def _evaluate(
    pool: TFBatchValueScorePool, rows: Any, *, request_id: str
) -> tuple[Any, Any, Mapping[str, Any]]:
    values, scores, metadata = pool.evaluate(rows, request_id=request_id)
    worker_pids = metadata.get("startup_worker_pids", ())
    if len(tuple(worker_pids)) != int(pool.config.worker_count):
        raise CampaignError("batch-native worker PID telemetry is incomplete")
    assigned_cpu_ids = tuple(
        row.get("assigned_cpu")
        for row in metadata.get("startup_worker_metadata", ())
    )
    expected_full_batch = (
        pool.config.batch_per_worker is not None
        and int(rows.shape[0])
        == int(pool.config.worker_count) * int(pool.config.batch_per_worker)
    )
    if expected_full_batch and len(set(metadata.get("worker_result_pids", ()))) != int(
        pool.config.worker_count
    ):
        raise CampaignError(
            "full training batch did not use every configured persistent worker"
        )
    THREAD_AUDIT.check(
        worker_pids,
        assigned_cpu_ids,
        int(pool.config.worker_count),
        int(pool.config.cores_per_worker),
    )
    parent_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    combined = parent_bytes + max(
        int(metadata.get("active_worker_ru_maxrss_sum_bytes", 0)),
        int(metadata.get("startup_worker_ru_maxrss_sum_bytes", 0)),
    )
    if combined > HOST_RAM_CAP_BYTES:
        raise CampaignError("combined parent/worker RSS exceeded 64 GiB")
    return values, scores, {
        **metadata,
        "parent_ru_maxrss_bytes": parent_bytes,
        "combined_ru_maxrss_bytes": combined,
        "thread_audit": THREAD_AUDIT.payload(),
    }


def _trainer(target: Any, stream: Stream) -> NeuTraReverseKLTrainer:
    config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy().tolist()),
        target_parameter_names=FREE_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=PARAMETERS["learning_rate"],
        initialization_scale=PARAMETERS["initialization_scale"],
        gradient_clip_norm=PARAMETERS["gradient_clip_norm"],
        initialization_seed=stream.initialization_seed,
        jit_compile=True,
    )
    return NeuTraReverseKLTrainer(target, config)


def _batch(seed: tuple[int, int], fold: int, size: int) -> tf.Tensor:
    return tf.random.stateless_normal((int(size), 4), tf.random.experimental.stateless_fold_in(tf.constant(seed, tf.int32), int(fold)), dtype=tf.float64)


def _host_step(result: Any) -> Mapping[str, Any]:
    return {"step": int(result.step.numpy()), "loss": float(result.loss.numpy()), "surrogate": float(result.surrogate.numpy()), "target_value_mean": float(result.target_value_mean.numpy()), "logdet_mean": float(result.logdet_mean.numpy()), "gradient_norm": float(result.gradient_norm.numpy()), "clipped_gradient_norm": float(result.clipped_gradient_norm.numpy()), "clipping_applied": bool(result.clipping_applied.numpy())}


def _host_validation(validation: Any, step: int, learning_rate: float) -> Mapping[str, Any]:
    losses = [float(value) for value in validation.per_sample_loss.numpy().tolist()]
    targets = [float(value) for value in validation.target_value.numpy().tolist()]
    theta = validation.theta.numpy().tolist()
    logdet = [float(value) for value in validation.logdet.numpy().tolist()]
    scale_log = validation.scale_log.numpy().tolist()
    scale_logits = validation.scale_logits.numpy().tolist()
    hidden = validation.hidden_preactivations.numpy().tolist()
    all_scale = [value for row in scale_log for value in row]
    stages = len(all_scale) // (len(scale_log) * 4)
    all_logits = [value for row in scale_logits for stage in row for value in stage]
    all_hidden = [value for row in hidden for stage in row for layer in stage for value in layer]
    return {"step": int(step), "learning_rate": float(learning_rate), "per_sample_loss": losses, "mean_loss": sum(losses) / len(losses), "target_value_mean": sum(targets) / len(targets), "logdet_mean": sum(logdet) / len(logdet), "scale_log_min": min(all_scale), "scale_log_max": max(all_scale), "saturation_fraction": sum(abs(value) >= 0.95 for value in all_scale) / len(all_scale), "saturation_fraction_by_stage": [sum(abs(value) >= 0.95 for value in all_scale) / len(all_scale) for _ in range(stages)], "scale_logit_min": min(all_logits), "scale_logit_max": max(all_logits), "scale_logit_tail_fraction_by_stage": [sum(abs(value) >= math.atanh(0.95) for value in all_logits) / len(all_logits) for _ in range(stages)], "scale_logit_tail_threshold": math.atanh(0.95), "hidden_preactivation_min_by_stage": [min(all_hidden) for _ in range(stages)] if all_hidden else [0.0] * stages, "hidden_preactivation_max_by_stage": [max(all_hidden) for _ in range(stages)] if all_hidden else [0.0] * stages, "hidden_abs_tail_fraction_by_stage": [sum(abs(value) >= 5.0 for value in all_hidden) / len(all_hidden) for _ in range(stages)] if all_hidden else [0.0] * stages, "hidden_negative_tail_fraction_by_stage": [sum(value <= -5.0 for value in all_hidden) / len(all_hidden) for _ in range(stages)] if all_hidden else [0.0] * stages, "hidden_positive_tail_fraction_by_stage": [sum(value >= 5.0 for value in all_hidden) / len(all_hidden) for _ in range(stages)] if all_hidden else [0.0] * stages, "hidden_preactivation_abs_threshold": 5.0, "theta_min_by_coordinate": [min(row[index] for row in theta) for index in range(4)], "theta_max_by_coordinate": [max(row[index] for row in theta) for index in range(4)]}


def _paired_upper(initial: Sequence[float], final: Sequence[float]) -> Mapping[str, float]:
    differences = [float(right) - float(left) for left, right in zip(initial, final, strict=True)]
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    return {"mean_difference": mean, "standard_error": standard_error, "one_sided_95_upper": mean + 1.6694022215079607 * standard_error}


def _support(trainer: NeuTraReverseKLTrainer, target: Any, pool: TFBatchValueScorePool, request: str) -> Mapping[str, Any]:
    frozen = trainer.frozen_transport_payload(transport_id=request, target_signature=target.target_signature())
    loaded = load_frozen_neutra_artifact(frozen, expected_target_signature=target.target_signature())
    transport = loaded.transport
    rows = [tf.zeros((4,), tf.float64)]
    for index in range(4):
        direction = tf.one_hot(index, 4, dtype=tf.float64) * 4.0
        rows.extend((direction, -direction))
    z = tf.stack(rows)
    theta = transport.forward_batch(z)
    values, scores, metadata = _evaluate(pool, theta, request_id=request)
    replay_z = transport.inverse_theta_to_z_batch(theta)
    replay_theta = transport.forward_batch(replay_z)
    transformed_score = transport.pullback_score_batch(z, scores) + transport.log_abs_det_jacobian_score_batch(z)
    finite = all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in (theta, replay_z, replay_theta, transformed_score, values))
    residual = float(tf.reduce_max(tf.concat((tf.reshape(tf.abs(replay_z - z), [-1]), tf.reshape(tf.abs(replay_theta - theta), [-1])), axis=0)).numpy())
    return {"all_finite": finite, "roundtrip_max_abs": residual, "moderate_shell_max_inverse_radius": float(tf.reduce_max(tf.linalg.norm(replay_z, axis=-1)).numpy()), "transformed_score_max_abs": float(tf.reduce_max(tf.abs(transformed_score)).numpy()), "worker_backend": metadata, "probe_definition": "origin_plus_coordinate_shell_radius_4_in_neutra_z_chart"}


def _audit(trainer: NeuTraReverseKLTrainer, target: Any, pool: TFBatchValueScorePool, stream: Stream, request: str) -> Mapping[str, Any]:
    frozen = trainer.frozen_transport_payload(transport_id=request, target_signature=target.target_signature())
    loaded = load_frozen_neutra_artifact(frozen, expected_target_signature=target.target_signature())
    z = _batch(stream.validation_seed, 20260721, AUDIT_SIZE)
    theta = loaded.transport.forward_batch(z)
    logdet = loaded.transport.log_abs_det_jacobian_batch(z)
    values, _scores, metadata = _evaluate(pool, theta, request_id=request)
    losses = -values - logdet
    tf.debugging.assert_all_finite(losses, "audit losses")
    return {"batch_size": AUDIT_SIZE, "mean_loss": float(tf.reduce_mean(losses).numpy()), "per_sample_loss": losses.numpy().tolist(), "worker_backend": metadata, "audit_definition": "stateless_validation_seed_fold_20260721_final_only"}


def _load_resume(
    path: Path,
    *,
    stream: Stream,
    trainer: NeuTraReverseKLTrainer,
    controller: NeuTraPlateauController,
    minimum_prior_seconds: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]], int]:
    checkpoint_path = path.resolve()
    if not checkpoint_path.is_relative_to(ROOT) or not checkpoint_path.is_file():
        raise CampaignError("resume checkpoint must be a repository-local file")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    validate_joint_training_checkpoint(checkpoint)
    progress_path = checkpoint_path.parent / "progress.json"
    if not progress_path.is_file():
        raise CampaignError("resume checkpoint is missing sibling progress.json")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if progress.get("schema") != SCHEMA or progress.get("status") != "RUNNING":
        raise CampaignError("resume progress schema or status is invalid")
    if canonical(progress.get("stream")) != canonical(asdict(stream)):
        raise CampaignError("resume progress stream does not match --stream")
    receipts = list(progress.get("checkpoints", ()))
    if not receipts:
        raise CampaignError("resume progress has no checkpoint receipt")
    receipt = receipts[-1]
    if Path(receipt.get("path", "")).resolve() != checkpoint_path:
        raise CampaignError("resume checkpoint is not the latest progress receipt")
    if receipt.get("sha256") != sha256(checkpoint_path):
        raise CampaignError("resume checkpoint SHA-256 mismatch")
    if receipt.get("checkpoint_hash") != checkpoint.get("checkpoint_hash"):
        raise CampaignError("resume checkpoint hash receipt mismatch")
    prior_seconds = float(progress.get("campaign_elapsed_seconds", -1.0))
    if prior_seconds < 0.0 or float(minimum_prior_seconds) < prior_seconds:
        raise CampaignError(
            "--prior-wall-seconds must cover the elapsed time in resume progress"
        )
    trainer.restore_state(checkpoint["trainer_state"])
    controller.restore_state(checkpoint["controller_state"])
    if controller.status not in {"running", "stopped"}:
        raise CampaignError("resume plateau controller status is invalid")
    best_state = checkpoint.get("best_trainer_state")
    if best_state is None:
        raise CampaignError("resume checkpoint has no eligible best trainer state")
    if best_state.get("state_hash") != controller.best_trainer_state_hash:
        raise CampaignError("resume best state does not match plateau controller")
    history = list(progress.get("history", ()))
    if not history or int(history[0].get("step", -1)) != 0:
        raise CampaignError("resume progress is missing the initial validation")
    program_step = int(progress.get("last_program_step", -1))
    if program_step != int(controller.last_observation_step or -1):
        raise CampaignError("resume program step does not match plateau controller")
    return checkpoint, best_state, history, receipts, program_step


def _stream(
    target: Any,
    pool: TFBatchValueScorePool,
    stream: Stream,
    budget: Budget,
    output: Path,
    *,
    resume_checkpoint: Path | None,
    debug_stop_after_steps: int | None,
) -> Mapping[str, Any]:
    trainer = _trainer(target, stream)
    controller = NeuTraPlateauController(NeuTraPlateauConfig(validation_check_every=CHECK_EVERY, patience_steps=CHECK_EVERY, max_steps=MAX_STEPS, initial_learning_rate=PARAMETERS["learning_rate"], learning_rate_factor=0.5, post_repair_no_improvement_cycles=2, saturation_repair_enabled=False, roundtrip_max_abs=1.0e-9, moderate_shell_max_inverse_radius=4.30))
    validation_z = _batch(stream.validation_seed, 0, VALIDATION_SIZE)
    if resume_checkpoint is None:
        theta, _ = trainer.forward_and_logdet(validation_z)
        values, _scores, _metadata = _evaluate(pool, theta, request_id=f"{stream.label}-validation-0")
        validation = _host_validation(trainer.validation_batch_with_external_value(validation_z, values), 0, PARAMETERS["learning_rate"])
        probe = _support(trainer, target, pool, f"{stream.label}-support-0")
        state = trainer.state_payload()
        action = controller.observe(step=0, per_sample_loss=validation["per_sample_loss"], saturation_fraction=validation["saturation_fraction"], all_finite=probe["all_finite"], roundtrip_max_abs=probe["roundtrip_max_abs"], moderate_shell_max_inverse_radius=probe["moderate_shell_max_inverse_radius"], trainer_state_hash=state["state_hash"])
        best_state = state
        history = [{**validation, "support_probe": probe, "controller_action": action.payload()}]
        checkpoints: list[Mapping[str, Any]] = []
        program_step = 0
    else:
        _checkpoint, best_state, history, checkpoints, program_step = _load_resume(
            resume_checkpoint,
            stream=stream,
            trainer=trainer,
            controller=controller,
            minimum_prior_seconds=budget.prior_seconds,
        )
    initial_losses = history[0]["per_sample_loss"]
    pending_steps = (
        range(program_step + 1, MAX_STEPS + 1)
        if controller.status == "running"
        else ()
    )
    for step in pending_steps:
        # Preserve enough campaign time for the final support and audit batches.
        budget.require(180.0)
        z = _batch(stream.training_seed, step, BATCH_SIZE)
        theta, _ = trainer.forward_and_logdet(z)
        values, scores, _metadata = _evaluate(pool, theta, request_id=f"{stream.label}-train-{step}")
        step_result = trainer.train_step_with_external_value_score(z, values, scores)
        program_step = step
        if debug_stop_after_steps is not None and step >= debug_stop_after_steps:
            support = _support(trainer, target, pool, f"{stream.label}-support-debug-{step}")
            debug_result = {
                "schema": SCHEMA,
                "status": "CPU_DEBUG_SMOKE_COMPLETED",
                "q": 20,
                "stream": asdict(stream),
                "terminal_program_step": step,
                "training_step": _host_step(step_result),
                "trainer_state_hash": trainer.state_payload()["state_hash"],
                "support_probe": support,
                "execution_eligibility": {
                    "training_quality_eligible": False,
                    "hmc_eligible": False,
                    "transport_promotion_eligible": False,
                    "posterior_claim_eligible": False,
                    "reason": "explicit short CPU optimizer-update smoke",
                },
                "nonclaims": [
                    "debug smoke only",
                    "no plateau, heldout audit, HMC, posterior, or promotion claim",
                ],
            }
            write_json(output / "debug-smoke-result.json", debug_result)
            return {
                "label": stream.label,
                "status": debug_result["status"],
                "path": output.joinpath("debug-smoke-result.json").relative_to(ROOT).as_posix(),
                "sha256": sha256(output / "debug-smoke-result.json"),
                "terminal_program_step": step,
                "vetoes": [],
            }
        if step % CHECK_EVERY:
            continue
        theta, _ = trainer.forward_and_logdet(validation_z)
        values, _scores, _metadata = _evaluate(pool, theta, request_id=f"{stream.label}-validation-{step}")
        validation = _host_validation(trainer.validation_batch_with_external_value(validation_z, values), step, float(controller.current_learning_rate))
        probe = _support(trainer, target, pool, f"{stream.label}-support-{step}")
        state = trainer.state_payload()
        action = controller.observe(step=step, per_sample_loss=validation["per_sample_loss"], saturation_fraction=validation["saturation_fraction"], all_finite=probe["all_finite"], roundtrip_max_abs=probe["roundtrip_max_abs"], moderate_shell_max_inverse_radius=probe["moderate_shell_max_inverse_radius"], trainer_state_hash=state["state_hash"])
        if action.meaningful_improvement:
            best_state = state
        if action.should_reduce_learning_rate:
            trainer.restore_state(best_state)
            trainer.set_learning_rate(action.current_learning_rate)
        joint = joint_training_checkpoint_payload(trainer_state=trainer.state_payload(), controller_state=controller.state_payload(), best_trainer_state=best_state)
        checkpoint = output / f"checkpoint-{step:04d}.json"
        write_json(checkpoint, joint)
        checkpoints.append({"step": step, "path": checkpoint.relative_to(ROOT).as_posix(), "sha256": sha256(checkpoint), "checkpoint_hash": joint["checkpoint_hash"]})
        history.append({**validation, "support_probe": probe, "controller_action": action.payload()})
        write_json(output / "progress.json", {"schema": SCHEMA, "status": "RUNNING", "stream": asdict(stream), "last_program_step": step, "campaign_elapsed_seconds": budget.elapsed, "history": history, "checkpoints": checkpoints}, replace=True)
        if action.should_stop:
            break
    if controller.stop_reason is None:
        raise CampaignError("stream ended without a declared stop reason")
    budget.require(120.0)
    best_trainer = NeuTraReverseKLTrainer(target, trainer.config)
    best_trainer.restore_state(best_state)
    support = _support(best_trainer, target, pool, f"{stream.label}-support-final")
    audit = _audit(best_trainer, target, pool, stream, f"{stream.label}-audit-final")
    paired = _paired_upper(initial_losses, controller.best_per_sample_loss)
    vetoes = []
    if not support["all_finite"] or support["roundtrip_max_abs"] > 1.0e-9 or paired["one_sided_95_upper"] >= 0.0:
        vetoes.append("heldout_or_support_screen_failed")
    result = {"schema": SCHEMA, "q": 20, "status": "CPU_XLA_DIAGNOSTIC_SCREEN_PASSED" if not vetoes else "CPU_XLA_DIAGNOSTIC_SCREEN_VETOED", "stream": asdict(stream), "params": PARAMETERS, "stop_reason": controller.stop_reason, "best_step": controller.best_step, "terminal_program_step": program_step, "terminal_optimizer_step": int(trainer.step.numpy()), "learning_rate_reductions": controller.learning_rate_reductions, "history": history, "checkpoints": checkpoints, "paired_best_minus_initial": paired, "support_probe": support, "audit": audit, "vetoes": vetoes, "execution_eligibility": {"hmc_eligible": False, "transport_promotion_eligible": False, "posterior_claim_eligible": False, "reason": "CPU-only XLA NeuTra diagnostic exception"}}
    write_json(output / "result.json", result)
    return {"label": stream.label, "status": result["status"], "path": output.joinpath("result.json").relative_to(ROOT).as_posix(), "sha256": sha256(output / "result.json"), "best_step": result["best_step"], "terminal_program_step": result["terminal_program_step"], "stop_reason": result["stop_reason"], "audit_mean_loss": audit["mean_loss"], "vetoes": vetoes}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if not output.is_relative_to(ROOT):
        raise CampaignError("output root must be inside the repository")
    if args.resume_checkpoint is None:
        if output.exists() and any(output.iterdir()):
            raise CampaignError("fresh output root must be new or empty")
    else:
        checkpoint_path = args.resume_checkpoint.resolve()
        if output != checkpoint_path.parent.parent:
            raise CampaignError(
                "resume output root must be the checkpoint stream directory's parent"
            )
    output.mkdir(parents=True, exist_ok=True)
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    selected_streams = tuple(stream for stream in STREAMS if stream.label == args.stream)
    if len(selected_streams) != 1:
        raise CampaignError("exactly one known stream must be selected")
    budget = Budget(args.cap_seconds, prior_seconds=args.prior_wall_seconds)
    started = datetime.now(timezone.utc).isoformat()
    launch_index = len(tuple(output.glob("launch-attempt-*.json")))
    launch_path = output / f"launch-attempt-{launch_index:03d}.json"
    launch = {
        "schema": "bayesfilter.ssl_lstm.q20_cpu_xla_parallel_training_launch.v1",
        "status": "RUNNING",
        "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
        "command": " ".join(sys.argv),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "started_at_utc": started,
        "cap_seconds": args.cap_seconds,
        "prior_wall_seconds": args.prior_wall_seconds,
        "selected_stream": args.stream,
        "stream_seeds": asdict(selected_streams[0]),
        "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "jit_compile": True,
        "dtype": "float64",
        "target_signature": target.target_signature(),
        "target_adapter_signature": target.adapter_signature(),
        "source_sha256": {
            "plan": sha256(ROOT / PLAN),
            "script": sha256(Path(__file__).resolve()),
            "pool": sha256(ROOT / "bayesfilter/inference/tf_batch_value_score_pool.py"),
            "target": sha256(ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py"),
        },
        "cpu_process_topology": {
            "process_count": args.cpu_processes,
            "training_batch_per_process": args.batch_per_process,
            "training_batch_size": BATCH_SIZE,
        },
        "resume_checkpoint": (
            None if args.resume_checkpoint is None else args.resume_checkpoint.as_posix()
        ),
        "plan": PLAN.as_posix(),
        "output_root": args.output_root.as_posix(),
        "nonclaims": [
            "CPU-only XLA diagnostic/reference exception",
            "one seed is diagnostic only",
            "no HMC or promotion claim",
        ],
    }
    write_json(launch_path, launch)
    results = []
    resource_stop = None
    with _pool(args.cpu_processes, args.batch_per_process) as pool:
        for stream in selected_streams:
            try:
                results.append(_stream(target, pool, stream, budget, output / stream.label, resume_checkpoint=args.resume_checkpoint, debug_stop_after_steps=args.debug_stop_after_steps))
            except ResourceStop as exc:
                resource_stop = str(exc)
                break
    manifest = {**launch, "status": "FINISHED", "launch_attempt_path": launch_path.relative_to(ROOT).as_posix(), "finished_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": budget.elapsed, "thread_audit": THREAD_AUDIT.payload(), "host_ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "host_ram_cap_bytes": HOST_RAM_CAP_BYTES}
    write_json(launch_path, manifest, replace=True)
    completed_status = (
        "CPU_DEBUG_SMOKE_COMPLETED"
        if args.debug_stop_after_steps is not None and len(results) == 1
        else ("CPU_DIAGNOSTIC_COMPLETED" if len(results) == 1 else "INCOMPLETE")
    )
    payload = {"schema": SCHEMA, "status": "RESOURCE_STOP" if resource_stop else completed_status, "q": 20, "architecture": [32, 32], "batch_size": BATCH_SIZE, "selected_stream": args.stream, "results": results, "resource_stop": resource_stop, "run_manifest": manifest, "inference_status": {"hard_veto_screen": "thread, memory, finite, support, and artifact checks", "statistically_supported_ranking": "none", "descriptive_only_differences": ["loss", "runtime"], "default_readiness": "ineligible_cpu_diagnostic_exception", "next_evidence_needed": "independent seed replication and GPU/XLA claim-bearing training before HMC"}, "nonclaims": ["CPU-only XLA diagnostic/reference exception", "one seed is diagnostic only", "no HMC, posterior correctness, convergence, transport promotion, or scientific-validity claim"]}
    write_json(output / f"summary-attempt-{launch_index:03d}.json", payload)
    write_json(output / "summary.json", payload, replace=True)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=31500.0)
    parser.add_argument("--cpu-processes", type=int, default=4)
    parser.add_argument("--batch-per-process", type=int)
    parser.add_argument("--stream", choices=tuple(stream.label for stream in STREAMS), required=True)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--prior-wall-seconds", type=float, default=0.0)
    parser.add_argument("--debug-stop-after-steps", type=int)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0 or args.cap_seconds > MAX_CAMPAIGN_SECONDS:
        parser.error(f"--cap-seconds must be in (0, {MAX_CAMPAIGN_SECONDS:g}]")
    if not math.isfinite(args.prior_wall_seconds) or args.prior_wall_seconds < 0.0:
        parser.error("--prior-wall-seconds must be finite and nonnegative")
    if args.resume_checkpoint is None and args.prior_wall_seconds != 0.0:
        parser.error("--prior-wall-seconds requires --resume-checkpoint")
    if args.resume_checkpoint is not None and args.prior_wall_seconds <= 0.0:
        parser.error("--resume-checkpoint requires positive --prior-wall-seconds")
    if args.debug_stop_after_steps is not None:
        if args.debug_stop_after_steps <= 0 or args.debug_stop_after_steps >= CHECK_EVERY:
            parser.error(
                f"--debug-stop-after-steps must be in [1, {CHECK_EVERY - 1}]"
            )
        if args.resume_checkpoint is not None:
            parser.error("debug smoke cannot resume a campaign checkpoint")
    if args.cpu_processes <= 0 or args.cpu_processes > COMPUTE_CORE_LIMIT:
        parser.error(f"--cpu-processes must be in [1, {COMPUTE_CORE_LIMIT}]")
    if args.cpu_processes > len(os.sched_getaffinity(0)):
        parser.error("--cpu-processes exceeds CPUs available to the current process")
    if args.batch_per_process is not None:
        if args.batch_per_process <= 0:
            parser.error("--batch-per-process must be positive")
        if args.cpu_processes * args.batch_per_process != BATCH_SIZE:
            parser.error(
                "--cpu-processes times --batch-per-process must equal the training batch size"
            )
    payload = run(args)
    print(json.dumps({"status": payload["status"], "results": payload["results"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
