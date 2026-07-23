#!/usr/bin/env python3
"""Run q=20 SSL-LSTM NeuTra training on CPU with batch-native target shards."""

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
)
from bayesfilter.inference.tf_batch_value_score_pool import (
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import FREE_NAMES, PRIOR_CENTER


SCHEMA = "bayesfilter.ssl_lstm.q20_strict_cpu_batch_native_training.v1"
PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-strict-cpu-training-plan-2026-07-22.md")
MAX_STEPS = 2000
CHECK_EVERY = 250
BATCH_SIZE = 100
VALIDATION_SIZE = 64
AUDIT_SIZE = 256
THREAD_LIMIT = 50
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
    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def require(self, reserve: float = 0.0) -> None:
        if self.elapsed + float(reserve) >= self.seconds:
            raise ResourceStop("declared CPU campaign cap exhausted")


class ThreadAudit:
    def __init__(self) -> None:
        self.checks = 0
        self.maximum = 0
        self.snapshot: list[Mapping[str, Any]] = []

    def check(self, worker_pids: Sequence[int]) -> Mapping[str, Any]:
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
        if total > THREAD_LIMIT:
            raise CampaignError(f"process-tree native thread count {total} exceeds {THREAD_LIMIT}")
        return {"total": total, "rows": rows}

    def payload(self) -> Mapping[str, Any]:
        return {"thread_limit": THREAD_LIMIT, "check_count": self.checks, "maximum_process_tree_native_threads": self.maximum, "maximum_snapshot": self.snapshot, "passed": self.checks > 0 and self.maximum <= THREAD_LIMIT}


THREAD_AUDIT = ThreadAudit()


def _pool() -> TFBatchValueScorePool:
    return TFBatchValueScorePool(
        TFBatchValueScorePoolConfig(
            factory_path=("bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:batch_native_complexity_target_worker_factory"),
            factory_config={"q": 20, "principal_sqrt_backend": "tensorflow_eigh"},
            dimension=4,
            worker_count=4,
            cores_per_worker=1,
            batch_sizes=(2, 3, 16, 25, 64),
        )
    )


def _evaluate(
    pool: TFBatchValueScorePool, rows: Any, *, request_id: str
) -> tuple[Any, Any, Mapping[str, Any]]:
    values, scores, metadata = pool.evaluate(rows, request_id=request_id)
    worker_pids = metadata.get("startup_worker_pids", ())
    if len(tuple(worker_pids)) != 4:
        raise CampaignError("batch-native worker PID telemetry is incomplete")
    THREAD_AUDIT.check(worker_pids)
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
        jit_compile=False,
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


def _stream(target: Any, pool: TFBatchValueScorePool, stream: Stream, budget: Budget, output: Path) -> Mapping[str, Any]:
    trainer = _trainer(target, stream)
    controller = NeuTraPlateauController(NeuTraPlateauConfig(validation_check_every=CHECK_EVERY, patience_steps=CHECK_EVERY, max_steps=MAX_STEPS, initial_learning_rate=PARAMETERS["learning_rate"], learning_rate_factor=0.5, post_repair_no_improvement_cycles=2, saturation_repair_enabled=False, roundtrip_max_abs=1.0e-9, moderate_shell_max_inverse_radius=4.30))
    validation_z = _batch(stream.validation_seed, 0, VALIDATION_SIZE)
    theta, _ = trainer.forward_and_logdet(validation_z)
    values, _scores, metadata = _evaluate(pool, theta, request_id=f"{stream.label}-validation-0")
    validation = _host_validation(trainer.validation_batch_with_external_value(validation_z, values), 0, PARAMETERS["learning_rate"])
    probe = _support(trainer, target, pool, f"{stream.label}-support-0")
    state = trainer.state_payload()
    action = controller.observe(step=0, per_sample_loss=validation["per_sample_loss"], saturation_fraction=validation["saturation_fraction"], all_finite=probe["all_finite"], roundtrip_max_abs=probe["roundtrip_max_abs"], moderate_shell_max_inverse_radius=probe["moderate_shell_max_inverse_radius"], trainer_state_hash=state["state_hash"])
    best_state = state
    initial_losses = validation["per_sample_loss"]
    history = [{**validation, "support_probe": probe, "controller_action": action.payload()}]
    checkpoints = []
    for step in range(1, MAX_STEPS + 1):
        budget.require(60.0)
        z = _batch(stream.training_seed, step, BATCH_SIZE)
        theta, _ = trainer.forward_and_logdet(z)
        values, scores, _metadata = _evaluate(pool, theta, request_id=f"{stream.label}-train-{step}")
        trainer.train_step_with_external_value_score(z, values, scores)
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
        write_json(output / "progress.json", {"schema": SCHEMA, "status": "RUNNING", "stream": asdict(stream), "last_program_step": step, "history": history, "checkpoints": checkpoints}, replace=True)
        if action.should_stop:
            break
    if controller.stop_reason is None:
        raise CampaignError("stream ended without a declared stop reason")
    best_trainer = NeuTraReverseKLTrainer(target, trainer.config)
    best_trainer.restore_state(best_state)
    support = _support(best_trainer, target, pool, f"{stream.label}-support-final")
    audit = _audit(best_trainer, target, pool, stream, f"{stream.label}-audit-final")
    paired = _paired_upper(initial_losses, controller.best_per_sample_loss)
    vetoes = []
    if not support["all_finite"] or support["roundtrip_max_abs"] > 1.0e-9 or paired["one_sided_95_upper"] >= 0.0:
        vetoes.append("heldout_or_support_screen_failed")
    result = {"schema": SCHEMA, "q": 20, "status": "CPU_DIAGNOSTIC_SCREEN_PASSED" if not vetoes else "CPU_DIAGNOSTIC_SCREEN_VETOED", "stream": asdict(stream), "params": PARAMETERS, "stop_reason": controller.stop_reason, "best_step": controller.best_step, "terminal_program_step": int(trainer.step.numpy()), "learning_rate_reductions": controller.learning_rate_reductions, "history": history, "checkpoints": checkpoints, "paired_best_minus_initial": paired, "support_probe": support, "audit": audit, "vetoes": vetoes, "execution_eligibility": {"hmc_eligible": False, "transport_promotion_eligible": False, "posterior_claim_eligible": False, "reason": "CPU-only NeuTra diagnostic exception"}}
    write_json(output / "result.json", result)
    return {"label": stream.label, "status": result["status"], "path": output.joinpath("result.json").relative_to(ROOT).as_posix(), "sha256": sha256(output / "result.json"), "best_step": result["best_step"], "terminal_program_step": result["terminal_program_step"], "stop_reason": result["stop_reason"], "audit_mean_loss": audit["mean_loss"], "vetoes": vetoes}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = (ROOT / args.output_root).resolve()
    if output.exists() and any(output.iterdir()):
        raise CampaignError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    target = batch_native_complexity_posterior_target(20, jit_compile=False, principal_sqrt_backend="tensorflow_eigh")
    budget = Budget(args.cap_seconds)
    started = datetime.now(timezone.utc).isoformat()
    results = []
    resource_stop = None
    with _pool() as pool:
        for stream in STREAMS:
            try:
                results.append(_stream(target, pool, stream, budget, output / stream.label))
            except ResourceStop as exc:
                resource_stop = str(exc)
                break
    manifest = {"git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "command": " ".join(sys.argv), "python": sys.version.split()[0], "tensorflow": tf.__version__, "started_at_utc": started, "finished_at_utc": datetime.now(timezone.utc).isoformat(), "wall_seconds": budget.elapsed, "cap_seconds": args.cap_seconds, "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import", "jit_compile": False, "dtype": "float64", "thread_audit": THREAD_AUDIT.payload(), "host_ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024), "host_ram_cap_bytes": HOST_RAM_CAP_BYTES, "output_root": args.output_root.as_posix(), "plan": PLAN.as_posix()}
    payload = {"schema": SCHEMA, "status": "RESOURCE_STOP" if resource_stop else ("CPU_DIAGNOSTIC_COMPLETED" if len(results) == 2 else "INCOMPLETE"), "q": 20, "architecture": [32, 32], "batch_size": BATCH_SIZE, "results": results, "resource_stop": resource_stop, "run_manifest": manifest, "inference_status": {"hard_veto_screen": "thread, memory, finite, support, and artifact checks", "statistically_supported_ranking": "none", "descriptive_only_differences": ["loss", "runtime", "seed differences"], "default_readiness": "ineligible_cpu_diagnostic_exception", "next_evidence_needed": "GPU/XLA claim-bearing training before HMC"}, "nonclaims": ["CPU-only diagnostic/reference exception", "no HMC, posterior correctness, convergence, transport promotion, or scientific-validity claim"]}
    write_json(output / "summary.json", payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=13500.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0 or args.cap_seconds > 13500.0:
        parser.error("--cap-seconds must be in (0, 13500]")
    payload = run(args)
    print(json.dumps({"status": payload["status"], "results": payload["results"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
