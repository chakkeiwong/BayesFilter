#!/usr/bin/env python3
"""Run both q=20 NeuTra streams as a strict CPU-only diagnostic."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PARENT_INTRA_THREADS = 4
PARENT_INTER_THREADS = 1
WORKER_COUNT = 8
THREAD_LIMIT = 50
IS_POOL_WORKER_IMPORT = os.environ.get("BAYESFILTER_CPU_VALUE_SCORE_WORKER") == "1"
if not IS_POOL_WORKER_IMPORT:
    os.environ.update(
        {
            "CUDA_VISIBLE_DEVICES": "-1",
            "BAYESFILTER_CPU_VALUE_SCORE_WORKER": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "TF_NUM_INTRAOP_THREADS": str(PARENT_INTRA_THREADS),
            "TF_NUM_INTEROP_THREADS": str(PARENT_INTER_THREADS),
            "TF_CPP_MIN_LOG_LEVEL": "2",
        }
    )


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as _tensorflow

if not IS_POOL_WORKER_IMPORT:
    _tensorflow.config.threading.set_intra_op_parallelism_threads(
        PARENT_INTRA_THREADS
    )
    _tensorflow.config.threading.set_inter_op_parallelism_threads(
        PARENT_INTER_THREADS
    )

TRAINING_SCRIPT = (
    ROOT
    / "docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py"
)
SPEC = importlib.util.spec_from_file_location(
    "ssl_lstm_neutra_complexity_training_cpu_execution", TRAINING_SCRIPT
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the existing q=20 training implementation")
training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training
SPEC.loader.exec_module(training)
tf = training.tf


SCHEMA = "bayesfilter.ssl_lstm.q20_strict_cpu_training.v1"
PLAN = Path(
    "docs/plans/bayesfilter-ssl-lstm-q20-strict-cpu-training-plan-2026-07-22.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
PARAMS = training.TrialParameters(4.0e-4, 0.01, 10.0)
BATCH_SIZE = 100
HIDDEN_LAYERS = (32, 32)
HOST_RAM_CAP_BYTES = 64 * 1024**3


class StrictCPUTrainingError(RuntimeError):
    """Raised when the strict CPU diagnostic contract fails."""


def _canonical(payload: Any) -> bytes:
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _thread_count(pid: int) -> int | None:
    path = Path(f"/proc/{int(pid)}/status")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, PermissionError):
        return None
    for line in lines:
        if line.startswith("Threads:"):
            return int(line.split(":", 1)[1].strip())
    return None


class ThreadAudit:
    def __init__(self) -> None:
        self.check_count = 0
        self.maximum_total = 0
        self.maximum_rows: list[Mapping[str, Any]] = []

    def check(self, worker_pids: Sequence[int]) -> Mapping[str, Any]:
        rows = []
        for pid in (os.getpid(), *tuple(int(item) for item in worker_pids)):
            count = _thread_count(pid)
            if count is None:
                raise StrictCPUTrainingError(f"cannot audit process threads for pid {pid}")
            rows.append(
                {
                    "pid": pid,
                    "role": "parent" if pid == os.getpid() else "target_worker",
                    "threads": count,
                }
            )
        total = sum(int(row["threads"]) for row in rows)
        self.check_count += 1
        if total > self.maximum_total:
            self.maximum_total = total
            self.maximum_rows = rows
        if total > THREAD_LIMIT:
            raise StrictCPUTrainingError(
                f"process-tree native thread count {total} exceeds limit {THREAD_LIMIT}"
            )
        return {"total": total, "rows": rows}

    def payload(self) -> Mapping[str, Any]:
        return {
            "thread_limit": THREAD_LIMIT,
            "check_count": self.check_count,
            "maximum_process_tree_native_threads": self.maximum_total,
            "maximum_snapshot": self.maximum_rows,
            "passed": self.check_count > 0 and self.maximum_total <= THREAD_LIMIT,
        }


THREAD_AUDIT = ThreadAudit()
ORIGINAL_ENFORCE_HOST_MEMORY = training._enforce_host_memory
ORIGINAL_TRAINER_CONFIG = training.trainer_config


def _strict_enforce_host_memory(metadata: Mapping[str, Any]) -> int:
    combined = int(ORIGINAL_ENFORCE_HOST_MEMORY(metadata))
    THREAD_AUDIT.check(())
    return combined


def _strict_trainer_config(
    target: Any,
    stream: Any,
    params: Any,
    hidden_layers: tuple[int, ...] = HIDDEN_LAYERS,
) -> Any:
    config = ORIGINAL_TRAINER_CONFIG(target, stream, params, hidden_layers)
    return dataclasses.replace(config, jit_compile=False)


training._enforce_host_memory = _strict_enforce_host_memory
training.trainer_config = _strict_trainer_config


from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (  # noqa: E402
    batch_native_complexity_posterior_target,
)
from bayesfilter.inference.tf_batch_value_score_pool import (  # noqa: E402
    TFBatchValueScorePool,
    TFBatchValueScorePoolConfig,
)


class _BatchNativeBoundary:
    """Direct TensorFlow target boundary; no process pool or host row loop."""

    def __init__(self, target: Any) -> None:
        self.batch_native_target = target
        self._pool = TFBatchValueScorePool(
            TFBatchValueScorePoolConfig(
                factory_path=(
                    "bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:"
                    "batch_native_complexity_target_worker_factory"
                ),
                factory_config={"q": 20, "principal_sqrt_backend": "tensorflow_eigh"},
                dimension=4,
                worker_count=4,
                cores_per_worker=1,
                batch_sizes=(2, 3, 16, 25, 64),
            )
        )
        self._metadata = {
            "backend": "persistent_cpu_batch_native_tensorflow_pool",
            "evaluation_mode": "batch_native",
            "worker_backend": "batch_native_value_score",
            "configured_worker_count": 0,
            "compiled_batch_sizes": [],
            "jit_compile": False,
            "cuda_visible_devices": "-1",
            "tensorflow_gpu_devices": [],
            "startup_worker_pids": [],
            "active_worker_ru_maxrss_sum_bytes": 0,
            "startup_worker_ru_maxrss_sum_bytes": 0,
        }

    def __enter__(self) -> "_BatchNativeBoundary":
        self._pool.__enter__()
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self._pool.__exit__(_type, _value, _traceback)

    def metadata(self, *, request_id: str, mode: str) -> Mapping[str, Any]:
        parent_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        if parent_bytes > HOST_RAM_CAP_BYTES:
            raise training.HostMemoryVeto("parent RSS exceeded 64 GiB")
        thread_snapshot = THREAD_AUDIT.check(())
        return {
            **self._metadata,
            "request_id": str(request_id),
            "mode": str(mode),
            "parent_ru_maxrss_bytes": parent_bytes,
            "thread_snapshot": thread_snapshot,
        }

    def batch_value_and_score(self, rows: Any) -> tuple[Any, Any]:
        values, scores, metadata = self._pool.evaluate(
            rows, request_id="batch-native-training"
        )
        self._last_metadata = metadata
        return values, scores

    def _current_metadata(self, *, request_id: str, mode: str) -> Mapping[str, Any]:
        current = dict(self._last_metadata) if hasattr(self, "_last_metadata") else {}
        current.update(self.metadata(request_id=request_id, mode=mode))
        return current


def _batch_native_training_step(
    trainer: Any,
    boundary: _BatchNativeBoundary,
    z: Any,
    *,
    request_id: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    theta, _ = trainer.forward_and_logdet(z)
    values, scores = boundary.batch_value_and_score(tf.stop_gradient(theta))
    result = trainer.train_step_with_external_value_score(z, values, scores)
    return training._host_step(result), boundary._current_metadata(request_id=request_id, mode="value_score")


def _batch_native_validation(
    trainer: Any,
    boundary: _BatchNativeBoundary,
    z: Any,
    *,
    step: int,
    request_id: str,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    theta, _ = trainer.forward_and_logdet(z)
    values, _scores = boundary.batch_value_and_score(tf.stop_gradient(theta))
    validation = trainer.validation_batch_with_external_value(z, values)
    return (
        _batch_native_host_validation(
            validation,
            step=step,
            learning_rate=float(trainer.learning_rate_at(step).numpy()),
        ),
        boundary._current_metadata(request_id=request_id, mode="value_only"),
    )


def _batch_native_host_validation(
    validation: Any, *, step: int, learning_rate: float
) -> dict[str, Any]:
    """Materialize tensors only at the artifact/reporting boundary."""

    rows = {
        "per_sample_loss": validation.per_sample_loss.numpy().tolist(),
        "target_value": validation.target_value.numpy().tolist(),
        "theta": validation.theta.numpy().tolist(),
        "logdet": validation.logdet.numpy().tolist(),
        "scale_log": validation.scale_log.numpy().tolist(),
        "scale_logits": validation.scale_logits.numpy().tolist(),
        "hidden_preactivations": validation.hidden_preactivations.numpy().tolist(),
    }
    flat = [float(value) for value in rows["per_sample_loss"]]
    target_values = [float(value) for value in rows["target_value"]]
    theta = rows["theta"]
    logdet = [float(value) for value in rows["logdet"]]
    scale_log = rows["scale_log"]
    scale_logits = rows["scale_logits"]
    hidden = rows["hidden_preactivations"]
    if not all(math.isfinite(value) for value in flat + target_values + logdet):
        raise FloatingPointError("validation returned nonfinite values")
    stages = len(scale_log[0]) // len(theta[0])
    stage_scale = [
        [row[index * len(theta[0]) : (index + 1) * len(theta[0])] for index in range(stages)]
        for row in scale_log
    ]
    raw_threshold = math.atanh(0.95)
    all_scale = [value for row in scale_log for value in row]
    all_logits = [value for row in scale_logits for stage in row for value in stage]
    all_hidden = [value for row in hidden for stage in row for layer in stage for value in layer]
    return {
        "step": int(step),
        "learning_rate": float(learning_rate),
        "per_sample_loss": flat,
        "mean_loss": sum(flat) / len(flat),
        "target_value_mean": sum(target_values) / len(target_values),
        "logdet_mean": sum(logdet) / len(logdet),
        "scale_log_min": min(all_scale),
        "scale_log_max": max(all_scale),
        "saturation_fraction": sum(abs(value) >= 0.95 for value in all_scale) / len(all_scale),
        "saturation_fraction_by_stage": [
            sum(abs(row[index]) >= 0.95 for row in stage_scale for _ in [0]) / len(stage_scale)
            for index in range(stages)
        ],
        "scale_logit_min": min(all_logits),
        "scale_logit_max": max(all_logits),
        "scale_logit_tail_fraction_by_stage": [
            sum(abs(stage[index]) >= raw_threshold for row in scale_logits for stage in row)
            / len(scale_logits)
            for index in range(stages)
        ],
        "scale_logit_tail_threshold": raw_threshold,
        "hidden_preactivation_min_by_stage": [
            min(layer for row in hidden for stage in row for layer_values in stage for layer in layer_values)
            if hidden
            else 0.0
            for _ in range(stages)
        ],
        "hidden_preactivation_max_by_stage": [
            max(layer for row in hidden for stage in row for layer_values in stage for layer in layer_values)
            if hidden
            else 0.0
            for _ in range(stages)
        ],
        "hidden_abs_tail_fraction_by_stage": [
            sum(abs(value) >= 5.0 for value in all_hidden) / len(all_hidden)
            if all_hidden
            else 0.0
            for _ in range(stages)
        ],
        "hidden_negative_tail_fraction_by_stage": [
            sum(value <= -5.0 for value in all_hidden) / len(all_hidden)
            if all_hidden
            else 0.0
            for _ in range(stages)
        ],
        "hidden_positive_tail_fraction_by_stage": [
            sum(value >= 5.0 for value in all_hidden) / len(all_hidden)
            if all_hidden
            else 0.0
            for _ in range(stages)
        ],
        "hidden_preactivation_abs_threshold": 5.0,
        "theta_min_by_coordinate": [min(row[index] for row in theta) for index in range(len(theta[0]))],
        "theta_max_by_coordinate": [max(row[index] for row in theta) for index in range(len(theta[0]))],
    }


def _batch_native_paired_upper(initial: Sequence[float], final: Sequence[float]) -> dict[str, float]:
    differences = [float(right) - float(left) for left, right in zip(initial, final, strict=True)]
    mean = sum(differences) / len(differences)
    variance = sum((value - mean) ** 2 for value in differences) / (len(differences) - 1)
    standard_error = math.sqrt(variance / len(differences))
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "one_sided_95_upper": mean + 1.6694022215079607 * standard_error,
    }


def _batch_native_support_probe(
    transport: Any,
    boundary: _BatchNativeBoundary,
    *,
    request_id: str,
) -> dict[str, Any]:
    z = training._latent_shell()
    theta = transport.forward_batch(z)
    values, scores = boundary.batch_value_and_score(tf.stop_gradient(theta))
    replay_z = transport.inverse_theta_to_z_batch(theta)
    replay_theta = transport.forward_batch(replay_z)
    transformed_score = transport.pullback_score_batch(z, scores) + transport.log_abs_det_jacobian_score_batch(z)
    tensors = (theta, replay_z, replay_theta, transformed_score)
    all_finite = bool(
        all(tf.reduce_all(tf.math.is_finite(row)).numpy() for row in tensors)
        and tf.reduce_all(tf.math.is_finite(values)).numpy()
    )
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
        "all_finite": all_finite,
        "roundtrip_max_abs": roundtrip,
        "moderate_shell_max_inverse_radius": float(
            tf.reduce_max(tf.linalg.norm(replay_z, axis=-1)).numpy()
        ),
        "transformed_score_max_abs": float(tf.reduce_max(tf.abs(transformed_score)).numpy()),
        "worker_backend": boundary._current_metadata(request_id=request_id, mode="value_score"),
        "probe_definition": "origin_plus_coordinate_shell_radius_4_in_neutra_z_chart",
    }


def _batch_native_audit(
    transport: Any,
    boundary: _BatchNativeBoundary,
    z: Any,
    *,
    request_id: str,
) -> dict[str, Any]:
    theta = transport.forward_batch(z)
    logdet = transport.log_abs_det_jacobian_batch(z)
    values, _scores = boundary.batch_value_and_score(tf.stop_gradient(theta))
    losses = -values - logdet
    tf.debugging.assert_all_finite(losses, "batch-native audit loss")
    return {
        "batch_size": int(z.shape[0]),
        "mean_loss": float(tf.reduce_mean(losses).numpy()),
        "per_sample_loss": losses.numpy().tolist(),
        "worker_backend": boundary._current_metadata(request_id=request_id, mode="value_only"),
        "audit_definition": "stateless_validation_seed_fold_20260721_final_only",
    }


def _install_batch_native_route() -> None:
    """Replace the scalar CPU bridge used by the shared stream procedure."""

    training._external_training_step = _batch_native_training_step
    training._external_validation = _batch_native_validation
    training._paired_upper = _batch_native_paired_upper
    training.support_probe = _batch_native_support_probe
    def trainer_probe(
        trainer: Any,
        target: Any,
        pool: _BatchNativeBoundary,
        *,
        request_id: str,
    ) -> dict[str, Any]:
        payload = trainer.frozen_transport_payload(
            transport_id=f"{request_id}-checkpoint-probe",
            target_signature=target.target_signature(),
        )
        loaded = training.load_frozen_neutra_artifact(
            payload, expected_target_signature=target.target_signature()
        )
        return _batch_native_support_probe(
            loaded.transport, pool, request_id=request_id
        )

    training.trainer_support_probe = trainer_probe
    training._external_transport_audit = _batch_native_audit


def _repo_output(path: Path) -> Path:
    output = (ROOT / path).resolve()
    if not output.is_relative_to(ROOT):
        raise StrictCPUTrainingError("output root must remain inside the repository")
    return output


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise StrictCPUTrainingError(f"output already exists: {path}")
    path.write_bytes(_canonical(payload))


def _externalize_result(result: dict[str, Any], stream_dir: Path) -> Mapping[str, Any]:
    computed_status = str(result["status"])
    result["computed_training_screen_status"] = computed_status
    result["status"] = (
        "CPU_DIAGNOSTIC_SCREEN_PASSED"
        if computed_status == "ADMITTED"
        else "CPU_DIAGNOSTIC_SCREEN_VETOED"
    )
    result["execution_eligibility"] = {
        "training_backend": "strict_cpu_non_xla_diagnostic_exception",
        "hmc_eligible": False,
        "transport_promotion_eligible": False,
        "posterior_claim_eligible": False,
        "reason": "CPU-only NeuTra policy exception is diagnostic/reference only",
    }
    result["thread_audit"] = THREAD_AUDIT.payload()
    training.externalize_payload(
        result,
        key="best_trainer_state",
        path=stream_dir / "best-state.json",
    )
    training.externalize_payload(
        result,
        key="best_frozen_payload",
        path=stream_dir / "best-frozen-payload-diagnostic-only.json",
    )
    result_path = stream_dir / "result.json"
    training.write_json(result_path, result)
    return {
        "label": str(result["stream"]["label"]),
        "status": result["status"],
        "computed_training_screen_status": computed_status,
        "path": result_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(result_path),
        "best_step": result["best_step"],
        "terminal_program_step": result["terminal_program_step"],
        "stop_reason": result["stop_reason"],
        "audit_mean_loss": result["audit"]["mean_loss"],
        "vetoes": result["vetoes"],
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise StrictCPUTrainingError("CUDA must be hidden for strict CPU training")
    if tf.config.list_physical_devices("GPU"):
        raise StrictCPUTrainingError("strict CPU training found a visible GPU")
    affinity = sorted(int(item) for item in os.sched_getaffinity(0))
    if len(affinity) > THREAD_LIMIT:
        raise StrictCPUTrainingError("CPU affinity exceeds the strict thread limit")
    output = _repo_output(args.output_root)
    if output.exists() and any(output.iterdir()):
        raise StrictCPUTrainingError("output root must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    budget = training.Budget(args.cap_seconds)
    target = batch_native_complexity_posterior_target(
        20, jit_compile=False, principal_sqrt_backend="tensorflow_eigh"
    )
    _install_batch_native_route()
    results = []
    resource_stop = None
    hard_veto = None
    current_stream = None
    training.reset_training_interruption()
    training.install_training_signal_handlers()
    try:
        with _BatchNativeBoundary(target) as pool:
            for stream in training.STREAMS:
                current_stream = stream.label
                stream_dir = output / stream.label
                stream_dir.mkdir(parents=True, exist_ok=False)
                try:
                    result = training.run_final_stream(
                        target=target,
                        pool=pool,
                        stream=stream,
                        params=PARAMS,
                        budget=budget,
                        output_dir=stream_dir,
                        resume=False,
                        batch_size=BATCH_SIZE,
                        hidden_layers=HIDDEN_LAYERS,
                        saturation_repair_enabled=False,
                    )
                except training.ResourceStop as exc:
                    resource_stop = str(exc)
                    break
                results.append(_externalize_result(result, stream_dir))
    except (training.HostMemoryVeto, StrictCPUTrainingError) as exc:
        hard_veto = f"{type(exc).__name__}: {exc}"
    wall_seconds = time.perf_counter() - started
    source_paths = {
        "launcher": SCRIPT,
        "plan": PLAN,
        "training_implementation": TRAINING_SCRIPT.relative_to(ROOT),
        "trainer": Path("bayesfilter/inference/neutra_training.py"),
        "controller": Path("bayesfilter/inference/neutra_training_control.py"),
        "pool": Path("bayesfilter/inference/cpu_value_score_pool.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
    }
    status = (
        "HARD_VETO"
        if hard_veto is not None
        else "RESOURCE_STOP"
        if resource_stop is not None
        else "CPU_DIAGNOSTIC_COMPLETED"
        if len(results) == 2
        else "INCOMPLETE"
    )
    payload = {
        "schema": SCHEMA,
        "status": status,
        "q": 20,
        "architecture": list(HIDDEN_LAYERS),
        "batch_size": BATCH_SIZE,
        "params": dataclasses.asdict(PARAMS),
        "results": results,
        "active_stream_at_stop": current_stream,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "thread_contract": {
            "limit": THREAD_LIMIT,
            "parent_intra_op": PARENT_INTRA_THREADS,
            "parent_inter_op": PARENT_INTER_THREADS,
            "worker_count": WORKER_COUNT,
            "worker_intra_op_each": 1,
            "worker_inter_op_each": 1,
            "cpu_affinity": affinity,
            "audit": THREAD_AUDIT.payload(),
        },
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "platform": platform.platform(),
            "started_at_utc": started_utc,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": wall_seconds,
            "cap_seconds": args.cap_seconds,
            "cpu_gpu_status": "CPU-only; CUDA hidden before TensorFlow import",
            "jit_compile": False,
            "host_ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
            "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
            "random_seeds": {
                stream.label: dataclasses.asdict(stream) for stream in training.STREAMS
            },
            "output_root": args.output_root.as_posix(),
            "plan": PLAN.as_posix(),
            "source_paths": {key: path.as_posix() for key, path in source_paths.items()},
            "source_sha256": {key: _sha256(ROOT / path) for key, path in source_paths.items()},
        },
        "inference_status": {
            "hard_veto_screen": "see hard_veto and per-stream finite diagnostics",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "loss",
                "runtime",
                "best step",
                "learning-rate reductions",
                "seed differences",
            ],
            "default_readiness": "ineligible_cpu_diagnostic_exception",
            "next_evidence_needed": "GPU/XLA claim-bearing training before HMC",
        },
        "nonclaims": [
            "CPU-only diagnostic/reference exception",
            "no transport promotion or HMC eligibility",
            "no posterior-correctness, convergence, or scientific-validity claim",
            "no CPU/GPU or seed ranking",
            "a training-screen veto does not reject the NeuTra direction",
        ],
    }
    _write_json(output / "summary.json", payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=13500.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0:
        parser.error("--cap-seconds must be positive and finite")
    if args.cap_seconds > 13500.0:
        parser.error("--cap-seconds exceeds the reviewed CPU campaign cap")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run(args)
    print(json.dumps({"status": payload["status"], "results": payload["results"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
