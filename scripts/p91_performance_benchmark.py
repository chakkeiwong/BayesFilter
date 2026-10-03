#!/usr/bin/env python
"""P91 CPU/GPU benchmark harness for local Zhao-Cui SIR d18 targets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import tensorflow as tf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bayesfilter.highdim as highdim
from bayesfilter.testing.p91_complete_data_diagnostics_tf import complete_data_values_and_scores_tf


DTYPE = tf.float64
FINAL_TIME = 4
BATCH_SIZE = 4
REPEATS = 5
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase6-performance-benchmark-subplan-2026-06-29.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase6-performance-benchmark-result-2026-06-29.md"
)
PHASE7_SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase7-hmc-smoke-subplan-2026-06-29.md"
)


def _single_compiled(*, jit_compile: bool):
    @tf.function(
        jit_compile=jit_compile,
        input_signature=[
            tf.TensorSpec([3], DTYPE),
            tf.TensorSpec([FINAL_TIME + 1, 18], DTYPE),
            tf.TensorSpec([FINAL_TIME + 1, 9], DTYPE),
        ],
    )
    def value_and_score(
        theta: tf.Tensor,
        states: tf.Tensor,
        observations: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            value = highdim.zhao_cui_sir_austria_local_complete_data_log_density_xla(
                theta,
                states,
                observations,
            )
        score = tape.gradient(value, theta)
        if score is None:
            score = tf.fill(tf.shape(theta), tf.constant(float("nan"), dtype=DTYPE))
        return value, score

    return value_and_score


def _batched_compiled(*, jit_compile: bool):
    @tf.function(
        jit_compile=jit_compile,
        input_signature=[
            tf.TensorSpec([3], DTYPE),
            tf.TensorSpec([BATCH_SIZE, FINAL_TIME + 1, 18], DTYPE),
            tf.TensorSpec([BATCH_SIZE, FINAL_TIME + 1, 9], DTYPE),
        ],
    )
    def values_and_scores(
        theta: tf.Tensor,
        states: tf.Tensor,
        observations: tf.Tensor,
    ) -> tuple[tf.Tensor, tf.Tensor]:
        return complete_data_values_and_scores_tf(theta, states, observations)

    return values_and_scores


def _fixture_inputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    theta = tf.constant([0.0, 0.0, 0.0], dtype=DTYPE)
    model = highdim.parameterized_zhao_cui_sir_austria_model()
    states = [model.base_model.initial_mean]
    observations = []
    observation_offsets = tf.linspace(
        tf.constant(-0.2, dtype=DTYPE),
        tf.constant(0.2, dtype=DTYPE),
        model.observation_dim(),
    )
    for time_index in range(FINAL_TIME + 1):
        current = states[-1]
        observations.append(model.infectious_components(current)[0] + observation_offsets)
        if time_index < FINAL_TIME:
            transition_mean = model.transition_mean(theta, current[tf.newaxis, :])[0]
            perturbation = tf.linspace(
                tf.constant(-0.03, dtype=DTYPE),
                tf.constant(0.03, dtype=DTYPE),
                model.state_dim(),
            ) * tf.cast(time_index + 1, DTYPE)
            states.append(transition_mean + perturbation)
    single_states = tf.stack(states)
    single_observations = tf.stack(observations)
    batched_states = []
    batched_observations = []
    for index in range(BATCH_SIZE):
        shift = tf.cast(index, DTYPE)
        state_shift = tf.linspace(
            tf.constant(-0.01, dtype=DTYPE),
            tf.constant(0.01, dtype=DTYPE),
            model.state_dim(),
        ) * shift
        obs_shift = tf.linspace(
            tf.constant(-0.005, dtype=DTYPE),
            tf.constant(0.005, dtype=DTYPE),
            model.observation_dim(),
        ) * shift
        batched_states.append(single_states + state_shift[tf.newaxis, :])
        batched_observations.append(single_observations + obs_shift[tf.newaxis, :])
    return (
        theta,
        single_states,
        single_observations,
        tf.stack(batched_states),
        tf.stack(batched_observations),
    )


def _run_cell(
    *,
    name: str,
    fn: Any,
    args: tuple[tf.Tensor, ...],
    device_name: str,
    per_call_items: int,
) -> dict[str, Any]:
    trace_count_before = int(fn.experimental_get_tracing_count())
    retry_count = 0
    oom_status = False
    error = ""
    first_call_seconds = None
    second_call_seconds = None
    steady_seconds: list[float] = []
    repeated_seconds = None
    outputs = None
    trace_count_after_warmup = None
    trace_count_after_repeated = None
    try:
        start = time.perf_counter()
        with tf.device(device_name):
            outputs = fn(*args)
        _materialize(outputs)
        first_call_seconds = time.perf_counter() - start

        start = time.perf_counter()
        with tf.device(device_name):
            outputs = fn(*args)
        _materialize(outputs)
        second_call_seconds = time.perf_counter() - start
        trace_count_after_warmup = int(fn.experimental_get_tracing_count())

        for _ in range(REPEATS):
            start = time.perf_counter()
            with tf.device(device_name):
                outputs = fn(*args)
            _materialize(outputs)
            steady_seconds.append(time.perf_counter() - start)

        start = time.perf_counter()
        with tf.device(device_name):
            outputs = fn(*args)
        _materialize(outputs)
        repeated_seconds = time.perf_counter() - start
        trace_count_after_repeated = int(fn.experimental_get_tracing_count())
    except tf.errors.ResourceExhaustedError as exc:
        oom_status = True
        error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - runtime manifest path.
        error = f"{type(exc).__name__}: {exc}"

    if outputs is None:
        return {
            "name": name,
            "passed": False,
            "error": error or "no outputs produced",
            "retry_count": retry_count,
            "oom_status": oom_status,
            "post_warmup_retrace_detected": True,
            "all_finite": False,
            "output_devices": [],
            "gpu_output_devices": False,
        }

    values, scores = outputs
    all_finite = bool(
        tf.reduce_all(tf.math.is_finite(values)).numpy()
        and tf.reduce_all(tf.math.is_finite(scores)).numpy()
    )
    output_devices = tuple(output.device for output in outputs)
    gpu_output_devices = all("GPU" in device.upper() for device in output_devices)
    post_warmup_retrace_detected = trace_count_after_repeated != trace_count_after_warmup
    steady_mean = float(sum(steady_seconds) / len(steady_seconds)) if steady_seconds else None
    return {
        "name": name,
        "input_shapes": [tuple(int(dim) for dim in tensor.shape) for tensor in args],
        "input_dtypes": [tensor.dtype.name for tensor in args],
        "values": _tensor_to_float_list(values),
        "scores": _tensor_to_float_list(scores),
        "output_devices": output_devices,
        "all_finite": all_finite,
        "gpu_output_devices": gpu_output_devices,
        "first_call_seconds": first_call_seconds,
        "second_call_seconds": second_call_seconds,
        "steady_call_seconds": steady_seconds,
        "steady_mean_seconds": steady_mean,
        "steady_per_item_seconds": (
            steady_mean / float(per_call_items) if steady_mean is not None else None
        ),
        "repeated_call_seconds": repeated_seconds,
        "trace_count_before": trace_count_before,
        "trace_count_after_warmup": trace_count_after_warmup,
        "trace_count_after_repeated": trace_count_after_repeated,
        "post_warmup_retrace_detected": bool(post_warmup_retrace_detected),
        "retry_count": retry_count,
        "oom_status": oom_status,
        "error": error,
        "passed": bool(
            all_finite
            and retry_count == 0
            and not oom_status
            and not post_warmup_retrace_detected
            and not error
            and steady_mean is not None
        ),
    }


def _materialize(outputs: tuple[tf.Tensor, tf.Tensor]) -> None:
    for output in outputs:
        _ = output.numpy()


def _tensor_to_float_list(tensor: tf.Tensor) -> list[float]:
    return [float(value) for value in tf.reshape(tensor, [-1]).numpy()]


def _write_target_manifest(args: argparse.Namespace) -> int:
    target = str(args.target)
    actual_xla = _str_to_bool(args.xla)
    manifest_path = Path(str(args.manifest))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    device_name = "/CPU:0" if target == "cpu" else "/GPU:0"

    if target == "gpu" and not actual_xla:
        raise SystemExit("--target gpu requires --xla true under the P91 Phase 6 subplan")
    if target == "cpu" and actual_xla:
        raise SystemExit("--target cpu requires --xla false under the P91 Phase 6 subplan")

    devices = _device_payload()
    errors: list[str] = []
    if target == "gpu" and not tf.config.list_physical_devices("GPU"):
        errors.append("TensorFlow reported no physical GPU devices.")

    theta, states, observations, batched_states, batched_observations = _fixture_inputs()
    checks: list[dict[str, Any]] = []
    if not errors:
        single_fn = _single_compiled(jit_compile=actual_xla)
        batched_fn = _batched_compiled(jit_compile=actual_xla)
        checks.append(
            _run_cell(
                name=f"{target}_{'xla_' if actual_xla else ''}looped_single",
                fn=single_fn,
                args=(theta, states, observations),
                device_name=device_name,
                per_call_items=1,
            )
        )
        checks.append(
            _run_cell(
                name=f"{target}_{'xla_' if actual_xla else ''}batched",
                fn=batched_fn,
                args=(theta, batched_states, batched_observations),
                device_name=device_name,
                per_call_items=BATCH_SIZE,
            )
        )

    target_pathology = _target_pathology(target=target, actual_xla=actual_xla, checks=checks)
    status = (
        f"PASS_P91_PHASE6_{target.upper()}_{'XLA_' if actual_xla else ''}BENCHMARK"
        if not errors and not target_pathology["pathology_detected"]
        else f"BLOCK_P91_PHASE6_{target.upper()}_{'XLA_' if actual_xla else ''}BENCHMARK"
    )
    payload = {
        "schema_version": "p91.phase6.performance_benchmark.target.v1",
        "status": status,
        "target": target,
        "requested_xla": actual_xla,
        "actual_xla_status": actual_xla,
        "trusted_escalated_context_required": target == "gpu",
        "trusted_escalated_context_recorded": target == "gpu",
        "command": _target_command(target=target, actual_xla=actual_xla, manifest_path=manifest_path),
        "git_commit": _git_commit(),
        "worktree_status": _git_dirty_note(),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "tensorflow": _tf_build_info(),
        "devices": devices,
        "cpu_gpu_visibility": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
            "physical_gpus_visible_to_tensorflow": devices["physical_gpus"],
            "logical_gpus_visible_to_tensorflow": devices["logical_gpus"],
        },
        "input_contract": {
            "theta_shape": [3],
            "single_states_shape": [FINAL_TIME + 1, 18],
            "single_observations_shape": [FINAL_TIME + 1, 9],
            "batched_states_shape": [BATCH_SIZE, FINAL_TIME + 1, 18],
            "batched_observations_shape": [BATCH_SIZE, FINAL_TIME + 1, 9],
            "dtype": DTYPE.name,
            "random_seeds": "N/A; deterministic fixture",
            "repeats": REPEATS,
        },
        "checks": checks,
        "errors": errors,
        "pathology": target_pathology,
        "artifact_paths": {
            "plan": PLAN_PATH,
            "manifest": str(manifest_path),
            "result": RESULT_PATH,
            "phase7_subplan": PHASE7_SUBPLAN_PATH,
        },
        "nonclaims": _nonclaims(),
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "manifest": str(manifest_path)}, sort_keys=True))
    return 0 if status.startswith("PASS_") else 1


def _target_pathology(
    *,
    target: str,
    actual_xla: bool,
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = []
    for check in checks:
        if not bool(check.get("all_finite", False)):
            reasons.append(f"{check.get('name')}: nonfinite output value or score")
        if int(check.get("retry_count", 1)) > 0:
            reasons.append(f"{check.get('name')}: retry_count > 0")
        if bool(check.get("oom_status", False)):
            reasons.append(f"{check.get('name')}: OOM")
        if bool(check.get("post_warmup_retrace_detected", True)):
            reasons.append(f"{check.get('name')}: post-warmup retrace")
        if check.get("steady_mean_seconds") is None:
            reasons.append(f"{check.get('name')}: missing steady timing")
        if target == "gpu" and not bool(check.get("gpu_output_devices", False)):
            reasons.append(f"{check.get('name')}: missing GPU output devices")
    if target == "gpu" and not actual_xla:
        reasons.append("GPU benchmark missing actual_xla_status == true")
    if len(checks) == 2:
        single = checks[0].get("steady_per_item_seconds")
        batched = checks[1].get("steady_per_item_seconds")
        if single is None or batched is None:
            reasons.append("missing per-item timing for batch scaling")
        elif float(batched) > 10.0 * float(single):
            reasons.append("batched per-item steady time exceeds 10x looped single")
    return {
        "pathology_detected": bool(reasons),
        "reasons": reasons,
        "closed_rules": [
            "nonfinite output value or score",
            "retry_count > 0",
            "OOM",
            "post-warmup retrace",
            "missing steady timing",
            "batched per-item steady time > 10x looped single on same target",
            "GPU actual_xla_status != true",
            "trusted GPU output devices missing",
        ],
    }


def _write_merged_manifest(args: argparse.Namespace) -> int:
    inputs = [Path(path) for path in args.merge]
    manifest_path = Path(str(args.manifest))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in inputs]
    target_statuses = [manifest.get("status", "UNKNOWN") for manifest in manifests]
    any_blocked = any(not str(status).startswith("PASS_") for status in target_statuses)
    pathologies = []
    for manifest in manifests:
        pathology = manifest.get("pathology", {})
        if pathology.get("pathology_detected"):
            pathologies.extend(pathology.get("reasons", []))
    status = (
        "PASS_P91_PHASE6_PERFORMANCE_BENCHMARK"
        if not any_blocked and not pathologies
        else "BLOCK_P91_PHASE6_PERFORMANCE_BENCHMARK"
    )
    payload = {
        "schema_version": "p91.phase6.performance_benchmark.combined.v1",
        "status": status,
        "command": _merge_command(inputs, manifest_path),
        "git_commit": _git_commit(),
        "worktree_status": _git_dirty_note(),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "target_statuses": target_statuses,
        "pathology_detected": bool(pathologies),
        "pathology_reasons": pathologies,
        "manifests": manifests,
        "artifact_paths": {
            "plan": PLAN_PATH,
            "manifest": str(manifest_path),
            "result": RESULT_PATH,
            "phase7_subplan": PHASE7_SUBPLAN_PATH,
            "inputs": [str(path) for path in inputs],
        },
        "nonclaims": _nonclaims(),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "manifest": str(manifest_path)}, sort_keys=True))
    return 0 if status.startswith("PASS_") else 1


def _target_command(*, target: str, actual_xla: bool, manifest_path: Path) -> str:
    prefix = "CUDA_VISIBLE_DEVICES=-1 " if target == "cpu" else ""
    return (
        f"{prefix}python scripts/p91_performance_benchmark.py --target {target} "
        f"--xla {str(actual_xla).lower()} --manifest {manifest_path}"
    )


def _merge_command(inputs: list[Path], manifest_path: Path) -> str:
    return (
        "python scripts/p91_performance_benchmark.py --merge "
        + " ".join(str(path) for path in inputs)
        + f" --manifest {manifest_path}"
    )


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "UNKNOWN"


def _git_dirty_note() -> str:
    completed = subprocess.run(
        ["git", "status", "--short"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "UNKNOWN"
    return "dirty research worktree" if completed.stdout.strip() else "clean"


def _tf_build_info() -> dict[str, Any]:
    build_info = dict(getattr(tf.sysconfig, "get_build_info", lambda: {})())
    return {
        "tensorflow_version": tf.__version__,
        "cuda_version": build_info.get("cuda_version", "N/A"),
        "cudnn_version": build_info.get("cudnn_version", "N/A"),
        "is_cuda_build": bool(tf.test.is_built_with_cuda()),
    }


def _device_payload() -> dict[str, Any]:
    physical_gpus = tf.config.list_physical_devices("GPU")
    logical_gpus = tf.config.list_logical_devices("GPU")
    return {
        "physical_gpus": [device.name for device in physical_gpus],
        "logical_gpus": [device.name for device in logical_gpus],
        "gpu_names": [
            tf.config.experimental.get_device_details(device).get("device_name", "N/A")
            for device in physical_gpus
        ],
    }


def _nonclaims() -> list[str]:
    return [
        "no universal GPU speed superiority",
        "no score identity proof",
        "no exact likelihood correctness",
        "no HMC posterior validity",
        "no packaging/default readiness",
        "no production readiness",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("cpu", "gpu"))
    parser.add_argument("--xla", default="false")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--merge", nargs="*")
    args = parser.parse_args()
    if args.merge:
        return _write_merged_manifest(args)
    if args.target is None:
        raise SystemExit("--target is required unless --merge is used")
    return _write_target_manifest(args)


if __name__ == "__main__":
    raise SystemExit(main())
