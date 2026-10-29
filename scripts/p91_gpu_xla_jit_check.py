#!/usr/bin/env python
"""P91 trusted GPU/XLA capability check for local Zhao-Cui SIR d18 targets."""

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
PLAN_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase5-gpu-xla-jit-subplan-2026-06-29.md"
)
RESULT_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase5-gpu-xla-jit-result-2026-06-29.md"
)
PHASE6_SUBPLAN_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase6-performance-benchmark-subplan-2026-06-29.md"
)
DEFAULT_MANIFEST_PATH = (
    "docs/plans/"
    "bayesfilter-highdim-zhao-cui-p91-phase5-gpu-xla-jit-manifest-2026-06-29.json"
)


def _single_value_and_score_compiled():
    @tf.function(
        jit_compile=True,
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


def _batched_value_and_score_compiled():
    @tf.function(
        jit_compile=True,
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


def _fixture_inputs() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    theta = tf.constant([0.0, 0.0, 0.0], dtype=DTYPE)
    base = highdim.zhao_cui_sir_austria_model()
    states = [base.initial_mean]
    observations = []
    observation_offsets = tf.linspace(
        tf.constant(-0.2, dtype=DTYPE),
        tf.constant(0.2, dtype=DTYPE),
        9,
    )
    for time_index in range(FINAL_TIME + 1):
        current = states[-1]
        observations.append(current[1::2] + observation_offsets)
        if time_index < FINAL_TIME:
            transition_mean = _transition_mean_eager(theta, current[tf.newaxis, :])[0]
            perturbation = tf.linspace(
                tf.constant(-0.03, dtype=DTYPE),
                tf.constant(0.03, dtype=DTYPE),
                18,
            ) * tf.cast(time_index + 1, DTYPE)
            states.append(transition_mean + perturbation)
    return theta, tf.stack(states), tf.stack(observations)


def _batched_fixture_inputs(
    states: tf.Tensor,
    observations: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
    state_rows = []
    observation_rows = []
    for index in range(BATCH_SIZE):
        shift = tf.cast(index, DTYPE)
        state_shift = tf.linspace(
            tf.constant(-0.01, dtype=DTYPE),
            tf.constant(0.01, dtype=DTYPE),
            18,
        ) * shift
        obs_shift = tf.linspace(
            tf.constant(-0.005, dtype=DTYPE),
            tf.constant(0.005, dtype=DTYPE),
            9,
        ) * shift
        state_rows.append(states + state_shift[tf.newaxis, :])
        observation_rows.append(observations + obs_shift[tf.newaxis, :])
    return tf.stack(state_rows), tf.stack(observation_rows)


def _transition_mean_eager(theta: tf.Tensor, state: tf.Tensor) -> tf.Tensor:
    # Deterministic fixture construction only; the GPU/XLA gate below tests the
    # package helper, not this host-side setup path.
    parameters = tf.reshape(tf.convert_to_tensor(theta, dtype=DTYPE), [3])
    state_tensor = tf.convert_to_tensor(state, dtype=DTYPE)
    kappa = tf.fill([9], tf.constant(0.1, dtype=DTYPE) * tf.exp(parameters[0]))
    nu = tf.fill([9], tf.constant(18.0, dtype=DTYPE) * tf.exp(parameters[1]))
    adjacency = tf.constant(
        [
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=DTYPE,
    )
    degree = tf.reduce_sum(adjacency, axis=1)
    step = tf.constant(0.005, dtype=DTYPE)
    current = state_tensor
    for _ in range(4):
        k1 = _rhs(current, kappa, nu, adjacency, degree)
        k2 = _rhs(current + 0.5 * step * k1, kappa, nu, adjacency, degree)
        k3 = _rhs(current + 0.5 * step * k2, kappa, nu, adjacency, degree)
        k4 = _rhs(current + 0.5 * step * k3, kappa, nu, adjacency, degree)
        current = current + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return current


def _rhs(
    state: tf.Tensor,
    kappa: tf.Tensor,
    nu: tf.Tensor,
    adjacency: tf.Tensor,
    degree: tf.Tensor,
) -> tf.Tensor:
    susceptible = state[:, 0::2]
    infectious = state[:, 1::2]
    susceptible_neighbor = (
        tf.linalg.matmul(susceptible, adjacency, transpose_b=True)
        - susceptible * degree[tf.newaxis, :]
    )
    infectious_neighbor = (
        tf.linalg.matmul(infectious, adjacency, transpose_b=True)
        - infectious * degree[tf.newaxis, :]
    )
    infection = kappa[tf.newaxis, :] * susceptible * infectious
    rhs_susceptible = -infection + 0.5 * susceptible_neighbor
    rhs_infectious = infection - nu[tf.newaxis, :] * infectious + 0.5 * infectious_neighbor
    return tf.reshape(tf.stack([rhs_susceptible, rhs_infectious], axis=2), [-1, 18])


def _run_compiled(
    name: str,
    fn: Any,
    args: tuple[tf.Tensor, ...],
) -> dict[str, Any]:
    trace_count_before = int(fn.experimental_get_tracing_count())
    timings = []
    outputs = None
    for _ in range(4):
        start = time.perf_counter()
        with tf.device("/GPU:0"):
            outputs = fn(*args)
        for output in outputs:
            _ = output.numpy()
        timings.append(time.perf_counter() - start)
    trace_count_after_warmup = int(fn.experimental_get_tracing_count())
    start = time.perf_counter()
    with tf.device("/GPU:0"):
        repeated_outputs = fn(*args)
    for output in repeated_outputs:
        _ = output.numpy()
    repeated_time = time.perf_counter() - start
    trace_count_after_repeated = int(fn.experimental_get_tracing_count())
    assert outputs is not None
    values, scores = outputs
    finite = bool(
        tf.reduce_all(tf.math.is_finite(values)).numpy()
        and tf.reduce_all(tf.math.is_finite(scores)).numpy()
    )
    output_devices = tuple(output.device for output in outputs)
    gpu_output_devices = all("GPU" in device.upper() for device in output_devices)
    retracing_pass = trace_count_after_repeated == trace_count_after_warmup
    return {
        "name": name,
        "input_shapes": [tuple(int(dim) for dim in tensor.shape) for tensor in args],
        "input_dtypes": [tensor.dtype.name for tensor in args],
        "values": _tensor_to_float_list(values),
        "scores": _tensor_to_float_list(scores),
        "output_devices": output_devices,
        "all_finite": finite,
        "gpu_output_devices": gpu_output_devices,
        "first_call_seconds": timings[0],
        "second_call_seconds": timings[1],
        "steady_call_seconds": timings[2:],
        "repeated_call_seconds": repeated_time,
        "trace_count_before": trace_count_before,
        "trace_count_after_warmup": trace_count_after_warmup,
        "trace_count_after_repeated": trace_count_after_repeated,
        "retracing_pass": retracing_pass,
        "passed": bool(finite and gpu_output_devices and retracing_pass),
    }


def _tensor_to_float_list(tensor: tf.Tensor) -> list[float]:
    return [float(value) for value in tf.reshape(tensor, [-1]).numpy()]


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    device_payload = _device_payload()
    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        payload = _manifest_payload(
            manifest_path=manifest_path,
            status="BLOCK_P91_PHASE5_NO_TENSORFLOW_GPU_DEVICE",
            device_payload=device_payload,
            checks=[],
            errors=["TensorFlow reported no physical GPU devices."],
        )
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return 1

    theta, states, observations = _fixture_inputs()
    batched_states, batched_observations = _batched_fixture_inputs(states, observations)
    errors = []
    checks = []
    try:
        single_fn = _single_value_and_score_compiled()
        batched_fn = _batched_value_and_score_compiled()
        checks.append(_run_compiled("single_local_complete_data", single_fn, (theta, states, observations)))
        checks.append(
            _run_compiled(
                "batched_local_complete_data",
                batched_fn,
                (theta, batched_states, batched_observations),
            )
        )
    except Exception as exc:  # pragma: no cover - manifest path for runtime failures.
        errors.append(f"{type(exc).__name__}: {exc}")

    status = (
        "PASS_P91_PHASE5_GPU_XLA_JIT_LOCAL_COMPLETE_DATA"
        if not errors and all(bool(check["passed"]) for check in checks)
        else "BLOCK_P91_PHASE5_GPU_XLA_JIT_LOCAL_COMPLETE_DATA"
    )
    payload = _manifest_payload(
        manifest_path=manifest_path,
        status=status,
        device_payload=device_payload,
        checks=checks,
        errors=errors,
    )
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": status, "manifest": str(manifest_path)}, sort_keys=True))
    return 0 if status.startswith("PASS_") else 1


def _manifest_payload(
    *,
    manifest_path: Path,
    status: str,
    device_payload: dict[str, Any],
    checks: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "p91.phase5.gpu_xla_jit.v1",
        "status": status,
        "git_commit": _git_commit(),
        "worktree_status": _git_dirty_note(),
        "command": (
            "python scripts/p91_gpu_xla_jit_check.py --manifest "
            "docs/plans/bayesfilter-highdim-zhao-cui-p91-phase5-gpu-xla-jit-manifest-2026-06-29.json"
        ),
        "python_executable": sys.executable,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "trusted_escalated_gpu_run": True,
        "tensorflow": _tf_build_info(),
        "devices": device_payload,
        "execution_target": "trusted GPU/XLA local complete-data value/score capability",
        "input_contract": {
            "theta_shape": [3],
            "single_states_shape": [FINAL_TIME + 1, 18],
            "single_observations_shape": [FINAL_TIME + 1, 9],
            "batched_states_shape": [BATCH_SIZE, FINAL_TIME + 1, 18],
            "batched_observations_shape": [BATCH_SIZE, FINAL_TIME + 1, 9],
            "dtype": DTYPE.name,
            "random_seeds": "N/A; deterministic fixture",
        },
        "checks": checks,
        "errors": errors,
        "primary_criterion": (
            "single and batched local complete-data value/score functions JIT "
            "compile and execute on /GPU:0 with finite outputs, GPU output "
            "devices, stable post-warmup tracing counts, and no OOM"
        ),
        "blocker_statuses_preserved": {
            "full_observed_data_filtering_score_identity": "NOT_CLAIMED",
            "previous_marginal_derivative": (
                "BLOCK_FIXED_TTSIRT_PREVIOUS_MARGINAL_DERIVATIVE_NOT_IMPLEMENTED"
            ),
            "fixed_ttsirt_transport_derivative": (
                "BLOCK_FIXED_TTSIRT_PROPOSAL_TRANSPORT_DERIVATIVE_NOT_IMPLEMENTED"
            ),
            "full_source_route_fd": "BLOCK_FULL_SOURCE_ROUTE_FD_NOT_CLAIMED",
        },
        "nonclaims": [
            "no full observed-data/filtering score identity",
            "no previous-marginal derivative readiness",
            "no fixed TTSIRT proposal/transport derivative readiness",
            "no GPU speed superiority",
            "no benchmark pass",
            "no HMC posterior validity",
            "no packaging/default readiness",
            "no production readiness",
        ],
        "artifact_paths": {
            "plan": PLAN_PATH,
            "manifest": str(manifest_path),
            "result": RESULT_PATH,
            "phase6_subplan": PHASE6_SUBPLAN_PATH,
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
