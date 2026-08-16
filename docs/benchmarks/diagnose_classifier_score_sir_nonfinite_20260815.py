"""Diagnose non-finite paths in the V7 SIR nested training bank."""

from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "docs/benchmarks/run_classifier_score_path_count_bundle_20260815.py"
OUTPUT_PATH = (
    ROOT
    / "docs/benchmarks/artifacts/classifier_score_path_count_scaling_20260815"
    / "sir_16384_nonfinite_diagnostic_attempt03.json"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("v7_path_count_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load V7 path-count runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def safe_float(value: tf.Tensor) -> float | None:
    result = float(value.numpy())
    return result if math.isfinite(result) else None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_single_path(
    runner, parameters: tf.Tensor, noise: tuple[tf.Tensor, ...], path_index: int
) -> dict[str, object]:
    sir = runner.sir
    state = sir.INITIAL_MEAN + noise[0][path_index]
    steps = []
    for time_index in range(50):
        current = state[None, :]
        substeps = []
        for substep_index in range(sir.SUBSTEPS):
            kappa = sir.BASE_KAPPA * tf.exp(parameters[0])
            nu = sir.BASE_NU * tf.exp(parameters[1])
            k1 = sir._rhs(current, kappa, nu)
            k2 = sir._rhs(current + 0.5 * sir.STEP * k1, kappa, nu)
            k3 = sir._rhs(current + 0.5 * sir.STEP * k2, kappa, nu)
            k4 = sir._rhs(current + 0.5 * sir.STEP * k3, kappa, nu)
            current = current + (sir.STEP / 6.0) * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            )
            substeps.append(
                {
                    "substep": substep_index + 1,
                    "finite": bool(tf.reduce_all(tf.math.is_finite(current)).numpy()),
                    "max_abs_state": safe_float(tf.reduce_max(tf.abs(current))),
                }
            )
        latent = current[0] + noise[1][path_index, time_index]
        susceptible = tf.maximum(latent[0::2], 0.0)
        infectious = latent[1::2]
        state = tf.reshape(
            tf.stack([susceptible, infectious], axis=1), [sir.STATE_DIMENSION]
        )
        observation = (
            infectious
            + sir.BASE_OBSERVATION_SCALE
            * tf.exp(parameters[2])
            * noise[2][path_index, time_index]
        )
        steps.append(
            {
                "time": time_index + 1,
                "finite_state": bool(tf.reduce_all(tf.math.is_finite(state)).numpy()),
                "finite_observation": bool(
                    tf.reduce_all(tf.math.is_finite(observation)).numpy()
                ),
                "max_abs_state": safe_float(tf.reduce_max(tf.abs(state))),
                "max_abs_observation": safe_float(tf.reduce_max(tf.abs(observation))),
                "substeps": substeps,
            }
        )
        if not steps[-1]["finite_observation"]:
            break
    return {"path_index": path_index, "steps": steps}


def main() -> None:
    started = time.perf_counter()
    runner = load_runner()
    simulator = runner.sir.make_compiled_observation_simulator(50)
    coordinate = 1
    bundle = 0
    count = 16384
    block_size = runner.SIMULATION_BLOCK
    direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
    cases = []
    failing_path_traces = []
    total_nonfinite_paths = 0
    baseline_prefix_nonfinite_paths = 0
    added_prefix_nonfinite_paths = 0

    for delta_index, delta in enumerate(runner.DELTAS):
        for sign_index, sign_name, sign in ((0, "minus", -1.0), (1, "plus", 1.0)):
            noise = runner.make_noise(
                "sir", count, (bundle, 10, coordinate, delta_index, sign_index)
            )
            parameters = runner.THETA + tf.cast(sign * delta, tf.float64) * direction
            for block_index, start in enumerate(range(0, count, block_size)):
                stop = start + block_size
                observations = simulator(
                    parameters, *(value[start:stop] for value in noise)
                )
                finite_elements = tf.math.is_finite(observations)
                finite_paths = tf.reduce_all(finite_elements, axis=(1, 2))
                nonfinite_path_count = int(
                    tf.reduce_sum(tf.cast(tf.logical_not(finite_paths), tf.int32)).numpy()
                )
                total_nonfinite_paths += nonfinite_path_count
                if block_index == 0:
                    baseline_prefix_nonfinite_paths += nonfinite_path_count
                else:
                    added_prefix_nonfinite_paths += nonfinite_path_count
                nonfinite_by_time = tf.reduce_sum(
                    tf.cast(
                        tf.logical_not(tf.reduce_all(finite_elements, axis=2)), tf.int32
                    ),
                    axis=0,
                )
                failing_times = tf.where(nonfinite_by_time > 0)
                first_failing_time = (
                    int(failing_times[0, 0].numpy()) + 1
                    if int(tf.shape(failing_times)[0].numpy()) > 0
                    else None
                )
                finite_values = tf.boolean_mask(observations, finite_elements)
                max_abs_finite = (
                    safe_float(tf.reduce_max(tf.abs(finite_values)))
                    if int(tf.size(finite_values).numpy()) > 0
                    else None
                )
                cases.append(
                    {
                        "delta": float(delta),
                        "sign": sign_name,
                        "block_index": block_index,
                        "row_start": start,
                        "row_stop": stop,
                        "nonfinite_path_count": nonfinite_path_count,
                        "first_failing_time": first_failing_time,
                        "nonfinite_path_indices": [
                            start + int(value)
                            for value in tf.reshape(
                                tf.where(tf.logical_not(finite_paths)), [-1]
                            ).numpy().tolist()
                        ],
                        "max_abs_finite_observation": max_abs_finite,
                    }
                )
                if nonfinite_path_count:
                    for local_index in tf.reshape(
                        tf.where(tf.logical_not(finite_paths)), [-1]
                    ).numpy().tolist():
                        failing_path_traces.append(
                            {
                                "delta": float(delta),
                                "sign": sign_name,
                                "trace": trace_single_path(
                                    runner, parameters, noise, start + int(local_index)
                                ),
                            }
                        )

    payload = {
        "schema": "bayesfilter.classifier_score_sir_nonfinite_diagnostic.v1",
        "status": "NONFINITE_FOUND" if total_nonfinite_paths else "ALL_FINITE",
        "bundle": bundle,
        "coordinate": coordinate,
        "path_count": count,
        "block_size": block_size,
        "total_nonfinite_paths_across_delta_sign_cases": total_nonfinite_paths,
        "baseline_8192_prefix_nonfinite_paths": baseline_prefix_nonfinite_paths,
        "added_8192_prefix_nonfinite_paths": added_prefix_nonfinite_paths,
        "cases": cases,
        "failing_path_traces": failing_path_traces,
        "git_commit": runner.git_commit(),
        "command": [sys.executable, *sys.argv],
        "source_sha256": {
            str(RUNNER_PATH.relative_to(ROOT)): sha256(RUNNER_PATH),
            str(runner.SIR_PATH.relative_to(ROOT)): sha256(runner.SIR_PATH),
            str(Path(__file__).resolve().relative_to(ROOT)): sha256(
                Path(__file__).resolve()
            ),
        },
        "cuda_visible_devices": runner.os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "selected_nvidia_smi_index": runner.os.environ.get(
            "BAYESFILTER_SELECTED_NVIDIA_SMI_INDEX", "unset"
        ),
        "selected_gpu_uuid": runner.os.environ.get(
            "BAYESFILTER_SELECTED_GPU_UUID", "unset"
        ),
        "selected_gpu_name": runner.os.environ.get(
            "BAYESFILTER_SELECTED_GPU_NAME", "unset"
        ),
        "gpu_memory_policy": runner.GPU_MEMORY_POLICY,
        "wall_time_seconds": time.perf_counter() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "interpretation_guard": (
            "Diagnostic only. No rows were dropped, clipped, replaced, or resampled."
        ),
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(runner.safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(runner.safe(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
