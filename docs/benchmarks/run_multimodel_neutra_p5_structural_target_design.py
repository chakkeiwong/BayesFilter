#!/usr/bin/env python3
"""Execute the reviewed P5 structural target-design gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


PROGRAM_ID = "multimodel-neutra-filter-posterior-20260715"
PLAN_FILE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p5-structural-target-design-subplan-2026-07-16.md"
)
RESULT_FILE = (
    "docs/plans/bayesfilter-multimodel-neutra-filter-posterior-"
    "p5-structural-target-design-result-2026-07-16.md"
)
DESIGN_SEEDS = ((20260716, 15101), (20260716, 15102), (20260716, 15103))
FINAL_SEED = (20260716, 15001)
PRIOR_PREDICTIVE_SEED = (20260716, 15201)
HORIZONS = (50, 100, 200)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--gpu-canary-only",
        action="store_true",
        help="run only the trusted GPU/XLA value-score canary",
    )
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(value: tf.Tensor) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_sha256(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode("utf-8")).hexdigest()


def _design_points() -> tf.Tensor:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        structural_truth_source,
    )

    truth = structural_truth_source()
    eye = 0.5 * tf.eye(5, dtype=tf.float64)
    return tf.concat([truth[None, :], truth[None, :] + eye, truth[None, :] - eye], axis=0)


def _eigen_summary(matrix: tf.Tensor) -> dict[str, Any]:
    symmetric = 0.5 * (matrix + tf.linalg.matrix_transpose(matrix))
    eigenvalues = tf.linalg.eigvalsh(symmetric)
    maximum = eigenvalues[..., -1]
    minimum = eigenvalues[..., 0]
    rank = tf.reduce_sum(
        tf.cast(eigenvalues > 1.0e-8 * maximum[..., None], tf.int32), axis=-1
    )
    condition = maximum / tf.maximum(minimum, tf.constant(1.0e-300, tf.float64))
    valid_psd = minimum >= -1.0e-8 * tf.maximum(
        tf.ones_like(maximum), maximum
    )
    passed = tf.logical_and(
        valid_psd,
        tf.logical_and(
            tf.equal(rank, 5),
            tf.logical_and(minimum >= 0.10, condition <= 1.0e6),
        ),
    )
    return {
        "eigenvalues": eigenvalues,
        "minimum_eigenvalue": minimum,
        "maximum_eigenvalue": maximum,
        "numerical_rank": rank,
        "condition_number": condition,
        "psd_passed": valid_psd,
        "information_gate_passed": passed,
    }


def _run_design_audit(checkpoint_root: Path | None = None) -> dict[str, Any]:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_TRUTH_PHYSICAL,
        simulate_structural_trajectories_tf,
        structural_likelihood_information_tf,
    )

    points = _design_points()
    physical_truth = STRUCTURAL_TRUTH_PHYSICAL[None, :]

    @tf.function(
        input_signature=[
            tf.TensorSpec([11, 5], tf.float64),
            tf.TensorSpec([200, 1], tf.float64),
        ],
        jit_compile=True,
    )
    def compiled_information(
        source_points: tf.Tensor, observations: tf.Tensor
    ) -> tuple[Mapping[str, tf.Tensor], Mapping[str, tf.Tensor]]:
        return (
            structural_likelihood_information_tf(
                source_points,
                observations=observations,
                finite_difference_step=tf.constant(5.0e-5, tf.float64),
            ),
            structural_likelihood_information_tf(
                source_points,
                observations=observations,
                finite_difference_step=tf.constant(1.0e-4, tf.float64),
            ),
        )

    rows: list[dict[str, Any]] = []
    all_t100 = True
    monotonic_all = True
    for seed in DESIGN_SEEDS:
        _states, observations_batch, residuals = simulate_structural_trajectories_tf(
            physical_truth,
            horizon=200,
            seed=tf.constant(seed, tf.int32),
        )
        observations = observations_batch[0]
        fine_information, coarse_information = compiled_information(points, observations)
        information = fine_information["cumulative_information"]
        derivative_scale = tf.maximum(
            tf.ones_like(fine_information["d_predictive_mean"]),
            tf.abs(fine_information["d_predictive_mean"]),
        )
        mean_derivative_relative_gap = tf.reduce_max(
            tf.abs(
                fine_information["d_predictive_mean"]
                - coarse_information["d_predictive_mean"]
            )
            / derivative_scale
        )
        log_variance_derivative_scale = tf.maximum(
            tf.ones_like(fine_information["d_log_innovation_variance"]),
            tf.abs(fine_information["d_log_innovation_variance"]),
        )
        log_variance_derivative_relative_gap = tf.reduce_max(
            tf.abs(
                fine_information["d_log_innovation_variance"]
                - coarse_information["d_log_innovation_variance"]
            )
            / log_variance_derivative_scale
        )
        derivative_step_stability_passed = tf.logical_and(
            mean_derivative_relative_gap <= 5.0e-3,
            log_variance_derivative_relative_gap <= 5.0e-3,
        )
        horizon_matrices = tf.gather(
            information,
            tf.constant([horizon - 1 for horizon in HORIZONS], tf.int32),
            axis=1,
        )
        summaries = [_eigen_summary(horizon_matrices[:, index]) for index in range(3)]
        t100_pass = bool(
            tf.logical_and(
                tf.reduce_all(summaries[1]["information_gate_passed"]),
                derivative_step_stability_passed,
            ).numpy()
        )
        all_t100 = all_t100 and t100_pass
        increments = horizon_matrices[:, 1:] - horizon_matrices[:, :-1]
        increment_eigenvalues = tf.linalg.eigvalsh(
            0.5 * (increments + tf.linalg.matrix_transpose(increments))
        )
        scale = tf.maximum(
            tf.ones(tf.shape(increment_eigenvalues)[:-1], tf.float64),
            tf.reduce_max(tf.abs(horizon_matrices[:, 1:]), axis=[-2, -1]),
        )
        monotonic = tf.reduce_all(
            increment_eigenvalues[..., 0] >= -1.0e-8 * scale
        )
        monotonic_all = monotonic_all and bool(monotonic.numpy())
        rows.append(
            {
                "seed": seed,
                "observation_sha256": _tensor_sha256(observations),
                "maximum_deterministic_residual": tf.reduce_max(tf.abs(residuals)),
                "horizons": {
                    str(horizon): summaries[index]
                    for index, horizon in enumerate(HORIZONS)
                },
                "information_increment_minimum_eigenvalues": increment_eigenvalues[..., 0],
                "information_nondecreasing": monotonic,
                "fine_finite_difference_step": 5.0e-5,
                "coarse_finite_difference_step": 1.0e-4,
                "mean_derivative_relative_step_gap": mean_derivative_relative_gap,
                "log_variance_derivative_relative_step_gap": (
                    log_variance_derivative_relative_gap
                ),
                "derivative_step_stability_passed": derivative_step_stability_passed,
                "t100_gate_passed": t100_pass,
            }
        )
        if checkpoint_root is not None:
            _write_json(
                checkpoint_root / f"design-seed-{seed[1]}.json",
                rows[-1],
            )
        print(f"DESIGN_SEED_COMPLETE {seed[1]}", flush=True)
    if all_t100 and monotonic_all:
        decision = "ADMIT_T100_TARGET_DESIGN_HORIZON"
    else:
        t200_pass = all(
            all(bool(item) for item in row["horizons"]["200"]["information_gate_passed"].numpy())
            for row in rows
        )
        decision = (
            "TARGET_DESIGN_HORIZON_REPLAN_REQUIRED"
            if t200_pass
            else "TARGET_BLOCKED_LIKELIHOOD_INFORMATION"
        )
    return {
        "design_points_source": points,
        "horizons": HORIZONS,
        "rows": rows,
        "all_t100_information_gates_passed": all_t100,
        "all_information_matrices_nondecreasing": monotonic_all,
        "decision": decision,
        "passed": decision == "ADMIT_T100_TARGET_DESIGN_HORIZON",
    }


def _run_prior_predictive() -> dict[str, Any]:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        structural_prior_predictive_tf,
    )

    @tf.function(jit_compile=True)
    def compiled() -> dict[str, tf.Tensor]:
        return dict(
            structural_prior_predictive_tf(
                batch_size=4096,
                horizon=200,
                seed=tf.constant(PRIOR_PREDICTIVE_SEED, tf.int32),
            )
        )

    result = compiled()
    rows: dict[str, Any] = {}
    for horizon in HORIZONS:
        states = result["states"][:, :horizon]
        observations = result["observations"][:, :horizon]
        values = tf.concat([tf.abs(states), tf.abs(observations)], axis=2)
        magnitude = tf.reduce_max(values, axis=[1, 2])
        finite = tf.reduce_all(tf.math.is_finite(values), axis=[1, 2])
        valid = tf.logical_and(finite, magnitude <= 1.0e6)
        sorted_magnitude = tf.sort(magnitude)
        indices = tf.constant(
            [int(0.50 * 4095), int(0.95 * 4095), int(0.99 * 4095), 4095],
            tf.int32,
        )
        rows[str(horizon)] = {
            "valid_fraction": tf.reduce_mean(tf.cast(valid, tf.float64)),
            "valid_count": tf.reduce_sum(tf.cast(valid, tf.int32)),
            "magnitude_q50_q95_q99_max": tf.gather(sorted_magnitude, indices),
            "passed": tf.reduce_mean(tf.cast(valid, tf.float64)) >= 0.99,
        }
    passed = all(bool(row["passed"].numpy()) for row in rows.values())
    output = {
        "batch_size": 4096,
        "seed": PRIOR_PREDICTIVE_SEED,
        "compiled_xla": True,
        "rows": rows,
        "maximum_deterministic_residual": tf.reduce_max(
            tf.abs(result["deterministic_residuals"])
        ),
        "passed": passed,
    }
    print("PRIOR_PREDICTIVE_COMPLETE", flush=True)
    return output


def _run_gpu_canary() -> dict[str, Any]:
    policy = configure_tensorflow_gpu_memory_growth(tf)
    devices = tf.config.list_logical_devices("GPU")
    if not devices:
        raise RuntimeError("trusted GPU canary requires a visible TensorFlow GPU")
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_TRUTH_PHYSICAL,
        simulate_structural_trajectories_tf,
        structural_truth_source,
        structural_ukf_likelihood_value_score_status,
    )

    physical = STRUCTURAL_TRUTH_PHYSICAL[None, :]
    _states, observations, _residuals = simulate_structural_trajectories_tf(
        physical,
        horizon=20,
        seed=tf.constant(FINAL_SEED, tf.int32),
    )

    @tf.function(
        input_signature=[tf.TensorSpec([2, 5], tf.float64)], jit_compile=True
    )
    def compiled(theta: tf.Tensor):
        return structural_ukf_likelihood_value_score_status(
            theta,
            observations=observations[0],
            principal_sqrt_backend="tensorflow_eigh",
        )

    truth = structural_truth_source()
    theta = tf.stack([truth, truth + 0.1], axis=0)
    value, score, status = compiled(theta)
    device_names = [tensor.device for tensor in (value, score, status["status_code"])]
    passed = (
        bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
        and bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
        and all("GPU" in name.upper() for name in device_names)
    )
    return {
        "passed": passed,
        "jit_compile": True,
        "logical_gpus": [device.name for device in devices],
        "output_devices": device_names,
        "memory_policy": policy,
        "value": value,
        "score": score,
        "status": status,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }


def _final_dataset() -> dict[str, Any]:
    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_TRUTH_PHYSICAL,
        simulate_structural_trajectories_tf,
    )

    states, observations, residuals = simulate_structural_trajectories_tf(
        STRUCTURAL_TRUTH_PHYSICAL[None, :],
        horizon=100,
        seed=tf.constant(FINAL_SEED, tf.int32),
    )
    return {
        "seed": FINAL_SEED,
        "horizon": 100,
        "time_order": "x0_from_initial_then_y0; transitions_and_observations_for_t1_to_t99",
        "states": states[0],
        "observations": observations[0],
        "state_sha256": _tensor_sha256(states[0]),
        "observation_sha256": _tensor_sha256(observations[0]),
        "maximum_deterministic_residual": tf.reduce_max(tf.abs(residuals)),
    }


def main() -> None:
    args = _parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if args.gpu_canary_only:
        gpu = _run_gpu_canary()
        result = {
            "schema": "bayesfilter.multimodel_neutra_p5_structural_gpu_canary.v1",
            "program_id": PROGRAM_ID,
            "gpu_canary": gpu,
            "passed": gpu["passed"],
            "elapsed_seconds": time.perf_counter() - started,
        }
        _write_json(output_root / "gpu_canary.json", result)
        return

    from bayesfilter.testing.structural_ukf_neutra_target_design_tf import (
        STRUCTURAL_PARAMETER_LOWER,
        STRUCTURAL_PARAMETER_NAMES,
        STRUCTURAL_PARAMETER_UPPER,
        STRUCTURAL_TRUTH_PHYSICAL,
        simulate_structural_trajectories_tf,
        structural_likelihood_information_tf,
        structural_negative_control_one_step_tf,
        structural_ukf_likelihood_value_score_status,
    )

    _write_json(
        output_root / "attempt_status.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p5_attempt_status.v1",
            "status": "STARTED",
            "program_id": PROGRAM_ID,
            "output_root": str(output_root),
        },
    )
    design = _run_design_audit(output_root)
    _write_json(output_root / "design_audit.json", design)
    prior_predictive = _run_prior_predictive()
    _write_json(output_root / "prior_predictive.json", prior_predictive)
    negative_control = structural_negative_control_one_step_tf()
    _write_json(output_root / "negative_control.json", dict(negative_control))
    negative_control_passed = (
        abs(float(negative_control["structural_innovation_variance"].numpy()[0]) - 0.6121674304) <= 5e-6
        and abs(float(negative_control["negative_control_innovation_variance"].numpy()[0]) - 0.6521674304) <= 5e-6
        and bool(tf.reduce_any(tf.not_equal(negative_control["negative_control_pointwise_residuals"], 0.0)).numpy())
        and float(negative_control["negative_control_k_variance_increment"].numpy()) == 0.04
    )
    admitted = design["passed"] and prior_predictive["passed"] and negative_control_passed
    final_dataset = _final_dataset() if admitted else None
    decision = (
        "ADMIT_STRUCTURAL_TARGET_DESIGN_READY_FOR_R1B"
        if admitted
        else design["decision"]
        if not design["passed"]
        else "TARGET_BLOCKED_PRIOR_PREDICTIVE_OR_NEGATIVE_CONTROL"
    )
    result = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_target_design.v1",
        "program_id": PROGRAM_ID,
        "decision": decision,
        "passed": admitted,
        "typed_target_signature_issued": False,
        "plan_file": PLAN_FILE,
        "result_file": RESULT_FILE,
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "device_intent": "CPU XLA target design; trusted GPU/XLA canary is a separate command",
        "parameter_names": STRUCTURAL_PARAMETER_NAMES,
        "parameter_lower": STRUCTURAL_PARAMETER_LOWER,
        "parameter_upper": STRUCTURAL_PARAMETER_UPPER,
        "truth_physical": STRUCTURAL_TRUTH_PHYSICAL,
        "truth_role": "synthetic_data_generation_and_design_center_only",
        "design_audit": design,
        "prior_predictive": prior_predictive,
        "negative_control": negative_control,
        "negative_control_passed": negative_control_passed,
        "final_dataset": final_dataset,
        "source_hashes": {
            "simulate_structural_trajectories_tf": _source_sha256(simulate_structural_trajectories_tf),
            "structural_likelihood_information_tf": _source_sha256(structural_likelihood_information_tf),
            "structural_ukf_likelihood_value_score_status": _source_sha256(structural_ukf_likelihood_value_score_status),
        },
        "nonclaims": [
            "no typed posterior identity issued in target-design rung",
            "no posterior correctness or global identifiability claim",
            "no filter exactness, HMC convergence, NeuTra, Zhao-Cui, calibration, or readiness claim",
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_json(output_root / "result.json", result)
    manifest = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_target_design_manifest.v1",
        "program_id": PROGRAM_ID,
        "git_commit": result["git_commit"],
        "command": " ".join(os.sys.argv),
        "output_root": str(output_root),
        "plan_file": PLAN_FILE,
        "result_file": str(output_root / "result.json"),
        "result_note": RESULT_FILE,
        "device_intent": result["device_intent"],
        "random_seeds": {
            "design": DESIGN_SEEDS,
            "prior_predictive": PRIOR_PREDICTIVE_SEED,
            "final_dataset": FINAL_SEED,
        },
        "wall_time_seconds": result["elapsed_seconds"],
    }
    _write_json(output_root / "run_manifest.json", manifest)
    hashes = {
        "schema": "bayesfilter.multimodel_neutra_p5_structural_hashes.v1",
        "artifacts": {
            "result.json": _sha256(output_root / "result.json"),
            "run_manifest.json": _sha256(output_root / "run_manifest.json"),
        },
    }
    _write_json(output_root / "artifact_hashes.json", hashes)
    _write_json(
        output_root / "attempt_status.json",
        {
            "schema": "bayesfilter.multimodel_neutra_p5_attempt_status.v1",
            "status": "COMPLETED",
            "program_id": PROGRAM_ID,
            "decision": decision,
            "passed": admitted,
            "result_sha256": hashes["artifacts"]["result.json"],
        },
    )


if __name__ == "__main__":
    main()
