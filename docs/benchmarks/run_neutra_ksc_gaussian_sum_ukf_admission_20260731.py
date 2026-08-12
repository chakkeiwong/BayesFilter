#!/usr/bin/env python3
"""Bounded Gaussian-sum KSC repair admission diagnostic.

This is a CPU/reference lane.  It evaluates the repaired bounded mixture
recurrence at the same frozen audit points and T=20 dense reference used by
the original KSC-UKF admission gate.  It never launches NeuTra training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")

ROOT = Path(__file__).resolve().parents[2]
PLAN = "docs/plans/bayesfilter-neutra-four-blocked-target-repair-and-admission-plan-2026-07-31.md"
PREFIX_HORIZON = 20
DENSE_RADIUS = 8.0
DENSE_ORDERS = (401, 601)
FD_STEPS = (3.0e-5, 1.0e-5)
VALUE_GAP_PER_OBSERVATION_MAX = 1.0e-3
SCORE_GAP_MAX = 1.0e-2
DENSE_VALUE_ORDER_GAP_PER_OBSERVATION_MAX = 2.0e-4
DENSE_FD_STEP_GAP_MAX = 2.0e-3
DENSE_FD_ORDER_GAP_MAX = 2.0e-3
# Keep the bounded repair within the declared CPU memory budget.  Larger caps
# are not a free admission ladder and require a separately reviewed campaign.
COMPONENT_CAPS = (7, 16, 32, 64, 128, 256)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "numpy"):
        return _json_ready(value.numpy())
    if hasattr(value, "tolist") and hasattr(value, "shape"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _tensor_hash(tf: Any, value: Any) -> str:
    return hashlib.sha256(bytes(tf.io.serialize_tensor(value).numpy())).hexdigest()


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def run(output_root: Path, *, gpu_canary: bool = False) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"fresh output root required: {output_root}")
    output_root.mkdir(parents=True)
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    if not gpu_canary:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import tensorflow as tf
    import tensorflow_probability as tfp

    memory_policy = None
    if gpu_canary:
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(True)

    from bayesfilter.highdim.sv_mixture_cut4 import (
        StochasticVolatilitySSM,
        ksc_1998_log_chi_square_mixture,
        scalar_sv_mixture_dense_reference,
    )
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        generate_frozen_exact_sv_dataset_tf,
        source_chart_physical_parameters,
    )
    from bayesfilter.testing.ksc_gaussian_sum_ukf_neutra_target_tf import (
        ksc_gaussian_sum_ukf_likelihood_value_score_status,
    )
    from bayesfilter.testing.ksc_ukf_neutra_target_tf import (
        KSC_UKF_RAW_OBSERVATION_SHA256,
        KSC_UKF_STATE_SHA256,
        transformed_ksc_observations,
    )

    states, raw_observations = generate_frozen_exact_sv_dataset_tf()
    state_hash = _tensor_hash(tf, states)
    observation_hash = _tensor_hash(tf, raw_observations)
    if state_hash != KSC_UKF_STATE_SHA256:
        raise RuntimeError("frozen KSC state hash mismatch")
    if observation_hash != KSC_UKF_RAW_OBSERVATION_SHA256:
        raise RuntimeError("frozen KSC observation hash mismatch")

    fixed = tf.constant(
        [[-1.0, -1.0], [-1.0, 1.0], [0.0, 0.0], [1.0, -1.0], [1.0, 1.0]],
        tf.float64,
    )
    normal = tfp.distributions.Normal(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64))
    truth = normal.quantile(
        tf.constant([(0.6 - 0.1) / 0.8, (0.4 - 0.1) / 0.8], tf.float64)
    )
    audit_points = tf.concat((fixed, truth[None, :]), axis=0)
    prefix_observations = raw_observations[:PREFIX_HORIZON]
    transformed = transformed_ksc_observations(prefix_observations)
    mixture = ksc_1998_log_chi_square_mixture()

    gamma, beta = source_chart_physical_parameters(audit_points)
    legacy_theta = tf.stack((normal.quantile(gamma), tf.math.log(beta)), axis=1)
    model = StochasticVolatilitySSM(sigma=1.0)

    def dense_value(theta_row: Any, *, order: int) -> Any:
        with tf.device("/CPU:0"):
            return scalar_sv_mixture_dense_reference(
                model,
                theta_row,
                prefix_observations,
                order=order,
                radius=DENSE_RADIUS,
            ).log_likelihood

    dense_values = {
        order: tf.stack([dense_value(row, order=order) for row in tf.unstack(legacy_theta)])
        for order in DENSE_ORDERS
    }

    def dense_fd(order: int, epsilon: float) -> Any:
        rows = []
        for row in tf.unstack(audit_points):
            coordinates = []
            for coordinate in range(2):
                direction = tf.one_hot(coordinate, 2, dtype=tf.float64)
                plus = dense_value(
                    tf.stack(
                        (
                            normal.quantile(source_chart_physical_parameters((row + epsilon * direction)[None, :])[0])[0],
                            tf.math.log(source_chart_physical_parameters((row + epsilon * direction)[None, :])[1])[0],
                        )
                    ),
                    order=order,
                )
                minus = dense_value(
                    tf.stack(
                        (
                            normal.quantile(source_chart_physical_parameters((row - epsilon * direction)[None, :])[0])[0],
                            tf.math.log(source_chart_physical_parameters((row - epsilon * direction)[None, :])[1])[0],
                        )
                    ),
                    order=order,
                )
                coordinates.append((plus - minus) / (2.0 * epsilon))
            rows.append(tf.stack(coordinates))
        return tf.stack(rows)

    dense_fd_values = {
        (order, step): dense_fd(order, step)
        for order, step in ((401, 3.0e-5), (401, 1.0e-5), (601, 3.0e-5))
    }
    dense_order_gap = float(
        tf.reduce_max(tf.abs(dense_values[401] - dense_values[601])).numpy()
        / PREFIX_HORIZON
    )
    dense_fd_step_gap = float(
        tf.reduce_max(tf.abs(dense_fd_values[(401, 3.0e-5)] - dense_fd_values[(401, 1.0e-5)])).numpy()
    )
    dense_fd_order_gap = float(
        tf.reduce_max(tf.abs(dense_fd_values[(401, 3.0e-5)] - dense_fd_values[(601, 3.0e-5)])).numpy()
    )
    dense_reference_passed = bool(
        dense_order_gap <= DENSE_VALUE_ORDER_GAP_PER_OBSERVATION_MAX
        and dense_fd_step_gap <= DENSE_FD_STEP_GAP_MAX
        and dense_fd_order_gap <= DENSE_FD_ORDER_GAP_MAX
        and all(bool(tf.reduce_all(tf.math.is_finite(x)).numpy()) for x in dense_values.values())
        and all(bool(tf.reduce_all(tf.math.is_finite(x)).numpy()) for x in dense_fd_values.values())
    )

    cap_rows = []
    for cap in COMPONENT_CAPS:
        value, score, status = ksc_gaussian_sum_ukf_likelihood_value_score_status(
            audit_points,
            transformed_observations=transformed,
            mixture_weights=mixture.weights,
            mixture_means=mixture.means,
            mixture_variances=mixture.variances,
            component_cap=cap,
        )
        reversed_value, reversed_score, reversed_status = ksc_gaussian_sum_ukf_likelihood_value_score_status(
            tf.reverse(audit_points, axis=(0,)),
            transformed_observations=transformed,
            mixture_weights=mixture.weights,
            mixture_means=mixture.means,
            mixture_variances=mixture.variances,
            component_cap=cap,
        )
        value_gap = tf.abs(value - dense_values[601]) / PREFIX_HORIZON
        score_gap = tf.reduce_max(tf.abs(score - dense_fd_values[(601, 3.0e-5)]), axis=1)
        valid = bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())
        row = {
            "component_cap": cap,
            "value": value,
            "score": score,
            "value_gap_per_observation": value_gap,
            "score_gap": score_gap,
            "minimum_retained_mass_fraction": status["minimum_retained_mass_fraction"],
            "minimum_premerge_top_weight_mass_fraction": status["minimum_premerge_top_weight_mass_fraction"],
            "maximum_active_component_count": status["maximum_active_component_count"],
            "valid": valid,
            "permutation_value_gap": float(tf.reduce_max(tf.abs(value - tf.reverse(reversed_value, axis=(0,)))).numpy()),
            "permutation_score_gap": float(tf.reduce_max(tf.abs(score - tf.reverse(reversed_score, axis=(0,)))).numpy()),
            "permutation_status_equal": bool(tf.reduce_all(tf.equal(status["status_code"], tf.reverse(reversed_status["status_code"], axis=(0,)))).numpy()),
        }
        row["passed"] = bool(
            dense_reference_passed
            and valid
            and float(tf.reduce_max(value_gap).numpy()) <= VALUE_GAP_PER_OBSERVATION_MAX
            and float(tf.reduce_max(score_gap).numpy()) <= SCORE_GAP_MAX
            and row["permutation_value_gap"] <= 1.0e-12
            and row["permutation_score_gap"] <= 1.0e-12
            and row["permutation_status_equal"]
        )
        cap_rows.append(row)

    canary = None
    if gpu_canary:
        canary_cap = 32

        @tf.function(
            input_signature=[tf.TensorSpec([None, 2], tf.float64)],
            jit_compile=True,
            reduce_retracing=True,
        )
        def compiled_candidate(theta):
            return ksc_gaussian_sum_ukf_likelihood_value_score_status(
                theta,
                transformed_observations=transformed,
                mixture_weights=mixture.weights,
                mixture_means=mixture.means,
                mixture_variances=mixture.variances,
                component_cap=canary_cap,
            )

        with tf.device("/GPU:0"):
            canary_value, canary_score, canary_status = compiled_candidate(audit_points)
        cpu_row = next(row for row in cap_rows if row["component_cap"] == canary_cap)
        canary = {
            "component_cap": canary_cap,
            "value": canary_value,
            "score": canary_score,
            "status_code": canary_status["status_code"],
            "valid": bool(tf.reduce_all(canary_status["valid_pre_regularized_score"]).numpy()),
            "value_gap_to_cpu": float(tf.reduce_max(tf.abs(canary_value - tf.convert_to_tensor(cpu_row["value"], tf.float64))).numpy()),
            "score_gap_to_cpu": float(tf.reduce_max(tf.abs(canary_score - tf.convert_to_tensor(cpu_row["score"], tf.float64))).numpy()),
            "xla": True,
            "device": "/GPU:0",
        }

    passed_caps = [row["component_cap"] for row in cap_rows if row["passed"]]
    gpu_canary_passed = bool(
        gpu_canary
        and canary is not None
        and canary["valid"]
        and canary["value_gap_to_cpu"] <= 1.0e-10
        and canary["score_gap_to_cpu"] <= 1.0e-8
    )
    if gpu_canary:
        decision = (
            "ADMIT_KSC_GAUSSIAN_SUM_UKF_GPU_CANARY"
            if passed_caps and gpu_canary_passed
            else "KEEP_KSC_UKF_TARGET_BLOCKED_GPU_CANARY"
        )
    else:
        decision = (
            "CPU_FILTER_ADMITTED_KSC_GAUSSIAN_SUM_UKF_GPU_CANARY_PENDING"
            if passed_caps
            else "KEEP_KSC_UKF_TARGET_BLOCKED_GAUSSIAN_SUM_REPAIR_NOT_ADMITTED"
        )
    result = {
        "schema": "bayesfilter.neutra_ksc_gaussian_sum_ukf_admission.v1",
        "status": "TERMINAL_KSC_GAUSSIAN_SUM_UKF_REPAIR_ADMISSION",
        "decision": decision,
        "target": {
            "horizon": PREFIX_HORIZON,
            "raw_observation_sha256": observation_hash,
            "state_sha256": state_hash,
            "transform": "log(y^2 + 1e-8)",
            "audit_point_count": int(audit_points.shape[0]),
            "mixture_component_count": int(mixture.weights.shape[0]),
        },
        "candidate": {
            "route": "bounded_deterministic_mass_preserving_clustered_gaussian_sum_ukf",
            "component_caps": COMPONENT_CAPS,
            "rows": cap_rows,
            "passed_caps": passed_caps,
        },
        "gpu_xla_canary": canary,
        "gpu_xla_canary_passed": gpu_canary_passed,
        "dense_reference": {
            "orders": DENSE_ORDERS,
            "radius": DENSE_RADIUS,
            "value_order_gap_per_observation": dense_order_gap,
            "fd_step_gap": dense_fd_step_gap,
            "fd_order_gap": dense_fd_order_gap,
            "passed": dense_reference_passed,
        },
        "thresholds": {
            "value_gap_per_observation_max": VALUE_GAP_PER_OBSERVATION_MAX,
            "score_gap_max": SCORE_GAP_MAX,
            "dense_value_order_gap_per_observation_max": DENSE_VALUE_ORDER_GAP_PER_OBSERVATION_MAX,
            "dense_fd_step_gap_max": DENSE_FD_STEP_GAP_MAX,
            "dense_fd_order_gap_max": DENSE_FD_ORDER_GAP_MAX,
        },
        "training_launched": False,
        "nonclaims": [
            "not exact-SV evidence",
            "not NeuTra or HMC evidence",
            "no ranking beyond the frozen admission screen",
        ],
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "plan": PLAN,
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python_executable": sys.executable,
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "device": "/GPU:0 canary plus /CPU:0 dense reference" if gpu_canary else "CPU reference; GPU intentionally hidden",
            "jit_compile": bool(gpu_canary),
            "gpu_memory_policy": memory_policy,
            "tf32_execution_enabled": bool(gpu_canary),
            "dtype": "float64",
            "random_seeds": {"dataset": 81101, "filter": "deterministic"},
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(output_root),
            "result_file": str(output_root / "result.json"),
        },
    }
    _write_new_json(output_root / "result.json", result)
    _write_new_json(output_root / "run_manifest.json", result["run_manifest"])
    hashes = {
        str(path.relative_to(output_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new_json(output_root / "artifact_hashes.json", {"schema": "bayesfilter.neutra_ksc_gaussian_sum_ukf_artifact_hashes.v1", "artifacts": hashes})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-canary", action="store_true")
    args = parser.parse_args()
    result = run(args.output_root, gpu_canary=args.gpu_canary)
    print(json.dumps({"decision": result["decision"], "passed_caps": result["candidate"]["passed_caps"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
