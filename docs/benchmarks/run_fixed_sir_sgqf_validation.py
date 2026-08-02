#!/usr/bin/env python3
"""Validate the source-order fixed Austria SIR SGQF value route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _json_safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _git_diff_hash() -> str:
    result = subprocess.run(
        ["git", "diff", "--no-ext-diff", "--binary"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/bayesfilter-mpl")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import tensorflow as tf
    import tensorflow_probability as tfp

    memory_policy: dict[str, object] | None = None
    physical = tf.config.list_physical_devices("GPU")
    if args.mode == "gpu":
        from bayesfilter.runtime.gpu_memory_policy import (
            configure_tensorflow_gpu_memory_growth,
        )

        memory_policy = dict(
            configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        )
        tf.config.set_soft_device_placement(False)
        logical = tf.config.list_logical_devices("GPU")
        if not logical:
            raise RuntimeError("fixed SIR GPU validation requires a logical GPU")
    else:
        logical = tf.config.list_logical_devices("GPU")
        if physical or logical:
            raise RuntimeError("CPU mode must hide all GPU devices before TensorFlow import")

    from bayesfilter.highdim.fixed_sir_sgqf_tf import (
        fixed_sir_sgqf_value_only_status,
        make_fixed_sir_sgqf_route,
    )

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    route = make_fixed_sir_sgqf_route()

    if args.mode == "cpu":

        @tf.function(jit_compile=True)
        def cpu_xla(observations):
            with tf.device("/CPU:0"):
                return fixed_sir_sgqf_value_only_status(observations)

        cpu_value, cpu_status = cpu_xla(route.observations)
        cpu_value_float = float(cpu_value.numpy())
        cpu_status_payload = _json_safe(cpu_status)
        cpu_device = str(cpu_value.device)
        if "CPU" not in cpu_device.upper():
            raise RuntimeError(f"CPU reference escaped CPU placement: {cpu_device}")
    else:
        if args.cpu_reference is None:
            raise ValueError("GPU mode requires --cpu-reference")
        reference = json.loads(args.cpu_reference.read_text(encoding="utf-8"))
        if reference.get("route_identity") != route.route_identity:
            raise ValueError("CPU reference route identity mismatch")
        cpu_reference = reference.get("cpu_xla", {})
        cpu_device = str(cpu_reference.get("result_device", ""))
        cpu_status_payload = dict(cpu_reference.get("status", {}))
        cpu_value_float = float(cpu_reference["value"])
        if "CPU" not in cpu_device.upper():
            raise ValueError("CPU reference artifact is not CPU-placed")
        if int(cpu_status_payload.get("status_code", 1)) != 0:
            raise ValueError("CPU reference artifact failed covariance/status gate")

    gpu_payload: dict[str, object] | None = None
    hard_vetoes: list[str] = []
    if int(cpu_status_payload["status_code"]) != 0:
        hard_vetoes.append("cpu_covariance_or_finite_status_failed")
    if args.mode == "gpu":

        @tf.function(jit_compile=True)
        def gpu_xla(observations):
            with tf.device("/GPU:0"):
                return fixed_sir_sgqf_value_only_status(observations)

        gpu_value, gpu_status = gpu_xla(route.observations)
        gpu_device = str(gpu_value.device)
        absolute_difference = abs(float(gpu_value.numpy()) - cpu_value_float)
        tolerance = 1.0e-9 * max(1.0, abs(cpu_value_float))
        allocator = tf.config.experimental.get_memory_info("GPU:0")
        if "GPU" not in gpu_device.upper():
            hard_vetoes.append("gpu_result_not_placed_on_gpu")
        if int(gpu_status["status_code"].numpy()) != 0:
            hard_vetoes.append("gpu_covariance_or_finite_status_failed")
        if absolute_difference > tolerance:
            hard_vetoes.append("cpu_gpu_value_parity_failed")
        gpu_payload = {
            "value": float(gpu_value.numpy()),
            "status": _json_safe(gpu_status),
            "result_device": gpu_device,
            "jit_compile": True,
            "absolute_difference_from_cpu": absolute_difference,
            "parity_tolerance": tolerance,
            "allocator_bytes": {key: int(value) for key, value in allocator.items()},
        }

    comparator_payload: dict[str, object] | None = None
    if args.mode == "cpu" and args.run_comparators:
        from bayesfilter.testing.sir_filter_neutra_target_design_tf import (
            sir_bootstrap_pf_log_likelihood_tf,
        )

        @tf.function(jit_compile=True)
        def compiled_pf(observations):
            return sir_bootstrap_pf_log_likelihood_tf(
                tf.zeros([3], dtype=tf.float64),
                observations=observations,
                particle_count=args.pf_particles,
                replicate_count=args.pf_replicates,
                seed=tf.constant([20260722, 903], dtype=tf.int32),
            )

        pf_values = compiled_pf(route.observations)
        pf_mean = tf.reduce_mean(pf_values)
        pf_standard_error = tf.math.reduce_std(pf_values) / tf.sqrt(
            tf.cast(args.pf_replicates, tf.float64)
        )
        comparator_payload = {
            "role": "explanatory_value_diagnostics_only",
            "bootstrap_pf": {
                "particle_count": args.pf_particles,
                "replicate_count": args.pf_replicates,
                "seed": [20260722, 903],
                "values": _json_safe(pf_values),
                "mean": float(pf_mean.numpy()),
                "standard_error_of_mean": float(pf_standard_error.numpy()),
                "absolute_mean_gap_from_sgqf": float(
                    tf.abs(pf_mean - cpu_value_float).numpy()
                ),
                "inference_status": (
                    "descriptive_only_few_replicates_no_ranking_or_accuracy_promotion"
                ),
            },
        }

    payload: dict[str, object] = {
        "schema_version": "bayesfilter.fixed_sir_sgqf_validation.v2",
        "status": "PASS" if not hard_vetoes else "BLOCKED",
        "mode": args.mode,
        "row_id": route.manifest["row_id"],
        "route_identity": route.route_identity,
        "route_manifest": dict(route.manifest),
        "cpu_xla": {
            "value": cpu_value_float,
            "status": cpu_status_payload,
            "result_device": cpu_device,
            "jit_compile": True,
            "reference_artifact": (
                _display_path(args.cpu_reference)
                if args.cpu_reference is not None
                else None
            ),
        },
        "gpu_xla": gpu_payload,
        "comparators": comparator_payload,
        "hard_vetoes": hard_vetoes,
        "engineering_decision": (
            "route_executes_full_T20_under_requested_XLA_device"
            if not hard_vetoes
            else "route_failed_requested_engineering_gate"
        ),
        "numerical_decision": (
            "finite_covariance_valid_level2_candidate"
            if not hard_vetoes
            else "numerical_candidate_blocked"
        ),
        "scientific_decision": (
            "value_only_viable_candidate_not_accuracy_or_superiority_evidence"
        ),
        "nonclaims": [
            "no exact-likelihood claim",
            "no statistically supported method ranking",
            "no SGQF superiority claim",
            "no score because the fixed source row has no free theta",
            "PF differences are descriptive only",
            "SIR-UKF is owner-excluded and was not evaluated",
        ],
        "owner_exclusions": {
            "SIR-UKF": "OWNER_EXCLUDED_METHOD_NOT_APPLICABLE",
        },
        "memory_policy": memory_policy,
        "physical_gpus": [device.name for device in physical],
        "logical_gpus": [device.name for device in logical],
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_diff_sha256": _git_diff_hash(),
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "tensorflow_probability": tfp.__version__,
            "platform": platform.platform(),
            "host": platform.node(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
            "tf_force_gpu_allow_growth": os.environ.get(
                "TF_FORCE_GPU_ALLOW_GROWTH", "unset"
            ),
            "started_utc": started_utc,
            "wall_seconds": time.perf_counter() - started,
            "plan": (
                "docs/plans/bayesfilter-sgqf-whole-highdim-leaderboard-repair-"
                "master-program-2026-07-22.md"
            ),
            "result": _display_path(args.output_root / "result.json"),
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    result_path = args.output_root / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {result_path}")
    result_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if hard_vetoes:
        raise RuntimeError(f"fixed SIR SGQF validation vetoes: {hard_vetoes}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-comparators", action="store_true")
    parser.add_argument("--cpu-reference", type=Path, default=None)
    parser.add_argument("--pf-particles", type=int, default=2048)
    parser.add_argument("--pf-replicates", type=int, default=8)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.mode == "gpu" and args.run_comparators:
        parser.error("comparators are CPU-only diagnostics in this runner")
    if args.mode == "cpu" and args.cpu_reference is not None:
        parser.error("--cpu-reference applies only to GPU mode")
    if args.pf_particles < 2 or args.pf_replicates < 2:
        parser.error("PF diagnostics require at least 2 particles and 2 replicates")
    _run(args)


if __name__ == "__main__":
    main()
