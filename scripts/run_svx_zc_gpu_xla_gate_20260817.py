#!/usr/bin/env python3
"""Run the current SVX-ZC value/score GPU-XLA parity gate."""

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
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
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
    if hasattr(value, "tolist"):
        return _json_ready(value.tolist())
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"fresh output root required: {output_root}")
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()

    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)

    from bayesfilter.ssm import stable_ssm_target_signature
    from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (
        make_actual_sv_zc_neutra_adapter,
    )

    adapter = make_actual_sv_zc_neutra_adapter()
    target_signature = stable_ssm_target_signature(adapter.contract)
    capability = adapter.value_score_capability()
    theta = tf.constant(
        [
            [0.6, 0.4],
            [0.2, -0.3],
            [0.2533471031357998, -0.4054651081081643],
        ],
        tf.float64,
    )

    with tf.device("/CPU:0"):
        cpu_value, cpu_score, cpu_status = (
            adapter.neutra_batch_log_prob_and_grad_status(theta)
        )
        epsilon = tf.constant(1.0e-6, tf.float64)
        fd_columns = []
        for coordinate in range(2):
            direction = tf.one_hot(coordinate, 2, dtype=tf.float64)[None, :]
            plus = adapter.log_prob(theta + epsilon * direction)
            minus = adapter.log_prob(theta - epsilon * direction)
            fd_columns.append((plus - minus) / (2.0 * epsilon))
        finite_difference = tf.stack(fd_columns, axis=1)

    @tf.function(
        input_signature=[tf.TensorSpec([None, 2], tf.float64)],
        jit_compile=True,
        reduce_retracing=True,
    )
    def compiled(values: tf.Tensor):
        return adapter.neutra_batch_log_prob_and_grad_status(values)

    with tf.device("/GPU:0"):
        gpu_value, gpu_score, gpu_status = compiled(theta)
        reverse_value, reverse_score, reverse_status = compiled(
            tf.reverse(theta, axis=(0,))
        )

    value_gap = float(tf.reduce_max(tf.abs(cpu_value - gpu_value)).numpy())
    score_gap = float(tf.reduce_max(tf.abs(cpu_score - gpu_score)).numpy())
    fd_gap = float(tf.reduce_max(tf.abs(cpu_score - finite_difference)).numpy())
    permutation_value_gap = float(
        tf.reduce_max(tf.abs(gpu_value - tf.reverse(reverse_value, axis=(0,)))).numpy()
    )
    permutation_score_gap = float(
        tf.reduce_max(tf.abs(gpu_score - tf.reverse(reverse_score, axis=(0,)))).numpy()
    )
    required_status = {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    }
    status_fields_present = required_status <= set(gpu_status)
    status_all_zero = bool(
        tf.reduce_all(tf.equal(gpu_status["status_code"], 0)).numpy()
    )
    score_valid = bool(
        tf.reduce_all(gpu_status["valid_pre_regularized_score"]).numpy()
    )
    reverse_status_equal = bool(
        tf.reduce_all(
            tf.equal(
                gpu_status["status_code"],
                tf.reverse(reverse_status["status_code"], axis=(0,)),
            )
        ).numpy()
    )
    finite = bool(tf.reduce_all(tf.math.is_finite(gpu_value)).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(gpu_score)).numpy()
    )
    thresholds = {
        "cpu_gpu_value_max_abs": 1.0e-10,
        "cpu_gpu_score_max_abs": 1.0e-9,
        "same_program_fd_max_abs": 2.0e-6,
        "permutation_value_max_abs": 1.0e-12,
        "permutation_score_max_abs": 1.0e-12,
    }
    passed = bool(
        finite
        and status_fields_present
        and status_all_zero
        and score_valid
        and reverse_status_equal
        and value_gap <= thresholds["cpu_gpu_value_max_abs"]
        and score_gap <= thresholds["cpu_gpu_score_max_abs"]
        and fd_gap <= thresholds["same_program_fd_max_abs"]
        and permutation_value_gap <= thresholds["permutation_value_max_abs"]
        and permutation_score_gap <= thresholds["permutation_score_max_abs"]
    )
    try:
        allocator = tf.config.experimental.get_memory_info("GPU:0")
    except (ValueError, RuntimeError):
        allocator = {"current": None, "peak": None}

    result = {
        "schema": "bayesfilter.svx_zc.current_gpu_xla_gate.v1",
        "decision": "PASS_CURRENT_SVX_ZC_GPU_XLA_GATE" if passed else "FAIL_CURRENT_SVX_ZC_GPU_XLA_GATE",
        "passed": passed,
        "target_scope": adapter.target_scope,
        "target_signature": target_signature,
        "adapter_signature": adapter.adapter_signature(),
        "score_backend_id": adapter.score_backend_id,
        "runtime_autodiff_for_hmc": adapter.runtime_autodiff_for_hmc,
        "current_capability": {
            "value_score_authority": capability.value_score_authority,
            "xla_hmc_ready": capability.xla_hmc_ready,
            "full_chain_xla_diagnostic_ready": capability.full_chain_xla_diagnostic_ready,
            "runtime_backend": capability.runtime_backend,
        },
        "gates": {
            "finite": finite,
            "status_fields_present": status_fields_present,
            "status_all_zero": status_all_zero,
            "score_valid": score_valid,
            "cpu_gpu_value_max_abs": value_gap,
            "cpu_gpu_score_max_abs": score_gap,
            "same_program_fd_max_abs": fd_gap,
            "permutation_value_max_abs": permutation_value_gap,
            "permutation_score_max_abs": permutation_score_gap,
            "permutation_status_equal": reverse_status_equal,
        },
        "thresholds": thresholds,
        "gpu_allocator_bytes": allocator,
        "gpu_memory_policy": memory_policy,
        "nonclaims": [
            "current target GPU/XLA mechanics only",
            "does not establish exact filtering or exact posterior correctness",
            "does not by itself rewrite repository capability or registry status",
        ],
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "python": sys.executable,
            "platform": platform.platform(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
            "device": "/GPU:0 mapped from CUDA_VISIBLE_DEVICES",
            "dtype": "float64",
            "jit_compile": True,
            "tf32_execution_enabled": True,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "wall_time_seconds": time.monotonic() - started,
            "output_root": str(output_root),
        },
    }
    _write_new(output_root / "result.json", result)
    _write_new(output_root / "run_manifest.json", result["run_manifest"])
    hashes = {
        str(path.relative_to(output_root)): _sha256(path)
        for path in sorted(output_root.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_new(
        output_root / "artifact_hashes.json",
        {"schema": "bayesfilter.svx_zc.current_gpu_xla_hashes.v1", "artifacts": hashes},
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.output_root)
    print(json.dumps({"decision": result["decision"], "gates": result["gates"]}, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

