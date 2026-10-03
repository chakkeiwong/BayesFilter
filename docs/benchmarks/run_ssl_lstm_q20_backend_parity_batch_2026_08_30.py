#!/usr/bin/env python3
"""Run a bounded same-input q=20 compiled/strict backend parity check."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json_safe(value: Any, tf: Any) -> Any:
    if tf.is_tensor(value):
        return _json_safe(value.numpy(), tf)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, tf) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, tf) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist(), tf)
    if hasattr(value, "item"):
        return _json_safe(value.item(), tf)
    return str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RuntimeError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _git(command: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            tuple(command), cwd=ROOT, text=True, stderr=subprocess.STDOUT
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"unavailable:{type(exc).__name__}"


def _nvidia_snapshot() -> Mapping[str, Any]:
    command = (
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        return {"command": list(command), "rows": output.strip().splitlines()}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"command": list(command), "error": type(exc).__name__}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, choices=(8, 32), default=8)
    return parser.parse_args()


def _target_program(
    tf: Any, backend: str, batch_size: int, *, jit_compile: bool
) -> tuple[Any, Any]:
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    bridge = make_q20_tempered_bridge(
        20, jit_compile=jit_compile, principal_sqrt_backend=backend
    )

    @tf.function(
        input_signature=[tf.TensorSpec([batch_size, 4], tf.float64)],
        jit_compile=jit_compile,
        reduce_retracing=False,
    )
    def program(theta: Any) -> tuple[Any, Any, Any]:
        return bridge.value_score_status(theta, tf.constant(0.5, tf.float64))

    return bridge, program


def _call_payload(tf: Any, rows: tuple[Any, Any, Mapping[str, Any]]) -> Mapping[str, Any]:
    value, score, status = rows
    valid_key = "bridge_valid" if "bridge_valid" in status else "valid_pre_regularized_score"
    valid = tf.convert_to_tensor(status[valid_key], tf.bool)
    status_code = tf.convert_to_tensor(status["status_code"], tf.int32)
    finite = tf.reduce_all(
        tf.concat(
            (
                tf.reshape(tf.math.is_finite(value), [-1]),
                tf.reshape(tf.math.is_finite(score), [-1]),
                tf.reshape(valid, [-1]),
            ),
            axis=0,
        )
    )
    hard_valid = tf.reduce_all(tf.logical_and(valid, tf.equal(status_code, 0)))
    return {
        "value": value,
        "score": score,
        "status_code": status_code,
        "valid": valid,
        "finite": bool(finite.numpy()),
        "hard_valid": bool(hard_valid.numpy()),
        "valid_count": int(tf.reduce_sum(tf.cast(valid, tf.int32)).numpy()),
    }


def _compare(tf: Any, custom: Mapping[str, Any], strict: Mapping[str, Any]) -> Mapping[str, Any]:
    value_delta = tf.abs(strict["value"] - custom["value"])
    score_delta = tf.abs(strict["score"] - custom["score"])
    score_scale = tf.maximum(tf.abs(custom["score"]), tf.constant(1.0, tf.float64))
    value_max = float(tf.reduce_max(value_delta).numpy())
    score_max = float(tf.reduce_max(score_delta).numpy())
    score_rel = float(tf.reduce_max(score_delta / score_scale).numpy())
    value_atol = 1.0e-8
    score_atol = 1.0e-7
    score_rtol = 1.0e-7
    score_ok = bool(
        tf.reduce_all(
            tf.logical_or(
                score_delta <= tf.constant(score_atol, tf.float64),
                score_delta / score_scale <= tf.constant(score_rtol, tf.float64),
            )
        ).numpy()
    )
    status_equal = bool(
        tf.reduce_all(strict["status_code"] == custom["status_code"]).numpy()
    ) and bool(tf.reduce_all(strict["valid"] == custom["valid"]).numpy())
    passed = (
        custom["finite"]
        and strict["finite"]
        and custom["hard_valid"]
        and strict["hard_valid"]
        and value_max <= value_atol
        and score_ok
        and status_equal
    )
    return {
        "passed": passed,
        "value_max_abs": value_max,
        "score_max_abs": score_max,
        "score_max_relative_scaled": score_rel,
        "value_atol": value_atol,
        "score_atol": score_atol,
        "score_rtol": score_rtol,
        "status_equal": status_equal,
    }


def main() -> int:
    args = _args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").strip().lower() != "true":
        raise RuntimeError("GPU parity requires TF_FORCE_GPU_ALLOW_GROWTH=true")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") in {"", "-1"}:
        raise RuntimeError("GPU parity requires one explicit visible GPU")

    import tensorflow as tf

    # This must precede all BayesFilter imports: module-level tensors can
    # otherwise initialize a logical GPU before the allocator policy is set.
    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError("parity requires exactly one visible logical GPU")

    started = time.monotonic()
    theta = tf.random.stateless_normal(
        [args.batch_size, 4], tf.constant((20260830, 62001), tf.int32), dtype=tf.float64
    )
    # Keep the fixed comparison bank in a numerically benign neighborhood of
    # the prior center while retaining non-identical rows.
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER

    theta = tf.convert_to_tensor(PRIOR_CENTER, tf.float64)[tf.newaxis, :] + 0.02 * theta
    # The repository custom op has no XLA GPU kernel.  Keep it in its
    # supported graph mode for semantic comparison; the strict route remains
    # the XLA execution route used by C2.
    custom, custom_program = _target_program(
        tf, "compiled_custom_op", args.batch_size, jit_compile=False
    )
    strict, strict_program = _target_program(
        tf, "tensorflow_eigh_strict", args.batch_size, jit_compile=True
    )
    if custom.target_signature != strict.target_signature:
        raise RuntimeError("backend target signatures differ")
    custom_started = time.monotonic()
    custom_rows = _call_payload(tf, custom_program(theta))
    custom_elapsed = time.monotonic() - custom_started
    strict_started = time.monotonic()
    strict_rows = _call_payload(tf, strict_program(theta))
    strict_elapsed = time.monotonic() - strict_started
    parity = _compare(tf, custom_rows, strict_rows)
    if not parity["passed"]:
        raise RuntimeError(f"backend parity failed: {parity}")

    manifest: dict[str, Any] = {
        "schema": "bayesfilter.ssl_lstm_q20.backend_parity_batch.v1",
        "status": "PASS_Q20_BACKEND_PARITY_BATCH",
        "batch_size": args.batch_size,
        "beta": 0.5,
        "target_signature": custom.target_signature,
        "custom_backend": "compiled_custom_op",
        "strict_backend": "tensorflow_eigh_strict",
        "custom_jit_compile": False,
        "strict_jit_compile": True,
        "custom_call": {"elapsed_seconds": custom_elapsed, **custom_rows},
        "strict_call": {"elapsed_seconds": strict_elapsed, **strict_rows},
        "parity": parity,
        "memory_policy": memory_policy,
        "logical_gpus": [str(device.name) for device in logical],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tf_force_gpu_allow_growth": os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"),
        "tf32_execution_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "jit_compile": True,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "tensorflow": tf.__version__,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "gpu_snapshot_before": _nvidia_snapshot(),
        "git_commit": _git(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _git(("git", "status", "--porcelain")),
        "command": sys.argv,
        "output_dir": str(output_dir),
        "wall_time_seconds": time.monotonic() - started,
        "source_hashes": {
            path: _sha256(ROOT / path)
            for path in (
                "bayesfilter/inference/tempered_target_tf.py",
                "bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py",
                "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py",
                "bayesfilter/runtime/gpu_memory_policy.py",
                "docs/benchmarks/run_ssl_lstm_q20_backend_parity_batch_2026_08_30.py",
            )
        },
        "nonclaims": [
            "batch parity and backend feasibility only",
            "no target default promotion",
            "no whitening, mode discovery, posterior, HMC, superiority, or scaling claim",
        ],
    }
    safe = _json_safe(manifest, tf)
    safe["manifest_hash"] = _stable_hash(safe)
    _write_json(output_dir / "run_manifest.json", safe)
    print(json.dumps({"status": safe["status"], "output_dir": str(output_dir)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Failure artifacts are intentionally best-effort and never replace a
        # complete manifest. The launcher preserves the process traceback.
        try:
            parsed = _args()
            path = parsed.output_dir.expanduser().resolve() / "failure.json"
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                _write_json(
                    path,
                    {
                        "schema": "bayesfilter.ssl_lstm_q20.backend_parity_batch.v1",
                        "status": "FAIL_Q20_BACKEND_PARITY_BATCH",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "command": sys.argv,
                    },
                )
        finally:
            raise
