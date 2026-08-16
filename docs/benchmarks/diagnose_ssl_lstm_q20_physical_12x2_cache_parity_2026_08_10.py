#!/usr/bin/env python3
"""Diagnose terminal cache parity for the failed immutable 12x2 canary."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANARY_RUNNER = ROOT / (
    "docs/benchmarks/"
    "run_ssl_lstm_q20_physical_distributed_replica_12x2_canary_2026_08_10.py"
)
FAILED_ROOT = ROOT / (
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r5-topology-12x2-canary"
)
OUTPUT_ROOT = ROOT / (
    "docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/"
    "r5-cache-parity-diagnosis"
)
FINAL = OUTPUT_ROOT / "diagnosis.json"
FAILED_CANARY_SHA256 = "08e9d29fee2af56aeadc3622f01a6f97487384c4446e01f16fc00dedb2ecb3ac"


class CacheParityDiagnosisError(RuntimeError):
    """Raised when the bounded diagnosis cannot produce valid evidence."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CacheParityDiagnosisError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CacheParityDiagnosisError(f"refusing to overwrite {path}")
    encoded = json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(path)


def _read_tensor(path: Path, tf: Any, dtype: Any) -> Any:
    return tf.io.parse_tensor(path.read_bytes(), out_type=dtype)


def main() -> None:
    started = time.perf_counter()
    if tuple(sorted(os.sched_getaffinity(0))) != tuple(range(32, 48)):
        raise CacheParityDiagnosisError("parent CPU affinity mismatch")
    if _sha(FAILED_ROOT / "canary.json") != FAILED_CANARY_SHA256:
        raise CacheParityDiagnosisError("failed canary identity mismatch")

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if tf.config.list_physical_devices("GPU"):
        raise CacheParityDiagnosisError("CPU-only diagnosis found visible GPU")

    from bayesfilter.inference.tf_batch_value_score_pool import TFBatchValueScorePool

    canary = _load_module("physical_12x2_canary_cache_diagnosis", CANARY_RUNNER)
    support = canary._load_checkpoint_runner()
    geometry = json.loads((ROOT / canary.GEOMETRY).read_text(encoding="utf-8"))
    chart = support._chart(tf, geometry)
    latent = tf.ensure_shape(
        _read_tensor(FAILED_ROOT / "transition-state.tftensor", tf, tf.float64),
        (len(support.BETAS), support.CHAINS, 4),
    )
    cached_value = tf.ensure_shape(
        _read_tensor(
            FAILED_ROOT / "transition-base_target_log_prob.tftensor", tf, tf.float64
        ),
        (len(support.BETAS), support.CHAINS),
    )
    cached_score = tf.ensure_shape(
        _read_tensor(FAILED_ROOT / "transition-base_score.tftensor", tf, tf.float64),
        (len(support.BETAS), support.CHAINS, 4),
    )
    rows = tf.reshape(latent, (canary.ROWS, 4))
    theta = chart["center"] + tf.matmul(rows, chart["factor"], transpose_b=True)
    permutation = tf.roll(tf.range(canary.ROWS, dtype=tf.int32), shift=-1, axis=0)
    inverse_permutation = tf.argsort(permutation)

    def convert(value: Any, score: Any) -> tuple[Any, Any]:
        return (
            tf.convert_to_tensor(value, tf.float64) + chart["log_abs_determinant"],
            tf.matmul(tf.convert_to_tensor(score, tf.float64), chart["factor"]),
        )

    with TFBatchValueScorePool(canary._pool_config()) as pool:
        original_started = time.perf_counter()
        original_value, original_score, original_status, original_metadata = (
            pool.evaluate_with_status(theta, request_id="cache-diagnosis-original")
        )
        original_seconds = time.perf_counter() - original_started
        original_value, original_score = convert(original_value, original_score)

        permuted_started = time.perf_counter()
        permuted_value, permuted_score, permuted_status, permuted_metadata = (
            pool.evaluate_with_status(
                tf.gather(theta, permutation), request_id="cache-diagnosis-permuted"
            )
        )
        permuted_seconds = time.perf_counter() - permuted_started
        permuted_value, permuted_score = convert(permuted_value, permuted_score)
        permuted_value = tf.gather(permuted_value, inverse_permutation)
        permuted_score = tf.gather(permuted_score, inverse_permutation)
        permuted_status = {
            key: tf.gather(value, inverse_permutation)
            for key, value in permuted_status.items()
        }

    flat_cached_value = tf.reshape(cached_value, (canary.ROWS,))
    flat_cached_score = tf.reshape(cached_score, (canary.ROWS, 4))
    original_value_error = tf.abs(original_value - flat_cached_value)
    original_score_error = tf.reduce_max(
        tf.abs(original_score - flat_cached_score), axis=1
    )
    grouping_value_error = tf.abs(original_value - permuted_value)
    grouping_score_error = tf.reduce_max(tf.abs(original_score - permuted_score), axis=1)
    original_valid = tf.logical_and(
        tf.convert_to_tensor(original_status["status_code"], tf.int32) == 0,
        tf.convert_to_tensor(original_status["valid_pre_regularized_score"], tf.bool),
    )
    permuted_valid = tf.logical_and(
        tf.convert_to_tensor(permuted_status["status_code"], tf.int32) == 0,
        tf.convert_to_tensor(permuted_status["valid_pre_regularized_score"], tf.bool),
    )
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_12x2_cache_parity_diagnosis.v1",
        "status": "CACHE_PARITY_DIAGNOSIS_COMPLETE",
        "failed_canary_sha256": _sha(FAILED_ROOT / "canary.json"),
        "configuration": {
            "workers": canary.WORKERS,
            "rows_per_worker": canary.ROWS_PER_WORKER,
            "jit_compile": True,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        },
        "all_status_valid": bool(
            tf.reduce_all(tf.logical_and(original_valid, permuted_valid)).numpy()
        ),
        "original_vs_cached": {
            "value_max_abs_error": tf.reduce_max(original_value_error),
            "score_max_abs_error": tf.reduce_max(original_score_error),
            "value_row_abs_errors": original_value_error,
            "score_row_max_abs_errors": original_score_error,
        },
        "changed_pair_grouping": {
            "permutation": permutation,
            "value_max_abs_error": tf.reduce_max(grouping_value_error),
            "score_max_abs_error": tf.reduce_max(grouping_score_error),
            "value_row_abs_errors": grouping_value_error,
            "score_row_max_abs_errors": grouping_score_error,
        },
        "timing": {
            "original_seconds": original_seconds,
            "permuted_seconds": permuted_seconds,
            "wall_seconds": time.perf_counter() - started,
        },
        "worker_identity": {
            "original": support._worker_identity(original_metadata),
            "permuted": support._worker_identity(permuted_metadata),
        },
        "nonclaims": (
            "targeted cache-parity localization only",
            "no sampler, topology, travel, convergence, or posterior claim",
        ),
    }
    if not payload["all_status_valid"]:
        raise CacheParityDiagnosisError("diagnostic target status invalid")
    _write_json(FINAL, payload)
    print(json.dumps({"status": payload["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
