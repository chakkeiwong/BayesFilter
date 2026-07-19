"""Persistent CPU process pool for scalar value/score target evaluations.

The pool is the execution boundary used by the DSGE-HMC-style NeuTra training
route.  Spawned workers hide CUDA before importing TensorFlow, construct one
target, and evaluate scalar rows eagerly.  The parent process owns the
transport, optimizer, and custom-gradient bridge.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import importlib
import multiprocessing
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_WORKER_TARGET: Any | None = None
_WORKER_METADATA: Mapping[str, Any] | None = None
_CPU_WORKER_ENV = "BAYESFILTER_CPU_VALUE_SCORE_WORKER"


def _array_hash(rows: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(rows, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _worker_init(factory_path: str, factory_config: Mapping[str, Any], cores: int) -> None:
    """Initialize a worker with fail-closed CPU visibility."""

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("CPU value/score worker must inherit CUDA_VISIBLE_DEVICES=-1")
    for key, value in {
        "OMP_NUM_THREADS": str(max(1, int(cores))),
        "OPENBLAS_NUM_THREADS": str(max(1, int(cores))),
        "MKL_NUM_THREADS": str(max(1, int(cores))),
        "NUMEXPR_NUM_THREADS": str(max(1, int(cores))),
        "TF_NUM_INTRAOP_THREADS": str(max(1, int(cores))),
        "TF_NUM_INTEROP_THREADS": "1",
    }.items():
        os.environ[key] = value
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU value/score worker found a visible GPU")
    module_name, symbol_name = str(factory_path).split(":", 1)
    factory = getattr(importlib.import_module(module_name), symbol_name)
    target = factory(dict(factory_config))
    method = getattr(target, "eager_value_and_score", None)
    if not callable(method):
        method = target.value_and_score

    @tf.function(
        input_signature=(tf.TensorSpec([int(target.parameter_dim)], tf.float64),),
        jit_compile=False,
        reduce_retracing=True,
    )
    def scalar_value_score(row: Any) -> tuple[Any, Any]:
        return method(row)

    scalar_value_score(tf.zeros([int(target.parameter_dim)], tf.float64))
    global _WORKER_TARGET, _WORKER_METADATA
    _WORKER_TARGET = scalar_value_score
    _WORKER_METADATA = {
        "pid": int(os.getpid()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "jit_compile": False,
        "tensorflow_gpu_devices": [],
        "worker_backend": "scalar_eager_value_score",
    }


def _worker_eval(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _WORKER_TARGET is None or _WORKER_METADATA is None:
        raise RuntimeError("CPU value/score worker was not initialized")
    import tensorflow as tf

    rows = np.asarray(payload["rows"], dtype=np.float64)
    started = time.perf_counter()
    values: list[float] = []
    scores: list[np.ndarray] = []
    for row in rows:
        value, score = _WORKER_TARGET(tf.convert_to_tensor(row, tf.float64))
        value_np = float(tf.convert_to_tensor(value, tf.float64).numpy())
        score_np = np.asarray(tf.convert_to_tensor(score, tf.float64).numpy(), dtype=np.float64)
        values.append(value_np)
        scores.append(score_np)
    values_np = np.asarray(values, dtype=np.float64)
    scores_np = np.asarray(scores, dtype=np.float64)
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "input_hash": str(payload["input_hash"]),
        "values": values_np,
        "scores": scores_np,
        "runtime_seconds": float(time.perf_counter() - started),
        "worker_metadata": dict(_WORKER_METADATA),
    }


@dataclass(frozen=True)
class CPUValueScorePoolConfig:
    worker_factory_path: str
    worker_config: Mapping[str, Any]
    dimension: int
    worker_count: int = 4
    cores_per_worker: int = 1
    timeout_seconds: float = 600.0

    def __post_init__(self) -> None:
        if ":" not in str(self.worker_factory_path):
            raise ValueError("worker_factory_path must be module:callable")
        if int(self.dimension) <= 0 or int(self.worker_count) <= 0:
            raise ValueError("dimension and worker_count must be positive")
        if int(self.cores_per_worker) <= 0 or float(self.timeout_seconds) <= 0.0:
            raise ValueError("cores_per_worker and timeout_seconds must be positive")


class CPUValueScorePool:
    """Persistent spawn pool returning ordered scalar target values and scores."""

    def __init__(self, config: CPUValueScorePoolConfig) -> None:
        self.config = config
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._opened = False

    def __enter__(self) -> "CPUValueScorePool":
        if self._opened:
            raise RuntimeError("CPU value/score pool is already open")
        self._opened = True
        return self

    def evaluate(self, rows: Any, *, request_id: str) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
        if not self._opened:
            raise RuntimeError("CPU value/score pool must be opened as a context manager")
        matrix = np.asarray(rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != int(self.config.dimension):
            raise ValueError("rows must have shape [batch, dimension]")
        if matrix.shape[0] == 0 or not np.all(np.isfinite(matrix)):
            raise ValueError("rows must be nonempty and finite")
        input_hash = _array_hash(matrix)
        worker_count = min(int(self.config.worker_count), int(matrix.shape[0]))
        # Keep the fail-closed CPU environment in place until the executor has
        # submitted its first tasks.  Spawn inherits the environment at that
        # boundary; restoring it earlier is a GPU-visibility race.
        previous = os.environ.get("CUDA_VISIBLE_DEVICES")
        previous_marker = os.environ.get(_CPU_WORKER_ENV)
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        os.environ[_CPU_WORKER_ENV] = "1"
        try:
            if self._executor is None:
                self._executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(self.config.worker_count),
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=_worker_init,
                    initargs=(
                        str(self.config.worker_factory_path),
                        dict(self.config.worker_config),
                        int(self.config.cores_per_worker),
                    ),
                )
            futures = []
            for worker_index in range(worker_count):
                start = (matrix.shape[0] * worker_index) // worker_count
                stop = (matrix.shape[0] * (worker_index + 1)) // worker_count
                futures.append(self._executor.submit(_worker_eval, {
                    "worker_index": worker_index,
                    "item_start": start,
                    "item_stop": stop,
                    "input_hash": input_hash,
                    "rows": matrix[start:stop],
                }))
        finally:
            if previous is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous
            if previous_marker is None:
                os.environ.pop(_CPU_WORKER_ENV, None)
            else:
                os.environ[_CPU_WORKER_ENV] = previous_marker
        results = [future.result(timeout=float(self.config.timeout_seconds)) for future in futures]
        results.sort(key=lambda result: int(result["item_start"]))
        if any(result["input_hash"] != input_hash for result in results):
            raise RuntimeError("worker input identity mismatch")
        values = np.concatenate([np.asarray(result["values"], dtype=np.float64) for result in results])
        scores = np.concatenate([np.asarray(result["scores"], dtype=np.float64) for result in results], axis=0)
        if values.shape != (matrix.shape[0],) or scores.shape != matrix.shape:
            raise RuntimeError("worker value/score shape mismatch")
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(scores)):
            raise FloatingPointError("worker returned nonfinite values or scores")
        return values, scores, {
            "request_id": str(request_id),
            "input_hash": input_hash,
            "worker_count": worker_count,
            "worker_pids": sorted({int(result["worker_metadata"]["pid"]) for result in results}),
            "worker_metadata": [dict(result["worker_metadata"]) for result in results],
            "runtime_seconds": float(sum(float(result["runtime_seconds"]) for result in results)),
            "backend": "persistent_cpu_worker_value_score_custom_gradient_bridge",
            "jit_compile": False,
        }

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._opened = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = ["CPUValueScorePool", "CPUValueScorePoolConfig"]
