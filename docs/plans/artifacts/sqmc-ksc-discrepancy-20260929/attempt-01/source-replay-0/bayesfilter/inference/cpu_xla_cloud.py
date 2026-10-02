"""Explicit persistent CPU/XLA process pool for independent score-cloud rows.

This module does not provide a GPU fallback. A caller deliberately selects the
CPU route, supplies an importable value/score factory, and owns the context
manager lifetime. Each spawned child hides CUDA before importing TensorFlow,
compiles one static ``B=1`` function, and reuses it for all assigned rows.
"""

from __future__ import annotations

import concurrent.futures
import importlib
import multiprocessing
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np

from bayesfilter.cpu_xla_worker_bootstrap import (
    evaluate_cpu_xla_row,
    initialize_cpu_xla_worker,
)


_SPAWN_ENV_LOCK = threading.Lock()


@dataclass(frozen=True)
class CPUXLACloudConfig:
    """Process-pool policy for an explicitly selected CPU/XLA cloud route."""

    worker_factory_path: str
    dimension: int
    worker_count: int | None = None
    set_affinity: bool = True
    factory_config: Mapping[str, Any] | None = None
    heartbeat_seconds: float | None = 30.0

    def __post_init__(self) -> None:
        path = str(self.worker_factory_path)
        if ":" not in path or path.startswith(":") or path.endswith(":"):
            raise ValueError("worker_factory_path must be 'module:callable'")
        object.__setattr__(self, "worker_factory_path", path)
        dimension = int(self.dimension)
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        object.__setattr__(self, "dimension", dimension)
        if self.worker_count is not None:
            workers = int(self.worker_count)
            if workers <= 0:
                raise ValueError("worker_count must be positive")
            object.__setattr__(self, "worker_count", workers)
        object.__setattr__(self, "set_affinity", bool(self.set_affinity))
        object.__setattr__(
            self,
            "factory_config",
            {} if self.factory_config is None else dict(self.factory_config),
        )
        if self.heartbeat_seconds is not None:
            heartbeat = float(self.heartbeat_seconds)
            if not np.isfinite(heartbeat) or heartbeat <= 0.0:
                raise ValueError("heartbeat_seconds must be positive finite")
            object.__setattr__(self, "heartbeat_seconds", heartbeat)


@dataclass(frozen=True)
class CPUXLACloudResult:
    values: np.ndarray
    scores: np.ndarray
    worker_pids: tuple[int, ...]
    worker_count: int
    core_count: int
    core_count_source: str
    worker_bootstrap_records: tuple[Mapping[str, Any], ...]
    batch_size: int = 1
    jit_compile: bool = True
    device: str = "CPU"
    automatic_fallback_used: bool = False

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float).copy()
        scores = np.asarray(self.scores, dtype=float).copy()
        if values.ndim != 1 or scores.ndim != 2 or scores.shape[0] != values.shape[0]:
            raise ValueError("values/scores must have [rows] and [rows, dimension] shapes")
        values.setflags(write=False)
        scores.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "worker_pids", tuple(int(pid) for pid in self.worker_pids))
        object.__setattr__(
            self,
            "worker_bootstrap_records",
            tuple(dict(record) for record in self.worker_bootstrap_records),
        )

    def payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.cpu_xla_cloud.v1",
            "row_count": int(self.values.shape[0]),
            "dimension": int(self.scores.shape[1]),
            "worker_pids": list(self.worker_pids),
            "worker_count": self.worker_count,
            "core_count": self.core_count,
            "core_count_source": self.core_count_source,
            "worker_bootstrap_records": [
                dict(record) for record in self.worker_bootstrap_records
            ],
            "batch_size": self.batch_size,
            "jit_compile": self.jit_compile,
            "device": self.device,
            "cuda_visible_devices": "-1",
            "automatic_fallback_used": self.automatic_fallback_used,
            "nonclaims": [
                "explicit CPU/XLA cloud evaluation only",
                "no GPU fallback",
                "no CPU/GPU performance claim",
                "no HMC or posterior claim",
            ],
        }


class CPUXLACloudEvaluator:
    """Persistent spawn pool; use as a context manager and call repeatedly."""

    def __init__(self, config: CPUXLACloudConfig) -> None:
        self.config = config
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._open = False
        self._core_count, self._core_count_source = detected_cpu_core_count()
        self._worker_count = default_cpu_worker_count(
            task_count=None,
            worker_override=config.worker_count,
            detected_cores=self._core_count,
        )

    @property
    def worker_count(self) -> int:
        return self._worker_count

    def __enter__(self) -> "CPUXLACloudEvaluator":
        if self._open:
            raise RuntimeError("CPU/XLA evaluator is already open")
        self._open = True
        return self

    def evaluate(
        self,
        points: Any,
        *,
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> CPUXLACloudResult:
        if not self._open:
            raise RuntimeError("CPU/XLA evaluator must be opened with a context manager")
        rows = np.asarray(points, dtype=float)
        if (
            rows.ndim != 2
            or rows.shape[1] != self.config.dimension
            or rows.shape[0] == 0
            or not np.all(np.isfinite(rows))
        ):
            raise ValueError("points must be a nonempty finite [rows, dimension] matrix")
        tasks = [(index, rows[index].tolist()) for index in range(rows.shape[0])]
        if self._executor is None:
            self._worker_count = default_cpu_worker_count(
                task_count=int(rows.shape[0]),
                worker_override=self.config.worker_count,
                detected_cores=self._core_count,
            )
            pending = self._launch_pool_and_submit(tasks)
        else:
            pending = {
                self._executor.submit(evaluate_cpu_xla_row, task): task[0]
                for task in tasks
            }
        _emit_progress(
            progress_callback,
            "cpu_xla_cloud_evaluation_started",
            row_count=int(rows.shape[0]),
            worker_count=self._worker_count,
            completed_rows=0,
            semantic_progress=True,
        )
        started_at = time.monotonic()
        outputs = []
        completed = 0
        heartbeat_sequence = 0
        timeout = self.config.heartbeat_seconds
        while pending:
            done, _not_done = concurrent.futures.wait(
                pending,
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                heartbeat_sequence += 1
                _emit_progress(
                    progress_callback,
                    "cpu_xla_cloud_liveness_heartbeat",
                    row_count=int(rows.shape[0]),
                    worker_count=self._worker_count,
                    completed_rows=completed,
                    heartbeat_sequence=heartbeat_sequence,
                    elapsed_seconds=time.monotonic() - started_at,
                    semantic_progress=False,
                    heartbeat_is_not_progress=True,
                )
                continue
            for future in done:
                outputs.append(future.result())
                pending.pop(future)
                completed += 1
                _emit_progress(
                    progress_callback,
                    "cpu_xla_cloud_row_completed",
                    row_count=int(rows.shape[0]),
                    worker_count=self._worker_count,
                    completed_rows=completed,
                    semantic_progress=True,
                )
        outputs.sort(key=lambda item: item[0])
        _emit_progress(
            progress_callback,
            "cpu_xla_cloud_evaluation_completed",
            row_count=int(rows.shape[0]),
            worker_count=self._worker_count,
            completed_rows=int(rows.shape[0]),
            semantic_progress=True,
        )
        values = np.asarray([item[1] for item in outputs], dtype=float)
        scores = np.asarray([item[2] for item in outputs], dtype=float)
        pids = tuple(sorted({int(item[3]) for item in outputs}))
        bootstrap_by_pid = {
            int(item[3]): {"worker_pid": int(item[3]), **dict(item[4])}
            for item in outputs
        }
        return CPUXLACloudResult(
            values=values,
            scores=scores,
            worker_pids=pids,
            worker_count=self._worker_count,
            core_count=self._core_count,
            core_count_source=self._core_count_source,
            worker_bootstrap_records=tuple(
                bootstrap_by_pid[pid] for pid in sorted(bootstrap_by_pid)
            ),
        )

    def _launch_pool_and_submit(
        self, tasks: list[tuple[int, list[float]]]
    ) -> dict[concurrent.futures.Future, int]:
        """Start every child with CPU-only env inherited before spawn re-imports."""

        with _SPAWN_ENV_LOCK:
            previous_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
            previous_growth = os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH")
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            try:
                self._executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=self._worker_count,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_cpu_xla_worker,
                    initargs=(
                        self.config.worker_factory_path,
                        self.config.dimension,
                        dict(self.config.factory_config),
                        self.config.set_affinity,
                    ),
                )
                # Task count is at least worker count, so all persistent children
                # are launched while the inherited environment is fail-closed.
                return {
                    self._executor.submit(evaluate_cpu_xla_row, task): task[0]
                    for task in tasks
                }
            finally:
                if previous_cuda is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda
                if previous_growth is None:
                    os.environ.pop("TF_FORCE_GPU_ALLOW_GROWTH", None)
                else:
                    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = previous_growth

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._open = False

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def detected_cpu_core_count() -> tuple[int, str]:
    """Return physical cores when psutil is available, else logical fallback."""

    try:
        psutil = importlib.import_module("psutil")
        physical = psutil.cpu_count(logical=False)
    except (ImportError, AttributeError):
        physical = None
    if physical is not None and int(physical) > 0:
        return int(physical), "psutil_physical"
    logical = os.cpu_count()
    if logical is None or int(logical) <= 0:
        return 1, "minimum_fallback"
    return int(logical), "os_logical_fallback"


def default_cpu_worker_count(
    *,
    task_count: int | None,
    worker_override: int | None = None,
    detected_cores: int | None = None,
) -> int:
    """Use one third of detected cores, clamped to tasks; allow explicit override."""

    cores = detected_cpu_core_count()[0] if detected_cores is None else int(detected_cores)
    if cores <= 0:
        raise ValueError("detected_cores must be positive")
    workers = max(1, cores // 3) if worker_override is None else int(worker_override)
    if workers <= 0:
        raise ValueError("worker_override must be positive")
    if task_count is not None:
        tasks = int(task_count)
        if tasks <= 0:
            raise ValueError("task_count must be positive")
        workers = min(workers, tasks)
    return workers


def _emit_progress(
    callback: Callable[[Mapping[str, Any]], None] | None,
    stage: str,
    **payload: Any,
) -> None:
    if callback is not None:
        callback({"stage": str(stage), **payload})
