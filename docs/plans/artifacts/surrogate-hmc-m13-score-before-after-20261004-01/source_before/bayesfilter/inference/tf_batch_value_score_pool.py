"""Persistent CPU workers for batch-native TensorFlow value/score targets."""

from __future__ import annotations

import concurrent.futures
import importlib
import multiprocessing
import os
import resource
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping


_TARGET: Any | None = None
_READY_BARRIER: Any | None = None
_METADATA: Mapping[str, Any] | None = None
_STATUS_CALLS: Mapping[int, Any] = {}

# Preserve admission for the historical worker route while allowing the
# current status-bearing batch-native target to use the same CPU pool.
_ADMITTED_EVALUATION_POLICIES = frozenset(
    {
        "batch_native_tensorflow_no_row_mapping_v1",
        "batch_native_tensorflow_status_no_row_mapping_v2",
    }
)


def _worker_environment(cores: int) -> Mapping[str, str]:
    threads = str(int(cores))
    return {
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "OMP_NUM_THREADS": threads,
        "OPENBLAS_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "NUMEXPR_NUM_THREADS": threads,
        "TF_NUM_INTRAOP_THREADS": threads,
        "TF_NUM_INTEROP_THREADS": "1",
    }


def _worker_init(
    factory_path: str,
    factory_config: Mapping[str, Any],
    cores: int,
    batch_sizes: tuple[int, ...],
    barrier: Any,
    assigned_cpu: int | None,
) -> None:
    global _TARGET, _READY_BARRIER, _METADATA, _STATUS_CALLS

    expected = _worker_environment(cores)
    mismatched = {
        key: os.environ.get(key) for key, value in expected.items() if os.environ.get(key) != value
    }
    if mismatched:
        raise RuntimeError("batch worker environment mismatch: " + ", ".join(sorted(mismatched)))
    assigned_cpu = None if assigned_cpu is None else int(assigned_cpu)
    if assigned_cpu is not None:
        # Pin before importing TensorFlow so lazily created native threads
        # inherit the worker's single-core affinity.
        os.sched_setaffinity(0, {assigned_cpu})
    # Thread counts are supplied through the environment before TensorFlow is
    # imported; changing them after import can initialize the context too
    # early and raises in spawned workers.
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("batch-native CPU worker found a visible GPU")
    module_name, symbol_name = str(factory_path).split(":", 1)
    factory = getattr(importlib.import_module(module_name), symbol_name)
    target = factory(dict(factory_config))
    if getattr(target, "evaluation_policy", None) not in _ADMITTED_EVALUATION_POLICIES:
        raise RuntimeError("worker target is not an admitted batch-native route")
    if assigned_cpu is not None:
        _bind_process_threads_to_cpu(assigned_cpu)
    _TARGET = target
    status_method = getattr(target, "neutra_batch_log_prob_and_grad_status", None)
    status_jit_compile = bool(getattr(target, "_jit_compile", False))
    _STATUS_CALLS = (
        {
            int(size): tf.function(
                status_method,
                input_signature=(
                    tf.TensorSpec([int(size), int(target.parameter_dim)], tf.float64),
                ),
                jit_compile=status_jit_compile,
                reduce_retracing=False,
            )
            for size in batch_sizes
        }
        if callable(status_method)
        else {}
    )
    _READY_BARRIER = barrier
    _METADATA = {
        "pid": os.getpid(),
        "assigned_cpu": assigned_cpu,
        "thread_affinity": _thread_affinity_snapshot(),
        "worker_backend": "batch_native_value_score",
        "evaluation_policy": target.evaluation_policy,
        "compiled_batch_sizes": list(batch_sizes),
        "status_jit_compile": status_jit_compile,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_gpu_devices": [],
        "target_signature": target.target_signature(),
        "adapter_signature": target.adapter_signature(),
    }


def _worker_ready() -> Mapping[str, Any]:
    if _TARGET is None or _READY_BARRIER is None or _METADATA is None:
        raise RuntimeError("batch-native worker initialization is incomplete")
    try:
        _READY_BARRIER.wait()
    except threading.BrokenBarrierError as exc:
        raise RuntimeError("batch-native worker startup barrier failed") from exc
    return {
        "metadata": dict(_METADATA),
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }


def _thread_affinity_snapshot() -> list[Mapping[str, Any]]:
    rows = []
    task_root = f"/proc/{os.getpid()}/task"
    for name in os.listdir(task_root):
        if not str(name).isdigit():
            continue
        task_id = int(name)
        try:
            affinity = sorted(os.sched_getaffinity(task_id))
        except ProcessLookupError:
            continue
        rows.append({"tid": task_id, "affinity": affinity})
    return rows


def _worker_evaluate(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if _TARGET is None or _METADATA is None:
        raise RuntimeError("batch-native worker is unavailable")
    import tensorflow as tf

    rows = tf.io.parse_tensor(payload["rows"], out_type=tf.float64)
    rows = tf.ensure_shape(rows, [int(payload["row_count"]), int(_TARGET.parameter_dim)])
    started = time.perf_counter()
    preserve_status = bool(payload.get("preserve_status", False))
    if preserve_status:
        status_call = _STATUS_CALLS.get(int(payload["row_count"]))
        if not callable(status_call):
            raise RuntimeError(
                "status-preserving evaluation requires a compiled target row-status call"
            )
        value, score, raw_status = status_call(rows)
        status = {
            str(key): tf.convert_to_tensor(tensor)
            for key, tensor in raw_status.items()
        }
        for key, tensor in status.items():
            if tensor.shape.rank != 1 or tensor.shape[0] != int(payload["row_count"]):
                raise RuntimeError(
                    f"target status field {key!r} must have shape [row_count]"
                )
    else:
        value, score = _TARGET.batch_value_and_score(rows)
        status = {}
    assigned_cpu = _METADATA.get("assigned_cpu")
    if assigned_cpu is not None:
        _bind_process_threads_to_cpu(int(assigned_cpu))
    if not preserve_status:
        tf.debugging.assert_all_finite(value, "batch-native worker value")
        tf.debugging.assert_all_finite(score, "batch-native worker score")
    return {
        "worker_index": int(payload["worker_index"]),
        "item_start": int(payload["item_start"]),
        "item_stop": int(payload["item_stop"]),
        "request_id": str(payload["request_id"]),
        "worker_pid": os.getpid(),
        "assigned_cpu": assigned_cpu,
        "value": bytes(tf.io.serialize_tensor(value).numpy()),
        "score": bytes(tf.io.serialize_tensor(score).numpy()),
        "status": {
            key: {
                "dtype": tensor.dtype.name,
                "tensor": bytes(tf.io.serialize_tensor(tensor).numpy()),
            }
            for key, tensor in status.items()
        },
        "runtime_seconds": time.perf_counter() - started,
        "ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "metadata": dict(_METADATA),
    }


def _bind_process_threads_to_cpu(cpu_id: int) -> None:
    """Bind all current worker threads to one logical CPU.

    TensorFlow creates housekeeping threads even with one configured compute
    thread. Binding every existing task keeps those native threads on the
    worker's assigned core instead of turning them into additional compute
    capacity.
    """

    cpu = int(cpu_id)
    if cpu < 0:
        raise ValueError("assigned CPU must be nonnegative")
    task_root = f"/proc/{os.getpid()}/task"
    for _attempt in range(2):
        task_ids = tuple(
            int(name)
            for name in os.listdir(task_root)
            if str(name).isdigit()
        )
        for task_id in task_ids:
            try:
                os.sched_setaffinity(task_id, {cpu})
            except ProcessLookupError:
                continue
    disallowed = []
    for name in os.listdir(task_root):
        if not str(name).isdigit():
            continue
        task_id = int(name)
        try:
            affinity = os.sched_getaffinity(task_id)
        except ProcessLookupError:
            continue
        if affinity != {cpu}:
            disallowed.append({"tid": task_id, "affinity": sorted(affinity)})
    if disallowed:
        raise RuntimeError(
            f"worker thread affinity did not bind to CPU {cpu}: {disallowed}"
        )


@dataclass(frozen=True)
class TFBatchValueScorePoolConfig:
    factory_path: str
    factory_config: Mapping[str, Any]
    dimension: int
    worker_count: int = 8
    cores_per_worker: int = 1
    batch_sizes: tuple[int, ...] = (1, 2, 8, 12, 13, 32)
    batch_per_worker: int | None = None
    worker_cpu_ids: tuple[int, ...] = ()
    timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if ":" not in self.factory_path:
            raise ValueError("factory_path must be module:callable")
        if int(self.dimension) <= 0 or int(self.worker_count) <= 0:
            raise ValueError("dimension and worker_count must be positive")
        if int(self.cores_per_worker) <= 0 or float(self.timeout_seconds) <= 0.0:
            raise ValueError("cores_per_worker and timeout_seconds must be positive")
        if any(int(size) <= 0 for size in self.batch_sizes):
            raise ValueError("batch_sizes must be positive")
        if self.batch_per_worker is not None and int(self.batch_per_worker) <= 0:
            raise ValueError("batch_per_worker must be positive when configured")
        cpu_ids = tuple(int(value) for value in self.worker_cpu_ids)
        if cpu_ids:
            if len(cpu_ids) != int(self.worker_count):
                raise ValueError("worker_cpu_ids must match worker_count")
            if len(set(cpu_ids)) != len(cpu_ids) or any(value < 0 for value in cpu_ids):
                raise ValueError("worker_cpu_ids must be unique and nonnegative")
            unavailable = sorted(set(cpu_ids) - set(os.sched_getaffinity(0)))
            if unavailable:
                raise ValueError(f"worker_cpu_ids are outside the current CPU affinity: {unavailable}")


class TFBatchValueScorePool:
    """Process-sharded rank-2 TensorFlow target evaluator."""

    def __init__(self, config: TFBatchValueScorePoolConfig) -> None:
        self.config = config
        self._executor: concurrent.futures.ProcessPoolExecutor | None = None
        self._pinned_executors: tuple[concurrent.futures.ProcessPoolExecutor, ...] = ()
        self._startup: Mapping[str, Any] | None = None

    def __enter__(self) -> "TFBatchValueScorePool":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        for executor in self._pinned_executors:
            executor.shutdown(wait=True, cancel_futures=True)
        self._pinned_executors = ()

    def abort(self) -> None:
        executors = (
            self._pinned_executors
            if self._pinned_executors
            else (() if self._executor is None else (self._executor,))
        )
        self._executor = None
        self._pinned_executors = ()
        if not executors:
            return
        for executor in executors:
            processes = getattr(executor, "_processes", {})
            for process in tuple(processes.values()):
                if process.is_alive():
                    process.terminate()
            executor.shutdown(wait=True, cancel_futures=True)

    def evaluate(
        self, rows: Any, *, request_id: str
    ) -> tuple[Any, Any, Mapping[str, Any]]:
        values, scores, _status, metadata = self._evaluate(
            rows, request_id=request_id, preserve_status=False
        )
        return values, scores, metadata

    def evaluate_with_status(
        self, rows: Any, *, request_id: str
    ) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
        """Return raw target outputs and per-row status without finite asserts.

        A completed invalid target evaluation is data, not a worker failure. The
        caller must reject the full batch before any optimizer update.
        """

        values, scores, status, metadata = self._evaluate(
            rows, request_id=request_id, preserve_status=True
        )
        return values, scores, status, metadata

    def _evaluate(
        self, rows: Any, *, request_id: str, preserve_status: bool
    ) -> tuple[Any, Any, Mapping[str, Any], Mapping[str, Any]]:
        import tensorflow as tf

        matrix = tf.convert_to_tensor(rows, tf.float64)
        if matrix.shape.rank != 2 or matrix.shape[1] != int(self.config.dimension):
            raise ValueError("rows must have static shape [batch, dimension]")
        row_count = matrix.shape[0]
        if row_count is None:
            raise ValueError("rows must have a static leading batch dimension")
        if int(row_count) <= 0:
            raise ValueError("batch must be nonempty")
        tf.debugging.assert_all_finite(matrix, "batch-native pool input")
        request = str(request_id)
        if not request:
            raise ValueError("request_id must be nonempty")
        self._ensure_started()
        if self._executor is None and not self._pinned_executors:
            raise RuntimeError("batch-native worker pool failed to start")
        if self.config.batch_per_worker is None:
            if int(row_count) < int(self.config.worker_count):
                raise ValueError("batch must contain at least one row per configured worker")
            bounds = [
                (
                    int(row_count) * task_index // int(self.config.worker_count),
                    int(row_count) * (task_index + 1) // int(self.config.worker_count),
                )
                for task_index in range(int(self.config.worker_count))
            ]
        else:
            shard = int(self.config.batch_per_worker)
            bounds = [
                (start, min(start + shard, int(row_count)))
                for start in range(0, int(row_count), shard)
            ]
        futures = []
        for task_index, (start, stop) in enumerate(bounds):
            shard_size = stop - start
            if shard_size not in set(int(size) for size in self.config.batch_sizes):
                raise ValueError(f"undeclared batch-native shard size {shard_size}")
            serialized = bytes(tf.io.serialize_tensor(matrix[start:stop]).numpy())
            executor = (
                self._pinned_executors[task_index % len(self._pinned_executors)]
                if self._pinned_executors
                else self._executor
            )
            assert executor is not None
            futures.append(
                executor.submit(
                    _worker_evaluate,
                    {
                        "worker_index": task_index,
                        "item_start": start,
                        "item_stop": stop,
                        "row_count": shard_size,
                        "request_id": request,
                        "rows": serialized,
                        "preserve_status": bool(preserve_status),
                    },
                )
            )
        try:
            results = [future.result(timeout=self.config.timeout_seconds) for future in futures]
        except BaseException:
            self.abort()
            raise
        results.sort(key=lambda row: int(row["item_start"]))
        values = tf.concat(
            [tf.io.parse_tensor(row["value"], out_type=tf.float64) for row in results], axis=0
        )
        scores = tf.concat(
            [tf.io.parse_tensor(row["score"], out_type=tf.float64) for row in results], axis=0
        )
        values = tf.ensure_shape(values, [int(row_count)])
        scores = tf.ensure_shape(scores, [int(row_count), int(self.config.dimension)])
        status: dict[str, Any] = {}
        if preserve_status:
            status_fields = tuple(sorted(results[0]["status"]))
            if not status_fields or any(
                tuple(sorted(row["status"])) != status_fields for row in results
            ):
                raise RuntimeError("worker target status fields are missing or inconsistent")
            for key in status_fields:
                dtype_name = str(results[0]["status"][key]["dtype"])
                if any(
                    str(row["status"][key]["dtype"]) != dtype_name for row in results
                ):
                    raise RuntimeError(f"worker target status dtype changed for {key}")
                dtype = tf.dtypes.as_dtype(dtype_name)
                status[key] = tf.ensure_shape(
                    tf.concat(
                        [
                            tf.io.parse_tensor(
                                row["status"][key]["tensor"], out_type=dtype
                            )
                            for row in results
                        ],
                        axis=0,
                    ),
                    [int(row_count)],
                )
        else:
            tf.debugging.assert_all_finite(values, "batch-native pooled values")
            tf.debugging.assert_all_finite(scores, "batch-native pooled scores")
        metadata = self._metadata(results, request=request)
        return values, scores, status, metadata

    def _ensure_started(self) -> None:
        if self._executor is not None or self._pinned_executors:
            return
        environment = _worker_environment(self.config.cores_per_worker)
        previous = {key: os.environ.get(key) for key in environment}
        os.environ.update(environment)
        try:
            context = multiprocessing.get_context("spawn")
            barrier = context.Barrier(
                int(self.config.worker_count), timeout=float(self.config.timeout_seconds)
            )
            if self.config.worker_cpu_ids:
                # One single-process executor per CPU makes shard-to-core
                # assignment deterministic instead of relying on pool scheduling.
                self._pinned_executors = tuple(
                    concurrent.futures.ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=context,
                        initializer=_worker_init,
                        initargs=(
                            self.config.factory_path,
                            dict(self.config.factory_config),
                            int(self.config.cores_per_worker),
                            tuple(int(size) for size in self.config.batch_sizes),
                            barrier,
                            int(cpu_id),
                        ),
                    )
                    for cpu_id in self.config.worker_cpu_ids
                )
                readiness = [executor.submit(_worker_ready) for executor in self._pinned_executors]
            else:
                self._executor = concurrent.futures.ProcessPoolExecutor(
                    max_workers=int(self.config.worker_count),
                    mp_context=context,
                    initializer=_worker_init,
                    initargs=(
                        self.config.factory_path,
                        dict(self.config.factory_config),
                        int(self.config.cores_per_worker),
                        tuple(int(size) for size in self.config.batch_sizes),
                        barrier,
                        None,
                    ),
                )
                readiness = [
                    self._executor.submit(_worker_ready)
                    for _ in range(int(self.config.worker_count))
                ]
            rows = [future.result(timeout=self.config.timeout_seconds) for future in readiness]
        except BaseException:
            self.abort()
            raise
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        metadata = sorted(
            (dict(row["metadata"]) for row in rows),
            key=lambda row: (
                row.get("assigned_cpu") is None,
                -1 if row.get("assigned_cpu") is None else int(row["assigned_cpu"]),
                int(row["pid"]),
            ),
        )
        pids = sorted({int(row["pid"]) for row in metadata})
        if len(pids) != int(self.config.worker_count):
            self.abort()
            raise RuntimeError("not every batch-native worker initialized")
        if any(row["worker_backend"] != "batch_native_value_score" for row in metadata):
            self.abort()
            raise RuntimeError("worker backend admission failed")
        if self.config.worker_cpu_ids:
            realized = {int(row["assigned_cpu"]) for row in metadata}
            if realized != set(int(value) for value in self.config.worker_cpu_ids):
                self.abort()
                raise RuntimeError("persistent worker CPU assignment is incomplete")
        self._startup = {
            "startup_worker_pids": pids,
            "startup_worker_metadata": metadata,
            "startup_worker_ru_maxrss_sum_bytes": sum(
                int(row["ru_maxrss_bytes"]) for row in rows
            ),
        }

    def _metadata(
        self, results: list[Mapping[str, Any]], *, request: str
    ) -> Mapping[str, Any]:
        assert self._startup is not None
        return {
            "backend": "persistent_cpu_batch_native_tensorflow_pool",
            "evaluation_mode": "batch_native",
            "worker_backend": "batch_native_value_score",
            "configured_worker_count": int(self.config.worker_count),
            "batch_per_worker": self.config.batch_per_worker,
            "compiled_batch_sizes": list(self.config.batch_sizes),
            "task_count": len(results),
            "worker_shard_sizes": [
                int(row["item_stop"]) - int(row["item_start"]) for row in results
            ],
            "request_id": request,
            "worker_runtime_seconds": [float(row["runtime_seconds"]) for row in results],
            "worker_runtime_max_seconds": max(float(row["runtime_seconds"]) for row in results),
            "worker_result_pids": [int(row["worker_pid"]) for row in results],
            "worker_assigned_cpu_ids": [
                row.get("assigned_cpu") for row in results
            ],
            "worker_tasks": [
                {
                    "worker_index": int(row["worker_index"]),
                    "item_start": int(row["item_start"]),
                    "item_stop": int(row["item_stop"]),
                    "worker_pid": int(row["worker_pid"]),
                    "assigned_cpu": row.get("assigned_cpu"),
                }
                for row in results
            ],
            "active_worker_ru_maxrss_sum_bytes": sum(
                int(row["ru_maxrss_bytes"]) for row in results
            ),
            **self._startup,
        }


__all__ = ["TFBatchValueScorePool", "TFBatchValueScorePoolConfig"]
