"""Framework-free subprocess supervision for independent GPU tuning tasks.

Workers own their fresh output directories. Only the parent writes supervision
records, handles deadlines, and reconciles complete results. This module grants
no compute allocation; a campaign caller must reserve and settle worker time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import signal
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from bayesfilter.runtime.display_gpu_policy import POLICY_ID as GPU_POLICY_ID


POLL_SECONDS = 0.05


class ParallelTuningError(RuntimeError):
    """Process-level tuning orchestration failed its engineering contract."""


@dataclass(frozen=True)
class ParallelTuningTask:
    """One independent worker invocation and its private evidence root."""

    task_id: str
    scope_index: int
    gpu_uuid: str
    output_dir: Path
    command: tuple[str, ...]

    @property
    def supervision_dir(self) -> Path:
        return self.output_dir.with_name(self.output_dir.name + "-supervision")

    def payload(self) -> Mapping[str, Any]:
        return {
            "task_id": self.task_id,
            "scope_index": self.scope_index,
            "gpu_uuid": self.gpu_uuid,
            "output_dir": str(self.output_dir.resolve()),
            "command": list(self.command),
        }


def validate_tasks(tasks: Sequence[ParallelTuningTask]) -> tuple[ParallelTuningTask, ...]:
    """Require disjoint fresh worker/control roots and one process per UUID."""

    normalized = tuple(replace(task, output_dir=task.output_dir.expanduser().resolve()) for task in tasks)
    if not normalized:
        raise ParallelTuningError("at least one tuning task is required")
    task_ids = [task.task_id for task in normalized]
    gpu_uuids = [task.gpu_uuid for task in normalized]
    if len(task_ids) != len(set(task_ids)):
        raise ParallelTuningError("tuning task IDs must be unique")
    if len(gpu_uuids) != len(set(gpu_uuids)):
        raise ParallelTuningError("one process per GPU is required; GPU UUID reused")
    roots = []
    for task in normalized:
        if not task.task_id.strip():
            raise ParallelTuningError("tuning task ID must be nonempty")
        if not task.gpu_uuid.startswith("GPU-") or "," in task.gpu_uuid:
            raise ParallelTuningError("tuning workers require one physical GPU UUID")
        if type(task.scope_index) is not int or task.scope_index < 0:
            raise ParallelTuningError("scope index must be a nonnegative integer")
        if not task.command or any(not isinstance(part, str) or not part for part in task.command):
            raise ParallelTuningError("tuning worker command must contain nonempty strings")
        for path in (task.output_dir, task.supervision_dir):
            root = path.expanduser().resolve()
            if root.exists():
                raise ParallelTuningError(f"refusing to reuse tuning worker root: {root}")
            if any(root == other or root in other.parents or other in root.parents for other in roots):
                raise ParallelTuningError("worker and supervision roots must not overlap")
            roots.append(root)
    return normalized


def worker_environment(
    base_environment: Mapping[str, str] | None, gpu_uuid: str
) -> dict[str, str]:
    """Establish pinning and on-demand allocation before child Python starts."""

    if not gpu_uuid.startswith("GPU-") or "," in gpu_uuid:
        raise ParallelTuningError("worker environment requires one GPU UUID")
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update({
        "CUDA_VISIBLE_DEVICES": gpu_uuid,
        "BAYESFILTER_SELECTED_GPU_UUID": gpu_uuid,
        "BAYESFILTER_GPU_SELECTION_POLICY_ID": GPU_POLICY_ID,
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "BAYESFILTER_PRELOAD_CUSTOM_OP": "0",
        "PYTHONUNBUFFERED": "1",
    })
    return environment


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _terminate_process(process: subprocess.Popen[Any], *, force: bool) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def run_parallel_tuning_wave(
    tasks: Sequence[ParallelTuningTask],
    *,
    timeout_seconds: float,
    terminate_grace_seconds: float = 30.0,
    base_environment: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    required_artifact: str = "run_manifest.json",
    artifact_validator: Callable[[ParallelTuningTask, Mapping[str, Any]], None] | None = None,
    monitor_callback: Callable[[], None] | None = None,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
) -> Mapping[str, Any]:
    """Supervise one wave, including cleanup within the declared wall bound.

    A campaign validator must check scientific receipts; absent one, only a
    JSON object with status='completed' passes this engineering-only helper.
    Early exits retain their individual lifetimes even if a sibling times out.
    """

    normalized = validate_tasks(tasks)
    timeout = float(timeout_seconds)
    grace = float(terminate_grace_seconds)
    if not math.isfinite(timeout) or timeout <= 0.0:
        raise ParallelTuningError("timeout_seconds must be positive and finite")
    if not math.isfinite(grace) or grace < 0.0 or grace >= timeout:
        raise ParallelTuningError("termination grace must be nonnegative and below timeout")
    artifact_relative = Path(required_artifact)
    if artifact_relative.is_absolute() or ".." in artifact_relative.parts:
        raise ParallelTuningError("required artifact must be relative to its worker root")
    started = time.monotonic()
    stop_at = started + timeout - grace
    processes: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    wave_error: str | None = None
    wave_status = "completed"

    def collect(task_id: str) -> bool:
        worker = processes[task_id]
        process = worker["process"]
        returncode = process.poll()
        if returncode is None:
            return False
        ended = time.monotonic()
        worker["stdout"].close()
        worker["stderr"].close()
        task = worker["task"]
        artifact = task.output_dir / artifact_relative
        status = worker.get("stop_reason") or ("completed" if returncode == 0 else "failed")
        artifact_error = None
        artifact_hash = None
        if status == "completed":
            try:
                content = artifact.read_bytes()
                payload = json.loads(content, parse_constant=_reject_constant)
                if not isinstance(payload, Mapping):
                    raise ValueError("worker result is not a JSON object")
                if artifact_validator is not None:
                    artifact_validator(task, payload)
                elif payload.get("status") != "completed":
                    raise ValueError("worker result status is not completed")
                artifact_hash = hashlib.sha256(content).hexdigest()
            except Exception as exc:
                status = "invalid_required_artifact"
                artifact_error = f"{type(exc).__name__}: {exc}"
        result = {
            "schema": "bayesfilter.parallel_tuning_worker_result.v1",
            "status": status,
            "task": task.payload(),
            "pid": process.pid,
            "returncode": returncode,
            "started_at_unix": worker["started_at_unix"],
            "elapsed_seconds": max(0.0, ended - worker["started"]),
            "required_artifact": str(artifact.resolve()),
            "required_artifact_sha256": artifact_hash,
            "artifact_error": artifact_error,
            "supervision_dir": str(task.supervision_dir.resolve()),
        }
        results[task_id] = result
        del processes[task_id]
        try:
            _write_json(task.supervision_dir / "parallel_worker_result.json", result)
        except OSError as exc:
            result["status"] = "supervision_write_failed"
            result["artifact_error"] = f"{type(exc).__name__}: {exc}"
        return True

    try:
        for task in normalized:
            if time.monotonic() >= stop_at:
                raise TimeoutError("wave deadline reached during launch")
            task.supervision_dir.mkdir(parents=True, exist_ok=False)
            stdout = (task.supervision_dir / "worker.stdout.log").open("x", encoding="utf-8")
            stderr = (task.supervision_dir / "worker.stderr.log").open("x", encoding="utf-8")
            environment = worker_environment(base_environment, task.gpu_uuid)
            environment["BAYESFILTER_WORKER_OUTPUT_DIR"] = str(task.output_dir.resolve())
            child_started = time.monotonic()
            child_unix = time.time()
            try:
                process = popen_factory(
                    list(task.command),
                    cwd=None if cwd is None else str(Path(cwd).resolve()),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=True,
                )
            except BaseException:
                stdout.close()
                stderr.close()
                raise
            processes[task.task_id] = {
                "task": task, "process": process, "stdout": stdout, "stderr": stderr,
                "started": child_started, "started_at_unix": child_unix,
            }
            _write_json(task.supervision_dir / "parallel_worker_start.json", {
                "schema": "bayesfilter.parallel_tuning_worker_start.v1",
                "status": "RUNNING", "task": task.payload(), "pid": process.pid,
                "started_at_unix": child_unix, "timeout_seconds": timeout,
                "terminate_grace_seconds": grace, "grace_included_in_timeout": True,
                "tf_force_gpu_allow_growth": environment["TF_FORCE_GPU_ALLOW_GROWTH"],
            })
        while processes:
            for task_id in tuple(processes):
                collect(task_id)
            if not processes:
                break
            if time.monotonic() >= stop_at:
                raise TimeoutError("wave deadline reached")
            if monitor_callback is not None:
                monitor_callback()
            time.sleep(min(POLL_SECONDS, max(0.0, stop_at - time.monotonic())))
    except BaseException as exc:
        wave_error = f"{type(exc).__name__}: {exc}"
        wave_status = "timed_out" if isinstance(exc, TimeoutError) else "failed"
    finally:
        if processes:
            for task_id in tuple(processes):
                collect(task_id)
            cancelled = tuple(worker["process"] for worker in processes.values())
            for worker in processes.values():
                worker["stop_reason"] = wave_status
                _terminate_process(worker["process"], force=False)
            grace_end = min(started + timeout, time.monotonic() + grace)
            while processes and time.monotonic() < grace_end:
                for task_id in tuple(processes):
                    collect(task_id)
                if processes:
                    time.sleep(min(POLL_SECONDS, max(0.0, grace_end - time.monotonic())))
            for process in cancelled:
                _terminate_process(process, force=True)
            for worker in tuple(processes.values()):
                worker["process"].wait()
                collect(worker["task"].task_id)
    for task in normalized:
        if task.task_id not in results:
            results[task.task_id] = {
                "status": "not_launched", "task": task.payload(),
                "elapsed_seconds": 0.0, "pid": None, "returncode": None,
            }
    if wave_status == "completed" and any(row["status"] != "completed" for row in results.values()):
        wave_status = "failed"
    return {
        "schema": "bayesfilter.parallel_tuning_wave_result.v1",
        "status": wave_status,
        "error": wave_error,
        "task_count": len(normalized),
        "completed_count": sum(row["status"] == "completed" for row in results.values()),
        "results": tuple(results[task.task_id] for task in normalized),
        "elapsed_seconds": time.monotonic() - started,
        "worker_seconds": sum(row["elapsed_seconds"] for row in results.values()),
        "timeout_seconds": timeout,
        "terminate_grace_seconds": grace,
        "grace_included_in_timeout": True,
    }


__all__ = [
    "ParallelTuningError", "ParallelTuningTask", "run_parallel_tuning_wave",
    "validate_tasks", "worker_environment",
]
