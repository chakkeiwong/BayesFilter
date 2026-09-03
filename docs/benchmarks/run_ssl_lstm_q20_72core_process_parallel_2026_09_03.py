#!/usr/bin/env python3
"""Run the staged q=20 72-core process-parallel mechanics campaign.

The controller starts isolated workers in three sequential barriers.  Workers
run the existing TensorFlow/TFP fixed-transport verification program; they do
not average maps, replace the target, or introduce a vectorized/pfor path.
CPU execution is deliberately an engineering diagnostic exception to the
repository GPU default.  The script is therefore not a posterior sampler
entry point and emits explicit nonclaims in every summary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Direct execution from ``docs/benchmarks`` does not put the repository root
# on ``sys.path``. Establish it before importing repository-owned modules.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.process_topology import (
    BarrierTopology,
    ProcessTopologyError,
    Q20ProcessTopology,
)


SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-72core-process-parallel-plan-2026-09-03.md"
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BACKEND = "tensorflow_eigh_strict"
DIMENSION = 4
CHAIN_COUNT = 4
BETAS = (0.0, 0.5, 1.0)
COMPONENTS = ("phase9a-chart-0", "phase9a-chart-1")
STEP_GRID = (0.25, 0.55, 0.85, 1.20)
LEAPFROG_GRID = (3, 8)
INITIAL_STATE_BANK = (
    (0.0, 0.0, 0.0, 0.0),
    (0.25, 0.0, 0.0, 0.0),
    (-0.25, 0.0, 0.0, 0.0),
    (0.0, 0.25, 0.0, 0.0),
)
WORKER_PYTHON = Path(
    os.environ.get("Q20_PROCESS_PYTHON", "/home/ubuntu/anaconda3/envs/tfgpu/bin/python")
)
TOPOLOGY = Q20ProcessTopology()
CANARY_CAP_SECONDS = 1200.0
FULL_CAP_SECONDS = 14400.0
TOTAL_CAP_SECONDS = 15600.0
CANARY_COUNTS = {
    "screen_num_results": 1,
    "screen_num_burnin_steps": 1,
    "selection_num_results": 4,
    "selection_num_burnin_steps": 1,
    "verification_num_results": 4,
    "verification_num_burnin_steps": 1,
}
FULL_COUNTS = {
    "screen_num_results": 4,
    "screen_num_burnin_steps": 2,
    "selection_num_results": 16,
    "selection_num_burnin_steps": 4,
    "verification_num_results": 16,
    "verification_num_burnin_steps": 4,
}
FIXTURE_CHECKPOINTS = (
    ROOT
    / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/"
    "phase9a-full-replay/attempt-02/chart-0/beta-0/chart_checkpoint.json",
    ROOT
    / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/"
    "phase9a-full-replay/attempt-02/chart-0/beta-0.5/chart_checkpoint.json",
    ROOT
    / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/"
    "phase9a-full-replay/attempt-02/chart-0/beta-1/chart_checkpoint.json",
    ROOT
    / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/"
    "phase9a-full-replay/attempt-02/chart-1/beta-0/chart_checkpoint.json",
    ROOT
    / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/"
    "phase9a-full-replay/attempt-02/chart-1/beta-0.5/chart_checkpoint.json",
    ROOT
    / "docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-09-02/"
    "phase9a-full-replay/attempt-02/chart-1/beta-1/chart_checkpoint.json",
)


class ParallelCampaignError(RuntimeError):
    """Raised when the process campaign cannot satisfy its contract."""


class ParallelCampaignDeadline(ParallelCampaignError):
    """A declared wall-cap stop with a durable, partial barrier receipt."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        super().__init__(
            f"{payload.get('phase', 'unknown')} barrier stopped at the declared campaign deadline"
        )


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ParallelCampaignError(f"refusing to overwrite artifact: {path}")
    path.write_bytes(_canonical(payload))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ParallelCampaignError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    """Convert TensorFlow values without importing TensorFlow in the controller."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        return _json_ready(numpy_method())
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _json_ready(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_ready(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        # Non-finite diagnostics are expected failure data for some candidate
        # probes. Preserve them as explicit JSON values instead of allowing
        # one invalid diagnostic to destroy the worker's durable failure row.
        return {"__nonfinite__": str(value).lower()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def _git_payload() -> Mapping[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(args, cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    status = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty": bool(status),
    }


def _available_cpu_ids() -> tuple[int, ...]:
    try:
        return tuple(sorted(int(value) for value in os.sched_getaffinity(0)))
    except AttributeError as exc:
        raise ParallelCampaignError("Linux CPU affinity is required") from exc


def _scope_specs(checkpoint_paths: Sequence[Path]) -> tuple[Mapping[str, Any], ...]:
    if len(checkpoint_paths) != len(BETAS) * len(COMPONENTS):
        raise ParallelCampaignError("six chart/beta checkpoint paths are required")
    rows = []
    index = 0
    for chart_index, component_id in enumerate(COMPONENTS):
        for beta in BETAS:
            path = Path(checkpoint_paths[index]).resolve()
            if not path.is_file():
                raise ParallelCampaignError(f"checkpoint is missing: {path}")
            rows.append(
                {
                    "scope_index": index,
                    "scope_id": f"chart-{chart_index}-beta-{beta:g}",
                    "chart_index": chart_index,
                    "component_id": component_id,
                    "beta": beta,
                    "checkpoint_path": str(path),
                }
            )
            index += 1
    return tuple(rows)


def _seed_bases(scope_index: int) -> Mapping[str, tuple[int, int]]:
    # Distinct hundred-thousand bands make accidental cross-role collisions
    # obvious in manifests; _run_verification folds candidate/replication IDs
    # into each base with TensorFlow stateless fold-in.
    index = int(scope_index)
    return {
        "tune": (20260903, 100000 + index * 1000),
        "screen": (20260903, 200000 + index * 1000),
        "selection": (20260903, 300000 + index * 1000),
        "verification": (20260903, 400000 + index * 1000),
    }


def _candidate_rows() -> tuple[Mapping[str, Any], ...]:
    rows = []
    index = 0
    for step in STEP_GRID:
        for leapfrog in LEAPFROG_GRID:
            rows.append(
                {
                    "candidate_index": index,
                    "step_size": step,
                    "num_leapfrog_steps": leapfrog,
                }
            )
            index += 1
    return tuple(rows)


def _base_task(
    *,
    phase: str,
    scope: Mapping[str, Any],
    candidate: Mapping[str, Any] | None,
    counts: Mapping[str, int],
    profile_id: str,
    replication: int | None = None,
    include_tensors: bool = False,
    skip: bool = False,
) -> dict[str, Any]:
    scope_index = int(scope["scope_index"])
    candidate_index = None if candidate is None else int(candidate["candidate_index"])
    if phase == "screen":
        task_id = f"scope-{scope_index:02d}-candidate-{candidate_index:02d}"
    elif phase == "selection":
        task_id = (
            f"scope-{scope_index:02d}-candidate-{candidate_index:02d}"
            f"-replication-{int(replication):02d}"
        )
    elif phase == "scope_finalize":
        task_id = f"scope-{scope_index:02d}-finalize"
    else:
        task_id = f"{phase}-scope-{scope_index:02d}-candidate-{candidate_index}"
    task = {
        "schema": "bayesfilter.q20.process_task.v1",
        "task_id": task_id,
        "phase": phase,
        "profile_id": profile_id,
        "scope_index": scope_index,
        "scope_id": str(scope["scope_id"]),
        "chart_index": int(scope["chart_index"]),
        "component_id": str(scope["component_id"]),
        "beta": float(scope["beta"]),
        "checkpoint_path": str(scope["checkpoint_path"]),
        "candidate_index": candidate_index,
        "step_size": None if candidate is None else float(candidate["step_size"]),
        "num_leapfrog_steps": (
            None if candidate is None else int(candidate["num_leapfrog_steps"])
        ),
        "replication": None if replication is None else int(replication),
        "counts": {str(key): int(value) for key, value in counts.items()},
        "seed_bases": {
            key: list(value) for key, value in _seed_bases(scope_index).items()
        },
        "include_tensors": bool(include_tensors),
        "skip": bool(skip),
    }
    return task


def _partition_tasks(tasks: Sequence[Mapping[str, Any]], worker_count: int) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    count = int(worker_count)
    if count <= 0:
        raise ParallelCampaignError("worker_count must be positive")
    values = tuple(tasks)
    return tuple(
        values[index * len(values) // count : (index + 1) * len(values) // count]
        for index in range(count)
    )


def _worker_environment(cores: int, cpu_ids: Sequence[int], *, gpu: bool = False) -> dict[str, str]:
    core_text = str(int(cores))
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0" if gpu else "-1",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true" if gpu else "false",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "OMP_NUM_THREADS": core_text,
            "OPENBLAS_NUM_THREADS": core_text,
            "MKL_NUM_THREADS": core_text,
            "NUMEXPR_NUM_THREADS": core_text,
            "TF_NUM_INTRAOP_THREADS": core_text,
            "TF_NUM_INTEROP_THREADS": "1",
            "PYTHONUNBUFFERED": "1",
            "Q20_ASSIGNED_CPU_IDS": ",".join(str(int(value)) for value in cpu_ids),
        }
    )
    return environment


def _task_result_path(worker_dir: Path, task_id: str) -> Path:
    safe = str(task_id).replace("/", "_").replace("..", "_")
    return worker_dir / "tasks" / f"{safe}.json"


def _worker_setup(args: argparse.Namespace) -> tuple[Any, Any, Mapping[str, Any]]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ParallelCampaignError("CPU worker requires CUDA_VISIBLE_DEVICES=-1")
    expected = tuple(
        int(value)
        for value in str(os.environ.get("Q20_ASSIGNED_CPU_IDS", "")).split(",")
        if value
    )
    if not expected:
        raise ParallelCampaignError("worker CPU affinity assignment is missing")
    os.sched_setaffinity(0, set(expected))
    if tuple(sorted(os.sched_getaffinity(0))) != tuple(sorted(expected)):
        raise ParallelCampaignError("worker process affinity did not bind exactly")

    # TensorFlow is imported only after CUDA is hidden and the environment
    # thread limits are installed.
    import tensorflow as tf

    tf.config.threading.set_intra_op_parallelism_threads(int(args.cores))
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise ParallelCampaignError("CPU worker found a visible TensorFlow GPU")

    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    bridge = make_q20_tempered_bridge(
        20, jit_compile=True, principal_sqrt_backend=BACKEND
    )
    if str(bridge.target_signature) != TARGET_SIGNATURE:
        raise ParallelCampaignError("q=20 target signature mismatch in worker")
    thread_rows = []
    task_root = f"/proc/{os.getpid()}/task"
    for name in os.listdir(task_root):
        if str(name).isdigit():
            tid = int(name)
            try:
                thread_rows.append(
                    {"tid": tid, "affinity": sorted(os.sched_getaffinity(tid))}
                )
            except ProcessLookupError:
                pass
    metadata = {
        "pid": os.getpid(),
        "worker_index": int(args.worker_index),
        "cores": int(args.cores),
        "assigned_cpu_ids": list(expected),
        "process_affinity": sorted(os.sched_getaffinity(0)),
        "thread_affinity": thread_rows,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_gpu_devices": [
            str(device) for device in tf.config.list_logical_devices("GPU")
        ],
        "tensorflow": tf.__version__,
        "jit_compile": True,
        "dtype": "float64",
        "target_signature": str(bridge.target_signature),
        "bridge_signature": str(bridge.signature),
        "backend": BACKEND,
    }
    return tf, bridge, metadata


def _load_chart(
    tf: Any,
    checkpoint_path: Path,
    task: Mapping[str, Any],
    *,
    expected_bridge_signature: str,
) -> Any:
    from bayesfilter.inference.tempered_transport_ensemble_tf import (
        restore_trainable_transport_checkpoint,
        transport_preflight_state_hash,
    )

    wrapper = _read_json(checkpoint_path)
    checkpoint = wrapper.get("checkpoint", wrapper)
    if not isinstance(checkpoint, Mapping):
        raise ParallelCampaignError("chart checkpoint payload is not a mapping")
    if str(checkpoint.get("target_signature")) != TARGET_SIGNATURE:
        raise ParallelCampaignError("chart checkpoint target signature mismatch")
    if str(checkpoint.get("bridge_signature")) != str(expected_bridge_signature):
        raise ParallelCampaignError("chart checkpoint bridge signature mismatch")
    if str(checkpoint.get("component_id")) != str(task["component_id"]):
        raise ParallelCampaignError("chart component identity mismatch")
    if not math.isclose(float(checkpoint.get("beta")), float(task["beta"]), abs_tol=0.0):
        raise ParallelCampaignError("chart beta identity mismatch")
    restored = restore_trainable_transport_checkpoint(
        checkpoint,
        expected_context={
            "target_signature": TARGET_SIGNATURE,
            "component_id": str(task["component_id"]),
        },
    )
    state_hash = transport_preflight_state_hash(restored)
    if state_hash != str(checkpoint.get("transport_state_hash")):
        raise ParallelCampaignError("restored chart state hash mismatch")
    binder = getattr(restored, "bind_frozen_identity", None)
    if not callable(binder):
        raise ParallelCampaignError("restored chart cannot bind frozen identity")
    binder(
        {
            "checkpoint_sha256": str(checkpoint.get("checkpoint_hash")),
            "training_state_hash": str(checkpoint.get("transport_state_hash")),
            "transport_tensor_hash": state_hash,
        }
    )
    return restored


def _build_scope_runtime(tf: Any, bridge: Any, task: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    from bayesfilter.inference.batched_value_score import (
        FixedTransportValueScoreAdapter,
    )
    from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import (
        FixedTransportReusableRunnerPool,
    )

    chart = _load_chart(
        tf,
        Path(str(task["checkpoint_path"])),
        task,
        expected_bridge_signature=str(bridge.signature),
    )
    base = bridge.fixed_beta_adapter(float(task["beta"]))
    target_scope = f"{base.target_scope}:q20_72core:{task['scope_id']}"
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=base,
        transport=chart,
        target_scope=target_scope,
        evidence_path=str(PLAN.relative_to(ROOT)),
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    return adapter, chart, FixedTransportReusableRunnerPool()


def _make_config(task: Mapping[str, Any], target_scope: str) -> Any:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import (
        FixedTransportHMCKernelTuningConfig,
    )

    counts = {str(key): int(value) for key, value in task["counts"].items()}
    seeds = {
        key: tuple(int(value) for value in values)
        for key, values in task["seed_bases"].items()
    }
    return FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.10,
        maximum_candidate_step_size=2.0,
        step_size_candidates=STEP_GRID,
        leapfrog_grid=LEAPFROG_GRID,
        chain_count=CHAIN_COUNT,
        initial_state_bank=INITIAL_STATE_BANK,
        target_accept_prob=0.70,
        acceptance_band=(0.45, 0.90),
        repair_band=(0.30, 0.95),
        budget_schedule=(2, 2, 2),
        tune_num_results=2,
        screen_num_results=counts["screen_num_results"],
        screen_num_burnin_steps=counts["screen_num_burnin_steps"],
        selection_policy="replicated_min_bulk_ess_per_gradient",
        selection_replications=2,
        selection_num_results=counts["selection_num_results"],
        selection_num_burnin_steps=counts["selection_num_burnin_steps"],
        verification_num_results=counts["verification_num_results"],
        verification_num_burnin_steps=counts["verification_num_burnin_steps"],
        require_modern_rank_normalized_verification=False,
        report_modern_rank_normalized_verification=False,
        tune_seed_base=seeds["tune"],
        screen_seed_base=seeds["screen"],
        selection_seed_base=seeds["selection"],
        verification_seed_base=seeds["verification"],
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope=target_scope,
        target_status_trace_policy="per_chain_step",
        tuning_policy="measured_joint_grid_v1",
    )


def _run_one_task(
    tf: Any,
    bridge: Any,
    task: Mapping[str, Any],
    runtimes: dict[str, tuple[Any, Any, Any]],
    worker_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    started = time.perf_counter()
    if bool(task.get("skip", False)):
        return {
            "schema": "bayesfilter.q20.process_task_result.v1",
            "status": "SKIPPED_NO_NOMINEE",
            "task": dict(task),
            "worker": dict(worker_metadata),
            "elapsed_seconds": 0.0,
        }
    scope_id = str(task["scope_id"])
    runtime = runtimes.get(scope_id)
    if runtime is None:
        runtime = _build_scope_runtime(tf, bridge, task)
        runtimes[scope_id] = runtime
    adapter, _chart, runner_pool = runtime
    target_scope = str(adapter.target_scope)
    config = _make_config(task, target_scope)
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import _run_verification

    captured: dict[str, Any] = {}

    def run_full_chain(adapter_value: Any, initial_state: Any, chain_config: Any) -> Any:
        result = runner_pool(adapter_value, initial_state, chain_config)
        captured["result"] = result
        return result

    candidate_index = int(task["candidate_index"])
    step = float(task["step_size"])
    leapfrog = int(task["num_leapfrog_steps"])
    common = {
        "config": config,
        "adapter": adapter,
        "z0": tf.zeros([DIMENSION], tf.float64),
        "step": step,
        "leapfrog": leapfrog,
        "candidate_index": candidate_index,
        "run_full_chain": run_full_chain,
        "passthrough_exceptions": (),
    }
    phase = str(task["phase"])
    if phase == "screen":
        check = _run_verification(probe_only=True, **common)
    elif phase == "selection":
        check = _run_verification(
            probe_only=False,
            selection_replication=int(task["replication"]),
            **common,
        )
    elif phase == "scope_finalize":
        check = _run_verification(
            probe_only=False,
            post_selection_heldout=True,
            **common,
        )
    else:
        raise ParallelCampaignError(f"unknown task phase: {phase}")
    payload: dict[str, Any] = {
        "schema": "bayesfilter.q20.process_task_result.v1",
        "status": "COMPLETE",
        "task": dict(task),
        "worker": dict(worker_metadata),
        "check": _json_ready(check),
        "runner_evidence": _json_ready(runner_pool.evidence()),
        "elapsed_seconds": time.perf_counter() - started,
        "rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    if bool(task.get("include_tensors", False)) and "result" in captured:
        result = captured["result"]
        payload["parity_tensors"] = {
            "samples": _json_ready(result.samples),
            "target_log_prob": _json_ready(result.trace["target_log_prob"]),
            "target_score": _json_ready(result.trace["target_score"]),
            "log_accept_ratio": _json_ready(result.trace["log_accept_ratio"]),
        }
    return payload


def _worker_main(args: argparse.Namespace) -> int:
    worker_dir = Path(args.output_dir).resolve()
    worker_dir.mkdir(parents=True, exist_ok=False)
    tasks_payload = _read_json(Path(args.task_file).resolve())
    tasks = tasks_payload.get("tasks")
    if not isinstance(tasks, list):
        raise ParallelCampaignError("worker task file must contain a tasks list")
    try:
        tf, bridge, metadata = _worker_setup(args)
        _write_json(
            worker_dir / "worker_ready.json",
            {
                "schema": "bayesfilter.q20.process_worker_ready.v1",
                "status": "READY",
                "phase": str(args.phase),
                "task_count": len(tasks),
                "metadata": dict(metadata),
                "started_at_utc": _utc_now(),
            },
        )
        runtimes: dict[str, tuple[Any, Any, Any]] = {}
        result_paths = []
        for raw_task in tasks:
            if not isinstance(raw_task, Mapping):
                raise ParallelCampaignError("worker task is not a mapping")
            task = dict(raw_task)
            task_path = _task_result_path(worker_dir, str(task["task_id"]))
            _write_json(
                worker_dir / "tasks" / f"{task['task_id']}.start.json",
                {
                    "schema": "bayesfilter.q20.process_task_start.v1",
                    "status": "RUNNING",
                    "task": task,
                    "worker": dict(metadata),
                    "started_at_utc": _utc_now(),
                },
            )
            try:
                result = _run_one_task(tf, bridge, task, runtimes, metadata)
            except Exception as exc:  # Candidate-local failure is durable data.
                result = {
                    "schema": "bayesfilter.q20.process_task_result.v1",
                    "status": "FAILURE",
                    "task": task,
                    "worker": dict(metadata),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                    "elapsed_seconds": None,
                }
            _write_json(task_path, result)
            result_paths.append(str(task_path.relative_to(worker_dir)))
        summary = {
            "schema": "bayesfilter.q20.process_worker_summary.v1",
            "status": "COMPLETE",
            "phase": str(args.phase),
            "worker_index": int(args.worker_index),
            "metadata": dict(metadata),
            "task_count": len(tasks),
            "result_paths": result_paths,
            "runner_scope_count": len(runtimes),
            "finished_at_utc": _utc_now(),
        }
        _write_json(worker_dir / "worker_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        failure = {
            "schema": "bayesfilter.q20.process_worker_failure.v1",
            "status": "FAILURE",
            "phase": str(args.phase),
            "worker_index": int(args.worker_index),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=30),
        }
        try:
            _write_json(worker_dir / "worker_failure.json", failure)
        except Exception:
            pass
        print(json.dumps(failure, sort_keys=True), flush=True)
        return 2


def _wait_for_paths(paths: Sequence[Path], processes: Sequence[subprocess.Popen[str]], deadline: float) -> None:
    while True:
        if all(path.is_file() for path in paths):
            return
        if any(process.poll() is not None and process.returncode != 0 for process in processes):
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(0.1)


def _terminate_processes(processes: Sequence[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    limit = time.monotonic() + 15.0
    for process in processes:
        while process.poll() is None and time.monotonic() < limit:
            time.sleep(0.1)
        if process.poll() is None:
            process.kill()
    for process in processes:
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            pass


def _partial_task_results(output_dir: Path) -> tuple[Mapping[str, Any], ...]:
    """Read only durable task envelopes from a possibly interrupted barrier."""

    if (output_dir / "tasks").is_dir():
        worker_dirs = (output_dir,)
    else:
        worker_dirs = tuple(
            path for path in sorted(output_dir.glob("worker-*")) if path.is_dir()
        )
    rows: list[Mapping[str, Any]] = []
    for worker_dir in worker_dirs:
        for path in sorted((worker_dir / "tasks").glob("*.json")):
            if path.name.endswith(".start.json"):
                continue
            try:
                row = _read_json(path)
            except (OSError, json.JSONDecodeError, ParallelCampaignError):
                # A process can be terminated while a task envelope is being
                # written.  The timeout receipt records that artifact gap.
                continue
            if isinstance(row.get("task"), Mapping):
                rows.append(row)
    rows.sort(key=lambda row: str(row.get("task", {}).get("task_id", "")))
    return tuple(rows)


def _partial_coverage(
    results: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> Mapping[str, Any]:
    """Return non-throwing coverage data for a cap-stopped barrier."""

    expected = [str(value) for value in expected_ids]
    actual = [str(row.get("task", {}).get("task_id", "")) for row in results]
    counts: dict[str, int] = {}
    for task_id in actual:
        counts[task_id] = counts.get(task_id, 0) + 1
    duplicates = sorted(task_id for task_id, count in counts.items() if count > 1)
    missing = sorted(task_id for task_id in expected if task_id not in counts)
    failures = sum(1 for row in results if row.get("status") == "FAILURE")
    return {
        "expected_task_count": len(expected),
        "returned_task_count": len(actual),
        "unique_task_count": len(set(actual)),
        "candidate_local_failure_count": failures,
        "all_tasks_durable": not missing and not duplicates,
        "missing_task_count": len(missing),
        "missing_task_ids": missing,
        "duplicate_task_ids": duplicates,
    }


def _barrier_deadline_payload(
    *,
    phase: str,
    barrier: BarrierTopology,
    tasks: Sequence[Mapping[str, Any]],
    output_dir: Path,
    processes: Sequence[subprocess.Popen[str]],
    stage: str,
) -> Mapping[str, Any]:
    """Terminate a cap-stopped barrier and persist its exact partial state."""

    before = [process.poll() for process in processes]
    _terminate_processes(processes)
    workers = []
    all_results = _partial_task_results(output_dir)
    expected_ids = [str(task.get("task_id", "")) for task in tasks]
    for index, process in enumerate(processes):
        worker_dir = output_dir / f"worker-{index}"
        worker_results = _partial_task_results(worker_dir)
        workers.append(
            {
                "worker_index": index,
                "returncode_before_termination": before[index],
                "returncode_after_termination": process.returncode,
                "ready": (worker_dir / "worker_ready.json").is_file(),
                "summary": (worker_dir / "worker_summary.json").is_file(),
                "failure": (worker_dir / "worker_failure.json").is_file(),
                "started_task_count": len(
                    list((worker_dir / "tasks").glob("*.start.json"))
                ),
                "durable_task_count": len(worker_results),
                "durable_task_ids": [
                    str(row.get("task", {}).get("task_id", ""))
                    for row in worker_results
                ],
            }
        )
    payload = {
        "schema": "bayesfilter.q20.process_barrier_timeout.v1",
        "status": "CAP_STOP_INCOMPLETE",
        "phase": phase,
        "stage": stage,
        "barrier": barrier.payload(),
        "output_dir": str(output_dir.relative_to(ROOT)),
        "expected_task_count": len(expected_ids),
        "expected_task_ids": expected_ids,
        "partial_coverage": _partial_coverage(all_results, expected_ids),
        "workers": workers,
        "terminated_at_utc": _utc_now(),
        "stop_reason": "declared_global_campaign_wall_cap",
        "nonclaims": [
            "partial task records are not a complete staged comparison",
            "no posterior, convergence, whitening, mode-discovery, or ranking claim",
        ],
    }
    receipt = output_dir / "barrier_timeout.json"
    if not receipt.exists():
        _write_json(receipt, payload)
    return payload


def _raise_barrier_deadline(
    *,
    phase: str,
    barrier: BarrierTopology,
    tasks: Sequence[Mapping[str, Any]],
    output_dir: Path,
    processes: Sequence[subprocess.Popen[str]],
    stage: str,
) -> None:
    raise ParallelCampaignDeadline(
        _barrier_deadline_payload(
            phase=phase,
            barrier=barrier,
            tasks=tasks,
            output_dir=output_dir,
            processes=processes,
            stage=stage,
        )
    )


def _launch_barrier(
    *,
    phase: str,
    barrier: BarrierTopology,
    tasks: Sequence[Mapping[str, Any]],
    output_dir: Path,
    available_cpu_ids: Sequence[int],
    deadline: float,
) -> tuple[Mapping[str, Any], ...]:
    assignments = TOPOLOGY.assignments(barrier.name, available_cpu_ids)
    if len(assignments) != barrier.worker_count:
        raise ParallelCampaignError("topology assignment count mismatch")
    if sum(len(row.cpu_ids) for row in assignments) != barrier.worker_core_total:
        raise ParallelCampaignError("topology assignment core count mismatch")
    output_dir.mkdir(parents=True, exist_ok=False)
    partitions = _partition_tasks(tasks, barrier.worker_count)
    processes: list[subprocess.Popen[str]] = []
    logs = []
    try:
        for index, (assignment, partition) in enumerate(zip(assignments, partitions, strict=True)):
            worker_dir = output_dir / f"worker-{index}"
            task_file = output_dir / f"worker-{index}-tasks.json"
            _write_json(
                task_file,
                {
                    "schema": "bayesfilter.q20.process_task_file.v1",
                    "phase": phase,
                    "worker_index": index,
                    "assignment": assignment.payload(),
                    "tasks": [dict(task) for task in partition],
                },
            )
            stderr = (output_dir / f"worker-{index}.stderr.log").open("w", encoding="utf-8")
            logs.append(stderr)
            command = [
                str(WORKER_PYTHON),
                str(SCRIPT),
                "--worker",
                "--phase",
                phase,
                "--worker-index",
                str(index),
                "--cores",
                str(barrier.cores_per_worker),
                "--cpu-ids",
                ",".join(str(value) for value in assignment.cpu_ids),
                "--task-file",
                str(task_file),
                "--output-dir",
                str(worker_dir),
            ]
            environment = _worker_environment(barrier.cores_per_worker, assignment.cpu_ids)
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    bufsize=1,
                )
            )
        ready_paths = [output_dir / f"worker-{index}" / "worker_ready.json" for index in range(barrier.worker_count)]
        _wait_for_paths(ready_paths, processes, deadline)
        if not all(path.is_file() for path in ready_paths):
            if time.monotonic() >= deadline and not any(
                (output_dir / f"worker-{index}" / "worker_failure.json").is_file()
                for index in range(barrier.worker_count)
            ):
                _raise_barrier_deadline(
                    phase=phase,
                    barrier=barrier,
                    tasks=tasks,
                    output_dir=output_dir,
                    processes=processes,
                    stage="worker_readiness",
                )
            _terminate_processes(processes)
            raise ParallelCampaignError(f"{phase} barrier did not reach worker readiness")
        ready_rows = [_read_json(path) for path in ready_paths]
        for index, row in enumerate(ready_rows):
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ParallelCampaignError("worker readiness metadata is missing")
            expected = sorted(assignments[index].cpu_ids)
            if metadata.get("process_affinity") != expected:
                raise ParallelCampaignError(f"worker {index} process affinity mismatch")
            if metadata.get("assigned_cpu_ids") != expected:
                raise ParallelCampaignError(f"worker {index} assigned CPU mismatch")
            if metadata.get("tensorflow_gpu_devices") != []:
                raise ParallelCampaignError(f"worker {index} unexpectedly sees a GPU")
            if metadata.get("jit_compile") is not True:
                raise ParallelCampaignError(f"worker {index} did not enable XLA")
            if metadata.get("target_signature") != TARGET_SIGNATURE:
                raise ParallelCampaignError(f"worker {index} target signature mismatch")
        summary_paths = [
            output_dir / f"worker-{index}" / "worker_summary.json"
            for index in range(barrier.worker_count)
        ]
        _wait_for_paths(summary_paths, processes, deadline)
        if not all(path.is_file() for path in summary_paths):
            if time.monotonic() >= deadline and not any(
                (output_dir / f"worker-{index}" / "worker_failure.json").is_file()
                for index in range(barrier.worker_count)
            ):
                _raise_barrier_deadline(
                    phase=phase,
                    barrier=barrier,
                    tasks=tasks,
                    output_dir=output_dir,
                    processes=processes,
                    stage="worker_summary",
                )
            _terminate_processes(processes)
            details = []
            for index, process in enumerate(processes):
                failure_path = output_dir / f"worker-{index}" / "worker_failure.json"
                details.append(
                    _read_json(failure_path)
                    if failure_path.is_file()
                    else {"returncode": process.returncode}
                )
            raise ParallelCampaignError(f"{phase} worker failure: {details}")
        # A durable summary can be visible just before the child exits. Join
        # each child before interpreting its return code.
        join_timed_out = False
        for process in processes:
            remaining = max(0.1, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                join_timed_out = True
        if join_timed_out and time.monotonic() >= deadline:
            _raise_barrier_deadline(
                phase=phase,
                barrier=barrier,
                tasks=tasks,
                output_dir=output_dir,
                processes=processes,
                stage="worker_join",
            )
        if not all(process.poll() == 0 for process in processes):
            details = []
            for index, process in enumerate(processes):
                failure_path = output_dir / f"worker-{index}" / "worker_failure.json"
                details.append(_read_json(failure_path) if failure_path.is_file() else {"returncode": process.returncode})
            _terminate_processes(processes)
            raise ParallelCampaignError(f"{phase} worker failure: {details}")
        task_results = []
        for index in range(barrier.worker_count):
            worker_dir = output_dir / f"worker-{index}"
            summary = _read_json(worker_dir / "worker_summary.json")
            for relative in summary.get("result_paths", ()):
                task_results.append(_read_json(worker_dir / str(relative)))
        task_results.sort(key=lambda row: str(row.get("task", {}).get("task_id", "")))
        return tuple(task_results)
    finally:
        if any(process.poll() is None for process in processes):
            _terminate_processes(processes)
        for log in logs:
            log.close()


def _flatten(value: Any) -> list[float]:
    if isinstance(value, (tuple, list)):
        result: list[float] = []
        for item in value:
            result.extend(_flatten(item))
        return result
    try:
        return [float(value)]
    except (TypeError, ValueError):
        raise ParallelCampaignError(f"parity value is not numeric: {value!r}")


def _assert_close_nested(left: Any, right: Any, *, rtol: float, atol: float, label: str) -> None:
    a = _flatten(left)
    b = _flatten(right)
    if len(a) != len(b):
        raise ParallelCampaignError(f"{label} shape/length differs: {len(a)} != {len(b)}")
    for index, (x, y) in enumerate(zip(a, b, strict=True)):
        if not math.isclose(x, y, rel_tol=rtol, abs_tol=atol):
            raise ParallelCampaignError(f"{label} differs at flat index {index}: {x} != {y}")


def _run_serial_parity(
    *,
    task: Mapping[str, Any],
    output_dir: Path,
    available_cpu_ids: Sequence[int],
    deadline: float,
) -> Mapping[str, Any]:
    # A one-worker barrier uses the same worker protocol and four-core setting
    # as the screen worker, but runs alone to provide a fixed-seed reference.
    custom = BarrierTopology(
        name="parity",
        worker_count=1,
        cores_per_worker=4,
        work_unit_count=1,
    )
    # The topology allocator is intentionally strict about named barriers;
    # use the first four IDs directly for this auxiliary, non-budgeted check.
    assignment = tuple(sorted(int(value) for value in available_cpu_ids[:4]))
    if len(assignment) != 4:
        raise ParallelCampaignError("four CPUs are required for parity")
    output_dir.mkdir(parents=True, exist_ok=False)
    worker_dir = output_dir / "worker-0"
    task_file = output_dir / "worker-0-tasks.json"
    _write_json(
        task_file,
        {
            "schema": "bayesfilter.q20.process_task_file.v1",
            "phase": "parity",
            "worker_index": 0,
            "assignment": {"worker_index": 0, "cpu_ids": list(assignment), "cores": 4},
            "tasks": [dict(task)],
        },
    )
    stderr = (output_dir / "worker-0.stderr.log").open("w", encoding="utf-8")
    command = [
        str(WORKER_PYTHON),
        str(SCRIPT),
        "--worker",
        "--phase",
        "parity",
        "--worker-index",
        "0",
        "--cores",
        "4",
        "--cpu-ids",
        ",".join(str(value) for value in assignment),
        "--task-file",
        str(task_file),
        "--output-dir",
        str(worker_dir),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=_worker_environment(4, assignment),
        stdout=subprocess.PIPE,
        stderr=stderr,
        text=True,
    )
    try:
        _wait_for_paths([worker_dir / "worker_ready.json"], [process], deadline)
        if not (worker_dir / "worker_ready.json").is_file():
            raise ParallelCampaignError("serial parity worker did not start")
        _wait_for_paths([worker_dir / "worker_summary.json"], [process], deadline)
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        if process.poll() != 0:
            failure = worker_dir / "worker_failure.json"
            raise ParallelCampaignError(
                f"serial parity worker failed: {_read_json(failure) if failure.is_file() else process.returncode}"
            )
        summary = _read_json(worker_dir / "worker_summary.json")
        relative = summary["result_paths"][0]
        return _read_json(worker_dir / str(relative))
    finally:
        if process.poll() is None:
            _terminate_processes([process])
        stderr.close()


def _compare_parity(serial: Mapping[str, Any], parallel: Mapping[str, Any]) -> Mapping[str, Any]:
    left = serial.get("parity_tensors")
    right = parallel.get("parity_tensors")
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        raise ParallelCampaignError("parity tensors are missing")
    checks = {}
    for key in ("samples", "target_log_prob", "log_accept_ratio"):
        _assert_close_nested(left[key], right[key], rtol=1.0e-9, atol=1.0e-9, label=key)
        checks[key] = {"rtol": 1.0e-9, "atol": 1.0e-9, "passed": True}
    _assert_close_nested(left["target_score"], right["target_score"], rtol=1.0e-8, atol=1.0e-8, label="target_score")
    checks["target_score"] = {"rtol": 1.0e-8, "atol": 1.0e-8, "passed": True}
    return {
        "schema": "bayesfilter.q20.process_parity.v1",
        "status": "PASS_SERIAL_PROCESS_PARITY",
        "task_id": parallel.get("task", {}).get("task_id"),
        "checks": checks,
        "serial_worker_pid": serial.get("worker", {}).get("pid"),
        "parallel_worker_pid": parallel.get("worker", {}).get("pid"),
        "serial_target_signature": serial.get("worker", {}).get("target_signature"),
        "parallel_target_signature": parallel.get("worker", {}).get("target_signature"),
    }


def _screen_tasks(scopes: Sequence[Mapping[str, Any]], counts: Mapping[str, int], profile_id: str) -> tuple[Mapping[str, Any], ...]:
    tasks = []
    for scope in scopes:
        for candidate in _candidate_rows():
            tasks.append(
                _base_task(
                    phase="screen",
                    scope=scope,
                    candidate=candidate,
                    counts=counts,
                    profile_id=profile_id,
                )
            )
    return tuple(tasks)


def _selection_tasks(
    scopes: Sequence[Mapping[str, Any]],
    screen_results: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
    profile_id: str,
) -> tuple[Mapping[str, Any], ...]:
    by_key = {
        (
            int(row.get("task", {}).get("scope_index", -1)),
            int(row.get("task", {}).get("candidate_index", -1)),
        ): row
        for row in screen_results
        if isinstance(row.get("task"), Mapping)
    }
    tasks = []
    candidates = _candidate_rows()
    for scope in scopes:
        scope_index = int(scope["scope_index"])
        for candidate in candidates:
            result = by_key.get((scope_index, int(candidate["candidate_index"])))
            check = result.get("check", {}) if isinstance(result, Mapping) else {}
            passed = (
                isinstance(result, Mapping)
                and result.get("status") == "COMPLETE"
                and isinstance(check, Mapping)
                and check.get("final_status") == "passed"
                and not check.get("hard_vetoes")
            )
            if not passed:
                continue
            for replication in (0, 1):
                tasks.append(
                    _base_task(
                        phase="selection",
                        scope=scope,
                        candidate=candidate,
                        counts=counts,
                        profile_id=profile_id,
                        replication=replication,
                    )
                )
    return tuple(tasks)


def _selection_score(row: Mapping[str, Any]) -> tuple[float, float, int, float, int] | None:
    check = row.get("check")
    if not isinstance(check, Mapping) or check.get("final_status") != "passed" or check.get("hard_vetoes"):
        return None
    diagnostics = check.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return None
    efficiency = diagnostics.get("selection_efficiency_diagnostics")
    if not isinstance(efficiency, Mapping):
        return None
    try:
        min_ess = float(efficiency["min_bulk_ess"])
        max_rhat = float(efficiency["max_rhat"])
        leapfrog = int(row["task"]["num_leapfrog_steps"])
        gradient_count = CHAIN_COUNT * (
            int(row["task"]["counts"]["selection_num_burnin_steps"])
            + int(row["task"]["counts"]["selection_num_results"])
        ) * leapfrog
        score = min_ess / float(gradient_count)
        step = float(row["task"]["step_size"])
        index = int(row["task"]["candidate_index"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    if not all(math.isfinite(value) for value in (score, max_rhat)) or score <= 0.0:
        return None
    return (-score, max_rhat, leapfrog, step, index)


def _nominations(
    scopes: Sequence[Mapping[str, Any]], selection_results: Sequence[Mapping[str, Any]]
) -> Mapping[int, Mapping[str, Any] | None]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for row in selection_results:
        task = row.get("task")
        if not isinstance(task, Mapping):
            continue
        key = (int(task.get("scope_index", -1)), int(task.get("candidate_index", -1)))
        grouped.setdefault(key, []).append(row)
    nominations: dict[int, Mapping[str, Any] | None] = {}
    for scope in scopes:
        scope_index = int(scope["scope_index"])
        candidates = []
        for candidate in _candidate_rows():
            rows = grouped.get((scope_index, int(candidate["candidate_index"])), [])
            if len(rows) != 2:
                continue
            scores = [_selection_score(row) for row in rows]
            if any(score is None for score in scores):
                continue
            assert scores[0] is not None and scores[1] is not None
            # The existing policy maximizes the minimum replication score,
            # then minimizes the largest R-hat, L, epsilon, and candidate index.
            key = (
                min(scores[0][0], scores[1][0]),
                max(scores[0][1], scores[1][1]),
                scores[0][2],
                scores[0][3],
                scores[0][4],
            )
            candidates.append((key, candidate))
        if candidates:
            _key, winner = min(candidates, key=lambda item: item[0])
            nominations[scope_index] = winner
        else:
            nominations[scope_index] = None
    return nominations


def _finalize_tasks(
    scopes: Sequence[Mapping[str, Any]],
    nominations: Mapping[int, Mapping[str, Any] | None],
    counts: Mapping[str, int],
    profile_id: str,
) -> tuple[Mapping[str, Any], ...]:
    tasks = []
    for scope in scopes:
        candidate = nominations.get(int(scope["scope_index"]))
        tasks.append(
            _base_task(
                phase="scope_finalize",
                scope=scope,
                candidate=candidate,
                counts=counts,
                profile_id=profile_id,
                skip=candidate is None,
            )
        )
    return tuple(tasks)


def _validate_task_results(
    results: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]
) -> Mapping[str, Any]:
    actual = [str(row.get("task", {}).get("task_id", "")) for row in results]
    if sorted(actual) != sorted(str(value) for value in expected_ids):
        raise ParallelCampaignError(
            f"task coverage mismatch: expected {len(expected_ids)}, got {len(actual)}"
        )
    if len(actual) != len(set(actual)):
        raise ParallelCampaignError("duplicate task result identity")
    failures = [row for row in results if row.get("status") == "FAILURE"]
    return {
        "expected_task_count": len(expected_ids),
        "returned_task_count": len(actual),
        "unique_task_count": len(set(actual)),
        "candidate_local_failure_count": len(failures),
        "all_tasks_durable": True,
    }


def _run_canary(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    available = _available_cpu_ids()
    TOPOLOGY.validate_available_cpu_ids(available)
    output = _new_output_root(args.output_root, "canary")
    output.mkdir(parents=True, exist_ok=False)
    scopes = _scope_specs(FIXTURE_CHECKPOINTS)
    profile_id = "q20_72core_process_canary_v1"
    counts = CANARY_COUNTS

    # The parity task is the first screen candidate.  It uses the historical
    # chart only as a mechanics fixture; no canary artifact can be promoted.
    parity_task = _base_task(
        phase="screen",
        scope=scopes[0],
        candidate=_candidate_rows()[0],
        counts=counts,
        profile_id=profile_id,
        include_tensors=True,
    )
    serial = _run_serial_parity(
        task=parity_task,
        output_dir=output / "parity-serial",
        available_cpu_ids=available,
        deadline=time.monotonic() + CANARY_CAP_SECONDS,
    )
    screen_tasks = tuple(
        {
            **dict(task),
            "include_tensors": task["task_id"] == parity_task["task_id"],
        }
        for task in _screen_tasks((scopes[0],), counts, profile_id)
    )
    screen_results = _launch_barrier(
        phase="screen",
        barrier=TOPOLOGY.screen,
        tasks=screen_tasks,
        output_dir=output / "screen",
        available_cpu_ids=available,
        deadline=time.monotonic() + CANARY_CAP_SECONDS,
    )
    parity_parallel = next(
        row for row in screen_results if row.get("task", {}).get("task_id") == parity_task["task_id"]
    )
    parity = _compare_parity(serial, parity_parallel)
    selection_task = _base_task(
        phase="selection",
        scope=scopes[0],
        candidate=_candidate_rows()[0],
        counts=counts,
        profile_id=profile_id,
        replication=0,
    )
    selection_task_1 = {**selection_task, "task_id": selection_task["task_id"].replace("replication-00", "replication-01"), "replication": 1}
    selection_results = _launch_barrier(
        phase="selection",
        barrier=TOPOLOGY.selection,
        tasks=(selection_task, selection_task_1),
        output_dir=output / "selection",
        available_cpu_ids=available,
        deadline=time.monotonic() + CANARY_CAP_SECONDS,
    )
    final_tasks = _finalize_tasks(
        scopes,
        {int(scope["scope_index"]): _candidate_rows()[0] for scope in scopes},
        counts,
        profile_id,
    )
    final_results = _launch_barrier(
        phase="scope_finalize",
        barrier=TOPOLOGY.scope_finalize,
        tasks=final_tasks,
        output_dir=output / "scope_finalize",
        available_cpu_ids=available,
        deadline=time.monotonic() + CANARY_CAP_SECONDS,
    )
    elapsed = time.perf_counter() - started
    if elapsed > CANARY_CAP_SECONDS:
        raise ParallelCampaignError("canary wall cap exceeded")
    summary = {
        "schema": "bayesfilter.q20.72core_process_canary.v1",
        "status": "PASS_72CORE_PROCESS_CANARY",
        "role": "engineering_and_q20_mechanics_only",
        "plan": str(PLAN.relative_to(ROOT)),
        "topology": TOPOLOGY.payload(),
        "available_logical_cpu_count": len(available),
        "available_logical_cpu_ids": list(available),
        "fixture_role": "historical_frozen_chart_mechanics_fixture_only",
        "target_signature": TARGET_SIGNATURE,
        "backend": BACKEND,
        "parity": parity,
        "barriers": {
            "screen": {
                "task_count": len(screen_results),
                "coverage": _validate_task_results(
                    screen_results, [task["task_id"] for task in screen_tasks]
                ),
            },
            "selection": {
                "task_count": len(selection_results),
                "coverage": _validate_task_results(
                    selection_results, [task["task_id"] for task in (selection_task, selection_task_1)]
                ),
            },
            "scope_finalize": {
                "task_count": len(final_results),
                "coverage": _validate_task_results(
                    final_results, [task["task_id"] for task in final_tasks]
                ),
            },
        },
        "wall_seconds": elapsed,
        "run_manifest": {
            "git": _git_payload(),
            "command": [str(value) for value in sys.argv],
            "python": sys.version,
            "python_executable": str(WORKER_PYTHON),
            "platform": platform.platform(),
            "started_at_utc": _utc_now(),
            "wall_cap_seconds": CANARY_CAP_SECONDS,
            "cpu_gpu_status": "CPU-only children; CUDA hidden before TensorFlow import",
            "xla": True,
            "memory_growth": "not applicable to CPU children; GPU hidden",
            "output_root": str(output.relative_to(ROOT)),
            "source_sha256": {
                "script": _sha256(SCRIPT),
                "plan": _sha256(PLAN),
            },
        },
        "inference_status": {
            "hard_veto_screen": "passed topology, identity, finite, and parity checks",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "worker timing/RSS only",
            "default_readiness": "not assessed; GPU default unchanged",
            "next_evidence_needed": "fresh chart preparation and full staged diagnostic",
        },
        "nonclaims": [
            "no whitening, mode discovery, convergence, posterior correctness, or sampler ranking",
            "no CPU default or GPU speedup claim",
            "historical fixture is not fresh tuning evidence",
        ],
    }
    _write_json(output / "canary_summary.json", summary)
    return summary


def _new_output_root(requested: Path | None, label: str) -> Path:
    if requested is not None:
        candidate = (ROOT / requested).resolve() if not requested.is_absolute() else requested.resolve()
        if candidate.exists():
            raise ParallelCampaignError(f"requested output root already exists: {candidate}")
        if not candidate.is_relative_to(ROOT):
            raise ParallelCampaignError("output root must be inside the repository")
        return candidate
    base = ROOT / "docs/plans/artifacts/ssl-lstm-q20-72core-process-parallel-2026-09-03" / label
    attempt = 1
    while True:
        candidate = base / f"attempt-{attempt:02d}"
        if not candidate.exists():
            return candidate
        attempt += 1


def _prepare_main(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise ParallelCampaignError("chart preparation requires CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
        raise ParallelCampaignError("chart preparation requires TF_FORCE_GPU_ALLOW_GROWTH=true")
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise ParallelCampaignError(f"expected one visible GPU for preparation, got {len(logical)}")
    from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge

    # ``docs`` is not a Python package. Load the source-owned Phase 9A chart
    # builder by its exact path without changing repository package layout.
    phase9a_path = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py"
    spec = importlib.util.spec_from_file_location("q20_phase9a_builder", phase9a_path)
    if spec is None or spec.loader is None:
        raise ParallelCampaignError("cannot load the source-owned Phase 9A chart builder")
    phase9a = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = phase9a
    spec.loader.exec_module(phase9a)
    _FULL_REPLAY_CANARY_PROFILE = phase9a._FULL_REPLAY_CANARY_PROFILE
    _build_fresh_chart = phase9a._build_fresh_chart
    _reliability = phase9a._reliability

    profile = replace(
        _FULL_REPLAY_CANARY_PROFILE,
        profile_id="q20_72core_fresh_chart_v1",
        plan_path=PLAN,
        initialization_roots=((20260903, 91001), (20260903, 91002)),
        preflight_roots=((20260903, 91101), (20260903, 91102)),
        training_roots=((20260903, 91201), (20260903, 91202)),
        tuning_roots=tuple((20260903, 91300 + index) for index in range(1, 7)),
        transition_root=(20260903, 91401),
        reliability_root=(20260903, 91501),
        scope_start=None,
        scope_limit=None,
    )
    bridge = make_q20_tempered_bridge(20, jit_compile=True, principal_sqrt_backend=BACKEND)
    if str(bridge.target_signature) != TARGET_SIGNATURE:
        raise ParallelCampaignError("preparation target signature mismatch")
    records = []
    beta_one_charts = []
    for chart_index, component_id in enumerate(COMPONENTS):
        charts, checkpoints, receipts = _build_fresh_chart(
            tf, bridge, chart_index, component_id, profile, output
        )
        records.append(
            {
                "chart_index": chart_index,
                "component_id": component_id,
                "checkpoint_hashes": [row["checkpoint"]["checkpoint_hash"] for row in checkpoints],
                "state_hashes": [row["state_hash"] for row in checkpoints],
                "preflight_count": len(receipts),
                "checkpoint_paths": [
                    str(output / f"chart-{chart_index}" / f"beta-{beta:g}" / "chart_checkpoint.json")
                    for beta in BETAS
                ],
            }
        )
        beta_one_charts.append(charts[1.0])
    # Reliability is a mechanics check on the newly prepared maps. It is not a
    # Gaussianization or posterior-quality claim.
    _reliability(tf, bridge, beta_one_charts, beta=1.0, profile=profile)
    checkpoint_paths = [
        output / f"chart-{chart}" / f"beta-{beta:g}" / "chart_checkpoint.json"
        for chart in range(len(COMPONENTS))
        for beta in BETAS
    ]
    manifest = {
        "schema": "bayesfilter.q20.72core_fresh_chart_manifest.v1",
        "status": "PASS_FRESH_CHART_PREPARATION",
        "profile_id": profile.profile_id,
        "target_signature": TARGET_SIGNATURE,
        "bridge_signature": str(bridge.signature),
        "backend": BACKEND,
        "memory_policy": _json_ready(memory),
        "logical_gpu": str(logical[0].name),
        "records": records,
        "checkpoint_paths": [str(path.relative_to(ROOT)) for path in checkpoint_paths],
        "checkpoint_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in checkpoint_paths},
        "git": _git_payload(),
        "plan": str(PLAN.relative_to(ROOT)),
        "nonclaims": ["fresh chart mechanics only; no posterior or whitening claim"],
    }
    _write_json(output / "prepare_manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0


def _run_prepare(output: Path, deadline: float) -> Mapping[str, Any]:
    if output.exists():
        raise ParallelCampaignError(f"preparation output already exists: {output}")
    # The preparation child creates ``output`` with ``exist_ok=False``. Keep
    # its log beside the reserved directory so early failures remain durable.
    output.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = output.parent / f"{output.name}.prepare.stderr.log"
    stderr = stderr_path.open("w", encoding="utf-8")
    command = [str(WORKER_PYTHON), str(SCRIPT), "--prepare", "--output-dir", str(output)]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF32": "1",
        }
    )
    process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE, stderr=stderr, text=True)
    try:
        while process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        if process.poll() is None:
            _terminate_processes([process])
            raise ParallelCampaignError("fresh chart preparation timed out")
        if process.returncode != 0:
            failure = output / "prepare_failure.json"
            if not failure.is_file():
                _write_json(
                    output.parent / f"{output.name}.prepare_failure.json",
                    {
                        "schema": "bayesfilter.q20.72core_prepare_failure.v1",
                        "status": "FAIL_FRESH_CHART_PREPARATION",
                        "returncode": process.returncode,
                        "stderr_path": str(stderr_path.relative_to(ROOT)),
                        "output_dir": str(output.relative_to(ROOT)),
                        "error": "preparation child exited without a manifest",
                    },
                )
            raise ParallelCampaignError(
                f"fresh chart preparation failed: {_read_json(failure) if failure.is_file() else process.returncode}"
            )
        return _read_json(output / "prepare_manifest.json")
    finally:
        if process.poll() is None:
            _terminate_processes([process])
        stderr.close()


def _full_cap_closeout(
    *,
    output: Path,
    started: float,
    preparation: Mapping[str, Any],
    scopes: Sequence[Mapping[str, Any]],
    profile_id: str,
    counts: Mapping[str, int],
    screen_tasks: Sequence[Mapping[str, Any]],
    screen_results: Sequence[Mapping[str, Any]],
    selection_tasks: Sequence[Mapping[str, Any]],
    selection_results: Sequence[Mapping[str, Any]],
    final_tasks: Sequence[Mapping[str, Any]],
    final_results: Sequence[Mapping[str, Any]],
    deadline: ParallelCampaignDeadline,
    available: Sequence[int],
) -> Mapping[str, Any]:
    """Persist a typed full-run result when a barrier reaches the wall cap."""

    elapsed = time.perf_counter() - started
    summary = {
        "schema": "bayesfilter.q20.72core_process_full_result.v1",
        "status": "CAP_STOP_INCOMPLETE",
        "role": "diagnostic_tuning_performance_only",
        "plan": str(PLAN.relative_to(ROOT)),
        "topology": TOPOLOGY.payload(),
        "target_signature": TARGET_SIGNATURE,
        "backend": BACKEND,
        "fresh_chart_preparation": preparation,
        "nominations": {
            str(key): value
            for key, value in _nominations(scopes, selection_results).items()
        },
        "barriers": {
            "screen": _partial_coverage(
                screen_results,
                [str(task["task_id"]) for task in screen_tasks],
            ),
            "selection": _partial_coverage(
                selection_results,
                [str(task["task_id"]) for task in selection_tasks],
            ),
            "scope_finalize": _partial_coverage(
                final_results,
                [str(task["task_id"]) for task in final_tasks],
            ),
        },
        "selection_task_count": len(selection_tasks),
        "scope_finalize_pass_count": sum(
            1
            for row in final_results
            if row.get("status") == "COMPLETE"
            and row.get("check", {}).get("final_status") == "passed"
        ),
        "wall_seconds": elapsed,
        "cap_stop": deadline.payload,
        "run_manifest": {
            "git": _git_payload(),
            "command": [str(value) for value in sys.argv],
            "python": sys.version,
            "python_executable": str(WORKER_PYTHON),
            "platform": platform.platform(),
            "started_at_utc": _utc_now(),
            "wall_cap_seconds": FULL_CAP_SECONDS,
            "total_campaign_cap_seconds": TOTAL_CAP_SECONDS,
            "available_logical_cpu_ids": list(available),
            "cpu_gpu_status": "CPU-only staged workers; fresh chart preparation used GPU0",
            "xla": True,
            "memory_growth": "verified in fresh chart preparation; CPU workers hide GPU",
            "output_root": str(output.relative_to(ROOT)),
            "source_sha256": {
                "script": _sha256(SCRIPT),
                "plan": _sha256(PLAN),
            },
        },
        "decision_table": {
            "decision": "declared campaign cap stopped the staged schedule",
            "primary_criterion": "complete staged task set under the declared cap",
            "veto_status": "resource/cap stop; incomplete barrier is not promotion evidence",
            "main_uncertainty": "unfinished tasks and descriptive CPU timing",
            "next_action": "repair/refresh the scheduling plan before any further full attempt",
            "not_concluded": "posterior correctness, convergence, whitening, mode discovery, or sampler ranking",
        },
        "inference_status": {
            "hard_veto_screen": "cap stop with explicit partial coverage; no complete full-run screen",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "completed-task timing and diagnostics only",
            "default_readiness": "not assessed; GPU default unchanged",
            "next_evidence_needed": "a reviewed schedule/cap repair and an independent sequential posterior-validation plan",
        },
        "nonclaims": [
            "no posterior convergence or correctness claim",
            "no whitening or mode-discovery claim",
            "no CPU default, GPU speedup, or high-dimensional scaling claim",
            "no Phase 9B admission",
            "partial task records are not a complete candidate comparison",
        ],
    }
    if not (output / "full_summary.json").exists():
        _write_json(output / "full_summary.json", summary)
    return summary


def _run_full(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    campaign_deadline = time.monotonic() + FULL_CAP_SECONDS
    available = _available_cpu_ids()
    TOPOLOGY.validate_available_cpu_ids(available)
    output = _new_output_root(args.output_root, "full")
    output.mkdir(parents=True, exist_ok=False)
    profile_id = "q20_72core_process_full_diagnostic_v1"
    counts = FULL_COUNTS
    prepare_dir = output / "fresh_chart_preparation"
    preparation = _run_prepare(
        prepare_dir,
        min(campaign_deadline, time.monotonic() + min(1800.0, FULL_CAP_SECONDS)),
    )
    checkpoint_paths = [ROOT / str(path) for path in preparation["checkpoint_paths"]]
    scopes = _scope_specs(checkpoint_paths)
    screen_tasks = _screen_tasks(scopes, counts, profile_id)
    screen_results: tuple[Mapping[str, Any], ...] = ()
    selection_tasks: tuple[Mapping[str, Any], ...] = ()
    selection_results: tuple[Mapping[str, Any], ...] = ()
    final_tasks: tuple[Mapping[str, Any], ...] = ()
    final_results: tuple[Mapping[str, Any], ...] = ()
    try:
        screen_results = _launch_barrier(
            phase="screen",
            barrier=TOPOLOGY.screen,
            tasks=screen_tasks,
            output_dir=output / "screen",
            available_cpu_ids=available,
            deadline=campaign_deadline,
        )
    except ParallelCampaignDeadline as exc:
        return _full_cap_closeout(
            output=output,
            started=started,
            preparation=preparation,
            scopes=scopes,
            profile_id=profile_id,
            counts=counts,
            screen_tasks=screen_tasks,
            screen_results=_partial_task_results(output / "screen"),
            selection_tasks=selection_tasks,
            selection_results=selection_results,
            final_tasks=final_tasks,
            final_results=final_results,
            deadline=exc,
            available=available,
        )
    screen_coverage = _validate_task_results(
        screen_results, [task["task_id"] for task in screen_tasks]
    )
    if not any(row.get("status") == "COMPLETE" for row in screen_results):
        raise ParallelCampaignError(
            "screen barrier returned no completed candidate; refusing to treat a common runtime failure as scope-local vetoes"
        )
    selection_tasks = _selection_tasks(scopes, screen_results, counts, profile_id)
    # Keep replication streams on separate workers, exactly as the requested
    # 2x8 allocation.  A candidate-local failure remains in its result row.
    selection_by_rep = {
        rep: tuple(task for task in selection_tasks if int(task["replication"]) == rep)
        for rep in (0, 1)
    }
    try:
        selection_results = _launch_barrier(
            phase="selection",
            barrier=TOPOLOGY.selection,
            tasks=selection_by_rep[0] + selection_by_rep[1],
            output_dir=output / "selection",
            available_cpu_ids=available,
            deadline=campaign_deadline,
        )
    except ParallelCampaignDeadline as exc:
        return _full_cap_closeout(
            output=output,
            started=started,
            preparation=preparation,
            scopes=scopes,
            profile_id=profile_id,
            counts=counts,
            screen_tasks=screen_tasks,
            screen_results=screen_results,
            selection_tasks=selection_tasks,
            selection_results=_partial_task_results(output / "selection"),
            final_tasks=final_tasks,
            final_results=final_results,
            deadline=exc,
            available=available,
        )
    selection_coverage = _validate_task_results(
        selection_results, [task["task_id"] for task in selection_tasks]
    )
    nominations = _nominations(scopes, selection_results)
    final_tasks = _finalize_tasks(scopes, nominations, counts, profile_id)
    try:
        final_results = _launch_barrier(
            phase="scope_finalize",
            barrier=TOPOLOGY.scope_finalize,
            tasks=final_tasks,
            output_dir=output / "scope_finalize",
            available_cpu_ids=available,
            deadline=campaign_deadline,
        )
    except ParallelCampaignDeadline as exc:
        return _full_cap_closeout(
            output=output,
            started=started,
            preparation=preparation,
            scopes=scopes,
            profile_id=profile_id,
            counts=counts,
            screen_tasks=screen_tasks,
            screen_results=screen_results,
            selection_tasks=selection_tasks,
            selection_results=selection_results,
            final_tasks=final_tasks,
            final_results=_partial_task_results(output / "scope_finalize"),
            deadline=exc,
            available=available,
        )
    final_coverage = _validate_task_results(
        final_results, [task["task_id"] for task in final_tasks]
    )
    elapsed = time.perf_counter() - started
    if elapsed > FULL_CAP_SECONDS:
        raise ParallelCampaignError("full staged campaign wall cap exceeded")
    final_passes = sum(
        1
        for row in final_results
        if row.get("status") == "COMPLETE"
        and row.get("check", {}).get("final_status") == "passed"
    )
    summary = {
        "schema": "bayesfilter.q20.72core_process_full_result.v1",
        "status": "PASS_72CORE_STAGED_MECHANICS" if final_passes == len(scopes) else "COMPLETE_WITH_SCOPE_VETOES",
        "role": "diagnostic_tuning_performance_only",
        "plan": str(PLAN.relative_to(ROOT)),
        "topology": TOPOLOGY.payload(),
        "target_signature": TARGET_SIGNATURE,
        "backend": BACKEND,
        "fresh_chart_preparation": preparation,
        "nominations": {str(key): value for key, value in nominations.items()},
        "barriers": {
            "screen": screen_coverage,
            "selection": selection_coverage,
            "scope_finalize": final_coverage,
        },
        "selection_task_count": len(selection_tasks),
        "scope_finalize_pass_count": final_passes,
        "wall_seconds": elapsed,
        "run_manifest": {
            "git": _git_payload(),
            "command": [str(value) for value in sys.argv],
            "python": sys.version,
            "python_executable": str(WORKER_PYTHON),
            "platform": platform.platform(),
            "started_at_utc": _utc_now(),
            "wall_cap_seconds": FULL_CAP_SECONDS,
            "total_campaign_cap_seconds": TOTAL_CAP_SECONDS,
            "available_logical_cpu_ids": list(available),
            "cpu_gpu_status": "CPU-only staged workers; fresh chart preparation used GPU0",
            "xla": True,
            "memory_growth": "verified in fresh chart preparation; CPU workers hide GPU",
            "output_root": str(output.relative_to(ROOT)),
            "source_sha256": {"script": _sha256(SCRIPT), "plan": _sha256(PLAN)},
        },
        "decision_table": {
            "decision": "parallel mechanics schedule completed" if final_passes == len(scopes) else "schedule completed with scope-local vetoes",
            "primary_criterion": "all declared staged task identities and artifacts",
            "veto_status": "no infrastructure veto" if final_passes == len(scopes) else "scope-local numerical/tuning vetoes present",
            "main_uncertainty": "descriptive CPU timing and short-chain diagnostics",
            "next_action": "write/refresh a separately reviewed Phase 9B plan only after terminal review",
            "not_concluded": "posterior correctness, convergence, whitening, mode discovery, or sampler ranking",
        },
        "inference_status": {
            "hard_veto_screen": "identity/finite/resource/artifact status recorded per task",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "timing, RSS, acceptance, and ESS diagnostics",
            "default_readiness": "not assessed; GPU default unchanged",
            "next_evidence_needed": "terminal review and an independent sequential posterior-validation plan",
        },
        "nonclaims": [
            "no posterior convergence or correctness claim",
            "no whitening or mode-discovery claim",
            "no CPU default, GPU speedup, or high-dimensional scaling claim",
            "no Phase 9B admission",
        ],
    }
    _write_json(output / "full_summary.json", summary)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--phase", default="screen")
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--cpu-ids", default="")
    parser.add_argument("--task-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    modes = sum(bool(value) for value in (args.canary, args.full, args.prepare, args.worker))
    if modes != 1:
        parser.error("choose exactly one of --canary, --full, --prepare, or --worker")
    if args.worker and (args.task_file is None or args.output_dir is None):
        parser.error("--worker requires --task-file and --output-dir")
    if args.prepare and args.output_dir is None:
        parser.error("--prepare requires --output-dir")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.worker:
            return _worker_main(args)
        if args.prepare:
            return _prepare_main(args)
        result = _run_canary(args) if args.canary else _run_full(args)
        print(json.dumps(_json_ready(result), indent=2, sort_keys=True), flush=True)
        return 0
    except (ParallelCampaignError, ProcessTopologyError) as exc:
        if args.prepare and args.output_dir is not None:
            output = Path(args.output_dir).resolve()
            if output.is_dir() and not (output / "prepare_failure.json").exists():
                try:
                    _write_json(
                        output / "prepare_failure.json",
                        {
                            "schema": "bayesfilter.q20.72core_prepare_failure.v1",
                            "status": "FAIL_FRESH_CHART_PREPARATION",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                except Exception:
                    pass
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if args.prepare and args.output_dir is not None:
            output = Path(args.output_dir).resolve()
            if output.is_dir() and not (output / "prepare_failure.json").exists():
                try:
                    _write_json(
                        output / "prepare_failure.json",
                        {
                            "schema": "bayesfilter.q20.72core_prepare_failure.v1",
                            "status": "FAIL_FRESH_CHART_PREPARATION",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                            "traceback": traceback.format_exc(limit=30),
                        },
                    )
                except Exception:
                    pass
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
