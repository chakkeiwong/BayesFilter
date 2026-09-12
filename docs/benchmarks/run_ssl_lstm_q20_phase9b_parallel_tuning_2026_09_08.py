#!/usr/bin/env python3
"""Phase 0 process-parallel tuning diagnostic; never a P1 or posterior run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-execution-plan-2026-09-07.md"
RESULT = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-parallel-tuning-gpu-result-2026-09-08.md"
SOURCE_RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9a_fresh_tuning_preflight_2026_08_31.py"
CLAIM_BOUNDARY = "phase9b_p0_parallel_tuning_diagnostic_only"
ARMS = ("factor", "strict")
SCOPE_INDEX = 2
CHART_INDEX = 0
BETA = 1.0
WORKER_PEAK_MIB = 4096
MONITOR_INTERVAL_SECONDS = 5.0
MONITOR_TIMEOUT_SECONDS = 1.0

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ParallelDiagnosticError(RuntimeError):
    """The bounded Phase 0 tuning diagnostic cannot preserve its contract."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _load_source():
    spec = importlib.util.spec_from_file_location("phase9b_parallel_phase9a_source", SOURCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ParallelDiagnosticError("cannot load the source-owned tuning runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _source_hashes() -> dict[str, str]:
    paths = {SCRIPT, SOURCE_RUNNER}
    for package in ("inference", "nonlinear", "ops", "runtime"):
        for path in (ROOT / "bayesfilter" / package).rglob("*"):
            if path.is_file() and path.suffix in {".py", ".so"}:
                paths.add(path)
    paths.add(ROOT / "bayesfilter/__init__.py")
    return {str(path.relative_to(ROOT)): _sha256(path) for path in sorted(paths)}


def _profile(source: Any, output: Path, arm: str, worker_seconds: float):
    if arm not in ARMS:
        raise ParallelDiagnosticError("unexpected tuning arm")

    def seed(role: str) -> tuple[int, int]:
        digest = hashlib.sha256(f"{output.resolve()}:{arm}:{role}".encode()).digest()
        return (int.from_bytes(digest[:4], "big") % (2**31 - 1),
                int.from_bytes(digest[4:8], "big") % (2**31 - 1024))

    return replace(
        source._FACTOR_TUNING_R2_PROFILE,
        profile_id=f"phase9b_p0_parallel_{arm}_{_json_hash(str(output.resolve()))[:12]}",
        plan_path=PLAN,
        initialization_roots=tuple(seed(f"initialization-{index}") for index in range(2)),
        preflight_roots=tuple(seed(f"preflight-{index}") for index in range(2)),
        training_roots=tuple(seed(f"training-{index}") for index in range(2)),
        tuning_roots=tuple(seed(f"tuning-{index}") for index in range(6)),
        transition_root=seed("transition"), reliability_root=seed("reliability"),
        scope_start=SCOPE_INDEX, scope_limit=1, material_cap_seconds=worker_seconds,
        principal_sqrt_backend=source.FACTOR_BACKEND if arm == "factor" else source.STRICT_BACKEND,
    )


def _job(source: Any, output: Path, arm: str, worker_seconds: float, sources: Mapping[str, str], *, smoke_only: bool = False):
    profile = _profile(source, output, arm, worker_seconds)
    return json.loads(json.dumps({
        "schema": "bayesfilter.phase9b_parallel_tuning_job.v1",
        "arm": arm, "campaign_output": str(output.resolve()),
        "output_dir": str((output / arm).resolve()), "worker_seconds": worker_seconds,
        "scope": {"scope_index": SCOPE_INDEX, "chart_index": CHART_INDEX, "beta": BETA},
        "profile": profile.payload(), "target_signature": source.EXPECTED_TARGET_SIGNATURE,
        "source_hashes": dict(sources), "plan_hash": _sha256(PLAN),
        "claim_boundary": CLAIM_BOUNDARY,
        "mode": "gpu_smoke" if smoke_only else "tuning",
    }))


def _require_framework_free() -> None:
    if any(name in sys.modules for name in ("tensorflow", "tensorflow_probability", "jax", "torch")):
        raise ParallelDiagnosticError("coordinator/worker setup must precede framework import")


def _gpu_smoke(tf: Any, device_name: str, job: Mapping[str, Any]) -> Mapping[str, Any]:
    @tf.function(input_signature=[tf.TensorSpec((16, 16), tf.float64)], jit_compile=True)
    def kernel(values):
        return tf.linalg.matmul(values, values, transpose_b=True) + tf.eye(16, dtype=tf.float64)

    with tf.device(device_name):
        values = tf.random.stateless_normal(
            (16, 16), seed=job["profile"]["seed_namespace"]["initialization_roots"][0], dtype=tf.float64
        )
        timings = []
        outputs = []
        for _ in range(2):
            started = time.monotonic()
            output = kernel(values)
            finite = bool(tf.reduce_all(tf.math.is_finite(output)).numpy())
            timings.append(time.monotonic() - started)
            if not finite or "GPU:0" not in output.device:
                raise ParallelDiagnosticError("GPU smoke returned nonfinite or misplaced values")
            outputs.append(output)
        hlo = str(kernel.experimental_get_compiler_ir(values)(stage="hlo"))
    with tf.device("/CPU:0"):
        reference_values = tf.identity(values)
        reference = tf.linalg.matmul(reference_values, reference_values, transpose_b=True) + tf.eye(16, dtype=tf.float64)
        errors = [float(tf.reduce_max(tf.abs(tf.identity(output) - reference)).numpy()) for output in outputs]
    if max(errors) > 1e-10 or kernel.experimental_get_tracing_count() != 1 or not hlo:
        raise ParallelDiagnosticError("GPU smoke failed repeated-shape XLA/reference checks")
    return {
        "status": "PASS_GPU_XLA_STARTUP_SMOKE", "device": device_name,
        "call_seconds": timings, "max_absolute_errors": errors,
        "tolerance": 1e-10, "tracing_count": kernel.experimental_get_tracing_count(),
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "role": "startup_mechanics_only_no_target_or_tuning_evidence",
    }


def _worker(job_path: Path) -> int:
    _require_framework_free()
    job = json.loads(job_path.read_text())
    source = _load_source()
    expected = _job(source, Path(job["campaign_output"]), job["arm"], job["worker_seconds"],
                    _source_hashes(), smoke_only=job.get("mode") == "gpu_smoke")
    if job != expected:
        raise ParallelDiagnosticError("worker source, plan, profile or scope changed")
    output = Path(job["output_dir"])
    if os.environ.get("BAYESFILTER_WORKER_OUTPUT_DIR") != str(output):
        raise ParallelDiagnosticError("worker output environment mismatch")
    uuid = os.environ.get("BAYESFILTER_SELECTED_GPU_UUID", "")
    if not uuid.startswith("GPU-") or os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise ParallelDiagnosticError("worker requires exactly its assigned GPU UUID")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ParallelDiagnosticError("memory growth must precede TensorFlow import")
    from bayesfilter.runtime.display_gpu_policy import (
        POLICY_ID, check_runtime_headroom, probe_inventory, select_gpus,
    )

    if os.environ.get("BAYESFILTER_GPU_SELECTION_POLICY_ID") != POLICY_ID:
        raise ParallelDiagnosticError("worker GPU policy mismatch")
    placement = select_gpus(probe_inventory(), len(ARMS), estimated_peak_mib=WORKER_PEAK_MIB)
    selected = next((row for row in placement["selected"] if row["uuid"] == uuid), None)
    if selected is None:
        raise ParallelDiagnosticError("assigned GPU is no longer eligible before framework import")
    selection = {"selected": selected}
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write_json(output / "run_start.json", {"job": job, "pid": os.getpid(), "placement": placement,
                                           "started_at_unix": time.time(), "command": sys.argv,
                                           "environment": {name: os.environ.get(name) for name in (
                                               "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
                                               "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                                               "OMP_NUM_THREADS", "TF_XLA_FLAGS", "XLA_FLAGS",
                                               "NVIDIA_TF32_OVERRIDE", "CONDA_DEFAULT_ENV")}})
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        _write_json(output / "memory_policy.json", memory)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        tf.config.set_soft_device_placement(False)
        logical = tf.config.list_logical_devices("GPU")
        if len(logical) != 1 or len(tf.config.list_physical_devices("GPU")) != 1:
            raise ParallelDiagnosticError("worker must see exactly one physical and logical GPU")
        if job["mode"] == "gpu_smoke":
            smoke = _gpu_smoke(tf, logical[0].name, job)
            _write_json(output / "smoke_receipt.json", smoke)
            allocator = tf.config.experimental.get_memory_info(logical[0].name)
            if int(allocator["peak"]) > WORKER_PEAK_MIB * 1024**2:
                raise ParallelDiagnosticError("GPU smoke exceeded the worker allocator screen")
            headroom = check_runtime_headroom(selection, int(allocator["peak"]))
            if _source_hashes() != job["source_hashes"] or _sha256(PLAN) != job["plan_hash"]:
                raise ParallelDiagnosticError("source or plan changed during GPU smoke")
            _write_json(output / "run_manifest.json", {
                "status": "PASS_PHASE0_PARALLEL_GPU_SMOKE", "job": job,
                "pid": os.getpid(), "gpu_uuid": uuid, "placement": placement,
                "memory_policy": memory, "allocator": allocator, "headroom": headroom,
                "tensorflow": tf.__version__, "jit_compile": True,
                "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
                "wall_time_seconds": time.monotonic() - started,
                "artifact_hashes": {str(path.relative_to(output)): _sha256(path)
                                    for path in sorted(output.rglob("*.json"))},
                "claim_boundary": CLAIM_BOUNDARY, "p1_closeout_issued": False,
            })
            return 0
        from bayesfilter.inference.tempered_target_tf import make_q20_tempered_bridge
        from bayesfilter.inference.tuning_contract import hmc_tuning_interface_capability
        import tensorflow_probability as tfp

        capability = hmc_tuning_interface_capability("tune_fixed_transport_hmc_kernel")
        if capability.capability_status != "tested_supported" or not capability.artifact_authority:
            raise ParallelDiagnosticError("frozen-transport public tuner is not artifact-authoritative")
        profile = _profile(source, Path(job["campaign_output"]), job["arm"], job["worker_seconds"])
        with tf.device(logical[0].name):
            bridge = make_q20_tempered_bridge(20, jit_compile=True,
                                            principal_sqrt_backend=profile.principal_sqrt_backend)
            if bridge.target_signature != job["target_signature"]:
                raise ParallelDiagnosticError("worker target signature changed")
            chart_started = time.monotonic()
            charts, _, _ = source._build_fresh_chart(
                tf, bridge, CHART_INDEX, source.COMPONENT_IDS[CHART_INDEX], profile, output
            )
            chart_seconds = time.monotonic() - chart_started
            _write_json(output / "tuning_start.json", {"started_at_unix": time.time(),
                                                       "chart_seconds": chart_seconds})
            tune_started = time.monotonic()
            row = source._tune_scope(tf, bridge, charts[BETA], chart_index=CHART_INDEX,
                                     beta=BETA, output_dir=output / "tuning", profile=profile,
                                     scope_index=SCOPE_INDEX)
            tuning_seconds = time.monotonic() - tune_started
        allocator = tf.config.experimental.get_memory_info(logical[0].name)
        if int(allocator["peak"]) > WORKER_PEAK_MIB * 1024**2:
            raise ParallelDiagnosticError("worker exceeded inherited 4 GiB allocator screen")
        headroom = check_runtime_headroom(selection, int(allocator["peak"]))
        if _source_hashes() != job["source_hashes"] or _sha256(PLAN) != job["plan_hash"]:
            raise ParallelDiagnosticError("source or plan changed during tuning")
        public_row = {key: value for key, value in row.items() if not key.startswith("_")}
        _write_json(output / "tuning_receipt.json", source._json_ready(public_row))
        artifacts = {str(path.relative_to(output)): _sha256(path)
                     for path in sorted(output.rglob("*.json"))}
        _write_json(output / "run_manifest.json", {
            "status": "PASS_PHASE0_PARALLEL_TUNING_WORKER", "job": job,
            "pid": os.getpid(), "gpu_uuid": uuid, "placement": placement,
            "memory_policy": memory, "allocator": allocator, "headroom": headroom,
            "tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__,
            "jit_compile": True,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "training_batch_size": source.BATCH_SIZE, "samplewise_scalar_fallback": False,
            "training_target_backend": "batch_native_q20_tempered_bridge",
            "chart_seconds": chart_seconds, "tuning_seconds": tuning_seconds,
            "wall_time_seconds": time.monotonic() - started, "artifact_hashes": artifacts,
            "claim_boundary": CLAIM_BOUNDARY, "p1_closeout_issued": False,
        })
        return 0
    except BaseException as exc:
        _write_json(output / "failure.json", {"status": "FAIL_PARALLEL_TUNING_WORKER",
                                              "error": f"{type(exc).__name__}: {exc}",
                                              "wall_time_seconds": time.monotonic() - started})
        raise


def _validate_result(task: Any, payload: Mapping[str, Any], job: Mapping[str, Any]) -> None:
    smoke_only = job.get("mode") == "gpu_smoke"
    expected_status = "PASS_PHASE0_PARALLEL_GPU_SMOKE" if smoke_only else "PASS_PHASE0_PARALLEL_TUNING_WORKER"
    if payload.get("status") != expected_status or payload.get("job") != job:
        raise ParallelDiagnosticError("worker result is incomplete or has wrong job lineage")
    if payload.get("gpu_uuid") != task.gpu_uuid or payload.get("claim_boundary") != CLAIM_BOUNDARY:
        raise ParallelDiagnosticError("worker device/claim boundary mismatch")
    memory = payload.get("memory_policy", {})
    if (memory.get("all_physical_devices_memory_growth") is not True
            or memory.get("configured_before_logical_device_initialization") is not True
            or payload.get("jit_compile") is not True or payload.get("tf32_enabled") is not True
            or payload.get("headroom", {}).get("status") != "pass"):
        raise ParallelDiagnosticError("worker runtime policy receipt is incomplete")
    hashes = payload.get("artifact_hashes", {})
    required = {"memory_policy.json", "smoke_receipt.json"} if smoke_only else {
        "memory_policy.json", "tuning_receipt.json", "tuning/fixed_transport_hmc_tuning_result.json"
    }
    if not required <= hashes.keys():
        raise ParallelDiagnosticError("worker required tuning evidence is missing")
    for name, digest in hashes.items():
        path = (task.output_dir / name).resolve()
        if task.output_dir.resolve() not in path.parents or _sha256(path) != digest:
            raise ParallelDiagnosticError("worker artifact checksum or location mismatch")
    if json.loads((task.output_dir / "memory_policy.json").read_text()) != memory:
        raise ParallelDiagnosticError("startup memory verification differs from worker result")
    if smoke_only:
        smoke = json.loads((task.output_dir / "smoke_receipt.json").read_text())
        errors = smoke.get("max_absolute_errors", ())
        if (smoke.get("status") != "PASS_GPU_XLA_STARTUP_SMOKE" or smoke.get("tracing_count") != 1
                or not smoke.get("hlo_sha256") or len(smoke.get("call_seconds", ())) != 2
                or len(errors) != 2 or any(not math.isfinite(error) or error > 1e-10 for error in errors)):
            raise ParallelDiagnosticError("GPU smoke evidence is incomplete")
        return
    receipt = json.loads((task.output_dir / "tuning_receipt.json").read_text())
    tuning = receipt["tuning_result"]
    serialized_tuning = json.loads((task.output_dir / "tuning/fixed_transport_hmc_tuning_result.json").read_text())
    if tuning != serialized_tuning:
        raise ParallelDiagnosticError("worker tuning receipt differs from the public tuning artifact")
    selection = tuning["candidate_selection"]
    profile = job["profile"]
    count = len(profile["step_size_candidates"]) * len(profile["leapfrog_grid"])
    declared_pairs = {(step, leapfrog) for step in profile["step_size_candidates"]
                      for leapfrog in profile["leapfrog_grid"]}
    measured_pairs = {(row["selected_step_size"], row["num_leapfrog_steps"])
                      for row in tuning.get("candidates", ())}
    if (selection.get("all_candidate_pairs_measured") is not True
            or selection.get("final_status") != "passed"
            or selection.get("seed_ledger", {}).get("all_seeds_unique") is not True
            or selection.get("heldout_verification", {}).get("final_status") != "passed"
            or len(tuning.get("candidates", ())) != count
            or measured_pairs != declared_pairs
            or tuning.get("tuning_policy") != "measured_joint_grid_v1"
            or receipt.get("scope_index") != SCOPE_INDEX
            or not receipt.get("handoff_hash")):
        raise ParallelDiagnosticError("worker has incomplete grid, seed or heldout evidence")


def _headroom_monitor(tasks: tuple[Any, ...], output: Path, wave_index: int):
    from bayesfilter.runtime.display_gpu_policy import HEADROOM_MIB

    next_check = time.monotonic() + MONITOR_INTERVAL_SECONDS
    check_index = 0

    def check() -> None:
        nonlocal next_check, check_index
        if time.monotonic() < next_check:
            return
        command = ["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"]
        text = subprocess.check_output(command, text=True, timeout=MONITOR_TIMEOUT_SECONDS)
        free_by_uuid = {uuid.strip(): float(free) for uuid, free in csv.reader(text.splitlines())}
        _write_json(output / f"wave-{wave_index}-headroom-{check_index}.json", {
            "captured_at_unix": time.time(), "command": command, "free_mib_by_uuid": free_by_uuid,
        })
        for task in tasks:
            free = free_by_uuid.get(task.gpu_uuid, -1.0)
            if not math.isfinite(free) or free < HEADROOM_MIB:
                raise ParallelDiagnosticError(f"5 GiB runtime headroom lost on {task.gpu_uuid}")
        check_index += 1
        next_check = time.monotonic() + MONITOR_INTERVAL_SECONDS

    return check


def _check_budget(ledger: Any, jobs: list[Mapping[str, Any]], worker_seconds: float) -> None:
    payload = ledger.read()
    required = len(jobs) * worker_seconds
    if ledger.remaining_seconds() < required:
        raise ParallelDiagnosticError(f"complete worker allocations require {required:g} worker-seconds; "
                                      f"ledger has {ledger.remaining_seconds():g}")
    if payload["reserved_seconds"] != 0:
        raise ParallelDiagnosticError("ledger already has active work; coordinator must be sole writer")
    if (payload["claim_boundary"] != CLAIM_BOUNDARY
            or payload["source_hash"] != _json_hash(jobs[0]["source_hashes"])
            or payload["plan_hash"] != jobs[0]["plan_hash"]):
        raise ParallelDiagnosticError("ledger scope/source/plan mismatch; no automatic migration or budget reset")
    seed_rows = []
    for job in jobs:
        namespace = job["profile"]["seed_namespace"]
        for name, values in namespace.items():
            if name.endswith("_roots"):
                for first, second in values:
                    offsets = (0, 100, 150, 200) if name == "tuning_roots" else (0,)
                    seed_rows.extend((first, second + offset) for offset in offsets)
            else:
                seed_rows.append(tuple(values))
    if len(seed_rows) != len(set(seed_rows)):
        raise ParallelDiagnosticError("worker seed namespaces overlap")


def _coordinate(output: Path, ledger: Any, worker_seconds: float, grace: float, max_workers: int, *, smoke_only: bool = False) -> Mapping[str, Any]:
    from bayesfilter.runtime.display_gpu_policy import probe_inventory, select_gpus
    from bayesfilter.runtime.parallel_tuning import ParallelTuningTask, run_parallel_tuning_wave

    source = _load_source()
    sources = _source_hashes()
    jobs = [_job(source, output, arm, worker_seconds, sources, smoke_only=smoke_only) for arm in ARMS]
    _check_budget(ledger, jobs, worker_seconds)
    if output.exists():
        raise ParallelDiagnosticError("refusing to reuse campaign output")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write_json(output / "run_start.json", {
        "status": "RUNNING_PHASE0_PARALLEL_TUNING", "command": sys.argv, "jobs": jobs,
        "started_at_unix": time.time(), "python": sys.executable, "platform": platform.platform(),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True),
        "plan_file": str(PLAN), "result_file": str(RESULT), "budget_before": ledger.read(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)), "gpu_execution_requires_trusted_permissions": True,
    })
    attempt_id = output.name
    binding = {"source_hash": _json_hash(jobs[0]["source_hashes"]), "plan_hash": jobs[0]["plan_hash"]}
    ledger.start_attempt(attempt_id=attempt_id, output_root=output,
                         seed_namespace={job["arm"]: job["profile"] for job in jobs}, **binding)
    reserved = []
    settled = set()
    waves = []
    measurements = {}
    inflight = ()
    wave_started = None
    error = None
    try:
        for job in jobs:
            ledger.reserve_chunk(attempt_id=attempt_id, arm=job["arm"], chunk_index=0,
                                 reserve_seconds=worker_seconds, output_root=output,
                                 seed_namespace=job["profile"], **binding)
            reserved.append(job["arm"])
            _write_json(output / f"{job['arm']}-job.json", job)
        pending = list(jobs)
        while pending:
            if _source_hashes() != jobs[0]["source_hashes"] or _sha256(PLAN) != binding["plan_hash"]:
                raise ParallelDiagnosticError("source or plan changed before launch")
            selection = select_gpus(probe_inventory(), min(max_workers, len(pending)),
                                    estimated_peak_mib=WORKER_PEAK_MIB)
            devices = selection["selected"]
            if not devices:
                raise ParallelDiagnosticError("no eligible GPU; no worker launched")
            batch = pending[:len(devices)]
            tasks = tuple(ParallelTuningTask(
                job["arm"], SCOPE_INDEX, device["uuid"], Path(job["output_dir"]),
                (sys.executable, str(SCRIPT), "--worker-job", str(output / f"{job['arm']}-job.json")),
            ) for job, device in zip(batch, devices))
            _write_json(output / f"wave-{len(waves)}-placement.json", selection)
            environment = dict(os.environ)
            threads = max(1, len(os.sched_getaffinity(0)) // len(tasks))
            environment.update({"TF_NUM_INTRAOP_THREADS": str(threads), "TF_NUM_INTEROP_THREADS": "1",
                                "OMP_NUM_THREADS": str(threads)})
            expected_jobs = {job["arm"]: job for job in batch}
            inflight = tasks
            wave_started = time.monotonic()
            wave = run_parallel_tuning_wave(
                tasks, timeout_seconds=worker_seconds, terminate_grace_seconds=grace,
                base_environment=environment, cwd=ROOT,
                artifact_validator=lambda task, payload: _validate_result(task, payload, expected_jobs[task.task_id]),
                monitor_callback=_headroom_monitor(tasks, output, len(waves)),
            )
            measurements.update({row["task"]["task_id"]: row for row in wave["results"]})
            inflight = ()
            waves.append(wave)
            _write_json(output / f"wave-{len(waves) - 1}-result.json", wave)
            for row in wave["results"]:
                arm = row["task"]["task_id"]
                ledger.settle_arm(attempt_id=attempt_id, arm=arm, measured_seconds=row["elapsed_seconds"],
                                  status=row["status"], failure_class=None if row["status"] == "completed" else "worker_or_evidence_failure")
                settled.add(arm)
            if wave["status"] != "completed":
                raise ParallelDiagnosticError(f"incomplete tuning wave: {wave.get('error')}")
            pending = pending[len(batch):]
        if _source_hashes() != sources or _sha256(PLAN) != binding["plan_hash"]:
            raise ParallelDiagnosticError("source or plan changed before reconciliation")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        for arm in reserved:
            if arm not in settled:
                row = measurements.get(arm)
                uncertain = any(task.task_id == arm for task in inflight)
                elapsed = row["elapsed_seconds"] if row is not None else (
                    time.monotonic() - wave_started if uncertain else 0.0
                )
                status = row["status"] if row is not None else (
                    "supervisor_failure_conservative_wall_estimate" if uncertain else "not_launched"
                )
                ledger.settle_arm(attempt_id=attempt_id, arm=arm, measured_seconds=elapsed, status=status)
                settled.add(arm)
        ledger.finish_attempt(attempt_id=attempt_id, status="completed" if error is None else "failed")
    status_suffix = "GPU_SMOKE" if smoke_only else "TUNING_DIAGNOSTIC"
    result = {
        "status": f"{'PASS' if error is None else 'FAIL'}_PHASE0_PARALLEL_{status_suffix}",
        "error": error, "waves": waves, "wall_time_seconds": time.monotonic() - started,
        "worker_seconds": sum(wave["worker_seconds"] for wave in waves), "budget_after": ledger.read(),
        "claim_boundary": CLAIM_BOUNDARY, "p1_closeout_issued": False,
        "maximum_concurrent_workers": max((wave["task_count"] for wave in waves), default=0),
        "candidate_level_parallelism": False,
        "mode": "gpu_smoke" if smoke_only else "tuning",
    }
    _write_json(output / "run_manifest.json", result)
    return result


def _interrupt(signum: int, _frame: Any) -> None:
    raise InterruptedError(f"parallel diagnostic interrupted by signal {signum}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-job", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--worker-seconds", type=float)
    parser.add_argument("--terminate-grace-seconds", type=float, default=30.0)
    parser.add_argument("--max-workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    try:
        _require_framework_free()
        os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
        signal.signal(signal.SIGTERM, _interrupt)
        signal.signal(signal.SIGINT, _interrupt)
        if args.worker_job is not None:
            return _worker(args.worker_job.resolve())
        if args.plan_only:
            print(json.dumps({"status": "PLAN_ONLY_NO_GPU_WORK", "arms": ARMS,
                              "source_hash": _json_hash(_source_hashes()), "plan_hash": _sha256(PLAN),
                              "claim_boundary": CLAIM_BOUNDARY, "max_workers": args.max_workers,
                              "required_allocation": "2 * worker-seconds, summed over processes",
                              "existing_funded_ledger_required": True, "p1_closeout_issued": False}, indent=2))
            return 0
        seconds = args.worker_seconds
        grace = args.terminate_grace_seconds
        if (args.ledger is None or args.output_dir is None or seconds is None
                or not math.isfinite(seconds) or not math.isfinite(grace) or not 0 <= grace < seconds):
            raise ParallelDiagnosticError("runtime needs an existing funded --ledger, fresh --output-dir, "
                                          "and finite --worker-seconds greater than termination grace")
        from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger

        result = _coordinate(args.output_dir.resolve(), CampaignBudgetLedger(args.ledger),
                             seconds, grace, args.max_workers, smoke_only=args.smoke_only)
        print(json.dumps({"status": result["status"], "error": result["error"]}))
        return 0 if result["error"] is None else 2
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED_OR_FAILED_PARALLEL_TUNING",
                          "error": f"{type(exc).__name__}: {exc}"}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
