"""Optional complete-fit process isolation for diagnostic validation campaigns.

The parent is framework-free. Each child owns one complete fit and exits
normally, releasing framework caches between fits. Process failure and saved
numerical evidence are recorded separately; neither is silently replaced.
"""
from __future__ import annotations

from collections import Counter
import gc
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from .designs import ValidationDesign
from .storage import read_json, write_json, file_hash


def _engine(design):
    # Both assessment modules stay framework-free until a numerical child runs.
    if design.engine == "reference_mean":
        from .engines import reference_mean
        return reference_mean
    from .engines import pipeline
    return pipeline


def resource_snapshot():
    """Linux resident memory and live-object diagnostics, without collecting GC."""
    status = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:", "Threads:")):
            key, value = line.split(":", 1)
            status[key] = value.strip()
    counts = Counter()
    for obj in gc.get_objects():
        cls = type(obj)
        module = cls.__module__
        if isinstance(module, str) and module.startswith(("tensorflow", "bayesfilter")) and (
                "Graph" in cls.__name__ or "Function" in cls.__name__
                or "Runner" in cls.__name__):
            counts[module + "." + cls.__name__] += 1
    result = {"pid": os.getpid(), "proc_status": status,
              "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "live_objects": dict(sorted(counts.items()))}
    if "tensorflow" in sys.modules:
        import tensorflow as tf
        from tensorflow.python.eager import context
        try:
            result["registered_tf_functions"] = len(context.context().list_function_names())
        except (AttributeError, RuntimeError) as exc:
            result["function_count_unavailable"] = str(exc)
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
            result["gpu_allocator_bytes"] = {
                device.name: tf.config.experimental.get_memory_info(device.name)
                for device in tf.config.list_logical_devices("GPU")}
    return result


def fit_worker(design_file, root, replication, budget, attempt, *, profile_execution=False):
    """CLI child; configure device before importing the numerical pipeline."""
    from .execution import configure_worker, source_state
    from .profiling import HostProfile

    started = time.monotonic()
    root = Path(root)
    path = root / f"replication-{replication:04d}"
    path.mkdir(parents=True, exist_ok=True)
    prefix = path / f"process-attempt-{attempt:03d}"
    design = ValidationDesign.from_payload(read_json(design_file))
    requested = profile_execution or design.options.get("profile_execution", False)
    profile = HostProfile(prefix, requested=requested, scope="isolated_numerical_child")
    manifest = {"command": [sys.executable, *sys.argv], "environment": sys.executable,
                "design_identity": design.identity, "replication": replication,
                "seed": design.seed, "data": design.options.get("data"),
                "source": source_state(), "budget_seconds": budget,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "plan_file": design.options.get("plan_file", design.numerical_provenance),
                "result_file": str(path / "independent_assessment.json"),
                "profiling": {"requested": requested, "scope": "isolated_numerical_child",
                              "status_file": str(prefix) + "-profile.json",
                              "start_boundary": "after_framework_and_pipeline_imports"}}
    try:
        manifest["runtime"] = configure_worker(design)
        run_replication = _engine(design).run_replication
        manifest["setup_seconds_before_profile"] = time.monotonic() - started
        write_json(str(prefix) + "-manifest.json", manifest)
        profile.start()
        before = resource_snapshot()
        run_replication(design, root, replication, started + budget)
        write_json(str(prefix) + "-resources.json", {
            "before": before, "after": resource_snapshot(),
            "elapsed_before_shutdown_seconds": time.monotonic() - started,
            "shutdown_success_established": False})
        return 0
    except Exception as exc:
        write_json(str(prefix) + "-failure.json", {
            "exception": type(exc).__name__, "reason": str(exc),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        traceback.print_exc()
        return 1
    finally:
        profile.finish()


def _supervise(command, log, seconds, device):
    """Child shares the outer worker's group so campaign timeout kills both."""
    started = time.monotonic()
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
    if device == "cpu_reference":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    with Path(log).open("w") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=env)
        try:
            code = process.wait(timeout=seconds)
            status = "complete" if code == 0 else "failed"
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            code, status = process.returncode, "timed_out"
    return {"status": status, "exit_code": code, "pid": process.pid,
            "elapsed_seconds": time.monotonic() - started, "command": command,
            "log": str(log)}


def _record(path, replication, receipt, design):
    assessment = path / "independent_assessment.json"
    if assessment.is_file():
        row = read_json(assessment)
    elif design.engine == "reference_mean":
        row = _engine(design).unavailable_record(design, replication, "execution_failed")
    else:
        row = {"replication": replication, "inventory": {"failures": []}, "members": [],
               "tuning_completion": "execution_failed", "pipeline": str(path / "pipeline.json")}
    if receipt["status"] != "complete":
        row["execution_failure"] = receipt
    row["process_execution"] = receipt
    return row


def run_isolated_replications(design, root, *, deadline, profile_execution=False):
    """Sequential complete-fit children; native checkpoints support local retry.

    A failed process with a final assessment is preserved and never rerun.
    Partial fits may resume only under the same source/design and unused cap.
    """
    if "tensorflow" in sys.modules:
        raise RuntimeError("isolated-fit coordinator must not initialize TensorFlow")
    from .execution import source_state
    summarize_replications = _engine(design).summarize_replications
    from .profiling import profile_report

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    requested = profile_execution or design.options.get("profile_execution", False)
    identity = {"design_identity": design.identity, "source_identity": source_state()["identity"]}
    binding = root / "isolation_identity.json"
    if binding.exists() and read_json(binding) != identity:
        raise ValueError("isolated resume requires identical design and source")
    write_json(binding, identity)
    design_file = root / "isolated_design.json"
    write_json(design_file, design.payload())
    records = []
    for replication in range(design.replications):
        path = root / f"replication-{replication:04d}"
        path.mkdir(parents=True, exist_ok=True)
        attempts = sorted(path.glob("process-attempt-*-exit.json"))
        prior = [read_json(p) for p in attempts]
        assessment = path / "independent_assessment.json"
        if prior and assessment.exists():
            last = prior[-1]
            if last.get("assessment_sha256") != file_hash(assessment):
                raise ValueError("saved independent assessment changed after process exit")
            records.append(_record(path, replication, last, design))
            continue
        if assessment.exists():
            raise ValueError("isolated assessment has no process exit receipt")
        # An interrupted parent must not silently reallocate a child's budget.
        launches = list(path.glob("process-attempt-*-launch.json"))
        if len(launches) != len(attempts):
            raise ValueError("unreconciled child launch; inspect process before resuming")
        remaining = min(deadline - time.monotonic(),
                        design.options["fit_process_timeout_seconds"]
                        - sum(p["elapsed_seconds"] for p in prior))
        if remaining <= 0:
            if prior:
                records.append(_record(path, replication, prior[-1], design))
            if time.monotonic() >= deadline:
                break
            continue
        attempt = len(attempts) + 1
        prefix = path / f"process-attempt-{attempt:03d}"
        command = [sys.executable, "-m", "bayesfilter.testing.inference_validation",
                   "_pipeline_fit", str(design_file.resolve()), str(root.resolve()),
                   str(replication), str(remaining), str(attempt)]
        if requested:
            command.append("--profile-execution")
        write_json(str(prefix) + "-launch.json", {"command": command, "budget_seconds": remaining})
        receipt = _supervise(command, str(prefix) + ".log", remaining, design.device)
        if assessment.exists():
            receipt["assessment_sha256"] = file_hash(assessment)
        elif receipt["status"] == "complete":
            receipt["status"] = "missing_assessment"
        write_json(str(prefix) + "-exit.json", receipt)
        records.append(_record(path, replication, receipt, design))
        # Infrastructure failure needs diagnosis before another expensive fit.
        if receipt["status"] != "complete":
            break
    result = summarize_replications(design, records)
    result["execution_mode"] = "one_process_per_complete_fit"
    result["framework_initialized_in_coordinator"] = "tensorflow" in sys.modules
    if requested:
        result["profiling"] = profile_report(root, isolated=True, requested=True)
    write_json(root / "assessment.json", result)
    return result
