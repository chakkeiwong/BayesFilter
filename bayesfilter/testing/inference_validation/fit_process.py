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
        from tensorflow.python.eager import context
        try:
            result["registered_tf_functions"] = len(context.context().list_function_names())
        except (AttributeError, RuntimeError) as exc:
            result["function_count_unavailable"] = str(exc)
    return result


def fit_worker(design_file, root, replication, budget, attempt):
    """CLI child; configure device before importing the numerical pipeline."""
    from .execution import configure_worker, source_state

    started = time.monotonic()
    root = Path(root)
    path = root / f"replication-{replication:04d}"
    path.mkdir(parents=True, exist_ok=True)
    prefix = path / f"process-attempt-{attempt:03d}"
    design = ValidationDesign.from_payload(read_json(design_file))
    manifest = {"command": [sys.executable, *sys.argv], "environment": sys.executable,
                "design_identity": design.identity, "replication": replication,
                "seed": design.seed, "data": design.options.get("data"),
                "source": source_state(), "budget_seconds": budget,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "plan_file": design.options.get("plan_file", design.numerical_provenance),
                "result_file": str(path / "independent_assessment.json")}
    try:
        manifest["runtime"] = configure_worker(design)
        write_json(str(prefix) + "-manifest.json", manifest)
        from .engines.pipeline import run_replication
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


def _record(path, replication, receipt):
    assessment = path / "independent_assessment.json"
    row = (read_json(assessment) if assessment.is_file() else {
        "replication": replication, "inventory": {"failures": []}, "members": [],
        "tuning_completion": "execution_failed", "pipeline": str(path / "pipeline.json")})
    if receipt["status"] != "complete":
        row["execution_failure"] = receipt
    row["process_execution"] = receipt
    return row


def run_isolated_replications(design, root, *, deadline):
    """Sequential complete-fit children; native checkpoints support local retry.

    A failed process with a final assessment is preserved and never rerun.
    Partial fits may resume only under the same source/design and unused cap.
    """
    if "tensorflow" in sys.modules:
        raise RuntimeError("isolated-fit coordinator must not initialize TensorFlow")
    from .execution import source_state
    from .engines.pipeline import summarize_replications

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
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
            records.append(_record(path, replication, last))
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
                records.append(_record(path, replication, prior[-1]))
            if time.monotonic() >= deadline:
                break
            continue
        attempt = len(attempts) + 1
        prefix = path / f"process-attempt-{attempt:03d}"
        command = [sys.executable, "-m", "bayesfilter.testing.inference_validation",
                   "_pipeline_fit", str(design_file.resolve()), str(root.resolve()),
                   str(replication), str(remaining), str(attempt)]
        write_json(str(prefix) + "-launch.json", {"command": command, "budget_seconds": remaining})
        receipt = _supervise(command, str(prefix) + ".log", remaining, design.device)
        if assessment.exists():
            receipt["assessment_sha256"] = file_hash(assessment)
        elif receipt["status"] == "complete":
            receipt["status"] = "missing_assessment"
        write_json(str(prefix) + "-exit.json", receipt)
        records.append(_record(path, replication, receipt))
        # Infrastructure failure needs diagnosis before another expensive fit.
        if receipt["status"] != "complete":
            break
    result = summarize_replications(design, records)
    result["execution_mode"] = "one_process_per_complete_fit"
    result["framework_initialized_in_coordinator"] = "tensorflow" in sys.modules
    write_json(root / "assessment.json", result)
    return result
