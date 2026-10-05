"""Optional complete-fit process isolation for diagnostic validation campaigns.

The parent is framework-free. Each child owns one complete fit and exits
normally, releasing framework caches between fits. Process failure and saved
numerical evidence are recorded separately; neither is silently replaced.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import gc
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from .designs import ValidationDesign
from .storage import read_json, write_json, file_hash
from .timeout_policy import (
    TimeoutPolicy,
    FitAllowance,
    wait_for_gpu_admission,
)


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


def fit_worker(design_file, root, replication, budget, attempt, *, profile_execution=False,
               reuse_leapfrog_graphs=False):
    """CLI child; configure device before importing the numerical pipeline."""
    from .execution import configure_worker, source_state
    from .profiling import HostProfile

    started = time.monotonic()
    root = Path(root)
    path = root / f"replication-{replication:04d}"
    path.mkdir(parents=True, exist_ok=True)
    prefix = path / f"process-attempt-{attempt:03d}"
    design = ValidationDesign.from_payload(read_json(design_file))
    if type(reuse_leapfrog_graphs) is not bool:
        raise TypeError("reuse_leapfrog_graphs must be boolean")
    requested = profile_execution or design.options.get("profile_execution", False)
    profile = HostProfile(prefix, requested=requested, scope="isolated_numerical_child")
    manifest = {"command": [sys.executable, *sys.argv], "environment": sys.executable,
                "design_identity": design.identity, "replication": replication,
                "seed": design.seed, "data": design.options.get("data"),
                "source": source_state(), "budget_seconds": budget,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "plan_file": design.options.get("plan_file", design.numerical_provenance),
                "result_file": str(path / "independent_assessment.json"),
                "reuse_leapfrog_graphs": reuse_leapfrog_graphs or design.engine == "reference_mean",
                "profiling": {"requested": requested, "scope": "isolated_numerical_child",
                              "status_file": str(prefix) + "-profile.json",
                              "start_boundary": "after_framework_and_pipeline_imports"}}
    try:
        manifest["runtime"] = configure_worker(design)
        run_replication = _engine(design).run_replication
        execution_options = ({"reuse_leapfrog_graphs": True}
                             if reuse_leapfrog_graphs and design.engine != "reference_mean" else {})
        manifest["setup_seconds_before_profile"] = time.monotonic() - started
        write_json(str(prefix) + "-manifest.json", manifest)
        profile.start()
        before = resource_snapshot()
        from bayesfilter.runtime.execution_budget import execution_budget
        allowance = FitAllowance(str(prefix) + "-allowance.json")
        if allowance.path.exists():
            with execution_budget(check=allowance.available):
                run_replication(design, root, replication, None, **execution_options)
        else:
            with execution_budget(deadline=started + budget):
                run_replication(design, root, replication, started + budget, **execution_options)
        write_json(str(prefix) + "-resources.json", {
            "before": before, "after": resource_snapshot(),
            "elapsed_before_shutdown_seconds": time.monotonic() - started,
            "shutdown_success_established": False})
        return 0
    except Exception as exc:
        from bayesfilter.runtime.execution_budget import ExecutionBudgetExceeded
        from bayesfilter.inference.hmc_preparation import HMCPreparationBudgetExceeded
        if isinstance(exc, (ExecutionBudgetExceeded, HMCPreparationBudgetExceeded)):
            write_json(str(prefix) + "-budget.json", {"status": "budget_exhausted",
                "exception": type(exc).__name__, "reason": str(exc),
                "elapsed_seconds": time.monotonic() - started})
            return 75
        write_json(str(prefix) + "-failure.json", {
            "exception": type(exc).__name__, "reason": str(exc),
            "traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - started})
        traceback.print_exc()
        return 1
    finally:
        profile.finish()


def _supervise(command, log, seconds, device, **options):
    from .fit_supervision import supervise_fit
    return supervise_fit(command, log, seconds, device, **options)


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


def archive_interrupted_preparation(path, attempt):
    """Preserve a nonresumable progress-only preparation before a paid retry.

    Full tuning checkpoints are resumed normally. Unexpected files or a final
    preparation failure are left to the public tuner's fail-closed validation.
    """
    directory=Path(path)/"tuning"
    if not directory.is_dir() or {p.name for p in directory.iterdir()} != {"preparation_progress.json"}:
        return None
    progress=read_json(directory/"preparation_progress.json")
    interrupted=(progress.get("status")=="running" or (
        progress.get("status") in {"failed","deferred"}
        and (progress.get("failure") or {}).get("type")=="HMCPreparationBudgetExceeded"))
    if not interrupted or progress.get("artifact_authority") is not False:
        return None
    archived=Path(path)/f"interrupted-preparation-before-attempt-{attempt:03d}"
    if archived.exists():
        raise ValueError("preparation retry archive already exists")
    directory.rename(archived)
    return str(archived)


def contention_retry_budget(prior, base_seconds, policy):
    """Unused original fit cap, only for bounded incomplete resource recovery.

    The caller separately checks absence of a final assessment. Time consumed
    by every attempt counts, even if it produced no committed numerical chunk.
    """
    if not prior or len(prior) > policy.max_contention_retries:
        return None
    last = prior[-1]
    observed = last.get("observed_contention_seconds", 0.)
    if (last.get("status") not in {"timed_out", "budget_exhausted"}
            or type(observed) not in (int, float) or not math.isfinite(observed) or observed <= 0):
        return None
    spent = [row["elapsed_seconds"] for row in prior]
    if any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in spent):
        raise ValueError("invalid cumulative fit receipt")
    remaining = base_seconds + policy.max_extension_seconds - sum(spent)
    return remaining if remaining > policy.shutdown_grace_seconds else None


def run_isolated_replications(design, root, *, deadline, profile_execution=False,
                              reuse_leapfrog_graphs=False):
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
    if type(reuse_leapfrog_graphs) is not bool:
        raise TypeError("reuse_leapfrog_graphs must be boolean")
    if reuse_leapfrog_graphs:
        identity["reuse_leapfrog_graphs"] = True
    binding = root / "isolation_identity.json"
    if binding.exists() and read_json(binding) != identity:
        raise ValueError("isolated resume requires identical design and source")
    write_json(binding, identity)
    design_file = root / "isolated_design.json"
    write_json(design_file, design.payload())
    records = []
    timeout_policy = TimeoutPolicy.from_options(design.options)
    pending = list(range(design.replications))
    while pending:
        replication = pending.pop(0)
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
            if last["status"] not in {"complete", "timed_out", "budget_exhausted"}:
                break
            continue
        if assessment.exists():
            raise ValueError("isolated assessment has no process exit receipt")
        # An interrupted parent must not silently reallocate a child's budget.
        launches = list(path.glob("process-attempt-*-launch.json"))
        if len(launches) != len(attempts):
            raise ValueError("unreconciled child launch; inspect process before resuming")
        if (prior and any(p.get("contention_recovery") for p in prior)
                and len(prior) > timeout_policy.max_contention_retries):
            records.append(_record(path, replication, prior[-1], design))
            continue
        retry_remaining = contention_retry_budget(
            prior, design.options["fit_process_timeout_seconds"], timeout_policy)
        if (retry_remaining is None and prior and prior[-1]["status"] in {"timed_out", "budget_exhausted"}
                and prior[-1].get("allocation_limiter") != "cell"):
            # A cooperative stop reserves serialization time, not a new slot
            # for framework startup on the next invocation.
            records.append(_record(path, replication, prior[-1], design))
            continue
        # The declared fit timeout is the cumulative base allocation.  Any
        # contention grace is consumed inside _supervise and is never folded
        # into a fresh retry, so a retry cannot reset or double-count it.
        fit_remaining = (design.options["fit_process_timeout_seconds"]
                         - sum(p["elapsed_seconds"] for p in prior))
        attempt_policy = timeout_policy
        if retry_remaining is not None:
            # Recovery consumes the remaining *original* hard allowance.
            # It cannot earn the same contention extension for a second time.
            fit_remaining = retry_remaining
            attempt_policy = replace(timeout_policy, max_extension_seconds=0.)
        cell_remaining = deadline - time.monotonic()
        remaining = min(cell_remaining, fit_remaining)
        if remaining <= 0:
            if prior:
                records.append(_record(path, replication, prior[-1], design))
            if time.monotonic() >= deadline:
                break
            continue
        if design.device == "gpu" and timeout_policy.gpu_admission_wait_seconds:
            admission = wait_for_gpu_admission(device=design.device, policy=timeout_policy, deadline=deadline)
            admission_root = path / "admission"
            write_json(admission_root / f"check-{len(list(admission_root.glob('check-*.json')))+1:03d}.json", admission)
            if not admission["admitted"]:
                # Defer later slots too; do not create a sequence of competing
                # numerical workers during the same busy interval.
                break
            cell_remaining = deadline-time.monotonic()
            remaining = min(cell_remaining, fit_remaining)
            if remaining <= 0:
                break
        attempt = len(attempts) + 1
        archived_preparation=archive_interrupted_preparation(path,attempt) if prior else None
        prefix = path / f"process-attempt-{attempt:03d}"
        command = [sys.executable, "-m", "bayesfilter.testing.inference_validation",
                   "_pipeline_fit", str(design_file.resolve()), str(root.resolve()),
                   str(replication), str(remaining), str(attempt)]
        if requested:
            command.append("--profile-execution")
        if reuse_leapfrog_graphs:
            command.append("--reuse-leapfrog-graphs")
        write_json(str(prefix) + "-launch.json", {"command": command, "budget_seconds": remaining,
            "archived_unfinished_preparation":archived_preparation,
            "contention_recovery":retry_remaining is not None,
            "prior_attempt_seconds":sum(p["elapsed_seconds"] for p in prior),
            "original_fit_cap_seconds":design.options["fit_process_timeout_seconds"]
                + timeout_policy.max_extension_seconds})
        receipt = _supervise(
            command,
            str(prefix) + ".log",
            remaining,
            design.device,
            progress_root=path,
            cell_deadline=deadline,
            timeout_policy=attempt_policy,
            allowance_path=str(prefix) + "-allowance.json",
            budget_receipt_path=str(prefix) + "-budget.json",
        )
        receipt["allocation_limiter"] = "cell" if cell_remaining < fit_remaining else "fit"
        receipt["contention_recovery"] = retry_remaining is not None
        receipt["cumulative_fit_seconds"] = sum(p["elapsed_seconds"] for p in prior)+receipt["elapsed_seconds"]
        receipt["fit_base_budget_remaining_seconds"] = max(0.,
            design.options["fit_process_timeout_seconds"]-receipt["cumulative_fit_seconds"])
        receipt["fit_hard_budget_remaining_seconds"] = max(0.,
            design.options["fit_process_timeout_seconds"]+timeout_policy.max_extension_seconds
            -receipt["cumulative_fit_seconds"])
        if assessment.exists():
            receipt["assessment_sha256"] = file_hash(assessment)
        elif receipt["status"] == "complete":
            receipt["status"] = "missing_assessment"
        write_json(str(prefix) + "-exit.json", receipt)
        recovery_budget = contention_retry_budget(
            [*prior, receipt], design.options["fit_process_timeout_seconds"], timeout_policy)
        if (not assessment.exists() and recovery_budget is not None
                and deadline-time.monotonic() > timeout_policy.shutdown_grace_seconds):
            # Same process-independent source/design binding and native paths;
            # the next child reloads completed candidate/posterior chunks.
            pending.insert(0, replication)
            continue
        records.append(_record(path, replication, receipt, design))
        # A local budget timeout is an unavailable fit, not proof that the
        # source or harness is invalid. Continue independent slots while the
        # cell still has its declared aggregate budget. Other failures require
        # diagnosis before another expensive fit.
        if receipt["status"] not in {"complete", "timed_out", "budget_exhausted"}:
            break
    result = summarize_replications(design, records)
    result["execution_mode"] = "one_process_per_complete_fit"
    result["framework_initialized_in_coordinator"] = "tensorflow" in sys.modules
    if requested:
        result["profiling"] = profile_report(root, isolated=True, requested=True)
    write_json(root / "assessment.json", result)
    return result
