"""Campaign-funded continuation of an unchanged, resource-interrupted fit.

The framework-free coordinator can differ from the frozen numerical worker.
An explicit cumulative allocation can exceed the original fit cap; it never
changes the design, source, seeds, numerical criteria, or earlier charges.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import shutil
import sys
import time

from .designs import ValidationDesign, digest
from .fit_supervision import supervise_fit
from .storage import file_hash, read_json, write_json
from .timeout_policy import TimeoutPolicy

RESOURCE_STOPS = {"timed_out", "budget_exhausted"}


def checked_frozen_source(source):
    source = Path(source).resolve()
    snapshot = read_json(source / "source_snapshot.json")
    expected = snapshot.get("source", snapshot)
    files = {str(p.relative_to(source)): file_hash(p)
             for p in sorted((source / "bayesfilter").rglob("*.py"))}
    if files != expected["files"] or digest(files) != expected["identity"]:
        raise ValueError("frozen numerical source changed")
    dependencies = source.parent / "dependencies.json"
    if dependencies.exists():
        if any(file_hash(source / p) != sha for p, sha in read_json(dependencies).items()):
            raise ValueError("frozen numerical dependencies changed")
    return expected


def durable_progress(receipt):
    progress = receipt.get("progress", {})
    before, after = progress.get("initial", {}), progress.get("last", {})
    return any(after.get(key, 0) > before.get(key, 0) for key in
        ("evidence_count", "partial_chunks", "posterior_chunks", "preparation_completions"))


def _seconds(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(name + " must be finite and nonnegative")
    return value


def continuation_allowance(prior, *, cumulative_cap_seconds, quantum_seconds,
                           outer_remaining_seconds, max_additional_attempts,
                           checked_startup_repair=False):
    """Explicit allocation, with all prior attempts and restarts counted."""
    for value, name in ((cumulative_cap_seconds, "cumulative cap"),
                        (quantum_seconds, "quantum"), (outer_remaining_seconds, "outer remaining")):
        _seconds(value, name)
    if type(max_additional_attempts) is not int or max_additional_attempts < 1:
        raise ValueError("additional attempt cap must be a positive integer")
    eligible = prior[-1] if prior else {}
    if checked_startup_repair:
        eligible = prior[-2] if len(prior) >= 2 else {}
    if eligible.get("status") not in RESOURCE_STOPS or not durable_progress(eligible):
        return 0.
    if sum(bool(row.get("campaign_continuation")) for row in prior) >= max_additional_attempts:
        return 0.
    spent = sum(_seconds(row["elapsed_seconds"], "attempt elapsed") for row in prior)
    return max(0., min(quantum_seconds, outer_remaining_seconds, cumulative_cap_seconds - spent))


def immutable_fit_evidence(path):
    """Completed numerical records are immutable; controller checkpoints advance."""
    path = Path(path)
    files = {path / "fit_identity.json", path / "tuning/execution_spec.json"}
    for pattern in ("process-attempt-*-exit.json", "process-attempt-*-manifest.json",
                    "tuning/numerical_*/*.json", "members/*/member.json",
                    "members/*/result.json", "members/*/*.tensor*",
                    "members/*/posterior_chunks/committed/**/*"):
        files.update(p for p in path.glob(pattern) if p.is_file())
    return {str(p): file_hash(p) for p in sorted(files) if p.is_file()}


def _runtime_checked(path, prefix, design, identity):
    manifest = read_json(str(prefix) + "-manifest.json")
    if (manifest["design_identity"] != design.identity or
            manifest["source"]["identity"] != identity["source_identity"]):
        raise ValueError("continued child changed design or source")
    if design.device == "gpu":
        runtime = manifest["runtime"]
        memory = runtime["memory_policy"]
        if (runtime.get("jit_compile") is not True or "GPU" not in runtime.get("gpu_tensor_device", "")
                or not memory.get("configured_before_logical_device_initialization")
                or not memory.get("all_physical_devices_memory_growth")
                or not memory.get("physical_devices")):
            raise ValueError("continued fit lacks GPU/XLA/memory-growth provenance")
    return manifest


def check_checkpoint_device(cell, replication=0):
    """Check the saved device binding without initializing a framework."""
    spec = Path(cell) / f"replication-{replication:04d}/tuning/execution_spec.json"
    if not spec.exists():
        return
    runtime = read_json(spec)["execution"]["runtime_policy"]
    if (runtime.get("device_type") == "GPU" and
            (runtime["cuda_visible_devices"] != os.environ.get("CUDA_VISIBLE_DEVICES")
             or os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true")):
        raise ValueError("checkpoint requires original GPU selection and memory growth")


def checked_device_startup_repair(path, exits, prior):
    """Recognize only an unchanged checkpoint rejected by device reconstruction."""
    if len(prior) < 2 or prior[-1].get("status") != "failed":
        return None
    last = prior[-1]
    failure_path = Path(str(exits[-1]).removesuffix("-exit.json") + "-failure.json")
    if not failure_path.exists():
        return None
    failure = read_json(failure_path)
    if (failure.get("exception") != "ValueError"
            or failure.get("reason") != "execution device or numerical policy mismatch"):
        return None
    if not (path / "tuning/execution_spec.json").exists():
        raise ValueError("missing original execution device specification")
    if (prior[-2].get("status") not in RESOURCE_STOPS or not durable_progress(prior[-2])
            or durable_progress(last) or last.get("preserved_evidence_verified") is not True):
        raise ValueError("device startup failure is not an unchanged resource checkpoint")
    record = Path(last["allocation_file"]).parent
    before = record / "before"
    checkpoint = before / "tuning/tuning_checkpoint.json"
    if not checkpoint.exists():
        raise ValueError("missing before-startup checkpoint")
    for p in before.rglob("*"):
        if p.is_file() and file_hash(p) != file_hash(path / p.relative_to(before)):
            raise ValueError("checkpoint changed during device startup failure")
    preserved = read_json(record / "preserved-evidence.json")
    if any(file_hash(p) != sha for p, sha in preserved.items()):
        raise ValueError("evidence changed during device startup failure")
    check_checkpoint_device(path.parent, int(path.name.removeprefix("replication-")))
    return {"reason": "unchanged_checkpoint_after_device_binding_startup_failure",
            "failed_receipt": str(exits[-1]), "failure_record": str(failure_path),
            "failure_receipt_sha256": file_hash(exits[-1]),
            "checkpoint_sha256": file_hash(checkpoint), "prior_costs_retained": True}


def continue_frozen_fit(*, cell, source, replication, output, cumulative_cap_seconds,
                        quantum_seconds, deadline, max_additional_attempts,
                        repair_device_startup_failure=False):
    """Run at most one additional child; the caller owns the enclosing budget.

    ``output`` is a fresh attempt directory. Native checkpoint paths stay fixed.
    Completed assessments, including failures, are returned without execution.
    Only a resource stop with durable progress can consume additional time.
    """
    invocation_started = time.monotonic()
    cell, source, output = map(lambda p: Path(p).resolve(), (cell, source, output))
    frozen = checked_frozen_source(source)
    design = ValidationDesign.from_payload(read_json(cell / "isolated_design.json"))
    identity = read_json(cell / "isolation_identity.json")
    expected = {"design_identity": design.identity, "source_identity": frozen["identity"]}
    if identity.get("reuse_leapfrog_graphs") is True:
        expected["reuse_leapfrog_graphs"] = True
    if identity != expected:
        raise ValueError("continuation requires identical original design and source")
    path = cell / f"replication-{replication:04d}"
    fit = read_json(path / "fit_identity.json")
    if (fit["source_identity"] != frozen["identity"] or fit["design"] != design.identity
            or fit["fit_id"] != replication or fit["data"] != digest(design.options.get("data"))
            or fit["reuse_leapfrog_graphs"] != identity.get("reuse_leapfrog_graphs", False)):
        raise ValueError("fit identity changed")
    exits = sorted(path.glob("process-attempt-*-exit.json"))
    launches = sorted(path.glob("process-attempt-*-launch.json"))
    if len(launches) != len(exits) or any(
            a.name.replace("-launch.json", "-exit.json") != b.name for a, b in zip(launches, exits)):
        raise ValueError("unreconciled child launch; inspect process before continuing")
    prior = [read_json(p) for p in exits]
    assessment = path / "independent_assessment.json"
    if assessment.exists():
        if not prior or prior[-1].get("assessment_sha256") != file_hash(assessment):
            raise ValueError("completed assessment changed or has no exit receipt")
        return {"status": "reused_final_assessment", "new_elapsed_seconds": 0.,
                "receipt": str(exits[-1]), "original_status": prior[-1]["status"]}
    startup_repair = (checked_device_startup_repair(path, exits, prior)
                      if repair_device_startup_failure else None)
    check_checkpoint_device(cell, replication)
    seconds = continuation_allowance(prior, cumulative_cap_seconds=cumulative_cap_seconds,
        quantum_seconds=quantum_seconds, outer_remaining_seconds=max(0., deadline - time.monotonic()),
        max_additional_attempts=max_additional_attempts, checked_startup_repair=startup_repair is not None)
    if seconds <= 0:
        return {"status": "ineligible_or_allocation_exhausted", "new_elapsed_seconds": 0.}
    if not (path / "tuning/tuning_checkpoint.json").exists():
        return {"status": "no_resumable_numerical_checkpoint", "new_elapsed_seconds": 0.}
    attempt = len(prior) + 1
    prefix = path / f"process-attempt-{attempt:03d}"
    output.mkdir(parents=True, exist_ok=False)
    preserved = immutable_fit_evidence(path)
    write_json(output / "preserved-evidence.json", preserved)
    for name in ("tuning/tuning_checkpoint.json", "pipeline.json", "posterior_selection.json"):
        if (path / name).exists():
            saved = output / "before" / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path / name, saved)
    command = [sys.executable, "-m", "bayesfilter.testing.inference_validation", "_pipeline_fit",
        str(cell / "isolated_design.json"), str(cell), str(replication), str(seconds), str(attempt)]
    if identity.get("reuse_leapfrog_graphs"):
        command.append("--reuse-leapfrog-graphs")
    policy = TimeoutPolicy(gpu_admission_mode="shared", max_extension_seconds=0.,
                           extension_mode="observed_intervals")
    allocation = {"campaign_continuation": True, "command": command, "cwd": str(source),
        "source_identity": frozen["identity"], "design_identity": design.identity,
        "prior_attempt_seconds": sum(r["elapsed_seconds"] for r in prior),
        "cumulative_fit_cap_seconds": cumulative_cap_seconds, "allocation_seconds": seconds,
        "max_additional_attempts": max_additional_attempts, "allocation_file": str(output / "allocation.json"),
        "original_design_cap_unchanged": True}
    if startup_repair is not None:
        allocation["diagnosed_startup_repair"] = startup_repair
    write_json(output / "allocation.json", allocation)
    write_json(str(prefix) + "-launch.json", allocation)
    receipt = supervise_fit(command, str(prefix) + ".log", seconds, design.device,
        progress_root=path, cell_deadline=deadline, timeout_policy=policy,
        allowance_path=str(prefix) + "-allowance.json", budget_receipt_path=str(prefix) + "-budget.json",
        cwd=source)
    receipt.update(campaign_continuation=True, allocation_file=str(output / "allocation.json"),
        cumulative_fit_cap_seconds=cumulative_cap_seconds,
        cumulative_fit_seconds=allocation["prior_attempt_seconds"] + receipt["elapsed_seconds"],
        allocation_limiter="campaign_allocation")
    # Save the terminal process fact before any post-run validation can fail.
    write_json(str(prefix) + "-exit.json", receipt)
    try:
        if any(file_hash(p) != sha for p, sha in preserved.items()):
            raise ValueError("completed numerical evidence changed during continuation")
        _runtime_checked(path, prefix, design, identity)
        if assessment.exists():
            receipt["assessment_sha256"] = file_hash(assessment)
        elif receipt["status"] == "complete":
            raise ValueError("continued fit exited without final assessment")
        receipt["preserved_evidence_verified"] = True
    except Exception as exc:
        receipt.update(status="continuation_validation_failed", validation_error=str(exc))
        raise
    finally:
        receipt["continuation_invocation_seconds"] = time.monotonic() - invocation_started
        write_json(str(prefix) + "-exit.json", receipt)
        write_json(output / "result.json", {**receipt, "receipt": str(prefix) + "-exit.json"})
    return {**receipt, "receipt": str(prefix) + "-exit.json"}


def start_frozen_fit(*, cell, source, design, quantum_seconds, deadline):
    """Start a fresh complete fit using the same receipt format as continuation.

    The independent campaign allocator owns wall-time caps; numerical settings
    and the public tuner remain those in the explicit design and frozen source.
    """
    invocation_started = time.monotonic()
    cell, source = Path(cell).resolve(), Path(source).resolve()
    frozen = checked_frozen_source(source)
    _seconds(quantum_seconds, "quantum")
    seconds = min(quantum_seconds, deadline - time.monotonic())
    if seconds <= 0:
        raise ValueError("no time to start frozen fit")
    cell.mkdir(parents=True, exist_ok=False)
    identity = {"design_identity": design.identity, "source_identity": frozen["identity"],
                "reuse_leapfrog_graphs": True}
    write_json(cell / "isolated_design.json", design.payload())
    write_json(cell / "isolation_identity.json", identity)
    path = cell / "replication-0000"
    path.mkdir()
    prefix = path / "process-attempt-001"
    command = [sys.executable, "-m", "bayesfilter.testing.inference_validation", "_pipeline_fit",
        str(cell / "isolated_design.json"), str(cell), "0", str(seconds), "1", "--reuse-leapfrog-graphs"]
    write_json(str(prefix) + "-launch.json", {"command": command, "budget_seconds": seconds,
        "source_identity": frozen["identity"], "design_identity": design.identity,
        "campaign_initial_allocation": True})
    receipt = supervise_fit(command, str(prefix) + ".log", seconds, design.device,
        progress_root=path, cell_deadline=deadline,
        timeout_policy=TimeoutPolicy(gpu_admission_mode="shared"),
        allowance_path=str(prefix) + "-allowance.json", budget_receipt_path=str(prefix) + "-budget.json",
        cwd=source)
    write_json(str(prefix) + "-exit.json", receipt)
    try:
        _runtime_checked(path, prefix, design, identity)
        assessment = path / "independent_assessment.json"
        if assessment.exists():
            receipt["assessment_sha256"] = file_hash(assessment)
        elif receipt["status"] == "complete":
            raise ValueError("new fit exited without final assessment")
    except Exception as exc:
        receipt.update(status="initial_validation_failed", validation_error=str(exc))
        raise
    finally:
        receipt["invocation_seconds"] = time.monotonic() - invocation_started
        write_json(str(prefix) + "-exit.json", receipt)
    return {**receipt, "receipt": str(prefix) + "-exit.json"}
