#!/usr/bin/env python3
"""Resume unchanged public qualification from a copied full-model checkpoint.

The original frozen numerical source owns target construction, checkpoint
validation, tuning decisions and retained exports. This launcher adds no pilot,
numerical policy, posterior sampling or artifact authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_parent(parent):
    """Check recorded provenance before copying or importing a numerical module."""
    parent = Path(parent).resolve()
    manifest, result, supervisor = (read(parent/name) for name in
                                  ("manifest.json", "result.json", "supervisor.json"))
    if (supervisor["status"] not in {"resource_stopped", "supervisor_budget_exhausted"}
            or result["status"] != supervisor["status"]
            or manifest["status"] != supervisor["status"]):
        raise ValueError("a terminal resource-censored parent is required")
    if manifest["design"].get("posterior") is not None:
        raise ValueError("this command resumes qualification only")
    if manifest["design"].get("method") not in {"fixed_hmc", "randomized_hmc", "snaper", "mams"}:
        raise ValueError("unsupported or excluded method")
    if manifest.get("use_xla") is not True or manifest.get("cpu_reference") is not False:
        raise ValueError("the full-model continuation requires the original GPU/XLA scope")
    source = Path(manifest["command"][1]).resolve().parents[1]
    if read(source/".validation-source.json")["git_commit"] != manifest["git_commit"]:
        raise ValueError("original numerical source identity changed")
    for path, expected in manifest["source_and_input_sha256"].items():
        if digest(path) != expected:
            raise ValueError("original source/input changed: " + path)
    design = read(parent/"search/search-design.json")
    checkpoint = read(parent/"search/tuning/tuning_checkpoint.json")
    if (design["binding_hash"] != manifest["binding_hash"]
            or checkpoint["binding_hash"] != manifest["binding_hash"]):
        raise ValueError("parent search/checkpoint binding differs")
    if not design["requested_primary_pairs"]:
        raise ValueError("actual pilot proposal denominator is required")
    return source, manifest


def copy_search(parent, output):
    """Copy evidence, never alter the old fit or rerun its pilot."""
    parent, output = Path(parent).resolve(), Path(output).resolve()
    search = output/"search"
    if search.exists():
        raise FileExistsError(search)
    if output == parent or parent in output.parents:
        raise ValueError("continuation must be outside the original run")
    original = parent/"search"
    preserved = {str(p): digest(p) for p in original.rglob("*") if p.is_file()}
    search.mkdir(parents=True)
    shutil.copytree(original/"tuning", search/"tuning")
    shutil.copyfile(original/"search-design.json", search/"search-design.json")
    write(output/"parent-evidence.json", preserved)
    return search


def inherited_attempts(checkpoint):
    """Find charged calls without durable results; only these may be retried."""
    payload = read(checkpoint)
    events = payload["result"]["accounting_events"]
    directory = Path(checkpoint).parent
    returned = set()
    for key in payload["numerical_evidence_hashes"]:
        record = read(directory/"numerical_evidence"/(key+".json"))
        returned.update((record["work"]["work_item_id"], c["trial_ordinal"], c["trial_chunk_index"])
                        for c in record.get("chunks", ()))
    for work_id, keys in payload["partial_chunks"].items():
        for key in keys:
            chunk = read(directory/"numerical_chunks"/(key+".json"))
            returned.add((work_id, chunk["trial_ordinal"], chunk["trial_chunk_index"]))
    charged = {(e["work_item_id"], e["trial_ordinal"], e["trial_chunk_index"])
               for e in events if e["event"] == "numerical_chunk_charged"}
    return events, charged-returned


def resume_search(output, adapter):
    """Use the public resume/export APIs; numerical replay remains their job."""
    from bayesfilter.inference import resume_hmc_candidate_set_tuning
    from bayesfilter.inference.hmc_candidate_set_retained import (
        export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners)
    from bayesfilter.testing.inference_validation.modern_tuning_consumer import candidate_inventory
    from scripts.resume_modern_hmc_bridge import audit_recovery

    output = Path(output)
    checkpoint = output/"search/tuning/tuning_checkpoint.json"
    before = read(checkpoint)
    events, missing = inherited_attempts(checkpoint)
    run = resume_hmc_candidate_set_tuning(checkpoint, adapter=adapter)
    binding = run.adapter._execution_binding
    # Persisted tuples become lists; compare JSON values, not Python containers.
    after = json.loads(json.dumps(run.result.payload(), allow_nan=False))
    if (binding.binding_hash != before["binding_hash"]
            or after["config"] != before["result"]["config"]
            or after["candidates"][:len(before["result"]["candidates"])] != before["result"]["candidates"]
            or after["observations"][:len(before["result"]["observations"])] != before["result"]["observations"]):
        raise ValueError("continuation changed inherited identity, candidates or evidence")
    audit = audit_recovery(run.result, binding, inherited_events=events,
                          inherited_missing_keys=missing)
    paths = export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=binding, output_dir=output/"search/members")
    members = load_hmc_candidate_retained_runners(tuple(paths.values()), adapter=adapter)
    if set(members) != set(run.result.verified_candidate_ids):
        raise ValueError("retained export/reload lost a verified member")
    for path, expected in read(output/"parent-evidence.json").items():
        if digest(path) != expected:
            raise ValueError("parent evidence changed during continuation: " + path)
    design = read(output/"search/search-design.json")
    inventory = candidate_inventory(run.result, design["requested_primary_pairs"])
    summary = dict(**inventory, status="completed" if run.result.completion_status == "complete"
        else "resource_stopped" if run.result.completion_status == "partial_budget" else "repair_required",
        completion_status=run.result.completion_status,
        verified_candidate_ids=list(run.result.verified_candidate_ids),
        candidate_states=dict(run.result.candidate_states), work=audit,
        members={key: str(path) for key, path in paths.items()}, parent_evidence_unchanged=True,
        pilot_rerun=False, posterior_assessment="not_run", method_ranking=None, release_qualified=False)
    write(output/"result.json", summary)
    return run, members, summary


def restored_fixture(parent, *, use_xla):
    # Use the same factory/identity convention as prepared_binding. The bridge
    # mechanics helper adds a "sha256:" prefix to this metadata field.
    from bayesfilter.testing.inference_validation.modern_nonlinear import make_nonlinear_comparison_target
    fixture = make_nonlinear_comparison_target(parent["design"]["target"], jit_compile=use_xla)
    if fixture.metadata["data_sha256"] != parent["target"]["data_sha256"]:
        raise ValueError("restored target data differs")
    if fixture.metadata["horizon"] != parent["target"]["horizon"]:
        raise ValueError("restored target horizon differs")
    return fixture


def worker(args):
    source, parent = checked_parent(args.parent)
    sys.path.insert(0, str(source))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from bayesfilter.runtime.execution_budget import execution_budget
    fixture = restored_fixture(parent, use_xla=True)
    manifest = read(args.output/"manifest.json")
    manifest.update(status="restoring_and_qualifying", memory_policy=memory,
        tf_version=tf.__version__, use_xla=True,
        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    write(args.output/"manifest.json", manifest)
    started = time.monotonic()
    with execution_budget(check=lambda: time.monotonic()-started < args.max_wall_seconds,
                          replace_wall_time_limit=True):
        _, _, summary = resume_search(args.output, fixture.adapter)
    manifest.update(status=summary["status"], finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/"manifest.json", manifest)
    return 0 if summary["status"] == "completed" else 1


def supervise(args):
    source, parent = checked_parent(args.parent)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != parent["cuda_visible_devices"]:
        raise ValueError("continuation must retain the original device")
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.monotonic()
    copy_search(args.parent, args.output)
    manifest = dict(status="starting", plan=args.plan, parent_run=str(args.parent),
        numerical_source=str(source), git_commit=parent["git_commit"], command=[sys.executable,*sys.argv],
        launcher_sha256=digest(__file__), parent_manifest_sha256=digest(args.parent/"manifest.json"),
        started_utc=datetime.now(timezone.utc).isoformat(), max_wall_seconds=args.max_wall_seconds,
        environment=sys.executable, pilot_rerun=False, posterior_assessment="not_run")
    write(args.output/"manifest.json", manifest)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--parent", str(args.parent),
        "--output", str(args.output), "--plan", args.plan, "--max-wall-seconds", str(args.max_wall_seconds)]
    interrupted = None
    with subprocess.Popen(command, start_new_session=True) as process:
        try:
            code = process.wait(timeout=max(0., args.max_wall_seconds-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            interrupted = "supervisor_budget_exhausted"
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGKILL)
            code = process.wait()
            interrupted = "supervisor_interrupted"
        except BaseException:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    manifest = read(args.output/"manifest.json")
    if interrupted is not None or not (args.output/"result.json").exists():
        sys.path.insert(0, str(source))
        from scripts.run_modern_hmc_frozen_full_model import finalize_interruption
        status = interrupted or "repair_required"
        finalize_interruption(args.output, parent["design"], status=status)
        manifest["status"] = status
    manifest.update(finished_utc=datetime.now(timezone.utc).isoformat())
    write(args.output/"manifest.json", manifest)
    write(args.output/"supervisor.json", dict(command=command, exit_code=code,
        status=manifest["status"], elapsed_seconds=time.monotonic()-started,
        max_wall_seconds=args.max_wall_seconds))
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wall-seconds", type=float, required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.parent, args.output = args.parent.resolve(), args.output.resolve()
    if not math.isfinite(args.max_wall_seconds) or args.max_wall_seconds <= 0:
        raise ValueError("explicit finite positive allowance required")
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("memory growth must be configured before framework import")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
