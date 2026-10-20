#!/usr/bin/env python3
"""Resume a frozen N5 mechanics checkpoint under an explicit new wall allowance.

Preserve the parent attempt. This diagnostic launcher changes no numerical
policy and grants no posterior or release authority.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PLAN = "docs/plans/bayesfilter-modern-hmc-n5-campaign-2026-10-08.md"


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def checked_spec(checkpoint):
    checkpoint = Path(checkpoint)
    spec = json.loads((checkpoint.parent / "execution_spec.json").read_text())["execution"]
    for path, digest in spec["source_closure"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("source changed; cannot resume frozen checkpoint: " + path)
    if not spec["config"]["use_xla"] or spec["runtime_policy"]["device_type"] != "GPU":
        raise ValueError("this recovery requires the original GPU/XLA scope")
    return spec


def _key(event):
    return (event["work_item_id"], event["trial_ordinal"], event["trial_chunk_index"])


def recovery_gpu_inventory(visible):
    """Permit shared admission for the original device with recorded headroom.

    The readiness helper has an idle-only selection policy. Only that explicit
    busy-device result permits the trusted shared-memory check below; driver,
    visibility and unexpected probe failures still fail closed.
    """
    probe = subprocess.run([str(Path.home()/".codex/bin/codex-gpu-probe"),
        "--framework", "nvidia", "--gpu", visible], capture_output=True, text=True,
        timeout=15)
    payload = json.loads(probe.stdout)
    if probe.returncode == 0 and payload["status"] == "PASSED":
        return payload
    if payload.get("error") != f"requested_gpu_compute_busy:{visible}":
        raise RuntimeError("trusted GPU readiness failed: " + str(payload))
    inventory = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15, check=True)
    rows = [dict(index=int(row[0]), uuid=row[1].strip(), name=row[2].strip(),
                 memory_used_mib=float(row[3]), memory_total_mib=float(row[4]),
                 utilization_percent=float(row[5])) for row in csv.reader(inventory.stdout.splitlines())]
    selected = next(row for row in rows if str(row["index"]) == visible)
    if selected["memory_total_mib"] - selected["memory_used_mib"] < 4096:
        raise RuntimeError("shared GPU has less than the declared 4096 MiB free-memory screen")
    return dict(status="SHARED_ADMITTED", gpu_inventory=rows, idle_probe=payload,
                admission_policy="trusted_shared_device_with_memory_growth", free_memory_screen_mib=4096,
                timestamp_utc=datetime.now(timezone.utc).isoformat(), scientific_authority=False)


def audit_recovery(result, binding, *, inherited_events, inherited_missing_keys):
    """Audit committed evidence once while charging every lost native attempt.

    The numerical controller already permits a retry of an uncommitted chunk.
    The original mechanics auditor expects unique charge keys. Give it an
    explicit unique-chunk view, then restore all attempted costs in the report.
    The actual result and checkpoint remain unchanged.
    """
    from bayesfilter.testing.inference_validation.modern_bridge_audit import audit_native_evidence

    events = list(result.accounting_events)
    # JSON round-tripping normalizes tuple/list representation only.
    if json.loads(json.dumps(events[:len(inherited_events)])) != json.loads(json.dumps(inherited_events)):
        raise ValueError("recovery changed inherited accounting")
    previous = Counter(_key(e) for e in inherited_events if e["event"] == "numerical_chunk_charged")
    unique, attempts = {}, Counter()
    for event in events:
        if event["event"] != "numerical_chunk_charged":
            continue
        key = _key(event)
        attempts[key] += 1
        if key in unique and json.loads(json.dumps(unique[key])) != json.loads(json.dumps(event)):
            raise ValueError("retry charge differs from the frozen trial")
        unique[key] = event
    for key, count in attempts.items():
        allowed = max(1, previous[key]) + (key in inherited_missing_keys)
        if count > allowed or (count > previous[key] and previous[key] and key not in inherited_missing_keys):
            raise ValueError("retry of committed or unrecorded trial")
    view = replace(result, accounting_events=tuple(unique.values()))
    audit = audit_native_evidence(view, binding)
    retry_work = sum((attempts[key] - 1) * event["gradient_work"] for key, event in unique.items())
    audit["unique_chunk_gradient_work"] = audit["charged_gradient_work"]
    audit["charged_gradient_work"] += retry_work
    audit["interrupted_retry_gradient_work"] = retry_work
    audit["attempted_chunk_count"] = sum(attempts.values())
    audit["interrupted_attempt_count"] = sum(attempts.values()) - audit["returned_chunk_count"]
    return audit


def worker(checkpoint, output, max_wall):
    started = time.monotonic()
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH", "").lower() != "true":
        raise ValueError("memory growth must be configured before TensorFlow import")
    spec = checked_spec(checkpoint)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != spec["runtime_policy"]["cuda_visible_devices"]:
        raise ValueError("resume must preserve the recorded visible device")
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    from scripts.run_modern_hmc_bridge import _fixture
    target = spec["target_lineage"]["model"]
    fixture, _, _, metadata = _fixture(target, use_xla=True)
    if metadata["data_sha256"] != spec["target_lineage"]["data"]:
        raise ValueError("saved target data changed")
    from bayesfilter.inference.hmc_candidate_set_checkpoint import load_numerical_tuning_checkpoint
    from bayesfilter.inference import resume_hmc_candidate_set_tuning
    from bayesfilter.runtime.execution_budget import execution_budget
    binding, controller = load_numerical_tuning_checkpoint(checkpoint, adapter=fixture.adapter)
    inherited_events = list(controller.result().accounting_events)
    # Chunks store the work id in their work record, unlike accounting events.
    returned = {(row["work"]["work_item_id"], c["trial_ordinal"], c["trial_chunk_index"])
                for row in binding._evidence.values() for c in row.get("chunks", ())}
    returned.update((wid, c["trial_ordinal"], c["trial_chunk_index"])
                    for wid, chunks in binding._partial.items() for c in chunks)
    missing = {_key(e) for e in inherited_events if e["event"] == "numerical_chunk_charged"} - returned
    before = dict(binding_hash=binding.binding_hash, config=controller.config.payload(),
                  inherited_attempts=sum(e["event"] == "numerical_chunk_charged" for e in inherited_events),
                  inherited_returned=len(returned), inherited_missing_keys=sorted(missing),
                  memory_policy=memory, data_sha256=metadata["data_sha256"],
                  source_closure=spec["source_closure"], tensorflow=tf.__version__)
    write_json(output / "recovery-before.json", before)
    deadline = started + max_wall
    with execution_budget(check=lambda: time.monotonic() < deadline, replace_wall_time_limit=True):
        run = resume_hmc_candidate_set_tuning(checkpoint, adapter=fixture.adapter)
    binding = run.adapter._execution_binding
    if binding.binding_hash != before["binding_hash"] or run.result.config.payload() != before["config"]:
        raise ValueError("recovery changed frozen numerical identity")
    work = audit_recovery(run.result, binding, inherited_events=inherited_events,
                          inherited_missing_keys=missing)
    from bayesfilter.inference.hmc_candidate_set_retained import (
        export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners)
    exported = export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=binding, output_dir=output / "members")
    loaded = load_hmc_candidate_retained_runners(tuple(exported.values()), adapter=fixture.adapter)
    if set(loaded) != set(run.result.verified_candidate_ids):
        raise ValueError("recovery export lost a verified member")
    expected = {c.candidate_id for c in run.result.candidates}
    passed = (run.result.completion_status == "complete" and len(expected) == 2
              and expected == set(loaded) and work["returned_chunk_count"] == 64
              and work["unobserved_attempted_chunk_count"] == 0)
    result = dict(status="mechanics_verified" if passed else "mechanics_incomplete",
        target=target, work=work, binding_hash=binding.binding_hash,
        members={key: str(path) for key, path in exported.items()}, wall_seconds=time.monotonic()-started,
        gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"),
        posterior_qualification=False, method_ranking=None, plan=PLAN)
    write_json(output / "result.json", result)
    print(json.dumps(dict(status=result["status"], wall_seconds=result["wall_seconds"],
                          returned=work["returned_chunk_count"], verified=len(loaded))), flush=True)
    return 0 if passed else 1


def supervise(args):
    if not math.isfinite(args.max_wall_time_seconds) or args.max_wall_time_seconds <= 30:
        raise ValueError("wall cap must exceed the 30-second cleanup reserve")
    spec = checked_spec(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = dict(schema="bayesfilter.modern_hmc_bridge_recovery.v1", status="running",
        command=[sys.executable, *sys.argv], plan=PLAN, checkpoint_parent=str(args.checkpoint.resolve()),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        started_utc=datetime.now(timezone.utc).isoformat(), environment=sys.executable,
        wall_cap_seconds=args.max_wall_time_seconds, result_file=str(args.output / "result.json"),
        launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gpu_trust_basis="trusted_escalated_gpu_execution", numerical_policy_unchanged=True)
    write_json(args.output / "manifest.json", manifest)
    try:
        visible = spec["runtime_policy"]["cuda_visible_devices"]
        inventory = recovery_gpu_inventory(visible)
        write_json(args.output / "gpu-probe.json", inventory)
        parent_probe = next((parent/"gpu-probe.json" for parent in args.checkpoint.parents
                             if (parent/"gpu-probe.json").is_file()), None)
        if parent_probe is None:
            raise ValueError("original physical GPU provenance is missing")
        previous = json.loads(parent_probe.read_text())
        old_device = next(d for d in previous["gpu_inventory"] if str(d["index"]) == visible)
        new_device = next(d for d in inventory["gpu_inventory"] if str(d["index"]) == visible)
        if old_device["uuid"] != new_device["uuid"]:
            raise ValueError("original physical GPU changed")
        destination = args.output / "cell"
        shutil.copytree(args.checkpoint.parent, destination)
        env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true", CUDA_VISIBLE_DEVICES=visible,
                   TF_CPP_MIN_LOG_LEVEL="3", OMP_NUM_THREADS="2",
                   TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="2")
        remaining = args.max_wall_time_seconds - (time.monotonic() - started)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
            "--checkpoint", str(destination / "tuning_checkpoint.json"), "--output", str(args.output),
            "--max-wall-time-seconds", str(remaining - 30)]
        manifest.update(worker_command=command, physical_gpu_uuid=new_device["uuid"],
                        cuda_visible_devices=visible)
        write_json(args.output / "manifest.json", manifest)
        with (args.output / "worker.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=remaining)
        manifest.update(status="finished" if result.returncode == 0 else "repair_needed",
                        returncode=result.returncode)
    except subprocess.TimeoutExpired:
        manifest.update(status="wall_budget_exhausted", partial_evidence_preserved=True)
    except Exception as exc:
        manifest.update(status="launch_failed", error_type=type(exc).__name__, error=str(exc))
    manifest.update(wall_seconds=time.monotonic()-started, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    return 0 if manifest["status"] == "finished" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-wall-time-seconds", type=float, required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        try:
            return worker(args.checkpoint, args.output, args.max_wall_time_seconds)
        except Exception as exc:
            write_json(args.output / "failure.json", dict(error_type=type(exc).__name__, error=str(exc)))
            raise
    return supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
