"""Existing-budget supervision of the eight-sweep fixed-bank GPU check."""

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import uuid


ROOT = Path(__file__).resolve().parents[6]
DIAGNOSTIC = Path(__file__).resolve().parent
CAMPAIGN = DIAGNOSTIC.parent
sys.path.insert(0, str(ROOT))
from bayesfilter.runtime.durable_tensor_checkpoint import durable_json, payload_hash
from bayesfilter.runtime.display_gpu_policy import probe_inventory, select_gpus
from bayesfilter.runtime.parallel_tuning import ParallelTuningTask, run_parallel_tuning_wave
from bayesfilter.runtime.campaign_budget_ledger import CampaignBudgetLedger


spec = importlib.util.spec_from_file_location("eight_sweep_recovery", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
CAP_SECONDS = 900.0


def validate_artifact(task, payload):
    if (payload.get("status") != "completed" or payload.get("passed") is not True
            or payload.get("sources") != current or payload.get("gpu_uuid") != task.gpu_uuid
            or not payload.get("memory_policy", {}).get("all_physical_devices_memory_growth")
            or payload.get("jit_compile") is not True):
        raise ValueError("GPU fixed-bank validation did not pass its declared checks")


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Supervisor requires hidden GPUs and pre-import memory-growth policy")
with (CAMPAIGN / ".coordinator.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = json.loads((CAMPAIGN / "campaign-start.json").read_bytes())
    previous = json.loads((CAMPAIGN / "source-migration.json").read_bytes())
    recovery._validate_source_migration(start, CAMPAIGN, previous["to_sources"])
    current = recovery.source_hashes()
    changed = sorted(path for path in set(previous["to_sources"]) | set(current)
                     if previous["to_sources"].get(path) != current.get(path))
    if changed != [recovery.NUMERICAL_CORE_PATH]:
        raise RuntimeError(f"unexpected change since four-sweep migration: {changed}")
    validation = json.loads((DIAGNOSTIC / "eight-sweep-validation-a8df05dcf2/validation.json").read_bytes())
    if current != validation["current_sources"] or hashlib.sha256(recovery.PLAN.read_bytes()).hexdigest() != start["plan_hash"]:
        raise RuntimeError("Current sources differ from CPU-validated repair or frozen plan")
    ledger = CampaignBudgetLedger(CAMPAIGN / "campaign_budget_ledger.json")
    payload = ledger.read()
    if (payload["source_hash"] != payload_hash(start["sources"]) or payload["plan_hash"] != start["plan_hash"]
            or payload["total_budget_seconds"] != recovery.TOTAL_GPU_SECONDS or payload["reserved_seconds"] != 0.0
            or any(attempt["status"] == "running" for attempt in payload["attempts"])):
        raise RuntimeError("Campaign binding changed or live reservations remain")
    if ledger.remaining_seconds() < CAP_SECONDS:
        raise RuntimeError("Fixed-bank diagnostic does not fit remaining budget")
    selection = select_gpus(probe_inventory(), 1, estimated_peak_mib=4096)
    if not selection["selected"]:
        raise RuntimeError("No eligible GPU for fixed-bank validation")
    device = selection["selected"][0]
    attempt = "eight-sweep-gpu-validation-" + uuid.uuid4().hex[:10]
    output = CAMPAIGN / "launches" / attempt
    output.mkdir(parents=True, exist_ok=False)
    binding = {"source_hash": payload_hash(start["sources"]), "plan_hash": start["plan_hash"]}
    namespace = {"mode": "eight_sweep_fixed_bank_validation", "random_draws": False}
    source = DIAGNOSTIC / "preflight_bank_gpu_validation.py"
    command = ("timeout", "--signal=TERM", "--kill-after=30s", "870s", sys.executable,
               str(source), "--gpu-uuid", device["uuid"], "--output-dir", str(output / "worker"))
    task = ParallelTuningTask("numerical-validation", 0, device["uuid"], output / "worker", command)
    durable_json(output / "placement.json", selection)
    durable_json(output / "job.json", {"sources": current, "command": command, "cap_seconds": CAP_SECONDS,
                 "plan": str(ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md")})
    for path in (source, Path(__file__), ROOT / recovery.NUMERICAL_CORE_PATH):
        shutil.copy2(path, output / path.name)
    ledger.start_attempt(attempt_id=attempt, output_root=output, seed_namespace=namespace, **binding)
    ledger.reserve_chunk(attempt_id=attempt, arm=task.task_id, chunk_index=0, reserve_seconds=CAP_SECONDS,
                         output_root=output, seed_namespace=namespace, **binding)
    observed = None
    try:
        wave = run_parallel_tuning_wave(
            (task,), timeout_seconds=CAP_SECONDS, terminate_grace_seconds=30.0, cwd=ROOT,
            base_environment={**os.environ, "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"},
            artifact_validator=validate_artifact,
            monitor_callback=recovery.load_parallel()._headroom_monitor((task,), output, 0),
        )
        durable_json(output / "wave.json", wave)
        observed = wave["results"][0]
    finally:
        measured = observed["elapsed_seconds"] if observed else CAP_SECONDS
        status = observed["status"] if observed else "conservative_unobserved_worker_bound"
        ledger.settle_arm(attempt_id=attempt, arm=task.task_id, measured_seconds=measured, status=status,
                          repair="eight-sweep GPU fixed-bank numerical validation")
        ledger.finish_attempt(attempt_id=attempt, status=status)
        durable_json(output / "summary.json", {"result": observed, "ledger": ledger.read()})
    print(json.dumps({"output": str(output), "result": observed, "remaining_gpu_seconds": ledger.remaining_seconds()}))
    raise SystemExit(0 if observed and observed["status"] == "completed" else 1)
