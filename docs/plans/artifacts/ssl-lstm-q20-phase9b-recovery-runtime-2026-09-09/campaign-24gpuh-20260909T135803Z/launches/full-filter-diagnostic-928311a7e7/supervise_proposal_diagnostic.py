"""Bounded debugging supervisor using the existing campaign's accounting."""

import fcntl
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bayesfilter.runtime.durable_tensor_checkpoint import durable_json, payload_hash
from bayesfilter.runtime.display_gpu_policy import probe_inventory, select_gpus
from bayesfilter.runtime.parallel_tuning import ParallelTuningTask, run_parallel_tuning_wave

spec = importlib.util.spec_from_file_location("diagnostic_recovery_supervisor", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=("prefix", "cached-gradient", "covariance", "eigensystem", "refinement", "full-filter"), default="prefix")
args = parser.parse_args()
cap = 1800.0 if args.mode == "prefix" else 900.0
with (CAMPAIGN / ".coordinator.lock").open("a+b") as lock:
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    start, ledger = recovery.initialize_campaign(CAMPAIGN, resume=True)
    if ledger.read()["reserved_seconds"]:
        raise RuntimeError("Existing live reservations must be reconciled first")
    selection = select_gpus(probe_inventory(), 2, estimated_peak_mib=4096)
    expected = json.loads((CAMPAIGN / "strict-gpu.json").read_text())["uuid"]
    if not any(row["uuid"] == expected for row in selection["selected"]):
        raise RuntimeError("The original strict comparator GPU is unavailable")
    output = CAMPAIGN / "launches" / (args.mode + "-diagnostic-" + uuid.uuid4().hex[:10])
    output.mkdir()
    attempt = output.name
    binding = {"source_hash": payload_hash(start["sources"]), "plan_hash": start["plan_hash"]}
    seed_namespace = {"mode": args.mode + "_diagnostic", "original_transition_prefix": 159}
    ledger.start_attempt(attempt_id=attempt, output_root=output, seed_namespace=seed_namespace, **binding)
    ledger.reserve_chunk(attempt_id=attempt, arm="strict", chunk_index=0, reserve_seconds=cap,
                         output_root=output, seed_namespace=seed_namespace, **binding)
    source_name = {"covariance": "trace_covariance_diagnostic.py", "eigensystem": "gpu_eigensystem_diagnostic.py", "refinement": "gpu_eigensystem_diagnostic.py", "full-filter": "full_filter_refinement_diagnostic.py"}.get(args.mode, "replay_proposal_diagnostic.py")
    source = Path(__file__).with_name(source_name)
    mode_arguments = {"prefix": ("--sample-chain-results", "159"), "cached-gradient": ("--cached-gradient",), "covariance": (), "eigensystem": (), "refinement": ("--refine",), "full-filter": ()}[args.mode]
    command = ("timeout", "--signal=TERM", "--kill-after=30s", f"{cap - 30:g}s", sys.executable,
               str(source), "--gpu-uuid", expected, "--output-dir", str(output / "strict"), *mode_arguments)
    task = ParallelTuningTask("strict", 2, expected, output / "strict", command)
    durable_json(output / "placement.json", selection)
    durable_json(output / "strict-job.json", {
        "command": command, "gpu_uuid": expected, "cap_seconds": cap,
        "diagnostic_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
        "sources": recovery.source_hashes(), "scientific_sources_unchanged": True,
    })
    (output / source.name).write_bytes(source.read_bytes())
    (output / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    if args.mode in ("refinement", "full-filter"):
        helper = source.with_name("jacobi_refinement_diagnostic.py")
        (output / helper.name).write_bytes(helper.read_bytes())
    observed = None
    try:
        wave = run_parallel_tuning_wave(
            (task,), timeout_seconds=cap, terminate_grace_seconds=30.0, cwd=ROOT,
            base_environment={**os.environ, "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1"},
            monitor_callback=recovery.load_parallel()._headroom_monitor((task,), output, 0),
        )
        durable_json(output / "wave-0.json", wave)
        observed = wave["results"][0]
    finally:
        ledger.settle_arm(attempt_id=attempt, arm="strict",
                          measured_seconds=observed["elapsed_seconds"] if observed else cap,
                          status=observed["status"] if observed else "conservative_unobserved_worker_bound",
                          repair="proposal telemetry localization; no numerical source change")
        ledger.finish_attempt(attempt_id=attempt, status=observed["status"] if observed else "failed")
        durable_json(output / "summary.json", {"result": observed, "ledger": ledger.read()})
    print(json.dumps({"output": str(output), "result": observed, "remaining_gpu_seconds": ledger.remaining_seconds()}))
    if observed["status"] != "completed":
        raise SystemExit(1)
