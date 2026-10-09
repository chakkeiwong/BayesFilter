"""Wait for capacity, finish the bounded C1 recovery, then run frozen SSM stages.

This parent is framework-free. Queue waiting is a separate, conservatively
charged reservation and cannot exhaust a numerical preflight's allowance.
"""
from pathlib import Path
import sys

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datetime import datetime, timezone
from dataclasses import replace
import os
import signal
import subprocess
import time
from types import SimpleNamespace

import run_hmc_c1_recovery_then_ssm as recovery
from bayesfilter.testing.inference_validation import ssm_campaign as campaign
from bayesfilter.testing.inference_validation.execution import plan_suite
from bayesfilter.testing.inference_validation.storage import read_json, write_json
from bayesfilter.testing.inference_validation.timeout_policy import (
    TimeoutPolicy, default_workload_probe, gpu_admission_supported,
)


def check_wall_time(config, seconds=0.):
    now = time.time()
    start = config["campaign_started_epoch"]
    if now >= start+42*3600 or now+seconds > start+46*3600:
        raise ValueError("continuation would exceed original 42/46-hour deadlines")


def charged_ssm_seconds(config, ledger):
    roots = [Path(config["root"])]
    if config.get("ssm_root"):
        roots.append(Path(config["ssm_root"]))
    roots.extend(Path(p) for p in config.get("ssm_accounting_roots", []))
    inherited = set(config["prior_ssm_receipts"])
    return sum(row["elapsed_seconds"] for row in ledger["records"]
        if row["receipt"] in inherited or (
            any(Path(row["receipt"]).is_relative_to(root) for root in roots)
            and "recovery-execution.json" != Path(row["receipt"]).name))


def check_completed_recovery(result_path, repo):
    """Confirm every original failure has a clean, unchanged numerical result."""
    result = read_json(result_path)
    rows = result["outcomes"]
    if (result.get("planned") != 10 or result.get("completed") != 10
            or tuple(sorted(row["replication"] for row in rows)) != recovery.INDICES
            or any(row["status"] != "complete" for row in rows)):
        raise ValueError("recovery is incomplete; cannot skip remaining original seeds")
    _, source, original = recovery.original_paths(Path(repo))
    source_record = recovery.checked_original_source(source)
    if result["numerical_source_identity"] != source_record["identity"]:
        raise ValueError("completed recovery source mismatch")
    checked = {}
    for cell in {Path(row["receipt"]).parent.parent for row in rows}:
        for evidence in recovery.completed_recoveries(cell, original):
            checked[evidence["receipt"]] = evidence["replication"]
    if any(checked.get(row["receipt"]) != row["replication"] for row in rows):
        raise ValueError("completed recovery receipt or replication changed")
    return {"completed":10,"result_file":str(result_path),"source_identity":source_record["identity"]}


def check_ssm_allocation(config, seconds):
    check_wall_time(config, seconds)
    ledger = read_json(config["ledger"])
    available = campaign.free_gpu_seconds(ledger, c1_active=campaign.c1_is_active())
    if seconds > available or charged_ssm_seconds(config, ledger)+seconds > config["ssm_gpu_cap_seconds"]:
        raise ValueError("continuation exceeds cumulative SSM or grant allowance")


def pricing_only_resource_failures(prepared):
    """Failed budget-limited lanes stay unpriced; unknown failures need review."""
    index_path = Path(prepared)/"pricing/run_index.json"
    if not index_path.exists():
        return False
    index = read_json(index_path)
    expected = read_json(Path(prepared)/"pricing.json")
    if set(index["jobs"]) != {d["design_id"] for d in expected["designs"]}:
        return False
    found = False
    for key, job in index["jobs"].items():
        if job["status"] == "complete":
            continue
        found = True
        attempts = job.get("attempts", [])
        if attempts and attempts[-1].get("status") == "timed_out":
            continue
        exits = sorted((Path(prepared)/"pricing"/key/"replication-0000").glob("process-attempt-*-exit.json"))
        if (job["status"] != "failed" or not exits
                or read_json(exits[-1]).get("status") not in {"timed_out", "budget_exhausted"}):
            return False
    return found


def wait_for_capacity(config, label, devices, *, probe=None):
    """Select eligible capacity under the explicit idle/shared policy."""
    root = Path(config["root"])
    ledger_path = Path(config["ledger"])
    ledger = read_json(ledger_path)
    used = sum(r["elapsed_seconds"] for r in ledger["records"]
        if r.get("accounting") == "capacity_queue_enclosing_charge")
    seconds = max(0., min(config["queue_cap_seconds"]-used,
        config["campaign_started_epoch"]+42*3600-time.time()))
    check_ssm_allocation(config, seconds)
    if seconds <= 0:
        return None
    receipt_path = root/f"queue-{label}-execution.json"
    if receipt_path.exists():
        raise ValueError("capacity attempt already exists; preserve prior evidence")
    ledger["active_reservation"] = {"service":config["service"], "receipt":str(receipt_path),
        "maximum_gpu_seconds_including_shutdown":seconds}
    write_json(ledger_path, ledger)
    started = time.monotonic()
    deadline = started+seconds
    prior_device = os.environ.get("CUDA_VISIBLE_DEVICES")
    selected = None
    scans = 0
    policy = TimeoutPolicy(gpu_admission_mode=config.get("gpu_admission_mode", "idle"))
    probe = probe or default_workload_probe
    try:
        while time.monotonic() < deadline:
            samples = []
            for device in devices:
                if time.monotonic() >= deadline:
                    break
                os.environ["CUDA_VISIBLE_DEVICES"] = device
                try:
                    probe_policy = replace(policy,workload_probe_timeout_seconds=min(
                        policy.workload_probe_timeout_seconds,max(.001,(deadline-time.monotonic())/2)))
                    sample = dict(probe("gpu", None, probe_policy))
                except Exception as exc:
                    sample = {"trusted_for_extension":False,"reason":str(exc)}
                samples.append({"uuid":device, **sample})
                if (policy.gpu_admission_mode == "idle" and gpu_admission_supported(sample, policy)
                        and time.monotonic() < deadline):
                    selected = device
                    break
            if policy.gpu_admission_mode == "shared" and time.monotonic() < deadline:
                eligible = [sample for sample in samples if gpu_admission_supported(sample, policy)]
                if eligible:
                    best = min(eligible, key=lambda sample: (
                        sample.get("contended") is not False,
                        sample["gpu"].get("utilization_gpu_pct", 100.),
                        sample["gpu"]["memory_used_mib"]-sample["gpu"]["memory_total_mib"]))
                    selected = best["uuid"]
            scans += 1
            write_json(root/f"queue-{label}-progress.json", {"scans":scans, "latest_samples":samples,
                "elapsed_seconds":time.monotonic()-started, "selected_gpu":selected,
                "gpu_admission_mode":policy.gpu_admission_mode,
                "state":"admitted" if selected else "waiting_for_capacity"})
            if selected:
                break
            delay = min(policy.telemetry_interval_seconds, deadline-time.monotonic())
            if delay > 0:
                time.sleep(delay)
    finally:
        if prior_device is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES",None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = prior_device
        receipt = {"receipt":str(receipt_path),"resource":"gpu", "elapsed_seconds":time.monotonic()-started,
            "selected_gpu":selected, "scans":scans, "exit_code":0 if selected else 75,
            "gpu_admission_mode":policy.gpu_admission_mode,
            "accounting":"capacity_queue_enclosing_charge", "gpu_framework_initialized":False,
            "interpretation":"conservative wall-time charge; no numerical GPU worker launched",
            "ended_utc":datetime.now(timezone.utc).isoformat()}
        write_json(receipt_path,receipt)
        recovery.settle(ledger_path,receipt)
    return selected


def run(config):
    root = Path(config["root"])
    prepared = Path(config["ssm_root"])
    if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH","").lower() != "true":
        raise ValueError("memory growth must be enabled before launch")
    if (prepared/"checkpoint.json").exists():
        raise ValueError("prepared continuation was already attempted; preserve its checkpoint")
    def status(stage, state, **details):
        write_json(root/"status.json", {"stage":stage,"state":state,
            "updated_utc":datetime.now(timezone.utc).isoformat(), **details})
    try:
        if config.get("completed_recovery_result"):
            completed = check_completed_recovery(config["completed_recovery_result"],config["repo"])
            status("recovery", "completed_evidence_reused", **completed)
            write_json(root/"reused-recovery.json",completed)
        else:
            status("capacity", "waiting_for_capacity")
            device = wait_for_capacity(config, "recovery", config["gpu_uuids"])
            if device is None:
                status("capacity", "deferred_capacity_wait_exhausted")
                return 75
            os.environ["CUDA_VISIBLE_DEVICES"] = device
            check_wall_time(config, recovery.RECOVERY_SECONDS+30)
            status("recovery", "running", gpu_uuid=device)
            args = SimpleNamespace(repo=Path(config["repo"]), output=root/"recovery-r2",
                ssm_root=prepared, ledger=Path(config["ledger"]),service=config["service"],
                previous_cell=Path(config["previous_cell"]))
            old, new, source = recovery.run_recovery(args)
            subprocess.run([sys.executable,"-c",recovery.SUMMARY_CODE,str(old),str(new),
                str(root/"recovered-beta-summary.json"),str(args.previous_cell)],cwd=source,
                env={**os.environ,"CUDA_VISIBLE_DEVICES":"-1"},check=True,timeout=60)
        # Preflight has not yet bound the SSM device. Select it once here;
        # subsequent stages must retain that same device for valid pricing.
        status("ssm_capacity", "waiting_for_capacity")
        device = wait_for_capacity(config, "ssm", config["gpu_uuids"])
        if device is None:
            status("ssm_capacity", "deferred_capacity_wait_exhausted")
            return 75
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        write_json(prepared/"checkpoint.json", {"started_epoch":config["campaign_started_epoch"],
            "stages":{},"plan_file":config["plan_file"],"gpu_uuid":device})
        for stage in ("preflight-mechanics","preflight-pipeline","pricing","main"):
            status(stage,"waiting_for_capacity",gpu_uuid=device)
            if wait_for_capacity(config, stage, [device]) is None:
                status(stage,"deferred_capacity_wait_exhausted",gpu_uuid=device)
                return 75
            main_cap = None
            if stage == "main":
                ledger = read_json(config["ledger"])
                main_cap = max(0., config["ssm_gpu_cap_seconds"]-charged_ssm_seconds(config,ledger)-30.)
                campaign.price(prepared, ledger, allocation_cap_seconds=main_cap)
            envelope = plan_suite(read_json(prepared/(stage+".json")))["maximum_worker_seconds"]+30.
            check_ssm_allocation(config,envelope)
            status(stage,"running",gpu_uuid=device)
            receipt = campaign.run_stage(prepared,stage,Path(config["ledger"]),
                gpu_admission_mode=config.get("gpu_admission_mode", "idle"),
                allocation_cap_seconds=main_cap)
            if receipt.get("exit_code",0) != 0:
                if stage == "pricing" and pricing_only_resource_failures(prepared):
                    status(stage,"resource_limited_lanes_unpriced",receipt=receipt)
                    continue
                status(stage,"requires_diagnosis",receipt=receipt)
                return 1
        status("terminal","complete; inspect per-slot scientific results")
        return 0
    except BaseException as exc:
        status("interrupted", "requires_diagnosis",exception=type(exc).__name__,reason=str(exc))
        raise
    finally:
        campaign.summarize(prepared)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",type=Path)
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    raise SystemExit(run(read_json(args.config)))


if __name__ == "__main__":
    main()
