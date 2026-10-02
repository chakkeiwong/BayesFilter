"""Device-assigned process queue with aggregate campaign accounting (no TF)."""
from __future__ import annotations

import os
import math
from pathlib import Path
import signal
import subprocess
import time

from bayesfilter.inference.q20_campaign_runtime import (
    CampaignBudgetError, _process_start, atomic_json, source_snapshot, utc_now,
)
from bayesfilter.inference.q20_production_config import digest


def execute_queue(campaign, jobs, devices):
    """Reserve all caps up front, then settle each owned worker separately.

    Each job has stage, command, request, cap_seconds and diagnostic. A lost
    coordinator is handled by Campaign.recover's full-cap charge per worker.
    The caller owns the campaign lock throughout, including sequential slots.
    """
    if campaign._lock is None:
        raise RuntimeError("queue requires the campaign lock")
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("queue requires distinct assigned devices")
    jobs = list(jobs)
    if len({j["stage"] for j in jobs}) != len(jobs):
        raise ValueError("queue stages must be unique")
    total = sum(j["cap_seconds"] for j in jobs)
    diagnostic = sum(j["cap_seconds"] for j in jobs if j["diagnostic"])
    if total > campaign.remaining() or diagnostic > campaign.remaining(True):
        raise CampaignBudgetError("aggregate queue reservation exceeds allowance")
    grace = campaign.config["execution"]["termination_grace_seconds"]
    for job in jobs:
        if not math.isfinite(job["cap_seconds"]) or not grace < job["cap_seconds"] <= campaign.stage_remaining(job["stage"]):
            raise CampaignBudgetError("queue job exceeds cumulative stage allowance")
    pending, live, completed = list(jobs), {}, []

    def settle(device, status=None):
        process, stream, attempt, started = live.pop(device)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        campaign._cleanup_group(process.pid)
        stream.close()
        code = process.returncode
        status = status or ("completed" if code == 0 else "timed_out" if code in (124, 137, -9, -15) else "failed")
        elapsed = time.monotonic()-started
        atomic_json(Path(attempt["directory"])/"supervisor.json",
                    {"elapsed_seconds": elapsed, "status": status, "returncode": code})
        campaign._settle(attempt, elapsed, status, code)
        campaign.save()
        completed.append(attempt)

    try:
        while pending or live:
            for device in devices:
                if device in live or not pending:
                    continue
                if source_snapshot(campaign.repo) != campaign.state["sources"]:
                    raise ValueError("queue execution sources drifted")
                job = pending.pop(0)
                cap = job["cap_seconds"]
                folder = campaign.root/"attempts"/f"{len(campaign.state['attempts']):05d}-{job['stage']}"
                folder.mkdir(parents=True, exist_ok=False)
                request = {**job["request"], "gpu": device,
                           "cooperative_seconds": cap-2*grace}
                atomic_json(folder/"request.json", request)
                argv = [str(x).replace("{attempt}", str(folder)) for x in job["command"]]
                argv = ["timeout", "--signal=TERM", f"--kill-after={grace}s", f"{cap-grace}s", *argv]
                attempt = {"stage": job["stage"], "status": "running", "directory": str(folder),
                    "command": argv, "diagnostic": job["diagnostic"], "cap_seconds": cap,
                    "started_at": utc_now(), "started_epoch": time.time(), "pid": None,
                    "process_start": None, "request_hash": digest(request), "gpu": device,
                    "accounting_basis": "sum_of_supervisor_measured_worker_wall"}
                campaign.state["attempts"].append(attempt)
                campaign.state["status"] = "PARALLEL_TRAINING_REPAIR_RUNNING"
                campaign.save()
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device),
                    "TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_NUM_INTRAOP_THREADS": "2",
                    "TF_NUM_INTEROP_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1", "BAYESFILTER_PRELOAD_CUSTOM_OP": "0",
                    "PYTHONUNBUFFERED": "1", **job.get("environment", {})}
                stream = (folder/"console.log").open("wb")
                started = time.monotonic()
                try:
                    process = subprocess.Popen(argv, cwd=campaign.repo, env=env, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    campaign._settle(attempt, time.monotonic()-started, "launch_failed", None)
                    campaign.save()
                    raise
                live[device] = process, stream, attempt, started
                attempt.update(pid=process.pid, process_start=_process_start(process.pid))
                campaign.save()
            for device in list(live):
                if live[device][0].poll() is not None:
                    settle(device)
            if live:
                time.sleep(.2)
    except BaseException:
        for device in list(live):
            settle(device, "interrupted")
        raise
    return completed
