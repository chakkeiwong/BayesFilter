#!/usr/bin/env python3
"""Fixed operations for the authorized q20 campaign; no arbitrary launch inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

REPO = Path("/home/ubuntu/python/BayesFilter")
ROOT = REPO / "docs/plans/artifacts/q20-master-operations-2026-09-21"
OLD = REPO / "docs/plans/artifacts/ssl-lstm-q20-deadline-continuation-2026-09-21"
OLD_UNIT = "bayesfilter-q20-deadline-20260925-1800-r2.service"
PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
ENVIRONMENT = {"TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_NUM_INTRAOP_THREADS": "2",
    "TF_NUM_INTEROP_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1", "BAYESFILTER_PRELOAD_CUSTOM_OP": "0"}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def service(unit):
    result = subprocess.run(["systemctl", "--user", "show", unit,
        "--property=LoadState,ActiveState,SubState,MainPID,ExecMainStartTimestamp,RuntimeMaxUSec,Result"],
        capture_output=True, text=True, timeout=15.)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "service inspection failed")
    return dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)


def active(record):
    return record.get("ActiveState") in {"active", "activating", "deactivating", "reloading"}


def status():
    job = read(ROOT / "job.json")
    launch = read(ROOT / "launch.json") if (ROOT / "launch.json").is_file() else {}
    runtime_root = Path(job.get("output_root", ROOT))
    current = read(runtime_root / "status.json") if (runtime_root / "status.json").is_file() else {}
    campaign = runtime_root / job.get("campaign_name", "campaign-01")
    if not (campaign / "campaign.json").is_file():
        campaign = Path(job["parent_campaign"])
    state = read(campaign / "campaign.json")
    inflight = sum(max(0., time.time()-a["started_epoch"]) for a in state["attempts"] if a["status"] == "running")
    next_phase = read(campaign / "next-phase.json") if (campaign / "next-phase.json").is_file() else None
    return {"observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "deadline_local": job["deadline_local"], "campaign_status": state["status"],
        "remaining_campaign_hours_observed": max(0., state["campaign_limit"]-state["spent_seconds"]-inflight)/3600,
        "supervisor": service(launch["unit"]) if launch else {"status": "not_started"},
        "parent_service": service(job["parent_unit"]), "current": current,
        "next_phase": next_phase, "plan_file": job["plan_file"]}


def ensure():
    job = read(ROOT / "job.json")
    driver = Path(job["driver_path"])
    if hashlib.sha256(driver.read_bytes()).hexdigest() != job["driver_sha256"]:
        raise ValueError("prepared supervisor changed; inspect and validate before activation")
    grace = 5  # inherited supervisor stop grace, also reserved before calendar end
    with (ROOT / "control.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        launch_path = ROOT / "launch.json"
        if launch_path.is_file():
            previous = read(launch_path)
            observed = service(previous["unit"])
            if active(observed):
                return {"action": "already_running", "unit": previous["unit"], "service": observed}
        if time.time() + 2*grace >= job["deadline_epoch"]:
            raise ValueError("calendar deadline reached; ensure cannot renew it")
        campaign_root = Path(job.get("output_root", ROOT)) / job.get("campaign_name", "campaign-01")
        campaign = campaign_root / "campaign.json"
        if campaign.is_file():
            saved = read(campaign)
            if not (saved["status"].startswith("running:") or saved["status"] in {
                    "CONTINUATION_PREPARED", "WAITING_FOR_GPU", "MASTER_INTERRUPTED"}):
                return {"action": "inspection_required", "campaign_status": saved["status"],
                        "next_phase": str(campaign_root / "next-phase.json")}
        old = service(OLD_UNIT)
        replaced = False
        if active(old):
            old_status = read(OLD / "status.json")
            if (old_status.get("status") != "WAITING_FOR_EXISTING_MASTER"
                    or old_status.get("numerical_work_launched") is not False
                    or (OLD / "campaign-01/campaign.json").exists()):
                raise RuntimeError("previous continuation has taken over; inspect before replacing its supervisor")
            subprocess.run(["systemctl", "--user", "stop", OLD_UNIT], check=True, timeout=20.)
            replaced = True
        launches = ROOT / "launches"
        launches.mkdir(exist_ok=True)
        number = len(list(launches.glob("*.json"))) + 1
        unit = f"bayesfilter-q20-master-operations-20260921-{number:02d}"
        runtime = math.floor(job["deadline_epoch"] - time.time()) - 2*grace
        command = ["systemd-run", "--user", "--unit="+unit,
            "--property=WorkingDirectory="+job["source_root"],
            "--property=RuntimeMaxSec="+str(runtime), "--property=TimeoutStopSec="+str(grace),
            "--property=KillMode=control-group", "--property=StandardOutput=append:"+str(ROOT/"console.log"),
            "--property=StandardError=inherit",
            *("--setenv="+key+"="+value for key, value in ENVIRONMENT.items()),
            PYTHON, str(driver), "supervise", "--job", str(ROOT/"job.json")]
        record = {"unit": unit+".service", "command": command, "environment": ENVIRONMENT,
            "deadline_local": job["deadline_local"], "runtime_max_seconds": runtime,
            "stop_grace_seconds": grace, "driver_sha256": job["driver_sha256"],
            "control_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "replaced_waiting_observer": replaced, "numerical_parent_restarted": False,
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "launching"}
        write(launches / f"{number:03d}.json", record)
        subprocess.run(command, check=True, timeout=20.)
        record.update(status="running", service=service(record["unit"]))
        write(launches / f"{number:03d}.json", record)
        write(launch_path, record)
        return {"action": "started", **record}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "ensure"))
    args = parser.parse_args()
    print(json.dumps(status() if args.action == "status" else ensure(), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
