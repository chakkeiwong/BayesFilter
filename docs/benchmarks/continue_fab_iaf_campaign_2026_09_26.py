"""Finish budget-stopped q20 arms from exact state within the original total cap."""
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26"
OUTPUT = BASE / "q20-continuation-r1"
PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
TOTAL = 16200.


def read(path):
    return json.loads(path.read_text())


def main():
    OUTPUT.mkdir(parents=True, exist_ok=False)
    running, completed, pending = {}, [], []
    base_charge = 0.

    def snapshot(status):
        data = {"status": status, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "plan": "docs/plans/bayesfilter-fab-iaf-training-campaign-2026-09-26.md",
            "total_worker_budget_seconds": TOTAL, "previous_worker_seconds": base_charge,
            "completed": completed, "pending": pending,
            "running": [{k: v for k, v in row.items() if k not in {"process", "log", "started"}}
                        for row in running.values()]}
        temporary = OUTPUT / "queue.json.tmp"
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        temporary.replace(OUTPUT / "queue.json")

    snapshot("waiting_for_original_queue")
    while True:
        previous = read(BASE / "q20-r1/queue.json")
        if previous["status"] == "finished":
            break
        time.sleep(5)
    base_charge = sum(row["process_wall_seconds"] for row in previous["completed"])
    base_charge += read(BASE / "coverage-pricing-r1.json")["wall_seconds"]
    if len(previous["completed"]) != 6 or any(row["status"] not in {"complete", "partial_budget"} for row in previous["completed"]):
        snapshot("engineering_failure_requires_repair")
        return 1
    for row in previous["completed"]:
        if row["status"] == "partial_budget":
            progress = read(Path(row["output"]) / "progress.json")
            durations = [r["seconds"] / (r.get("update_count_this_pass", 1) or 1)
                         for r in progress if r.get("update_count_this_pass", 1)]
            pending.append({"arm": row["arm"], "seed": row["seed"], "source": row["output"],
                "starting_updates": row["optimizer_updates"],
                "seconds_per_update": statistics.median(durations[-10:])})
    pending.sort(key=lambda r: (r["arm"] == "fab", r["seed"] == 0, r["seed"]))
    observed = read(BASE / "q20-r1/reverse_kl-seed0-gpu2/result.json")
    last_progress = read(BASE / "q20-r1/reverse_kl-seed0-gpu2/progress.json")[-1]
    diagnostic_seconds = observed["wall_seconds"] - last_progress["elapsed_seconds"]
    stopped = False
    while pending or running:
        for gpu in (1, 2):
            if gpu in running or not pending:
                continue
            job = next((p for p in pending if not (p["arm"] == "fab" and
                any(q["arm"] == "reverse_kl" and q["seed"] == p["seed"]
                    for q in pending + list(running.values())))), None)
            if job is None:
                continue
            cached = (BASE / f"reference-cache/q20-seed{job['seed']}.json").exists()
            diagnostic = diagnostic_seconds * (0.75 if cached else 1.)
            needed = (240 - job["starting_updates"]) * job["seconds_per_update"] + 1.15 * diagnostic + 80.
            desired = math.ceil(needed / 25.) * 25
            available = TOTAL - base_charge - sum(r["process_wall_seconds"] for r in completed) - sum(r["worker_cap"] for r in running.values())
            if available < desired and running:
                continue  # Reclaim actual unused time before allocating the next arm.
            cap = min(desired, math.floor(available))
            if cap < diagnostic + 100:
                if running:
                    continue
                stopped = True
                break
            pending.remove(job)
            destination = OUTPUT / f"{job['arm']}-seed{job['seed']}-gpu{gpu}"
            command = ["timeout", "--signal=TERM", "--kill-after=10s", f"{cap - 10}s", PYTHON,
                "docs/benchmarks/run_fab_iaf_training_campaign_2026_09_26.py", "--arm", job["arm"],
                "--target", "q20", "--gpu", str(gpu), "--seed", str(job["seed"]),
                "--updates", "240", "--worker-seconds", str(cap - 20),
                "--resume-from", job["source"], "--output", str(destination)]
            log = destination.with_suffix(".log").open("w")
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true"),
                                       stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = {**job, "output": str(destination), "command": command, "worker_cap": cap,
                "pid": process.pid, "process": process, "log": log, "started": time.monotonic()}
            print(f"resume {job['arm']} seed={job['seed']} from={job['starting_updates']} cap={cap}", flush=True)
        for gpu, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            row["log"].close()
            result_path = Path(row["output"]) / "result.json"
            result = read(result_path) if result_path.exists() else {"status": "missing_result"}
            completed.append({k: v for k, v in row.items() if k not in {"process", "log", "started"}} |
                {"process_wall_seconds": time.monotonic() - row["started"], "exit_code": code,
                 "status": result["status"], "optimizer_updates": result.get("optimizer_updates")})
            del running[gpu]
            print(f"finished {row['arm']} seed={row['seed']} status={result['status']} updates={result.get('optimizer_updates')}", flush=True)
        snapshot("budget_exhausted" if stopped else "running")
        if stopped:
            break
        if running or pending:
            time.sleep(5)
    status = "complete" if not pending and all(r["status"] == "complete" for r in completed) else "incomplete_within_budget"
    snapshot(status)
    return 0 if status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
