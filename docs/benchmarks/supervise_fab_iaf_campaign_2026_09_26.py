"""Bounded local queue for the six reviewed q20 training arms; no framework imports."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26/q20-r1"
PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
RUNNER = "docs/benchmarks/run_fab_iaf_training_campaign_2026_09_26.py"


def main():
    BASE.mkdir(parents=True, exist_ok=False)
    jobs = [(arm, seed, 3600 if arm == "fab" else 1800)
            for seed in range(3) for arm in ("fab", "reverse_kl")]
    running, completed = {}, []
    started = time.monotonic()
    env = dict(os.environ, TF_FORCE_GPU_ALLOW_GROWTH="true")

    def snapshot():
        body = {"plan": "docs/plans/bayesfilter-fab-iaf-training-campaign-2026-09-26.md",
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running" if running or jobs else "finished",
            "worker_budget_seconds": 16200, "devices": [1, 2],
            "completed": completed,
            "running": [{k: v for k, v in row.items() if k not in {"process", "log", "started"}}
                        for row in running.values()],
            "pending": jobs, "supervisor_wall_seconds": time.monotonic() - started}
        tmp = BASE / "queue.json.tmp"
        tmp.write_text(json.dumps(body, indent=2) + "\n")
        tmp.replace(BASE / "queue.json")

    while jobs or running:
        for gpu in (1, 2):
            if gpu in running or not jobs:
                continue
            arm, seed, cap = jobs.pop(0)
            output = BASE / f"{arm}-seed{seed}-gpu{gpu}"
            command = ["timeout", "--signal=TERM", "--kill-after=10s", f"{cap - 10}s",
                PYTHON, RUNNER, "--arm", arm, "--target", "q20", "--gpu", str(gpu),
                "--seed", str(seed), "--updates", "240", "--worker-seconds", str(cap - 20),
                "--output", str(output)]
            log = (BASE / f"{arm}-seed{seed}-gpu{gpu}.log").open("w")
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[gpu] = {"arm": arm, "seed": seed, "gpu": gpu, "output": str(output),
                "command": command, "pid": process.pid, "worker_cap": cap,
                "process": process, "log": log, "started": time.monotonic()}
            print(f"started {arm} seed={seed} gpu={gpu} pid={process.pid}", flush=True)
            snapshot()
        for gpu, row in list(running.items()):
            code = row["process"].poll()
            if code is None:
                continue
            wall = time.monotonic() - row["started"]
            row["log"].close()
            result_path = Path(row["output"]) / "result.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else {"status": "missing_result"}
            completed.append({k: v for k, v in row.items() if k not in {"process", "log", "started"}} |
                {"exit_code": code, "process_wall_seconds": wall,
                 "status": result.get("status"), "optimizer_updates": result.get("optimizer_updates")})
            del running[gpu]
            print(f"finished {row['arm']} seed={row['seed']} status={result.get('status')} updates={result.get('optimizer_updates')} seconds={wall:.1f}", flush=True)
            # An engineering failure pauses the queue for localized repair.
            # Poor finite geometry/coverage is deliberately not a stop signal.
            if result.get("status") not in {"complete", "partial_budget"}:
                jobs.clear()
            snapshot()
        if running:
            time.sleep(5)
    snapshot()
    return 0 if len(completed) == 6 and all(r["status"] == "complete" for r in completed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
