"""Bounded CPU-only supervisor for independent R failure diagnostics.

No numerical filter kernel or selection logic lives here. Existing phase-16
artifacts are read-only; every attempt writes a new directory and manifest.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

WORKDIR = Path(__file__).resolve().parents[2]
ROOT = WORKDIR / "docs/plans/artifacts/iapf-score-failure-remedies-20260923-01"
PLAN = "docs/plans/iapf-score-failure-remedies-2026-09-23.md"
WORKER = "docs/benchmarks/diagnose_iapf_score_failures.R"
SOURCES = [WORKER, "docs/benchmarks/diagnose_iapf_score_failures.py",
           "docs/benchmarks/reference_iapf_paper.R",
           "docs/benchmarks/reference_iapf_author_choices.R",
           "docs/benchmarks/diagnostic_iapf_paper_score.R", PLAN]
BASE_SOURCES = SOURCES[2:5]
START_CPU = 65708.04461661764
START_GPU = 171882.52032958285
CAP = 7200.0
ATTEMPT_CAP = 300.0
ENV = {"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
       "VECLIB_MAXIMUM_THREADS": "1"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def task(mode, data, seed, label, method):
    return dict(mode=mode, data_seed=data, seed=seed, label=label, method=method,
                name=f"{mode}_d{data}_r{label}_{method}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Use --execute after reading the phase-17 plan")
    ROOT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKDIR, text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=WORKDIR, text=True).strip()
    if branch != "ledh-refactor-with-policy-fix":
        raise RuntimeError(f"Unexpected checkout: {branch}")
    frozen = {p: digest(WORKDIR / p) for p in BASE_SOURCES}
    mechanisms = [task("mechanism", 971080, 1772000+i, i, "score") for i in (291, 300, 694, 722)]
    repairs = [task("learner", 971080, 1772000+i, i, "score") for i in (844, 891, 947, 989)]
    repairs += [task("learner", 971080, 1772000+i, i, "score_qr_rescue") for i in (291, 300, 694, 722)]
    fresh = []
    for data, base in ((981080, 1972000), (991080, 1982000)):
        for label in range(1, 5):
            fresh += [task("learner", data, base+label, label, method)
                      for method in ("score", "score_qr_rescue", "qr")]
            fresh += [task("controls", data, base+10000+label, label, "controls")]
    stages = [mechanisms, repairs, fresh]
    jobs = [json.loads(p.read_text()) for p in sorted(ROOT.glob("*-launch.json"))]
    if any(j["status"] == "running" for j in jobs):
        raise RuntimeError("Reconcile existing running-attempt manifests before resuming")
    completed = {j["task"]["name"]: j for j in jobs if j["status"] == "complete"}
    run_manifest = dict(created_utc=utc(), git_commit=commit, branch=branch,
        working_directory=str(WORKDIR), environment=ENV, cpu_only=True,
        gpu_intentionally_hidden=True, R_version=subprocess.check_output(
            ["/usr/bin/Rscript", "--version"], text=True, stderr=subprocess.STDOUT).strip(),
        plan=PLAN, result=str(ROOT / "result.md"), initial_cpu_seconds=START_CPU,
        initial_gpu_seconds=START_GPU, task_count=45,
        source_hashes={p: digest(WORKDIR / p) for p in SOURCES})
    manifest_path = ROOT / f"supervisor-{len(list(ROOT.glob('supervisor-*.json')))+1:02d}.json"
    write_json(manifest_path, run_manifest)

    def budget(active=0):
        charged = sum(j.get("wall_seconds", 0) for j in jobs if j["status"] != "running")
        value = dict(cpu_charged_seconds=charged, gpu_charged_seconds=0,
            cpu_reserved_seconds=active*ATTEMPT_CAP, phase_cpu_cap_seconds=CAP,
            remaining_campaign_cpu_seconds=START_CPU-charged,
            remaining_campaign_gpu_seconds=START_GPU, attempts=len(jobs))
        write_json(ROOT / "budget.json", value)
        return charged

    def launch(item):
        attempt = 1 + sum(j["task"]["name"] == item["name"] for j in jobs)
        name = f"{item['name']}_attempt{attempt:02d}"
        directory = ROOT / name
        directory.mkdir()
        if item["mode"] == "verify":
            with (directory / "pairs.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["data_seed", "label", "score", "rescue"])
                writer.writeheader()
                for data, base in ((981080, 1972000), (991080, 1982000)):
                    for label in range(1, 5):
                        keys = [task("learner", data, base+label, label, m)["name"]
                                for m in ("score", "score_qr_rescue")]
                        writer.writerow(dict(data_seed=data, label=label,
                            score=str(Path(completed[keys[0]]["output_directory"]) / "learner.rds"),
                            rescue=str(Path(completed[keys[1]]["output_directory"]) / "learner.rds")))
        command = ["/usr/bin/Rscript", WORKER, str(directory), item["mode"],
                   str(item["data_seed"]), str(item["seed"]), str(item["label"]), item["method"]]
        entry = dict(task=item, attempt=attempt, command=command, environment=ENV,
            git_commit=commit, plan=PLAN, started_utc=utc(), status="running",
            output_directory=str(directory), source_hashes={p: digest(WORKDIR / p) for p in SOURCES},
            manifest_path=str(ROOT / f"{name}-launch.json"))
        jobs.append(entry)
        write_json(Path(entry["manifest_path"]), entry)

        def run():
            started = time.monotonic()
            with (directory / "R.log").open("w") as log:
                try:
                    result = subprocess.run(command, cwd=WORKDIR, env={**os.environ, **ENV},
                        stdout=log, stderr=subprocess.STDOUT, timeout=ATTEMPT_CAP)
                    entry["exit_code"] = result.returncode
                    entry["status"] = "complete" if result.returncode == 0 else "failed"
                except subprocess.TimeoutExpired:
                    entry["status"] = "timeout"
                    entry["exit_code"] = None
            entry["wall_seconds"] = time.monotonic()-started
            entry["finished_utc"] = utc()
            entry["sources_unchanged"] = all(digest(WORKDIR/p) == v for p, v in entry["source_hashes"].items())
            entry["output_hashes"] = {p.name: digest(p) for p in directory.iterdir() if p.is_file()}
            if not entry["sources_unchanged"]:
                entry["status"] = "source_changed"
            write_json(Path(entry["manifest_path"]), entry)
            return entry
        return run

    def execute_stage(items):
        pending = [item for item in items if item["name"] not in completed]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            active = {}
            blocked = False
            while pending or active:
                while pending and len(active) < 2 and not blocked:
                    if any(digest(WORKDIR/p) != v for p, v in frozen.items()):
                        raise RuntimeError("Frozen numerical source changed")
                    if len(jobs) >= 48 or budget(len(active))+(len(active)+1)*ATTEMPT_CAP > CAP:
                        blocked = True
                        break
                    item = pending.pop(0)
                    active[pool.submit(launch(item))] = item
                budget(len(active))
                write_json(ROOT / "checkpoint.json", dict(status="running", updated_utc=utc(),
                    completed=len(completed), active=[i["name"] for i in active.values()],
                    remaining_in_stage=len(pending), next_action="Await workers; inspect stage before advancing"))
                if not active:
                    if pending:
                        raise RuntimeError("Budget/attempt limit reached")
                    break
                done, _ = concurrent.futures.wait(active, timeout=30,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:
                    print(f"Running {len(active)}; completed {len(completed)}/45", flush=True)
                for future in done:
                    item = active.pop(future)
                    result = future.result()
                    print(f"{item['name']}: {result['status']} {result['wall_seconds']:.2f}s", flush=True)
                    if result["status"] == "complete":
                        completed[item["name"]] = result
                    else:
                        blocked = True
                if blocked and not active:
                    raise RuntimeError("Stage stopped for failed attempt or exhausted budget; inspect preserved log")
        budget()

    for stage in stages:
        execute_stage(stage)
    execute_stage([task("verify", 0, 0, 0, "paired_identity")])
    write_json(ROOT / "completed.json", completed)
    budget()
    write_json(ROOT / "checkpoint.json", dict(status="complete_pending_interpretation",
        updated_utc=utc(), completed=len(completed), active=[],
        next_action="Interpret mechanism, cap extensions, healthy identity and conditional controls"))
    print(f"Complete: {len(completed)} tasks. Artifacts: {ROOT}", flush=True)


if __name__ == "__main__":
    main()
