"""Versioned launcher/accounting for the independent iAPF mechanism campaign."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "docs/plans/artifacts/iapf-renewed-mechanism-20260922-01"
PLAN = "docs/plans/iapf-renewed-mechanism-campaign-2026-09-22.md"
OLD = REPO / "docs/plans/artifacts/iapf-r-independent-validation-20260922-01"
REFERENCES = ["docs/benchmarks/reference_iapf_paper.R",
              "docs/benchmarks/reference_iapf_author_choices.R",
              "docs/benchmarks/reference_iapf_constrained_diagnostic.R"]

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage")
    p.add_argument("--driver", default="docs/benchmarks/diagnose_iapf_guide_gap.R")
    p.add_argument("--limit", type=float, default=1800)
    p.add_argument("--input", action="append", default=[])
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("args", nargs="*")
    a = p.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    attempts = sorted(ROOT.glob("attempt*-*/manifest.json"))
    previous = [json.loads(x.read_text()) for x in attempts]
    if any(x["status"] == "running" for x in previous):
        raise RuntimeError("An unfinished attempt must be reconciled before another launch")
    used = sum(x.get("CPU_worker_seconds",x["wall_seconds"]) for x in previous)
    used += sum(json.loads(x.read_text()).get("CPU_worker_seconds",0.)
                for x in ROOT.glob("*tests.json"))
    used += sum(json.loads(x.read_text()).get("CPU_reference_seconds",
                json.loads(x.read_text())["GPU_process_seconds"])
                for x in ROOT.glob("gpu-attempt*/manifest.json"))
    if not 1 <= a.workers <= 4 or not 0 < a.limit*a.workers <= 172800-used:
        raise ValueError("Attempt exceeds remaining CPU allocation")
    frozen = json.loads((OLD/"manifest.json").read_text())["frozen_reference_source_sha256"]
    for name in REFERENCES:
        if sha(REPO/name) != frozen[name]:
            raise RuntimeError(f"Frozen source changed: {name}")
    attempt = ROOT / f"attempt{len(attempts)+1:03d}-{a.stage}"
    attempt.mkdir()
    sources = list(dict.fromkeys(REFERENCES + [a.driver,
        "docs/benchmarks/reference_iapf_guide_diagnostics.R",
        "docs/benchmarks/run_iapf_mechanism_attempt.py"]))
    captured = {}
    for name in sources:
        target = attempt/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target)
        captured[name] = sha(target)
    inputs = [OLD/"manifest.json", OLD/"cells-completion.csv"] + [Path(x).resolve() for x in a.input]
    if a.stage == "saved":
        for row in csv.DictReader((OLD/"cells-completion.csv").open()):
            directory = OLD/row["attempt"]/"results"
            inputs.extend(directory/name for name in
                ("data.rds", "qr-learning.rds", "ridge_bounded1-learning.rds"))
    inputs = {str(x):sha(x) for x in dict.fromkeys(inputs)}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", IAPF_DIAGNOSTIC_WORKERS=str(a.workers))
    command = [shutil.which("Rscript"), "--vanilla", str(attempt/"source"/a.driver),
               a.stage, str(attempt/"results")] + a.args
    if a.stage == "saved":
        command.append(str(OLD))
    manifest = dict(stage=a.stage, status="running", command=command, plan=PLAN,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
        git_branch=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip(),
        dirty_worktree=True, CPU_only=True, GPU_intentionally_hidden=True,
        environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","IAPF_DIAGNOSTIC_WORKERS")},
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), source_sha256=captured,
        input_sha256=inputs, timeout_seconds=a.limit, wall_seconds=0.,
        seeds="Recorded in captured driver and output rows", source_snapshot=str(attempt/"source"),
        data_version="GJL Section 5.2; saved immutable validation or explicitly fresh seeded data",
        R_version=subprocess.check_output(["Rscript","--version"],stderr=subprocess.STDOUT,text=True).strip())
    path = attempt/"manifest.json"
    path.write_text(json.dumps(manifest,indent=2)+"\n")
    start = time.monotonic()
    try:
        with (attempt/"run.log").open("w") as log:
            process = subprocess.Popen(command,cwd=attempt/"source",env=env,stdout=log,
                                       stderr=subprocess.STDOUT,start_new_session=True)
            try:
                returncode = process.wait(timeout=a.limit)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
                raise
        manifest.update(status="complete" if returncode==0 else "failed",
                        returncode=returncode)
    except subprocess.TimeoutExpired:
        manifest.update(status="timeout",returncode=None)
    finally:
        manifest.update(wall_seconds=time.monotonic()-start,
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        manifest["CPU_worker_seconds"] = a.workers*manifest["wall_seconds"]
        manifest["output_sha256"] = {str(x.relative_to(attempt)):sha(x)
            for x in sorted((attempt/"results").rglob("*")) if x.is_file()}
        manifest["remaining_CPU_seconds"] = 172800-used-manifest["CPU_worker_seconds"]
        path.write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({k:manifest[k] for k in ("stage","status","wall_seconds","remaining_CPU_seconds")}) + " " + str(path))
    if manifest["status"] != "complete":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
