"""Bounded local supervisor for an independent R diagnostic campaign."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "docs/plans/artifacts/iapf-r-targeted-fitting-repair-20260922-01"
PLAN = "docs/plans/iapf-r-targeted-fitting-repair-2026-09-22.md"
DEADLINE = dt.datetime.fromisoformat("2026-09-21T20:04:26+00:00")
CAP = 4800.0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments_in):
    stage, timeout_text, *arguments = arguments_in
    ROOT.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(p.read_text()) for p in ROOT.glob("attempt*/manifest.json")]
    if any(p["status"] == "running" for p in prior):
        raise RuntimeError("A previous attempt is still running or needs reconciliation")
    used = sum(p.get("wall_seconds", 0.0) for p in prior)
    remaining_wall = (DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds()
    limit = min(float(timeout_text), CAP - used, remaining_wall)
    if limit <= 0 or len(prior) >= 30:
        raise RuntimeError("Campaign continuation budget/deadline exhausted")
    attempt = ROOT / f"attempt{len(prior)+1:02d}-{stage}"
    attempt.mkdir()
    snapshot = attempt / "source"
    files = [REPO / PLAN, Path(__file__).resolve()]
    files += [REPO / "docs/benchmarks" / name for name in [
        "reference_iapf_paper.R", "reference_iapf_author_choices.R",
        "reference_iapf_constrained_diagnostic.R", "run_iapf_fitting_repair.R"]]
    files += sorted((REPO / "docs/benchmarks").glob("iapf_fitting_repair_*.R"))
    files += [REPO / "tests/reference_iapf_author_choices.R"]
    for path in files:
        destination = snapshot / path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    (snapshot / "docs/plans/artifacts").symlink_to(REPO / "docs/plans/artifacts")
    output = attempt / "results"
    command = ["/usr/bin/Rscript", "--vanilla", str(snapshot / "docs/benchmarks/run_iapf_fitting_repair.R"),
               stage, str(output), *arguments]
    env_fields = dict(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    environment = {**os.environ, **env_fields}
    started = dt.datetime.now(dt.timezone.utc)
    manifest = dict(status="running", stage=stage, git_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        command=command, working_directory=str(snapshot), environment=env_fields,
        CPU_only=True, GPU_intentionally_hidden=True, plan=PLAN,
        result="../result.md", started_utc=started.isoformat(), timeout_seconds=limit,
        source_sha256={str(p.relative_to(REPO)): sha(p) for p in files},
        arguments=arguments, R_version=subprocess.check_output(
            ["Rscript", "--version"], stderr=subprocess.STDOUT, text=True).strip())
    path = attempt / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    t0 = time.monotonic()
    try:
        with (attempt / "command.log").open("w") as log:
            completed = subprocess.run(command, cwd=snapshot, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=limit)
        manifest.update(status="complete" if completed.returncode == 0 else "failed",
                        exit_code=completed.returncode)
    except subprocess.TimeoutExpired:
        manifest.update(status="resource_censored", exit_code=124)
    finally:
        elapsed = time.monotonic() - t0
        manifest.update(wall_seconds=elapsed,
                        finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                        campaign_worker_seconds=used + elapsed,
                        remaining_campaign_seconds=CAP - used - elapsed,
                        cumulative_prior_campaign_seconds=109582.18247977401 + used + elapsed)
        inputs = output / "input-paths.txt"
        if inputs.exists():
            manifest["input_sha256"] = {p: sha(REPO / p) for p in inputs.read_text().splitlines() if p}
        manifest["output_sha256"] = {str(p.relative_to(attempt)): sha(p)
                                     for p in sorted(output.glob("*")) if p.is_file()}
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ["status", "exit_code", "wall_seconds",
                                             "campaign_worker_seconds", "remaining_campaign_seconds"]}
                     | {"attempt": str(attempt.relative_to(REPO))}), flush=True)
    return manifest["exit_code"]


def main():
    if sys.argv[1:] == ["bridge_ladder"]:
        for seed in (92100180, 92100280):
            for replication in range(1, 5):
                code = run(["bridge", "120", str(seed), str(replication)])
                if code not in (0, 124):
                    return code
        return 0
    if sys.argv[1:] == ["ladder"]:
        for dimension in (5, 20, 80):
            for data_index in (1, 2):
                seed = 92200000 + 10000 * data_index + dimension
                for arm in ("qr", "ridge_bounded1"):
                    timeout = 360 if dimension == 80 else 120
                    code = run(["learning", str(timeout), str(dimension), str(seed), arm])
                    if code not in (0, 124):
                        return code
        return 0
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
