"""Serial supervisor for the independent, CPU-only R validation campaign."""
from __future__ import annotations

import csv
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
ROOT = REPO / "docs/plans/artifacts/iapf-r-independent-validation-20260922-01"
PLAN = "docs/plans/iapf-r-independent-validation-2026-09-22.md"
PARENT = REPO / "docs/plans/artifacts/iapf-r-targeted-fitting-repair-20260922-01/manifest.json"
DEADLINE = dt.datetime.fromisoformat("2026-09-21T20:04:26+00:00")
CAP = 4100.0
CORES = [f"docs/benchmarks/{name}" for name in (
    "reference_iapf_paper.R", "reference_iapf_author_choices.R",
    "reference_iapf_constrained_diagnostic.R")]
FILES = CORES + [PLAN, "docs/benchmarks/run_iapf_independent_validation.R",
    "docs/benchmarks/run_iapf_independent_validation.py",
    "docs/benchmarks/report_iapf_independent_validation.R"]
ENV = dict(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def manifests():
    return [(p, json.loads(p.read_text())) for p in sorted(ROOT.glob("attempt*/manifest.json"))]


def launch(stage, arguments=(), timeout=120):
    prior = manifests()
    assert not any(m["status"] == "running" for _, m in prior)
    frozen = json.loads(PARENT.read_text())["final_source_sha256"]
    for rel in CORES:
        assert sha(REPO / rel) == frozen[rel], f"Frozen source changed: {rel}"
    used = sum(m.get("wall_seconds", 0.0) for _, m in prior)
    wall_remaining = (DEADLINE - dt.datetime.now(dt.timezone.utc)).total_seconds()
    reserve = 0 if stage == "report" else globals().get("CLOSURE_RESERVE_SECONDS", 120)
    limit = min(timeout, CAP - used, wall_remaining - reserve)
    if limit <= 0 or len(prior) >= 90:
        return None
    attempt = ROOT / f"attempt{len(prior)+1:03d}-{stage}"
    attempt.mkdir()
    source = attempt / "source"
    for rel in FILES:
        destination = source / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, destination)
    output = attempt / "results"
    if stage == "report":
        command = ["/usr/bin/Rscript", "--vanilla", str(source / FILES[-1]),
                   str(ROOT), str(output), str(ROOT / globals().get("CELL_INDEX", "cells.csv"))]
    else:
        command = ["/usr/bin/Rscript", "--vanilla",
                   str(source / "docs/benchmarks/run_iapf_independent_validation.R"),
                   stage, str(output), *map(str, arguments)]
    manifest = dict(status="running", stage=stage, arguments=list(arguments),
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        command=command, working_directory=str(source), environment=ENV,
        CPU_only=True, GPU_intentionally_hidden=True, plan=PLAN,
        result=str(output.relative_to(REPO)),
        started_utc=dt.datetime.now(dt.timezone.utc).isoformat(), timeout_seconds=limit,
        source_sha256={rel: sha(REPO / rel) for rel in FILES},
        input_sha256={str(PARENT.relative_to(REPO)): sha(PARENT)},
        R_version=subprocess.check_output(["/usr/bin/Rscript", "--version"],
                                          stderr=subprocess.STDOUT, text=True).strip())
    if stage == "report":
        for p in sorted(ROOT.glob("attempt*-pair/results/*.csv")):
            manifest["input_sha256"][str(p.relative_to(REPO))] = sha(p)
        cell_index = ROOT / globals().get("CELL_INDEX", "cells.csv")
        manifest["input_sha256"][str(cell_index.relative_to(REPO))] = sha(cell_index)
    path = attempt / "manifest.json"
    write_json(path, manifest)
    start = time.monotonic()
    try:
        with (attempt / "command.log").open("w") as log:
            done = subprocess.run(command, cwd=source, env={**os.environ, **ENV},
                                  stdout=log, stderr=subprocess.STDOUT, timeout=limit)
        manifest.update(status="complete" if done.returncode == 0 else "failed",
                        exit_code=done.returncode)
    except subprocess.TimeoutExpired:
        manifest.update(status="resource_censored", exit_code=124)
    finally:
        manifest["wall_seconds"] = time.monotonic() - start
        manifest["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["campaign_worker_seconds"] = used + manifest["wall_seconds"]
        manifest["remaining_campaign_seconds"] = max(0, CAP-manifest["campaign_worker_seconds"])
        manifest["output_sha256"] = {str(p.relative_to(attempt)): sha(p)
            for p in sorted(output.glob("*")) if p.is_file()}
        write_json(path, manifest)
    print(json.dumps(dict(attempt=attempt.name, stage=stage, arguments=list(arguments),
        status=manifest["status"], worker_seconds=manifest["wall_seconds"],
        used_seconds=manifest["campaign_worker_seconds"])), flush=True)
    return path, manifest


def save_cells(cells):
    with (ROOT / "cells.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)


def main():
    ROOT.mkdir(exist_ok=True)
    if sys.argv[1:] == ["checks"]:
        found = launch("checks", timeout=60)
        return 0 if found and found[1]["status"] == "complete" else 1
    if sys.argv[1:] == ["report"]:
        found = launch("report", timeout=60)
        return 0 if found and found[1]["status"] == "complete" else 1
    if sys.argv[1:] != ["campaign"]:
        raise SystemExit("Use checks, campaign or report")
    prior = manifests()
    assert any(m["stage"] == "checks" and m["status"] == "complete" for _, m in prior)
    cells = [dict(dimension=d, data_index=j, replication=r, status="pending", attempt="")
             for r in range(1, 9) for j in (1, 2) for d in (80, 40, 20, 10, 5)]
    for cell in cells:
        for path, m in prior:
            if m["stage"] == "pair" and m["arguments"] == [cell[k] for k in (
                    "dimension", "data_index", "replication")]:
                cell.update(status=m["status"], attempt=str(path.parent.relative_to(ROOT)))
    save_cells(cells)
    for cell in cells:
        if cell["status"] != "pending":
            continue
        arguments = [cell[k] for k in ("dimension", "data_index", "replication")]
        found = launch("pair", arguments, 180 if cell["dimension"] == 80 else 120)
        if found is None:
            break
        path, m = found
        cell.update(status=m["status"], attempt=str(path.parent.relative_to(ROOT)))
        save_cells(cells)
        if m["status"] == "failed":
            break
    save_cells(cells)
    launch("report", timeout=60)
    all_records = manifests()
    status = "complete" if all(c["status"] == "complete" for c in cells) else "incomplete"
    write_json(ROOT / "execution-summary.json", dict(status=status,
        planned_cells=len(cells), complete_cells=sum(c["status"] == "complete" for c in cells),
        worker_seconds=sum(m["wall_seconds"] for _, m in all_records),
        attempts=len(all_records), deadline_utc=DEADLINE.isoformat(),
        numerical_cap_seconds=CAP, CPU_only=True, GPU_intentionally_hidden=True))
    print(json.dumps(json.loads((ROOT / "execution-summary.json").read_text())), flush=True)
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
