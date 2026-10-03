"""Bounded local supervisor for the explicitly independent R reference campaign."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/plans/artifacts/iapf-r-equation15-resolution-20260921-01"
PLAN = ROOT / "docs/plans/iapf-r-equation15-resolution-2026-09-21.md"
DEADLINE = datetime.fromisoformat("2026-09-21T20:04:26+00:00").timestamp()
BUDGET = 10000
PREVIOUS = 107291.71576361393
ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
LOCK = threading.Lock()
RESERVED = 0


def rows(p):
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def attempts():
    p = OUT / "attempts.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def manifest(status):
    entries = attempts()
    used = sum(x["wall_seconds"] for x in entries)
    data = dict(plan=str(PLAN), result=str(OUT / "result.md"), status=status,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        environment={k:ENV[k] for k in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")},
        cpu_only=True, R_version="4.1.2", original_deadline_utc="2026-09-21T20:04:26+00:00",
        previous_worker_seconds=PREVIOUS, local_budget_seconds=BUDGET,
        worker_seconds=used, aggregate_campaign_seconds=PREVIOUS+used,
        remaining_seconds=min(BUDGET-used,172800-PREVIOUS-used),
        attempts=len(entries), max_workers=2, reserved_seconds=RESERVED,
        updated_utc=datetime.now(timezone.utc).isoformat(), source_snapshot=str(OUT / "source-v1"))
    p = OUT / "manifest.json"
    temporary = p.with_suffix(".tmp")
    temporary.write_text(json.dumps(data,indent=2)+"\n")
    temporary.replace(p)
    return data


def snapshot():
    destination = OUT / "source-v1"
    if destination.exists():
        hashes = json.loads((destination / "sha256.json").read_text())
        for name, expected in hashes.items():
            if digest(destination / name) != expected:
                raise RuntimeError("source snapshot changed: "+name)
        return destination
    files = ["docs/benchmarks/"+name for name in (
        "reference_iapf_paper.R", "reference_iapf_plausible_choices.R",
        "reference_iapf_eq15_solvers.R", "diagnose_iapf_r_validation_tails.R",
        "run_iapf_eq15_resolution.R", "compare_iapf_r_fitting.R",
        "run_iapf_eq15_resolution.py")]
    files += [str(PLAN.relative_to(ROOT)), "tests/reference_iapf_eq15_solvers.R"]
    for name in files:
        target = destination / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT / name,target)
    (destination / "sha256.json").write_text(json.dumps({f:digest(destination/f) for f in files},indent=2)+"\n")
    return destination


def run_job(spec):
    global RESERVED
    job_id, mode, d, seed, arm, rep, maxit, cap = spec
    with LOCK:
        existing = [x for x in attempts() if x.get("job_id")==job_id]
        if existing:
            return existing[-1]
        current = manifest("executing "+mode)
        available = min(current["remaining_seconds"]-RESERVED, DEADLINE-time.time()-5)
        if cap>available or current["attempts"]>=200:
            return dict(job_id=job_id,status="budget_stop",wall_seconds=0)
        RESERVED += cap
        parent = OUT / "runs" / job_id
        parent.mkdir(parents=True,exist_ok=False)
        source = OUT / "source-v1"
        if mode=="timing":
            cmd=["Rscript",str(source/"docs/benchmarks/compare_iapf_r_fitting.R"),
                str(source),str(parent/"results"),"baselines",str(d),"2",str(rep),str(seed),"200"]
        else:
            cmd=["Rscript",str(source/"docs/benchmarks/run_iapf_eq15_resolution.R"),
                str(source),str(parent/"results"),mode,str(d),str(seed),arm,str(rep),str(maxit)]
        entry=dict(job_id=job_id,mode=mode,dimension=d,data_seed=seed,arm=arm,
            replication=rep,maxit=maxit,command=cmd,timeout_seconds=cap,
            started_utc=datetime.now(timezone.utc).isoformat(),output=str(parent/"results"))
        (parent/"launch.json").write_text(json.dumps(entry,indent=2)+"\n")
    started=time.monotonic()
    try:
        with (parent/"worker.log").open("w") as log:
            process=subprocess.run(cmd,cwd=ROOT,env=ENV,stdout=log,stderr=subprocess.STDOUT,timeout=cap)
        entry.update(exit_code=process.returncode,status="finished" if process.returncode==0 else "worker_error")
    except subprocess.TimeoutExpired:
        entry.update(exit_code=None,status="timeout")
    finally:
        entry["wall_seconds"]=time.monotonic()-started
        entry["ended_utc"]=datetime.now(timezone.utc).isoformat()
        with LOCK:
            RESERVED -= cap
            with (OUT/"attempts.jsonl").open("a") as f:
                f.write(json.dumps(entry)+"\n")
            manifest("workers executing" if RESERVED else "stage workers finished")
    print(json.dumps({k:entry[k] for k in ("job_id","status","wall_seconds")}),flush=True)
    return entry


def batch(jobs, workers=2):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        output=list(pool.map(run_job,jobs))
    return output


def probe_jobs(arms, stage, d, seed, reps):
    return [(f"{stage}-d{d}-{seed}-{arm}-r{rep}","probe",d,seed,arm,rep,200,
             180 if arm not in ("qr","bpf","fully_adapted","sis") else 60)
            for rep in reps for arm in arms]


def repair_limits(entries):
    repairs=[]
    for e in entries:
        if e["status"]!="finished":
            continue
        failure=rows(Path(e["output"])/"failure.csv")
        if failure and "iteration limit" in failure[0].get("optimizer_message","").lower():
            repairs.append((e["job_id"]+"-maxit600","probe",e["dimension"],e["data_seed"],
                e["arm"],e["replication"],600,300))
    return batch(repairs) if repairs else []


def valid_arm(arm):
    required={(d,rep) for d in (5,20) for rep in (101,102)}
    present=set()
    for e in attempts():
        if e.get("arm")!=arm or e.get("mode")!="probe" or e.get("data_seed")!=91210000+e["dimension"]:
            continue
        if e["status"]!="finished":
            continue
        record=rows(Path(e["output"])/"replicates.csv")
        if len(record)==1 and record[0]["status"]=="complete" and record[0]["tail_pass"]=="TRUE":
            present.add((e["dimension"],e["replication"]))
    return required<=present


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("fixtures","probes","validation","timing","all"))
    args=parser.parse_args(); snapshot()
    design=dict(fixture_data_seeds=[91110000+d+1000*s for s in (1,2) for d in (5,20,80)],
        probe_data_seeds=[91210005,91210020],probe_replications=[101,102],
        validation_data_seeds=[91310020,91320020],validation_replications=list(range(201,217)),
        timing_data_seeds=[91410005,91410020],timing_replications=[301,302],
        oracle_data_seed=89800080,oracle_replications=list(range(1,17)),
        frozen_before_launch=True,selection_order=["joint_qr","profiled_qr","joint_moments","profiled_moments"])
    design_path=OUT/"design.json"
    if design_path.exists():
        assert json.loads(design_path.read_text())==design
    else:
        design_path.write_text(json.dumps(design,indent=2)+"\n")
    if args.phase in ("fixtures","all"):
        jobs=[(f"fixtures-d{d}-data{s}","fixtures",d,91110000+d+1000*s,"all",s,200,180)
            for s in (1,2) for d in (5,20,80)]
        jobs.append(("exact-oracle-d80","oracle",80,89800080,"exact",1,200,180))
        batch(jobs)
    if args.phase in ("probes","all"):
        jobs=[]
        for d in (5,20):
            jobs+=probe_jobs(["joint_qr","profiled_qr","qr","bpf","fully_adapted","sis"],
                "probe",d,91210000+d,(101,102))
        result=batch(jobs); repair_limits(result)
        if not any(valid_arm(arm) for arm in ("joint_qr","profiled_qr")):
            repair=[]
            for d in (5,20):
                repair+=probe_jobs(["joint_moments","profiled_moments"],"repair",d,91210000+d,(101,102))
            repair_limits(batch(repair))
    if args.phase in ("validation","all"):
        selected=next((arm for arm in design["selection_order"] if valid_arm(arm)),None)
        selection=dict(selected=selected,criterion="complete d5/d20 probes and recorded tail checks; deterministic order, not ranking")
        (OUT/"selection.json").write_text(json.dumps(selection,indent=2)+"\n")
        if selected:
            jobs=[]
            for seed in design["validation_data_seeds"]:
                jobs+=probe_jobs([selected,"qr","bpf","fully_adapted","sis"],"validation",20,seed,range(201,217))
            batch(jobs)
    if args.phase in ("timing","all"):
        batch([(f"timing-d{d}","timing",d,91410000+d,"baselines",301,200,120) for d in (5,20)],workers=1)
    manifest("requested stages finished; final analysis pending")


if __name__=="__main__":
    main()
