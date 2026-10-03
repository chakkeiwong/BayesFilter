"""Resumable, bounded independent CPU R campaign; no production entry point."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from run_iapf_r_replication import snapshot_sources, r_worker_command, run_bounded_worker
import run_iapf_r_fitting_comparison as checks

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"docs/plans/artifacts/iapf-r-24hour-campaign-20260921-01"
PLAN = "docs/plans/iapf-r-24hour-campaign-2026-09-21.md"
WORKER = "docs/benchmarks/compare_iapf_r_fitting.R"


def save(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def read_rows(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_attempts():
    path = OUT/"attempts.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    path = OUT/"manifest.json"
    if path.exists():
        return json.loads(path.read_text())
    ledger = dict(plan=PLAN,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        authorized_start_utc="2026-09-20T20:04:26+00:00",deadline_utc="2026-09-21T20:04:26+00:00",
        wall_budget_seconds=86400,budget_seconds=172800,max_workers=2,launch_limit=10000,
        worker_seconds=0,attempt_count=0,phases={},status="prepared",
        environment=dict(CUDA_VISIBLE_DEVICES="-1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",
        device="independent CPU R reference; GPU intentionally hidden",python=os.sys.executable,
        r_version=subprocess.check_output(["Rscript","--version"],stderr=subprocess.STDOUT,text=True).strip()),
        old_balances_transferred=False,result=str(OUT/"result.md"))
    save(path,ledger)
    return ledger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase_file",type=Path)
    args = parser.parse_args()
    phase = json.loads(args.phase_file.read_text())
    ledger = initialize()
    if ledger.get("inflight"):
        raise RuntimeError("recorded workers require reconciliation before a second supervisor")
    identifier = phase["phase"]
    phase_dir = OUT/identifier
    snapshots = phase_dir/"source-snapshot"
    if identifier not in ledger["phases"]:
        phase_dir.mkdir()
        save(phase_dir/"phase.json",phase)
        sources = [PLAN,phase["plan"],WORKER,"docs/benchmarks/reference_iapf_paper.R",
            "docs/benchmarks/reference_iapf_plausible_choices.R","docs/benchmarks/diagnose_iapf_r_validation_tails.R",
            "docs/benchmarks/run_iapf_r_replication.py","docs/benchmarks/run_iapf_r_fitting_comparison.py",
            "docs/benchmarks/run_iapf_r_campaign_phase.py"]
        sources += [name for name in ("docs/benchmarks/summarize_iapf_r_campaign_phase.py",
                    "docs/benchmarks/supervise_iapf_r_24hour_campaign.py") if (ROOT/name).exists()]
        ledger["phases"][identifier] = dict(spec_sha256=digest(phase_dir/"phase.json"),
            source_hashes=snapshot_sources(ROOT,list(dict.fromkeys(sources)),snapshots),status="prepared")
    elif json.loads((phase_dir/"phase.json").read_text()) != phase:
        raise RuntimeError("phase specification changed")
    for name, expected in ledger["phases"][identifier]["source_hashes"].items():
        if digest(snapshots/name) != expected:
            raise RuntimeError("source snapshot changed")
    checks.OUT = OUT
    attempts = load_attempts()
    ledger["worker_seconds"] = sum(a["worker_seconds"] for a in attempts)
    ledger["attempt_count"] = len(attempts)
    jobs = []
    methods = lambda arm: ["qr","short_qr","bpf","fully_adapted","sis"] if arm=="controller" else ["qr","bpf","fully_adapted","sis"] if arm=="baselines" else [arm]
    for cell in phase["cells"]:
        for first in range(cell["first"],cell["first"]+cell["repeats"],cell.get("chunk_size",1)):
            repeat = min(cell.get("chunk_size",1),cell["first"]+cell["repeats"]-first)
            key = f"{identifier}-{cell['cell_id']}-{first}"
            prior = [a for a in attempts if a["job_key"]==key]
            recorded = []
            for a in prior:
                recorded.extend((r["replication"],r["method"]) for r in read_rows(OUT/a["name"]/"results/replicates.csv"))
            expected = {(str(i),m) for i in range(first,first+repeat) for m in methods(cell["arm"])}
            if len(set(recorded))!=len(recorded) or not set(recorded).issubset(expected):
                raise RuntimeError("duplicate or unexpected resumed evidence")
            if set(recorded)==expected:
                continue
            jobs.append(dict(job_key=key,phase=identifier,cell_id=cell["cell_id"],stage=cell["stage"],
                arm=cell["arm"],dimension=cell["dimension"],data_seed=cell["data_seed"],first=first,repeats=repeat,
                fit_maxit=200,timeout_seconds=cell["timeout_seconds"]*(2**len(prior)),
                completed_before=sorted(recorded),retry_count=len(prior)))
    if phase.get("interleave_cells"):
        jobs.sort(key=lambda job:(job["first"],job["dimension"]))
    deadline = datetime.fromisoformat(ledger["deadline_utc"]).timestamp()
    env = dict(os.environ,CUDA_VISIBLE_DEVICES="-1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")

    def checkpoint(status,active):
        ledger["status"]=status
        ledger["remaining_seconds"]=ledger["budget_seconds"]-ledger["worker_seconds"]
        ledger["remaining_wall_seconds"]=max(0,deadline-time.time())
        ledger["inflight"]=list(active.values())
        save(OUT/"manifest.json",ledger)
        (OUT/"checkpoint.md").write_text(
            "# Active independent R 24-hour campaign\n\n"
            f"Phase: {identifier}. Status: {status}.\nQuestion and criteria: {phase['plan']}.\n"
            f"Worker use {ledger['worker_seconds']:.3f}/172800 seconds; "
            f"{ledger['attempt_count']}/10000 launches; deadline {ledger['deadline_utc']}.\n"
            f"Active {len(active)}; queued {len(jobs)}. Source snapshots and attempts.jsonl preserve evidence.\n"
            "Next: finish this frozen phase, analyze complete cells, then execute the reviewed next phase. "
            "Candidate rejection does not stop the program. Local repairs use the same seeds and data.\n")

    def execute(spec):
        path=OUT/spec["name"]
        path.mkdir(parents=True)
        skip_file=path/"completed-pairs.csv"
        with skip_file.open("w",newline="") as stream:
            writer=csv.writer(stream);writer.writerow(["replication","method"]);writer.writerows(spec["completed_before"])
        spec["command"]=r_worker_command(snapshots,WORKER,path/"results",spec["arm"],spec["dimension"],
            spec["repeats"],spec["first"],spec["data_seed"],spec["fit_maxit"],skip_file)
        spec["started_utc"]=datetime.now(timezone.utc).isoformat()
        started=time.monotonic()
        with (path/"worker.log").open("w") as log:
            try:
                spec["exit_code"]=run_bounded_worker(spec["command"],cwd=ROOT,env=env,log=log,timeout=spec["timeout_seconds"])
            except subprocess.TimeoutExpired:
                spec["exit_code"]=124
                spec["failure_class"]="bounded_worker_timeout"
        spec["worker_seconds"]=time.monotonic()-started
        spec["completed_utc"]=datetime.now(timezone.utc).isoformat()
        for name in ("observations.csv","kalman.csv"):
            if (path/"results"/name).exists(): spec[name+"_sha256"]=digest(path/"results"/name)
        try:
            spec["inspection"]=checks.inspect(spec)
        except Exception as error:
            spec["inspection"]={"eligible":False,"records_ok":False,"error":repr(error)}
            spec["failure_class"]="artifact_inspection_failure"
        save(path/"attempt-result.json",spec)
        return spec

    active={}
    blocked=False
    validity_stop=False
    data_hashes={}
    with ThreadPoolExecutor(max_workers=2) as pool:
        while jobs or active:
            while jobs and len(active)<2 and not blocked:
                spec=jobs[0]
                reserved=sum(s["timeout_seconds"]+1 for s in active.values())
                cap=spec["timeout_seconds"]
                if (cap+1>ledger["budget_seconds"]-ledger["worker_seconds"]
                    or time.time()+cap+1>deadline or ledger["attempt_count"]+len(active)>=10000):
                    blocked=True
                    break
                if cap+1+reserved>ledger["budget_seconds"]-ledger["worker_seconds"]:
                    # Active jobs can finish below their reserved caps. Recheck
                    # after completion rather than declaring a budget stop now.
                    break
                jobs.pop(0)
                number=ledger["attempt_count"]+len(active)+1
                spec["name"]=f"{identifier}/attempt{number:05d}-{spec['cell_id']}-{spec['first']}-r{spec['retry_count']}"
                future=pool.submit(execute,dict(spec))
                active[future]=spec
                checkpoint("running",active)
                print("BEGIN",spec["name"],flush=True)
            if not active: break
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            if not done:
                checkpoint("running",active)
                continue
            for future in done:
                spec=future.result()
                del active[future]
                with (OUT/"attempts.jsonl").open("a") as stream:stream.write(json.dumps(spec,allow_nan=False)+"\n")
                attempts.append(spec)
                ledger["worker_seconds"]+=spec["worker_seconds"]
                ledger["attempt_count"]+=1
                print("END",spec["name"],spec["exit_code"],round(spec["worker_seconds"],3),flush=True)
                observed_hashes=tuple(spec.get(name+"_sha256") for name in ("observations.csv","kalman.csv"))
                if spec["cell_id"] in data_hashes and observed_hashes != data_hashes[spec["cell_id"]]:
                    ledger["continuation_veto"]="data/oracle identity changed within cell"
                    validity_stop=blocked=True
                elif all(observed_hashes):
                    data_hashes[spec["cell_id"]]=observed_hashes
                if not spec["inspection"].get("records_ok",False):
                    ledger["continuation_veto"]="completed records failed mandatory artifact validation"
                    validity_stop=blocked=True
                if spec["exit_code"]==124 and spec["retry_count"]<2:
                    rows=read_rows(OUT/spec["name"]/"results/replicates.csv")
                    done_keys=spec["completed_before"]+[[r["replication"],r["method"]] for r in rows]
                    if len(done_keys)<spec["repeats"]*len(methods(spec["arm"])):
                        retry={k:spec[k] for k in ("job_key","phase","cell_id","stage","arm","dimension","data_seed","first","repeats","fit_maxit")}
                        retry.update(completed_before=done_keys,retry_count=spec["retry_count"]+1,timeout_seconds=2*spec["timeout_seconds"])
                        jobs.append(retry)
                checkpoint("running",active)
    ledger["phases"][identifier]["status"]=("validity boundary" if validity_stop else
        "budget boundary" if jobs else "workers finished; analysis pending")
    checkpoint(ledger["phases"][identifier]["status"],{})


if __name__=="__main__":
    main()
