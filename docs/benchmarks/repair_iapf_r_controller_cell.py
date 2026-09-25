"""Complete a timed-out CPU R reference cell within its original allocation."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_iapf_r_replication import snapshot_sources, r_worker_command, run_bounded_worker
from run_iapf_r_fitting_comparison import ROOT, OUT, WORKER, inspect, read_csv, write_json


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ledger = json.loads((OUT/"manifest.json").read_text())
    if ledger.get("inflight") or ledger.get("controller_repair"):
        raise RuntimeError("another launch or repair is already recorded")
    original = next(a for a in ledger["attempts"] if a["name"] == "attempt14-controller-89400080")
    if original["exit_code"] != 124 or original["inspection"]["failed_rows"]:
        raise RuntimeError("repair is restricted to the recorded resource timeout")
    for section, directory in ((ledger,"source-snapshot"),(ledger["followthrough"],"followthrough-source-snapshot")):
        for name, expected in section["source_hashes"].items():
            if digest(OUT/directory/name) != expected:
                raise RuntimeError("previous evidence source snapshot changed")
    algorithm_sources = ["docs/benchmarks/reference_iapf_paper.R",
                         "docs/benchmarks/reference_iapf_plausible_choices.R",
                         "docs/benchmarks/diagnose_iapf_r_validation_tails.R"]
    for name in algorithm_sources:
        if digest(ROOT/name) != ledger["followthrough"]["source_hashes"][name]:
            raise RuntimeError("repair changed the scientific computation")
    old_path = OUT/original["name"]/"results"
    old_rows = read_csv(old_path/"replicates.csv")
    done = {(r["replication"],r["method"]) for r in old_rows if r["status"] == "complete"}
    expected = {(str(i),m) for i in range(2501,2509)
                for m in ("qr","short_qr","bpf","fully_adapted","sis")}
    if len(done) != len(old_rows) or not done < expected or not original["inspection"]["records_ok"]:
        raise RuntimeError("completed evidence is invalid or no work is missing")
    cap = min(400,int(ledger["remaining_seconds"])-1)
    if cap < 1 or len(ledger["attempts"]) >= ledger["launch_limit"]:
        raise RuntimeError("original allocation exhausted")
    number = len(ledger["attempts"])+1
    spec = dict(name=f"attempt{number:02d}-controller-completion-89400080",stage="controller_repair",
                arm="controller",dimension=80,repeats=8,first=2501,data_seed=89400080,fit_maxit=200,
                timeout_seconds=cap,completed_before=sorted(done))
    path = OUT/spec["name"]
    path.mkdir()
    skips = path/"completed-pairs.csv"
    write_rows(skips,[dict(replication=i,method=m) for i,m in sorted(done)])
    note = ("# Controller timeout repair\n\n"
            f"Attempt14 completed {len(done)}/40 method/replica pairs without a numerical error. "
            f"Resume only the {40-len(done)} missing pairs, with {cap} seconds reserved from "
            f"the remaining {ledger['remaining_seconds']:.6f} seconds; no new allocation.\n\n"
            "Skeptical audit: PASS. Every method resets its recorded RNG seed. The original "
            "algorithm, data, particle counts, criteria and CPU environment are unchanged. "
            "The added skip list affects scheduling only. Verify data hashes and complete unique "
            "40-pair coverage before interpreting the combined cell. Filter all diagnostic tables "
            "by complete method/replica pairs, so interrupted records cannot become evidence. "
            "Preserve both attempts and charge all elapsed worker time, including lost work. "
            "Timeout/incomplete coverage blocks this cell's promotion, not the research direction.\n")
    (path/"repair-note.md").write_text(note)
    sources = algorithm_sources+[WORKER,ledger["plan"],ledger["followthrough"]["plan"],
        "docs/benchmarks/run_iapf_r_replication.py","docs/benchmarks/run_iapf_r_fitting_comparison.py",
        "docs/benchmarks/repair_iapf_r_controller_cell.py","docs/benchmarks/summarize_iapf_r_fitting_comparison.py"]
    snapshots = OUT/"repair-source-snapshot"
    if snapshots.exists():
        raise RuntimeError("repair snapshot already exists")
    hashes = snapshot_sources(ROOT,sources,snapshots)
    ledger["controller_repair"] = dict(source_hashes=hashes,completed_pairs_sha256=digest(skips),
                                      reason="bounded_worker_timeout",original_allocation_unchanged=True)
    spec["command"] = r_worker_command(snapshots,WORKER,path/"results","controller",80,8,2501,89400080,200,skips)
    spec["started_utc"] = datetime.now(timezone.utc).isoformat()
    ledger["inflight"] = [spec]
    ledger["status"] = "controller missing-pair repair running"
    write_json(OUT/"manifest.json",ledger)
    (OUT/"checkpoint.md").write_text(note+"\nNext: finish the repair, validate the combined cell, then terminal analysis.\n")
    env = dict(os.environ,CUDA_VISIBLE_DEVICES="-1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1")
    print("BEGIN",spec["name"],"missing",40-len(done),"cap",cap,flush=True)
    started = time.monotonic()
    with (path/"worker.log").open("w") as log:
        try:
            spec["exit_code"] = run_bounded_worker(spec["command"],cwd=ROOT,env=env,log=log,timeout=cap)
        except subprocess.TimeoutExpired:
            spec["exit_code"] = 124
            spec["failure_class"] = "bounded_worker_timeout"
    spec["worker_seconds"] = time.monotonic()-started
    spec["completed_utc"] = datetime.now(timezone.utc).isoformat()
    spec["inspection"] = inspect(spec)
    ledger["attempts"].append(spec)
    ledger.pop("inflight")
    ledger["worker_seconds"] = sum(a["worker_seconds"] for a in ledger["attempts"])
    ledger["remaining_seconds"] = ledger["budget_seconds"]-ledger["worker_seconds"]
    ledger["status"] = "controller repair finished; combined-cell validation pending"
    write_json(OUT/"manifest.json",ledger)
    new_path = path/"results"
    for name in ("observations.csv","kalman.csv"):
        if digest(new_path/name) != digest(old_path/name):
            raise RuntimeError("repair data/oracle changed")
        spec[name+"_sha256"] = digest(new_path/name)
    new_rows = read_csv(new_path/"replicates.csv")
    combined = old_rows+new_rows
    keys = [(r["replication"],r["method"]) for r in combined]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate repaired method/replica")
    derived = dict(name="derived-controller-89400080",stage="controller",arm="controller",
                   dimension=80,repeats=8,first=2501,data_seed=89400080,fit_maxit=200,
                   exit_code=0 if set(keys)==expected and all(r["status"]=="complete" for r in combined) else 124,
                   parent_attempts=[original["name"],spec["name"]],derived_without_additional_worker=True)
    destination = OUT/derived["name"]/"results"
    destination.mkdir(parents=True)
    write_rows(destination/"replicates.csv",combined)
    provenance = {}
    for name in ("prefixes.csv","fits.csv","tails.csv"):
        rows = []
        for source, records in ((old_path,old_rows),(new_path,new_rows)):
            complete = {(r["replication"],r["method"]) for r in records if r["status"]=="complete"}
            rows.extend(r for r in read_csv(source/name) if (r["replication"],r["method"]) in complete)
            if (source/name).exists():
                provenance[str((source/name).relative_to(OUT))] = digest(source/name)
        write_rows(destination/name,rows)
    for name in ("observations.csv","kalman.csv","settings.R"):
        shutil.copyfile(old_path/name,destination/name)
    for source,records in ((old_path,old_rows),(new_path,new_rows)):
        provenance[str((source/"replicates.csv").relative_to(OUT))] = digest(source/"replicates.csv")
        for row in records:
            if row["status"]=="complete" and row["method"] in ("qr","short_qr"):
                file = source/f"{row['method']}-{row['replication']}.rds"
                shutil.copyfile(file,destination/file.name)
                provenance[str(file.relative_to(OUT))] = digest(file)
    derived["inspection"] = inspect(derived)
    write_json(destination.parent/"merge-provenance.json",dict(source_hashes=provenance,
               parents=derived["parent_attempts"],complete_pair_filter=True,inspection=derived["inspection"]))
    original["used_in_derived"] = derived["name"]
    spec["used_in_derived"] = derived["name"]
    ledger["derived_cells"] = [derived]
    ledger["status"] = "controller cell assembled; terminal analysis pending"
    write_json(OUT/"manifest.json",ledger)
    (OUT/"checkpoint.md").write_text(note+f"\nRepair elapsed {spec['worker_seconds']:.6f}; "
        f"total used {ledger['worker_seconds']:.6f}/2400; remaining {ledger['remaining_seconds']:.6f}. "
        f"Combined inspection: {derived['inspection']}.\nNext: terminal analysis, result review and master update.\n")
    print("END",spec["exit_code"],spec["worker_seconds"],derived["inspection"],flush=True)


if __name__ == "__main__":
    main()
