"""Four bounded CPU reference follow-ups; preserves the original campaign."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from run_iapf_r_replication import snapshot_sources, r_worker_command, run_bounded_worker
import run_iapf_r_plausible_choices as reporting
from run_iapf_r_replication_gap_audit import ROOT, OUT, PLAN, WORKER, DIAGNOSTIC


def main():
    ledger = json.loads((OUT / "manifest.json").read_text())
    assert len(ledger["attempts"]) == 6
    assert ledger["status"] == "workers finished; source audit and terminal review pending"
    jobs = [
        dict(name="attempt07-fit-sensitivity-fixed-scale", mode="fit", timeout_seconds=120),
        dict(name="attempt08-pre-doubling-guides", mode="pre", timeout_seconds=400),
        dict(name="attempt09-validation-d40-completion", dimension=40, first=1828,
             repeats=5, data_seed=87000040, baselines=True, timeout_seconds=200),
        dict(name="attempt10-d20-mean-confirmation", dimension=20, first=1901,
             repeats=64, data_seed=87000020, baselines=False, timeout_seconds=400)]
    assert sum(j["timeout_seconds"] for j in jobs) <= ledger["remaining_worker_seconds"]
    snapshot = OUT / "repair-source-snapshot"
    snapshot.mkdir(exist_ok=False)
    names = list(ledger["source_hashes"]) + [
        "docs/benchmarks/continue_iapf_r_replication_gap_audit.py",
        str((OUT / "repair-note.md").relative_to(ROOT))]
    hashes = snapshot_sources(ROOT, names, snapshot)
    # The mean and completion jobs retain the original numerical core/worker.
    for name in ("docs/benchmarks/reference_iapf_paper.R", WORKER,
                 "docs/benchmarks/diagnose_iapf_r_validation_tails.R"):
        assert hashes[name] == ledger["source_hashes"][name]
    ledger["repair_source_hashes"] = hashes
    ledger["repair_driver_command"] = [
        "/home/chakwong/anaconda3/envs/tftwogpu/bin/python", str(Path(__file__).resolve())]
    ledger["repair_started_utc"] = datetime.now(timezone.utc).isoformat()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    reporting.OUT = OUT

    def checkpoint():
        ledger["worker_seconds"] = sum(a["worker_seconds"] for a in ledger["attempts"])
        ledger["remaining_worker_seconds"] = ledger["budget_seconds"] - ledger["worker_seconds"]
        reporting.write_json(OUT / "manifest.json", ledger)
        (OUT / "checkpoint.md").write_text(
            f"# iAPF replication-gap audit\n\nPlan: {PLAN}\n\n"
            f"{ledger['status']}. Completed launches: {len(ledger['attempts'])}/10; "
            f"worker seconds: {ledger['worker_seconds']:.3f}/2400. "
            "Outstanding workers retain their full reservations.\n")

    def execute(spec):
        spec = dict(spec)
        directory = OUT / spec["name"]
        directory.mkdir(exist_ok=False)
        if "mode" in spec:
            runner, args = DIAGNOSTIC, [spec["mode"]]
        else:
            spec.update(stage="validation", arm="delayed")
            runner, args = WORKER, ["delayed", spec["dimension"], spec["repeats"],
                spec["first"], spec["data_seed"], "yes" if spec["baselines"] else "no"]
        command = r_worker_command(snapshot, runner, directory / "results", *args)
        spec.update(command=command, source_snapshot=str(snapshot.relative_to(ROOT)),
                    started_utc=datetime.now(timezone.utc).isoformat())
        reporting.write_json(directory / "launch.json", spec)
        print("BEGIN", spec["name"], flush=True)
        started = time.monotonic()
        with (directory / "run.log").open("w") as log:
            try:
                spec["exit_code"] = run_bounded_worker(command, cwd=ROOT, env=env,
                    log=log, timeout=spec["timeout_seconds"])
            except subprocess.TimeoutExpired:
                spec["exit_code"] = 124
                spec["failure_class"] = "bounded_worker_timeout"
        spec["worker_seconds"] = time.monotonic() - started
        spec["completed_utc"] = datetime.now(timezone.utc).isoformat()
        print("END", spec["name"], spec["exit_code"], round(spec["worker_seconds"], 3), flush=True)
        return spec

    ledger["status"] = "bounded repairs and one fixed confirmation batch running"
    checkpoint()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute, spec) for spec in jobs]
        for future in as_completed(futures):
            ledger["attempts"].append(future.result())
            checkpoint()
    ledger["status"] = "all planned workers finished; terminal analysis pending"
    ledger["finished_utc"] = datetime.now(timezone.utc).isoformat()
    checkpoint()
    print("COMPLETE", ledger["worker_seconds"], flush=True)


if __name__ == "__main__":
    main()
