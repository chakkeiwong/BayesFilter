"""Execute bounded thesis-guided iAPF diagnostics; independent CPU reference."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/plans/artifacts/iapf-thesis-reconstruction-20260927-01"
OLD = ROOT / "docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01/attempt03"
PLAN = ROOT / "docs/plans/iapf-thesis-guided-fit-reconstruction-20260927.md"
PRIOR_CPU = 199.100153
CAP = 600.0

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("calibration", "box_repair"), default="calibration")
    args = parser.parse_args()
    prior = [json.loads(p.read_text()) for p in sorted(OUT.glob("attempt*/manifest.json"))]
    if any(m["status"] == "RUNNING" for m in prior):
        raise SystemExit("Unfinished attempt requires investigation")
    if any(m["status"] == "COMPLETE" and m.get("phase", "calibration") == args.phase for m in prior):
        raise SystemExit("Requested phase already completed; preserved evidence will not be overwritten")
    spent = sum(m["wall_seconds"] for m in prior)
    remaining = min(CAP-spent, 1800-PRIOR_CPU-spent)
    if len(prior) >= 3 or remaining < 1:
        raise SystemExit("Campaign attempt/CPU cap exhausted")
    historical = json.loads((OLD/"manifest.json").read_text())
    script_name = ("diagnose_iapf_thesis_reconstruction.R" if args.phase == "calibration"
                   else "diagnose_iapf_thesis_box_repair.R")
    selected = [OLD/"sources/reference_iapf_paper.R",
                OLD/"sources/reference_iapf_scale_aware_fit.R",
                ROOT/"docs/benchmarks"/script_name, Path(__file__).resolve(),
                ROOT/"scripts/run_replication_master.py", PLAN, OUT/"source-recovery.md"]
    old_sources = {Path(k).name: v for k, v in historical["source_sha256"].items()}
    for path in selected[:2]:
        if sha(path) != old_sources[path.name]:
            raise SystemExit("Historical source checksum mismatch: "+str(path))
    inputs = sorted(OLD.glob("*.rds"))
    if len(inputs) != 6:
        raise SystemExit("Expected exactly six saved calibration inputs")
    for path in inputs:
        if sha(path) != historical["output_sha256"][path.name]:
            raise SystemExit("Historical input checksum mismatch: "+str(path))
    previous = None
    if args.phase == "box_repair":
        calibration = [m for m in prior if m["status"] == "COMPLETE"
                       and m.get("phase", "calibration") == "calibration"]
        if len(calibration) != 1:
            raise SystemExit("Exactly one completed calibration is required")
        cal = calibration[0]
        previous = Path(cal["output_root"])
        for relative, digest in cal["output_sha256"].items():
            if sha(previous/relative) != digest:
                raise SystemExit("Frozen calibration checksum mismatch: "+relative)
        with (previous/"decision.csv").open() as stream:
            decision = next(csv.DictReader(stream))
        if (decision["numerical_controls"] != "TRUE" or decision["nominated_arms"] != "0"
                or decision["bounded_fits"] != "8" or decision["constrained_stationary"] != "0"):
            raise SystemExit("Box-repair prerequisites do not match the reviewed plan")
        inputs += [previous/"controls.rds", previous/"heuristic_starts.csv"]
        inputs += sorted(previous.glob("*.profiled.scaled.tight.[24].rds"))
        if len(inputs) != 16:
            raise SystemExit("Expected six original inputs, one control, one heuristic table and eight fits")
    paper_paths = [ROOT/".localresources/papers/guarniero-2017-iapf-thesis.pdf",
                   ROOT/".localresources/papers/guarniero-2017-iapf-thesis.txt",
                   ROOT/".localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt"]
    attempt = OUT / f"attempt{len(prior)+1:02d}"
    sources = attempt/"sources"
    sources.mkdir(parents=True, exist_ok=False)
    for path in selected:
        shutil.copy2(path, sources/path.name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = ["/usr/bin/Rscript", "--vanilla", str(sources/script_name),
               str(attempt), str(OLD), str(sources)]
    if previous is not None:
        command.append(str(previous))
    manifest = dict(status="RUNNING", phase=args.phase, start_utc=now(), controller_pid=os.getpid(),
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        command=command, python_executable=sys.executable,
        environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
        R_version=subprocess.check_output(["/usr/bin/Rscript","--version"],stderr=subprocess.STDOUT,text=True).strip(),
        hardware="Independent R/FP64 CPU reference; GPU deliberately hidden",gpu_seconds=0,
        data_version="Frozen d5 T100 N1000 attempt03; calibration permits conditional fresh draws",
        seeds=dict(control=9271001,calibration=[9262001,9262002],fresh=[9274001,9274002],
                   fresh_APF=[9274101,9274102]),
        source_sha256={str(p):sha(p) for p in selected},
        input_sha256={str(p):sha(p) for p in inputs},
        paper_sha256={str(p):sha(p) for p in paper_paths},
        historical_manifest_sha256=sha(OLD/"manifest.json"),
        previous_manifest_sha256=sha(previous/"manifest.json") if previous else None,
        prior_combined_cpu_seconds=PRIOR_CPU,prior_phase_cpu_seconds=spent,
        total_phase_cpu_cap=CAP,timeout_seconds=remaining,plan=str(PLAN),
        result=str(OUT/"result.md"),output_root=str(attempt))
    dest=attempt/"manifest.json"
    def write():
        dest.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n")
    write()
    started=time.monotonic()
    try:
        with (attempt/"run.log").open("w") as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            manifest["worker_pid"]=process.pid
            write()
            try:
                code=process.wait(timeout=remaining)
                manifest.update(status="COMPLETE" if code==0 else "FAILED",exit_code=code)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                manifest.update(status="TIMEOUT",exit_code=process.returncode)
    except Exception as exc:
        manifest.update(status="FAILED",exit_code=None,error=str(exc))
        raise
    finally:
        manifest["wall_seconds"]=time.monotonic()-started
        manifest["end_utc"]=now()
        manifest["combined_cpu_seconds"]=PRIOR_CPU+spent+manifest["wall_seconds"]
        manifest["output_sha256"]={str(p.relative_to(attempt)):sha(p) for p in attempt.rglob("*")
                                   if p.is_file() and p!=dest}
        write()
    print(json.dumps({k:manifest[k] for k in ("status","phase","exit_code","wall_seconds","combined_cpu_seconds")}))
    print(str(dest))
    if manifest["status"]!="COMPLETE":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
