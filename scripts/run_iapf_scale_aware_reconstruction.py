"""Bounded CPU-only R reference campaign; no production promotion authority."""
import argparse
import csv
import datetime
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
REF = ROOT / ".localresources/worktrees/ledh-iapf-execution-20260922"
PLAN = ROOT / "docs/plans/iapf-scale-aware-reconstruction-20260926.md"
CAMPAIGN = ROOT / "docs/plans/artifacts/iapf-scale-aware-reconstruction-20260926-01"
BENCH = ROOT / "docs/benchmarks"
SOURCES = [BENCH / "reference_iapf_scale_aware_fit.R",
           BENCH / "diagnose_iapf_scale_aware_reconstruction.R",
           REF / "docs/benchmarks/reference_iapf_paper.R"]

def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def git(path, *args):
    return subprocess.check_output(["git", *args], cwd=path, text=True).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["controls", "backward", "repair", "pilot"], required=True)
    parser.add_argument("--timeout", type=int, default=400)
    args = parser.parse_args()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(p.read_text()) for p in sorted(CAMPAIGN.glob("attempt*/manifest.json"))]
    if any(m["status"] == "RUNNING" for m in previous):
        raise SystemExit("Unfinished attempt must be investigated before retry")
    if len(previous) >= 6:
        raise SystemExit("Six-launch cap reached")
    spent = sum(m.get("wall_seconds", 0) for m in previous)
    timeout = min(args.timeout, 1500 - spent)
    if timeout < 1:
        raise SystemExit("CPU budget exhausted")
    dependency = None
    if args.phase != "controls":
        needed = "controls" if args.phase == "backward" else "backward"
        repair_attempts = [m for m in previous if m["phase"] == "repair" and m["status"] == "COMPLETED"]
        if args.phase == "backward" and repair_attempts:
            needed = "repair"
        eligible = [m for m in previous if m["phase"] == needed and m["status"] == "COMPLETED"]
        if not eligible:
            raise SystemExit("Missing successful prerequisite phase: " + needed)
        parent = eligible[-1]
        changed = [path for path in SOURCES if parent["source_sha256"][str(path)] != sha(path)]
        if changed and args.phase != "repair":
            # A no-nominee repair may proceed only to baseline-only reporting.
            baseline_only = False
            if args.phase == "pilot" and repair_attempts:
                repair = repair_attempts[-1]
                with (pathlib.Path(repair["output_root"]) / "repair_decision.csv").open() as stream:
                    decision = next(csv.DictReader(stream))
                with (pathlib.Path(parent["output_root"]) / "backward_outcomes.csv").open() as stream:
                    rows = list(csv.DictReader(stream))
                baseline_only = (decision["nominee_count"] == "0"
                    and all(r["status"] == "fit_failed" for r in rows)
                    and all(repair["source_sha256"][str(p)] == sha(p) for p in SOURCES))
            if not baseline_only:
                raise SystemExit("Source changed since prerequisite; rerun qualification")
        dependency = pathlib.Path(parent["output_root"])
    attempt = CAMPAIGN / f"attempt{len(previous)+1:02d}"
    attempt.mkdir(exist_ok=False)
    sources = attempt / "sources"
    sources.mkdir()
    hashes = {}
    for path in [*SOURCES, PLAN, pathlib.Path(__file__).resolve()]:
        shutil.copy2(path, sources / path.name)
        hashes[str(path)] = sha(path)
    inputs = {str(p): sha(p) for p in dependency.glob("*.csv")} if dependency else {}
    if args.phase == "repair":
        control = [m for m in previous if m["phase"] == "controls" and m["status"] == "COMPLETED"][-1]
        control_input = pathlib.Path(control["output_root"]) / "controls.rds"
        shutil.copy2(control_input, sources / "recovery_inputs.rds")
        inputs[str(control_input)] = sha(control_input)
        for p in dependency.glob("min_mass_*.rds"):
            inputs[str(p)] = sha(p)
    command = ["/usr/bin/Rscript", "--vanilla",
               str(sources / "diagnose_iapf_scale_aware_reconstruction.R"),
               args.phase, str(attempt), str(dependency or "none")]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    manifest = dict(phase=args.phase, status="RUNNING", start_utc=stamp(), command=command,
        git_commit=git(ROOT, "rev-parse", "HEAD"), branch=git(ROOT, "branch", "--show-current"),
        reference_git_commit=git(REF, "rev-parse", "HEAD"), reference_branch=git(REF, "branch", "--show-current"),
        source_sha256=hashes, input_sha256=inputs, dependency=str(dependency) if dependency else None,
        environment={k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        interpreter=subprocess.check_output(["/usr/bin/Rscript", "--version"], stderr=subprocess.STDOUT, text=True).strip(),
        hardware="Independent FP64 CPU reference; GPU deliberately hidden", GPU_seconds=0,
        controller_pid=os.getpid(), CPU_wall_limit=timeout, total_CPU_wall_limit=1500,
        previous_CPU_wall_seconds=spent, prior_combined_CPU_wall_seconds=89.936265,
        data="Paper d=5 T=100 linear Gaussian generator; seeds bind the generated data",
        seeds=dict(controls="9261001:9261008; 9261099; 9261100", calibration=[9262001, 9262002],
                   untouched=9263001, backward=[9264001, 9264002], adaptive=9264000, evaluation="9265001:9265008"),
        plan=str(PLAN), result=str(attempt / "result.md"), output_root=str(attempt))
    write(attempt / "manifest.json", manifest)
    started = time.monotonic()
    code = None
    with (attempt / "worker.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        manifest["worker_pid"] = proc.pid
        write(attempt / "manifest.json", manifest)
        try:
            code = proc.wait(timeout=timeout)
            status = "COMPLETED" if code == 0 else "FAILED"
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            code = proc.returncode
            status = "TIMEOUT"
    manifest.update(status=status, exit_code=code, end_utc=stamp(), wall_seconds=time.monotonic()-started)
    manifest["output_sha256"] = {str(p.relative_to(attempt)): sha(p) for p in attempt.iterdir() if p.is_file() and p.name not in ("manifest.json", "result.md")}
    write(attempt / "manifest.json", manifest)
    with (CAMPAIGN / "attempts.jsonl").open("a") as stream:
        stream.write(json.dumps(manifest) + "\n")
    summary = "No phase status emitted. See worker.log."
    phase_file = attempt / "phase_status.csv"
    if phase_file.exists():
        with phase_file.open() as stream:
            summary = str(list(csv.DictReader(stream)))
    (attempt / "result.md").write_text(f"# {args.phase} attempt\n\n{status}; exit {code}; {manifest['wall_seconds']:.6f} CPU wall seconds; GPU 0.\n\n{summary}\n\nIndependent R reference only. See manifest.json for exact provenance and the campaign result for interpretation.\n")
    print(json.dumps({k: manifest[k] for k in ("phase", "status", "exit_code", "wall_seconds", "output_root")}), flush=True)
    if status != "COMPLETED":
        raise SystemExit(124 if status == "TIMEOUT" else (code or 1))

if __name__ == "__main__":
    main()
