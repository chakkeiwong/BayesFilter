"""Bounded CPU-only author selection diagnostic supervisor."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE = BASE / "native-selection-20261013"
PLAN = "docs/plans/controlled-smc-kdm-native-selection-plan-20261013.md"
RESULT = "docs/plans/controlled-smc-kdm-native-selection-result-20261013.md"
PY = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    tmp.replace(path)


def main():
    sys.path.insert(0, str(ROOT / "docs/benchmarks"))
    from run_replication_gap_localization import campaign
    balance = campaign()
    assert not balance["running"]
    prior = list(PHASE.glob("attempt*/manifest.json"))
    assert len(prior) < 2 and all(read(p)["status"] != "RUNNING" for p in prior)
    remaining = min(3600-sum(read(p)["resource_wall_seconds"]["cpu"] for p in prior),
        balance["remaining_seconds"]["cpu"], balance["tranche_remaining_seconds"]["cpu"])
    assert remaining > 180
    out = PHASE / ("attempt%02d" % (len(prior)+1))
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), ROOT / PLAN, ROOT / "docs/benchmarks/reference_kdm_native_selection.py",
        ROOT / "docs/benchmarks/reference_kdm_neural_exact_replay.py"]
    for path in files:
        dest = out / "sources" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    command = [PY, "-I", "-B", str(ROOT / "docs/benchmarks/reference_kdm_native_selection.py"), "--output", str(out / "diagnostic")]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
        LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    manifest = dict(status="RUNNING", supervisor_pid=os.getpid(), command=command,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        git_dirty=subprocess.check_output(["git", "status", "--short", "--untracked-files=no"], cwd=ROOT, text=True),
        plan_file=PLAN, result_file=RESULT, start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        environment=PY, cpu_only=True, gpu_intentionally_hidden=True, jit_compile="N/A: original PyTorch author reference",
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in files},
        data_version="pinned units01/02 full training/validation; exact input hashes in diagnostic/inputs.json",
        seed_base=2026101300, seeds="explicit per-call seeds in diagnostic/evaluations.jsonl; training seed_base+1,+2",
        phase_cap_cpu_seconds=3600, per_launch_cap_seconds=1800, attempt_cap=2,
        resource_wall_seconds=dict(cpu=0., gpu=0.), campaign_remaining_before=balance["remaining_seconds"],
        output=str(out.relative_to(ROOT)), replication="INCOMPLETE")
    write(out / "manifest.json", manifest)
    started = time.monotonic()
    proc = None
    decision = dict(status="FAILED")
    try:
        with (out / "execution.log").open("w") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            manifest["worker_pid"] = proc.pid
            write(out / "manifest.json", manifest)
            code = proc.wait(timeout=min(1800, remaining))
            if code:
                raise RuntimeError("Author selection diagnostic exited " + str(code))
        decision = read(out / "diagnostic/result.json")
        manifest["status"] = "COMPLETED"
    except BaseException:
        manifest["status"] = "FAILED"
        decision["error"] = traceback.format_exc()
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        elapsed = time.monotonic()-started
        manifest["resource_wall_seconds"] = dict(cpu=elapsed, gpu=0.)
        manifest["end_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(out / "decision.json", decision)
        write(out / "manifest.json", manifest)
        ledger = BASE / "reference-diagnostics.json"
        records = read(ledger)
        identity = "kdm-native-selection-" + out.name + "_cpu"
        assert not any(x["id"] == identity for x in records)
        records.append(dict(id=identity, resource="cpu", charged_seconds=elapsed,
            summary=str((out / "decision.json").relative_to(ROOT)), summary_sha256=sha(out / "decision.json"),
            manifest=str((out / "manifest.json").relative_to(ROOT)), plan_file=PLAN, result_file=RESULT,
            stage="KDM_NATIVE_SELECTION_DIAGNOSTIC", after_attempt=balance["attempts"],
            next_action="Audit conditional native validation result and independent-unit precision feasibility",
            executions=[dict(path=str(out.relative_to(ROOT)), resource="cpu", wall_seconds=elapsed, status=manifest["status"])]))
        write(ledger, records)
    print(json.dumps(dict(status=manifest["status"], output=str(out), charged_cpu_seconds=elapsed)), flush=True)
    return 0 if manifest["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
