"""Bounded local driver for the September 17 filter execution repair.

This is host orchestration, not a numerical implementation. The command prefix
is deliberately stable. Only registered test groups and fixtures can execute;
there is no arbitrary command, network, installation, merge, or push action.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/plans/artifacts/filter-gradient-repair-20260917"
PLAN = "docs/plans/filter_gradient_repair_master_20260917.md"
BASELINE = "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf"
BASELINE_ROOT = Path("/tmp/bayesfilter-filter-repair-baseline-3582b4ac")
BUDGET_SECONDS = {"CPU": 8 * 3600, "GPU": 4 * 3600}
TEST_GROUPS = {
    "primitives": ("tests/highdim/test_retained_moments.py", "tests/test_filter_repair_primitives.py"),
    "kalman": ("tests/test_compiled_filter_parity_tf.py", "tests/test_filter_runtime_policy.py"),
    "sgqf": ("tests/test_fixed_sgqf_tf.py", "tests/test_fixed_sgqf_scores_tf.py", "tests/test_fixed_sgqf_integration_tf.py", "tests/test_predator_prey_sgqf_neutra_target.py"),
    "genut": ("tests/highdim/test_cubature_genut_batch.py", "tests/highdim/test_genut_batch_primal_parity.py", "tests/highdim/test_genut_batch_general_route_parity.py", "tests/highdim/test_ledh_contract_e_canonical_lgssm_phase5.py"),
    "tt": ("tests/highdim/test_squared_tt_density.py", "tests/highdim/test_zhao_cui_actual_sv_batched_tt_tf.py", "tests/highdim/test_zhao_cui_frozen_proposal_apf_tf.py"),
    "consumers": ("tests/test_filter_repair_consumers.py",),
    "policy": ("tests/test_filter_repair_campaign.py", "tests/test_filter_repair_policy.py"),
}
FIXTURES = ("rectangular", "factor", "covariance", "sinkhorn_jvp", "sqmc", "dns", "retained_moments", "sgqf_derivatives", "joint_target", "genut", "contract_e", "tt", "tt_adjoint", "apf", "particle", "cpu_pool")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def source_hashes():
    paths = git("ls-files", "--cached", "--others", "--exclude-standard", "bayesfilter", "experiments/dpf_implementation/tf_tfp", "scripts", "tests").splitlines()
    return {p: sha(ROOT / p) for p in sorted(set(paths)) if p.endswith(".py") and (ROOT / p).is_file()}


def records():
    return [json.loads(p.read_text()) for p in sorted(OUTPUT.glob("run-*/run.json"))]


def charged_seconds(rows, device):
    # A crashed/unfinished run is charged its whole reserved timeout on resume.
    return sum(row.get("elapsed_seconds", row["timeout_seconds"]) for row in rows if row["device"] == device)


def save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def ensure_baseline():
    marker = BASELINE_ROOT / "source-manifest.json"
    if marker.exists():
        manifest = json.loads(marker.read_text())
        if manifest["commit"] != BASELINE or any(sha(BASELINE_ROOT / p) != digest for p, digest in manifest["files"].items()):
            raise RuntimeError("Baseline snapshot changed; comparison is invalid")
        return
    BASELINE_ROOT.mkdir(exist_ok=False)
    archive = subprocess.check_output(["git", "archive", BASELINE, "bayesfilter", "experiments/dpf_implementation/tf_tfp"], cwd=ROOT, timeout=60)
    with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
        handle.extractall(BASELINE_ROOT, filter="data")
    save_json(marker, {"commit": BASELINE, "files": {str(p.relative_to(BASELINE_ROOT)): sha(p) for p in BASELINE_ROOT.rglob("*") if p.is_file()}})


def run_job(args):
    rows = records()
    device = args.device
    timeout = 900 if args.action == "test" else 300
    if charged_seconds(rows, device) + timeout > BUDGET_SECONDS[device]:
        raise RuntimeError(f"{device} campaign budget exhausted")
    key = [args.action, args.group, args.arm, args.fixture, args.jit, args.size, args.repeat, device]
    hashes = source_hashes()
    attempts = [row for row in rows if row["key"] == key and row["source_sha256"] == hashes]
    if len(attempts) >= 3:
        raise RuntimeError("Three attempts consumed for this exact job; inspect/repair scope before retry")
    directory = OUTPUT / f"run-{len(rows) + 1:05d}"
    directory.mkdir(exist_ok=False)
    result = directory / "result.json"
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "2" if device == "GPU" else "-1", "TF_FORCE_GPU_ALLOW_GROWTH": "true", "TF_NUM_INTRAOP_THREADS": "2", "TF_NUM_INTEROP_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg", "PYTHONHASHSEED": "0"})
    if args.action == "test":
        command = [sys.executable, "scripts/filter_repair_test_worker.py", "-q", *TEST_GROUPS[args.group], f"--junitxml={directory / 'junit.xml'}"]
    elif args.action == "measure":
        ensure_baseline()
        source = BASELINE_ROOT if args.arm == "before" else ROOT
        command = [sys.executable, str(ROOT / "scripts/filter_repair_benchmark_worker.py"), "--source-root", str(source), "--fixture", args.fixture, "--jit", args.jit, "--size", str(args.size), "--device", device, "--output", str(result)]
    elif args.action == "audit":
        command = [sys.executable, "scripts/audit_filter_gradient_policy.py", "--output", str(directory / "audit.json.gz"), "--markdown", str(directory / "audit.md")]
    elif args.action == "compare":
        command = [sys.executable, "scripts/compare_filter_repair_campaign.py", "--output", str(result)]
    else:
        raise ValueError(args.action)
    record = {"schema": "filter_repair_run.v1", "key": key, "started_utc": datetime.now(timezone.utc).isoformat(), "state": "running", "device": device, "timeout_seconds": timeout, "command": command, "cwd": str(ROOT), "environment": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS", "PYTHONHASHSEED")}, "git_head": git("rev-parse", "HEAD"), "git_diff_stat": git("diff", "--stat"), "source_sha256": hashes, "plan": PLAN, "result": str(result), "log": str(directory / "process.log")}
    save_json(directory / "run.json", record)
    started = time.monotonic()
    print(json.dumps({"run": str(directory), "command": command}), flush=True)
    with (directory / "process.log").open("x") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = 124
    record.update(state="passed" if code == 0 else "failed", exit_code=code, elapsed_seconds=time.monotonic() - started)
    save_json(directory / "run.json", record)
    print(json.dumps({"state": record["state"], "elapsed_seconds": record["elapsed_seconds"], "log": record["log"]}), flush=True)
    if code:
        print((directory / "process.log").read_text()[-14000:])
    return code


def gate():
    ledger = json.loads((ROOT / "docs/plans/filter_gradient_repair_ledger_20260917.json").read_text())
    pending = [item["id"] for item in ledger["findings"] if item["status"] != "closed" or not item.get("evidence") or any(not (ROOT / path).is_file() for path in item.get("evidence", []))]
    if {item["id"] for item in ledger["findings"]} != {f"F{i:02d}" for i in range(1, 21)}:
        pending.append("incomplete_finding_inventory")
    rows = records()
    current = source_hashes()
    missing = []
    for group in TEST_GROUPS:
        candidates = [row for row in rows if row["key"][:2] == ["test", group] and row["state"] == "passed" and row["source_sha256"] == current]
        if not candidates:
            missing.append(group)
    comparisons = [row for row in rows if row["key"][0] == "compare" and row["state"] == "passed" and row["source_sha256"] == current]
    complete = not pending and not missing and bool(comparisons)
    print(json.dumps({"merge_allowed": complete, "open_findings": pending, "missing_current_tests": missing, "current_comparison": bool(comparisons)}, indent=2))
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "test", "measure", "audit", "compare", "gate"))
    parser.add_argument("--group", choices=tuple(TEST_GROUPS), default="policy")
    parser.add_argument("--fixture", choices=FIXTURES, default="covariance")
    parser.add_argument("--arm", choices=("before", "after"), default="after")
    parser.add_argument("--jit", choices=("on", "off"), default="on")
    parser.add_argument("--size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repeat", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--device", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action == "status":
            rows = records()
            print(json.dumps({"branch": git("branch", "--show-current"), "baseline": BASELINE, "runs": len(rows), "budget_seconds": BUDGET_SECONDS, "charged_seconds": {d: charged_seconds(rows, d) for d in BUDGET_SECONDS}, "artifact_root": str(OUTPUT)}, indent=2))
            return 0
        if args.action == "gate":
            return gate()
        return run_job(args)


if __name__ == "__main__":
    raise SystemExit(main())
