"""CPU-only regression preflight for the prepared M13 repeatability batch."""
from pathlib import Path
import json
import os
import resource
import signal
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.insert(0, "/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917")
from bayesfilter.inference.surrogate_hmc_closure_program import preceding_charges, ROOT
from bayesfilter.inference.surrogate_hmc_program import validated_budget, run_job, write_json, file_hash

output = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260924-repeatability-preflight-02"
previous, receipts = preceding_charges(output)
if output.exists():
    raise SystemExit("fresh preflight output required")
if previous["cpu_worker_seconds"] + 360 > 54000:
    raise SystemExit("insufficient CPU preflight budget")
output.mkdir()
plan = "docs/plans/surrogate-hmc-m13-repeatability-2026-09-24.md"
paths = [
    "bayesfilter/testing/surrogate_hmc_m13_repeatability.py",
    "bayesfilter/testing/surrogate_hmc_closure_diagnostics.py",
    "bayesfilter/inference/surrogate_hmc_closure_program.py",
    "tests/test_surrogate_hmc_m13_repeatability.py", plan]
for name in paths:
    path = output / "source" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((ROOT / name).read_bytes())
(output / "preflight-command.py").write_bytes(Path(__file__).read_bytes())
ledger = validated_budget(1., 360.)
usage = resource.getrusage(resource.RUSAGE_SELF)
cpu_start = usage.ru_utime + usage.ru_stime
wall_start = time.monotonic()
manifest = dict(command=[sys.executable, str(Path(__file__).resolve())], cwd=str(ROOT),
    plan=plan, git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    environment=sys.executable, cpu_only=True, gpu_intentionally_hidden=True,
    test_seeds="fixture-defined deterministic tests; no inference",
    source_sha256={name:file_hash(ROOT/name) for name in paths},
    output=str(output), status="RUNNING")
write_json(output/"run-manifest.json", manifest)
command = [sys.executable, "-m", "pytest", "-q",
    "tests/test_surrogate_hmc_closure_program.py", "tests/test_surrogate_hmc_m13_repeatability.py",
    "tests/test_surrogate_hmc_m13_consumer.py", "tests/test_surrogate_hmc_program.py",
    "tests/test_surrogate_hmc_dependency_execution.py"]
try:
    record = run_job(command, output/"review", ledger, device="cpu", timeout=240., cpu_allowance=300.)
    manifest["status"] = record["status"]
finally:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    ledger["charged"]["cpu_worker_seconds"] += usage.ru_utime + usage.ru_stime - cpu_start
    accounting = dict(charged_this_run=ledger["charged"], previous_charge=previous,
        cumulative={k:previous[k]+ledger["charged"][k] for k in previous},
        jobs=ledger["jobs"])
    manifest.update(wall_seconds=time.monotonic()-wall_start,
        source_unchanged=all(file_hash(ROOT/name)==value for name,value in manifest["source_sha256"].items()))
    write_json(output/"accounting.json", accounting)
    write_json(output/"run-manifest.json", manifest)
    (output/"result.md").write_text("# CPU-only M13 preparation check\n\n"+manifest["status"]+
        "\n\nGPU devices intentionally hidden. No numerical M13 GPU replay or inference was run. "
        "See review/command.log and accounting.json.\n")
    print(json.dumps(dict(status=manifest["status"], charged=ledger["charged"], cumulative=accounting["cumulative"],
        source_unchanged=manifest["source_unchanged"], log=str(output/"review/command.log"))))
raise SystemExit(0 if manifest["status"]=="COMPLETED" else 1)
