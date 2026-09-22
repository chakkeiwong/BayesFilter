"""Bounded CPU diagnostic launcher; no HMC campaign or posterior authority."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).resolve().parent
repo = root.parents[4]
output = root / sys.argv[1]
output.mkdir(exist_ok=False)
command = ["/tmp/hmc-reference-20260922/bin/python", "-m", "pytest", "-q", "--tb=short", "--disable-warnings",
           "-p", "no:cacheprovider", "--junitxml=" + str(output / "tests.xml"), *sys.argv[2:]]
spent = sum(json.loads(p.read_text())["wall_seconds"] for p in root.glob("tests-*/result.json"))
remaining = int(1800 - spent)
if remaining <= 0:
    raise RuntimeError("focused diagnostic allocation exhausted")
env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", TF_FORCE_GPU_ALLOW_GROWTH="true",
           TF_NUM_INTRAOP_THREADS="1", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="1",
           OPENBLAS_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", PYTHONPATH=str(repo),
           HMC_REPAIR_TEST_OUTPUT=str(output))
tracked = subprocess.check_output(["git", "diff", "--binary"], cwd=repo)
paths = ["bayesfilter/inference/hmc_bootstrap.py", "bayesfilter/inference/hmc_mass_adaptation.py",
         "bayesfilter/inference/hmc_convergence.py", "bayesfilter/inference/hmc_kernel_tuning.py",
         "bayesfilter/inference/hmc_posterior_diagnostics.py", "bayesfilter/inference/hmc_ess.py"]
record = dict(command=command, environment="tfgpu + independent ArviZ 0.21 reference venv",
    gpu_status="intentionally hidden (CUDA_VISIBLE_DEVICES=-1)",
    cpu_threads=1, seeds="fixed in named tests", data_version="synthetic fixtures in named tests",
    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
    worktree_diff_sha256=hashlib.sha256(tracked).hexdigest(),
    source_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths},
    plan_file="docs/plans/bayesfilter-macrofinance-bootstrap-diagnostics-repair-2026-09-22.md",
    result_file="docs/plans/bayesfilter-macrofinance-bootstrap-diagnostics-repair-2026-09-22.md",
    artifacts=[str(output / "tests.log"), str(output / "tests.xml")],
    started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), timeout_seconds=remaining,
    role="engineering_regression_not_posterior_or_performance_evidence")
(output / "manifest.json").write_text(json.dumps(record, indent=2)+"\n")
started = time.monotonic()
with (output / "tests.log").open("w") as log:
    try:
        result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=remaining)
        code = result.returncode
    except subprocess.TimeoutExpired:
        code = 124
record.update(exit_code=code, wall_seconds=time.monotonic()-started)
(output / "result.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps({"result":str(output / "result.json"), "exit_code":code,
                  "cpu_worker_seconds":record["wall_seconds"]}))
sys.exit(code)
