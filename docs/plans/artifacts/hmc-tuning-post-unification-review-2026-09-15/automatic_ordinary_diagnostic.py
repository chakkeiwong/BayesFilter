"""Full automatic ordinary CPU mechanics check; no scientific qualification."""
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU debugging only: hide GPUs before imports")
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
manifest = {"command": sys.argv, "baseline": subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
    check=True).stdout.strip(), "role": "CPU/non-XLA mechanics exception; no posterior or ranking claim",
    "environment": {key: os.environ.get(key) for key in
        ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_TEST_DEVICE_SCOPE")},
    "target": "two-dimensional standard Gaussian", "initial_position": [0., 0.],
    "external_timeout_seconds": 180}
(root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
try:
    from tests.test_hmc_candidate_set_execution import GaussianTarget
    from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
    config = HMCKernelTuningConfig.standard(use_xla=False, target_scope="candidate-bridge-test")
    run = tune_hmc_kernel(adapter=GaussianTarget(), initial_position=[0., 0.], config=config,
        target_lineage={"model": "two-dimensional standard Gaussian", "data": "none", "prior": "standard normal"},
        source_paths=[str(ROOT / "tests/test_hmc_candidate_set_execution.py")],
        output_dir=root / "tuning")
    result = {"returned": True, "completion": run.result.completion_status,
        "verified_count": len(run.result.verified_candidate_ids),
        "candidate_count": len(run.result.candidates),
        "candidates": [{"L": c.leapfrog_steps, "epsilon": c.epsilon,
            "state": run.result.candidate_states[c.candidate_id]} for c in run.result.candidates],
        "work_count": len(run.result.work_items), "budget_used": run.result.budget_used_units,
        "nominee": run.result.payload()["nominee_id"], "base_seed": config.seed,
        "epsilon_domain": run.result.scope.epsilon_domain}
except Exception as exc:
    result = {"returned": False, "exception_type": type(exc).__name__, "message": str(exc),
        "traceback": traceback.format_exc()}
result["wall_seconds_including_imports"] = time.monotonic() - started
(root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
