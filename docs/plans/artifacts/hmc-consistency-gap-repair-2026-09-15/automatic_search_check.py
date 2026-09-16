"""Bounded automatic-search mechanics checks, no posterior or ranking claim."""
import os
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU debugging only; hide GPUs before framework imports")

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=False)
case = sys.argv[2]
started = time.monotonic()

import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.inference import HMCKernelTuningConfig, tune_hmc_kernel
from tests.test_hmc_candidate_set_execution import GaussianTarget


class AnisotropicGaussian(GaussianTarget):
    """Independent Gaussian with convenience fixture variances 1 and 9."""
    def adapter_signature(self):
        return "review-anisotropic-gaussian-variance-1-9-v1"

    def log_prob_and_grad(self, theta):
        theta = tf.convert_to_tensor(theta, tf.float64)
        precision = tf.constant([1., 1./9.], tf.float64)
        return -.5 * tf.reduce_sum(theta * theta * precision, axis=-1), -theta * precision


manifest = {"baseline_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
    capture_output=True, text=True, check=True).stdout.strip(), "argv": sys.argv,
    "python": platform.python_version(), "tensorflow": tf.__version__, "tfp": tfp.__version__,
    "environment": {name: os.environ.get(name) for name in (
        "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "BAYESFILTER_TEST_DEVICE_SCOPE")},
    "role": "CPU/non-XLA complete automatic tuning mechanics; no posterior or ranking claim",
    "case": case, "seed": [20260621, 8], "initial_position": [0., 0.],
    "variance_provenance": "unit Gaussian baseline; variances 1 and 9 are a convenience anisotropic regression fixture",
    "sources": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (ROOT / "bayesfilter/inference").glob("*.py")}}
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
try:
    target = GaussianTarget() if case == "isotropic" else AnisotropicGaussian()
    cfg = HMCKernelTuningConfig.standard(use_xla=False, target_scope="candidate-bridge-test")
    run = tune_hmc_kernel(adapter=target, initial_position=[0., 0.], config=cfg,
        parameter_scales=None if case == "isotropic" else [1., 3.],
        target_lineage={"model": case + " exact Gaussian", "data": "none", "prior": "Gaussian"},
        source_paths=[__file__, str(ROOT / "tests/test_hmc_candidate_set_execution.py")],
        output_dir=destination / "tuning")
    result = {"completion": run.result.completion_status,
        "verified_count": len(run.result.verified_candidate_ids),
        "candidate_count": len(run.result.candidates),
        "work_units": run.result.budget_used_units,
        "states": run.result.candidate_states,
        "verified_pairs": [{"L": c.leapfrog_steps, "epsilon": c.epsilon}
            for c in run.result.candidates if c.candidate_id in run.result.verified_candidate_ids]}
    result["passed"] = result["completion"] == "complete" and result["verified_count"] > 0
except Exception as exc:
    result = {"passed": False, "exception": type(exc).__name__, "reason": str(exc),
              "traceback": traceback.format_exc()}
result["wall_seconds_including_imports"] = time.monotonic() - started
(destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if result["passed"] else 1)
