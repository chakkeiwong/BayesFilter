"""Diagnostic-only arithmetic audit; no sampling and no production claims."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

started = time.monotonic()
cpu_started = time.process_time()
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
source = ROOT / "scripts/phase2_toy_potential_test.py"
spec = importlib.util.spec_from_file_location("recovered_toy", source)
toy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(toy)

current = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], tf.float64)
proposal = tf.constant([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], tf.float64)
current_expected = tf.constant([0.5, 2.0], tf.float64)
proposal_expected = tf.constant([2.0, 0.0], tf.float64)
current_actual = toy.neg_log_prob(current)
proposal_actual = toy.neg_log_prob(proposal)
# Zero momenta isolate the potential part of the runner's acceptance formula.
actual_log_ratios = current_actual - proposal_actual + tf.zeros([2], tf.float64)
expected_log_ratios = current_expected - proposal_expected
log_uniform = tf.constant([-1.0, -1.0], tf.float64)
actual_accept = log_uniform < actual_log_ratios
expected_accept = log_uniform < expected_log_ratios
assert current_actual.shape.rank == 0
assert bool(tf.reduce_any(actual_accept != expected_accept).numpy())

# The moment comparison uses population covariance. These samples therefore
# represent fitted Gaussian variances 0.01 and 0.04, with zero means.
samples_a = tf.constant([[-0.1], [0.1]], tf.float64)
samples_b = tf.constant([[-0.2], [0.2]], tf.float64)
reported_distance = toy.compute_wasserstein_gaussian_approx(samples_a, samples_b)
exact_gaussian_w2 = 0.1
assert abs(reported_distance - 0.03) < 1e-12
assert reported_distance < exact_gaussian_w2

result = {
    "classification": "diagnostic_only_deterministic_counterexamples",
    "historical_program": "September 7 manual Gaussian toy HMC, not DCTR",
    "tuning_status": "historical fixed controls; no tuning authority",
    "cpu_only": True,
    "gpu_intentionally_hidden": True,
    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
    "jit_compile": False,
    "execution_exception": "tiny CPU eager reference arithmetic; no chain",
    "source_path": str(source.relative_to(ROOT)),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    "tensorflow_version": tf.__version__,
    "target": "independent N(0, diag(1, 1/4, 1/9)) chains",
    "energy_check": {
        "current_expected_per_chain": current_expected.numpy().tolist(),
        "current_actual_scalar": float(current_actual.numpy()),
        "proposal_expected_per_chain": proposal_expected.numpy().tolist(),
        "proposal_actual_scalar": float(proposal_actual.numpy()),
        "expected_log_acceptance_ratios": expected_log_ratios.numpy().tolist(),
        "actual_log_acceptance_ratios": actual_log_ratios.numpy().tolist(),
        "log_uniforms": log_uniform.numpy().tolist(),
        "expected_accept": expected_accept.numpy().tolist(),
        "actual_accept": actual_accept.numpy().tolist(),
        "verdict": "wrong relative to independent per-chain MH acceptance",
    },
    "distance_check": {
        "reported_frobenius_moment_statistic": reported_distance,
        "exact_gaussian_wasserstein_distance": exact_gaussian_w2,
        "verdict": "neither exact W2 nor the asserted upper bound",
    },
    "historical_claim_status": "recorded PASS cannot establish valid sampling",
    "research_direction_status": "not rejected; endpoint-corrected mechanism remains viable",
    "random_seeds": "N/A: deterministic arithmetic, no sampling",
    "wall_seconds": time.monotonic() - started,
    "process_cpu_seconds": time.process_time() - cpu_started,
    "plan": "docs/plans/ledh-surrogate-hmc-recovery-plan-2026-09-18.md",
}
(OUT / "toy-arithmetic-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({
    "energy": result["energy_check"]["verdict"],
    "distance": result["distance_check"]["verdict"],
    "wall_seconds": result["wall_seconds"],
    "artifact": str(OUT / "toy-arithmetic-audit.json"),
}, indent=2))
