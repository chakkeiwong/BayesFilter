"""CPU-only arithmetic audit; these fixed arrays are not posterior samples."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
started = time.monotonic()

import numpy as np
from scipy.stats import norm, rankdata
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference import hmc_convergence as shared
from bayesfilter.inference import hmc_posterior_diagnostics as posterior


def normalize(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(-1, values.shape[-1])
    ranks = np.column_stack([
        rankdata(flat[:, j], method="average") for j in range(flat.shape[-1])
    ])
    return norm.ppf((ranks - 3 / 8) / (flat.shape[0] + 1 / 4)).reshape(values.shape)


def split(values: np.ndarray) -> np.ndarray:
    half = len(values) // 2
    return np.concatenate((values[:half], values[half:]), axis=1)


def rhat(values: np.ndarray) -> np.ndarray:
    count = values.shape[0]
    within = np.var(values, axis=0, ddof=1).mean(axis=0)
    between_div_count = np.var(values.mean(axis=0), axis=0, ddof=1)
    return np.sqrt(((count - 1) / count * within + between_div_count) / within)


def tfp_ratio(values: np.ndarray) -> np.ndarray:
    n, m, _ = values.shape
    within = np.var(values, axis=0, ddof=1).mean(axis=0)
    between_div_count = np.var(values.mean(axis=0), axis=0, ddof=1)
    variance_plus = (n - 1) / n * within + between_div_count
    return ((m + 1) / m) * variance_plus / within - (n - 1) / (m * n)


# Convenience arithmetic geometry: 64 even draws and four chains, two columns
# to exercise continuous ranks and ties. No seed or stochastic model is used.
t = np.arange(64, dtype=float)[:, None]
c = np.arange(4, dtype=float)[None, :]
values = np.stack((
    np.sin(0.7 * t + c) + np.cos(0.11 * t + 0.2 * c) + 0.3 * c,
    ((3 * t + c) % 7) + (c == 3),
), axis=-1)
tensor = tf.constant(values, tf.float64)
folded = np.abs(values - np.median(values, axis=(0, 1), keepdims=True))
expected_bulk = rhat(split(normalize(values)))
expected_folded = rhat(split(normalize(folded)))
expected_max = np.maximum(expected_bulk, expected_folded)
shared_normal = shared._rank_normalize(tensor).numpy()
posterior_normal = posterior._rank_normalize(tensor).numpy()
shared_report = shared.rank_normalized_split_rhat_summary(tensor)
posterior_report = posterior.rank_normalized_split_rhat(tf.transpose(tensor, (1, 0, 2)))

np.testing.assert_allclose(shared_normal, normalize(values), rtol=0, atol=1e-12)
assert np.max(np.abs(posterior_normal - normalize(values))) > 1e-6
np.testing.assert_allclose(
    shared_report["rhat"],
    np.maximum(tfp_ratio(split(normalize(values))), tfp_ratio(split(normalize(folded)))),
    rtol=0, atol=1e-12,
)
assert np.max(np.abs(np.array(shared_report["rhat"]) - expected_max)) > 1e-6
assert np.max(np.abs(posterior_report["maximum"].numpy() - expected_max)) > 1e-6

result = {
    "role": "deterministic_arithmetic_counterexample_only",
    "sample_shape_draw_chain_parameter": list(values.shape),
    "sample_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    "published_bulk_rhat": expected_bulk.tolist(),
    "published_folded_rhat": expected_folded.tolist(),
    "published_max_rhat": expected_max.tolist(),
    "shared_reported_rhat": list(shared_report["rhat"]),
    "posterior_reported_rhat": posterior_report["maximum"].numpy().tolist(),
    "posterior_rank_normalization_max_abs_error": float(np.max(np.abs(posterior_normal - normalize(values)))),
    "shared_rank_normalization_max_abs_error": float(np.max(np.abs(shared_normal - normalize(values)))),
    "verdict": "both_report_paths_differ_from_published_modern_rhat_for_distinct_reasons",
    "not_concluded": ["target-specific burn-in sufficiency", "sampler convergence", "GPU or XLA qualification", "estimator performance ranking"],
}
output = Path(__file__).parent
(output / "diagnostic-formula-result.json").write_text(json.dumps(result, indent=2) + "\n")

sources = [
    ROOT / "bayesfilter/inference/hmc_convergence.py",
    ROOT / "bayesfilter/inference/hmc_posterior_diagnostics.py",
    ROOT / "bayesfilter/inference/neutra_hmc.py",
    Path(tfp.mcmc.diagnostic.__file__),
    Path(__file__),
]
sources.extend(sorted((ROOT / ".localresources/papers/hmc_warmup_precision_20260915").glob("*")))
manifest = {
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "command": "env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/plans/artifacts/hmc-warmup-precision-review-2026-09-15/diagnostic_formula_check.py",
    "environment": {"python": sys.version, "tensorflow": tf.__version__, "tensorflow_probability": tfp.__version__, "conda_env": "tfgpu"},
    "device_scope": "CPU-only arithmetic diagnostic; GPUs intentionally hidden before imports; no GPU probe",
    "jit_compile": False,
    "jit_exception": "independent-reference arithmetic audit, no sampler or default qualification",
    "seeds": "N/A: deterministic fixed arrays",
    "data_version": "N/A: formula fixtures; array checksum recorded",
    "wall_seconds": time.monotonic() - started,
    "plan": "docs/plans/bayesfilter-hmc-warmup-precision-repair-plan-2026-09-15.md",
    "result": str((output / "diagnostic-formula-result.json").relative_to(ROOT)),
    "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources if path.is_file()},
}
(output / "run-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps({"result": result, "wall_seconds": manifest["wall_seconds"]}, indent=2))
