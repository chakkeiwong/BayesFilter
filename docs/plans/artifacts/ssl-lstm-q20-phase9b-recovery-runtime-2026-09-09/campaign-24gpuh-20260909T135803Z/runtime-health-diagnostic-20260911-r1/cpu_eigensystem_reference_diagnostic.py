"""Independent CPU-only diagnostic for one saved covariance; never runtime."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU reference must hide GPU devices before import")
ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("cpu-eigensystem-r2")
OUTPUT.mkdir(exist_ok=False)
sys.path.insert(0, str(ROOT))
started = time.monotonic()

import tensorflow as tf
import scipy
import scipy.linalg
import mpmath as mp
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json

trace_root = CAMPAIGN / "launches/covariance-diagnostic-6a6c7c4a1b/strict/tensors"
input_path = CAMPAIGN / "launches/cached-gradient-diagnostic-2dededa013/strict/transition-158-L2.json"
saved = json.loads(input_path.read_text())
with DurableTensorCheckpoint(trace_root, json.loads((trace_root / "identity.json").read_text())) as store:
    result, receipt = store.load("instrumented", {"parameters": saved["values"]["proposed_model_state"]})
matrix = result[3]["incoming_augmented_covariance"][17, 3]
rows = matrix.numpy().tolist()
if not bool(tf.reduce_all(matrix == tf.transpose(matrix)).numpy()):
    raise RuntimeError("Stored covariance is not exactly symmetric")

reports = {}
native_values, native_vectors = tf.linalg.eigh(matrix)
reports["tensorflow_cpu"] = {
    "minimum_eigenvalue": float(native_values[0].numpy()),
    "maximum_eigenpair_residual": float(tf.reduce_max(tf.abs(matrix @ native_vectors - native_vectors * native_values[None, :])).numpy()),
}
for driver in ("evd", "evr", "ev"):
    values, vectors = scipy.linalg.eigh(matrix.numpy(), driver=driver)
    reports["scipy_" + driver] = {
        "minimum_eigenvalue": float(values[0]),
        "maximum_eigenpair_residual": float(abs(matrix.numpy() @ vectors - vectors * values[None, :]).max()),
    }
durable_json(OUTPUT / "float64-reference.json", reports)

mp.mp.dps = 60
exact_rows = []
for row in rows:
    exact_row = []
    for value in row:
        numerator, denominator = value.as_integer_ratio()
        exact_row.append(mp.mpf(numerator) / denominator)
    exact_rows.append(exact_row)
reference = mp.matrix(exact_rows)
factor = mp.cholesky(reference)
values = mp.eigsy(reference, eigvals_only=True)
if not values[values.rows - 1, 0] > values[0] > 0:
    raise RuntimeError("Independent reference eigenvalues do not confirm SPD")
reports["mpmath_60_digits"] = {
    "minimum_eigenvalue": str(values[0]), "maximum_eigenvalue": str(values[values.rows - 1, 0]),
    "minimum_cholesky_diagonal": str(min(factor[index, index] for index in range(factor.rows))),
    "maximum_cholesky_residual": str(max(abs(value) for value in reference - factor * factor.T)),
    "float_inputs_converted_by_exact_integer_ratio": True,
}
manifest = {
    "status": "completed", "evidence_role": "independent_reference_only",
    "reports": reports, "shape": matrix.shape.as_list(), "filter_time_zero_based": 17, "chain_zero_based": 3,
    "gpu_devices_intentionally_hidden": True, "tensorflow": tf.__version__, "scipy": scipy.__version__, "mpmath": mp.__version__,
    "command": [sys.executable, *sys.argv],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "matrix_sha256": hashlib.sha256(tf.io.serialize_tensor(matrix).numpy()).hexdigest(),
    "source_bundle": str(trace_root / "committed/instrumented"), "bundle_checksum_verified": True,
    "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "jit_compile": False, "jit_exception": "Independent CPU tiny matrix reference",
    "seed_policy": "No new randomness", "wall_seconds": time.monotonic() - started,
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result_file": str(OUTPUT / "reference-result.json"),
}
durable_json(OUTPUT / "reference-result.json", manifest)
print(json.dumps(reports), flush=True)
