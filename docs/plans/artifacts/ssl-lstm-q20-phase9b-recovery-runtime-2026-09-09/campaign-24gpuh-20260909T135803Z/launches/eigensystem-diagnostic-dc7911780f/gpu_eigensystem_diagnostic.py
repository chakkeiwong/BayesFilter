"""Fixed-matrix GPU eigensolver diagnostic, not a target or tuning route."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-uuid", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
    raise RuntimeError("Diagnostic GPU environment mismatch")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Memory growth must precede framework import")
ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(exist_ok=False)
sys.path.insert(0, str(ROOT))
started = time.monotonic()

import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
tf.config.set_soft_device_placement(False)
if len(tf.config.list_logical_devices("GPU")) != 1:
    raise RuntimeError("Exactly one GPU required")

from tensorflow.compiler.tf2xla.ops import gen_xla_ops
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json

spec = importlib.util.spec_from_file_location("eigen_recovery_diagnostic", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
job = json.loads((CAMPAIGN / "launches/runtime-c48f1f0535/strict-job.json").read_text())
if recovery.source_hashes() != job["sources"]:
    raise RuntimeError("Scientific source closure changed")
input_path = CAMPAIGN / "launches/cached-gradient-diagnostic-2dededa013/strict/transition-158-L2.json"
saved = json.loads(input_path.read_text())
trace_root = CAMPAIGN / "launches/covariance-diagnostic-6a6c7c4a1b/strict/tensors"
with DurableTensorCheckpoint(trace_root, json.loads((trace_root / "identity.json").read_text())) as store:
    recorded, _ = store.load("instrumented", {"parameters": saved["values"]["proposed_model_state"]})
matrices = recorded[3]["incoming_augmented_covariance"][17]
reference_path = Path(__file__).with_name("cpu-eigensystem-r2") / "reference-result.json"
reference = json.loads(reference_path.read_text())
reference_minimum = float(reference["reports"]["mpmath_60_digits"]["minimum_eigenvalue"])


def compiled_solver(epsilon):
    @tf.function(input_signature=(tf.TensorSpec(matrices.shape, tf.float64),), jit_compile=True, autograph=False)
    def solve(batch):
        if epsilon is None:
            return tf.linalg.eigh(batch)
        return gen_xla_ops.xla_self_adjoint_eig(batch, lower=True, max_iter=100, epsilon=epsilon)

    return solve


solvers = {
    "tensorflow_default": compiled_solver(None),
    "explicit_1e_minus6": compiled_solver(1e-6),
    "explicit_1e_minus12": compiled_solver(1e-12),
    "explicit_8_double_epsilon": compiled_solver(8 * sys.float_info.epsilon),
}
identity = {"sources": job["sources"], "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_tensor_sha256": hashlib.sha256(tf.io.serialize_tensor(matrices).numpy()).hexdigest()}
durable_json(OUTPUT / "memory_policy.json", memory)
reports = {}
with DurableTensorCheckpoint(OUTPUT / "tensors", identity) as store:
    baseline_values = None
    for name, solver in solvers.items():
        call_started = time.monotonic()
        values, vectors = store.run(name, {}, lambda: tuple(solver(matrices)))
        if baseline_values is None:
            baseline_values = values
        (OUTPUT / (name + ".hlo.txt")).write_text(solver.experimental_get_compiler_ir(matrices)(stage="hlo"))
        residual = matrices @ vectors - vectors * values[:, None, :]
        orthogonality = tf.linalg.matrix_transpose(vectors) @ vectors - tf.eye(matrices.shape[-1], dtype=tf.float64)[None]
        reports[name] = {
            "minimum_eigenvalues": values[:, 0].numpy().tolist(),
            "failed_chain_reference_error": float(values[3, 0].numpy()) - reference_minimum,
            "maximum_eigenpair_residual_per_chain": tf.reduce_max(tf.abs(residual), axis=(-2, -1)).numpy().tolist(),
            "frobenius_eigenpair_residual_per_chain": tf.linalg.norm(residual, axis=(-2, -1)).numpy().tolist(),
            "maximum_orthogonality_error_per_chain": tf.reduce_max(tf.abs(orthogonality), axis=(-2, -1)).numpy().tolist(),
            "eigenvalues_exact_to_default": bool(tf.reduce_all(values == baseline_values).numpy()),
            "first_call_with_compile_checkpoint_seconds": time.monotonic() - call_started,
        }
durable_json(OUTPUT / "eigensystem-result.json", reports)
if recovery.source_hashes() != job["sources"]:
    raise RuntimeError("Scientific source closure changed during diagnostic")
durable_json(OUTPUT / "run_manifest.json", {
    "status": "completed", "evidence_role": "fixed_matrix_debugging_only", "command": [sys.executable, *sys.argv],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "tensorflow": tf.__version__, "gpu_uuid": args.gpu_uuid, "memory_policy": memory,
    "jit_compile": True, "tf32_enabled": True, "dtype": "float64",
    "allocator": tf.config.experimental.get_memory_info("GPU:0"), "seed_policy": "No new randomness; fixed saved matrix batch",
    "reference_path": str(reference_path), "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "identity": identity, "wall_seconds": time.monotonic() - started,
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result_file": str(OUTPUT / "eigensystem-result.json"), "nonclaims": ["No full-filter repair", "No tuning or sampler admission"],
})
print(json.dumps(reports), flush=True)
