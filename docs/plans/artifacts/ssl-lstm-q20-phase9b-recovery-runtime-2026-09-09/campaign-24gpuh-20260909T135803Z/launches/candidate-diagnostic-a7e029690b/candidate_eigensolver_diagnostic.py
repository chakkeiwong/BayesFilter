"""Bounded check of the integrated candidate; no tuning or sampler admission."""

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
if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid or os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("GPU pinning and memory growth must precede import")
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

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import BatchNativeSSLLSTMComplexityPosteriorTarget

spec = importlib.util.spec_from_file_location("candidate_recovery_diagnostic", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
job = json.loads((OUTPUT.parent / "strict-job.json").read_text())
sources = recovery.source_hashes()
if sources != job["sources"]:
    raise RuntimeError("Candidate source closure changed")
input_path = CAMPAIGN / "launches/cached-gradient-diagnostic-2dededa013/strict/transition-158-L2.json"
saved = json.loads(input_path.read_text())
parameters = tf.constant(saved["values"]["proposed_model_state"], tf.float64)
reference_path = Path(__file__).with_name("cpu-full-filter-r1") / "filter-result.json"
reference = json.loads(reference_path.read_text())["unchanged"]
reference_value = tf.constant(reference["value"], tf.float64)
reference_score = tf.constant(reference["score"], tf.float64)
reports = {}
durable_json(OUTPUT / "memory_policy.json", memory)
with DurableTensorCheckpoint(OUTPUT / "tensors", {"sources": sources, "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest()}) as store:
    for backend in ("tensorflow_eigh_strict", "tensorflow_eigh_strict_factor_cached"):
        target = BatchNativeSSLLSTMComplexityPosteriorTarget(20, jit_compile=True, principal_sqrt_backend=backend)

        @tf.function(input_signature=(tf.TensorSpec((4, 4), tf.float64),), jit_compile=True, autograph=False)
        def evaluate(batch):
            model, derivatives = target._batched_components(batch)
            value, score, diagnostics = core.tf_batched_svd_sigma_point_value_and_score(
                target.config.observations, model, derivatives, backend="tf_principal_sqrt_ukf",
                principal_sqrt_backend=backend,
            )
            return value, score, diagnostics["principal_sqrt_target_row_class_code"]

        before = time.monotonic()
        value, score, status = store.run(backend, {}, lambda: evaluate(parameters))
        if not bool(tf.reduce_all(status == 0).numpy()):
            raise RuntimeError("Repaired candidate endpoint still invalid")
        tf.debugging.assert_near(value, reference_value, atol=1e-10, rtol=1e-12)
        tf.debugging.assert_near(score, reference_score, atol=1e-9, rtol=1e-10)
        hlo = evaluate.experimental_get_compiler_ir(parameters)(stage="hlo")
        (OUTPUT / (backend + ".hlo.txt")).write_text(hlo)
        reports[backend] = {
            "row_class": status.numpy().tolist(), "value": value.numpy().tolist(), "score": score.numpy().tolist(),
            "maximum_value_residual": float(tf.reduce_max(tf.abs(value - reference_value)).numpy()),
            "maximum_scaled_score_residual": float(tf.reduce_max(tf.abs(score - reference_score) / tf.maximum(1.0, tf.abs(reference_score))).numpy()),
            "tracing_count": evaluate.experimental_get_tracing_count(), "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "seconds_including_compile_and_checkpoint": time.monotonic() - before,
        }
        if reports[backend]["tracing_count"] != 1:
            raise RuntimeError("Unexpected retracing")
    controls = tf.linalg.diag(tf.constant([[1e-13, 1.0, 1.0], [-1e-8, 1.0, 1.0]], tf.float64))

    @tf.function(input_signature=(tf.TensorSpec((2, 3, 3), tf.float64),), jit_compile=True, autograph=False)
    def control_solver(batch):
        return core._refined_symmetric_eigh(batch)

    eigenvalues, _ = store.run("odd_dimension_sign_controls", {}, lambda: control_solver(controls))
    if not bool(((eigenvalues[0, 0] > 0.0) & (eigenvalues[1, 0] < 0.0)).numpy()):
        raise RuntimeError("Eigensolver changed a control's sign")
if sources != recovery.source_hashes():
    raise RuntimeError("Candidate source closure changed during run")
durable_json(OUTPUT / "candidate-result.json", reports)
durable_json(OUTPUT / "run_manifest.json", {
    "status": "completed", "evidence_role": "candidate_numerical_repair_check_only",
    "sources": sources, "memory_policy": memory, "gpu_uuid": args.gpu_uuid,
    "tensorflow": tf.__version__, "jit_compile": True, "tf32_enabled": True,
    "allocator": tf.config.experimental.get_memory_info("GPU:0"),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "command": [sys.executable, *sys.argv], "seed_policy": "No new randomness",
    "data_sha256": hashlib.sha256(tf.io.serialize_tensor(target.config.observations).numpy()).hexdigest(),
    "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "wall_seconds": time.monotonic() - started,
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result_file": str(OUTPUT / "candidate-result.json"), "p1_launched": False, "tuning_launched": False,
})
print(json.dumps(reports), flush=True)
