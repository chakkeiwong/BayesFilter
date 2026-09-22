"""Fixed-bank GPU/XLA numerical diagnostic; no optimizer or sampler."""

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
    raise RuntimeError("UUID pinning and growth must precede TensorFlow import")
ROOT = Path(__file__).resolve().parents[6]
DIAGNOSTIC = Path(__file__).resolve().parent
CAMPAIGN = DIAGNOSTIC.parent
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT))
started = time.monotonic()

import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
tf.config.set_soft_device_placement(False)
if len(tf.config.list_logical_devices("GPU")) != 1:
    raise RuntimeError("Exactly one logical GPU required")

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import BatchNativeSSLLSTMComplexityPosteriorTarget


spec = importlib.util.spec_from_file_location("gpu_bank_recovery", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
job = json.loads((OUTPUT.parent / "job.json").read_bytes())
sources = recovery.source_hashes()
if sources != job["sources"] or core._refined_symmetric_eigh.__kwdefaults__["sweeps"] != 8:
    raise RuntimeError("Unexpected current source closure or sweep cap")
reference_path = DIAGNOSTIC / "preflight-eight-sweep-bank-r1/result.json"
reference = json.loads(reference_path.read_bytes())
reference_manifest = json.loads((reference_path.parent / "run_manifest.json").read_bytes())
validation_path = DIAGNOSTIC / "eight-sweep-validation-a8df05dcf2/validation.json"
validation = json.loads(validation_path.read_bytes())
for relative, expected in validation["artifact_sha256"].items():
    if relative.startswith("preflight-eight-sweep-bank-r1/") and hashlib.sha256((DIAGNOSTIC / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError("Saved CPU reference checksum changed")
endpoint_path = CAMPAIGN / "launches/cached-gradient-diagnostic-2dededa013/strict/transition-158-L2.json"
endpoint_parameters = json.loads(endpoint_path.read_bytes())["values"]["proposed_model_state"]
endpoint_reference_path = DIAGNOSTIC / "cpu-full-filter-r1/filter-result.json"
endpoint_reference = json.loads(endpoint_reference_path.read_bytes())["unchanged"]
inputs = {}
reports = {}
manifest = {
    "status": "started", "sources": sources, "memory_policy": memory,
    "command": [sys.executable, *sys.argv], "python": sys.executable,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "gpu_uuid": args.gpu_uuid, "tensorflow": tf.__version__, "jit_compile": True, "tf32_enabled": True,
    "environment": {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "seed_policy": "Saved physical banks; no new randomness", "optimizer_updates": 0, "sampler_transitions": 0,
    "evidence_role": "fixed_bank_numerical_equivalence_only", "trust_basis": "trusted_escalated_gpu_execution",
    "inputs": inputs, "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    "endpoint_reference_sha256": hashlib.sha256(endpoint_reference_path.read_bytes()).hexdigest(),
    "endpoint_input_sha256": hashlib.sha256(endpoint_path.read_bytes()).hexdigest(),
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result_file": str(OUTPUT / "result.json"), "data_sha256": reference_manifest["data_sha256"],
    "nonclaims": ["No tuning or sampling", "No convergence or ranking", "No universal eigensolver guarantee"],
}
durable_json(OUTPUT / "run_manifest.json", manifest)

with DurableTensorCheckpoint(OUTPUT / "tensors", {"sources": sources, "gpu_uuid": args.gpu_uuid}) as store:
    for arm, backend in (("factor", "tensorflow_eigh_strict_factor_cached"), ("strict", "tensorflow_eigh_strict")):
        input_path = DIAGNOSTIC / "cpu-preflight-native-r1" / f"{arm}-status.json"
        input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if input_hash != reference_manifest["inputs"][arm]["sha256"]:
            raise RuntimeError("Fixed physical input bank changed")
        inputs[arm] = {"path": str(input_path), "sha256": input_hash}
        physical = json.loads(input_path.read_bytes())["physical"]
        cases = (
            ("bank", physical, reference[arm]["native"]["likelihood"], reference[arm]["native"]["analytic_score"]),
            ("endpoint", endpoint_parameters, endpoint_reference["value"], endpoint_reference["score"]),
        )
        target = BatchNativeSSLLSTMComplexityPosteriorTarget(20, jit_compile=True, principal_sqrt_backend=backend)
        if hashlib.sha256(tf.io.serialize_tensor(target.config.observations).numpy()).hexdigest() != manifest["data_sha256"]:
            raise RuntimeError("Observation data changed")
        for label, bank, native_value, native_score in cases:
            with tf.device("/GPU:0"):
                parameters = tf.constant(bank, tf.float64)
                output = store.run(arm + "-" + label, {"physical": bank},
                    lambda: target.batch_prior_likelihood_value_score_status(parameters))
            value, score, _, _, status = output
            expected_value = tf.constant(native_value, tf.float64)
            expected_score = tf.constant(native_score, tf.float64)
            compiled = target._compiled_component_batches[len(bank)]
            report = {
                "invalid_rows": tf.where(~status["valid_pre_regularized_score"])[:, 0].numpy().tolist(),
                "value_agreement": bool(tf.reduce_all(tf.abs(value - expected_value) <= 1e-10 + 1e-12 * tf.abs(expected_value)).numpy()),
                "score_agreement": bool(tf.reduce_all(tf.abs(score - expected_score) <= 1e-9 + 1e-10 * tf.abs(expected_score)).numpy()),
                "maximum_value_residual": float(tf.reduce_max(tf.abs(value - expected_value)).numpy()),
                "maximum_scaled_score_residual": float(tf.reduce_max(tf.abs(score - expected_score) / (1.0 + tf.abs(expected_score))).numpy()),
                "tracing_count": compiled.experimental_get_tracing_count(), "device": value.device,
            }
            hlo = compiled.experimental_get_compiler_ir(parameters)(stage="hlo")
            (OUTPUT / f"{arm}-{label}.hlo.txt").write_text(hlo)
            report["hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
            report["passed"] = not report["invalid_rows"] and report["value_agreement"] and report["score_agreement"] and report["tracing_count"] == 1 and "GPU:0" in report["device"]
            reports[arm + "-" + label] = report
            durable_json(OUTPUT / "result.json", recovery.diagnostic_payload(reports))
            print(json.dumps({"case": arm + "-" + label, **report}), flush=True)

    @tf.function(input_signature=(tf.TensorSpec((2, 3, 3), tf.float64),), jit_compile=True, autograph=False)
    def check_sign(batch):
        return core._refined_symmetric_eigh(batch)

    with tf.device("/GPU:0"):
        controls = tf.linalg.diag(tf.constant([[1e-13, 1.0, 1.0], [-1e-8, 1.0, 1.0]], tf.float64))
        values, _ = store.run("odd-sign-controls", {}, lambda: check_sign(controls))
    sign_passed = bool(((values[0, 0] > 0.0) & (values[1, 0] < 0.0)).numpy())
if sources != recovery.source_hashes():
    raise RuntimeError("Source closure changed during GPU validation")
manifest.update(status="completed", passed=sign_passed and all(row["passed"] for row in reports.values()),
                sign_controls_passed=sign_passed, allocator=tf.config.experimental.get_memory_info("GPU:0"),
                wall_seconds=time.monotonic() - started)
durable_json(OUTPUT / "run_manifest.json", manifest)
print(json.dumps({"passed": manifest["passed"], "wall_seconds": manifest["wall_seconds"]}), flush=True)
raise SystemExit(0 if manifest["passed"] else 1)
