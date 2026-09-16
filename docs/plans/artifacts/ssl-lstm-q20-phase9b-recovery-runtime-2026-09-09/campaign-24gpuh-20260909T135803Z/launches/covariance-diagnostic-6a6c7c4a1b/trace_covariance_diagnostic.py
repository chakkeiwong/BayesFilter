"""Saved-endpoint covariance debugging; no tuning, target repair, or inference."""

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-uuid")
parser.add_argument("--output-dir", type=Path, required=True)
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != (args.gpu_uuid or "-1"):
    raise RuntimeError("Diagnostic device environment mismatch")
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

memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True) if args.gpu_uuid else None
if args.gpu_uuid:
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    if len(tf.config.list_logical_devices("GPU")) != 1:
        raise RuntimeError("Exactly one GPU is required")

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import BatchNativeSSLLSTMComplexityPosteriorTarget

spec = importlib.util.spec_from_file_location(
    "covariance_recovery_diagnostic",
    ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py",
)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
job = json.loads((CAMPAIGN / "launches/runtime-c48f1f0535/strict-job.json").read_text())
if recovery.source_hashes() != job["sources"]:
    raise RuntimeError("Scientific source closure changed")
input_path = CAMPAIGN / "launches/cached-gradient-diagnostic-2dededa013/strict/transition-158-L2.json"
saved = json.loads(input_path.read_text())
parameters = tf.constant(saved["values"]["proposed_model_state"], tf.float64)
target = BatchNativeSSLLSTMComplexityPosteriorTarget(
    20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh_strict",
)

FIELDS = {
    "incoming_augmented_covariance": "aug_covariance",
    "placement_minimum_eigenvalue": "placement.min_eigenvalue",
    "placement_invalid_count": "placement.classified_invalid_count",
    "predicted_covariance": "predicted_covariance",
    "updated_covariance": "covariance",
    "innovation_covariance": "innovation_factor.implemented_covariance",
    "raw_innovation_covariance": "raw_innovation_covariance",
    "cross_covariance": "cross_covariance",
    "kalman_gain": "kalman_gain",
    "centered_state_points": "centered_x",
    "centered_observation_points": "centered_y",
    "updated_covariance_derivative": "d_covariance",
    "placement_factor": "placement.factor",
}


def instrument_recursion():
    original = core.tf_batched_svd_sigma_point_value_and_score_with_rule
    parsed = ast.parse(inspect.getsource(original))
    function = parsed.body[0]
    loop = next(node for node in function.body if isinstance(node, ast.FunctionDef) and node.name == "_loop_body")
    loop.args.args.append(ast.arg(arg="diagnostic_history"))
    updates = ", ".join(
        f"diagnostic_history[{index}].write(t, tf.cast({expression}, tf.float64))"
        for index, expression in enumerate(FIELDS.values())
    )
    loop.body.insert(-1, ast.parse(f"diagnostic_history = ({updates},)").body[0])
    if not isinstance(loop.body[-1], ast.Return) or not isinstance(loop.body[-1].value, ast.Tuple):
        raise RuntimeError("Unexpected original loop return structure")
    loop.body[-1].value.elts.append(ast.Name(id="diagnostic_history", ctx=ast.Load()))
    assignments = [node for node in function.body if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call)
                   and ast.unparse(node.value.func) == "tf.while_loop"]
    if len(assignments) != 1:
        raise RuntimeError("Expected one original time recursion")
    assignment = assignments[0]
    assignment.targets[0].elts.append(ast.Name(id="diagnostic_history", ctx=ast.Store()))
    buffers = ast.parse(
        "(" + ", ".join("tf.TensorArray(tf.float64, size=n_timesteps, clear_after_read=False)" for _ in FIELDS) + ",)",
        mode="eval",
    ).body
    assignment.value.args[2].elts.append(buffers)
    if not isinstance(function.body[-1], ast.Return) or not isinstance(function.body[-1].value, ast.Tuple):
        raise RuntimeError("Unexpected original function return structure")
    function.body[-1].value.elts.append(ast.parse(
        "tuple(buffer.stack() for buffer in diagnostic_history)", mode="eval").body)
    ast.fix_missing_locations(parsed)
    generated = ast.unparse(parsed) + "\n"
    generated_path = OUTPUT / "instrumented_recursion_diagnostic.py"
    generated_path.write_text(generated)
    namespace = dict(vars(core))
    exec(compile(parsed, str(generated_path), "exec"), namespace)
    return namespace[function.name]


instrumented = instrument_recursion()


def compile_evaluation(callback, with_history):
    @tf.function(input_signature=(tf.TensorSpec((4, 4), tf.float64),), jit_compile=True, autograph=False)
    def evaluate(batch):
        model, derivatives = target._batched_components(batch)
        _, _, state_dim, innovation_dim, _ = core._check_model_derivative_shapes(model, derivatives)
        rule, backend = core._rule_for_backend("tf_principal_sqrt_ukf", state_dim + innovation_dim)
        result = callback(
            target.config.observations, model, derivatives, sigma_rule=rule,
            backend_name=backend, placement_floor=0.0, innovation_floor=1e-12,
            principal_sqrt_backend="tensorflow_eigh_strict",
        )
        value, score, diagnostics = result[:3]
        numeric_status = {key: item for key, item in diagnostics.items() if item.dtype != tf.string}
        history = dict(zip(FIELDS, result[3])) if with_history else {}
        return value, score, numeric_status, history, rule.covariance_weights, model.observation_covariance

    return evaluate


baseline = compile_evaluation(core.tf_batched_svd_sigma_point_value_and_score_with_rule, False)
diagnostic = compile_evaluation(instrumented, True)
identity = {
    "original_sources": job["sources"],
    "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
}
durable_json(OUTPUT / "memory_policy.json", memory)
with DurableTensorCheckpoint(OUTPUT / "tensors", identity) as store:
    baseline_result = store.run("unchanged", {"parameters": saved["values"]["proposed_model_state"]}, lambda: baseline(parameters))
    diagnostic_result = store.run("instrumented", {"parameters": saved["values"]["proposed_model_state"]}, lambda: diagnostic(parameters))

history = diagnostic_result[3]
weights = diagnostic_result[4]
state_points = history["centered_state_points"]
observation_points = history["centered_observation_points"]
gain = history["kalman_gain"]
innovation = history["innovation_covariance"]
cross = history["cross_covariance"]
updated = history["updated_covariance"]
predicted = history["predicted_covariance"]
if not all(bool(tf.reduce_all(tf.math.is_finite(value)).numpy()) for value in history.values()):
    raise RuntimeError("Saved covariance diagnostic contains nonfinite tensors")

residual_points = state_points - tf.einsum("tbnm,tbrm->tbrn", gain, observation_points)
observation_point_covariance = tf.einsum("r,tbri,tbrj->tbij", weights, observation_points, observation_points)
effective_noise = innovation - observation_point_covariance
residual_covariance = tf.einsum("r,tbri,tbrj->tbij", weights, residual_points, residual_points)
residual_covariance += gain @ effective_noise @ tf.linalg.matrix_transpose(gain)
residual_covariance = core._symmetrize(residual_covariance)
gain_times_innovation = gain @ innovation
subtraction = gain_times_innovation @ tf.linalg.matrix_transpose(gain)
independent_subtraction = core._symmetrize(predicted - subtraction)
solve_residual = gain_times_innovation - cross


def minimum_spectrum(matrix):
    return tf.reduce_min(tf.linalg.eigvalsh(core._symmetrize(matrix)), axis=-1)


def materialize(value):
    return tf.nest.map_structure(lambda item: item.numpy().tolist(), value)


summary = {
    "original_failed_chain_one_based": 4,
    "original_failure_minimum_eigenvalue": saved["values"]["proposed_status"]["min_placement_eigenvalue"],
    "unchanged_status": materialize(baseline_result[2]),
    "instrumented_status": materialize(diagnostic_result[2]),
    "values_exact": bool(tf.reduce_all(baseline_result[0] == diagnostic_result[0]).numpy()),
    "scores_exact": bool(tf.reduce_all(baseline_result[1] == diagnostic_result[1]).numpy()),
    "maximum_value_residual": float(tf.reduce_max(tf.abs(baseline_result[0] - diagnostic_result[0])).numpy()),
    "maximum_score_residual": float(tf.reduce_max(tf.abs(baseline_result[1] - diagnostic_result[1])).numpy()),
    "invalid_events_zero_based_time_chain": tf.where(history["placement_invalid_count"] > 0).numpy().tolist(),
    "covariance_weight_minimum": float(tf.reduce_min(weights).numpy()),
    "per_time_chain": materialize({
        "incoming_min_eigenvalue": minimum_spectrum(history["incoming_augmented_covariance"]),
        "placement_reported_min_eigenvalue": history["placement_minimum_eigenvalue"],
        "predicted_min_eigenvalue": minimum_spectrum(predicted),
        "updated_min_eigenvalue": minimum_spectrum(updated),
        "residual_form_min_eigenvalue": minimum_spectrum(residual_covariance),
        "effective_noise_min_eigenvalue": minimum_spectrum(effective_noise),
        "predicted_max_abs": tf.reduce_max(tf.abs(predicted), axis=(-2, -1)),
        "subtraction_max_abs": tf.reduce_max(tf.abs(subtraction), axis=(-2, -1)),
        "updated_max_abs": tf.reduce_max(tf.abs(updated), axis=(-2, -1)),
        "covariance_derivative_max_abs": tf.reduce_max(tf.abs(history["updated_covariance_derivative"]), axis=(-3, -2, -1)),
        "gain_solve_residual_norm": tf.linalg.norm(solve_residual, axis=(-2, -1)),
        "residual_form_difference_norm": tf.linalg.norm(residual_covariance - updated, axis=(-2, -1)),
        "independent_subtraction_difference_norm": tf.linalg.norm(independent_subtraction - updated, axis=(-2, -1)),
    }),
    "nonclaims": ["No target repair", "No tuning or inference", "Instrumentation equality is measured, not assumed"],
}
durable_json(OUTPUT / "covariance-summary.json", recovery.diagnostic_payload(summary))
if recovery.source_hashes() != job["sources"]:
    raise RuntimeError("Scientific source closure changed during diagnostic")
durable_json(OUTPUT / "run_manifest.json", {
    "status": "completed_debugging_only", "command": [sys.executable, *sys.argv],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "environment": {key: os.environ.get(key) for key in ("CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "tensorflow": tf.__version__, "gpu_uuid": args.gpu_uuid, "memory_policy": memory,
    "gpu_devices_intentionally_hidden": not args.gpu_uuid, "jit_compile": True,
    "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
    "allocator": tf.config.experimental.get_memory_info("GPU:0") if args.gpu_uuid else None,
    "seed_policy": "No new randomness; saved finite parameter batch",
    "input_path": str(input_path), "provenance": identity,
    "data_sha256": hashlib.sha256(tf.io.serialize_tensor(target.config.observations).numpy()).hexdigest(),
    "generated_source_sha256": hashlib.sha256((OUTPUT / "instrumented_recursion_diagnostic.py").read_bytes()).hexdigest(),
    "wall_seconds": time.monotonic() - started, "output_root": str(OUTPUT),
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result_file": str(OUTPUT / "covariance-summary.json"),
})
print(json.dumps({key: summary[key] for key in ("values_exact", "scores_exact", "invalid_events_zero_based_time_chain")}), flush=True)
