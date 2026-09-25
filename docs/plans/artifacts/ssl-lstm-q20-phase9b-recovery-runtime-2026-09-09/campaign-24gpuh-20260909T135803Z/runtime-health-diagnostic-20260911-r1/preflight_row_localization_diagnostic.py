"""CPU-only diagnostic copies of the fixed-row filter and eigenpair checks."""

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--whole-bank", action="store_true")
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("This reference/localization diagnostic requires hidden GPUs")
ROOT = Path(__file__).resolve().parents[6]
CAMPAIGN = Path(__file__).resolve().parents[1]
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT))
started = time.monotonic()
input_path = Path(__file__).parent / "cpu-xla-preflight-r1/factor-status.json"
input_payload = json.loads(input_path.read_text())
selected = input_payload["physical"] if args.whole_bank else [input_payload["physical"][15]]
identity = {
    "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
    "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "core_sha256": hashlib.sha256((ROOT / "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py").read_bytes()).hexdigest(),
}
manifest = {
    "status": "started", "started_unix": time.time(), "identity": identity,
    "command": [sys.executable, *sys.argv], "wall_cap_seconds": 300,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "environment": {key: os.environ.get(key) for key in (
        "CONDA_PREFIX", "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
        "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "environment_python": sys.executable,
    "gpu_intentionally_hidden": True, "gpu_worker_seconds": 0,
    "cpu_jit_compile": True, "seeds": "N/A: fixed saved physical parameters; no random draws",
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result": str(OUTPUT / "result.json"),
    "batch_size": len(selected), "original_row_zero_based": 15,
    "whole_bank_replay": args.whole_bank, "optimizer_updates": 0, "sampler_transitions": 0,
    "evidence_role": "numerical_localization_only_not_gpu_clearance",
}
(OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
shutil.copy2(__file__, OUTPUT / Path(__file__).name)

import tensorflow as tf
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import BatchNativeSSLLSTMComplexityPosteriorTarget


parameters = tf.constant(selected, tf.float64)
target = BatchNativeSSLLSTMComplexityPosteriorTarget(
    20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh_strict_factor_cached",
)
FIELDS = {
    "incoming_covariance": "aug_covariance",
    "placement_minimum": "placement.min_eigenvalue",
    "placement_invalid_count": "placement.classified_invalid_count",
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
    loop.body[-1].value.elts.append(ast.Name(id="diagnostic_history", ctx=ast.Load()))
    assignments = [node for node in function.body if isinstance(node, ast.Assign)
                   and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "tf.while_loop"]
    if len(assignments) != 1:
        raise RuntimeError("Unexpected time recursion structure")
    assignment = assignments[0]
    assignment.targets[0].elts.append(ast.Name(id="diagnostic_history", ctx=ast.Store()))
    assignment.value.args[2].elts.append(ast.parse(
        "(" + ", ".join("tf.TensorArray(tf.float64, size=n_timesteps, clear_after_read=False)" for _ in FIELDS) + ",)",
        mode="eval").body)
    function.body[-1].value.elts.append(ast.parse(
        "tuple(buffer.stack() for buffer in diagnostic_history)", mode="eval").body)
    ast.fix_missing_locations(parsed)
    generated_path = OUTPUT / "instrumented_recursion_diagnostic.py"
    generated_path.write_text(ast.unparse(parsed) + "\n")
    namespace = dict(vars(core))
    exec(compile(parsed, str(generated_path), "exec"), namespace)
    return namespace[function.name]


def instrument_refinement():
    parsed = ast.parse(inspect.getsource(core._refined_symmetric_eigh))
    function = parsed.body[0]
    function.body[-1] = ast.parse(
        "return values, vectors, valid, residual_norm, matrix_norm, "
        "tf.linalg.norm(orthogonality, axis=[-2, -1]), roundoff_bound, working"
    ).body[0]
    ast.fix_missing_locations(parsed)
    generated_path = OUTPUT / "instrumented_refinement_diagnostic.py"
    generated_path.write_text(ast.unparse(parsed) + "\n")
    namespace = dict(vars(core))
    exec(compile(parsed, str(generated_path), "exec"), namespace)
    return namespace[function.name]


def compile_filter(callback, with_history):
    @tf.function(input_signature=(tf.TensorSpec((len(selected), 4), tf.float64),), jit_compile=True, autograph=False)
    def evaluate(batch):
        model, derivatives = target._batched_components(batch)
        _, _, state_dim, innovation_dim, _ = core._check_model_derivative_shapes(model, derivatives)
        rule, backend = core._rule_for_backend("tf_principal_sqrt_ukf", state_dim + innovation_dim)
        result = callback(
            target.config.observations, model, derivatives, sigma_rule=rule,
            backend_name=backend, placement_floor=0.0, innovation_floor=1e-12,
            principal_sqrt_backend="tensorflow_eigh_strict_factor_cached",
        )
        return result[0], result[1], {
            key: value for key, value in result[2].items() if value.dtype != tf.string
        }, dict(zip(FIELDS, result[3])) if with_history else {}
    return evaluate


baseline = compile_filter(core.tf_batched_svd_sigma_point_value_and_score_with_rule, False)
instrumented = compile_filter(instrument_recursion(), True)
refinement = instrument_refinement()
manifest["tensorflow"] = tf.__version__
manifest["data_sha256"] = hashlib.sha256(tf.io.serialize_tensor(target.config.observations).numpy()).hexdigest()
durable_json(OUTPUT / "run_manifest.json", manifest)


def host(tensors):
    return tf.nest.map_structure(lambda tensor: tensor.numpy().tolist(), tensors)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


with DurableTensorCheckpoint(OUTPUT / "tensors", identity) as store:
    baseline_result = store.run("baseline", {"parameters": selected}, lambda: baseline(parameters))
    print(json.dumps({"stage": "baseline", "status": host(baseline_result[2])}), flush=True)
    traced_result = store.run("instrumented", {"parameters": selected}, lambda: instrumented(parameters))
    history = traced_result[3]
    invalid_events = tf.where(history["placement_invalid_count"] > 0).numpy().tolist()
    report = {
        "invalid_events_time_row_zero_based": invalid_events,
        "baseline_status": host(baseline_result[2]), "instrumented_status": host(traced_result[2]),
        "value_exact": bool(tf.reduce_all(baseline_result[0] == traced_result[0]).numpy()),
        "score_exact": bool(tf.reduce_all(baseline_result[1] == traced_result[1]).numpy()),
        "max_value_residual": float(tf.reduce_max(tf.abs(baseline_result[0] - traced_result[0])).numpy()),
        "max_score_residual": float(tf.reduce_max(tf.abs(baseline_result[1] - traced_result[1])).numpy()),
        "per_time_minimum": host(history["placement_minimum"]), "refinement": {},
    }
    durable_json(OUTPUT / "result.json", json_safe(report))
    print(json.dumps({"stage": "instrumented", "invalid_events": invalid_events,
                      "value_exact": report["value_exact"], "score_exact": report["score_exact"]}), flush=True)
    if invalid_events:
        failed_time, failed_row = invalid_events[0]
        matrix = history["incoming_covariance"][failed_time, failed_row:failed_row + 1]
        store.run("failed-covariance", {}, lambda: matrix)
        native_values, native_vectors = tf.linalg.eigh(core._symmetrize(matrix))
        native_residual = matrix @ native_vectors - native_vectors * native_values[:, tf.newaxis, :]
        report["native_cpu"] = {
            "minimum_eigenvalue": float(tf.reduce_min(native_values).numpy()),
            "maximum_eigenvalue": float(tf.reduce_max(native_values).numpy()),
            "matrix_norm": float(tf.linalg.norm(matrix).numpy()),
            "residual_norm": float(tf.linalg.norm(native_residual).numpy()),
        }
        for sweep_count in (1, 4, 8, 12):
            @tf.function(input_signature=(tf.TensorSpec(matrix.shape, tf.float64),), jit_compile=True, autograph=False)
            def check_refinement(batch):
                return refinement(batch, sweeps=sweep_count)
            result = store.run(f"refinement-{sweep_count}", {}, lambda: check_refinement(matrix))
            values, vectors, valid, residual_norm, matrix_norm, orthogonality_norm, bound, working = result
            report["refinement"][str(sweep_count)] = {
                "valid": host(valid), "minimum_eigenvalue": float(tf.reduce_min(values).numpy()),
                "residual_norm": host(residual_norm), "matrix_norm": host(matrix_norm),
                "orthogonality_norm": host(orthogonality_norm), "bound": host(bound),
                "remaining_offdiagonal_norm": float(tf.linalg.norm(working - tf.linalg.diag(tf.linalg.diag_part(working))).numpy()),
            }
            durable_json(OUTPUT / "result.json", json_safe(report))
            print(json.dumps({"sweeps": sweep_count, **report["refinement"][str(sweep_count)]}), flush=True)

if identity["core_sha256"] != hashlib.sha256((ROOT / "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py").read_bytes()).hexdigest():
    raise RuntimeError("Numerical core changed during diagnostic")
manifest.update(status="completed", wall_seconds=time.monotonic() - started)
durable_json(OUTPUT / "run_manifest.json", manifest)
print(json.dumps({"status": "completed", "wall_seconds": manifest["wall_seconds"]}), flush=True)
