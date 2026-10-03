"""Isolated full-filter/score debugging; never a tuning or admitted target route."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

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
        raise RuntimeError("Exactly one GPU required")

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import BatchNativeSSLLSTMComplexityPosteriorTarget
from jacobi_refinement_diagnostic import refine_eigh

spec = importlib.util.spec_from_file_location("filter_recovery_diagnostic", ROOT / "docs/benchmarks/run_ssl_lstm_q20_phase9b_recovery_runtime_2026_09_09.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)
job = json.loads((CAMPAIGN / "launches/runtime-c48f1f0535/strict-job.json").read_text())
if recovery.source_hashes() != job["sources"]:
    raise RuntimeError("Scientific source closure changed")
input_path = CAMPAIGN / "launches/cached-gradient-diagnostic-2dededa013/strict/transition-158-L2.json"
saved = json.loads(input_path.read_text())
parameters = tf.constant(saved["values"]["proposed_model_state"], tf.float64)
target = BatchNativeSSLLSTMComplexityPosteriorTarget(20, jit_compile=bool(args.gpu_uuid), principal_sqrt_backend="tensorflow_eigh_strict")


class DiagnosticLinalg:
    def __init__(self, sweeps):
        self.sweeps = sweeps

    def __getattr__(self, name):
        return getattr(tf.linalg, name)

    def eigh(self, matrix):
        if matrix.shape[-1] == 1:
            return tf.linalg.diag_part(matrix), tf.ones_like(matrix)
        return refine_eigh(matrix, sweeps=self.sweeps)


class DiagnosticTensorFlow:
    def __init__(self, sweeps):
        self.linalg = DiagnosticLinalg(sweeps)

    def __getattr__(self, name):
        return getattr(tf, name)


def cloned_recursion(sweeps):
    namespace = dict(vars(core))
    namespace["tf"] = DiagnosticTensorFlow(sweeps)
    for name, function in vars(core).items():
        if isinstance(function, types.FunctionType) and function.__module__ == core.__name__:
            copied = types.FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
            copied.__kwdefaults__ = function.__kwdefaults__
            namespace[name] = copied
    return namespace["tf_batched_svd_sigma_point_value_and_score"]


def compile_evaluation(callback):
    @tf.function(input_signature=(tf.TensorSpec((4, 4), tf.float64),), jit_compile=bool(args.gpu_uuid), autograph=False)
    def evaluate(batch):
        model, derivatives = target._batched_components(batch)
        value, score, diagnostics = callback(
            target.config.observations, model, derivatives, backend="tf_principal_sqrt_ukf",
            placement_floor=0.0, innovation_floor=1e-12, principal_sqrt_backend="tensorflow_eigh_strict",
        )
        return value, score, {key: item for key, item in diagnostics.items() if item.dtype != tf.string}

    return evaluate


evaluators = {"unchanged": compile_evaluation(core.tf_batched_svd_sigma_point_value_and_score)}
if args.gpu_uuid:
    evaluators.update({"refined_three": compile_evaluation(cloned_recursion(3)),
                       "refined_four": compile_evaluation(cloned_recursion(4))})
identity = {"sources": job["sources"], "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
            "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "refinement_source_sha256": hashlib.sha256(Path(__file__).with_name("jacobi_refinement_diagnostic.py").read_bytes()).hexdigest()}
durable_json(OUTPUT / "memory_policy.json", memory)
reports = {}
with DurableTensorCheckpoint(OUTPUT / "tensors", identity) as store:
    for name, evaluate in evaluators.items():
        before = time.monotonic()
        value, score, diagnostics = store.run(name, {}, lambda: evaluate(parameters))
        first_seconds = time.monotonic() - before
        report = {
            "value": value.numpy().tolist(), "score": score.numpy().tolist(),
            "status": tf.nest.map_structure(lambda item: item.numpy().tolist(), diagnostics),
            "first_call_with_compile_checkpoint_seconds": first_seconds,
        }
        reports[name] = report
        durable_json(OUTPUT / (name + "-result.json"), recovery.diagnostic_payload(report))
        if name == "refined_three" or (args.gpu_uuid and name == "unchanged"):
            continue
        finite_differences = {}
        for step in (1e-4, 3e-5):
            columns = []
            statuses = []
            for coordinate in range(4):
                offset = tf.one_hot(coordinate, 4, dtype=tf.float64)[None] * step
                step_key = str(step).replace(".", "p")
                plus = store.run(f"{name}-plus-{step_key}-{coordinate}", {}, lambda: evaluate(parameters + offset))
                minus = store.run(f"{name}-minus-{step_key}-{coordinate}", {}, lambda: evaluate(parameters - offset))
                columns.append((plus[0] - minus[0]) / (2.0 * step))
                statuses.append({"plus": plus[2]["principal_sqrt_target_row_class_code"].numpy().tolist(),
                                 "minus": minus[2]["principal_sqrt_target_row_class_code"].numpy().tolist()})
            numerical_score = tf.stack(columns, axis=-1)
            relative_error = tf.abs(numerical_score - score) / tf.maximum(1.0, tf.abs(score))
            finite_differences[str(step)] = {"score": numerical_score.numpy().tolist(),
                                           "scaled_error": relative_error.numpy().tolist(), "statuses": statuses,
                                           "maximum_scaled_error": float(tf.reduce_max(relative_error).numpy())}
            durable_json(OUTPUT / (name + "-finite-difference-" + str(step).replace(".", "p") + ".json"), recovery.diagnostic_payload(finite_differences[str(step)]))
        report["finite_differences"] = finite_differences
        warmed_seconds = []
        for _ in range(2):
            before = time.monotonic()
            tf.reduce_sum(evaluate(parameters)[0]).numpy()
            warmed_seconds.append(time.monotonic() - before)
        report["warmed_call_seconds"] = warmed_seconds
durable_json(OUTPUT / "filter-result.json", recovery.diagnostic_payload(reports))
if recovery.source_hashes() != job["sources"]:
    raise RuntimeError("Scientific source closure changed during diagnostic")
durable_json(OUTPUT / "run_manifest.json", {
    "status": "completed", "evidence_role": "full_filter_debugging_only", "command": [sys.executable, *sys.argv],
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "tensorflow": tf.__version__, "gpu_uuid": args.gpu_uuid, "memory_policy": memory,
    "jit_compile": bool(args.gpu_uuid), "gpu_devices_intentionally_hidden": not args.gpu_uuid,
    "jit_exception": None if args.gpu_uuid else "Independent CPU tiny full-filter reference",
    "allocator": tf.config.experimental.get_memory_info("GPU:0") if args.gpu_uuid else None,
    "seed_policy": "No new randomness; saved parameter batch and deterministic coordinate perturbations",
    "data_sha256": hashlib.sha256(tf.io.serialize_tensor(target.config.observations).numpy()).hexdigest(),
    "identity": identity, "wall_seconds": time.monotonic() - started,
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result_file": str(OUTPUT / "filter-result.json"), "nonclaims": ["No runtime source change", "No tuning or sampler admission"],
})
print(json.dumps({name: {key: report[key] for key in ("value", "first_call_with_compile_checkpoint_seconds")} for name, report in reports.items()}), flush=True)
