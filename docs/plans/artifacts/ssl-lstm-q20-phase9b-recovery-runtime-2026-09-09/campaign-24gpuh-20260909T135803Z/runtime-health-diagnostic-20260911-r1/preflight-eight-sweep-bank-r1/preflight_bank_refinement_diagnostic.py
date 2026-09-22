"""Same-input CPU reference and eight-sweep XLA prototype, with no training."""

import argparse
import functools
import hashlib
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
args = parser.parse_args()
if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("CPU-only diagnostic requires intentionally hidden GPUs")
ROOT = Path(__file__).resolve().parents[6]
OUTPUT = args.output_dir.resolve()
OUTPUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT))
started = time.monotonic()
core_path = ROOT / "bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py"
source_hash = hashlib.sha256(core_path.read_bytes()).hexdigest()
identity = {"core_sha256": source_hash,
            "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
manifest = {
    "status": "started", "started_unix": time.time(), "identity": identity,
    "command": [sys.executable, *sys.argv], "wall_cap_seconds": 900,
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "environment_python": sys.executable,
    "environment": {key: os.environ.get(key) for key in (
        "CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS")},
    "gpu_intentionally_hidden": True, "gpu_worker_seconds": 0,
    "candidate_sweeps": 8, "candidate_cpu_jit_compile": True,
    "reference_cpu_jit_compile": False, "reference_eigensolver": "native_cpu_tf.linalg.eigh",
    "optimizer_updates": 0, "sampler_transitions": 0,
    "plan": "docs/plans/bayesfilter-ssl-lstm-q20-phase9b-runtime-health-repair-2026-09-11.md",
    "result": str(OUTPUT / "result.json"), "seeds": "N/A: saved parameter banks; no new random draws",
    "evidence_role": "CPU_numerical_equivalence_not_GPU_readiness", "inputs": {},
}
(OUTPUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
shutil.copy2(__file__, OUTPUT / Path(__file__).name)

import tensorflow as tf
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint, durable_json
from bayesfilter.nonlinear import experimental_batched_svd_sigma_point_tf as core
from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import BatchNativeSSLLSTMComplexityPosteriorTarget


original_refinement = core._refined_symmetric_eigh
reports = {}
manifest["tensorflow"] = tf.__version__


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def host(tensors):
    return tf.nest.map_structure(lambda tensor: tensor.numpy().tolist(), tensors)


with DurableTensorCheckpoint(OUTPUT / "tensors", identity) as store:
    for arm, backend in (
        ("factor", "tensorflow_eigh_strict_factor_cached"),
        ("strict", "tensorflow_eigh_strict"),
    ):
        input_path = Path(__file__).parent / "cpu-preflight-native-r1" / f"{arm}-status.json"
        payload = json.loads(input_path.read_text())
        parameters = tf.constant(payload["physical"], tf.float64)
        manifest["inputs"][arm] = {"path": str(input_path),
                                  "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                                  "batch_size": int(parameters.shape[0])}
        outputs = {}
        for mode in ("native", "prototype"):
            jit_compile = mode == "prototype"
            core._refined_symmetric_eigh = (
                functools.partial(original_refinement, sweeps=8)
                if jit_compile else lambda matrix, **kwargs: tf.linalg.eigh(matrix)
            )
            target = BatchNativeSSLLSTMComplexityPosteriorTarget(
                20, jit_compile=jit_compile, principal_sqrt_backend=backend,
            )
            manifest["data_sha256"] = hashlib.sha256(tf.io.serialize_tensor(target.config.observations).numpy()).hexdigest()
            durable_json(OUTPUT / "run_manifest.json", manifest)
            checkpoint_started = time.monotonic()
            output = store.run(arm + "-" + mode, {"input": manifest["inputs"][arm]},
                lambda: target.batch_prior_likelihood_value_score_status(parameters))
            outputs[mode] = output
            likelihood, score, prior, prior_score, status = output
            invalid = tf.where(~status["valid_pre_regularized_score"])[:, 0].numpy().tolist()
            reports.setdefault(arm, {})[mode] = {
                "invalid_rows": invalid, "status": host(status),
                "likelihood": host(likelihood), "analytic_score": host(score),
                "wall_seconds": time.monotonic() - checkpoint_started,
            }
            durable_json(OUTPUT / "result.json", json_safe(reports))
            print(json.dumps({"arm": arm, "mode": mode, "invalid_rows": invalid,
                              "wall_seconds": reports[arm][mode]["wall_seconds"]}), flush=True)
        native, prototype = outputs["native"], outputs["prototype"]
        value_difference = tf.abs(native[0] - prototype[0])
        score_difference = tf.abs(native[1] - prototype[1])
        checks = {
            "all_rows_valid": not reports[arm]["native"]["invalid_rows"] and not reports[arm]["prototype"]["invalid_rows"],
            "likelihood_agreement": bool(tf.reduce_all(value_difference <= 1e-10 + 1e-12 * tf.abs(native[0])).numpy()),
            "analytic_score_agreement": bool(tf.reduce_all(score_difference <= 1e-9 + 1e-10 * tf.abs(native[1])).numpy()),
            "maximum_likelihood_residual": float(tf.reduce_max(value_difference).numpy()),
            "maximum_scaled_score_residual": float(tf.reduce_max(score_difference / (1.0 + tf.abs(native[1]))).numpy()),
        }
        checks["passed"] = all(checks[key] for key in ("all_rows_valid", "likelihood_agreement", "analytic_score_agreement"))
        reports[arm]["comparison"] = checks
        durable_json(OUTPUT / "result.json", json_safe(reports))
        print(json.dumps({"arm": arm, "comparison": checks}), flush=True)
core._refined_symmetric_eigh = original_refinement
if source_hash != hashlib.sha256(core_path.read_bytes()).hexdigest():
    raise RuntimeError("Numerical source changed during diagnostic")
manifest.update(status="completed", wall_seconds=time.monotonic() - started,
                passed=all(reports[arm]["comparison"]["passed"] for arm in reports))
durable_json(OUTPUT / "run_manifest.json", manifest)
print(json.dumps({"status": manifest["status"], "passed": manifest["passed"],
                  "wall_seconds": manifest["wall_seconds"]}), flush=True)
raise SystemExit(0 if manifest["passed"] else 1)
