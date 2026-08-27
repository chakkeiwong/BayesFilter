"""Apply source-faithful ETPF to a fresh theta-space q=20 bank.

The transformed empirical rows are evaluated by the q=20 target only for a
role-limited finite/status check.  No density or IID law is assigned to the
ETPF empirical transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Phase 29 requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 29 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("Phase 29 found a visible GPU in the reference lane")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.particle_authority_etpf_tf import second_order_etpf_transform


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase29_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
ETPF_MODULE = ROOT / "bayesfilter/testing/particle_authority_etpf_tf.py"


class Phase29Error(RuntimeError):
    """Raised when the fresh-bank ETPF receipt cannot be validated."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(item) for item in value.as_list()]
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase29Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise Phase29Error(f"refusing to overwrite artifact: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    path.write_bytes(encoded)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    observed = hashlib.sha256(encoded).hexdigest()
    if observed != str(receipt["sha256"]):
        raise Phase29Error(f"tensor digest mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Corrected q=20 Fresh-Theta ETPF",
        "",
        f"Status: `{result['status']}`",
        "",
        "The source-faithful ETPF map acted on a fresh theta in R^4 bank. The empirical transform has no assigned density or IID claim.",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for key, value in result["hard_gates"].items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This is a role-limited ETPF integration receipt. It can refresh a GenUT scope decision, but it cannot replace the particle authority or define a density.",
            "",
            "## Nonclaims",
            "",
            "- transformed rows are not asserted IID or posterior draws",
            "- no empirical-transform proposal density or Jacobian is used",
            "- no mode-discovery, whitening, HMC, LEDH, or default claim",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--subset", type=int, default=32)
    parser.add_argument("--riccati-tolerance", type=float, default=1.0e-5)
    parser.add_argument("--riccati-max-steps", type=int, default=2000)
    args = parser.parse_args()
    for path in (args.authority_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase29Error("paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase29Error(f"refusing to overwrite output root: {output}")
    if int(args.subset) < 8:
        raise Phase29Error("subset must contain at least eight rows")
    output.mkdir(parents=True)
    started = time.perf_counter()

    authority = ROOT / args.authority_root
    pilot_path = authority / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase29Error("Phase 28 pilot is not an admitted finite theta-measure receipt")
    m0 = pilot["arms"].get("M0")
    if not isinstance(m0, Mapping):
        raise Phase29Error("M0 arm is missing")
    if m0.get("protocol", {}).get("measure") != "theta_R4":
        raise Phase29Error("M0 measure binding is not theta_R4")
    if m0.get("target_signature") != m0.get("protocol", {}).get("target_signature"):
        raise Phase29Error("M0 target signature/protocol mismatch")
    source_points = _load_tensor(m0["receipts"]["final_theta"])
    source_weights = _load_tensor(m0["receipts"]["final_normalized_weights"])
    if source_points.shape.rank != 2 or source_points.shape[1] != 4:
        raise Phase29Error(f"source particle shape is not [N,4]: {source_points.shape}")
    if source_weights.shape != (source_points.shape[0],):
        raise Phase29Error("source weight shape mismatch")
    tf.debugging.assert_non_negative(source_weights, "source weights")
    tf.debugging.assert_near(tf.reduce_sum(source_weights), tf.constant(1.0, tf.float64), atol=1.0e-10)
    count = int(source_points.shape[0])
    subset_count = min(int(args.subset), count)
    indices = [round(index * (count - 1) / (subset_count - 1)) for index in range(subset_count)]
    if len(set(indices)) != subset_count:
        raise Phase29Error("subset indices are not unique")
    index_tensor = tf.constant(indices, tf.int32)
    points = tf.gather(source_points, index_tensor)
    weights = tf.gather(source_weights, index_tensor)
    weights = weights / tf.reduce_sum(weights)

    analysis, diagnostics = second_order_etpf_transform(
        points,
        weights,
        regularization=10.0,
        sinkhorn_steps=400,
        riccati_step=0.1,
        riccati_max_steps=int(args.riccati_max_steps),
        riccati_tolerance=float(args.riccati_tolerance),
    )
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    values, scores, status = target.neutra_batch_log_prob_and_grad_status(analysis)
    valid = tf.logical_and(
        tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
        tf.cast(status["valid_pre_regularized_score"], tf.bool),
    )
    source_min = tf.reduce_min(points, axis=0)
    source_max = tf.reduce_max(points, axis=0)
    below = tf.reduce_sum(tf.cast(analysis < source_min[tf.newaxis, :], tf.int32))
    above = tf.reduce_sum(tf.cast(analysis > source_max[tf.newaxis, :], tf.int32))
    hard_gates = {
        "source_shape_N_by_4": source_points.shape == (count, 4),
        "source_weights_normalized": abs(float((tf.reduce_sum(source_weights) - 1.0).numpy())) <= 1.0e-10,
        "finite_transform": bool(tf.reduce_all(tf.math.is_finite(analysis)).numpy()),
        "riccati_converged": bool(tf.convert_to_tensor(diagnostics["riccati_converged"]).numpy()),
        "row_residual": float(diagnostics["corrected_row_residual"].numpy()) <= 2.0e-6,
        "column_residual": float(diagnostics["corrected_column_residual"].numpy()) <= 2.0e-6,
        "target_finite": bool(tf.reduce_all(tf.math.is_finite(values)).numpy()),
        "score_finite": bool(tf.reduce_all(tf.math.is_finite(scores)).numpy()),
        "target_status_valid": bool(tf.reduce_all(valid).numpy()),
        "output_shape_subset_by_4": analysis.shape == (subset_count, 4),
    }
    status_value = "PASS_FRESH_THETA_ETPF_ROLE_LIMITED" if all(hard_gates.values()) else "PHASE29_ETPF_FAIL_REPAIR"
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_etpf.v1",
        "status": status_value,
        "role": "source_faithful_etpf_fresh_theta_role_limited",
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(pilot_path),
            "protocol_hash": m0["configuration"]["protocol_hash"],
            "target_signature": m0["target_signature"],
            "measure": "theta_R4",
            "source_particle_count": count,
        },
        "subset": {"indices": indices, "count": subset_count},
        "controls": {
            "regularization": 10.0,
            "sinkhorn_steps": 400,
            "riccati_tolerance": float(args.riccati_tolerance),
            "riccati_max_steps": int(args.riccati_max_steps),
            "tolerance_provenance": "phase19_source_fixture_repair_warm_start",
        },
        "hard_gates": hard_gates,
        "diagnostics": {
            "source_weighted_mean": diagnostics["weighted_mean"],
            "analysis_mean": diagnostics["analysis_mean"],
            "mean_residual": diagnostics["mean_residual"],
            "covariance_residual": diagnostics["covariance_residual"],
            "negative_correction_fraction": diagnostics["corrected_negative_fraction"],
            "riccati_iterations": diagnostics["riccati_iterations"],
            "below_source_range_count": below,
            "above_source_range_count": above,
            "source_negative_mode_fraction": tf.reduce_mean(tf.cast(points[:, 2] < 0.0, tf.float64)),
            "analysis_negative_mode_fraction": tf.reduce_mean(tf.cast(analysis[:, 2] < 0.0, tf.float64)),
        },
        "target": {
            "values": values,
            "scores": scores,
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status["valid_pre_regularized_score"],
        },
        "receipts": {
            "source_subset_theta": _write_tensor(output / "source-subset-theta.tftensor", points),
            "source_subset_weights": _write_tensor(output / "source-subset-weights.tftensor", weights),
            "analysis_theta": _write_tensor(output / "analysis-theta.tftensor", analysis),
            "target_values": _write_tensor(output / "analysis-target-values.tftensor", values),
            "target_scores": _write_tensor(output / "analysis-target-scores.tftensor", scores),
        },
        "run_manifest": {
            "program": PLAN.as_posix(),
            "runner": RUNNER.as_posix(),
            "command": " ".join(sys.argv),
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "plan": _sha(PLAN),
                "runner": _sha(RUNNER),
                "etpf_module": _sha(ETPF_MODULE),
                "authority_pilot": _sha(pilot_path),
            },
        },
        "nonclaims": [
            "The transformed empirical rows have no assigned proposal density or Jacobian.",
            "ETPF moment and target/status checks do not establish IID samples, posterior correctness, or mode discovery.",
            "No SMC-U authority admission, LEDH, NeuTra, HMC, whitening, or default promotion.",
        ],
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": status_value, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if status_value == "PASS_FRESH_THETA_ETPF_ROLE_LIMITED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
