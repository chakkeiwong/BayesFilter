"""Audit the corrected q=20 parameter/state measure boundary.

This is a CPU-hidden, XLA reference diagnostic.  The only particle variable
constructed here is theta in R^4.  The q=20 UKF state and innovation dimensions
are reported as internal metadata and are never passed to ETPF.
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
    raise RuntimeError("Phase 27 requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 27 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("Phase 27 found a visible GPU in the reference lane")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.particle_authority_etpf_tf import (
    second_order_etpf_transform,
)


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase27_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
ETPF_MODULE = ROOT / "bayesfilter/testing/particle_authority_etpf_tf.py"

_LOG_TWO_PI = tf.constant(1.8378770664093453, tf.float64)
_SEED = (20260825, 2701)
_POINT_COUNT = 16
_TOLERANCE = 2.0e-10


class Phase27Error(RuntimeError):
    """Raised when the corrected contract cannot produce an auditable receipt."""


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
        raise Phase27Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _gaussian_log_density(points: tf.Tensor, center: tf.Tensor, std: float) -> tf.Tensor:
    points = tf.convert_to_tensor(points, tf.float64)
    center = tf.convert_to_tensor(center, tf.float64)
    scale = tf.constant(float(std), tf.float64)
    dimension = tf.cast(tf.shape(points)[1], tf.float64)
    standardized = (points - center[tf.newaxis, :]) / scale
    return -0.5 * tf.reduce_sum(tf.square(standardized), axis=1) - dimension * (
        tf.math.log(scale) + 0.5 * _LOG_TWO_PI
    )


def _target_status(target: Any, theta: tf.Tensor) -> Mapping[str, Any]:
    values, score, status = target.neutra_batch_log_prob_and_grad_status(theta)
    valid = tf.logical_and(
        tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
        tf.cast(status["valid_pre_regularized_score"], tf.bool),
    )
    finite = tf.logical_and(
        tf.reduce_all(tf.math.is_finite(values)),
        tf.reduce_all(tf.math.is_finite(score)),
    )
    return {
        "values": values,
        "scores": score,
        "status": status,
        "valid": valid,
        "finite": finite,
        "all_valid": tf.reduce_all(valid),
    }


def _affine_measure_contract(target_values: tf.Tensor, theta: tf.Tensor, center: tf.Tensor) -> Mapping[str, Any]:
    """Check chart cancellation while keeping theta-density labels explicit."""

    matrix = tf.constant(
        [[1.40, 0.15, 0.00, 0.00], [0.05, 0.90, 0.10, 0.00],
         [0.00, 0.05, 1.20, 0.20], [0.00, 0.00, 0.10, 0.80]],
        tf.float64,
    )
    determinant = tf.linalg.det(matrix)
    if not bool(tf.math.is_finite(determinant).numpy()) or float(determinant.numpy()) == 0.0:
        raise Phase27Error("affine chart matrix is singular or non-finite")
    z = tf.transpose(tf.linalg.solve(matrix, tf.transpose(theta - center[tf.newaxis, :])))
    recovered = center[tf.newaxis, :] + tf.matmul(z, matrix, transpose_b=True)
    round_trip = tf.reduce_max(tf.abs(recovered - theta))
    proposal_theta = _gaussian_log_density(theta, center, 2.0)
    # The target values are already log densities in theta measure.
    log_abs_det = tf.math.log(tf.abs(determinant))
    target_chart = target_values + log_abs_det
    proposal_chart = proposal_theta + log_abs_det
    theta_ratio = target_values - proposal_theta
    chart_ratio = target_chart - proposal_chart
    ratio_residual = tf.reduce_max(tf.abs(theta_ratio - chart_ratio))
    return {
        "matrix": matrix,
        "determinant": determinant,
        "log_abs_det": log_abs_det,
        "round_trip_residual": round_trip,
        "max_ratio_residual": ratio_residual,
        "target_log_theta": target_values,
        "proposal_log_theta": proposal_theta,
        "target_log_chart": target_chart,
        "proposal_log_chart": proposal_chart,
        "ratio_theta": theta_ratio,
        "ratio_chart": chart_ratio,
    }


def _make_theta_cloud(target: Any, count: int) -> tf.Tensor:
    center = tf.convert_to_tensor(target.config.prior_center, tf.float64)
    split = tf.random.experimental.stateless_split(tf.constant(_SEED, tf.int32), 2)
    noise = tf.random.stateless_normal((count, 4), seed=split[0], dtype=tf.float64)
    # This is a bounded diagnostic cloud around the declared prior center, not
    # a claimed posterior sample or an inherited authority bank.
    theta = center[tf.newaxis, :] + tf.constant(0.35, tf.float64) * noise
    return tf.ensure_shape(theta, [count, 4])


def _markdown(receipt: Mapping[str, Any]) -> str:
    gates = receipt["hard_gates"]
    lines = [
        "# Corrected q=20 Parameter Measure Contract",
        "",
        f"Status: `{receipt['status']}`",
        "",
        "The particle variable is theta in R^4. The UKF state and innovation are internal target-evaluation dimensions only.",
        "",
        "| Gate | Result |",
        "|---|---|",
    ]
    for key, value in gates.items():
        lines.append(f"| {key} | `{value}` |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "This receipt can admit the corrected shape/measure boundary to the next diagnostic phase only. It does not admit an SMC authority, an IID law, a posterior claim, or LEDH/HMC status.",
            "",
            "## Nonclaims",
            "",
            "- ETPF moment residuals do not define a density or IID samples.",
            "- A finite target batch does not prove mode discovery or posterior correctness.",
            "- The canonical LEDH rebuild remains separate and deferred.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--points", type=int, default=_POINT_COUNT)
    args = parser.parse_args()
    if args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise Phase27Error("output root must be repository-relative")
    if int(args.points) < 8:
        raise Phase27Error("at least eight theta rows are required")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase27Error(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()

    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    theta = _make_theta_cloud(target, int(args.points))
    if theta.shape != (int(args.points), 4):
        raise Phase27Error(f"particle shape is not [N,4]: {theta.shape}")
    target_eval = _target_status(target, theta)
    center = tf.convert_to_tensor(target.config.prior_center, tf.float64)
    chart = _affine_measure_contract(target_eval["values"], theta, center)

    weights = tf.nn.softmax(target_eval["values"] - tf.reduce_max(target_eval["values"]))
    analysis, etpf = second_order_etpf_transform(
        theta,
        weights,
        regularization=10.0,
        sinkhorn_steps=200,
        riccati_step=0.1,
        riccati_max_steps=500,
        riccati_tolerance=1.0e-4,
    )
    transformed_eval = _target_status(target, analysis)
    state_dim = int(target.config.static_config.augmented_state_dim)
    innovation_dim = int(target.config.static_config.latent_dim)
    hard_gates = {
        "theta_particle_shape_is_N_by_4": theta.shape == (int(args.points), 4),
        "target_parameter_dim_is_4": int(target.parameter_dim) == 4,
        "internal_state_dimension_recorded_not_particle_dimension": state_dim == 60 and state_dim != 4,
        "internal_innovation_dimension_recorded": innovation_dim == 20,
        "target_finite": bool(target_eval["finite"].numpy()),
        "target_status_valid": bool(target_eval["all_valid"].numpy()),
        "affine_round_trip": float(chart["round_trip_residual"].numpy()) <= _TOLERANCE,
        "chart_ratio_cancellation": float(chart["max_ratio_residual"].numpy()) <= _TOLERANCE,
        "etpf_output_shape_is_N_by_4": analysis.shape == (int(args.points), 4),
        "etpf_output_finite": bool(tf.reduce_all(tf.math.is_finite(analysis)).numpy()),
        "etpf_target_finite": bool(transformed_eval["finite"].numpy()),
        "etpf_target_status_valid": bool(transformed_eval["all_valid"].numpy()),
    }
    status = "PASS_CORRECTED_PARAMETER_MEASURE_CONTRACT" if all(hard_gates.values()) else "PHASE27_FAIL_REPAIR"
    receipt = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_parameter_measure_contract.v1",
        "status": status,
        "role": "parameter_space_shape_measure_and_etpf_boundary_diagnostic",
        "target": {
            "q": 20,
            "parameter_dim": int(target.parameter_dim),
            "target_signature": target.target_signature(),
            "adapter_signature": target.adapter_signature(),
            "target_scope": target.target_scope,
            "batch_size": int(args.points),
            "target_valid_count": int(tf.reduce_sum(tf.cast(target_eval["valid"], tf.int32)).numpy()),
            "target_status_codes": target_eval["status"]["status_code"],
        },
        "measure": {
            "particle_variable": "theta in R^4",
            "internal_ukf_state_dim": state_dim,
            "internal_innovation_dim": innovation_dim,
            "theta_density_ratio_definition": "target_log_theta - proposal_log_theta",
            "chart_ratio_definition": "(target_log_theta+log_abs_det_A)-(proposal_log_theta+log_abs_det_A)",
            "chart": chart,
        },
        "etpf": {
            "input_shape": theta.shape,
            "output_shape": analysis.shape,
            "weights": weights,
            "diagnostics": etpf,
            "transformed_target_status_codes": transformed_eval["status"]["status_code"],
        },
        "hard_gates": hard_gates,
        "diagnostics": {
            "source_mode_fraction_axis2_negative": tf.reduce_mean(tf.cast(theta[:, 2] < 0.0, tf.float64)),
            "analysis_mode_fraction_axis2_negative": tf.reduce_mean(tf.cast(analysis[:, 2] < 0.0, tf.float64)),
            "etpf_negative_entry_fraction": etpf["corrected_negative_fraction"],
            "etpf_covariance_residual": etpf["covariance_residual"],
            "etpf_riccati_converged": etpf["riccati_converged"],
        },
        "run_manifest": {
            "program": PLAN.as_posix(),
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
            "seed": list(_SEED),
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {
                "runner": _sha(RUNNER),
                "plan": _sha(PLAN),
                "etpf_module": _sha(ETPF_MODULE),
            },
        },
        "nonclaims": [
            "The finite theta cloud is not an SMC-U authority or posterior estimate.",
            "ETPF moment and target/status checks do not provide an empirical density or IID law.",
            "The internal 60D state and 20D innovation are not particle coordinates.",
            "No canonical LEDH, NeuTra training, HMC, mode-discovery, whitening, or default claim.",
        ],
    }
    _write_json(output / "result.json", receipt)
    (output / "result.md").write_text(_markdown(receipt), encoding="ascii")
    print(json.dumps({"status": status, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if status == "PASS_CORRECTED_PARAMETER_MEASURE_CONTRACT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
