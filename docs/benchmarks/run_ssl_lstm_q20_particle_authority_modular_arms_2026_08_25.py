"""Run explicitly labeled q20 modular-arm scaffolds on a fresh M0 pilot.

M1 is an affine finite-cloud moment scaffold, not the Acevedo ETPF. M2 is a
symmetric unscented sigma-point scaffold, not GenUT. M3 is only an affine
density scaffold and is not the canonical LEDH-PFPF route. M4 is an alias
diagnostic, not an implemented full ET-PF. None of these outputs can replace
the M0 authority or claim IID posterior samples.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("modular arm diagnostics require CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("modular arm diagnostics require TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("modular arm diagnostics found a visible GPU")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_modular_arms_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md"
CONTRACTS = ROOT / "bayesfilter/testing/particle_authority_contracts_tf.py"


class ModularArmError(RuntimeError):
    """Raised when a modular diagnostic cannot preserve its role boundary."""


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ModularArmError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="ascii")


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise ModularArmError(f"refusing to overwrite tensor: {path}")
    tensor = tf.convert_to_tensor(value)
    encoded = bytes(tf.io.serialize_tensor(tensor).numpy())
    path.write_bytes(encoded)
    return {"path": path.as_posix(), "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "dtype": tensor.dtype.name, "shape": list(tensor.shape)}


def _load_tensor(root: Path, receipt: Mapping[str, Any]) -> tf.Tensor:
    path = ROOT / str(receipt["path"])
    if not path.exists():
        raise ModularArmError(f"missing authority tensor: {path}")
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != receipt["sha256"]:
        raise ModularArmError(f"authority tensor hash mismatch: {path}")
    dtype = getattr(tf, str(receipt["dtype"]))
    return tf.io.parse_tensor(encoded, out_type=dtype)


def _load_authority(authority_root: Path) -> tuple[Mapping[str, Any], Mapping[str, tf.Tensor]]:
    pilot_path = authority_root / "pilot.json"
    if not pilot_path.exists():
        raise ModularArmError("authority pilot receipt is missing")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_GATE":
        raise ModularArmError("authority pilot did not pass its hard gates")
    arm = pilot["arms"].get("M0")
    if not isinstance(arm, Mapping):
        raise ModularArmError("M0 arm is missing")
    tensors = {name: _load_tensor(authority_root, receipt) for name, receipt in arm["receipts"].items()}
    required = {"final_theta", "final_normalized_weights", "final_target_log_prob", "final_proposal_log_prob"}
    if not required.issubset(tensors):
        raise ModularArmError("M0 receipt lacks required weighted cloud fields")
    return pilot, tensors


def _weighted_moments(points: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    weights = tf.convert_to_tensor(weights, tf.float64)
    points = tf.convert_to_tensor(points, tf.float64)
    weights = weights / tf.reduce_sum(weights)
    mean = tf.reduce_sum(weights[:, None] * points, axis=0)
    centered = points - mean[None, :]
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    return mean, 0.5 * (covariance + tf.transpose(covariance))


def _uniform_moments(points: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    mean = tf.reduce_mean(points, axis=0)
    centered = points - mean[None, :]
    count = tf.cast(tf.shape(points)[0], tf.float64)
    covariance = tf.einsum("ni,nj->ij", centered, centered) / count
    return mean, 0.5 * (covariance + tf.transpose(covariance))


def _m1_second_order_transform(points: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, Mapping[str, Any]]:
    weighted_mean, weighted_covariance = _weighted_moments(points, weights)
    source_mean, source_covariance = _uniform_moments(points)
    ridge = tf.constant(1.0e-8, tf.float64)
    source_chol = tf.linalg.cholesky(source_covariance + ridge * tf.eye(tf.shape(points)[1], dtype=tf.float64))
    target_chol = tf.linalg.cholesky(weighted_covariance + ridge * tf.eye(tf.shape(points)[1], dtype=tf.float64))
    centered = tf.transpose(points - source_mean[None, :])
    whitened = tf.linalg.triangular_solve(source_chol, centered, lower=True)
    transformed = tf.transpose(target_chol @ whitened) + weighted_mean[None, :]
    transformed_mean, transformed_covariance = _uniform_moments(transformed)
    mean_residual = tf.reduce_max(tf.abs(transformed_mean - weighted_mean))
    covariance_residual = tf.reduce_max(tf.abs(transformed_covariance - weighted_covariance))
    return transformed, {
        "weighted_mean": weighted_mean,
        "weighted_covariance": weighted_covariance,
        "transformed_mean": transformed_mean,
        "transformed_covariance": transformed_covariance,
        "max_mean_residual": mean_residual,
        "max_covariance_residual": covariance_residual,
        "ridge": ridge,
        "transform_kind": "affine_finite_cloud_second_order_moment_scaffold_not_etpf",
        "source_method_identity": "not_implemented_acevedo_second_order_etpf",
    }


def _m2_genut_sigma_points(points: tf.Tensor, weights: tf.Tensor) -> tuple[tf.Tensor, Mapping[str, Any]]:
    mean, covariance = _weighted_moments(points, weights)
    dimension = int(points.shape[1])
    chol = tf.linalg.cholesky(covariance + tf.constant(1.0e-8, tf.float64) * tf.eye(dimension, dtype=tf.float64))
    central_weight = tf.constant(0.10, tf.float64)
    side_weight = (1.0 - central_weight) / tf.cast(2 * dimension, tf.float64)
    scale = tf.sqrt(tf.cast(dimension, tf.float64) / (1.0 - central_weight))
    columns = tf.transpose(chol) * scale
    plus = mean[None, :] + columns
    minus = mean[None, :] - columns
    sigma_points = tf.concat((mean[None, :], plus, minus), axis=0)
    sigma_weights = tf.concat((central_weight[None], tf.fill((2 * dimension,), side_weight)), axis=0)
    sigma_mean = tf.reduce_sum(sigma_weights[:, None] * sigma_points, axis=0)
    centered = sigma_points - sigma_mean[None, :]
    sigma_covariance = tf.einsum("n,ni,nj->ij", sigma_weights, centered, centered)
    return sigma_points, {
        "sigma_weights": sigma_weights,
        "weighted_mean": mean,
        "weighted_covariance": covariance,
        "sigma_mean": sigma_mean,
        "sigma_covariance": sigma_covariance,
        "max_mean_residual": tf.reduce_max(tf.abs(sigma_mean - mean)),
        "max_covariance_residual": tf.reduce_max(tf.abs(sigma_covariance - covariance)),
        "point_count": 2 * dimension + 1,
        "transform_kind": "symmetric_ut_sigma_point_scaffold_not_genut",
        "source_method_identity": "not_implemented_ebeigbe_genut_skewness_kurtosis_constraints",
    }


def _evaluate_status(points: tf.Tensor, target: Any) -> Mapping[str, Any]:
    value, _score, status = target.neutra_batch_log_prob_and_grad_status(points)
    valid = tf.logical_and(status["status_code"] == 0, status["valid_pre_regularized_score"])
    return {
        "value": value,
        "valid": valid,
        "status_code": status["status_code"],
        "valid_count": tf.reduce_sum(tf.cast(valid, tf.int32)),
        "row_count": tf.size(valid),
    }


def _m3_affine_density_scaffold(points: tf.Tensor, proposal_log: tf.Tensor) -> Mapping[str, Any]:
    dimension = int(points.shape[1])
    center = tf.reduce_mean(points, axis=0)
    matrix = tf.constant(
        [[1.1, 0.08, 0.0, 0.0], [0.0, 0.9, 0.04, 0.0], [0.0, 0.0, 1.2, 0.05], [0.0, 0.0, 0.0, 0.8]],
        tf.float64,
    )
    if dimension != 4:
        raise ModularArmError("q20 affine scaffold expects four free coordinates")
    determinant = tf.linalg.det(matrix)
    transformed = center[None, :] + tf.matmul(points - center[None, :], matrix, transpose_b=True)
    inverse = tf.linalg.inv(matrix)
    recovered = center[None, :] + tf.matmul(transformed - center[None, :], inverse, transpose_b=True)
    inverse_log = proposal_log - tf.math.log(tf.abs(determinant))
    direct_log = proposal_log - tf.math.log(tf.abs(determinant))
    residual = tf.reduce_max(tf.abs(inverse_log - direct_log))
    return {
        "transformed": transformed,
        "recovered": recovered,
        "max_inverse_recovery_residual": tf.reduce_max(tf.abs(recovered - points)),
        "max_log_density_residual": residual,
        "determinant": determinant,
        "transform_kind": "affine_density_scaffold_not_canonical_ledh",
    }


def _arm_markdown(payload: Mapping[str, Any]) -> str:
    lines = [f"# {payload['arm']} Modular Arm Result", "", f"Status: `{payload['status']}`", "", f"Role: `{payload['role']}`", ""]
    lines.append("This artifact is auxiliary/scaffold evidence and cannot promote an authority or claim IID posterior samples.")
    lines.append("")
    lines.append("## Role boundary")
    lines.append("")
    lines.append(str(payload["role_boundary"]))
    lines.append("")
    lines.append("## Nonclaims")
    lines.append("")
    for item in payload["nonclaims"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.authority_root.is_absolute() or args.output_root.is_absolute() or ".." in args.output_root.parts:
        raise ModularArmError("paths must be repository-relative")
    if args.output_root.exists():
        raise ModularArmError(f"refusing to overwrite output root: {args.output_root}")
    args.output_root.mkdir(parents=True)
    started = time.perf_counter()
    pilot, tensors = _load_authority(ROOT / args.authority_root)
    points = tensors["final_theta"]
    weights = tensors["final_normalized_weights"]
    proposal_log = tensors["final_proposal_log_prob"]
    if points.shape.rank != 2 or points.shape[1] != 4:
        raise ModularArmError("authority cloud must have shape [N,4]")
    if not bool(tf.reduce_all(tf.math.is_finite(points)).numpy()):
        raise ModularArmError("authority cloud contains nonfinite values")
    target = batch_native_complexity_posterior_target(20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh")
    arms: dict[str, Any] = {}

    m1_points, m1_diag = _m1_second_order_transform(points, weights)
    m1_status = _evaluate_status(m1_points, target)
    m1 = {
        "schema": "bayesfilter.ssl_lstm.q20.modular_arm.v1",
        "arm": "M1",
        "status": "PASS_AUXILIARY_CONTRACT" if bool(tf.reduce_all(m1_status["valid"]).numpy()) and float(m1_diag["max_covariance_residual"].numpy()) < 1.0e-6 else "CANDIDATE_FAIL_REPAIR",
        "role": "finite_moment_transform_auxiliary",
        "role_boundary": "Second-order finite-cloud moment restoration; not a density, IID sample bank, or exact posterior authority.",
        "diagnostics": {**m1_diag, "target_valid_count": m1_status["valid_count"], "target_row_count": m1_status["row_count"]},
        "receipts": {"transformed_points": _write_tensor(args.output_root / "m1-transformed-points.tftensor", m1_points)},
        "nonclaims": ["No ETPF density identity is claimed for this affine scaffold.", "Bridge rows and covariance residuals are explanatory only."],
    }
    _write_json(args.output_root / "m1.json", m1)
    (args.output_root / "m1.md").write_text(_arm_markdown(m1), encoding="ascii")
    arms["M1"] = m1

    m2_points, m2_diag = _m2_genut_sigma_points(points, weights)
    m2_status = _evaluate_status(m2_points, target)
    m2 = {
        "schema": "bayesfilter.ssl_lstm.q20.modular_arm.v1",
        "arm": "M2",
        "status": "PASS_AUXILIARY_CONTRACT" if bool(tf.reduce_all(m2_status["valid"]).numpy()) and float(m2_diag["max_covariance_residual"].numpy()) < 1.0e-6 else "CANDIDATE_FAIL_REPAIR",
        "role": "symmetric_ut_sigma_point_auxiliary",
        "role_boundary": "Symmetric 2d+1 mean/covariance sigma-point diagnostic; not the GenUT skewness/kurtosis construction, a global multimodal density, or an IID replay bank.",
        "diagnostics": {**m2_diag, "target_valid_count": m2_status["valid_count"], "target_row_count": m2_status["row_count"]},
        "receipts": {"sigma_points": _write_tensor(args.output_root / "m2-sigma-points.tftensor", m2_points), "sigma_weights": _write_tensor(args.output_root / "m2-sigma-weights.tftensor", m2_diag["sigma_weights"])},
        "nonclaims": ["The sigma points do not establish global mode coverage.", "This symmetric rule is not a GenUT implementation and does not claim skewness or kurtosis matching.", "Moment residual does not establish density fidelity."],
    }
    _write_json(args.output_root / "m2.json", m2)
    (args.output_root / "m2.md").write_text(_arm_markdown(m2), encoding="ascii")
    arms["M2"] = m2

    m3_diag = _m3_affine_density_scaffold(points, proposal_log)
    m3 = {
        "schema": "bayesfilter.ssl_lstm.q20.modular_arm.v1",
        "arm": "M3",
        "status": "DESCRIPTIVE_ONLY",
        "role": "affine_density_scaffold",
        "role_boundary": "Affine change-of-variables diagnostic only; canonical LEDH-PFPF with Contract-E, UKF covariance lifecycle, and analytical recursive gradient is not implemented by this runner.",
        "diagnostics": m3_diag,
        "receipts": {"transformed_points": _write_tensor(args.output_root / "m3-affine-transformed-points.tftensor", m3_diag["transformed"])},
        "nonclaims": ["No canonical LEDH-PFPF admission.", "No q20 nonlinear flow determinant lifecycle claim."],
    }
    _write_json(args.output_root / "m3.json", m3)
    (args.output_root / "m3.md").write_text(_arm_markdown(m3), encoding="ascii")
    arms["M3"] = m3

    m4 = {
        "schema": "bayesfilter.ssl_lstm.q20.modular_arm.v1",
        "arm": "M4",
        "status": "DESCRIPTIVE_ONLY",
        "role": "approximate_etpf_comparator",
        "role_boundary": "The M1 affine scaffold is retained as an approximate finite-moment comparator; a full second-order ET-PF route is not implemented and this output cannot replace M0 or be called exact.",
        "diagnostics": {"source_cloud_uniform_moments": _uniform_moments(points), "m1_cloud_available": True},
        "nonclaims": ["No exact ET-PF theorem or posterior authority.", "No statistical ranking from this one pilot."],
    }
    _write_json(args.output_root / "m4.json", m4)
    (args.output_root / "m4.md").write_text(_arm_markdown(m4), encoding="ascii")
    arms["M4"] = m4

    receipt = {
        "schema": "bayesfilter.ssl_lstm.q20.modular_arms.v1",
        "status": "PASS_GATE" if all(arms[name]["status"] != "CANDIDATE_FAIL_REPAIR" for name in ("M1", "M2")) else "CANDIDATE_FAIL_REPAIR",
        "authority_input": args.authority_root.as_posix(),
        "pilot_protocol_hash": pilot["arms"]["M0"]["configuration"]["protocol_hash"],
        "target_signature": pilot["arms"]["M0"]["target_signature"],
        "arms": arms,
        "run_manifest": {
            "program": PLAN.as_posix(),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"runner": _sha(RUNNER), "plan": _sha(PLAN), "contracts": _sha(CONTRACTS)},
        },
        "nonclaims": [
            "M1/M2 are auxiliary finite-cloud diagnostics only.",
            "M3 is a scaffold and not canonical LEDH-PFPF.",
            "M4 is approximate by construction.",
            "No arm ranking, posterior correctness, HMC, or default promotion is claimed.",
        ],
    }
    _write_json(args.output_root / "modular-arms.json", receipt)
    (args.output_root / "result.md").write_text(
        "# Phase 3 Modular Arm Result\n\n"
        f"Status: `{receipt['status']}`\n\n"
        "M1 and M2 pass only their finite selected-moment/status contracts. M3 is an affine scaffold and M4 is an explicitly approximate comparator. The canonical LEDH-PFPF route remains outside this runner.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": receipt["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if receipt["status"] == "PASS_GATE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
