"""Probe GenUT feasibility in the corrected four-parameter theta measure."""

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
    raise RuntimeError("Phase 30 requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 30 requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("Phase 30 found a visible GPU in the reference lane")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.particle_authority_genut_tf import generalized_unscented_transform


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase30_2026_08_25.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
GENUT_MODULE = ROOT / "bayesfilter/testing/particle_authority_genut_tf.py"


class Phase30Error(RuntimeError):
    """Raised when the GenUT scope probe cannot preserve its receipt."""


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
        raise Phase30Error(f"refusing to overwrite artifact: {path}")
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _write_tensor(path: Path, value: Any) -> Mapping[str, Any]:
    if path.exists():
        raise Phase30Error(f"refusing to overwrite artifact: {path}")
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
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase30Error(f"tensor digest mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _evaluate_sigma(target: Any, points: tf.Tensor) -> Mapping[str, Any]:
    value, score, status = target.neutra_batch_log_prob_and_grad_status(points)
    valid = tf.logical_and(
        tf.equal(tf.convert_to_tensor(status["status_code"], tf.int32), 0),
        tf.cast(status["valid_pre_regularized_score"], tf.bool),
    )
    return {
        "finite": tf.logical_and(
            tf.reduce_all(tf.math.is_finite(value)),
            tf.reduce_all(tf.math.is_finite(score)),
        ),
        "all_valid": tf.reduce_all(valid),
        "valid_count": tf.reduce_sum(tf.cast(valid, tf.int32)),
        "row_count": tf.size(valid),
        "values": value,
        "scores": score,
        "status_code": status["status_code"],
        "valid_pre_regularized_score": status["valid_pre_regularized_score"],
    }


def _run_scope(name: str, points: tf.Tensor, weights: tf.Tensor, target: Any) -> Mapping[str, Any]:
    sigma_points, sigma_weights, diagnostics = generalized_unscented_transform(
        points, weights, ridge=1.0e-10
    )
    finite = bool(
        tf.reduce_all(tf.math.is_finite(sigma_points)).numpy()
        and tf.reduce_all(tf.math.is_finite(sigma_weights)).numpy()
    )
    feasible = bool(diagnostics["feasible"].numpy()) if finite else False
    payload: dict[str, Any] = {
        "scope": name,
        "input_count": int(points.shape[0]),
        "dimension": int(points.shape[1]),
        "finite": finite,
        "feasible": feasible,
        "diagnostics": diagnostics,
        "sigma_points": sigma_points,
        "sigma_weights": sigma_weights,
        "target_evaluated": False,
    }
    if feasible:
        target_eval = _evaluate_sigma(target, sigma_points)
        payload["target_evaluated"] = True
        payload["target"] = target_eval
        payload["target_all_valid"] = bool(target_eval["all_valid"].numpy())
        payload["status"] = (
            "PASS_PARAMETER_GENUT_ROLE_LIMITED"
            if bool(target_eval["finite"].numpy()) and payload["target_all_valid"]
            else "PARAMETER_GENUT_TARGET_FAIL_REPAIR"
        )
    else:
        payload["target_all_valid"] = False
        payload["status"] = "PARAMETER_GENUT_INFEASIBLE_SCOPE"
    return payload


def _markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Corrected q=20 Parameter-Space GenUT",
        "",
        f"Status: `{result['status']}`",
        "",
        "GenUT was evaluated in theta in R^4. Any global infeasibility is a scope result, not a theorem about all local uses.",
        "",
        "| Scope | Status | Feasible | Target valid |",
        "|---|---|---:|---:|",
    ]
    for name, payload in result["scopes"].items():
        lines.append(f"| {name} | `{payload['status']}` | `{payload['feasible']}` | `{payload['target_all_valid']}` |")
    lines.extend(
        [
            "",
            "## Nonclaims",
            "",
            "- sigma points are not IID posterior samples and receive no density claim",
            "- moment residuals do not establish mode discovery or posterior correctness",
            "- no clipping or negative-weight repair was applied",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    for path in (args.authority_root, args.output_root):
        if path.is_absolute() or ".." in path.parts:
            raise Phase30Error("paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase30Error(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    pilot_path = ROOT / args.authority_root / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase30Error("Phase 28 pilot is not a passing theta-measure receipt")
    m0 = pilot["arms"]["M0"]
    if m0["protocol"].get("measure") != "theta_R4":
        raise Phase30Error("M0 measure is not theta_R4")
    points = _load_tensor(m0["receipts"]["final_theta"])
    weights = _load_tensor(m0["receipts"]["final_normalized_weights"])
    if points.shape.rank != 2 or points.shape[1] != 4:
        raise Phase30Error(f"unexpected theta bank shape: {points.shape}")
    if weights.shape != (points.shape[0],):
        raise Phase30Error("theta bank weight shape mismatch")
    weights = weights / tf.reduce_sum(weights)
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    scopes: dict[str, Any] = {"global_theta_R4": _run_scope("global_theta_R4", points, weights, target)}
    global_feasible = bool(scopes["global_theta_R4"]["feasible"])
    # Local sign scopes are run only when the global construction is infeasible,
    # so they diagnose scope rather than silently replacing the global result.
    if not global_feasible:
        negative = points[:, 2] < 0.0
        for name, mask in (("negative_axis2_theta_R4", negative), ("positive_axis2_theta_R4", tf.logical_not(negative))):
            indices = tf.reshape(tf.where(mask), [-1])
            if int(indices.shape[0]) < 2:
                scopes[name] = {"scope": name, "status": "PARAMETER_GENUT_TOO_FEW_ROWS", "feasible": False, "target_all_valid": False}
                continue
            local_points = tf.gather(points, indices)
            local_weights = tf.gather(weights, indices)
            local_weights = local_weights / tf.reduce_sum(local_weights)
            scopes[name] = _run_scope(name, local_points, local_weights, target)
    if scopes["global_theta_R4"]["status"] == "PASS_PARAMETER_GENUT_ROLE_LIMITED":
        status_value = "PASS_PARAMETER_GENUT_ROLE_LIMITED"
    elif scopes["global_theta_R4"]["status"] == "PARAMETER_GENUT_INFEASIBLE_SCOPE":
        status_value = "PARAMETER_GENUT_GLOBAL_INFEASIBLE_SCOPE"
    else:
        status_value = "PARAMETER_GENUT_PROBE_FAIL_REPAIR"
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.corrected_theta_genut_scope.v1",
        "status": status_value,
        "role": "parameter_space_genut_scope_diagnostic",
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(pilot_path),
            "protocol_hash": m0["configuration"]["protocol_hash"],
            "target_signature": m0["target_signature"],
            "measure": "theta_R4",
            "dimension": 4,
        },
        "scopes": scopes,
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
                "genut_module": _sha(GENUT_MODULE),
                "authority_pilot": _sha(pilot_path),
            },
        },
        "nonclaims": [
            "GenUT is a moment/quadrature diagnostic, not a density or IID generator.",
            "Global infeasibility is scoped to this finite theta cloud and controls.",
            "No authority, posterior, mode-discovery, whitening, LEDH, HMC, or default claim.",
        ],
    }
    # Keep a compact receipt plus per-scope tensor artifacts, avoiding any
    # caller interpretation of sigma points as a persistent particle bank.
    for name, payload in scopes.items():
        if "sigma_points" in payload:
            payload.setdefault("receipts", {})["sigma_points"] = _write_tensor(output / f"{name}-sigma-points.tftensor", payload["sigma_points"])
            payload.setdefault("receipts", {})["sigma_weights"] = _write_tensor(output / f"{name}-sigma-weights.tftensor", payload["sigma_weights"])
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": status_value, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
