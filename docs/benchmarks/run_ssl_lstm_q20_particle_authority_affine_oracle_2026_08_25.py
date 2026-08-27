"""Compute an exact weighted affine-whitening oracle for one pilot bank."""

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
    raise RuntimeError("affine oracle requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("affine oracle requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("affine oracle found a visible GPU")

RUNNER = Path(__file__).resolve()
TOLERANCE = 1.0e-10


class OracleError(RuntimeError):
    """Raised when the finite affine oracle cannot be evaluated."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise OracleError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="ascii")


def _load(path: Path, dtype: Any) -> Any:
    if not path.is_file():
        raise OracleError(f"missing tensor: {path}")
    return tf.io.parse_tensor(tf.convert_to_tensor(path.read_bytes()), out_type=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.pilot_root.is_absolute() or args.output_root.is_absolute() or ".." in args.pilot_root.parts or ".." in args.output_root.parts:
        raise OracleError("paths must be repository-relative")
    pilot_root = ROOT / args.pilot_root
    output_root = ROOT / args.output_root
    if output_root.exists():
        raise OracleError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    pilot = json.loads((pilot_root / "pilot.json").read_text(encoding="utf-8"))
    arm = pilot["arms"]["M0"]
    receipts = arm["receipts"]
    paths = {}
    for name in ("final_theta", "final_normalized_weights"):
        stored = Path(str(receipts[name]["path"]))
        candidates = (stored, ROOT / stored, pilot_root / stored.name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None or _sha(path) != receipts[name]["sha256"]:
            raise OracleError(f"missing or hash-invalid receipt: {name}")
        paths[name] = path
    theta = _load(paths["final_theta"], tf.float64)
    weights = _load(paths["final_normalized_weights"], tf.float64)
    n = int(theta.shape[0])
    weights = weights / tf.reduce_sum(weights)
    mean = tf.reduce_sum(theta * weights[:, tf.newaxis], axis=0)
    centered = theta - mean
    covariance = tf.einsum("n,ni,nj->ij", weights, centered, centered)
    chol = tf.linalg.cholesky(covariance)
    whitened = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(centered)))
    whitened_mean = tf.reduce_sum(whitened * weights[:, tf.newaxis], axis=0)
    whitened_centered = whitened - whitened_mean
    whitened_covariance = tf.einsum("n,ni,nj->ij", weights, whitened_centered, whitened_centered)
    identity = tf.eye(4, dtype=tf.float64)
    offdiag = whitened_covariance - tf.linalg.diag(tf.linalg.diag_part(whitened_covariance))
    gates = {
        "finite_input": bool(tf.reduce_all(tf.math.is_finite(theta)).numpy() and tf.reduce_all(tf.math.is_finite(weights)).numpy()),
        "positive_definite_covariance": bool(tf.reduce_all(tf.linalg.diag_part(chol) > 0.0).numpy()),
        "whitened_mean_exact": float(tf.reduce_max(tf.abs(whitened_mean)).numpy()) <= TOLERANCE,
        "whitened_covariance_exact": float(tf.reduce_max(tf.abs(whitened_covariance - identity)).numpy()) <= TOLERANCE,
    }
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.affine_oracle.v1",
        "status": "PASS_AFFINE_ORACLE" if all(gates.values()) else "AFFINE_ORACLE_FAIL",
        "pilot_root": pilot_root.as_posix(),
        "particle_count": n,
        "gates": gates,
        "diagnostics": {
            "input_weight_sum": tf.reduce_sum(weights),
            "input_weighted_mean": mean,
            "input_covariance": covariance,
            "whitened_weighted_mean": whitened_mean,
            "whitened_covariance": whitened_covariance,
            "whitened_max_abs_mean": tf.reduce_max(tf.abs(whitened_mean)),
            "whitened_max_abs_covariance_residual": tf.reduce_max(tf.abs(whitened_covariance - identity)),
            "whitened_max_abs_offdiag": tf.reduce_max(tf.abs(offdiag)),
        },
        "run_manifest": {
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "wall_seconds": time.perf_counter() - started,
        },
        "nonclaims": [
            "The affine map matches finite weighted first and second moments only.",
            "It does not prove target density correctness, mode discovery, IID law, posterior correctness, or HMC readiness.",
        ],
    }
    _write_json(output_root / "result.json", result)
    (output_root / "result.md").write_text("# Weighted Affine Oracle\n\n" + f"Status: `{result['status']}`\n", encoding="ascii")
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] == "PASS_AFFINE_ORACLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
