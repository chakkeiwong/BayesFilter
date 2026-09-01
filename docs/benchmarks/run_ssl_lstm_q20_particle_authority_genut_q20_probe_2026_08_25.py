"""Run an actual-bank q20 GenUT feasibility and conditional status probe."""

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
    raise RuntimeError("q20 GenUT probe requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("q20 GenUT probe requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("q20 GenUT probe found a visible GPU")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.particle_authority_genut_tf import generalized_unscented_transform


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_genut_q20_probe_2026_08_25.py"
MODULE = ROOT / "bayesfilter/testing/particle_authority_genut_tf.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase21-genut-q20-probe-subplan-2026-08-25.md"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = ROOT / str(receipt["path"])
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != receipt["sha256"]:
        raise RuntimeError(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.authority_root.is_absolute() or args.output_root.is_absolute():
        raise RuntimeError("paths must be repository-relative")
    if ".." in args.authority_root.parts or ".." in args.output_root.parts:
        raise RuntimeError("paths may not contain parent traversal")
    output = ROOT / args.output_root
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()
    pilot_path = ROOT / args.authority_root / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_GATE":
        raise RuntimeError("authority pilot did not pass")
    m0 = pilot["arms"]["M0"]
    if m0["protocol"].get("mode_axis") != 2:
        raise RuntimeError("mode axis mismatch")
    points = _load_tensor(m0["receipts"]["final_theta"])
    weights = _load_tensor(m0["receipts"]["final_normalized_weights"])
    if points.shape != (300, 4) or weights.shape != (300,):
        raise RuntimeError("unexpected authority bank shape")
    sigma_points, sigma_weights, diagnostics = generalized_unscented_transform(
        points, weights
    )
    finite = bool(
        tf.reduce_all(tf.math.is_finite(sigma_points)).numpy()
        and tf.reduce_all(tf.math.is_finite(sigma_weights)).numpy()
    )
    feasible = bool(diagnostics["feasible"].numpy()) if finite else False
    target_payload: dict[str, Any] = {
        "evaluated": False,
        "valid_count": 0,
        "row_count": int(sigma_points.shape[0]),
    }
    if feasible:
        target = batch_native_complexity_posterior_target(
            20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
        )
        value, score, status = target.neutra_batch_log_prob_and_grad_status(sigma_points)
        valid = tf.logical_and(
            tf.equal(status["status_code"], 0),
            tf.cast(status["valid_pre_regularized_score"], tf.bool),
        )
        target_payload = {
            "evaluated": True,
            "valid_count": tf.reduce_sum(tf.cast(valid, tf.int32)),
            "row_count": tf.size(valid),
            "finite": tf.logical_and(
                tf.reduce_all(tf.math.is_finite(value)),
                tf.reduce_all(tf.math.is_finite(score)),
            ),
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status["valid_pre_regularized_score"],
            "values": value,
            "scores": score,
        }
    if not finite:
        status_value = "GENUT_Q20_PROBE_FAIL_REPAIR"
    elif not feasible:
        status_value = "GENUT_Q20_INFEASIBLE_SCOPE"
    elif not bool(target_payload["finite"].numpy()) or int(target_payload["valid_count"].numpy()) != int(target_payload["row_count"].numpy()):
        status_value = "GENUT_Q20_PROBE_FAIL_REPAIR"
    else:
        status_value = "PASS_GENUT_Q20_PROBE_ROLE_LIMITED"
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.genut_q20_probe.v1",
        "status": status_value,
        "role": "q20_genut_feasibility_and_status_probe",
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(pilot_path),
            "protocol_hash": m0["configuration"]["protocol_hash"],
            "target_signature": m0["target_signature"],
            "mode_axis": 2,
        },
        "hard_gates": {
            "input_finite": finite,
            "genut_feasible": feasible,
            "target_evaluated": target_payload["evaluated"],
            "target_all_valid": (
                bool(target_payload["evaluated"])
                and int(target_payload["valid_count"].numpy()) == int(target_payload["row_count"].numpy())
                if target_payload["evaluated"]
                else False
            ),
        },
        "diagnostics": diagnostics,
        "target": target_payload,
        "sigma": {
            "point_count": int(sigma_points.shape[0]),
            "weights": sigma_weights,
            "points": sigma_points,
            "mode_axis_negative_fraction": tf.reduce_mean(
                tf.cast(sigma_points[:, 2] < 0.0, tf.float64)
            ),
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
            ).strip(),
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_intentionally_hidden": True,
            "jit_compile": True,
            "wall_seconds": time.perf_counter() - started,
            "source_sha256": {"runner": _sha(RUNNER), "module": _sha(MODULE), "plan": _sha(PLAN)},
        },
        "nonclaims": [
            "sigma points are not IID posterior samples or an authority replacement",
            "infeasibility is a scope result, not a theorem about all local GenUT uses",
            "no global mode-discovery, density, posterior, HMC, or default claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 21 q20 GenUT Feasibility/Status Probe\n\n"
        f"Status: `{status_value}`\n\n"
        "This artifact tests GenUT feasibility on the actual weighted bank; "
        "it cannot promote sigma points to a density or authority.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": status_value, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
