"""Probe per-mode/local GenUT feasibility on the audited q20 bank."""

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
    raise RuntimeError("local GenUT probe requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("local GenUT probe requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("local GenUT probe found a visible GPU")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.particle_authority_genut_tf import generalized_unscented_transform


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_local_genut_probe_2026_08_25.py"
MODULE = ROOT / "bayesfilter/testing/particle_authority_genut_tf.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase22-local-genut-subplan-2026-08-25.md"


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


def _load(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = ROOT / str(receipt["path"])
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != receipt["sha256"]:
        raise RuntimeError(f"tensor hash mismatch: {path}")
    return tf.ensure_shape(
        tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"]))),
        receipt["shape"],
    )


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
    points = _load(m0["receipts"]["final_theta"])
    weights = _load(m0["receipts"]["final_normalized_weights"])
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    modes: dict[str, Any] = {}
    for name, negative in (("negative_axis2", True), ("positive_axis2", False)):
        mask = points[:, 2] < 0.0
        if not negative:
            mask = tf.logical_not(mask)
        indices = tf.reshape(tf.where(mask), [-1])
        count = int(tf.shape(indices)[0].numpy())
        if count < 2:
            modes[name] = {"status": "LOCAL_GENUT_TOO_FEW_ROWS", "count": count}
            continue
        local_points = tf.ensure_shape(tf.gather(points, indices), [count, 4])
        local_weights = tf.ensure_shape(tf.gather(weights, indices), [count])
        local_weights = local_weights / tf.reduce_sum(local_weights)
        sigma_points, sigma_weights, diagnostics = generalized_unscented_transform(
            local_points, local_weights
        )
        finite = bool(
            tf.reduce_all(tf.math.is_finite(sigma_points)).numpy()
            and tf.reduce_all(tf.math.is_finite(sigma_weights)).numpy()
        )
        feasible = bool(diagnostics["feasible"].numpy()) if finite else False
        payload: dict[str, Any] = {
            "count": count,
            "indices": indices,
            "weight_mass_before_renormalization": tf.reduce_sum(
                tf.gather(weights, indices)
            ),
            "finite": finite,
            "feasible": feasible,
            "diagnostics": diagnostics,
            "sigma_points": sigma_points,
            "sigma_weights": sigma_weights,
        }
        if feasible:
            value, score, status = target.neutra_batch_log_prob_and_grad_status(sigma_points)
            valid = tf.logical_and(
                tf.equal(status["status_code"], 0),
                tf.cast(status["valid_pre_regularized_score"], tf.bool),
            )
            payload["target"] = {
                "finite": tf.logical_and(
                    tf.reduce_all(tf.math.is_finite(value)),
                    tf.reduce_all(tf.math.is_finite(score)),
                ),
                "valid_count": tf.reduce_sum(tf.cast(valid, tf.int32)),
                "row_count": tf.size(valid),
                "values": value,
                "scores": score,
                "status_code": status["status_code"],
            }
            payload["status"] = (
                "PASS_LOCAL_GENUT_ROLE_LIMITED"
                if bool(payload["target"]["finite"].numpy())
                and int(payload["target"]["valid_count"].numpy()) == int(payload["target"]["row_count"].numpy())
                else "LOCAL_GENUT_TARGET_FAIL"
            )
        else:
            payload["status"] = "LOCAL_GENUT_INFEASIBLE_SCOPE"
        modes[name] = payload
    feasible_modes = [
        value for value in modes.values() if value.get("status") == "PASS_LOCAL_GENUT_ROLE_LIMITED"
    ]
    if len(feasible_modes) == 2:
        status_value = "PASS_LOCAL_GENUT_ROLE_LIMITED"
    elif feasible_modes:
        status_value = "LOCAL_GENUT_PARTIAL_FEASIBILITY"
    elif all(value.get("status") == "LOCAL_GENUT_INFEASIBLE_SCOPE" for value in modes.values()):
        status_value = "LOCAL_GENUT_INFEASIBLE_SCOPE"
    else:
        status_value = "LOCAL_GENUT_PROBE_FAIL_REPAIR"
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.local_genut_probe.v1",
        "status": status_value,
        "role": "q20_local_genut_feasibility_probe",
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(pilot_path),
            "protocol_hash": m0["configuration"]["protocol_hash"],
            "target_signature": m0["target_signature"],
            "mode_axis": 2,
        },
        "modes": modes,
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
            "axis2 sign subsets are finite diagnostics, not exhaustive modes",
            "local sigma points are not IID or a density/authority replacement",
            "no global mode-discovery, posterior, HMC, or default claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 22 Local GenUT Feasibility Probe\n\n"
        f"Status: `{status_value}`\n\n"
        "Axis-sign subsets are finite local diagnostics only.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": status_value, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
