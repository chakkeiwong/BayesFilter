"""Run a bounded q20 target/status probe for the source-faithful ETPF map."""

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
    raise RuntimeError("q20 ETPF probe requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("q20 ETPF probe requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("q20 ETPF probe found a visible GPU")

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.testing.particle_authority_etpf_tf import second_order_etpf_transform


RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_particle_authority_etpf_q20_probe_2026_08_25.py"
MODULE = ROOT / "bayesfilter/testing/particle_authority_etpf_tf.py"
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase19-etpf-q20-probe-subplan-2026-08-25.md"


class ProbeError(RuntimeError):
    """Raised when the q20 ETPF probe cannot preserve its contract."""


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
        raise ProbeError(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if not bool(tf.reduce_all(tf.math.is_finite(value)).numpy()):
        raise ProbeError(f"non-finite tensor: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--riccati-tolerance", type=float, default=1.0e-3)
    parser.add_argument("--riccati-max-steps", type=int, default=2000)
    args = parser.parse_args()
    if args.authority_root.is_absolute() or args.output_root.is_absolute():
        raise ProbeError("paths must be repository-relative")
    if ".." in args.authority_root.parts or ".." in args.output_root.parts:
        raise ProbeError("paths may not contain parent traversal")
    output = ROOT / args.output_root
    if output.exists():
        raise ProbeError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    started = time.perf_counter()

    pilot_path = ROOT / args.authority_root / "pilot.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if pilot.get("status") != "PASS_GATE":
        raise ProbeError("authority pilot did not pass")
    m0 = pilot["arms"]["M0"]
    if m0["protocol"].get("mode_axis") != 2:
        raise ProbeError("mode axis mismatch")
    if m0["protocol"].get("target_signature") != m0.get("target_signature"):
        raise ProbeError("target signature mismatch")
    points = _load_tensor(m0["receipts"]["final_theta"])
    weights = _load_tensor(m0["receipts"]["final_normalized_weights"])
    if points.shape != (300, 4) or weights.shape != (300,):
        raise ProbeError("unexpected authority bank shape")

    indices = [round(index * 299 / 31) for index in range(32)]
    if len(set(indices)) != 32:
        raise ProbeError("deterministic subset indices are not unique")
    index_tensor = tf.constant(indices, tf.int32)
    subset_points = tf.gather(points, index_tensor)
    subset_weights = tf.gather(weights, index_tensor)
    subset_weights = subset_weights / tf.reduce_sum(subset_weights)
    analysis, diagnostics = second_order_etpf_transform(
        subset_points,
        subset_weights,
        regularization=10.0,
        sinkhorn_steps=400,
        riccati_step=0.1,
        riccati_max_steps=args.riccati_max_steps,
        riccati_tolerance=args.riccati_tolerance,
    )
    target = batch_native_complexity_posterior_target(
        20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
    )
    value, score, status = target.neutra_batch_log_prob_and_grad_status(analysis)
    status_valid = tf.logical_and(
        tf.equal(status["status_code"], 0),
        tf.cast(status["valid_pre_regularized_score"], tf.bool),
    )
    finite_target = bool(
        tf.logical_and(
            tf.reduce_all(tf.math.is_finite(value)),
            tf.reduce_all(tf.math.is_finite(score)),
        ).numpy()
    )
    source_min = tf.reduce_min(subset_points, axis=0)
    source_max = tf.reduce_max(subset_points, axis=0)
    below = tf.reduce_sum(tf.cast(analysis < source_min[None, :], tf.int32))
    above = tf.reduce_sum(tf.cast(analysis > source_max[None, :], tf.int32))
    hard = {
        "finite_transform": bool(
            tf.reduce_all(tf.math.is_finite(analysis)).numpy()
        ),
        "riccati_converged": bool(diagnostics["riccati_converged"].numpy()),
        "row_residual": float(diagnostics["corrected_row_residual"].numpy()),
        "column_residual": float(diagnostics["corrected_column_residual"].numpy()),
        "mean_residual": float(diagnostics["mean_residual"].numpy()),
        "covariance_residual": float(diagnostics["covariance_residual"].numpy()),
        "target_finite": finite_target,
        "target_status_valid": bool(tf.reduce_all(status_valid).numpy()),
    }
    status_value = (
        "PASS_ETPF_Q20_PROBE_ROLE_LIMITED"
        if all(
            (
                hard["finite_transform"],
                hard["riccati_converged"],
                hard["row_residual"] <= 2.0e-6,
                hard["column_residual"] <= 2.0e-6,
                hard["mean_residual"] <= 1.0e-3,
                hard["covariance_residual"] <= 1.0e-3,
                hard["target_finite"],
                hard["target_status_valid"],
            )
        )
        else "ETPF_Q20_PROBE_FAIL_REPAIR"
    )
    mode_axis = 2
    result = {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.etpf_q20_probe.v1",
        "status": status_value,
        "role": "q20_source_faithful_etpf_integration_probe",
        "authority": {
            "root": args.authority_root,
            "pilot_sha256": _sha(pilot_path),
            "protocol_hash": m0["configuration"]["protocol_hash"],
            "target_signature": m0["target_signature"],
            "mode_axis": mode_axis,
            "riccati_tolerance": args.riccati_tolerance,
            "riccati_max_steps": args.riccati_max_steps,
        },
        "subset": {"indices": indices, "count": len(indices)},
        "hard_gates": hard,
        "diagnostics": diagnostics,
        "target": {
            "values": value,
            "scores": score,
            "status_code": status["status_code"],
            "valid_pre_regularized_score": status["valid_pre_regularized_score"],
        },
        "support_diagnostic": {
            "below_source_range_count": below,
            "above_source_range_count": above,
            "negative_correction_fraction": diagnostics["corrected_negative_fraction"],
            "mode_negative_fraction": tf.reduce_mean(
                tf.cast(analysis[:, mode_axis] < 0.0, tf.float64)
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
            "source_sha256": {
                "runner": _sha(RUNNER),
                "module": _sha(MODULE),
                "plan": _sha(PLAN),
            },
        },
        "nonclaims": [
            "the deterministic 32-row subset is not a q20 posterior estimate",
            "transformed rows are not IID and no density/Jacobian correction for the empirical transform is claimed",
            "no mode-discovery, posterior correctness, HMC, or default promotion claim",
        ],
    }
    (output / "result.json").write_text(
        json.dumps(_safe(result), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    (output / "result.md").write_text(
        "# Phase 19 q20 ETPF Integration Probe\n\n"
        f"Status: `{status_value}`\n\n"
        "This is a deterministic small-subset target/status probe, not q20 "
        "posterior or authority evidence.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": status_value, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if status_value == "PASS_ETPF_Q20_PROBE_ROLE_LIMITED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
