#!/usr/bin/env python3
"""Trusted GPU/XLA canary for frozen dense-IAF mathematical closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference import (  # noqa: E402
    finalize_dense_iaf_neutra_artifact_payload,
    load_frozen_neutra_artifact,
)


SCHEMA = "bayesfilter.ssl_lstm_neutra.phase2_dense_iaf_xla_canary.v1"
STATUS = "PHASE_2_DENSE_IAF_GPU_XLA_CANARY_PASSED"
TARGET_SIGNATURE = "549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-2-dense-iaf-closure-plan-2026-07-14.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-2-dense-iaf-closure-result-2026-07-14.md"
)


class CanaryError(RuntimeError):
    """Raised when the bounded dense-IAF canary fails."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(
        ("git", *args),
        cwd=ROOT,
        text=True,
    ).strip()


def _strict_write(path: Path, payload: dict[str, Any]) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CanaryError(f"output already exists: {path}")
    destination.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _payload() -> dict[str, Any]:
    raw = {
        "schema": "bayesfilter.neutra.dense_iaf_frozen_transport.v1",
        "transport_id": "phase2-dense-iaf-xla-canary",
        "dimension": 2,
        "target_signature": TARGET_SIGNATURE,
        "log_jacobian_available": True,
        "component_order": ("dense", "mix", "affine"),
        "components": (
            {
                "component_id": "dense",
                "kind": "dense_autoregressive_iaf",
                "dim": 2,
                "hidden_layers": (3,),
                "activation": "tanh",
                "s_max": 1.0,
                "masks_policy": "legacy_degree_masks_v1",
                "dtype": "float64",
                "weights": (
                    ((0.5, -0.25, 0.1), (0.75, 0.1, -0.3)),
                    (
                        (0.2, -0.4, 0.3, -0.2),
                        (0.1, 0.6, -0.5, 0.7),
                        (-0.2, 0.3, 0.4, -0.1),
                    ),
                ),
                "biases": ((0.05, -0.1, 0.2), (0.02, -0.03, 0.04, -0.05)),
            },
            {
                "component_id": "mix",
                "kind": "mixing_linear",
                "dim": 2,
                "dtype": "float64",
                "matrix": ((1.0, 0.25), (-0.5, 1.5)),
            },
            {
                "component_id": "affine",
                "kind": "affine_dense",
                "dim": 2,
                "dtype": "float64",
                "offset": (0.1, -0.2),
                "matrix": ((1.2, -0.3), (0.4, 0.9)),
            },
        ),
        "training_state_hash": "phase2-analytic-control-no-training",
        "nonclaims": (
            "dense-IAF mathematical closure canary only",
            "no NeuTra training claim",
            "no HMC or posterior claim",
            "no readiness or default claim",
        ),
    }
    return dict(finalize_dense_iaf_neutra_artifact_payload(raw))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_at = _now()
    started = time.perf_counter()

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise CanaryError("trusted Phase 2 canary requires a visible GPU")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    artifact = load_frozen_neutra_artifact(
        _payload(),
        expected_target_signature=TARGET_SIGNATURE,
    )
    transport = artifact.transport
    z = tf.constant(
        [[0.0, 0.0], [0.25, -0.35], [2.5, -3.0], [-4.0, 3.5]],
        dtype=tf.float64,
    )
    theta_score = tf.constant(
        [[0.7, -1.1], [-0.2, 0.5], [1.3, 0.4], [-0.6, 0.9]],
        dtype=tf.float64,
    )

    eager = (
        transport.forward_batch(z),
        transport.inverse_theta_to_z_batch(transport.forward_batch(z)),
        transport.log_abs_det_jacobian_batch(z),
        transport.pullback_score_batch(z, theta_score),
        transport.log_abs_det_jacobian_score_batch(z),
    )

    @tf.function(jit_compile=True, autograph=False)
    def compiled(values: tf.Tensor, scores: tf.Tensor):
        theta = transport.forward_batch(values)
        return (
            theta,
            transport.inverse_theta_to_z_batch(theta),
            transport.log_abs_det_jacobian_batch(values),
            transport.pullback_score_batch(values, scores),
            transport.log_abs_det_jacobian_score_batch(values),
        )

    with tf.device("/GPU:0"):
        actual = compiled(z, theta_score)
    _ = tuple(item.numpy() for item in actual)

    tolerance = 1.0e-11
    names = ("forward", "inverse", "logdet", "pullback", "logdet_score")
    residuals = {
        name: float(tf.reduce_max(tf.abs(got - expected)).numpy())
        for name, got, expected in zip(names, actual, eager)
    }
    roundtrip_residual = float(tf.reduce_max(tf.abs(actual[1] - z)).numpy())
    if any(value > tolerance for value in residuals.values()):
        raise CanaryError(f"eager/XLA residual exceeded tolerance: {residuals}")
    if roundtrip_residual > tolerance:
        raise CanaryError(f"roundtrip residual exceeded tolerance: {roundtrip_residual}")
    if not all(bool(tf.reduce_all(tf.math.is_finite(item)).numpy()) for item in actual):
        raise CanaryError("compiled output contains a nonfinite value")
    output_devices = tuple(sorted({item.device for item in actual}))
    if not output_devices or not all("GPU:" in item for item in output_devices):
        raise CanaryError(f"compiled outputs were not placed on GPU: {output_devices}")

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_at_utc": _now(),
        "checks": {
            "compiled_execution": True,
            "finite": True,
            "gpu_output": True,
            "jit_compile": True,
            "roundtrip_residual": roundtrip_residual,
            "eager_xla_max_abs_residuals": residuals,
            "tolerance": tolerance,
        },
        "transport": {
            "artifact_signature": artifact.artifact_signature,
            "target_signature": TARGET_SIGNATURE,
            "topology_hash": artifact.manifest.topology_hash,
            "tensor_hash": artifact.manifest.tensor_hash,
            "transport_hash": artifact.manifest.transport_hash,
        },
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "tensorflow_probability": "N/A: transport-only canary",
            "physical_gpus": tuple(item.name for item in gpus),
            "output_devices": output_devices,
            "dtype": "float64",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "seed": "N/A: fixed deterministic tensors",
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": time.perf_counter() - started,
            "output_path": args.output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
            "source_sha256": {
                "runner": _sha256(Path(__file__).resolve().relative_to(ROOT)),
                "transport": _sha256(Path("bayesfilter/inference/neutra_artifacts.py")),
                "focused_test": _sha256(Path("tests/test_dense_iaf_neutra_artifact_loader.py")),
            },
        },
        "nonclaims": (
            "engineering canary on a fixed two-dimensional analytic fixture only",
            "no transport quality or NeuTra training evidence",
            "no HMC, posterior, performance, readiness, or scientific claim",
        ),
    }
    _strict_write(args.output, payload)
    print(STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
