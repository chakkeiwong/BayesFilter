#!/usr/bin/env python3
"""Trusted GPU/XLA canary for the SSL-LSTM reverse-KL trainer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from bayesfilter.inference.neutra_artifacts import (  # noqa: E402
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (  # noqa: E402
    NeuTraReverseKLTrainer,
    NeuTraTrainerConfig,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    TARGET_SEMANTIC_SHA256,
    locked_ssl_lstm_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm_neutra.phase3_trainer_xla_canary.v1"
STATUS = "PHASE_3_REVERSE_KL_TRAINER_GPU_XLA_CANARY_PASSED"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-3-reverse-kl-trainer-plan-2026-07-14.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-3-reverse-kl-trainer-result-2026-07-14.md"
)


class CanaryError(RuntimeError):
    """Raised when the bounded trainer canary fails."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    destination = ROOT / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise CanaryError(f"output already exists: {path}")
    destination.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _finite(value: tf.Tensor) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(value)).numpy())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_at = _now()
    started = time.perf_counter()

    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise CanaryError("trusted Phase 3 canary requires a visible GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    previous_soft_placement = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)
    try:
        with tf.device("/GPU:0"):
            target = locked_ssl_lstm_posterior_target()
            config = NeuTraTrainerConfig(
                dimension=4,
                family="dense_iaf",
                hidden_layers=(4,),
                activation="tanh",
                s_max=1.0,
                initialization_scale=0.02,
                initialization_seed=(20260714, 2101),
                learning_rate=1.0e-3,
                gradient_clip_norm=10.0,
                jit_compile=True,
            )
            trainer = NeuTraReverseKLTrainer(target, config)
            z = trainer.sample_base(batch_size=4, seed=(20260714, 2101))
            before = tuple(tf.identity(variable) for variable in trainer.variables)
            first_started = time.perf_counter()
            first = trainer.train_step(z)
            first_host_rows = tuple(
                value.numpy()
                for value in (
                    first.loss,
                    first.surrogate,
                    first.target_value_mean,
                    first.logdet_mean,
                    first.gradient_norm,
                    first.clipped_gradient_norm,
                )
            )
            first_step_seconds = time.perf_counter() - first_started
            second_started = time.perf_counter()
            second = trainer.train_step(z)
            second_host_rows = tuple(
                value.numpy()
                for value in (
                    second.loss,
                    second.surrogate,
                    second.target_value_mean,
                    second.logdet_mean,
                    second.gradient_norm,
                    second.clipped_gradient_norm,
                )
            )
            second_step_seconds = time.perf_counter() - second_started
            after = tuple(tf.identity(variable) for variable in trainer.variables)
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)

    outputs = (
        first.loss,
        first.surrogate,
        first.target_value_mean,
        first.logdet_mean,
        first.gradient_norm,
        first.clipped_gradient_norm,
        second.loss,
        second.gradient_norm,
        *after,
    )
    if not all(_finite(value) for value in outputs):
        raise CanaryError("trainer output contains a nonfinite value")
    if not all(
        math.isfinite(float(value))
        for value in (*first_host_rows, *second_host_rows)
    ):
        raise CanaryError("host-synchronized trainer diagnostics are nonfinite")
    output_devices = tuple(sorted({value.device for value in outputs}))
    if not output_devices or not all("GPU:" in item for item in output_devices):
        raise CanaryError(f"trainer output was not GPU-resident: {output_devices}")
    update_norms = tuple(
        float(tf.linalg.norm(new - old).numpy()) for old, new in zip(before, after)
    )
    if not any(value > 0.0 for value in update_norms):
        raise CanaryError("two compiled steps did not update transport variables")
    if int(second.step.numpy()) != 2 or int(trainer.step.numpy()) != 2:
        raise CanaryError("trainer step counter did not reach two")

    state = trainer.state_payload()
    frozen = trainer.frozen_transport_payload(
        transport_id="ssl-lstm-phase3-canary-dense-iaf",
        target_signature=TARGET_SEMANTIC_SHA256,
    )
    loaded = load_frozen_neutra_artifact(
        frozen,
        expected_target_signature=TARGET_SEMANTIC_SHA256,
    )
    trainable_theta, trainable_logdet = trainer.forward_and_logdet(z)
    frozen_theta = loaded.transport.forward_batch(z)
    frozen_logdet = loaded.transport.log_abs_det_jacobian_batch(z)
    replay_residuals = {
        "forward": float(tf.reduce_max(tf.abs(trainable_theta - frozen_theta)).numpy()),
        "logdet": float(tf.reduce_max(tf.abs(trainable_logdet - frozen_logdet)).numpy()),
    }
    if any(value != 0.0 for value in replay_residuals.values()):
        raise CanaryError(f"frozen payload replay mismatch: {replay_residuals}")
    if loaded.manifest.training_state_hash != state["state_hash"]:
        raise CanaryError("frozen payload training-state hash mismatch")

    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_at_utc": _now(),
        "checks": {
            "compiled_update_count": 2,
            "finite": True,
            "host_synchronized_finite_checks": True,
            "gpu_output": True,
            "jit_compile": True,
            "target_tape_boundary": "target_value_score_outside_transport_tape",
            "update_norms": update_norms,
            "state_hash": state["state_hash"],
            "frozen_replay_residuals": replay_residuals,
            "first_loss": float(first.loss.numpy()),
            "second_loss": float(second.loss.numpy()),
            "first_gradient_norm": float(first.gradient_norm.numpy()),
            "second_gradient_norm": float(second.gradient_norm.numpy()),
            "first_clipped_gradient_norm": float(first.clipped_gradient_norm.numpy()),
            "second_clipped_gradient_norm": float(second.clipped_gradient_norm.numpy()),
            "compile_and_first_step_seconds": first_step_seconds,
            "steady_second_step_seconds": second_step_seconds,
        },
        "transport": {
            "target_signature": TARGET_SEMANTIC_SHA256,
            "artifact_signature": loaded.artifact_signature,
            "transport_hash": loaded.manifest.transport_hash,
            "topology_hash": loaded.manifest.topology_hash,
            "tensor_hash": loaded.manifest.tensor_hash,
            "training_state_hash": loaded.manifest.training_state_hash,
        },
        "run_manifest": {
            "git_commit": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain")),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "physical_gpus": tuple(device.name for device in physical_gpus),
            "output_devices": output_devices,
            "dtype": "float64",
            "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "jit_compile": True,
            "soft_device_placement_during_run": False,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "seed": [20260714, 2101],
            "batch_size": 4,
            "optimizer_steps": 2,
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": time.perf_counter() - started,
            "output_path": args.output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
            "source_sha256": {
                "runner": _sha256(Path(__file__).resolve().relative_to(ROOT)),
                "trainer": _sha256(Path("bayesfilter/inference/neutra_training.py")),
                "trainer_test": _sha256(Path("tests/test_neutra_reverse_kl_training.py")),
                "target": _sha256(Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py")),
            },
        },
        "nonclaims": (
            "two-step actual-target trainer mechanics canary only",
            "loss changes are descriptive and non-promotional",
            "no material transport training or candidate selection",
            "no HMC, posterior, predictive, performance, readiness, or scientific claim",
        ),
    }
    _write_json(args.output, payload)
    print(STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
