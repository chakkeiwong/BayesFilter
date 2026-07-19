#!/usr/bin/env python3
"""Trusted GPU/XLA timing canary for the transferred DSGE NeuTra procedure."""

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

from bayesfilter.inference.neutra_training import (  # noqa: E402
    DSGE_PAPER_TRAINING_BATCH_SIZE,
    DSGE_PAPER_TRAINING_STEPS,
    NeuTraReverseKLTrainer,
    dsge_paper_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    FREE_PARAMETER_NAMES,
    PRIOR_CENTER_VALUES,
    locked_ssl_lstm_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm_neutra.dsge_procedure_parity_timing_canary.v1"
STATUS = "DSGE_PROCEDURE_PARITY_GPU_XLA_TIMING_CANARY_PASSED"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-dsge-procedure-parity-repair-plan-2026-07-15.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-dsge-procedure-parity-repair-result-2026-07-15.md"
)
CANARY_SEED = (20260715, 4099)
MEASURED_STEPS = 5


class CanaryError(RuntimeError):
    """Raised when the exact-topology timing canary violates its contract."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _step_batch(trainer: NeuTraReverseKLTrainer, step: int) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant(CANARY_SEED, tf.int32),
        int(step),
    )
    return trainer.sample_base(batch_size=DSGE_PAPER_TRAINING_BATCH_SIZE, seed=seed)


def _host_step(result: Any) -> dict[str, Any]:
    return {
        "step": int(result.step.numpy()),
        "loss": float(result.loss.numpy()),
        "target_value_mean": float(result.target_value_mean.numpy()),
        "logdet_mean": float(result.logdet_mean.numpy()),
        "gradient_norm": float(result.gradient_norm.numpy()),
        "clipped_gradient_norm": float(result.clipped_gradient_norm.numpy()),
        "clipping_applied": bool(result.clipping_applied.numpy()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started_at = _now()
    started = time.perf_counter()

    physical_gpus = tf.config.list_physical_devices("GPU")
    if not physical_gpus:
        raise CanaryError("trusted timing canary requires a visible GPU")
    for gpu in physical_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    previous_soft_placement = tf.config.get_soft_device_placement()
    tf.config.set_soft_device_placement(False)

    try:
        with tf.device("/GPU:0"):
            target = locked_ssl_lstm_posterior_target()
            config = dsge_paper_neutra_config(
                dimension=4,
                fixed_translation=PRIOR_CENTER_VALUES,
                target_parameter_names=FREE_PARAMETER_NAMES,
                target_signature=target.target_signature(),
                target_adapter_signature=target.adapter_signature(),
                initialization_seed=CANARY_SEED,
                jit_compile=True,
            )
            trainer = NeuTraReverseKLTrainer(target, config)
            warmup_batch = _step_batch(trainer, 0)
            compile_started = time.perf_counter()
            warmup_result = trainer.train_step(warmup_batch)
            warmup_row = _host_step(warmup_result)
            compile_and_warmup_seconds = time.perf_counter() - compile_started

            hlo_text = trainer._compiled_train_step.experimental_get_compiler_ir(  # noqa: SLF001
                warmup_batch
            )(stage="hlo")
            if not isinstance(hlo_text, str) or "HloModule" not in hlo_text:
                raise CanaryError("compiled trainer did not expose HLO evidence")

            measured_rows = []
            measured_seconds = []
            for step in range(1, MEASURED_STEPS + 1):
                z = _step_batch(trainer, step)
                step_started = time.perf_counter()
                result = trainer.train_step(z)
                row = _host_step(result)
                measured_seconds.append(time.perf_counter() - step_started)
                measured_rows.append(row)

            validation = trainer.validation_batch(_step_batch(trainer, 100))
            validation_loss = float(tf.reduce_mean(validation.per_sample_loss).numpy())
            output_devices = tuple(
                sorted(
                    {
                        warmup_result.loss.device,
                        warmup_result.gradient_norm.device,
                        validation.per_sample_loss.device,
                        *(variable.device for variable in trainer.variables),
                    }
                )
            )
    finally:
        tf.config.set_soft_device_placement(previous_soft_placement)

    finite_values = (
        validation_loss,
        compile_and_warmup_seconds,
        *measured_seconds,
        *(value for row in (warmup_row, *measured_rows) for value in row.values() if not isinstance(value, bool)),
    )
    if not all(math.isfinite(float(value)) for value in finite_values):
        raise CanaryError("timing canary produced a nonfinite value")
    if not output_devices or not all("GPU:" in value for value in output_devices):
        raise CanaryError(f"timing canary output was not GPU-resident: {output_devices}")
    if int(trainer.step.numpy()) != MEASURED_STEPS + 1:
        raise CanaryError("timing canary optimizer step count mismatch")

    mean_step_seconds = sum(measured_seconds) / len(measured_seconds)
    max_step_seconds = max(measured_seconds)
    conservative_seconds_per_seed = max_step_seconds * DSGE_PAPER_TRAINING_STEPS
    conservative_pair_seconds = 2.0 * conservative_seconds_per_seed
    payload = {
        "schema": SCHEMA,
        "status": STATUS,
        "created_at_utc": _now(),
        "checks": {
            "exact_preset": config.manifest_payload(),
            "compile_and_warmup_seconds": compile_and_warmup_seconds,
            "measured_step_seconds": measured_seconds,
            "mean_step_seconds": mean_step_seconds,
            "max_step_seconds": max_step_seconds,
            "conservative_5000_step_seed_seconds": conservative_seconds_per_seed,
            "conservative_two_seed_seconds": conservative_pair_seconds,
            "warmup": warmup_row,
            "measured": measured_rows,
            "validation_loss": validation_loss,
            "hlo_sha256": _stable_hash(hlo_text),
            "hlo_characters": len(hlo_text),
            "finite": True,
            "gpu_output": True,
            "jit_compile": True,
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
            "data_version": target.target_signature(),
            "seed": list(CANARY_SEED),
            "batch_size": DSGE_PAPER_TRAINING_BATCH_SIZE,
            "compile_warmup_steps": 1,
            "measured_steps": MEASURED_STEPS,
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": time.perf_counter() - started,
            "output_path": args.output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
            "source_sha256": {
                "runner": _sha256(Path(__file__).resolve().relative_to(ROOT)),
                "trainer": _sha256(Path("bayesfilter/inference/neutra_training.py")),
                "artifact_loader": _sha256(Path("bayesfilter/inference/neutra_artifacts.py")),
                "parity_test": _sha256(Path("tests/test_neutra_dsge_procedure_parity.py")),
                "target": _sha256(Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py")),
            },
        },
        "nonclaims": (
            "compile and timing canary only",
            "loss and parameter movement are descriptive only",
            "no frozen or nominated transport candidate",
            "no material training, HMC, posterior, predictive, superiority, readiness, or scientific claim",
        ),
    }
    _write_json(args.output, payload)
    print(STATUS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
