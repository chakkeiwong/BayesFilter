#!/usr/bin/env python3
"""Trusted GPU/XLA compile canary for Phase 8 draw-chunked forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.nonlinear.ssl_lstm_predictive_tf import (  # noqa: E402
    SSLLSTMForecastConfig,
    forecast_ssl_lstm_paths,
    make_ssl_lstm_innovation_bank,
    ssl_lstm_forecast_compiled_program,
    ssl_lstm_terminal_compiled_program,
    ssl_lstm_terminal_covariance_audit_compiled_program,
)


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "plan-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
A0_LOCK_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"
)
A0_LOCK_SHA256 = (
    "1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383"
)
TERMINAL_VALIDATION_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "terminal-staged-audit-validation.json"
)
TERMINAL_VALIDATION_SHA256 = (
    "5aa08e674130c2a8b8e5fd7bf47989c68595f350a6a4d399ed1394f8883fad68"
)
FORECAST_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py")
PILOT_RUNNER = Path(
    "docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py"
)
ORIGINAL_STARTS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
DRAW_COUNT = 32
DRAW_CHUNK_SIZE = 16
SEED = (13001, 13002)


class ChunkRepairError(RuntimeError):
    """Raised when the chunk-repair canary fails closed."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ChunkRepairError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ChunkRepairError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise ChunkRepairError(f"expected JSON object: {path}")
    return value


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write(path: Path, payload: dict[str, Any]) -> None:
    absolute = _absolute(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise ChunkRepairError(f"refusing to overwrite receipt: {path}")
    absolute.write_bytes(_canonical(payload))


def _git(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ("git", *arguments),
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def _a0_points() -> tf.Tensor:
    if _sha256(A0_LOCK_PATH) != A0_LOCK_SHA256:
        raise ChunkRepairError("A0 target-lock identity drift")
    geometry = _strict_json(A0_LOCK_PATH).get("sampler_geometry")
    if not isinstance(geometry, dict):
        raise ChunkRepairError("A0 sampler geometry missing")
    center = tf.constant(geometry["center_free"]["values"], tf.float64)
    scale = tf.constant(geometry["scale"]["values"], tf.float64)
    factor_z = tf.constant(geometry["factor_z"]["values"], tf.float64)
    base = center + tf.constant(ORIGINAL_STARTS, tf.float64) @ tf.transpose(
        tf.linalg.diag(scale) @ factor_z
    )
    points = tf.tile(base, [DRAW_COUNT // len(ORIGINAL_STARTS), 1])
    if tuple(points.shape) != (DRAW_COUNT, 4):
        raise ChunkRepairError("chunk canary point shape is invalid")
    return points


def run_canary(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise ChunkRepairError("wall cap must be positive and finite")
    if DRAW_COUNT % DRAW_CHUNK_SIZE:
        raise ChunkRepairError("canary draw count must be divisible by chunk size")
    if _sha256(TERMINAL_VALIDATION_PATH) != TERMINAL_VALIDATION_SHA256:
        raise ChunkRepairError("terminal staged-validation identity drift")

    started_at = _now()
    started = time.perf_counter()
    config = SSLLSTMForecastConfig()
    config.assert_evidence_config()
    points = _a0_points()
    bank = make_ssl_lstm_innovation_bank(
        config,
        DRAW_COUNT,
        tf.constant(SEED, tf.int32),
        "independent_arm",
        1,
    )
    forecast = forecast_ssl_lstm_paths(
        points,
        bank,
        config,
        draw_chunk_size=DRAW_CHUNK_SIZE,
        runtime_execution_role="trusted_gpu_xla_canary",
        trust_basis="owner_designated_managed_session_visible_gpu_trusted",
    )
    elapsed = time.perf_counter() - started
    if elapsed > wall_cap_seconds:
        raise ChunkRepairError("chunk canary wall cap exceeded")
    if tuple(forecast.observations.shape) != (DRAW_COUNT, 2, 10, 1):
        raise ChunkRepairError("chunked forecast output shape is invalid")
    if forecast.provenance.draw_chunk_size != DRAW_CHUNK_SIZE:
        raise ChunkRepairError("chunk size was not preserved in provenance")
    if any(forecast.provenance.terminal_covariance_statuses):
        raise ChunkRepairError("terminal covariance status failed")
    if not all("GPU:" in device for device in forecast.provenance.output_devices):
        raise ChunkRepairError("chunked forecast outputs are not GPU resident")
    if not bool(tf.reduce_all(tf.math.is_finite(forecast.observations))):
        raise ChunkRepairError("chunked forecast output is nonfinite")

    trace_counts = {
        "terminal_filter_32": _trace_count(
            ssl_lstm_terminal_compiled_program(config, DRAW_COUNT)
        ),
        "terminal_covariance_audit_32": _trace_count(
            ssl_lstm_terminal_covariance_audit_compiled_program(DRAW_COUNT)
        ),
        "forecast_chunk_16": _trace_count(
            ssl_lstm_forecast_compiled_program(config, DRAW_CHUNK_SIZE)
        ),
    }
    if any(value != 1 for value in trace_counts.values()):
        raise ChunkRepairError(f"compiled trace gate failed: {trace_counts}")

    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_forecast_chunk_canary.v1",
        "status": "PASSED",
        "decision": "PHASE8_FORECAST_CHUNK_REPAIR_PASSED_EXACT_PREFIX_VALIDATION_REQUIRED",
        "contract": {
            "draw_count": DRAW_COUNT,
            "draw_chunk_size": DRAW_CHUNK_SIZE,
            "forecast_replication_count": 2,
            "forecast_horizon": 10,
            "chunk_count": DRAW_COUNT // DRAW_CHUNK_SIZE,
            "point_source": "tiled_four_A0_start_derived_points_engineering_only",
            "retained_samples_read": False,
            "confirmation_forecast_opened": False,
        },
        "terminal_validation_binding": {
            "path": TERMINAL_VALIDATION_PATH.as_posix(),
            "sha256": TERMINAL_VALIDATION_SHA256,
        },
        "forecast": {
            "observation_shape": list(forecast.observations.shape),
            "observation_tensor_sha256": hashlib.sha256(
                bytes(tf.io.serialize_tensor(forecast.observations).numpy())
            ).hexdigest(),
            "innovation_bank_signature": bank.content_signature,
            "innovation_tensor_hashes": bank.tensor_hashes(),
            "output_devices": list(forecast.provenance.output_devices),
            "terminal_status_nonzero_count": sum(
                value != 0 for value in forecast.provenance.terminal_covariance_statuses
            ),
        },
        "compile_trace_counts": trace_counts,
        "source_bindings": {
            name: {"path": path.as_posix(), "sha256": _sha256(path)}
            for name, path in {
                "plan": PLAN_PATH,
                "runner": SCRIPT_PATH,
                "forecast": FORECAST_SOURCE,
                "pilot_runner": PILOT_RUNNER,
            }.items()
        },
        "run_manifest": {
            "command": shlex.join((sys.executable, *sys.argv)),
            "cwd": str(ROOT),
            "interpreter": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": __import__("tensorflow_probability").__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_devices": [
                device.name for device in tf.config.list_physical_devices("GPU")
            ],
            "logical_devices": [
                device.name for device in tf.config.list_logical_devices("GPU")
            ],
            "jit_compile": True,
            "dtype": "float64",
            "tf32_enabled": bool(
                tf.config.experimental.tensor_float_32_execution_enabled()
            ),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(_git("status", "--porcelain").strip()),
            "random_seed": list(SEED),
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": elapsed,
            "wall_cap_seconds": wall_cap_seconds,
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
        },
        "nonclaims": [
            "engineering compile and draw-order canary only",
            "no retained G/H sample or confirmation forecast was read",
            "no calibration power or predictive-equivalence claim",
        ],
    }
    _write(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args()
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise ChunkRepairError("chunk canary requires a visible GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    with tf.device("/GPU:0"):
        payload = run_canary(
            output=args.output,
            wall_cap_seconds=float(args.wall_cap_seconds),
        )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
