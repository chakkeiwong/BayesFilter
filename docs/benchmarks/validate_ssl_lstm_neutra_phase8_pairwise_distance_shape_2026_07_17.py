#!/usr/bin/env python3
"""Exact-shape GPU/XLA canary for the Phase 8 pairwise-distance repair."""

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

import bayesfilter.inference.predictive_equivalence as predictive  # noqa: E402


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "plan-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
FAILURE_RECORD_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "target-pilot-repair-02-failure-record.json"
)
FAILURE_RECORD_SHA256 = (
    "f4de95bf6fc540b32d7d2ba7e06002600af5b3fb90512ef1ba5cb6a05bed4abc"
)
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
SHAPE = (8, 64, 2, 10)


class PairwiseShapeError(RuntimeError):
    """Raised when the exact-shape distance canary fails closed."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


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
        raise PairwiseShapeError(f"refusing to overwrite receipt: {path}")
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


def _fixture() -> tf.Tensor:
    values = tf.reshape(tf.range(math.prod(SHAPE), dtype=tf.float64), SHAPE)
    return tf.sin(values / 37.0) + values / 100003.0


def run_canary(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise PairwiseShapeError("wall cap must be positive and finite")
    if _sha256(FAILURE_RECORD_PATH) != FAILURE_RECORD_SHA256:
        raise PairwiseShapeError("repair-02 failure-record identity drift")
    started_at = _now()
    started = time.perf_counter()
    paths = _fixture()
    compiled = predictive.pooled_pairwise_distance_scale(paths)
    eager = predictive.pooled_pairwise_distance_scale(paths, jit_compile=False)
    wall_time = time.perf_counter() - started
    if wall_time > wall_cap_seconds:
        raise PairwiseShapeError("pairwise-distance canary wall cap exceeded")
    if "GPU:" not in str(compiled.median_distance.device):
        raise PairwiseShapeError("compiled distance output is not GPU resident")
    if not math.isfinite(float(compiled.median_distance)) or float(
        compiled.median_distance
    ) <= 0.0:
        raise PairwiseShapeError("compiled distance is nonpositive or nonfinite")
    residual = abs(float(compiled.median_distance - eager.median_distance))
    tolerance = 4096.0 * 2.220446049250313e-16 * max(
        1.0,
        abs(float(compiled.median_distance)),
        abs(float(eager.median_distance)),
    )
    if residual > tolerance:
        raise PairwiseShapeError("compiled/eager median parity failed")
    if int(compiled.positive_pair_count) != int(eager.positive_pair_count):
        raise PairwiseShapeError("compiled/eager positive-pair count differs")
    expected_total = math.prod(SHAPE[:-1]) * (math.prod(SHAPE[:-1]) - 1) // 2
    if int(compiled.total_pair_count) != expected_total:
        raise PairwiseShapeError("total-pair count differs")
    trace_count = predictive._pairwise_distance_scale_xla.experimental_get_tracing_count()
    if trace_count != 1:
        raise PairwiseShapeError(f"compiled distance kernel retraced: {trace_count}")

    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_pairwise_distance_shape_canary.v1",
        "status": "PASSED",
        "decision": "PHASE8_PAIRWISE_DISTANCE_EXACT_SHAPE_REPAIR_PASSED_PILOT_REPAIR_03_ELIGIBLE",
        "contract": {
            "path_shape": list(SHAPE),
            "path_count": math.prod(SHAPE[:-1]),
            "fixture_role": "deterministic_engineering_shape_fixture_only",
            "retained_samples_read": False,
            "forecast_artifacts_read": False,
            "g_h_difference_computed": False,
        },
        "repair_02_failure_binding": {
            "path": FAILURE_RECORD_PATH.as_posix(),
            "sha256": FAILURE_RECORD_SHA256,
        },
        "result": {
            "compiled_median_distance": float(compiled.median_distance),
            "eager_median_distance": float(eager.median_distance),
            "absolute_residual": residual,
            "tolerance": tolerance,
            "positive_pair_count": int(compiled.positive_pair_count),
            "total_pair_count": int(compiled.total_pair_count),
            "output_device": str(compiled.median_distance.device),
            "xla_trace_count": trace_count,
        },
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": _sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": _sha256(SCRIPT_PATH)},
            "predictive": {
                "path": PREDICTIVE_SOURCE.as_posix(),
                "sha256": _sha256(PREDICTIVE_SOURCE),
            },
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
            "random_seeds": "N/A deterministic range fixture",
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "wall_cap_seconds": wall_cap_seconds,
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
        },
        "nonclaims": [
            "fixed-shape pairwise-distance engineering canary only",
            "no target pilot, calibration, equivalence, posterior, or model claim",
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
        raise PairwiseShapeError("pairwise-distance canary requires a visible GPU")
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
                "median_distance": payload["result"]["compiled_median_distance"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
