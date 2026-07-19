#!/usr/bin/env python3
"""Exact G/H excluded-prefix validation of Phase 8 chunked forecasts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
from types import ModuleType
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesfilter.nonlinear.ssl_lstm_predictive_tf as predictive  # noqa: E402


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "plan-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PILOT_SCRIPT_PATH = Path(
    "docs/benchmarks/run_ssl_lstm_neutra_phase8_target_pilot_2026_07_17.py"
)
CHUNK_CANARY_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "forecast-chunk-repair-canary.json"
)
CHUNK_CANARY_SHA256 = (
    "e78e76203278548183f7974562249e3a292ae4f21e315cd137b955131e342587"
)
TIMEOUT_RECORD_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "target-pilot-repair-01-timeout-record.json"
)
TIMEOUT_RECORD_SHA256 = (
    "54d35f6c32babbb5b9e4f3c6a9b40323f0b8d0448de1715f1e6bccc66664109e"
)


class ExactPrefixChunkError(RuntimeError):
    """Raised when exact-prefix chunk validation fails closed."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, _absolute(path))
    if spec is None or spec.loader is None:
        raise ExactPrefixChunkError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise ExactPrefixChunkError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise ExactPrefixChunkError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise ExactPrefixChunkError(f"expected JSON object: {path}")
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
        raise ExactPrefixChunkError(f"refusing to overwrite receipt: {path}")
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


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def _validate_bindings() -> dict[str, Any]:
    if _sha256(CHUNK_CANARY_PATH) != CHUNK_CANARY_SHA256:
        raise ExactPrefixChunkError("chunk canary identity drift")
    canary = _strict_json(CHUNK_CANARY_PATH)
    if (
        canary.get("status") != "PASSED"
        or canary.get("decision")
        != "PHASE8_FORECAST_CHUNK_REPAIR_PASSED_EXACT_PREFIX_VALIDATION_REQUIRED"
        or canary.get("contract", {}).get("draw_chunk_size") != 16
        or canary.get("forecast", {}).get("terminal_status_nonzero_count") != 0
        or set(canary.get("compile_trace_counts", {}).values()) != {1}
    ):
        raise ExactPrefixChunkError("chunk canary contract drift")
    if _sha256(TIMEOUT_RECORD_PATH) != TIMEOUT_RECORD_SHA256:
        raise ExactPrefixChunkError("target-pilot timeout record identity drift")
    timeout_record = _strict_json(TIMEOUT_RECORD_PATH)
    if (
        timeout_record.get("status") != "FAILED_RESOURCE_TIMEOUT"
        or timeout_record.get("exit_code") != 124
        or timeout_record.get("receipt_written") is not False
    ):
        raise ExactPrefixChunkError("target-pilot timeout record drift")
    return {"canary": canary, "timeout": timeout_record}


def run_validation(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise ExactPrefixChunkError("wall cap must be positive and finite")
    bindings = _validate_bindings()
    started_at = _now()
    started = time.perf_counter()
    pilot = _load_module(PILOT_SCRIPT_PATH, "phase8_pilot_for_chunk_validation")
    if pilot.FORECAST_DRAW_CHUNK_SIZE != 16:
        raise ExactPrefixChunkError("pilot chunk size drift")
    entry = pilot.validate_entry_receipts()
    latent, archive_audit = pilot.read_frozen_pilot_prefix(entry["phase7"])
    config = predictive.SSLLSTMForecastConfig()
    config.assert_evidence_config()

    chart_rows: dict[str, Any] = {}
    bank_hashes: list[str] = []
    for chart in ("fresh-g", "fresh-h"):
        theta, mapping = pilot.map_pilot_to_theta(chart, latent[chart])
        flat_theta = tf.reshape(theta, [256, 4])
        bank = predictive.make_ssl_lstm_innovation_bank(
            config,
            256,
            tf.constant(pilot.PILOT_SEED, tf.int32),
            "independent_arm",
            pilot.ARM_IDS[chart],
        )
        bank_hashes.extend(bank.tensor_hashes().values())
        call_started = time.perf_counter()
        forecast = predictive.forecast_ssl_lstm_paths(
            flat_theta,
            bank,
            config,
            draw_chunk_size=pilot.FORECAST_DRAW_CHUNK_SIZE,
            runtime_execution_role="trusted_gpu_xla_canary",
            trust_basis="owner_designated_managed_session_visible_gpu_trusted",
        )
        call_elapsed = time.perf_counter() - call_started
        statuses = list(forecast.provenance.terminal_covariance_statuses)
        if any(statuses):
            raise ExactPrefixChunkError(f"terminal covariance status failed: {chart}")
        if forecast.provenance.draw_chunk_size != pilot.FORECAST_DRAW_CHUNK_SIZE:
            raise ExactPrefixChunkError(f"forecast chunk-size provenance failed: {chart}")
        if tuple(forecast.observations.shape) != (256, 2, 10, 1):
            raise ExactPrefixChunkError(f"forecast output shape failed: {chart}")
        if not bool(tf.reduce_all(tf.math.is_finite(forecast.observations))):
            raise ExactPrefixChunkError(f"forecast output is nonfinite: {chart}")
        if not all("GPU:" in device for device in forecast.provenance.output_devices):
            raise ExactPrefixChunkError(f"forecast output placement failed: {chart}")
        chart_rows[chart] = {
            "mapping": mapping,
            "draw_count": forecast.provenance.draw_count,
            "draw_chunk_size": forecast.provenance.draw_chunk_size,
            "chunk_count": 256 // forecast.provenance.draw_chunk_size,
            "forecast_replication_count": forecast.provenance.replication_count,
            "forecast_horizon": forecast.provenance.forecast_horizon,
            "elapsed_seconds": call_elapsed,
            "observation_shape": list(forecast.observations.shape),
            "observation_tensor_sha256": hashlib.sha256(
                bytes(tf.io.serialize_tensor(forecast.observations).numpy())
            ).hexdigest(),
            "innovation_bank_signature": bank.content_signature,
            "innovation_tensor_hashes": bank.tensor_hashes(),
            "terminal_status_nonzero_count": sum(status != 0 for status in statuses),
            "output_devices": list(forecast.provenance.output_devices),
        }
        if time.perf_counter() - started > wall_cap_seconds:
            raise ExactPrefixChunkError("exact-prefix chunk validation wall cap exceeded")
    if len(bank_hashes) != len(set(bank_hashes)):
        raise ExactPrefixChunkError("exact-prefix innovation tensor families overlap")

    trace_counts = {
        "terminal_filter_256": _trace_count(
            predictive.ssl_lstm_terminal_compiled_program(config, 256)
        ),
        "terminal_covariance_audit_256": _trace_count(
            predictive.ssl_lstm_terminal_covariance_audit_compiled_program(256)
        ),
        "forecast_chunk_16": _trace_count(
            predictive.ssl_lstm_forecast_compiled_program(config, 16)
        ),
    }
    if any(value != 1 for value in trace_counts.values()):
        raise ExactPrefixChunkError(f"compiled trace gate failed: {trace_counts}")
    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_forecast_chunk_exact_prefix.v1",
        "status": "PASSED",
        "decision": "PHASE8_FORECAST_CHUNK_EXACT_PREFIX_VALIDATED_PILOT_REPAIR_02_ELIGIBLE",
        "entry_bindings": {
            "forecast_chunk_canary": {
                "path": CHUNK_CANARY_PATH.as_posix(),
                "sha256": CHUNK_CANARY_SHA256,
                "decision": bindings["canary"]["decision"],
            },
            "target_pilot_repair_01_timeout": {
                "path": TIMEOUT_RECORD_PATH.as_posix(),
                "sha256": TIMEOUT_RECORD_SHA256,
                "decision": bindings["timeout"]["decision"],
            },
        },
        "scope": {
            "charts_evaluated": ["fresh-g", "fresh-h"],
            "pilot_draw_indices_per_chain": [0, 63],
            "points_per_chart": 256,
            "draw_chunk_size": 16,
            "forecast_replication_count": 2,
            "forecast_horizon": 10,
            "confirmation_suffix_selected": False,
            "confirmation_forecast_opened": False,
            "predictive_summary_computed": False,
            "g_h_predictive_difference_computed": False,
            "target_pilot_retried": False,
        },
        "archive_integrity": archive_audit,
        "charts": chart_rows,
        "innovation_tensor_families_disjoint": True,
        "compile_trace_counts": trace_counts,
        "source_bindings": {
            name: {"path": path.as_posix(), "sha256": _sha256(path)}
            for name, path in {
                "plan": PLAN_PATH,
                "runner": SCRIPT_PATH,
                "pilot_runner": PILOT_SCRIPT_PATH,
                "forecast": Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py"),
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
            "random_seeds": {
                "pilot_root": list(pilot.PILOT_SEED),
                "arm_ids": pilot.ARM_IDS,
            },
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "wall_cap_seconds": wall_cap_seconds,
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
        },
        "nonclaims": [
            "exact excluded-prefix chunk execution validation only",
            "no predictive summary, G/H difference, or Phase 9 confirmation opened",
            "no calibration power, equivalence, posterior, ranking, or model claim",
            "pass only makes a separately recorded target-pilot repair eligible",
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
        raise ExactPrefixChunkError("exact-prefix chunk validation requires a visible GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    with tf.device("/GPU:0"):
        payload = run_validation(
            output=args.output,
            wall_cap_seconds=float(args.wall_cap_seconds),
        )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "decision": payload["decision"],
                "g_elapsed_seconds": payload["charts"]["fresh-g"]["elapsed_seconds"],
                "h_elapsed_seconds": payload["charts"]["fresh-h"]["elapsed_seconds"],
                "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
