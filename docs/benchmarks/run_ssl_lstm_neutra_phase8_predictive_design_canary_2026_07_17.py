#!/usr/bin/env python3
"""Trusted GPU/XLA mechanics and timing canary for Phase 8 design work."""

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
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bayesfilter.inference.predictive_equivalence as predictive_equivalence  # noqa: E402
from bayesfilter.inference.predictive_equivalence import (  # noqa: E402
    PredictiveStatisticsConfig,
    adapt_ssl_lstm_observations,
    chain_batch_long_run_covariance,
    summarize_forecast_paths,
)
from bayesfilter.nonlinear.ssl_lstm_predictive_tf import (  # noqa: E402
    SSLLSTMForecastConfig,
    forecast_ssl_lstm_paths,
    make_ssl_lstm_innovation_bank,
    ssl_lstm_forecast_compiled_program,
    ssl_lstm_terminal_compiled_program,
)


PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "plan-2026-07-17.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-phase-8-predictive-design-refresh-"
    "result-2026-07-17.md"
)
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-8-predictive-design"
)
PHASE7_RECEIPT_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "phase-7-retained-admission/retained-acquisition.json"
)
PHASE7_RECEIPT_SHA256 = (
    "b79e5f6041e284de40bbd3834cc909fd12f45d012f172e570acccaa62dbe31a5"
)
A0_LOCK_PATH = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a0/target-lock.json"
)
A0_LOCK_SHA256 = (
    "1f7fccbeafbaa344a80e77c73b4356f44258b78a65ea2499e8ebd194b79a4383"
)
FORECAST_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py")
STATISTICS_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
TARGET_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py")
ORIGINAL_STARTS = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
SHARED_SEED = (10101, 10102)
INDEPENDENT_SEED = (10201, 10202)
RIDGE_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CONDITION_NUMBER_MAX = 1.0e8


class Phase8CanaryError(RuntimeError):
    """Raised when a Phase 8 canary invariant fails."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise Phase8CanaryError(f"duplicate JSON key {key!r}: {path}")
            result[key] = value
        return result

    def reject(value: str) -> None:
        raise Phase8CanaryError(f"nonfinite JSON constant {value!r}: {path}")

    value = json.loads(
        _absolute(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )
    if not isinstance(value, dict):
        raise Phase8CanaryError(f"expected JSON object: {path}")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if hasattr(value, "numpy"):
        return _json_safe(value.numpy())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _absolute(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise Phase8CanaryError(f"refusing to overwrite receipt: {path}")
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


def validate_phase7_receipt() -> dict[str, Any]:
    if _sha256(PHASE7_RECEIPT_PATH) != PHASE7_RECEIPT_SHA256:
        raise Phase8CanaryError("Phase 7 receipt byte identity drift")
    receipt = _strict_json(PHASE7_RECEIPT_PATH)
    if receipt.get("status") != "PASSED" or receipt.get("decision") != (
        "PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF"
    ):
        raise Phase8CanaryError("Phase 7 handoff decision drift")
    if receipt.get("both_charts_admitted") is not True or receipt.get("hard_vetoes") != []:
        raise Phase8CanaryError("Phase 7 admission state drift")
    if receipt.get("cross_replication_stability", {}).get("passed") is not True:
        raise Phase8CanaryError("Phase 7 replication stability drift")
    return {
        "path": PHASE7_RECEIPT_PATH.as_posix(),
        "sha256": PHASE7_RECEIPT_SHA256,
        "decision": receipt["decision"],
        "retained_samples_read_by_canary": False,
    }


def a0_start_points() -> tf.Tensor:
    if _sha256(A0_LOCK_PATH) != A0_LOCK_SHA256:
        raise Phase8CanaryError("A0 target-lock byte identity drift")
    lock = _strict_json(A0_LOCK_PATH)
    geometry = lock.get("sampler_geometry")
    if not isinstance(geometry, Mapping):
        raise Phase8CanaryError("A0 sampler geometry missing")
    center = tf.constant(geometry["center_free"]["values"], tf.float64)
    scale = tf.constant(geometry["scale"]["values"], tf.float64)
    factor_z = tf.constant(geometry["factor_z"]["values"], tf.float64)
    factor = tf.linalg.diag(scale) @ factor_z
    points = center + tf.constant(ORIGINAL_STARTS, tf.float64) @ tf.transpose(factor)
    if tuple(points.shape) != (4, 4) or not bool(
        tf.reduce_all(tf.math.is_finite(points)).numpy()
    ):
        raise Phase8CanaryError("A0 start-point reconstruction failed")
    return points


def make_canary_banks(config: SSLLSTMForecastConfig) -> dict[str, Any]:
    shared = make_ssl_lstm_innovation_bank(
        config,
        4,
        tf.constant(SHARED_SEED, tf.int32),
        "paired_diagnostic_shared",
        0,
    )
    independent_g = make_ssl_lstm_innovation_bank(
        config,
        4,
        tf.constant(INDEPENDENT_SEED, tf.int32),
        "independent_arm",
        1,
    )
    independent_h = make_ssl_lstm_innovation_bank(
        config,
        4,
        tf.constant(INDEPENDENT_SEED, tf.int32),
        "independent_arm",
        2,
    )
    banks = {
        "shared": shared,
        "independent-g": independent_g,
        "independent-h": independent_h,
    }
    signatures = [bank.content_signature for bank in banks.values()]
    if len(set(signatures)) != len(signatures):
        raise Phase8CanaryError("canary innovation-bank signatures are not distinct")
    tensor_hashes = [hash_value for bank in banks.values() for hash_value in bank.tensor_hashes().values()]
    if len(set(tensor_hashes)) != len(tensor_hashes):
        raise Phase8CanaryError("canary innovation tensor families overlap")
    return banks


def _trace_count(program: Any) -> int | None:
    method = getattr(program, "experimental_get_tracing_count", None)
    return None if method is None else int(method())


def _tensor_devices(tensors: Mapping[str, tf.Tensor]) -> dict[str, str]:
    return {name: str(tensor.device) for name, tensor in tensors.items()}


def _require_gpu_devices(devices: Mapping[str, str], *, surface: str) -> None:
    non_gpu = {name: device for name, device in devices.items() if "GPU:" not in device}
    if non_gpu:
        raise Phase8CanaryError(f"{surface} outputs are not GPU resident: {non_gpu}")


def _source_bindings() -> dict[str, Any]:
    return {
        name: {"path": path.as_posix(), "sha256": _sha256(path)}
        for name, path in {
            "plan": PLAN_PATH,
            "runner": SCRIPT_PATH,
            "forecast": FORECAST_SOURCE,
            "statistics": STATISTICS_SOURCE,
            "target": TARGET_SOURCE,
        }.items()
    }


def run_canary(*, output: Path, wall_cap_seconds: float) -> dict[str, Any]:
    if not math.isfinite(wall_cap_seconds) or wall_cap_seconds <= 0.0:
        raise Phase8CanaryError("wall cap must be positive and finite")
    started_at = _now()
    started = time.perf_counter()
    phase7 = validate_phase7_receipt()
    points = a0_start_points()
    config = SSLLSTMForecastConfig()
    config.assert_evidence_config()
    banks = make_canary_banks(config)
    forecast_rows: dict[str, Any] = {}
    path_objects: dict[str, Any] = {}
    for label, bank in banks.items():
        call_started = time.perf_counter()
        paths = forecast_ssl_lstm_paths(
            points,
            bank,
            config,
            runtime_execution_role="trusted_gpu_xla_canary",
            trust_basis="owner_designated_managed_session_visible_gpu_trusted",
        )
        elapsed = time.perf_counter() - call_started
        if any(status != 0 for status in paths.provenance.terminal_covariance_statuses):
            raise Phase8CanaryError(f"terminal covariance gate failed: {label}")
        if not paths.provenance.output_devices or not all(
            "GPU:" in device for device in paths.provenance.output_devices
        ):
            raise Phase8CanaryError(f"forecast outputs are not GPU resident: {label}")
        adapted = adapt_ssl_lstm_observations(paths.observations)
        summary_started = time.perf_counter()
        summary = summarize_forecast_paths(adapted, PredictiveStatisticsConfig())
        summary_elapsed = time.perf_counter() - summary_started
        terminal_devices = _tensor_devices(
            {
                "mean": paths.terminal.mean,
                "implemented_covariance": paths.terminal.implemented_covariance,
                "factor": paths.terminal.factor,
                "full_parameters": paths.terminal.full_parameters,
            }
        )
        summary_devices = _tensor_devices(
            {
                "means": summary.means,
                "variances": summary.variances,
                "log_variances": summary.log_variances,
                "central_moments": summary.central_moments,
                "quantiles": summary.quantiles,
                "cross_horizon_covariance": summary.cross_horizon_covariance,
            }
        )
        _require_gpu_devices(terminal_devices, surface=f"terminal {label}")
        _require_gpu_devices(summary_devices, surface=f"summary {label}")
        forecast_rows[label] = {
            "elapsed_seconds": elapsed,
            "summary_elapsed_seconds": summary_elapsed,
            "bank": {
                "role": bank.role,
                "arm_id": bank.arm_id,
                "root_seed": _json_safe(bank.root_seed),
                "content_signature": bank.content_signature,
                "tensor_hashes": bank.tensor_hashes(),
            },
            "forecast": {
                "draw_count": paths.provenance.draw_count,
                "replication_count": paths.provenance.replication_count,
                "horizon": paths.provenance.forecast_horizon,
                "cluster_unit": paths.provenance.cluster_unit,
                "jit_compile": paths.provenance.jit_compile,
                "execution_role": paths.provenance.execution_role,
                "output_devices": paths.provenance.output_devices,
                "innovation_bank_signature": paths.provenance.innovation_bank_signature,
                "free_draw_matrix_raw_sha256": paths.provenance.free_draw_matrix_raw_sha256,
                "terminal_output_devices": terminal_devices,
            },
            "summary": {
                "status": _json_safe(summary.status),
                "path_count": int(summary.path_count.numpy()),
                "all_primary_finite": bool(
                    tf.reduce_all(tf.math.is_finite(summary.means)).numpy()
                    and tf.reduce_all(tf.math.is_finite(summary.log_variances)).numpy()
                ),
                "output_devices": summary_devices,
            },
        }
        path_objects[label] = paths
        if time.perf_counter() - started > wall_cap_seconds:
            raise Phase8CanaryError("Phase 8 canary wall cap exceeded")

    shared_observations = tf.squeeze(path_objects["shared"].observations, axis=-1)
    covariance_features = tf.concat(
        (shared_observations, tf.square(shared_observations)), axis=-1
    )
    covariance_started = time.perf_counter()
    covariance = chain_batch_long_run_covariance(
        covariance_features,
        block_length=1,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
    )
    covariance_elapsed = time.perf_counter() - covariance_started
    if not covariance.inference_admissible:
        raise Phase8CanaryError("long-run covariance mechanics canary is inadmissible")
    if "GPU:" not in str(covariance.regularized_covariance.device):
        raise Phase8CanaryError("long-run covariance output is not GPU resident")

    terminal_program = ssl_lstm_terminal_compiled_program(config, 4)
    forecast_program = ssl_lstm_forecast_compiled_program(config, 4)
    trace_counts = {
        "terminal": _trace_count(terminal_program),
        "forecast": _trace_count(forecast_program),
        "statistics": _trace_count(predictive_equivalence._summary_xla),
        "long_run_covariance": _trace_count(
            predictive_equivalence._long_run_covariance_xla
        ),
    }
    if any(count != 1 for count in trace_counts.values()):
        raise Phase8CanaryError(f"compiled surface trace-count gate failed: {trace_counts}")
    wall_time = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm_neutra.phase8_predictive_design_canary.v1",
        "status": "PASSED",
        "decision": "PHASE8_ENGINEERING_CANARY_PASSED_RESOURCE_FREEZE_REQUIRED",
        "phase7_binding": phase7,
        "contract": {
            "point_source": "four_A0_start_derived_free_coordinate_points",
            "point_count": 4,
            "forecast_horizon": 10,
            "forecast_replication_count": 2,
            "bank_roles": ["paired_diagnostic_shared", "independent_arm", "independent_arm"],
            "retained_phase7_samples_read": False,
            "confirmatory_forecast_bank_opened": False,
            "calibration_parameters_frozen": False,
        },
        "forecasts": forecast_rows,
        "long_run_covariance": {
            "elapsed_seconds": covariance_elapsed,
            "feature_shape": list(covariance_features.shape),
            "block_length": covariance.block_length,
            "ridge_ladder": list(RIDGE_LADDER),
            "condition_number_max": CONDITION_NUMBER_MAX,
            "selected_ridge_index": int(covariance.selected_ridge_index.numpy()),
            "selected_ridge_multiplier": float(covariance.selected_ridge_multiplier.numpy()),
            "condition_number": float(covariance.condition_number.numpy()),
            "inference_admissible": covariance.inference_admissible,
            "output_device": str(covariance.regularized_covariance.device),
            "diagnostic_role": "singular_mechanics_fixture_not_weight_calibration",
        },
        "compile_trace_counts": trace_counts,
        "source_bindings": _source_bindings(),
        "run_manifest": {
            "command": " ".join(shlex.quote(item) for item in (sys.executable, *sys.argv)),
            "cwd": str(ROOT),
            "interpreter": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": __import__("tensorflow_probability").__version__,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_devices": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_devices": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "dtype": "float64",
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(_git("status", "--porcelain").strip()),
            "random_seeds": {
                "shared": list(SHARED_SEED),
                "independent_root": list(INDEPENDENT_SEED),
                "independent_arm_ids": [1, 2],
            },
            "started_at_utc": started_at,
            "completed_at_utc": _now(),
            "wall_time_seconds": wall_time,
            "wall_cap_seconds": wall_cap_seconds,
            "output_path": output.as_posix(),
            "plan_path": PLAN_PATH.as_posix(),
            "result_path": RESULT_PATH.as_posix(),
        },
        "nonclaims": [
            "four-point engineering and timing canary only",
            "no Phase 7 retained sample or confirmatory G/H forecast was read",
            "no calibrated margin, bandwidth, block length, bootstrap count, or power claim",
            "no predictive equivalence, posterior correctness, model adequacy, or ranking claim",
        ],
    }
    _write_json(output, payload)
    return payload


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise Phase8CanaryError("Phase 8 canary requires a visible trusted GPU")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise Phase8CanaryError("Phase 8 canary requires a logical GPU")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, required=True)
    args = parser.parse_args(argv)
    _require_gpu()
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
