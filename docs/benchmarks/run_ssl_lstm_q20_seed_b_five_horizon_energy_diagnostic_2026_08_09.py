#!/usr/bin/env python3
"""Five separate whole-path energy diagnostics for two fixed q=20 simulators."""

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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARKS = ROOT / "docs" / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-five-horizon-energy-diagnostic-plan-2026-08-09.md"
)
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_five_horizon_energy_diagnostic_2026_08_09.py"
)
ENERGY_SOURCE = Path("bayesfilter/testing/two_sample_energy_tf.py")
FORECAST_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py")
PARAMETER_ARTIFACT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-plugin-predictive-comparison-2026-08-08/r4/material.json"
)
PARAMETER_ARTIFACT_SHA256 = (
    "72ba9c7034e36f26e76d0d6542c3aa0ab6699e4d21fe0f727ca5dea275663f09"
)
DEFAULT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-five-horizon-energy-diagnostic-2026-08-09/r1"
)

TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BASE_ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
Q = 20
PARAMETER_DIM = 4
HORIZONS = (10, 20, 30, 50, 100)
SAMPLE_SIZE = 1000
PERMUTATION_COUNT = 9999
PERMUTATION_BATCH_SIZE = 250
ALPHA = 0.01
CANARY_HORIZON = 20
CANARY_SAMPLE_SIZE = 32
CANARY_PERMUTATIONS = 999
CANARY_CAP_SECONDS = 900.0
CAMPAIGN_CAP_SECONDS = 7200.0
THREADS = 8
SEED_WORD = 20260809
CANARY_SEEDS = {
    "left": (SEED_WORD, 210001),
    "right": (SEED_WORD, 210002),
    "permutation": (SEED_WORD, 210003),
}


class EnergyCampaignError(RuntimeError):
    """Raised when a campaign invariant or artifact contract fails."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_abs(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EnergyCampaignError(f"expected a JSON object: {path}")
    return payload


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "as_list"):
        return _safe(value.as_list())
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise EnergyCampaignError(f"refusing to overwrite artifact: {path}")
    absolute.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _write_tensor(path: Path, tensor: Any, tf: Any) -> dict[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise EnergyCampaignError(f"refusing to overwrite artifact: {path}")
    payload = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "dtype": tensor.dtype.name,
        "shape": tensor.shape,
    }


def classify_p_value(p_value: float, *, alpha: float = ALPHA) -> str:
    if not 0.0 < p_value <= 1.0:
        raise EnergyCampaignError("p-value must lie in (0, 1]")
    if not 0.0 < alpha < 1.0:
        raise EnergyCampaignError("alpha must lie in (0, 1)")
    return (
        "DISTINGUISHED_AT_1_PERCENT"
        if p_value < alpha
        else "NOT_DISTINGUISHED_AT_1_PERCENT"
    )


def _seeds(horizon: int) -> dict[str, tuple[int, int]]:
    if horizon not in HORIZONS:
        raise EnergyCampaignError("horizon lies outside the frozen diagnostic grid")
    root = 300000 + 1000 * horizon
    return {
        "left": (SEED_WORD, root + 1),
        "right": (SEED_WORD, root + 2),
        "permutation": (SEED_WORD, root + 3),
    }


def _configure_tensorflow() -> tuple[Any, Any, Any]:
    from ssl_lstm_q20_neutra_seed_b_terminal import configure_cpu_tensorflow

    tf = configure_cpu_tensorflow(threads=THREADS)
    if tf.config.list_physical_devices("GPU"):
        raise EnergyCampaignError("CPU diagnostic found a visible GPU")
    from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
        forecast_complexity_conditional_moments,
    )
    from bayesfilter.testing.two_sample_energy_tf import (
        whole_path_energy_permutation_test,
    )

    return tf, forecast_complexity_conditional_moments, whole_path_energy_permutation_test


def _parameters(tf: Any) -> tuple[Any, Any, dict[str, Any]]:
    if _sha(PARAMETER_ARTIFACT) != PARAMETER_ARTIFACT_SHA256:
        raise EnergyCampaignError("posterior-mean source artifact hash mismatch")
    source = _read_json(PARAMETER_ARTIFACT)
    if source.get("target_signature") != TARGET_SIGNATURE:
        raise EnergyCampaignError("posterior-mean source target signature mismatch")
    if source.get("base_adapter_signature") != BASE_ADAPTER_SIGNATURE:
        raise EnergyCampaignError("posterior-mean source adapter signature mismatch")
    summaries = source.get("parameter_summaries")
    if not isinstance(summaries, Mapping):
        raise EnergyCampaignError("posterior-mean source summaries are missing")
    posterior_mean = tf.convert_to_tensor(summaries.get("posterior_mean"), tf.float64)
    true = tf.convert_to_tensor(summaries.get("true"), tf.float64)
    if posterior_mean.shape != (PARAMETER_DIM,) or true.shape != (PARAMETER_DIM,):
        raise EnergyCampaignError("fixed parameter vectors have the wrong shape")
    try:
        tf.debugging.assert_all_finite(posterior_mean, "posterior mean must be finite")
        tf.debugging.assert_all_finite(true, "true parameter must be finite")
    except tf.errors.InvalidArgumentError as exc:
        raise EnergyCampaignError("fixed parameter vectors must be finite") from exc
    return posterior_mean, true, {
        "path": PARAMETER_ARTIFACT.as_posix(),
        "sha256": PARAMETER_ARTIFACT_SHA256,
    }


def _forecast_paths(
    tf: Any,
    forecast: Any,
    parameter: Any,
    *,
    horizon: int,
    sample_size: int,
    seed: tuple[int, int],
) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    result = forecast(
        tf.ensure_shape(parameter[tf.newaxis, :], [1, PARAMETER_DIM]),
        q=Q,
        seed=tf.constant(seed, tf.int32),
        replication_count=sample_size,
        horizon=horizon,
    )
    paths = tf.ensure_shape(
        tf.reshape(tf.convert_to_tensor(result.observations, tf.float64), [sample_size, horizon]),
        [sample_size, horizon],
    )
    if result.horizon != horizon or result.replication_count != sample_size:
        raise EnergyCampaignError("forecast result geometry mismatch")
    if not bool(tf.reduce_all(result.status)):
        raise EnergyCampaignError("forecast status failed")
    if not bool(tf.reduce_all(tf.math.is_finite(paths))):
        raise EnergyCampaignError("forecast paths are nonfinite")
    return paths, {
        "seed": seed,
        "construction_signature": result.construction_signature,
        "target_signature": result.target_signature,
        "shape": paths.shape,
        "elapsed_seconds": time.perf_counter() - started,
        "device": paths.device,
    }


def _summary(tf: Any, paths: Any) -> dict[str, Any]:
    return {
        "mean": tf.reduce_mean(paths, axis=0),
        "variance": tf.math.reduce_variance(paths, axis=0),
        "overall_mean": tf.reduce_mean(paths),
        "overall_variance": tf.math.reduce_variance(paths),
    }


def _permutation_summary(tf: Any, values: Any) -> dict[str, Any]:
    ordered = tf.sort(values)
    count = int(ordered.shape[0])

    def quantile(probability: float) -> Any:
        index = int(round(probability * (count - 1)))
        return ordered[index]

    return {
        "minimum": ordered[0],
        "q01": quantile(0.01),
        "q05": quantile(0.05),
        "median": quantile(0.50),
        "q95": quantile(0.95),
        "q99": quantile(0.99),
        "maximum": ordered[-1],
        "mean": tf.reduce_mean(ordered),
        "standard_deviation": tf.math.reduce_std(ordered),
    }


def _provenance() -> dict[str, Any]:
    return {
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha(PLAN),
        "runner": RUNNER.as_posix(),
        "runner_sha256": _sha(RUNNER),
        "energy_source": ENERGY_SOURCE.as_posix(),
        "energy_source_sha256": _sha(ENERGY_SOURCE),
        "forecast_source": FORECAST_SOURCE.as_posix(),
        "forecast_source_sha256": _sha(FORECAST_SOURCE),
    }


def _manifest(mode: str, started: float, cap_seconds: float, tf: Any) -> dict[str, Any]:
    return {
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"),
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        ),
        "command": " ".join(sys.argv),
        "mode": mode,
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python": platform.python_version(),
        "tensorflow": tf.__version__,
        "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cpu_threads": THREADS,
        "jit_compile": True,
        "cpu_only_diagnostic_exception": True,
        "wall_time_seconds": time.perf_counter() - started,
        "cap_seconds": cap_seconds,
    }


def _one_test(
    *,
    tf: Any,
    forecast: Any,
    energy_test: Any,
    left_parameter: Any,
    right_parameter: Any,
    horizon: int,
    sample_size: int,
    permutation_count: int,
    seeds: Mapping[str, tuple[int, int]],
    output_root: Path,
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    left, left_receipt = _forecast_paths(
        tf,
        forecast,
        left_parameter,
        horizon=horizon,
        sample_size=sample_size,
        seed=seeds["left"],
    )
    right, right_receipt = _forecast_paths(
        tf,
        forecast,
        right_parameter,
        horizon=horizon,
        sample_size=sample_size,
        seed=seeds["right"],
    )
    statistic_started = time.perf_counter()
    result = energy_test(
        left,
        right,
        permutation_count=permutation_count,
        seed=seeds["permutation"],
        permutation_batch_size=PERMUTATION_BATCH_SIZE,
        jit_compile=True,
    )
    statistic_seconds = time.perf_counter() - statistic_started
    tensor_path = output_root / f"{label}-permutation-statistics.tftensor"
    tensor_receipt = _write_tensor(tensor_path, result.permutation_statistics, tf)
    p_value = float(result.p_value)
    status = classify_p_value(p_value)
    return {
        "schema": "bayesfilter.ssl_lstm.q20_whole_path_energy_diagnostic.v1",
        "status": status,
        "horizon": horizon,
        "sample_size_per_arm": sample_size,
        "permutation_count": permutation_count,
        "p_value_resolution": 1.0 / float(permutation_count + 1),
        "alpha": ALPHA,
        "decision_rule": "reject equality iff p_value < alpha",
        "energy_statistic": result.statistic,
        "p_value": result.p_value,
        "exceedance_count": result.exceedance_count,
        "permutation_seed": seeds["permutation"],
        "permutation_statistics": tensor_receipt,
        "permutation_summary": _permutation_summary(tf, result.permutation_statistics),
        "forecast_receipts": {"left": left_receipt, "right": right_receipt},
        "descriptive_summaries": {"left": _summary(tf, left), "right": _summary(tf, right)},
        "descriptive_differences": {
            "mean": tf.reduce_mean(right, axis=0) - tf.reduce_mean(left, axis=0),
            "variance": tf.math.reduce_variance(right, axis=0) - tf.math.reduce_variance(left, axis=0),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "statistic_and_permutation_seconds": statistic_seconds,
        "jit_compile": True,
        "whole_path_test": True,
        "raw_shared_coordinates": True,
        "arm_specific_standardization_used": False,
        "nonclaims": (
            "not proof of equal DGPs",
            "not practical-equivalence evidence",
            "not a joint or multiplicity-adjusted test",
            "not evidence for horizons outside this row",
        ),
    }


def run(mode: str, output_root: Path, cap_seconds: float) -> dict[str, Any]:
    if mode not in {"canary", "campaign"}:
        raise EnergyCampaignError("mode must be canary or campaign")
    started = time.perf_counter()
    tf, forecast, energy_test = _configure_tensorflow()
    posterior_mean, true, parameter_receipt = _parameters(tf)
    output_root = Path(output_root)
    if mode == "canary":
        row = _one_test(
            tf=tf,
            forecast=forecast,
            energy_test=energy_test,
            left_parameter=true,
            right_parameter=true,
            horizon=CANARY_HORIZON,
            sample_size=CANARY_SAMPLE_SIZE,
            permutation_count=CANARY_PERMUTATIONS,
            seeds=CANARY_SEEDS,
            output_root=output_root,
            label="canary",
        )
        payload = {
            "schema": "bayesfilter.ssl_lstm.q20_five_horizon_energy_canary.v1",
            "status": "CANARY_PASSED",
            "role": "mechanics_only_true_vs_true_single_realization",
            "row": row,
            "parameter_source": parameter_receipt,
            "parameters": {"true": true},
            "provenance": _provenance(),
        }
        payload["run_manifest"] = _manifest(mode, started, cap_seconds, tf)
        if time.perf_counter() - started > cap_seconds:
            raise EnergyCampaignError("canary wall cap exceeded")
        _write_json(output_root / "canary.json", payload)
        return payload

    canary_path = output_root / "canary.json"
    if not _abs(canary_path).is_file():
        raise EnergyCampaignError("campaign requires canary.json")
    canary = _read_json(canary_path)
    if canary.get("status") != "CANARY_PASSED":
        raise EnergyCampaignError("campaign canary did not pass")
    if canary.get("provenance") != _safe(_provenance()):
        raise EnergyCampaignError("campaign canary source binding mismatch")

    rows = []
    for horizon in HORIZONS:
        label = f"t{horizon:03d}"
        row = _one_test(
            tf=tf,
            forecast=forecast,
            energy_test=energy_test,
            left_parameter=true,
            right_parameter=posterior_mean,
            horizon=horizon,
            sample_size=SAMPLE_SIZE,
            permutation_count=PERMUTATION_COUNT,
            seeds=_seeds(horizon),
            output_root=output_root,
            label=label,
        )
        row_payload = {
            **row,
            "parameter_source": parameter_receipt,
            "parameters": {"left_true": true, "right_posterior_mean": posterior_mean},
            "target_signature": TARGET_SIGNATURE,
            "base_adapter_signature": BASE_ADAPTER_SIGNATURE,
            "provenance": _provenance(),
        }
        _write_json(output_root / f"{label}.json", row_payload)
        rows.append(
            {
                "horizon": horizon,
                "status": row["status"],
                "energy_statistic": row["energy_statistic"],
                "p_value": row["p_value"],
                "exceedance_count": row["exceedance_count"],
                "receipt": (output_root / f"{label}.json").as_posix(),
                "receipt_sha256": _sha(output_root / f"{label}.json"),
                "elapsed_seconds": row["elapsed_seconds"],
            }
        )
        if time.perf_counter() - started > cap_seconds:
            raise EnergyCampaignError("campaign wall cap exceeded")
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_five_horizon_energy_diagnostic.v1",
        "status": "FIVE_DIAGNOSTICS_COMPLETED",
        "rows": rows,
        "horizons": HORIZONS,
        "diagnostic_count": len(HORIZONS),
        "sample_size_per_arm": SAMPLE_SIZE,
        "permutation_count_per_horizon": PERMUTATION_COUNT,
        "alpha_per_horizon": ALPHA,
        "joint_test_computed": False,
        "combined_p_value_computed": False,
        "multiplicity_adjustment_applied": False,
        "independent_test_fwer_arithmetic": 1.0 - (1.0 - ALPHA) ** len(HORIZONS),
        "independent_test_fwer_is_not_combined_p_value": True,
        "all_not_distinguished": all(
            row["status"] == "NOT_DISTINGUISHED_AT_1_PERCENT" for row in rows
        ),
        "parameters": {"true": true, "posterior_mean": posterior_mean},
        "parameter_source": parameter_receipt,
        "provenance": _provenance(),
        "nonclaims": (
            "five non-rejections would not prove equality",
            "no joint or familywise decision",
            "no predictive-equivalence conclusion",
            "no claim for untested horizons or the infinite process law",
        ),
    }
    payload["run_manifest"] = _manifest(mode, started, cap_seconds, tf)
    _write_json(output_root / "summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "campaign"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cap-seconds", type=float)
    args = parser.parse_args(argv)
    cap = (
        CANARY_CAP_SECONDS if args.mode == "canary" else CAMPAIGN_CAP_SECONDS
    ) if args.cap_seconds is None else args.cap_seconds
    payload = run(args.mode, args.output_root, cap)
    print(
        json.dumps(
            _safe(
                {
                    "mode": args.mode,
                    "status": payload["status"],
                    "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
                    "rows": payload.get("rows"),
                }
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
