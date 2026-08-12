#!/usr/bin/env python3
"""Compare q=20 true-parameter and seed-B plug-in predictive laws."""

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
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BENCHMARKS = ROOT / "docs" / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from ssl_lstm_q20_neutra_seed_b_terminal import build_seed_b_terminal


PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-seed-b-plugin-predictive-comparison-plan-2026-08-08.md")
ARCHIVE_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r2/sequential")
ARTIFACT_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-seed-b-plugin-predictive-comparison-2026-08-08/r1")
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BASE_ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
EXPECTED_DRAWS = 1000
CHAIN_COUNT = 4
PARAMETER_DIM = 4
FORECAST_ROWS = 1024
CANARY_ROWS = 32
CANARY_SEED = (20260808, 81001)
MATERIAL_SEED = (20260808, 82001)
Q = 20


class ComparisonError(RuntimeError):
    pass


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_abs(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ComparisonError(f"expected JSON object: {path}")
    return value


def _safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise ComparisonError(f"refusing to overwrite {path}")
    absolute.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parse_tensor(path: Path, dtype: Any = None) -> Any:
    import tensorflow as tf

    if dtype is None:
        dtype = tf.float64
    return tf.io.parse_tensor(_abs(path).read_bytes(), out_type=dtype)


def _load_retained() -> tuple[Any, dict[str, Any]]:
    import tensorflow as tf

    summary = _json(ARCHIVE_ROOT / "summary.json")
    if summary.get("status") != "SEQUENTIAL_SCREEN_PASSED" or summary.get("passed") is not True:
        raise ComparisonError("seed-B sequential screen is not passed")
    if int(summary.get("retained_results_per_chain", -1)) != EXPECTED_DRAWS:
        raise ComparisonError("retained draw count is not 1,000 per chain")
    if int(summary.get("warmup_results_per_chain", -1)) != 2000:
        raise ComparisonError("warm-up count does not match the declared archive")
    manifest_path = ARCHIVE_ROOT / "archive/seed-b-terminal-manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("warmup_excluded_from_posterior") is not True:
        raise ComparisonError("warm-up exclusion is not recorded")
    chunks = []
    receipts = []
    retained_dir = ARCHIVE_ROOT / "archive/retained"
    for index in (0, 1):
        receipt_path = retained_dir / f"seed-b-terminal-retained-{index:03d}-receipt.json"
        receipt = _json(receipt_path)
        sample = receipt.get("sample_receipt")
        if not isinstance(sample, dict):
            raise ComparisonError("retained receipt lacks sample descriptor")
        sample_path = Path(str(sample.get("path")))
        if not sample_path.is_absolute():
            sample_path = ROOT / sample_path
        if _sha(sample_path) != sample.get("sha256"):
            raise ComparisonError(f"retained sample hash mismatch for chunk {index}")
        tensor = _parse_tensor(sample_path)
        if tuple(tensor.shape) != (500, CHAIN_COUNT, PARAMETER_DIM):
            raise ComparisonError(f"retained sample shape mismatch for chunk {index}")
        if not bool(tf.reduce_all(tf.math.is_finite(tensor))):
            raise ComparisonError(f"retained sample nonfinite for chunk {index}")
        chunks.append(tensor)
        receipts.append({"receipt": receipt_path.as_posix(), "receipt_sha256": _sha(receipt_path), "sample": sample})
    samples = tf.concat(chunks, axis=0)
    if tuple(samples.shape) != (EXPECTED_DRAWS, CHAIN_COUNT, PARAMETER_DIM):
        raise ComparisonError("concatenated retained sample shape mismatch")
    return tf.transpose(samples, (1, 0, 2)), {
        "summary_sha256": _sha(ARCHIVE_ROOT / "summary.json"),
        "archive_manifest_sha256": _sha(manifest_path),
        "retained_receipts": receipts,
        "shape_chain_draw_parameter": list(tf.transpose(samples, (1, 0, 2)).shape),
    }


def _summary_parameters(z_chain_major: Any, transport: Any, provenance: dict[str, Any]) -> tuple[Any, Any, Any]:
    import tensorflow as tf

    if provenance.get("target_signature") != TARGET_SIGNATURE:
        raise ComparisonError("target signature mismatch")
    if provenance.get("target_adapter_signature") != BASE_ADAPTER_SIGNATURE:
        raise ComparisonError("base adapter signature mismatch")
    flat_z = tf.reshape(z_chain_major, (-1, PARAMETER_DIM))
    theta = tf.convert_to_tensor(transport.forward_z_to_theta_batch(flat_z), tf.float64)
    if tuple(theta.shape) != (EXPECTED_DRAWS * CHAIN_COUNT, PARAMETER_DIM):
        raise ComparisonError("mapped physical draw shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(theta))):
        raise ComparisonError("mapped physical draws are nonfinite")
    mean = tf.reduce_mean(theta, axis=0)
    median = tfp_percentile(theta, 50.0)
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER

    true = tf.convert_to_tensor(PRIOR_CENTER, tf.float64)
    return true, mean, median


def tfp_percentile(values: Any, percentile: float) -> Any:
    import tensorflow_probability as tfp

    return tfp.stats.percentile(values, percentile, axis=0, interpolation="linear")


def _forecast(label: str, parameter: Any, rows: int, seed: tuple[int, int]) -> dict[str, Any]:
    import tensorflow as tf
    from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
        forecast_complexity_conditional_moments,
    )

    started = time.perf_counter()
    parameter_row = tf.ensure_shape(parameter[tf.newaxis, :], [1, PARAMETER_DIM])
    result = forecast_complexity_conditional_moments(
        parameter_row,
        q=Q,
        seed=tf.constant(seed, tf.int32),
        replication_count=rows,
    )
    observations = tf.convert_to_tensor(result.observations, tf.float64)
    if not bool(tf.reduce_all(result.status)) or not bool(tf.reduce_all(tf.math.is_finite(observations))):
        raise ComparisonError(f"{label} forecast validity failed")
    conditional_means = tf.convert_to_tensor(result.conditional_means, tf.float64)
    conditional_variances = tf.convert_to_tensor(result.conditional_variances, tf.float64)
    elapsed = time.perf_counter() - started
    flat = tf.reshape(observations, (rows, 10))
    mean = tf.reduce_mean(flat, axis=0)
    variance = tf.math.reduce_variance(flat, axis=0)
    quantiles = tfp_percentile(flat, 5.0), tfp_percentile(flat, 50.0), tfp_percentile(flat, 95.0)
    return {
        "label": label,
        "parameter": parameter,
        "parameter_rows": 1,
        "replications": rows,
        "path_count": int(flat.shape[0]),
        "mean": mean,
        "variance": variance,
        "q05": quantiles[0],
        "q50": quantiles[1],
        "q95": quantiles[2],
        "forecast_signature": result.construction_signature,
        "elapsed_seconds": elapsed,
        "output_devices": sorted({str(t.device) for t in (result.observations, conditional_means, conditional_variances)}),
    }


def _diff(left: dict[str, Any], true: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf

    mean_delta = left["mean"] - true["mean"]
    var_delta = left["variance"] - true["variance"]
    return {
        "mean_delta": mean_delta,
        "variance_delta": var_delta,
        "absolute_mean_delta_max": tf.reduce_max(tf.abs(mean_delta)),
        "absolute_variance_delta_max": tf.reduce_max(tf.abs(var_delta)),
        "q05_delta": left["q05"] - true["q05"],
        "q50_delta": left["q50"] - true["q50"],
        "q95_delta": left["q95"] - true["q95"],
    }


def run(mode: str, output_root: Path, cap_seconds: float) -> dict[str, Any]:
    if mode not in {"canary", "material"}:
        raise ComparisonError("mode must be canary or material")
    started = time.perf_counter()
    _, transport, provenance = build_seed_b_terminal(
        threads=1,
        evidence_path=PLAN.as_posix(),
        target_scope_suffix="plugin_predictive_comparison",
    )
    import tensorflow as tf

    z, archive = _load_retained()
    true_parameter, mean_parameter, median_parameter = _summary_parameters(
        z, transport, dict(provenance)
    )
    rows = CANARY_ROWS if mode == "canary" else FORECAST_ROWS
    forecast_seed = CANARY_SEED if mode == "canary" else MATERIAL_SEED
    forecasts = {
        label: _forecast(label, value, rows, forecast_seed)
        for label, value in (("true", true_parameter), ("mean", mean_parameter), ("median", median_parameter))
    }
    if time.perf_counter() - started > cap_seconds:
        raise ComparisonError("wall cap exceeded")
    result = {
        "schema": "bayesfilter.ssl_lstm.q20_seed_b_plugin_predictive_comparison.v1",
        "status": "CANARY_PASSED" if mode == "canary" else "MATERIAL_COMPLETED",
        "mode": mode,
        "q": Q,
        "target_signature": TARGET_SIGNATURE,
        "base_adapter_signature": BASE_ADAPTER_SIGNATURE,
        "archive": archive,
        "parameter_summaries": {
            "true": true_parameter,
            "posterior_mean": mean_parameter,
            "posterior_median": median_parameter,
        },
        "forecasts": forecasts,
        "differences_vs_true": {
            "mean": _diff(forecasts["mean"], forecasts["true"]),
            "median": _diff(forecasts["median"], forecasts["true"]),
        },
        "contract": {
            "parameter_mixture_used": False,
            "parameter_rows_per_arm": 1,
            "repeated_parameter_rows_used": False,
            "forecast_seed": list(forecast_seed),
            "forecast_noise_replications_per_arm": rows,
            "forecast_replication_count": rows,
            "forecast_horizon": 10,
            "jit_compile": True,
            "cuda_visible_devices": "-1",
            "nonclaims": [
                "not posterior-predictive mixture evidence",
                "not independent posterior authority",
                "not mode-mass or parameter-posterior correctness",
                "not model adequacy or default readiness",
            ],
        },
        "run_manifest": {
            "command": " ".join(sys.argv),
            "environment": "tfgpu",
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
            "git_dirty": bool(subprocess.run(("git", "status", "--porcelain"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()),
            "wall_seconds": time.perf_counter() - started,
            "cap_seconds": cap_seconds,
        },
    }
    _write(output_root / f"{mode}.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "material"), required=True)
    parser.add_argument("--output-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--cap-seconds", type=float, default=900.0)
    args = parser.parse_args()
    payload = run(args.mode, args.output_root, args.cap_seconds)
    print(json.dumps(_safe({"status": payload["status"], "mode": payload["mode"], "wall_seconds": payload["run_manifest"]["wall_seconds"]}), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
