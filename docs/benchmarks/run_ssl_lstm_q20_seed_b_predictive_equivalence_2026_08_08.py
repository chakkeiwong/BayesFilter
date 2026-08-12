#!/usr/bin/env python3
"""Formal q=20 fixed-plug-in predictive equivalence comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
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

pe: Any = None


PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-seed-b-predictive-equivalence-plan-2026-08-08.md")
PLUGIN_PLAN = Path("docs/plans/bayesfilter-ssl-lstm-q20-seed-b-plugin-predictive-comparison-plan-2026-08-08.md")
ARCHIVE_ROOT = Path("docs/plans/artifacts/ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r2/sequential")
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BASE_ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
Q = 20
CHAIN_COUNT = 4
PARAMETER_DIM = 4
EXPECTED_DRAWS = 1000
HORIZON = 10
CALIBRATION_DRAWS = 1024
MATERIAL_DRAWS = 2048
REPLICATIONS = 2
CANARY_DRAWS = 32
BLOCK_LENGTH = 16
FEATURE_ALPHA = 0.03
MMD_ALPHA = 0.02
TOTAL_ALPHA = 0.05
MEAN_MARGIN = 0.15
LOG_VARIANCE_MARGIN = 0.13976194237515863
MMD_TOLERANCES = (0.005, 0.01, 0.02, 0.04, 0.08, 0.16)
RIDGE_LADDER = (0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6)
CONDITION_NUMBER_MAX = 1.0e8
CALIBRATION_ROOTS = (85000, 85010, 85020, 85030, 85040, 85050, 85060, 85070)
CALIBRATION_SHIFT_ROOTS = (85100, 85110, 85120, 85130, 85140, 85150, 85160, 85170)
MATERIAL_TRUE_SEEDS = ((20260808, 86001), (20260808, 86002), (20260808, 86003), (20260808, 86004))
MATERIAL_MEAN_SEEDS = ((20260808, 86101), (20260808, 86102), (20260808, 86103), (20260808, 86104))


class EquivalenceError(RuntimeError):
    pass


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(_abs(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EquivalenceError(f"expected JSON object: {path}")
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
        raise EquivalenceError(f"refusing to overwrite {path}")
    absolute.write_text(json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_retained() -> tuple[Any, dict[str, Any]]:
    import tensorflow as tf

    summary = _json(ARCHIVE_ROOT / "summary.json")
    if summary.get("status") != "SEQUENTIAL_SCREEN_PASSED" or summary.get("passed") is not True:
        raise EquivalenceError("seed-B sequential archive is not admitted")
    if int(summary.get("retained_results_per_chain", -1)) != EXPECTED_DRAWS:
        raise EquivalenceError("unexpected retained draw count")
    manifest_path = ARCHIVE_ROOT / "archive/seed-b-terminal-manifest.json"
    manifest = _json(manifest_path)
    if manifest.get("warmup_excluded_from_posterior") is not True:
        raise EquivalenceError("warm-up exclusion is not authenticated")
    chunks = []
    receipts = []
    for index in (0, 1):
        receipt_path = ARCHIVE_ROOT / "archive/retained" / f"seed-b-terminal-retained-{index:03d}-receipt.json"
        receipt = _json(receipt_path)
        sample = receipt.get("sample_receipt")
        if not isinstance(sample, dict):
            raise EquivalenceError("retained receipt lacks sample descriptor")
        sample_path = Path(str(sample.get("path")))
        if _sha(sample_path) != sample.get("sha256"):
            raise EquivalenceError(f"retained tensor hash mismatch: {index}")
        tensor = tf.io.parse_tensor(_abs(sample_path).read_bytes(), out_type=tf.float64)
        if tuple(tensor.shape) != (500, CHAIN_COUNT, PARAMETER_DIM):
            raise EquivalenceError("retained tensor shape mismatch")
        if not bool(tf.reduce_all(tf.math.is_finite(tensor))):
            raise EquivalenceError("retained tensor is nonfinite")
        chunks.append(tensor)
        receipts.append({"receipt": receipt_path.as_posix(), "receipt_sha256": _sha(receipt_path), "sample": sample})
    samples = tf.transpose(tf.concat(chunks, axis=0), (1, 0, 2))
    return samples, {"summary_sha256": _sha(ARCHIVE_ROOT / "summary.json"), "archive_manifest_sha256": _sha(manifest_path), "retained_receipts": receipts, "shape": list(samples.shape)}


def _mean_parameter(z: Any, transport: Any, provenance: dict[str, Any]) -> tuple[Any, Any]:
    import tensorflow as tf
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER

    if provenance.get("target_signature") != TARGET_SIGNATURE or provenance.get("target_adapter_signature") != BASE_ADAPTER_SIGNATURE:
        raise EquivalenceError("target or adapter signature mismatch")
    theta = tf.convert_to_tensor(transport.forward_z_to_theta_batch(tf.reshape(z, (-1, PARAMETER_DIM))), tf.float64)
    if tuple(theta.shape) != (CHAIN_COUNT * EXPECTED_DRAWS, PARAMETER_DIM) or not bool(tf.reduce_all(tf.math.is_finite(theta))):
        raise EquivalenceError("mapped retained draws invalid")
    return tf.convert_to_tensor(PRIOR_CENTER, tf.float64), tf.reduce_mean(theta, axis=0)


def _forecast(parameter: Any, seeds: tuple[tuple[int, int], ...], draws: int, *, shift: float = 0.0) -> Any:
    import tensorflow as tf
    from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import forecast_complexity_conditional_moments

    rows = []
    for lane, seed in enumerate(seeds):
        result = forecast_complexity_conditional_moments(tf.ensure_shape(parameter[tf.newaxis, :], [1, PARAMETER_DIM]), q=Q, seed=tf.constant(seed, tf.int32), replication_count=draws * REPLICATIONS)
        observations = tf.reshape(tf.convert_to_tensor(result.observations, tf.float64), [draws, REPLICATIONS, HORIZON])
        if shift:
            observations = observations + tf.constant(shift, tf.float64)
        if not bool(tf.reduce_all(result.status)) or not bool(tf.reduce_all(tf.math.is_finite(observations))):
            raise EquivalenceError("forecast validity failed")
        rows.append(observations)
    return tf.stack(rows, axis=0)


def _lane_seeds(root: int) -> tuple[tuple[int, int], ...]:
    return tuple((20260808, root + lane) for lane in range(CHAIN_COUNT))


def _features(paths: Any) -> tuple[Any, Any]:
    import tensorflow as tf

    # The statistical engine expects [chain, draw, replication, horizon].
    summary = pe.mean_log_variance_influence(paths, jit_compile=True)
    return summary.feature_estimate, summary.influence_values


def _design_calibration(true_parameter: Any) -> dict[str, Any]:
    import tensorflow as tf

    null_banks = []
    shift_banks = []
    for root, shift_root in zip(CALIBRATION_ROOTS, CALIBRATION_SHIFT_ROOTS):
        null_banks.append(_forecast(true_parameter, _lane_seeds(root), CALIBRATION_DRAWS))
        shift_banks.append(_forecast(true_parameter, _lane_seeds(shift_root), CALIBRATION_DRAWS, shift=0.20))
    # The distance-scale API is quadratic; freeze it on a predeclared calibration
    # subset, while the material MMD uses every material path.
    pooled = tf.concat(tuple(tf.concat((left[:, :128], right[:, :128]), axis=0) for left, right in zip(null_banks, shift_banks)), axis=0)
    scale = pe.pooled_pairwise_distance_scale(pooled, jit_compile=True)
    if not bool(tf.reduce_all(tf.math.is_finite(scale.median_distance))):
        raise EquivalenceError("q20 calibration distance scale is invalid")
    bandwidths = tf.constant((0.5, 1.0, 2.0), tf.float64) * scale.median_distance
    margins = tf.concat((tf.fill([HORIZON], tf.constant(MEAN_MARGIN, tf.float64)), tf.fill([HORIZON], tf.constant(LOG_VARIANCE_MARGIN, tf.float64))), axis=0)
    candidate_rows = []
    for tolerance in MMD_TOLERANCES:
        replication_rows = []
        for null_left, shift_right in zip(null_banks, shift_banks):
            null_left_feature, null_left_influence = _features(null_left)
            null_right_feature, null_right_influence = _features(shift_right - tf.constant(0.20, tf.float64))
            null_estimate = null_left_feature - null_right_feature
            null_influence = tf.concat((null_left_influence, -null_right_influence), axis=0)
            covariance = pe.chain_batch_long_run_covariance(null_influence, block_length=BLOCK_LENGTH, ridge_ladder=RIDGE_LADDER, condition_number_max=CONDITION_NUMBER_MAX)
            if not covariance.inference_admissible:
                raise EquivalenceError("q20 calibration covariance is inadmissible")
            se = tf.sqrt(tf.linalg.diag_part(covariance.regularized_covariance))
            interval = pe.simultaneous_feature_intervals(null_estimate, feature_alpha=FEATURE_ALPHA, method="bonferroni_studentized", standard_error=se)
            null_mmd = pe.cross_chain_linear_mmd(null_left, shift_right - tf.constant(0.20, tf.float64), bandwidths=bandwidths, mixture_weights=tf.fill([3], tf.constant(1.0 / 3.0, tf.float64)), chain_pair_schedule=tf.constant(((0, 1), (2, 3)), tf.int32), independent_arm_banks_verified=True, stationarity_verified=True, mixing_verified=True)
            null_interval = pe.cross_chain_mmd_upper_interval(null_mmd, mmd_alpha=MMD_ALPHA, block_length=BLOCK_LENGTH)
            shifted_mmd = pe.cross_chain_linear_mmd(null_left, shift_right, bandwidths=bandwidths, mixture_weights=tf.fill([3], tf.constant(1.0 / 3.0, tf.float64)), chain_pair_schedule=tf.constant(((0, 1), (2, 3)), tf.int32), independent_arm_banks_verified=True, stationarity_verified=True, mixing_verified=True)
            shifted_interval = pe.cross_chain_mmd_upper_interval(shifted_mmd, mmd_alpha=MMD_ALPHA, block_length=BLOCK_LENGTH)
            decision = pe.classify_predictive_evidence(interval, null_interval, margins=margins, mmd_tolerance=tf.constant(tolerance, tf.float64), total_alpha=TOTAL_ALPHA, feature_alpha=FEATURE_ALPHA, mmd_alpha=MMD_ALPHA)
            replication_rows.append({"status": decision.status, "null_upper": null_interval.upper, "shift_lower": shifted_interval.lower})
        candidate_rows.append({"tolerance": tolerance, "replications": replication_rows})
    selected = next((row["tolerance"] for row in candidate_rows if all(r["status"] == "PASS" and float(r["shift_lower"]) > row["tolerance"] for r in row["replications"])), None)
    if selected is None:
        return {"status": "CALIBRATION_INCONCLUSIVE", "reason": "no frozen MMD tolerance both passed every null replication and detected every controlled-shift replication", "center": tf.reduce_mean(pooled, axis=(0, 1, 2)), "scale": tf.math.reduce_std(tf.reshape(pooled, [-1, HORIZON]), axis=0), "median_distance": scale.median_distance, "bandwidths": bandwidths, "selected_mmd_tolerance": None, "candidate_rows": candidate_rows, "calibration_contract": {"draws_per_lane": CALIBRATION_DRAWS, "distance_scale_draws_per_lane": 128, "replications": len(CALIBRATION_ROOTS), "lanes": CHAIN_COUNT, "shift": 0.20, "margins_are_transferred_working_hypotheses": True}}
    return {"status": "CALIBRATION_PASSED", "center": tf.reduce_mean(pooled, axis=(0, 1, 2)), "scale": tf.math.reduce_std(tf.reshape(pooled, [-1, HORIZON]), axis=0), "median_distance": scale.median_distance, "bandwidths": bandwidths, "selected_mmd_tolerance": selected, "candidate_rows": candidate_rows, "calibration_contract": {"draws_per_lane": CALIBRATION_DRAWS, "distance_scale_draws_per_lane": 128, "replications": len(CALIBRATION_ROOTS), "lanes": CHAIN_COUNT, "shift": 0.20, "margins_are_transferred_working_hypotheses": True}}


def _material(true_parameter: Any, mean_parameter: Any, calibration: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf

    true_paths = _forecast(true_parameter, MATERIAL_TRUE_SEEDS, MATERIAL_DRAWS)
    mean_paths = _forecast(mean_parameter, MATERIAL_MEAN_SEEDS, MATERIAL_DRAWS)
    true_features, true_influence = _features(true_paths)
    mean_features, mean_influence = _features(mean_paths)
    estimate = mean_features - true_features
    influence = tf.concat((mean_influence, -true_influence), axis=0)
    covariance = pe.chain_batch_long_run_covariance(influence, block_length=BLOCK_LENGTH, ridge_ladder=RIDGE_LADDER, condition_number_max=CONDITION_NUMBER_MAX)
    if not covariance.inference_admissible:
        raise EquivalenceError("material feature covariance inadmissible")
    se = tf.sqrt(tf.linalg.diag_part(covariance.regularized_covariance))
    feature_interval = pe.simultaneous_feature_intervals(estimate, feature_alpha=FEATURE_ALPHA, method="bonferroni_studentized", standard_error=se)
    bandwidths = tf.convert_to_tensor(calibration["bandwidths"], tf.float64)
    mmd = pe.cross_chain_linear_mmd(mean_paths, true_paths, bandwidths=bandwidths, mixture_weights=tf.fill([3], tf.constant(1.0 / 3.0, tf.float64)), chain_pair_schedule=tf.constant(((0, 1), (2, 3)), tf.int32), independent_arm_banks_verified=True, stationarity_verified=True, mixing_verified=True)
    mmd_interval = pe.cross_chain_mmd_upper_interval(mmd, mmd_alpha=MMD_ALPHA, block_length=BLOCK_LENGTH)
    if not mmd_interval.inference_admissible:
        raise EquivalenceError("material MMD interval inadmissible")
    margins = tf.concat((tf.fill([HORIZON], tf.constant(MEAN_MARGIN, tf.float64)), tf.fill([HORIZON], tf.constant(LOG_VARIANCE_MARGIN, tf.float64))), axis=0)
    decision = pe.classify_predictive_evidence(feature_interval, mmd_interval, margins=margins, mmd_tolerance=tf.constant(float(calibration["selected_mmd_tolerance"]), tf.float64), total_alpha=TOTAL_ALPHA, feature_alpha=FEATURE_ALPHA, mmd_alpha=MMD_ALPHA)
    return {"status": decision.status, "decision": {"status": decision.status, "primary_interval_status": decision.primary_interval_status, "mmd_upper_bound_status": decision.mmd_upper_bound_status, "hard_veto_codes": list(decision.hard_veto_codes)}, "feature_estimate": estimate, "feature_interval": {"lower": feature_interval.lower, "upper": feature_interval.upper, "standard_error": feature_interval.standard_error, "critical_value": feature_interval.critical_value}, "mmd_interval": {"estimate": mmd_interval.estimate, "lower": mmd_interval.lower, "upper": mmd_interval.upper, "standard_error": mmd_interval.standard_error, "critical_value": mmd_interval.critical_value}, "raw_summaries": {"true": pe.summarize_forecast_paths(true_paths), "posterior_mean": pe.summarize_forecast_paths(mean_paths)}, "contract": {"draws_per_lane": MATERIAL_DRAWS, "replications": REPLICATIONS, "lanes": CHAIN_COUNT, "independent_banks": True, "parameter_mixture_used": False, "mmd_tolerance": calibration["selected_mmd_tolerance"]}}


def run(mode: str, output_root: Path, cap_seconds: float) -> dict[str, Any]:
    global pe
    if mode not in {"canary", "calibration", "material"}:
        raise EquivalenceError("mode must be canary, calibration, or material")
    started = time.perf_counter()
    _, transport, provenance = build_seed_b_terminal(threads=1, evidence_path=PLAN.as_posix(), target_scope_suffix="predictive_equivalence")
    from bayesfilter.inference import predictive_equivalence as predictive_api
    pe = predictive_api
    z, archive = _load_retained()
    true_parameter, mean_parameter = _mean_parameter(z, transport, dict(provenance))
    if mode == "canary":
        paths = _forecast(true_parameter, CALIBRATION_SEEDS, CANARY_DRAWS)
        payload = {"schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence.canary.v1", "status": "CANARY_PASSED", "path_shape": list(paths.shape), "finite": bool(__import__("tensorflow").reduce_all(__import__("tensorflow").math.is_finite(paths))), "jit_compile": True}
    elif mode == "calibration":
        calibration = _design_calibration(true_parameter)
        payload = {"schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence.calibration.v1", "status": calibration["status"], "calibration": calibration, "parameter_summaries": {"true": true_parameter}, "archive": archive, "target_signature": TARGET_SIGNATURE, "base_adapter_signature": BASE_ADAPTER_SIGNATURE}
    else:
        calibration_path = output_root / "calibration.json"
        if not _abs(calibration_path).is_file():
            raise EquivalenceError("material mode requires calibration.json")
        calibration = _json(calibration_path)["calibration"]
        if calibration.get("status") != "CALIBRATION_PASSED" or calibration.get("selected_mmd_tolerance") is None:
            closure = {"schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence.material.v1", "status": "MATERIAL_CLOSED_CALIBRATION_INCONCLUSIVE", "calibration_sha256": _sha(calibration_path), "reason": calibration.get("reason", "q20 calibration was not passed"), "parameter_mixture_used": False}
            _write(output_root / "material-closed.json", closure)
            raise EquivalenceError("material mode is closed because q20 calibration was not passed")
        material = _material(true_parameter, mean_parameter, calibration)
        payload = {"schema": "bayesfilter.ssl_lstm.q20_predictive_equivalence.material.v1", "archive": archive, "parameter_summaries": {"true": true_parameter, "posterior_mean": mean_parameter}, "target_signature": TARGET_SIGNATURE, "base_adapter_signature": BASE_ADAPTER_SIGNATURE, "calibration_sha256": _sha(calibration_path), "material": material}
    payload.update({"mode": mode, "run_manifest": {"command": " ".join(sys.argv), "python": platform.python_version(), "tensorflow": __import__("tensorflow").__version__, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "jit_compile": True, "wall_time_seconds": time.perf_counter() - started, "cap_seconds": cap_seconds, "cpu_only_reference_exception": True, "parameter_mixture_used": False}})
    if time.perf_counter() - started > cap_seconds:
        raise EquivalenceError("wall cap exceeded")
    _write(output_root / f"{mode}.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("canary", "calibration", "material"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cap-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    payload = run(args.mode, args.output_root, args.cap_seconds)
    print(json.dumps({"mode": args.mode, "status": payload.get("status", payload.get("material", {}).get("status")), "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
