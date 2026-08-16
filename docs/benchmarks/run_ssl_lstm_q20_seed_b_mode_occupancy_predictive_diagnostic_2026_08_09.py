#!/usr/bin/env python3
"""Diagnose seed-B retained mode coverage and fixed-mode predictive laws."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from ssl_lstm_q20_neutra_seed_b_terminal import build_seed_b_terminal


PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-plan-2026-08-09.md"
)
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_seed_b_mode_occupancy_predictive_diagnostic_2026_08_09.py"
)
ARCHIVE_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-terminal-neutra-validation-2026-08-07/r2/sequential"
)
MAP_ARTIFACT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-posterior-reference-2026-08-07/r3/map-progress.json"
)
ENERGY_SOURCE = Path("bayesfilter/testing/two_sample_energy_tf.py")
FORECAST_SOURCE = Path("bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py")
TRANSPORT_LOADER = Path("docs/benchmarks/ssl_lstm_q20_neutra_seed_b_terminal.py")
DEFAULT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-2026-08-09/r2"
)

TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
BASE_ADAPTER_SIGNATURE = "a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3"
HORIZONS = (10, 20, 30, 50, 100)
SAMPLE_SIZE = 1000
PERMUTATION_COUNT = 9999
PERMUTATION_BATCH_SIZE = 250
ALPHA = 0.01
CHAIN_COUNT = 4
RETAINED_PER_CHAIN = 1000
WARMUP_PER_CHAIN = 2000
PARAMETER_DIM = 4
OBSERVATION_WEIGHT_INDEX = 2
THREADS = 8
Q = 20
CANARY_SAMPLE_SIZE = 32
CANARY_HORIZON = 20
CANARY_PERMUTATIONS = 999
CANARY_CAP_SECONDS = 300.0
CAMPAIGN_CAP_SECONDS = 1200.0
SEED_WORD = 20260809
REPRESENTATIVE_CODES = {"plus": 31, "minus": 47}


class ModeDiagnosticError(RuntimeError):
    """Raised when a mode diagnostic invariant or artifact contract fails."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_abs(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ModeDiagnosticError(f"expected JSON object: {path}")
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
        raise ModeDiagnosticError(f"refusing to overwrite artifact: {path}")
    absolute.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _write_tensor(path: Path, tensor: Any, tf: Any) -> dict[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise ModeDiagnosticError(f"refusing to overwrite artifact: {path}")
    payload = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "dtype": tensor.dtype.name,
        "shape": tensor.shape,
    }


def classify_p_value(p_value: float) -> str:
    if not math.isfinite(p_value) or not 0.0 < p_value <= 1.0:
        raise ModeDiagnosticError("p-value must be finite and in (0,1]")
    return (
        "DISTINGUISHED_AT_1_PERCENT"
        if p_value < ALPHA
        else "NOT_DISTINGUISHED_AT_1_PERCENT"
    )


def _seeds(representative: str, horizon: int) -> dict[str, tuple[int, int]]:
    if representative not in REPRESENTATIVE_CODES or horizon not in HORIZONS:
        raise ModeDiagnosticError("invalid representative or horizon seed request")
    base = REPRESENTATIVE_CODES[representative] * 10000 + horizon * 10
    return {
        "true": (SEED_WORD, base + 1),
        "representative": (SEED_WORD, base + 2),
        "permutation": (SEED_WORD, base + 3),
    }


def _canary_seeds() -> dict[str, tuple[int, int]]:
    return {
        "left": (SEED_WORD, 900001),
        "right": (SEED_WORD, 900002),
        "permutation": (SEED_WORD, 900003),
    }


def _map_representatives(tf: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _read_json(MAP_ARTIFACT)
    starts = artifact.get("starts")
    if not isinstance(starts, list) or len(starts) < 2:
        raise ModeDiagnosticError("MAP artifact lacks multistart endpoints")
    eligible: dict[str, list[dict[str, Any]]] = {"plus": [], "minus": []}
    for row in starts:
        if not isinstance(row, Mapping):
            continue
        position = row.get("position")
        try:
            values = [float(value) for value in position]
            log_prob = float(row.get("log_prob"))
            score = float(row.get("score_inf_norm"))
        except (TypeError, ValueError):
            continue
        if (
            len(values) != PARAMETER_DIM
            or not all(math.isfinite(value) for value in values)
            or not math.isfinite(log_prob)
            or not math.isfinite(score)
            or score > 1.0e-5
            or values[OBSERVATION_WEIGHT_INDEX] == 0.0
        ):
            continue
        label = "plus" if values[OBSERVATION_WEIGHT_INDEX] > 0.0 else "minus"
        eligible[label].append(
            {
                "position": values,
                "log_prob": log_prob,
                "score_inf_norm": score,
                "start_index": int(row.get("start_index", -1)),
            }
        )
    if not eligible["plus"] or not eligible["minus"]:
        raise ModeDiagnosticError("MAP artifact lacks eligible endpoints of both signs")
    selected = {
        label: max(rows, key=lambda row: row["log_prob"])
        for label, rows in eligible.items()
    }
    representatives = {
        label: tf.ensure_shape(
            tf.convert_to_tensor(row["position"], tf.float64), [PARAMETER_DIM]
        )
        for label, row in selected.items()
    }
    receipt = {
        "path": MAP_ARTIFACT.as_posix(),
        "sha256": _sha(MAP_ARTIFACT),
        "eligibility": "finite_log_prob_and_coordinates_score_inf_norm_le_1e-5",
        "selection": "maximum_log_prob_within_observation_weight_sign",
        "eligible_count_by_sign": {
            label: len(rows) for label, rows in eligible.items()
        },
        "selected": selected,
    }
    return representatives, receipt


def _load_retained(tf: Any) -> tuple[Any, dict[str, Any]]:
    summary_path = ARCHIVE_ROOT / "summary.json"
    summary = _read_json(summary_path)
    if summary.get("status") != "SEQUENTIAL_SCREEN_PASSED" or summary.get("passed") is not True:
        raise ModeDiagnosticError("seed-B sequential archive is not passed")
    if int(summary.get("retained_results_per_chain", -1)) != RETAINED_PER_CHAIN:
        raise ModeDiagnosticError("unexpected retained draw count")
    if int(summary.get("warmup_results_per_chain", -1)) != WARMUP_PER_CHAIN:
        raise ModeDiagnosticError("unexpected warm-up count")
    manifest_path = ARCHIVE_ROOT / "archive/seed-b-terminal-manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("warmup_excluded_from_posterior") is not True:
        raise ModeDiagnosticError("warm-up exclusion is not authenticated")
    chunks = []
    receipts = []
    for index in (0, 1):
        receipt_path = (
            ARCHIVE_ROOT
            / "archive/retained"
            / f"seed-b-terminal-retained-{index:03d}-receipt.json"
        )
        receipt = _read_json(receipt_path)
        sample = receipt.get("sample_receipt")
        if not isinstance(sample, Mapping):
            raise ModeDiagnosticError("retained receipt lacks sample descriptor")
        sample_path = Path(str(sample.get("path")))
        if _sha(sample_path) != sample.get("sha256"):
            raise ModeDiagnosticError(f"retained tensor hash mismatch: {index}")
        tensor = tf.io.parse_tensor(_abs(sample_path).read_bytes(), out_type=tf.float64)
        if tuple(tensor.shape) != (500, CHAIN_COUNT, PARAMETER_DIM):
            raise ModeDiagnosticError("retained tensor shape mismatch")
        if not bool(tf.reduce_all(tf.math.is_finite(tensor))):
            raise ModeDiagnosticError("retained tensor is nonfinite")
        chunks.append(tensor)
        receipts.append(
            {
                "receipt": receipt_path.as_posix(),
                "receipt_sha256": _sha(receipt_path),
                "sample": sample,
            }
        )
    samples = tf.transpose(tf.concat(chunks, axis=0), (1, 0, 2))
    if tuple(samples.shape) != (CHAIN_COUNT, RETAINED_PER_CHAIN, PARAMETER_DIM):
        raise ModeDiagnosticError("concatenated retained sample shape mismatch")
    return samples, {
        "summary_path": summary_path.as_posix(),
        "summary_sha256": _sha(summary_path),
        "manifest_path": manifest_path.as_posix(),
        "manifest_sha256": _sha(manifest_path),
        "retained_receipts": receipts,
        "shape_chain_draw_parameter": list(samples.shape),
        "warmup_excluded": True,
    }


def _count_transitions(tf: Any, labels: Any) -> int:
    return int(tf.reduce_sum(tf.cast(labels[1:] != labels[:-1], tf.int64)))


def _quantile_summary(tfp: Any, values: Any) -> dict[str, Any]:
    return {
        "minimum": tfp.stats.percentile(values, 0.0, interpolation="linear"),
        "q05": tfp.stats.percentile(values, 5.0, interpolation="linear"),
        "median": tfp.stats.percentile(values, 50.0, interpolation="linear"),
        "q95": tfp.stats.percentile(values, 95.0, interpolation="linear"),
        "maximum": tfp.stats.percentile(values, 100.0, interpolation="linear"),
    }


def _occupancy(tf: Any, tfp: Any, z: Any, transport: Any, representatives: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    flat_z = tf.reshape(z, (-1, PARAMETER_DIM))
    theta_flat = tf.convert_to_tensor(
        transport.forward_z_to_theta_batch(flat_z), tf.float64
    )
    if tuple(theta_flat.shape) != (
        CHAIN_COUNT * RETAINED_PER_CHAIN,
        PARAMETER_DIM,
    ) or not bool(tf.reduce_all(tf.math.is_finite(theta_flat))):
        raise ModeDiagnosticError("mapped retained draws are invalid")
    theta = tf.reshape(
        theta_flat, (CHAIN_COUNT, RETAINED_PER_CHAIN, PARAMETER_DIM)
    )
    sign_label = tf.cast(theta[:, :, OBSERVATION_WEIGHT_INDEX] < 0.0, tf.int32)
    observation_weight = theta[:, :, OBSERVATION_WEIGHT_INDEX]
    rows = []
    for chain in range(CHAIN_COUNT):
        sign_chain = sign_label[chain]
        sign_plus_count = int(tf.reduce_sum(tf.cast(sign_chain == 0, tf.int64)))
        rows.append(
            {
                "chain": chain,
                "draw_count": RETAINED_PER_CHAIN,
                "observation_weight_half_space": {
                    "plus_count": sign_plus_count,
                    "minus_count": RETAINED_PER_CHAIN - sign_plus_count,
                    "plus_fraction": sign_plus_count / RETAINED_PER_CHAIN,
                    "minus_fraction": 1.0 - sign_plus_count / RETAINED_PER_CHAIN,
                    "transition_count": _count_transitions(tf, sign_chain),
                },
                "observation_weight_summary": _quantile_summary(
                    tfp, observation_weight[chain]
                ),
                "physical_parameter_mean": tf.reduce_mean(theta[chain], axis=0),
                "physical_parameter_median": tfp.stats.percentile(
                    theta[chain], 50.0, axis=0, interpolation="linear"
                ),
            }
        )
    total = CHAIN_COUNT * RETAINED_PER_CHAIN
    sign_plus = int(tf.reduce_sum(tf.cast(sign_label == 0, tf.int64)))
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_seed_b_mode_region_coverage.v2",
        "classification": "DESCRIPTIVE_HALF_SPACE_COVERAGE_ONLY",
        "parameter_names": (
            "latent_mean_weight.0.0",
            "latent_mean_bias.0",
            "observation_weight.0.0",
            "observation_bias.0",
        ),
        "coverage_coordinate": "observation_weight.0.0",
        "coverage_boundary": 0.0,
        "positive_map_observation_weight": representatives["plus"][OBSERVATION_WEIGHT_INDEX],
        "negative_map_observation_weight": representatives["minus"][OBSERVATION_WEIGHT_INDEX],
        "chains": rows,
        "pooled": {
            "draw_count": total,
            "positive_half_space_count": sign_plus,
            "negative_half_space_count": total - sign_plus,
            "positive_half_space_fraction": sign_plus / total,
            "negative_half_space_fraction": 1.0 - sign_plus / total,
            "observation_weight_summary": _quantile_summary(
                tfp, tf.reshape(observation_weight, [-1])
            ),
        },
        "physical_parameter_summary": {
            "mean": tf.reduce_mean(theta_flat, axis=0),
            "median": tfp.stats.percentile(
                theta_flat, 50.0, axis=0, interpolation="linear"
            ),
        },
        "nonclaims": (
            "not an estimate of integrated mode mass",
            "not formal basin-membership classification",
            "not an audit of unrecorded leapfrog trajectory states",
            "autocorrelated draws make occupancy fractions descriptive only",
            "two known centers do not rule out additional modes",
        ),
    }
    return theta, payload


def _forecast_paths(tf: Any, forecast: Any, parameter: Any, *, horizon: int, sample_size: int, seed: tuple[int, int]) -> tuple[Any, dict[str, Any]]:
    started = time.perf_counter()
    result = forecast(
        tf.ensure_shape(parameter[tf.newaxis, :], [1, PARAMETER_DIM]),
        q=Q,
        seed=tf.constant(seed, tf.int32),
        replication_count=sample_size,
        horizon=horizon,
    )
    paths = tf.reshape(
        tf.convert_to_tensor(result.observations, tf.float64),
        [sample_size, horizon],
    )
    if not bool(tf.reduce_all(result.status)) or not bool(
        tf.reduce_all(tf.math.is_finite(paths))
    ):
        raise ModeDiagnosticError("forecast path validity failed")
    return paths, {
        "parameter": parameter,
        "seed": seed,
        "sample_size": sample_size,
        "horizon": horizon,
        "shape": paths.shape,
        "construction_signature": result.construction_signature,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _path_summary(tf: Any, paths: Any) -> dict[str, Any]:
    return {
        "overall_mean": tf.reduce_mean(paths),
        "overall_variance": tf.math.reduce_variance(paths),
        "timewise_mean": tf.reduce_mean(paths, axis=0),
        "timewise_variance": tf.math.reduce_variance(paths, axis=0),
    }


def _one_test(tf: Any, forecast: Any, energy_test: Any, *, representative: str, parameter: Any, horizon: int, sample_size: int, permutation_count: int, seeds: Mapping[str, tuple[int, int]], output_root: Path, label: str) -> dict[str, Any]:
    started = time.perf_counter()
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER

    true_parameter = tf.convert_to_tensor(PRIOR_CENTER, tf.float64)
    true_paths, true_receipt = _forecast_paths(
        tf,
        forecast,
        true_parameter,
        horizon=horizon,
        sample_size=sample_size,
        seed=seeds["true"],
    )
    representative_paths, representative_receipt = _forecast_paths(
        tf,
        forecast,
        parameter,
        horizon=horizon,
        sample_size=sample_size,
        seed=seeds["representative"],
    )
    statistic_started = time.perf_counter()
    result = energy_test(
        true_paths,
        representative_paths,
        permutation_count=permutation_count,
        seed=seeds["permutation"],
        permutation_batch_size=PERMUTATION_BATCH_SIZE,
        jit_compile=True,
    )
    statistic_seconds = time.perf_counter() - statistic_started
    tensor_receipt = _write_tensor(
        output_root / f"{label}-permutation-statistics.tftensor",
        result.permutation_statistics,
        tf,
    )
    p_value = float(result.p_value)
    return {
        "schema": "bayesfilter.ssl_lstm.q20_mode_representative_energy_diagnostic.v1",
        "status": classify_p_value(p_value),
        "representative": representative,
        "parameters": {
            "true": true_parameter,
            "representative": parameter,
        },
        "horizon": horizon,
        "sample_size_per_arm": sample_size,
        "permutation_count": permutation_count,
        "alpha": ALPHA,
        "decision_rule": "reject equality iff p_value < alpha",
        "energy_statistic": result.statistic,
        "p_value": result.p_value,
        "exceedance_count": result.exceedance_count,
        "permutation_seed": seeds["permutation"],
        "permutation_statistics": tensor_receipt,
        "forecast_receipts": {
            "true": true_receipt,
            "representative": representative_receipt,
        },
        "descriptive_summaries": {
            "true": _path_summary(tf, true_paths),
            "representative": _path_summary(tf, representative_paths),
        },
        "descriptive_overall_mean_shift_representative_minus_true": (
            tf.reduce_mean(representative_paths) - tf.reduce_mean(true_paths)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "statistic_and_permutation_seconds": statistic_seconds,
        "jit_compile": True,
        "whole_path_test": True,
        "raw_shared_coordinates": True,
        "nonclaims": (
            "not equivalence evidence",
            "not a within-mode posterior-predictive test",
            "not a posterior-mixture test",
            "not a joint or multiplicity-adjusted test",
        ),
    }


def _provenance() -> dict[str, Any]:
    return {
        "plan": PLAN.as_posix(),
        "plan_sha256": _sha(PLAN),
        "runner": RUNNER.as_posix(),
        "runner_sha256": _sha(RUNNER),
        "transport_loader": TRANSPORT_LOADER.as_posix(),
        "transport_loader_sha256": _sha(TRANSPORT_LOADER),
        "energy_source": ENERGY_SOURCE.as_posix(),
        "energy_source_sha256": _sha(ENERGY_SOURCE),
        "forecast_source": FORECAST_SOURCE.as_posix(),
        "forecast_source_sha256": _sha(FORECAST_SOURCE),
    }


def _manifest(mode: str, started: float, cap_seconds: float, tf: Any) -> dict[str, Any]:
    visible_gpu = tf.config.list_physical_devices("GPU")
    if visible_gpu:
        raise ModeDiagnosticError("CPU-only diagnostic found a visible GPU")
    return {
        "git_commit": subprocess.run(
            ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
            capture_output=True, text=True
        ).stdout.strip(),
        "git_dirty": bool(
            subprocess.run(
                ("git", "status", "--porcelain"), cwd=ROOT, check=True,
                capture_output=True, text=True
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
        "sample_wise_loop_used": False,
        "wall_time_seconds": time.perf_counter() - started,
        "cap_seconds": cap_seconds,
        "artifact_root": DEFAULT_ROOT.as_posix(),
        "plan_file": PLAN.as_posix(),
        "result_file": (
            "docs/plans/"
            "bayesfilter-ssl-lstm-q20-seed-b-mode-occupancy-predictive-diagnostic-result-2026-08-09.md"
        ),
        "random_seeds": "representative/horizon seed table embedded in row receipts",
        "data_version": {
            "retained_archive_summary_sha256": _sha(ARCHIVE_ROOT / "summary.json"),
            "map_artifact_sha256": _sha(MAP_ARTIFACT),
        },
    }


def _setup() -> tuple[Any, Any, Any, Any, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    bridge, transport, transport_provenance = build_seed_b_terminal(
        threads=THREADS,
        evidence_path=PLAN.as_posix(),
        target_scope_suffix="mode_occupancy_predictive_diagnostic",
    )
    del bridge
    import tensorflow as tf
    import tensorflow_probability as tfp
    from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
        forecast_complexity_conditional_moments,
    )
    from bayesfilter.testing.two_sample_energy_tf import (
        whole_path_energy_permutation_test,
    )

    if transport_provenance.get("target_signature") != TARGET_SIGNATURE:
        raise ModeDiagnosticError("target signature mismatch")
    if transport_provenance.get("target_adapter_signature") != BASE_ADAPTER_SIGNATURE:
        raise ModeDiagnosticError("base adapter signature mismatch")
    z, archive_receipt = _load_retained(tf)
    representatives, map_receipt = _map_representatives(tf)
    theta, occupancy = _occupancy(tf, tfp, z, transport, representatives)
    return (
        tf,
        forecast_complexity_conditional_moments,
        whole_path_energy_permutation_test,
        theta,
        occupancy,
        representatives,
        {
            "transport": dict(transport_provenance),
            "archive": archive_receipt,
            "map": map_receipt,
        },
        _provenance(),
    )


def run(mode: str, output_root: Path, cap_seconds: float) -> dict[str, Any]:
    if mode not in {"canary", "campaign"}:
        raise ModeDiagnosticError("mode must be canary or campaign")
    started = time.perf_counter()
    (
        tf,
        forecast,
        energy_test,
        theta,
        occupancy,
        representatives,
        input_receipts,
        provenance,
    ) = _setup()
    output_root = Path(output_root)
    if mode == "canary":
        seeds = _canary_seeds()
        from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER

        row = _one_test(
            tf,
            forecast,
            energy_test,
            representative="true_control_null_canary",
            parameter=tf.convert_to_tensor(PRIOR_CENTER, tf.float64),
            horizon=CANARY_HORIZON,
            sample_size=CANARY_SAMPLE_SIZE,
            permutation_count=CANARY_PERMUTATIONS,
            seeds={
                "true": seeds["left"],
                "representative": seeds["right"],
                "permutation": seeds["permutation"],
            },
            output_root=output_root,
            label="canary",
        )
        payload = {
            "schema": "bayesfilter.ssl_lstm.q20_seed_b_mode_diagnostic_canary.v1",
            "status": "CANARY_PASSED",
            "role": "mechanics_only",
            "mapped_draw_shape": list(theta.shape),
            "occupancy": occupancy,
            "representatives": representatives,
            "row": row,
            "input_receipts": input_receipts,
            "provenance": provenance,
        }
        payload["run_manifest"] = _manifest(mode, started, cap_seconds, tf)
        if time.perf_counter() - started > cap_seconds:
            raise ModeDiagnosticError("canary wall cap exceeded")
        _write_json(output_root / "canary.json", payload)
        return payload

    canary_path = output_root / "canary.json"
    if not _abs(canary_path).is_file():
        raise ModeDiagnosticError("campaign requires canary.json")
    canary = _read_json(canary_path)
    if canary.get("status") != "CANARY_PASSED":
        raise ModeDiagnosticError("campaign canary did not pass")
    if canary.get("provenance") != _safe(provenance):
        raise ModeDiagnosticError("campaign canary source binding mismatch")

    occupancy_path = output_root / "occupancy.json"
    occupancy_payload = {
        **occupancy,
        "representatives": representatives,
        "input_receipts": input_receipts,
        "provenance": provenance,
    }
    _write_json(occupancy_path, occupancy_payload)
    rows = []
    for representative in ("plus", "minus"):
        for horizon in HORIZONS:
            label = f"{representative}-t{horizon:03d}"
            row = _one_test(
                tf,
                forecast,
                energy_test,
                representative=representative,
                parameter=representatives[representative],
                horizon=horizon,
                sample_size=SAMPLE_SIZE,
                permutation_count=PERMUTATION_COUNT,
                seeds=_seeds(representative, horizon),
                output_root=output_root,
                label=label,
            )
            row_payload = {
                **row,
                "input_receipts": input_receipts,
                "provenance": provenance,
            }
            row_path = output_root / f"{label}.json"
            _write_json(row_path, row_payload)
            rows.append(
                {
                    "representative": representative,
                    "horizon": horizon,
                    "status": row["status"],
                    "energy_statistic": row["energy_statistic"],
                    "p_value": row["p_value"],
                    "exceedance_count": row["exceedance_count"],
                    "overall_mean_shift": row[
                        "descriptive_overall_mean_shift_representative_minus_true"
                    ],
                    "elapsed_seconds": row["elapsed_seconds"],
                    "receipt": row_path.as_posix(),
                    "receipt_sha256": _sha(row_path),
                }
            )
            if time.perf_counter() - started > cap_seconds:
                raise ModeDiagnosticError("campaign wall cap exceeded")
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_seed_b_mode_occupancy_predictive_diagnostic.v2",
        "status": "TEN_DIAGNOSTICS_COMPLETED",
        "occupancy": {
            "receipt": occupancy_path.as_posix(),
            "receipt_sha256": _sha(occupancy_path),
            "pooled": occupancy["pooled"],
        },
        "representatives": representatives,
        "rows": rows,
        "horizons": HORIZONS,
        "sample_size_per_arm": SAMPLE_SIZE,
        "permutation_count_per_test": PERMUTATION_COUNT,
        "alpha_per_test": ALPHA,
        "joint_test_computed": False,
        "combined_p_value_computed": False,
        "multiplicity_adjustment_applied": False,
        "mode_mass_estimated": False,
        "posterior_predictive_mixture_computed": False,
        "input_receipts": input_receipts,
        "provenance": provenance,
        "nonclaims": (
            "not proof that multimodality caused posterior-mean rejection",
            "not integrated mode-mass evidence",
            "not a within-mode or mixture posterior-predictive test",
            "not equality evidence after non-rejection",
            "not a joint or familywise decision",
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
                    "occupancy": payload.get("occupancy", {}).get("pooled"),
                    "rows": payload.get("rows"),
                    "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"],
                }
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
