#!/usr/bin/env python3
"""Compare preserved PP-UKF retained samples against an archived same-target reference."""

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
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.inference.hmc_convergence import (
    RankNormalizedHMCThresholds,
    rank_normalized_hmc_diagnostics,
)


PLAN = Path("docs/plans/bayesfilter-pp-ukf-posterior-validation-plan-2026-07-30.md")
PUBLIC_RESULT = Path(
    "docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11/public_result.json"
)
REFERENCE_RESULT = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/plain-hmc-affine/attempt-01-20260715T152500Z/result.json"
)
REFERENCE_HASHES = Path(
    "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/plain-hmc-affine/attempt-01-20260715T152500Z/artifact_hashes.json"
)
REFERENCE_ROOT = REFERENCE_RESULT.parent
TARGET_SIGNATURE = "d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5"
TARGET_SCOPE = "PP-UKF-six-probit-initial-observation-first-v1"
PARAMETER_NAMES = (
    "r_source_probit",
    "K_source_probit",
    "a_source_probit",
    "s_source_probit",
    "u_source_probit",
    "v_source_probit",
)
BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_BATCH_SIZE = 20
BOOTSTRAP_SEED = (20260730, 3001)
MEAN_TOLERANCE_SCALE = 0.10
SD_TOLERANCE_SCALE = 0.10
QUANTILE_TOLERANCE_SCALE = 0.15
QUANTILES = (0.05, 0.50, 0.95)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _parse_tensor(path: Path) -> tf.Tensor:
    return tf.io.parse_tensor(path.read_bytes(), out_type=tf.float64)


def _finite_sample(
    path: Path,
    expected_sha256: str,
    expected_shape: tuple[int, int, int],
) -> tf.Tensor:
    if not path.is_file():
        raise ValueError(f"missing sample archive: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"sample archive hash mismatch: {path}")
    samples = _parse_tensor(path)
    if tuple(int(item) for item in samples.shape) != expected_shape:
        raise ValueError(f"sample archive shape mismatch: {path}: {samples.shape}")
    if not bool(tf.reduce_all(tf.math.is_finite(samples)).numpy()):
        raise ValueError(f"sample archive contains nonfinite values: {path}")
    return samples


def _candidate_samples(public: dict[str, Any]) -> list[dict[str, Any]]:
    if public.get("target_signature") != TARGET_SIGNATURE:
        raise ValueError("current PP-UKF result target signature mismatch")
    rows = public.get("candidate_rows")
    if not isinstance(rows, list) or len(rows) != 10:
        raise ValueError("current PP-UKF result must contain ten candidate rows")
    loaded = []
    for index, row in enumerate(rows):
        candidate = row["candidate"]
        if tuple(row["config"].get("parameter_names", PARAMETER_NAMES)) != PARAMETER_NAMES:
            raise ValueError(f"candidate {index} parameter names mismatch")
        archives = [
            item
            for item in row.get("archive", ())
            if item.get("stage") == "retained" and item.get("cumulative") is True
        ]
        if not archives:
            raise ValueError(f"candidate {index} has no cumulative retained archive")
        archive = max(archives, key=lambda item: int(item["shape"][0]))
        shape = tuple(int(item) for item in archive["shape"])
        if shape[1:] != (4, 6):
            raise ValueError(f"candidate {index} has unexpected archive shape: {shape}")
        samples = _finite_sample(
            _repo_path(archive["raw_path"]), archive["raw_sha256"], shape
        )
        loaded.append(
            {
                "candidate_index": index,
                "L": int(candidate["num_leapfrog_steps"]),
                "step_size": float(candidate["step_size"]),
                "candidate_id": candidate["candidate_id"],
                "draws_per_chain": shape[0],
                "archive_path": archive["raw_path"],
                "archive_sha256": archive["raw_sha256"],
                "samples": samples,
            }
        )
    return loaded


def _reference_samples(result: dict[str, Any], hashes: dict[str, Any]) -> dict[str, Any]:
    identity = result.get("target_identity", {})
    if identity.get("mathematical_target_signature") != TARGET_SIGNATURE:
        raise ValueError("reference mathematical target signature mismatch")
    reference_scope = identity.get("batch_execution_surface", {}).get("target_scope")
    if reference_scope != TARGET_SCOPE:
        raise ValueError("reference target scope mismatch")
    summary = result.get("posterior_summary", {})
    if tuple(summary.get("parameter_names", ())) != PARAMETER_NAMES:
        raise ValueError("reference parameter names mismatch")
    archive = result.get("sequential_run", {}).get("cumulative_archives", {}).get("retained", {})
    model_path = archive.get("model_path")
    if not model_path:
        model_path = "docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p4/PP-UKF/plain-hmc-affine/attempt-01-20260715T152500Z/samples/retained/cumulative/model.tensor"
    model_path_obj = Path(model_path)
    relative = (
        str(model_path_obj.relative_to(ROOT))
        if model_path_obj.is_absolute()
        else str(model_path_obj)
    )
    ledger_source = model_path_obj.relative_to(ROOT) if model_path_obj.is_absolute() else model_path_obj
    ledger_key = str(ledger_source.relative_to(REFERENCE_ROOT))
    expected_hash = hashes.get("artifacts", {}).get(ledger_key)
    if expected_hash is None:
        raise ValueError(f"reference hash ledger has no model archive entry: {ledger_key}")
    shape = (int(archive.get("sample_shape", [4000, 4, 6])[0]), 4, 6)
    samples = _finite_sample(_repo_path(relative), expected_hash, shape)
    return {
        "candidate_index": None,
        "L": None,
        "step_size": None,
        "candidate_id": "affine_plain_hmc_reference",
        "draws_per_chain": shape[0],
        "archive_path": relative,
        "archive_sha256": expected_hash,
        "samples": samples,
    }


def _linear_quantiles(values: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(values, tf.float64)
    sorted_values = tf.sort(values, axis=-2)
    count = int(sorted_values.shape[-2])
    positions = tf.constant(QUANTILES, tf.float64) * float(count - 1)
    lower = tf.cast(tf.floor(positions), tf.int32)
    upper = tf.cast(tf.math.ceil(positions), tf.int32)
    fraction = positions - tf.floor(positions)
    lower_values = tf.gather(sorted_values, lower, axis=-2)
    upper_values = tf.gather(sorted_values, upper, axis=-2)
    rank = len(sorted_values.shape)
    fraction_shape = [1] * (rank - 2) + [len(QUANTILES), 1]
    fraction = tf.reshape(fraction, fraction_shape)
    return lower_values + fraction * (upper_values - lower_values)


def _summary(samples: tf.Tensor) -> dict[str, tf.Tensor]:
    samples = tf.convert_to_tensor(samples, tf.float64)
    pooled = tf.reshape(samples, (-1, int(samples.shape[-1])))
    centered = pooled - tf.reduce_mean(pooled, axis=0, keepdims=True)
    # Match the archived comparator's population standard deviation summary.
    denominator = tf.cast(tf.shape(pooled)[0], tf.float64)
    covariance = tf.linalg.matmul(centered, centered, transpose_a=True) / denominator
    sd = tf.sqrt(tf.linalg.diag_part(covariance))
    correlation = covariance / (sd[:, None] * sd[None, :])
    return {
        "mean": tf.reduce_mean(pooled, axis=0),
        "sd": sd,
        "quantiles": _linear_quantiles(pooled),
        "covariance": covariance,
        "correlation": correlation,
    }


def _block_bootstrap_summary(
    samples: tf.Tensor, seed: tuple[int, int]
) -> dict[str, tf.Tensor]:
    samples = tf.convert_to_tensor(samples, tf.float64)
    draws, chains, dimension = (int(item) for item in samples.shape)
    block = max(20, int(math.sqrt(draws)))
    blocks_per_chain = int(math.ceil(draws / block))
    offsets = tf.range(block, dtype=tf.int32)
    chain_major = tf.transpose(samples, [1, 0, 2])
    means = []
    sds = []
    quantiles = []
    for batch_start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_BATCH_SIZE):
        active = min(BOOTSTRAP_BATCH_SIZE, BOOTSTRAP_REPLICATES - batch_start)
        selected_chains = tf.random.stateless_uniform(
            [active, chains],
            seed=tf.constant((seed[0] + 1, seed[1] + batch_start), tf.int32),
            minval=0,
            maxval=chains,
            dtype=tf.int32,
        )
        starts = tf.random.stateless_uniform(
            [active, chains, blocks_per_chain],
            seed=tf.constant((seed[0], seed[1] + batch_start), tf.int32),
            minval=0,
            maxval=draws,
            dtype=tf.int32,
        )
        indices = tf.math.floormod(
            starts[..., None] + offsets[None, None, None, :], draws
        )
        indices = tf.reshape(indices, [active, chains, -1])[:, :, :draws]
        selected = tf.gather(chain_major, selected_chains, axis=0)
        boot = tf.gather(selected, indices, axis=2, batch_dims=2)
        pooled = tf.reshape(boot, [active, -1, dimension])
        batch_mean = tf.reduce_mean(pooled, axis=1)
        centered = pooled - batch_mean[:, None, :]
        batch_sd = tf.sqrt(
            tf.reduce_sum(tf.square(centered), axis=1)
            / tf.cast(tf.shape(pooled)[1], tf.float64)
        )
        means.append(batch_mean)
        sds.append(batch_sd)
        quantiles.append(_linear_quantiles(pooled))
    return {
        "mean": tf.concat(means, axis=0),
        "sd": tf.concat(sds, axis=0),
        "quantiles": tf.concat(quantiles, axis=0),
        "block_length": tf.constant(block, tf.int32),
    }


def _compatibility(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    reference_summary: dict[str, tf.Tensor],
    reference_boot: dict[str, tf.Tensor],
) -> dict[str, Any]:
    candidate_summary = _summary(candidate["samples"])
    candidate_boot = _block_bootstrap_summary(
        candidate["samples"],
        (BOOTSTRAP_SEED[0], BOOTSTRAP_SEED[1] + candidate["candidate_index"]),
    )
    rows = []
    for parameter in range(6):
        reference_sd = float(reference_summary["sd"][parameter].numpy())
        checks = {}
        for statistic, scale in (("mean", MEAN_TOLERANCE_SCALE), ("sd", SD_TOLERANCE_SCALE)):
            difference = float(
                (candidate_summary[statistic][parameter] - reference_summary[statistic][parameter]).numpy()
            )
            bootstrap_difference = candidate_boot[statistic][:, parameter] - reference_boot[statistic][:, parameter]
            interval = _linear_interval(bootstrap_difference)
            tolerance = float(scale * reference_sd)
            equivalence_established = bool(
                interval[0] >= -tolerance and interval[1] <= tolerance
            )
            material_disagreement_supported = bool(
                interval[0] > tolerance or interval[1] < -tolerance
            )
            checks[statistic] = {
                "difference": difference,
                "interval_95": interval,
                "tolerance": tolerance,
                "point_estimate_within_tolerance": bool(abs(difference) <= tolerance),
                "equivalence_established": equivalence_established,
                "material_disagreement_supported": material_disagreement_supported,
                "status": (
                    "equivalence_established"
                    if equivalence_established
                    else "material_disagreement_supported"
                    if material_disagreement_supported
                    else "inconclusive"
                ),
                "passed": equivalence_established,
            }
        for quantile_index, quantile in enumerate(QUANTILES):
            difference = float(
                (
                    candidate_summary["quantiles"][quantile_index, parameter]
                    - reference_summary["quantiles"][quantile_index, parameter]
                ).numpy()
            )
            bootstrap_difference = candidate_boot["quantiles"][:, quantile_index, parameter] - reference_boot["quantiles"][:, quantile_index, parameter]
            interval = _linear_interval(bootstrap_difference)
            tolerance = float(QUANTILE_TOLERANCE_SCALE * reference_sd)
            equivalence_established = bool(
                interval[0] >= -tolerance and interval[1] <= tolerance
            )
            material_disagreement_supported = bool(
                interval[0] > tolerance or interval[1] < -tolerance
            )
            checks[f"q{int(quantile * 100):02d}"] = {
                "difference": difference,
                "interval_95": interval,
                "tolerance": tolerance,
                "point_estimate_within_tolerance": bool(abs(difference) <= tolerance),
                "equivalence_established": equivalence_established,
                "material_disagreement_supported": material_disagreement_supported,
                "status": (
                    "equivalence_established"
                    if equivalence_established
                    else "material_disagreement_supported"
                    if material_disagreement_supported
                    else "inconclusive"
                ),
                "passed": equivalence_established,
            }
        rows.append({"parameter": PARAMETER_NAMES[parameter], "checks": checks, "passed": all(item["passed"] for item in checks.values())})
    covariance_difference = candidate_summary["covariance"] - reference_summary["covariance"]
    correlation_difference = candidate_summary["correlation"] - reference_summary["correlation"]
    return {
        "candidate_index": candidate["candidate_index"],
        "candidate_id": candidate["candidate_id"],
        "L": candidate["L"],
        "step_size": candidate["step_size"],
        "draws_per_chain": candidate["draws_per_chain"],
        "archive_path": candidate["archive_path"],
        "archive_sha256": candidate["archive_sha256"],
        "summary": candidate_summary,
        "parameter_checks": rows,
        "equivalence_established": all(row["passed"] for row in rows),
        "material_disagreement_supported": any(
            item["material_disagreement_supported"]
            for row in rows
            for item in row["checks"].values()
        ),
        "all_point_estimates_within_tolerance": all(
            item["point_estimate_within_tolerance"]
            for row in rows
            for item in row["checks"].values()
        ),
        "decision": (
            "POSTERIOR_EQUIVALENCE_ESTABLISHED"
            if all(row["passed"] for row in rows)
            else "MATERIAL_POSTERIOR_DISAGREEMENT_SUPPORTED"
            if any(
                item["material_disagreement_supported"]
                for row in rows
                for item in row["checks"].values()
            )
            else "POSTERIOR_EQUIVALENCE_INCONCLUSIVE"
        ),
        "passed": all(row["passed"] for row in rows),
        "covariance_difference_frobenius": float(tf.linalg.norm(covariance_difference).numpy()),
        "correlation_difference_frobenius": float(tf.linalg.norm(correlation_difference).numpy()),
        "bootstrap_block_length": int(candidate_boot["block_length"].numpy()),
    }


def _linear_interval(values: tf.Tensor) -> tuple[float, float]:
    sorted_values = tf.sort(tf.reshape(tf.convert_to_tensor(values, tf.float64), [-1]))
    count = int(sorted_values.shape[0])
    positions = tf.constant([0.025, 0.975], tf.float64) * float(count - 1)
    lower = tf.cast(tf.floor(positions), tf.int32)
    upper = tf.cast(tf.math.ceil(positions), tf.int32)
    fraction = positions - tf.floor(positions)
    interval = tf.gather(sorted_values, lower) + fraction * (
        tf.gather(sorted_values, upper) - tf.gather(sorted_values, lower)
    )
    return tuple(float(item) for item in interval.numpy().tolist())


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, tf.Tensor):
        return value.numpy().tolist()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    output_root.mkdir(parents=True)
    started = datetime.now(timezone.utc)
    started_monotonic = time.monotonic()
    try:
        public = json.loads((ROOT / PUBLIC_RESULT).read_text())
        reference_result = json.loads((ROOT / REFERENCE_RESULT).read_text())
        reference_hashes = json.loads((ROOT / REFERENCE_HASHES).read_text())
        candidates = _candidate_samples(public)
        reference = _reference_samples(reference_result, reference_hashes)
        reference_summary = _summary(reference["samples"])
        reference_boot = _block_bootstrap_summary(
            reference["samples"], (BOOTSTRAP_SEED[0], BOOTSTRAP_SEED[1] + 100)
        )
        comparisons = [
            _compatibility(candidate, reference, reference_summary, reference_boot)
            for candidate in candidates
        ]
        for comparison, candidate in zip(comparisons, candidates):
            comparison["hmc_diagnostics"] = rank_normalized_hmc_diagnostics(
                candidate["samples"],
                parameter_names=PARAMETER_NAMES,
                thresholds=RankNormalizedHMCThresholds(
                    rhat_max=1.01, bulk_ess_min=1000.0, tail_ess_min=400.0
                ),
            )
    except Exception as exc:
        failure = {
            "schema": "bayesfilter.pp_ukf.posterior_validation.failure.v1",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "hmc_launched": False,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        }
        (output_root / "failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n", encoding="ascii"
        )
        raise
    result = {
        "schema": "bayesfilter.pp_ukf.posterior_validation.result.v1",
        "status": "completed_distributional_compatibility_screen",
        "plan": str(PLAN),
        "target_signature": TARGET_SIGNATURE,
        "target_scope": TARGET_SCOPE,
        "parameter_names": PARAMETER_NAMES,
        "reference": {key: value for key, value in reference.items() if key != "samples"},
        "reference_summary": reference_summary,
        "candidate_count": len(comparisons),
        "compatibility_pass_count": sum(bool(item["passed"]) for item in comparisons),
        "equivalence_established_count": sum(
            bool(item["equivalence_established"]) for item in comparisons
        ),
        "material_disagreement_supported_count": sum(
            bool(item["material_disagreement_supported"]) for item in comparisons
        ),
        "inconclusive_count": sum(
            not item["equivalence_established"]
            and not item["material_disagreement_supported"]
            for item in comparisons
        ),
        "ranking_performed": False,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "block_rule": "max(20, floor(sqrt(draws_per_chain))) contiguous circular blocks",
            "interval": "percentile 95% interval of candidate-reference statistic difference",
        },
        "tolerances": {
            "mean": "0.10 * reference SD",
            "sd": "0.10 * reference SD",
            "quantile": "0.15 * reference SD",
        },
        "comparisons": comparisons,
        "nonclaims": [
            "no exact posterior correctness",
            "no sampler superiority or best-candidate ranking",
            "no production or default readiness",
            "no broad PP-UKF scientific validity",
        ],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started_monotonic,
    }
    result_path = output_root / "public_result.json"
    result_path.write_text(json.dumps(_json_ready(result), indent=2, sort_keys=True) + "\n", encoding="ascii")
    manifest = {
        "schema": "bayesfilter.pp_ukf.posterior_validation.manifest.v1",
        "command": [str(Path(__file__).relative_to(ROOT)), "--output-root", str(args.output_root)],
        "plan": str(PLAN),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "tensorflow": tf.__version__,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cpu_only_postprocessing": True,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hmc_launched": False,
        "data_version": "zhao_cui_predator_prey_T20 observation sha256:dc63294b6e77913aef0c92796dd2d3c7a1721a766f976fcc392cd02a70754387",
        "random_seeds": {"bootstrap": BOOTSTRAP_SEED},
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.monotonic() - started_monotonic,
        "source_public_result": str(PUBLIC_RESULT),
        "source_public_result_sha256": _sha256(ROOT / PUBLIC_RESULT),
        "source_reference_result": str(REFERENCE_RESULT),
        "source_reference_result_sha256": _sha256(ROOT / REFERENCE_RESULT),
        "result_path": str(result_path.relative_to(ROOT)),
        "result_sha256": _sha256(result_path),
    }
    (output_root / "run_manifest.json").write_text(json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n", encoding="ascii")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
