#!/usr/bin/env python3
"""Emit a diagnostic-only average-OPG witness from the frozen LGSSM campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.score_diagnostics_tf import tf_score_comparison_diagnostics
from docs.benchmarks import aggregate_canonical_lgssm_kalman_certification as aggregate


SCHEMA_VERSION = "bayesfilter.model_agnostic_score_opg_lgssm_witness.v1"
COORDINATE_SYSTEM = "unconstrained_hmc_phi_atanh_scales_log"
DIAGONAL_SHRINKAGE = 0.0
BASE_RIDGE = 1.0
RIDGE_FLOOR = 0.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_aggregate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "aggregate_complete":
        raise ValueError("source aggregate is not complete")
    if payload.get("schema_version") != (
        "bayesfilter.canonical_lgssm_kalman_certification.v1"
    ):
        raise ValueError("unexpected source aggregate schema")
    return payload


def _load_arm(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "arm_complete":
        raise ValueError(f"source arm is not complete: {path}")
    return payload["result"]


def _resolve_source(source: str, aggregate_path: Path) -> Path:
    candidate = Path(source)
    if candidate.is_absolute():
        return candidate
    repo_candidate = ROOT / candidate
    if repo_candidate.exists():
        return repo_candidate.resolve()
    return (aggregate_path.parent / candidate).resolve()


def _kalman_prefix_hmc_scores(
    time_steps: int,
) -> tuple[list[float], tf.Tensor, tf.Tensor]:
    if time_steps <= 0:
        raise ValueError("time_steps must be positive")
    prefix_values: list[float] = []
    prefix_scores = []
    for prefix in range(1, time_steps + 1):
        value, score = aggregate._kalman(prefix)
        prefix_values.append(value)
        prefix_scores.append(tf.convert_to_tensor(score, tf.float64))
    stacked = tf.stack(prefix_scores, axis=0)
    increments = tf.concat([stacked[:1], stacked[1:] - stacked[:-1]], axis=0)
    return prefix_values, stacked, increments


def _candidate_metrics(
    candidate_scores: tf.Tensor, reference_increments: tf.Tensor
) -> tuple[dict[str, Any], dict[str, Any]]:
    ridge_scale = tf.ones([tf.shape(reference_increments)[1]], tf.float64)
    per_seed = tf_score_comparison_diagnostics(
        candidate_score=candidate_scores,
        reference_score_increments=reference_increments,
        diagonal_shrinkage=DIAGONAL_SHRINKAGE,
        base_ridge=BASE_RIDGE,
        ridge_floor=RIDGE_FLOOR,
        ridge_scale_diagonal=ridge_scale,
    )
    mean_score = tf.reduce_mean(candidate_scores, axis=0)
    mean = tf_score_comparison_diagnostics(
        candidate_score=mean_score,
        reference_score_increments=reference_increments,
        diagonal_shrinkage=DIAGONAL_SHRINKAGE,
        base_ridge=BASE_RIDGE,
        ridge_floor=RIDGE_FLOOR,
        ridge_scale_diagonal=ridge_scale,
    )
    per_seed_payload = {
        "score_error": per_seed.score_error.numpy().tolist(),
        "absolute_error_norm": per_seed.absolute_error_norm.numpy().tolist(),
        "relative_total_score_norm_error": (
            per_seed.relative_total_score_norm_error.numpy().tolist()
        ),
        "relative_increment_energy_error": (
            per_seed.relative_increment_energy_error.numpy().tolist()
        ),
        "rms_total_metric_error": per_seed.rms_total_metric_error.numpy().tolist(),
        "maximum_diagonal_standardized_error": (
            per_seed.maximum_diagonal_standardized_error.numpy().tolist()
        ),
    }
    mean_payload = {
        "candidate_score": mean_score.numpy().tolist(),
        "score_error": mean.score_error.numpy().tolist(),
        "absolute_error_norm": float(mean.absolute_error_norm.numpy()),
        "relative_total_score_norm_error": float(
            mean.relative_total_score_norm_error.numpy()
        ),
        "relative_increment_energy_error": float(
            mean.relative_increment_energy_error.numpy()
        ),
        "rms_total_metric_error": float(mean.rms_total_metric_error.numpy()),
        "maximum_diagonal_standardized_error": float(
            mean.maximum_diagonal_standardized_error.numpy()
        ),
    }
    return per_seed_payload, mean_payload


def build_payload(aggregate_path: Path) -> dict[str, Any]:
    aggregate_path = aggregate_path.resolve()
    source = _load_aggregate(aggregate_path)
    time_steps = int(source["time_steps"])
    source_paths = {
        name: _resolve_source(path, aggregate_path)
        for name, path in source["source_arms"].items()
    }
    arms = {name: _load_arm(path) for name, path in source_paths.items()}
    if set(arms) != {"contract_e", "no_reset"}:
        raise ValueError(f"unexpected source arms: {sorted(arms)}")
    expected_seeds = source["estimator_seeds"]
    for name, arm in arms.items():
        if int(arm["time_steps"]) != time_steps:
            raise ValueError(f"{name} time_steps differ from aggregate")
        if arm["estimator_seeds"] != expected_seeds:
            raise ValueError(f"{name} estimator seeds differ from aggregate")

    prefix_values, prefix_scores, increments = _kalman_prefix_hmc_scores(time_steps)
    source_oracle = tf.convert_to_tensor(source["kalman"]["hmc_score"], tf.float64)
    tf.debugging.assert_near(
        prefix_scores[-1], source_oracle, atol=2.0e-13, rtol=2.0e-13
    )
    tf.debugging.assert_near(
        tf.reduce_sum(increments, axis=0), source_oracle, atol=2.0e-13, rtol=2.0e-13
    )

    common = tf_score_comparison_diagnostics(
        candidate_score=source_oracle,
        reference_score_increments=increments,
        diagonal_shrinkage=DIAGONAL_SHRINKAGE,
        base_ridge=BASE_RIDGE,
        ridge_floor=RIDGE_FLOOR,
        ridge_scale_diagonal=tf.ones([tf.shape(increments)[1]], tf.float64),
    )
    arm_payloads = {}
    for output_name, source_name in (
        ("all_active_contract_e", "contract_e"),
        ("no_reset_weighted", "no_reset"),
    ):
        scores = tf.convert_to_tensor(
            arms[source_name]["per_seed_hmc_score"], tf.float64
        )
        per_seed, mean = _candidate_metrics(scores, increments)
        arm_payloads[output_name] = {
            "per_seed": per_seed,
            "mean_score_diagnostic": mean,
        }

    average_rank = int(tf.linalg.matrix_rank(common.average_opg).numpy())
    return {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_complete",
        "scientific_status": "descriptive_only_no_acceptance_threshold",
        "time_steps": time_steps,
        "parameter_count": int(increments.shape[1]),
        "parameter_names": list(aggregate.PARAMETER_NAMES),
        "coordinate_system": COORDINATE_SYSTEM,
        "reference": {
            "source": "exact_differentiated_kalman_prefix_likelihood",
            "prefix_log_likelihood": prefix_values,
            "prefix_hmc_score": prefix_scores.numpy().tolist(),
            "predictive_hmc_score_increments": increments.numpy().tolist(),
            "total_hmc_score": source_oracle.numpy().tolist(),
            "increment_sum_matches_total": True,
            "total_score_norm": float(common.reference_score_norm.numpy()),
            "increment_energy": float(common.increment_energy.numpy()),
        },
        "metric": {
            "construction": "regularized_average_predictive_score_opg",
            "diagonal_shrinkage": DIAGONAL_SHRINKAGE,
            "base_ridge": BASE_RIDGE,
            "ridge_floor": RIDGE_FLOOR,
            "ridge_scale_diagonal": common.ridge_scale_diagonal.numpy().tolist(),
            "settings_status": "diagnostic_convenience_not_scientific_defaults",
            "average_opg": common.average_opg.numpy().tolist(),
            "average_opg_eigenvalues": common.average_opg_eigenvalues.numpy().tolist(),
            "unregularized_numerical_rank": average_rank,
            "rank_upper_bound": min(time_steps, int(increments.shape[1])),
            "shrunk_average_opg": common.shrunk_average_opg.numpy().tolist(),
            "realized_ridge": float(common.realized_ridge.numpy()),
            "ridge_floor_active": bool(common.ridge_floor_active.numpy()),
            "average_metric": common.average_metric.numpy().tolist(),
            "total_metric": common.total_metric.numpy().tolist(),
            "total_metric_eigenvalues": common.total_metric_eigenvalues.numpy().tolist(),
            "total_metric_condition_proxy": float(
                common.total_metric_condition_proxy.numpy()
            ),
        },
        "arms": arm_payloads,
        "historical_screen_preserved": {
            "arm_screens": source["arm_screens"],
            "global_reset_direction": source["global_reset_direction"],
            "new_metrics_used_for_screen": False,
        },
        "source_artifacts": {
            "aggregate": str(aggregate_path),
            "aggregate_sha256": _sha256(aggregate_path),
            "arms": {name: str(path) for name, path in source_paths.items()},
            "arm_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        },
        "uncertainty_boundary": {
            "particle_seed_variation_role": "monte_carlo_uncertainty_only",
            "particle_seed_covariance_used_in_metric": False,
        },
        "nonclaims": [
            "not Kalman equivalence",
            "not nonlinear-model score correctness",
            "not HMC readiness",
            "not leaderboard admission",
            "not an optimal shrinkage or ridge selection",
            "not a statistically supported method ranking",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = build_payload(args.aggregate)
    _write_exclusive(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": payload["status"],
                "rank": payload["metric"]["unregularized_numerical_rank"],
            }
        )
    )


if __name__ == "__main__":
    main()
