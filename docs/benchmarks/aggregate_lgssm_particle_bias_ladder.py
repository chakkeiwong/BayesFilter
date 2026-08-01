#!/usr/bin/env python3
"""Aggregate independently tuned LGSSM particle scopes against Kalman."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
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

from bayesfilter.highdim.ledh_tuning_scope import (
    require_scope_match,
    scope_from_mapping,
)
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.score_diagnostics_tf import tf_score_comparison_diagnostics
from docs.benchmarks import aggregate_selected_lgssm_kalman_certification as prior


SCHEMA_VERSION = "bayesfilter.lgssm_particle_bias_ladder.v1"
CAMPAIGN_ID = "lgssm-particle-bias-ladder-20260720"
HORIZON = 50
EXPECTED = {
    2000: {
        "tuning_seeds": list(range(81900, 81916)),
        "claim_seeds": list(range(81920, 81936)),
        "controls": (20, 5),
        "microbatch_size": 4,
    },
    5000: {
        "tuning_seeds": list(range(82200, 82216)),
        "claim_seeds": list(range(82220, 82236)),
        "controls": (20, 5),
        "microbatch_size": 1,
    },
}
LABELS = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _require_scope_campaign(
    path: Path, *, num_particles: int
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign = _load(path)
    if campaign.get("status") != "SCOPE_CLAIM_PASS":
        raise ValueError(f"scope campaign is not a passing engineering claim: {path}")
    claim = campaign.get("claim", {})
    result = claim.get("result", {})
    expected = EXPECTED[num_particles]
    chunks = select_transport_chunks(num_particles)
    scope = campaign.get("tuning_scope", {})
    scope_valid = (
        scope.get("horizon") == HORIZON
        and scope.get("particle_count") == num_particles
        and scope.get("row_chunk_size") == chunks.row_chunk_size
        and scope.get("col_chunk_size") == chunks.col_chunk_size
        and scope.get("row_blocks") == chunks.row_blocks
        and scope.get("col_blocks") == chunks.col_blocks
        and scope.get("chunk_policy_id") == chunks.policy_id
        and scope.get("dtype") == "float32"
        and scope.get("tf32_enabled") is True
        and scope.get("jit_compile") is True
    )
    if not scope_valid:
        raise ValueError(f"invalid N={num_particles} tuning scope")
    selection_path = Path(campaign["selected_pair_artifact"]["path"])
    selection = _load(selection_path)
    require_scope_match(
        scope_from_mapping(scope), selection["tuning_scope"], label="selection"
    )
    require_scope_match(
        scope_from_mapping(scope), claim["tuning_scope"], label="claim"
    )
    sinkhorn_steps, balance_steps = expected["controls"]
    controls_valid = all(
        int(source.get("sinkhorn_steps", -1)) == sinkhorn_steps
        and int(source.get("balance_steps", -1)) == balance_steps
        for source in (campaign["selected_pair"], selection, claim, result)
    )
    partitions_valid = (
        campaign.get("partitions", {}).get("calibration")
        + campaign.get("partitions", {}).get("validation")
        == expected["tuning_seeds"]
        and campaign.get("partitions", {}).get("claim")
        == expected["claim_seeds"]
        and result.get("estimator_seeds") == expected["claim_seeds"]
    )
    identity = result.get("preparation_identity", {})
    device = result.get("device", {})
    graph = result.get("graph", {})
    production_valid = (
        result.get("time_steps") == HORIZON
        and result.get("num_particles") == num_particles
        and identity.get("transport_chunk_policy_id") == chunks.policy_id
        and identity.get("row_chunk_size") == chunks.row_chunk_size
        and identity.get("col_chunk_size") == chunks.col_chunk_size
        and identity.get("transport_block_grid")
        == [chunks.row_blocks, chunks.col_blocks]
        and result.get("microbatching", {}).get("seed_microbatch_size")
        == expected["microbatch_size"]
        and device.get("dtype") == "float32"
        and device.get("tf32_enabled") is True
        and device.get("jit_compile") is True
        and graph.get("python_horizon_unroll") is False
        and "StatelessWhile" in graph.get("while_operation_types", [])
    )
    binding = {
        "scope_valid": scope_valid,
        "controls_valid": controls_valid,
        "partitions_valid": partitions_valid,
        "production_valid": production_valid,
        "selection_excluded_kalman": selection.get("kalman_used") is False,
        "claim_engineering_pass": claim.get("status") == "PASS",
        "claim_hard_valid": result.get("hard_valid") is True,
    }
    binding["all_valid"] = all(binding.values())
    if not binding["all_valid"]:
        raise ValueError(f"invalid N={num_particles} selection/claim binding: {binding}")
    return campaign, result, binding


def _scope_result(
    campaign_path: Path, *, num_particles: int
) -> dict[str, Any]:
    campaign, result, binding = _require_scope_campaign(
        campaign_path, num_particles=num_particles
    )
    kalman_value, kalman_hmc_score, increments = prior._kalman_prefix(HORIZON)
    chain = prior._hmc_chain()
    node_kalman_hmc = (
        tf.convert_to_tensor(result["kalman_physical_score"], tf.float64) * chain
    )
    tf.debugging.assert_near(
        node_kalman_hmc, kalman_hmc_score, atol=2.0e-12, rtol=2.0e-12
    )
    if not math.isclose(
        float(result["kalman_value"]),
        kalman_value,
        rel_tol=2.0e-12,
        abs_tol=2.0e-12,
    ):
        raise ValueError(f"N={num_particles} Kalman target mismatch")
    values = [float(value) for value in result["per_seed_value"]]
    physical_scores = tf.convert_to_tensor(
        result["per_seed_physical_score"], tf.float64
    )
    hmc_scores = physical_scores * chain
    scales = [abs(kalman_value), *tf.abs(kalman_hmc_score).numpy().tolist()]
    relative_rows = []
    for value, score in zip(values, hmc_scores.numpy().tolist(), strict=True):
        relative_rows.append(
            [
                (value - kalman_value) / scales[0],
                *[
                    (candidate - oracle) / scale
                    for candidate, oracle, scale in zip(
                        score,
                        kalman_hmc_score.numpy().tolist(),
                        scales[1:],
                        strict=True,
                    )
                ],
            ]
        )
    intervals = {
        label: prior._interval([row[index] for row in relative_rows])
        for index, label in enumerate(LABELS)
    }
    screen = prior._screen(list(intervals.values()), bool(result["hard_valid"]))
    diagnostics = tf_score_comparison_diagnostics(
        candidate_score=hmc_scores,
        reference_score_increments=increments,
        diagonal_shrinkage=0.0,
        base_ridge=1.0,
        ridge_floor=0.0,
        ridge_scale_diagonal=tf.ones([5], tf.float64),
    )
    mean_diagnostics = tf_score_comparison_diagnostics(
        candidate_score=tf.reduce_mean(hmc_scores, axis=0),
        reference_score_increments=increments,
        diagonal_shrinkage=0.0,
        base_ridge=1.0,
        ridge_floor=0.0,
        ridge_scale_diagonal=tf.ones([5], tf.float64),
    )
    mean_value = statistics.mean(values)
    mean_hmc_score = tf.reduce_mean(hmc_scores, axis=0).numpy().tolist()
    return {
        "num_particles": num_particles,
        "screen": screen,
        "hard_valid": result["hard_valid"],
        "binding": binding,
        "controls": campaign["selected_pair"],
        "tuning_scope": campaign["tuning_scope"],
        "partitions": campaign["partitions"],
        "kalman": {
            "value": kalman_value,
            "hmc_score": kalman_hmc_score.numpy().tolist(),
            "physical_score": result["kalman_physical_score"],
            "predictive_hmc_score_increments": increments.numpy().tolist(),
            "predictive_increment_rms": tf.sqrt(
                tf.reduce_mean(tf.square(increments), axis=0)
            ).numpy().tolist(),
        },
        "candidate": {
            "mean_value": mean_value,
            "mean_hmc_score": mean_hmc_score,
            "mean_physical_score": result["aggregate_physical_score"],
            "absolute_value_error": mean_value - kalman_value,
            "absolute_hmc_score_error": [
                candidate - oracle
                for candidate, oracle in zip(
                    mean_hmc_score,
                    kalman_hmc_score.numpy().tolist(),
                    strict=True,
                )
            ],
        },
        "relative_error_intervals": intervals,
        "marginals": {
            "maximum_tv_column_error": result["maximum_tv_column_error"],
            "maximum_row_error": result["maximum_row_error"],
        },
        "opg_diagnostic_only": {
            "average_opg_eigenvalues": diagnostics.average_opg_eigenvalues.numpy().tolist(),
            "mean_rms_total_metric_error": float(
                mean_diagnostics.rms_total_metric_error.numpy()
            ),
            "mean_maximum_diagonal_standardized_error": float(
                mean_diagnostics.maximum_diagonal_standardized_error.numpy()
            ),
            "per_seed_rms_total_metric_error": diagnostics.rms_total_metric_error.numpy().tolist(),
        },
        "work": result["work"],
        "microbatching": result["microbatching"],
        "timing_seconds": result["timing_seconds"],
        "claim_wall_time_seconds": campaign["claim"]["wall_time_seconds"],
        "campaign_wall_time_seconds": campaign["wall_time_seconds"],
        "gpu_allocator_bytes": result["gpu_allocator_bytes"],
        "source_artifacts": {
            "campaign": str(campaign_path),
            "campaign_sha256": _sha256(campaign_path),
            "selected_pair": campaign["selected_pair_artifact"]["path"],
            "selected_pair_sha256": campaign["selected_pair_artifact"]["sha256"],
        },
    }


def _failed_claim_summary(path: Path) -> dict[str, Any]:
    campaign = _load(path)
    claim = campaign["claim"]
    result = claim["result"]
    failed_seeds = [
        microbatch["seeds"][0]
        for microbatch in claim["microbatches"]
        if microbatch["result"]["hard_valid"] is not True
    ]
    return {
        "status": campaign["status"],
        "selected_pair": campaign["selected_pair"],
        "claim_status": claim["status"],
        "claim_seeds": campaign["partitions"]["claim"],
        "failed_seed_ids": failed_seeds,
        "maximum_tv_column_error": result["maximum_tv_column_error"],
        "maximum_row_error": result["maximum_row_error"],
        "finite": result["finite"],
        "bitwise_replay": result["bitwise_replay"],
        "work_valid": result["work_valid"],
        "artifact": str(path),
        "artifact_sha256": _sha256(path),
        "used_for_repair_selection": False,
        "used_for_final_bias_screen": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prior-aggregate", type=Path, required=True)
    parser.add_argument("--n2000-campaign", type=Path, required=True)
    parser.add_argument("--n5000-failed-campaign", type=Path, required=True)
    parser.add_argument("--n5000-campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior_payload = _load(args.prior_aggregate.resolve())
    prior_t50 = prior_payload["horizons"]["50"]
    if prior_t50["screen"] != "screen_fail":
        raise ValueError("expected the frozen N=1024 T=50 screen-fail baseline")
    scopes = {
        "2000": _scope_result(
            args.n2000_campaign.resolve(), num_particles=2000
        ),
        "5000": _scope_result(
            args.n5000_campaign.resolve(), num_particles=5000
        ),
    }
    q1024 = prior_t50["relative_error_intervals"]["q_scale"]
    q2000 = scopes["2000"]["relative_error_intervals"]["q_scale"]
    q5000 = scopes["5000"]["relative_error_intervals"]["q_scale"]
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "aggregate_complete",
        "overall_screen": "screen_fail",
        "execution_environment": {
            "mode": "cpu_only_diagnostic_postprocessing",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_algorithm_rerun": False,
            "note": (
                "GPU devices were intentionally hidden before TensorFlow import; "
                "this artifact only validates and aggregates preserved GPU/XLA claims."
            ),
        },
        "prior_n1024_t50": {
            "screen": prior_t50["screen"],
            "controls": prior_t50["controls"],
            "q_scale_relative_error_interval": q1024,
            "value_relative_error_interval": prior_t50[
                "relative_error_intervals"
            ]["value"],
            "artifact": str(args.prior_aggregate.resolve()),
            "artifact_sha256": _sha256(args.prior_aggregate.resolve()),
        },
        "scopes": scopes,
        "failed_n5000_claim_repair_evidence": _failed_claim_summary(
            args.n5000_failed_campaign.resolve()
        ),
        "particle_scaling_interpretation": {
            "q_scale_mean_relative_error": {
                "1024": q1024["mean"],
                "2000": q2000["mean"],
                "5000": q5000["mean"],
            },
            "q_scale_seed_standard_deviation": {
                "1024": q1024["standard_deviation"],
                "2000": q2000["standard_deviation"],
                "5000": q5000["standard_deviation"],
            },
            "n5000_absolute_q_scale_bias_smaller_than_n1024": abs(q5000["mean"])
            < abs(q1024["mean"]),
            "n2000_absolute_q_scale_bias_smaller_than_n1024": abs(q2000["mean"])
            < abs(q1024["mean"]),
            "monotone_particle_count_improvement": False,
            "statistically_supported_ranking": False,
            "interpretation": (
                "N=5000 is descriptively much closer to Kalman for q_scale and "
                "has lower seed dispersion, but N=2000 worsened and neither new "
                "scope passed the frozen value-and-score screen."
            ),
        },
        "analysis": {
            "familywise_level": 0.95,
            "family_size_per_scope": 6,
            "degrees_of_freedom": 15,
            "critical_value": prior.CRITICAL_VALUE,
            "value_margin": prior.VALUE_MARGIN,
            "hmc_score_margin": prior.SCORE_MARGIN,
            "opg_metrics_used_for_screen": False,
        },
        "nonclaims": [
            "not a 1/N convergence-rate estimate",
            "not a statistically supported ranking across particle counts",
            "not HMC or posterior readiness",
            "not nonlinear-model score correctness",
            "not a universal tuning setting",
            "not method superiority",
        ],
    }
    _write_exclusive(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "overall_screen": payload["overall_screen"],
                "q_scale_means": payload["particle_scaling_interpretation"][
                    "q_scale_mean_relative_error"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
