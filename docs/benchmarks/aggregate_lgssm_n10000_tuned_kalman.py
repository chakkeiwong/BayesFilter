#!/usr/bin/env python3
"""Validate and aggregate the tuned N=10000 LGSSM claim against Kalman."""

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


SCHEMA_VERSION = "bayesfilter.lgssm_n10000_tuned_kalman.v1"
CAMPAIGN_ID = "lgssm-n10000-tuned-kalman-20260720"
HORIZON = 50
NUM_PARTICLES = 10_000
TUNING_SEEDS = list(range(82400, 82416))
CLAIM_SEEDS = list(range(82420, 82436))
SINKHORN_CANDIDATES = [20, 25, 30, 40]
BALANCE_CANDIDATES = [5, 8, 12, 16, 25, 32]
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


def _resolve_artifact(path_text: str, *, campaign_path: Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    repository_path = (ROOT / path).resolve()
    if repository_path.exists():
        return repository_path
    return (campaign_path.parent / path).resolve()


def _first_passing_pair(campaign: dict[str, Any]) -> tuple[int, int] | None:
    for record in campaign.get("tuning", []):
        if record.get("pair_pass") is True:
            return int(record["sinkhorn_steps"]), int(record["balance_steps"])
    return None


def _expected_candidate_order() -> list[tuple[int, int]]:
    return [
        (sinkhorn_steps, balance_steps)
        for sinkhorn_steps in SINKHORN_CANDIDATES
        for balance_steps in BALANCE_CANDIDATES
    ]


def _tuning_history_valid(campaign: dict[str, Any]) -> bool:
    tuning = campaign.get("tuning", [])
    recorded_pairs = [
        (int(record["sinkhorn_steps"]), int(record["balance_steps"]))
        for record in tuning
    ]
    expected_prefix = _expected_candidate_order()[: len(recorded_pairs)]
    if not recorded_pairs or recorded_pairs != expected_prefix:
        return False
    if any(record.get("pair_pass") is True for record in tuning[:-1]):
        return False
    if tuning[-1].get("pair_pass") is not True:
        return False
    for record in tuning:
        node = record.get("tuning_node", {})
        result = node.get("result", {})
        if node.get("seeds") != TUNING_SEEDS:
            return False
        if result.get("estimator_seeds") != TUNING_SEEDS:
            return False
        if result.get("kalman_value") is not None:
            return False
        if result.get("kalman_physical_score") is not None:
            return False
        if result.get("replay_checked") is not False:
            return False
    return True


def _require_campaign(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    campaign = _load(path)
    if campaign.get("status") != "SCOPE_CLAIM_PASS":
        raise ValueError("N=10000 campaign is not a passing engineering claim")
    if campaign.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("N=10000 campaign identity mismatch")

    chunks = select_transport_chunks(NUM_PARTICLES)
    scope = campaign.get("tuning_scope", {})
    scope_valid = (
        scope.get("horizon") == HORIZON
        and scope.get("particle_count") == NUM_PARTICLES
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
        raise ValueError("invalid N=10000 tuning scope")

    expected_partitions = {
        "calibration": TUNING_SEEDS[:8],
        "validation": TUNING_SEEDS[8:],
        "claim": CLAIM_SEEDS,
    }
    partitions_valid = campaign.get("partitions") == expected_partitions
    search_order_valid = campaign.get("search_order", {}) == {
        "sinkhorn_steps": SINKHORN_CANDIDATES,
        "balance_steps": BALANCE_CANDIDATES,
        "rule": "exhaust_balance_ladder_before_next_sinkhorn",
    }

    selected = campaign.get("selected_pair")
    if not isinstance(selected, dict):
        raise ValueError("campaign omitted selected controls")
    selected_pair = (
        int(selected["sinkhorn_steps"]), int(selected["balance_steps"])
    )
    selection_is_first_pass = _first_passing_pair(campaign) == selected_pair
    tuning_history_valid = _tuning_history_valid(campaign)

    selected_record = campaign.get("selected_pair_artifact", {})
    selection_path = _resolve_artifact(
        selected_record.get("path", ""), campaign_path=path
    )
    if not selection_path.is_file():
        raise ValueError("selected-control artifact is missing")
    selection_hash_valid = _sha256(selection_path) == selected_record.get("sha256")
    selection = _load(selection_path)
    require_scope_match(
        scope_from_mapping(scope), selection["tuning_scope"], label="selection"
    )

    candidate_path = _resolve_artifact(
        selection.get("candidate_artifact", ""), campaign_path=path
    )
    candidate_hash_valid = (
        candidate_path.is_file()
        and _sha256(candidate_path) == selection.get("candidate_artifact_sha256")
    )

    claim = campaign.get("claim", {})
    result = claim.get("result", {})
    require_scope_match(
        scope_from_mapping(scope), claim["tuning_scope"], label="claim"
    )
    controls_valid = all(
        int(source.get("sinkhorn_steps", -1)) == selected_pair[0]
        and int(source.get("balance_steps", -1)) == selected_pair[1]
        for source in (selection, claim, result)
    )

    identity = result.get("preparation_identity", {})
    device = result.get("device", {})
    graph = result.get("graph", {})
    production_valid = (
        result.get("time_steps") == HORIZON
        and result.get("num_particles") == NUM_PARTICLES
        and result.get("estimator_seeds") == CLAIM_SEEDS
        and result.get("microbatching", {}).get("seed_microbatch_size") == 1
        and identity.get("transport_chunk_policy_id") == chunks.policy_id
        and identity.get("row_chunk_size") == chunks.row_chunk_size
        and identity.get("col_chunk_size") == chunks.col_chunk_size
        and identity.get("transport_block_grid")
        == [chunks.row_blocks, chunks.col_blocks]
        and device.get("dtype") == "float32"
        and device.get("tf32_enabled") is True
        and device.get("jit_compile") is True
        and graph.get("python_horizon_unroll") is False
        and "StatelessWhile" in graph.get("while_operation_types", [])
    )
    direct_claim_valid = (
        claim.get("status") == "PASS"
        and result.get("hard_valid") is True
        and result.get("finite") is True
        and result.get("bitwise_replay") is True
        and result.get("work_valid") is True
        and result.get("maximum_tv_column_error", math.inf) <= 1.0e-4
        and result.get("maximum_row_error", math.inf) <= 1.0e-2
    )
    claim_path = path.parent / f"t{HORIZON}_fresh_claim_s16.json"
    standalone_claim_valid = claim_path.is_file() and _load(claim_path) == claim
    manifest_path = path.parent / "run_manifest.json"
    manifest = _load(manifest_path) if manifest_path.is_file() else {}
    manifest_valid = (
        manifest.get("campaign_id") == CAMPAIGN_ID
        and manifest.get("status") == "SCOPE_CLAIM_PASS"
        and manifest.get("result_sha256") == _sha256(path)
        and Path(manifest.get("result_path", "")).resolve() == path
        and manifest.get("seeds") == expected_partitions
    )
    binding = {
        "scope_valid": scope_valid,
        "partitions_valid": partitions_valid,
        "search_order_valid": search_order_valid,
        "tuning_history_valid": tuning_history_valid,
        "selection_is_first_blind_direct_gate_pass": selection_is_first_pass,
        "selection_excluded_kalman": selection.get("kalman_used") is False,
        "selection_hash_valid": selection_hash_valid,
        "candidate_hash_valid": candidate_hash_valid,
        "controls_valid": controls_valid,
        "production_valid": production_valid,
        "direct_claim_valid": direct_claim_valid,
        "standalone_claim_matches_campaign": standalone_claim_valid,
        "run_manifest_valid": manifest_valid,
    }
    binding["all_valid"] = all(binding.values())
    if not binding["all_valid"]:
        raise ValueError(f"invalid N=10000 campaign binding: {binding}")
    return campaign, result, binding


def _scope_result(path: Path) -> dict[str, Any]:
    campaign, result, binding = _require_campaign(path)
    kalman_value, kalman_hmc_score, increments = prior._kalman_prefix(HORIZON)
    chain = prior._hmc_chain()
    stored_kalman_hmc = (
        tf.convert_to_tensor(result["kalman_physical_score"], tf.float64) * chain
    )
    tf.debugging.assert_near(
        stored_kalman_hmc, kalman_hmc_score, atol=2.0e-12, rtol=2.0e-12
    )
    if not math.isclose(
        float(result["kalman_value"]),
        kalman_value,
        rel_tol=2.0e-12,
        abs_tol=2.0e-12,
    ):
        raise ValueError("N=10000 Kalman target mismatch")

    values = [float(value) for value in result["per_seed_value"]]
    physical_scores = tf.convert_to_tensor(
        result["per_seed_physical_score"], tf.float64
    )
    hmc_scores = physical_scores * chain
    kalman_scores = kalman_hmc_score.numpy().tolist()
    scales = [abs(kalman_value), *tf.abs(kalman_hmc_score).numpy().tolist()]
    relative_rows = [
        [
            (value - kalman_value) / scales[0],
            *[
                (candidate - oracle) / scale
                for candidate, oracle, scale in zip(
                    score, kalman_scores, scales[1:], strict=True
                )
            ],
        ]
        for value, score in zip(values, hmc_scores.numpy().tolist(), strict=True)
    ]
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
    mean_hmc_score = tf.reduce_mean(hmc_scores, axis=0).numpy().tolist()
    return {
        "num_particles": NUM_PARTICLES,
        "screen": screen,
        "hard_valid": result["hard_valid"],
        "binding": binding,
        "controls": campaign["selected_pair"],
        "tuning_scope": campaign["tuning_scope"],
        "partitions": campaign["partitions"],
        "kalman": {
            "value": kalman_value,
            "hmc_score": kalman_scores,
            "physical_score": result["kalman_physical_score"],
        },
        "candidate": {
            "mean_value": statistics.mean(values),
            "mean_hmc_score": mean_hmc_score,
            "mean_physical_score": result["aggregate_physical_score"],
            "absolute_value_error": statistics.mean(values) - kalman_value,
            "absolute_hmc_score_error": [
                candidate - oracle
                for candidate, oracle in zip(
                    mean_hmc_score, kalman_scores, strict=True
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
        },
        "work": result["work"],
        "microbatching": result["microbatching"],
        "timing_seconds": result["timing_seconds"],
        "claim_wall_time_seconds": campaign["claim"]["wall_time_seconds"],
        "campaign_wall_time_seconds": campaign["wall_time_seconds"],
        "gpu_allocator_bytes": result["gpu_allocator_bytes"],
        "source_artifacts": {
            "campaign": str(path),
            "campaign_sha256": _sha256(path),
            "selected_pair": campaign["selected_pair_artifact"]["path"],
            "selected_pair_sha256": campaign["selected_pair_artifact"]["sha256"],
        },
    }


def _require_n5000_baseline(path: Path) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("schema_version") != "bayesfilter.lgssm_particle_bias_ladder.v1":
        raise ValueError("unexpected N=5000 baseline schema")
    scope = payload.get("scopes", {}).get("5000", {})
    if scope.get("num_particles") != 5000 or scope.get("binding", {}).get(
        "all_valid"
    ) is not True:
        raise ValueError("N=5000 comparator is not an exact bound scope")
    return scope


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--n5000-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaign_path = args.campaign.resolve()
    baseline_path = args.n5000_aggregate.resolve()
    result = _scope_result(campaign_path)
    n5000 = _require_n5000_baseline(baseline_path)
    comparisons = {}
    for label in LABELS:
        n5000_interval = n5000["relative_error_intervals"][label]
        n10000_interval = result["relative_error_intervals"][label]
        comparisons[label] = {
            "n5000_mean_relative_error": n5000_interval["mean"],
            "n10000_mean_relative_error": n10000_interval["mean"],
            "n5000_standard_deviation": n5000_interval["standard_deviation"],
            "n10000_standard_deviation": n10000_interval["standard_deviation"],
            "n10000_absolute_mean_error_smaller_descriptively": abs(
                n10000_interval["mean"]
            )
            < abs(n5000_interval["mean"]),
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "aggregate_complete",
        "screen": result["screen"],
        "execution_environment": {
            "mode": "cpu_only_diagnostic_postprocessing",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_algorithm_rerun": False,
        },
        "n10000": result,
        "descriptive_comparison_with_n5000": comparisons,
        "cross_particle_ranking_statistically_supported": False,
        "analysis": {
            "familywise_level": 0.95,
            "family_size": 6,
            "degrees_of_freedom": 15,
            "critical_value": prior.CRITICAL_VALUE,
            "value_margin": prior.VALUE_MARGIN,
            "hmc_score_margin": prior.SCORE_MARGIN,
        },
        "source_artifacts": {
            "n5000_aggregate": str(baseline_path),
            "n5000_aggregate_sha256": _sha256(baseline_path),
        },
        "nonclaims": [
            "not a statistically supported ranking across particle counts",
            "not a 1/N convergence-rate estimate",
            "not HMC or posterior readiness",
            "not nonlinear-model score correctness",
            "not a universal tuning setting",
        ],
    }
    _write_exclusive(args.output.resolve(), payload)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "screen": result["screen"],
                "controls": result["controls"],
                "relative_error_means": {
                    label: result["relative_error_intervals"][label]["mean"]
                    for label in LABELS
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
