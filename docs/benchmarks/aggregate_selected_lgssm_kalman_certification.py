#!/usr/bin/env python3
"""Aggregate selected-control LGSSM nodes against the exact Kalman oracle."""

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

from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as canonical
from bayesfilter.score_diagnostics_tf import tf_score_comparison_diagnostics
from docs.benchmarks import run_canonical_lgssm_fused_ot_loop_repair as runner
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


SCHEMA_VERSION = "bayesfilter.selected_lgssm_kalman_certification.v1"
CAMPAIGN_ID = "lgssm-selected-controls-kalman-certification-20260719"
CRITICAL_VALUE = 3.036283222821165
VALUE_MARGIN = 0.001
SCORE_MARGIN = 0.05
EXPECTED = {
    10: {
        "sinkhorn_steps": 20,
        "balance_steps": 3,
        "seeds": list(range(81700, 81716)),
    },
    50: {
        "sinkhorn_steps": 20,
        "balance_steps": 8,
        "seeds": list(range(81820, 81836)),
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_node(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if payload.get("status") != "node_complete":
        raise ValueError(f"node is not complete: {path}")
    return payload["result"]


def _load_claim(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    result = payload.get("result")
    if not isinstance(result, dict) or payload.get("status") != "PASS":
        raise ValueError(f"stored selected-control claim is not PASS: {path}")
    return result


def _interval(values: list[float]) -> dict[str, float]:
    if len(values) != 16:
        raise ValueError("certification requires exactly 16 estimator seeds")
    mean = statistics.mean(values)
    stddev = statistics.stdev(values)
    standard_error = stddev / math.sqrt(len(values))
    radius = CRITICAL_VALUE * standard_error
    return {
        "mean": mean,
        "standard_deviation": stddev,
        "standard_error": standard_error,
        "critical_value": CRITICAL_VALUE,
        "lower": mean - radius,
        "upper": mean + radius,
    }


def _screen(intervals: list[dict[str, float]], hard_valid: bool) -> str:
    margins = (VALUE_MARGIN,) + (SCORE_MARGIN,) * 5
    if not hard_valid:
        return "screen_fail"
    if all(
        interval["lower"] >= -margin and interval["upper"] <= margin
        for interval, margin in zip(intervals, margins, strict=True)
    ):
        return "screen_pass"
    if any(
        interval["lower"] > margin or interval["upper"] < -margin
        for interval, margin in zip(intervals, margins, strict=True)
    ):
        return "screen_fail"
    return "inconclusive"


def _hmc_chain() -> tf.Tensor:
    theta = tf.constant(runner.THETA, tf.float64)
    return tf.concat([1.0 - tf.square(theta[:3]), theta[3:]], axis=0)


def _production_kalman(prefix: int) -> tuple[float, tf.Tensor]:
    """Differentiate Kalman on the exact float32 observation target used by LEDH."""

    observations = tf.cast(
        _lgssm_dataset(runner.DATASET_SEED)["observations"][:prefix], tf.float32
    )
    theta = tf.constant(runner.THETA, tf.float32)
    value, physical_score = runner._kalman(observations, theta, canonical)
    hmc_score = tf.convert_to_tensor(physical_score, tf.float64) * _hmc_chain()
    return value, hmc_score


def _kalman_prefix(time_steps: int) -> tuple[float, tf.Tensor, tf.Tensor]:
    values: list[float] = []
    scores: list[tf.Tensor] = []
    for prefix in range(1, time_steps + 1):
        value, score = _production_kalman(prefix)
        values.append(value)
        scores.append(score)
    stacked = tf.stack(scores, axis=0)
    increments = tf.concat([stacked[:1], stacked[1:] - stacked[:-1]], axis=0)
    tf.debugging.assert_near(
        tf.reduce_sum(increments, axis=0), stacked[-1], atol=2.0e-12, rtol=2.0e-12
    )
    return values[-1], stacked[-1], increments


def _require_binding(
    *,
    horizon: int,
    node: dict[str, Any],
    selection: dict[str, Any],
    claim: dict[str, Any],
) -> dict[str, Any]:
    expected = EXPECTED[horizon]
    controls_match = all(
        int(source[name]) == int(expected[name])
        for source in (node, selection, claim)
        for name in ("sinkhorn_steps", "balance_steps")
    )
    seeds_match = (
        node.get("estimator_seeds") == expected["seeds"]
        and claim.get("estimator_seeds") == expected["seeds"]
    )
    scope_match = (
        int(node.get("time_steps", -1)) == horizon
        and int(claim.get("time_steps", -1)) == horizon
        and int(node.get("num_particles", -1)) == 1024
        and int(claim.get("num_particles", -1)) == 1024
    )
    device = node.get("device", {})
    identity = node.get("preparation_identity", {})
    production_match = (
        device.get("dtype") == "float32"
        and device.get("tf32_enabled") is True
        and device.get("jit_compile") is True
        and identity.get("row_chunk_size") == 1024
        and identity.get("col_chunk_size") == 1024
        and node.get("graph", {}).get("python_horizon_unroll") is False
        and "StatelessWhile" in node.get("graph", {}).get("while_operation_types", [])
    )
    historical_replay = {
        "per_seed_value_exact": node.get("per_seed_value") == claim.get("per_seed_value"),
        "per_seed_physical_score_exact": (
            node.get("per_seed_physical_score")
            == claim.get("per_seed_physical_score")
        ),
    }
    all_valid = bool(
        controls_match
        and seeds_match
        and scope_match
        and production_match
        and selection.get("kalman_used") is False
    )
    return {
        "controls_match": controls_match,
        "seeds_match": seeds_match,
        "scope_match": scope_match,
        "production_match": production_match,
        "selection_excluded_kalman": selection.get("kalman_used") is False,
        "current_source_claim_revalidation_required": True,
        "historical_float32_exact_replay": historical_replay,
        "historical_float32_exact_replay_used_as_gate": False,
        "all_valid": all_valid,
    }


def _horizon_payload(
    *,
    horizon: int,
    node_path: Path,
    selection_path: Path,
    claim_path: Path,
) -> dict[str, Any]:
    node = _load_node(node_path)
    selection = _load_json(selection_path)
    claim = _load_claim(claim_path)
    binding = _require_binding(
        horizon=horizon, node=node, selection=selection, claim=claim
    )
    kalman_value, kalman_hmc_score, increments = _kalman_prefix(horizon)
    node_kalman_value = node.get("kalman_value")
    node_kalman_physical = node.get("kalman_physical_score")
    if node_kalman_value is None or node_kalman_physical is None:
        raise ValueError(f"node omitted Kalman diagnostics at T={horizon}")
    chain = _hmc_chain()
    node_kalman_hmc = tf.convert_to_tensor(node_kalman_physical, tf.float64) * chain
    tf.debugging.assert_near(
        node_kalman_hmc, kalman_hmc_score, atol=2.0e-12, rtol=2.0e-12
    )
    if not math.isclose(
        float(node_kalman_value), kalman_value, rel_tol=2.0e-12, abs_tol=2.0e-12
    ):
        raise ValueError(f"node Kalman value identity mismatch at T={horizon}")

    physical_scores = tf.convert_to_tensor(node["per_seed_physical_score"], tf.float64)
    hmc_scores = physical_scores * chain
    values = [float(value) for value in node["per_seed_value"]]
    scales = [abs(kalman_value), *tf.abs(kalman_hmc_score).numpy().tolist()]
    if any(not math.isfinite(scale) or scale == 0.0 for scale in scales):
        raise ValueError(f"invalid Kalman normalization scale at T={horizon}: {scales}")
    rows = []
    for value, score in zip(values, hmc_scores.numpy().tolist(), strict=True):
        rows.append(
            [
                (value - kalman_value) / scales[0],
                *[
                    (candidate - oracle) / scale
                    for candidate, oracle, scale in zip(
                        score, kalman_hmc_score.numpy().tolist(), scales[1:], strict=True
                    )
                ],
            ]
        )
    intervals = [_interval([row[index] for row in rows]) for index in range(6)]
    hard_valid = bool(
        node.get("hard_valid")
        and node.get("maximum_tv_column_error", float("inf")) <= 1.0e-4
        and node.get("maximum_row_error", float("inf")) <= 1.0e-2
        and binding["all_valid"]
    )

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
    labels = ("value", "phi1", "phi2", "phi3", "q_scale", "r_scale")
    return {
        "time_steps": horizon,
        "screen": _screen(intervals, hard_valid),
        "hard_valid": hard_valid,
        "binding": binding,
        "controls": {
            "sinkhorn_steps": node["sinkhorn_steps"],
            "balance_steps": node["balance_steps"],
        },
        "estimator_seeds": node["estimator_seeds"],
        "kalman": {
            "value": kalman_value,
            "hmc_score": kalman_hmc_score.numpy().tolist(),
            "physical_score": node_kalman_physical,
            "predictive_hmc_score_increments": increments.numpy().tolist(),
        },
        "candidate": {
            "mean_value": statistics.mean(values),
            "mean_hmc_score": tf.reduce_mean(hmc_scores, axis=0).numpy().tolist(),
            "mean_physical_score": node["aggregate_physical_score"],
        },
        "relative_error_intervals": dict(zip(labels, intervals, strict=True)),
        "marginals": {
            "maximum_tv_column_error": node["maximum_tv_column_error"],
            "maximum_row_error": node["maximum_row_error"],
        },
        "opg_diagnostic_only": {
            "settings": {
                "diagonal_shrinkage": 0.0,
                "base_ridge": 1.0,
                "ridge_floor": 0.0,
                "ridge_scale_diagonal": [1.0] * 5,
            },
            "average_opg_eigenvalues": diagnostics.average_opg_eigenvalues.numpy().tolist(),
            "mean_rms_total_metric_error": float(
                mean_diagnostics.rms_total_metric_error.numpy()
            ),
            "mean_maximum_diagonal_standardized_error": float(
                mean_diagnostics.maximum_diagonal_standardized_error.numpy()
            ),
            "per_seed_rms_total_metric_error": diagnostics.rms_total_metric_error.numpy().tolist(),
        },
        "timing_seconds": node["timing_seconds"],
        "gpu_allocator_bytes": node["gpu_allocator_bytes"],
        "source_artifacts": {
            "node": str(node_path),
            "node_sha256": _sha256(node_path),
            "selection": str(selection_path),
            "selection_sha256": _sha256(selection_path),
            "stored_claim": str(claim_path),
            "stored_claim_sha256": _sha256(claim_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for horizon in (10, 50):
        parser.add_argument(f"--t{horizon}-node", type=Path, required=True)
        parser.add_argument(f"--t{horizon}-selection", type=Path, required=True)
        parser.add_argument(f"--t{horizon}-claim", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    horizons = {
        str(horizon): _horizon_payload(
            horizon=horizon,
            node_path=getattr(args, f"t{horizon}_node").resolve(),
            selection_path=getattr(args, f"t{horizon}_selection").resolve(),
            claim_path=getattr(args, f"t{horizon}_claim").resolve(),
        )
        for horizon in (10, 50)
    }
    overall = (
        "screen_pass"
        if all(item["screen"] == "screen_pass" for item in horizons.values())
        else "not_closed"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "aggregate_complete",
        "overall_screen": overall,
        "horizons": horizons,
        "analysis": {
            "familywise_level": 0.95,
            "family_size_per_horizon": 6,
            "degrees_of_freedom": 15,
            "critical_value": CRITICAL_VALUE,
            "value_margin": VALUE_MARGIN,
            "hmc_score_margin": SCORE_MARGIN,
            "student_model_no_power_guarantee": True,
            "opg_metrics_used_for_screen": False,
        },
        "nonclaims": [
            "not a parameter-region certificate",
            "not HMC readiness",
            "not nonlinear-model score correctness",
            "not a universal tuning setting",
            "not leaderboard completeness",
            "not method superiority",
        ],
    }
    _write_exclusive(output, payload)
    print(json.dumps({"output": str(output), "overall_screen": overall}, indent=2))


if __name__ == "__main__":
    main()
