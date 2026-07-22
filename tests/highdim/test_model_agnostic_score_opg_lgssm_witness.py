from __future__ import annotations

import math
from pathlib import Path

import tensorflow as tf

from docs.benchmarks import emit_model_agnostic_score_opg_lgssm_witness as witness


ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT
    / "docs/benchmarks/artifacts/canonical_lgssm_balancing_kalman_repair_20260717"
    / "phase3/t2/aggregate_attempt01.json"
)


def test_kalman_predictive_score_increments_sum_to_total() -> None:
    _, prefix_scores, increments = witness._kalman_prefix_hmc_scores(2)
    tf.debugging.assert_near(
        tf.reduce_sum(increments, axis=0),
        prefix_scores[-1],
        atol=2.0e-13,
        rtol=2.0e-13,
    )


def test_witness_uses_reference_opg_and_preserves_historical_screen() -> None:
    payload = witness.build_payload(SOURCE)
    assert payload["status"] == "diagnostic_complete"
    assert payload["metric"]["construction"] == (
        "regularized_average_predictive_score_opg"
    )
    assert payload["metric"]["unregularized_numerical_rank"] <= 2
    assert payload["metric"]["rank_upper_bound"] == 2
    assert payload["uncertainty_boundary"]["particle_seed_covariance_used_in_metric"] is False
    assert payload["historical_screen_preserved"]["new_metrics_used_for_screen"] is False
    assert payload["historical_screen_preserved"]["arm_screens"] == {
        "all_active_contract_e": "inconclusive",
        "no_reset_weighted": "inconclusive",
    }
    assert payload["reference"]["increment_sum_matches_total"] is True


def test_witness_global_norm_reproduces_direct_contract_e_mean_calculation() -> None:
    payload = witness.build_payload(SOURCE)
    row = payload["arms"]["all_active_contract_e"]["mean_score_diagnostic"]
    direct = math.sqrt(sum(value * value for value in row["score_error"]))
    assert math.isclose(row["absolute_error_norm"], direct, rel_tol=1e-14)
    assert math.isclose(
        row["relative_total_score_norm_error"],
        direct / payload["reference"]["total_score_norm"],
        rel_tol=1e-14,
    )
