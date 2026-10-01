"""Independent endpoint and decision checks for the matrix campaign."""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "benchmarks"))
from run_ledh_younis_kdm_phase4b_campaign import (
    CampaignVeto,
    Engine,
    MARKS,
    power_requirement,
    preflight,
    seed_key,
    summarize,
)


def test_seed_schedule_is_disjoint_across_every_campaign_coordinate():
    keys = [
        tuple(seed_key(attempt, scope, split, replicate))
        for attempt in (1, 2, 3)
        for scope in range(3)
        for split in ("preflight", "calibration", "pilot", "validation")
        for replicate in range(500)
    ]
    assert len(keys) == len(set(keys))
    with pytest.raises(ValueError):
        seed_key(1, 0, "validation", 1000)


def test_power_respects_variance_of_the_ten_percent_margin():
    # Candidate is zero and baseline errors have squares 1,4,9. The margin
    # samples are -.9,-3.6,-8.1, whose sample variance is 13.23.
    result = power_requirement([0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 1.0)
    assert result["pilot_margin_variance"] == pytest.approx(13.23)
    expected = 20 * math.ceil(math.ceil(2.8**2 * 13.23 / 0.1**2) / 20)
    assert result["required_paths"] == expected
    assert result["executed_target_paths"] == 500
    assert result["nominal_power_adequate"] is False


def _rows(cheap_wins_in_low_group=False):
    result = []
    for i in range(100):
        exact = 0.0 if i < 50 else 10.0
        result.append(
            {
                "oracle": {"score": exact},
                "atom": {"score": exact + 1.0},
                "phase4b": {"score": exact + math.sqrt(0.8)},
                "phase4a": {"score": exact + 1.2},
                "bootstrap": {
                    "score": exact
                    + (0.1 if cheap_wins_in_low_group and i < 50 else 1.5)
                },
            }
        )
    return result


def test_conditional_heuristic_loss_vetoes_an_overall_primary_pass():
    power = {"nominal_power_adequate": True, "executed_target_paths": 100}
    result = summarize(_rows(True), 5.0, power, 42)
    assert result["primary_criterion_pass"] is True
    assert result["heuristic_dominance_verdict"] == "VETO"
    assert result["decision"] == "NO_PROMOTION"
    low = result["groups"]["low_absolute_exact_score"]["phase4b_comparisons"][
        "bootstrap"
    ]
    assert low["statistically_supported_loss"] is True


def test_underpowered_result_cannot_promote_even_with_a_negative_interval():
    power = {"nominal_power_adequate": False, "executed_target_paths": 100}
    result = summarize(_rows(), 5.0, power, 42)
    assert result["primary_criterion_pass"] is True
    assert result["decision"] == "NO_PROMOTION"
    assert "underpowered_at_500_path_cap" in result["promotion_vetoes"]


def test_campaign_calls_public_canonical_endpoint_and_consumes_status(monkeypatch):
    import bayesfilter.highdim.ledh_canonical_score_tf as canonical

    original = canonical.canonical_value_and_analytical_score
    calls = []

    def traced(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(canonical, "canonical_value_and_analytical_score", traced)
    engine = Engine(8, 2, jit_compile=False)
    path = engine.generate(tf.constant(seed_key(1, 2, "preflight", 0), tf.int32))
    engine.evaluate("atom", path)
    assert calls and calls[0]["reset_policy"] == "contract_e"
    assert calls[0]["pairwise_steps"] > 0 and calls[0]["coordinate_cap"] > 0
    monkeypatch.setattr(
        engine,
        "raw",
        lambda *args: {
            "value": tf.constant(1.0),
            "score": tf.constant(2.0),
            "valid": tf.constant(False),
        },
    )
    with pytest.raises(CampaignVeto, match="invalid"):
        engine.evaluate("atom", path)


def test_actual_matrix_fixture_total_derivatives_and_replay():
    result = preflight(Engine(8, 2, jit_compile=False))
    assert result["status"] == "PASS"
    for name in ("oracle", "atom", "phase4a", "bootstrap", *MARKS):
        assert result["methods"][name]["fd_relative_error"] < 2e-5
