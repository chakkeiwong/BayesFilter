from __future__ import annotations

from docs.benchmarks import aggregate_lgssm_n10000_tuned_kalman as aggregate


def test_first_passing_pair_is_selected_in_recorded_order() -> None:
    campaign = {
        "tuning": [
            {"sinkhorn_steps": 20, "balance_steps": 5, "pair_pass": False},
            {"sinkhorn_steps": 20, "balance_steps": 8, "pair_pass": True},
            {"sinkhorn_steps": 20, "balance_steps": 12, "pair_pass": True},
        ]
    }
    assert aggregate._first_passing_pair(campaign) == (20, 8)


def test_no_passing_pair_is_reported_as_none() -> None:
    campaign = {
        "tuning": [
            {"sinkhorn_steps": 20, "balance_steps": 5, "pair_pass": False}
        ]
    }
    assert aggregate._first_passing_pair(campaign) is None


def _record(sinkhorn_steps: int, balance_steps: int, pair_pass: bool) -> dict:
    return {
        "sinkhorn_steps": sinkhorn_steps,
        "balance_steps": balance_steps,
        "pair_pass": pair_pass,
        "tuning_node": {
            "seeds": aggregate.TUNING_SEEDS,
            "result": {
                "estimator_seeds": aggregate.TUNING_SEEDS,
                "kalman_value": None,
                "kalman_physical_score": None,
                "replay_checked": False,
            },
        },
    }


def test_tuning_history_requires_blind_declared_prefix() -> None:
    campaign = {
        "tuning": [
            _record(20, 5, False),
            _record(20, 8, True),
        ]
    }
    assert aggregate._tuning_history_valid(campaign)


def test_tuning_history_rejects_reordered_or_oracle_aware_record() -> None:
    reordered = {
        "tuning": [
            _record(20, 8, False),
            _record(20, 5, True),
        ]
    }
    assert not aggregate._tuning_history_valid(reordered)
    oracle_aware = {
        "tuning": [
            _record(20, 5, False),
            _record(20, 8, True),
        ]
    }
    oracle_aware["tuning"][0]["tuning_node"]["result"]["kalman_value"] = -1.0
    assert not aggregate._tuning_history_valid(oracle_aware)
