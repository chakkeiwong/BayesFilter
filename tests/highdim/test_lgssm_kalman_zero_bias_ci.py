from __future__ import annotations

from docs.benchmarks import aggregate_lgssm_kalman_zero_bias_ci as aggregate


def test_zero_bias_interval_contains_zero_when_mean_is_small() -> None:
    scope = {
        "num_particles": 5000,
        "hard_valid": True,
        "screen": "screen_fail",
        "binding": {"all_valid": True},
        "relative_error_intervals": {
            label: {"mean": 0.0, "standard_error": 0.01} for label in aggregate.LABELS
        },
    }
    result = aggregate._scope_zero_bias(scope)
    assert result["simultaneous_zero_bias_screen"] == "not_rejected"
    assert all(
        item["simultaneous_zero_bias_not_rejected"]
        for item in result["outputs"].values()
    )


def test_zero_bias_interval_rejects_persistent_q_bias() -> None:
    scope = {
        "num_particles": 5000,
        "hard_valid": True,
        "screen": "screen_fail",
        "binding": {"all_valid": True},
        "relative_error_intervals": {
            label: {"mean": 0.0, "standard_error": 0.01} for label in aggregate.LABELS
        },
    }
    scope["relative_error_intervals"]["q_scale"] = {
        "mean": -0.10,
        "standard_error": 0.02,
    }
    result = aggregate._scope_zero_bias(scope)
    assert result["outputs"]["q_scale"]["simultaneous_zero_bias_not_rejected"] is False
