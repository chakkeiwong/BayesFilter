from __future__ import annotations

from docs.benchmarks import aggregate_lgssm_n10000_single_seed_kalman as aggregate


def test_comparison_reports_absolute_error_improvement() -> None:
    result = aggregate._comparison(candidate=9.5, baseline=8.0, oracle=10.0)
    assert result["n10000_descriptively_closer"] is True
    assert result["absolute_error_change_n10000_minus_n5000"] == -1.5
    assert result["n10000_relative_error"] == -0.05


def test_relative_error_is_undefined_for_zero_oracle() -> None:
    assert aggregate._relative_error(1.0, 0.0) is None

