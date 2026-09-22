"""Missingness and distinct statistical targets in controller calibration."""
import pytest

from bayesfilter.testing.inference_validation.engines.controller_stopping_report import summarize_controller_records


def test_caps_and_missing_replications_keep_the_planned_denominator():
    rows = [{"rep":0,"status":"complete","passed":False,"retained_cap":True,
             "stopped":{"available":True,"covered":True},
             "fixed":{"available":True,"covered":False}},
            {"rep":1,"status":"complete","warmup_cap":True,
             "stopped":{"available":False,"covered":False}}]
    result = summarize_controller_records(rows, 4, oracle_alpha=.05 / 6, coverage_floor=.90)
    assert result["unstarted"] == 2 and not result["complete"]
    assert result["warmup_cap_count"] == result["retained_cap_count"] == 1
    assert result["intervals"]["stopped"]["unconditional"]["count"] == 1
    assert result["intervals"]["stopped"]["unconditional"]["total"] == 4
    assert result["intervals"]["stopped"]["conditional"]["total"] == 1
    assert result["intervals"]["fixed"]["unavailable"] == 3
    assert result["posterior_checks_passed"]["count"] == 0


def test_oracle_checks_transient_expectation_not_stationary_zero():
    rows = [{"rep":i,"status":"complete","fixed_oracle":{"available":True,
              "covered":False,"covered_transient_expectation":i<95}} for i in range(100)]
    result = summarize_controller_records(rows, 100, oracle_alpha=.05 / 6, coverage_floor=.90)
    assert result["oracle"]["covered_transient_expectation"] == 95
    assert result["oracle"]["screen_passed"]
    assert result["intervals"]["stopped"]["unavailable"] == 100


@pytest.mark.parametrize("indices", [(0,0), (-1,), (3,), (True,)])
def test_duplicate_or_unplanned_observations_cannot_inflate_precision(indices):
    with pytest.raises(ValueError, match="indices"):
        summarize_controller_records([{"rep":i} for i in indices], 3,
                                     oracle_alpha=.05 / 6, coverage_floor=.90)
