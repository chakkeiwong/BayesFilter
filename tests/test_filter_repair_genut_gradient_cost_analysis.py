"""Adverse diagnostic checks for saved gradient comparisons."""

import numpy as np
import pytest

from tests.test_filter_repair_campaign import load
from tests.test_filter_repair_genut_transitive import _write


def fixture():
    analyzer = load("analyze_filter_repair_genut_gradient_cost")
    reference = {name: np.array(1., dtype=np.float64) for name in analyzer.FIELDS}
    for name in ("particles", "source_gradient", "reset_gradient", "coefficients"):
        reference[name] = np.arange(12, dtype=np.float64).reshape(4, 3) / 4
    reference["weight_gradient"] = np.array([-.03125, 7., -17., 13.])
    reference["valid"] = np.array(True)
    actual = {name: array.astype(np.float32) for name, array in reference.items()}
    return analyzer, actual, reference


@pytest.mark.parametrize("fault", ["coefficient", "shape", "nan", "missing", "invalid"])
def test_ineligible_comparison_rejected(fault):
    analyzer, actual, reference = fixture()
    assert analyzer.compare_arrays(actual, reference, 4, 3)["complete_passed"]
    if fault == "coefficient":
        actual["coefficients"][0, 0] += 1e-7  # Below ordinary tolerance, still a different objective.
    elif fault == "shape":
        actual["weight_gradient"] = actual["weight_gradient"][:, None]
    elif fault == "nan":
        actual["source_gradient"][0, 0] = np.nan
    elif fault == "missing":
        del actual["reset_gradient"]
    else:
        actual["valid"] = np.array(False)
    with pytest.raises(AssertionError):
        analyzer.compare_arrays(actual, reference, 4, 3)


def test_report_disagreement_stays_separate_and_gradient_failure_vetoes():
    analyzer, actual, reference = fixture()
    actual[analyzer.REPORT_FIELD] = np.array(.99, dtype=np.float32)
    result = analyzer.compare_arrays(actual, reference, 4, 3)
    assert result["smooth_passed"] and not result["complete_passed"]
    actual["weight_gradient"][0] += 5e-5
    result = analyzer.compare_arrays(actual, reference, 4, 3)
    assert not result["smooth_passed"]
    assert result["fields"]["weight_gradient"]["failing_elements"] == 1


def test_saved_renewed_cost_cohort(request):
    analyzer = load("analyze_filter_repair_genut_gradient_cost")
    root = analyzer.Path("/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917")
    runs = (4246, 4247, 4248, 4249, 4250, 4251, 4252, 4253, 4254, 4255, 4256, 4257, 4258, 4259)
    result = analyzer.analyze(root, runs)
    result["artifact_runs"] = runs
    _write(request, "genut-bounded-gradient-cost-analysis.json", result)
    assert len(result["rows"]) == 12
    assert not result["all_smooth_passed"]
    assert not result["complete_record_gate_passed"]
    assert all(row["numerics"]["fields"]["weight_gradient"]["passed"] is False
               for row in result["rows"])
