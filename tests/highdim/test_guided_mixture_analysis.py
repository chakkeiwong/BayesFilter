"""Diagnostic-result pairing and veto tests; no numerical runtime dependency."""

import copy
import importlib.util
from pathlib import Path

import pytest


PATH = Path(__file__).resolve().parents[2] / "scripts/analyze_ledh_guided_mixture_score.py"
SPEC = importlib.util.spec_from_file_location("guided_analysis", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture_rows():
    return [{"condition": condition, "partition": partition, "arm": arm,
             "seed": index, "reference_score": [0.] * 4,
             "squared_error": [2. if arm == "fitted" else 1.] * 4,
             "value_error": .1, "evaluation_seconds": .01,
             "trace_min": {"valid": 1.}}
            for condition in range(4)
            for partition, count in (("validation", 2), ("final", 12))
            for arm in MODULE.ARMS for index in range(count)]


def test_heuristic_loss_veto_and_paired_error_sign():
    result = MODULE.analyze(fixture_rows(), resamples=10)
    assert result["heuristic_dominance_verdict"] == "VETO"
    assert result["nomination_criterion_passed"] is False
    assert len(result["heuristic_losses"]) == 16
    assert result["paired_comparisons"][0]["mean_error_difference"] == [1.] * 4
    assert result["paired_comparisons"][0]["paired_percentile_interval_95"] == [[1., 1.]] * 4


@pytest.mark.parametrize("fault", ["missing", "duplicate", "invalid", "unpaired", "reference"])
def test_invalid_evidence_rejected(fault):
    rows = fixture_rows()
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "invalid":
        rows[0]["trace_min"]["valid"] = 0.
    elif fault == "unpaired":
        rows[0]["seed"] = 100
    else:
        rows[0]["reference_score"][0] = 1.
    with pytest.raises(ValueError):
        MODULE.analyze(rows, resamples=10)
