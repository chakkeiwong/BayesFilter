"""Independent diagnostic tests for pairing and frozen factorial controls."""

import copy
import importlib.util
from pathlib import Path

import pytest

from bayesfilter.score_study.guidance_factorial_design import ARMS, POLICIES, SEED_BASE, make_factorial_arms

path = Path(__file__).resolve().parents[2] / "scripts/analyze_ledh_guidance_factorial.py"
spec = importlib.util.spec_from_file_location("factorial_analysis", path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def design():
    fixed = {"identity": [1., 0., 0.], "global": [.1, .9, 0.],
             "local": [.1, 0., .9], "equal": [1. / 3.] * 3}
    calibrated = {"current": [.1, .5, .4], "exact": [.2, .4, .4]}
    return make_factorial_arms(fixed, calibrated, {"current": object(), "exact": object()})


def rows():
    result = []
    for condition in range(4):
        for number, partition, count in ((1, "validation", 2), (2, "final", 12)):
            for index in range(count):
                seed = SEED_BASE + 1000 * condition + 100 * number + index
                for arm, beta, _ in design():
                    score = 1. if arm.startswith("future") else 2.
                    result.append({"condition": condition, "partition": partition, "seed": seed,
                                   "arm": arm, "beta": beta, "score": [score] * 4,
                                   "reference_score": [0.] * 4, "squared_error": [score ** 2] * 4,
                                   "value_error": 0., "paired_input_sha256": str(seed),
                                   "trace_min": {"valid": 1.}})
    return result


def test_crossed_arms_share_exact_beta_objects():
    arms = design()
    assert tuple(arm[0] for arm in arms) == ARMS
    keyed = {name: (beta, message) for name, beta, message in arms}
    for policy in POLICIES:
        assert keyed[f"current_{policy}"][0] is keyed[f"future_{policy}"][0]
        assert keyed[f"current_{policy}"][1] is not keyed[f"future_{policy}"][1]


def test_contrasts_and_interaction():
    result = analysis.analyze(rows(), resamples=20)
    assert result["rows"] == 616
    assert result["nomination_criterion_passed"]
    primary = [row for row in result["contrasts"] if row["role"] == "primary"]
    assert all(row["mean_error_difference"] == [-3.] * 4 for row in primary)
    interactions = [row for row in result["contrasts"] if row["contrast"] == "guidance_calibration_interaction"]
    assert all(row["mean_error_difference"] == [0.] * 4 for row in interactions)


@pytest.mark.parametrize("damage", ["missing", "duplicate", "invalid", "beta", "input", "score", "nonfinite"])
def test_rejects_corrupt_comparison(damage):
    data = copy.deepcopy(rows())
    if damage == "missing":
        data.pop()
    elif damage == "duplicate":
        data.append(data[0])
    elif damage == "invalid":
        data[0]["trace_min"]["valid"] = 0.
    elif damage == "beta":
        data[2]["beta"] = [.2, .8, 0.]
    elif damage == "input":
        data[2]["paired_input_sha256"] = "wrong"
    elif damage == "score":
        data[0]["score"][0] = 9.
    else:
        data[0]["score"][0] = float("nan")
    with pytest.raises(ValueError):
        analysis.analyze(data, resamples=20)
