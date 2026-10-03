"""Independent formula, adversary and denominator tests for endpoint diagnostics."""
import math

import pytest
from scipy import special

from bayesfilter.testing.inference_validation.engines.normal_endpoint import (
    OUTPUT_RULE, assess_endpoints, endpoint_from_pipeline, normal_reference,
)


def records(values):
    return [{"fit_id": str(i), "output_rule": OUTPUT_RULE, "status": "qualified",
             "z": z, "stream": [718, i]} for i, z in enumerate(values)]


def test_conjugate_reference_and_independent_test_formulas():
    assert normal_reference([2.]*6) == pytest.approx((1.92, .4))
    rows = records([-1., -.3, .7, 1.2])
    report = assess_endpoints(rows, [r["fit_id"] for r in rows])
    statistic = sum(r["z"] for r in rows) / 2
    square = sum(r["z"]**2 for r in rows)
    assert report["tests"]["mean"]["p_value"] == pytest.approx(math.erfc(abs(statistic)/math.sqrt(2)))
    cdf = special.gammainc(2, square/2)
    assert report["tests"]["second_moment"]["p_value"] == pytest.approx(2*min(cdf, 1-cdf))


def test_location_and_variance_adversaries_and_noop():
    # Symmetric deterministic reference quantiles are an arithmetic fixture,
    # not a simulated null-size estimate.
    z = [float(special.ndtri((i+.5)/512)) for i in range(512)]
    ids = [str(i) for i in range(512)]
    base = assess_endpoints(records(z), ids)
    assert not base["reject"]
    assert base == assess_endpoints(records([x+0 for x in z]), ids)
    for d in (.25, .5):
        assert assess_endpoints(records([x+d for x in z]), ids)["reject"]
    var = assess_endpoints(records([2*x for x in z]), ids)
    assert var["reject"] and var["tests"]["mean"]["p_value"] > .9


def test_missing_invalid_and_duplicate_records_cannot_be_detections():
    rows = records([1., 2.])
    assert assess_endpoints(rows[:1], ["0", "1"])["reject"] is None
    rows[1].update(status="unavailable", z=None)
    report = assess_endpoints(rows, ["0", "1"])
    assert report["planned"] == 2 and report["unavailable"] == 1
    with pytest.raises(ValueError, match="duplicate"):
        assess_endpoints([rows[0], rows[0]], ["0", "1"])
    for key, value in (("z", float("nan")), ("output_rule", "best_draw"), ("stream", None)):
        with pytest.raises(ValueError):
            assess_endpoints([{**rows[0], key: value}], ["0"])
    with pytest.raises(ValueError, match="streams"):
        assess_endpoints([{**r, "stream": [1, 2]} for r in records([1., 2.])], ["0", "1"])


def test_pipeline_output_rule_and_caps(tmp_path):
    import tensorflow as tf
    from bayesfilter.testing.inference_validation.storage import write_tensor
    path = write_tensor(tmp_path / "draws.tensor", tf.constant([[[2.]], [[2.12]]], tf.float64))
    member = {"candidate_id": "a", "status": "assessed", "draws_path": str(path),
        "warmup_exclusion_matches": True, "posterior": {"passed": True,
        "warmup_cap_hit": False, "retained_cap_hit": False, "hard_vetoes": [],
        "config": {"retained_seed": [1, 2]}}}
    pipeline = {"data": [2.]*6, "completion": "complete", "verified_candidate_ids": ["b", "a"],
        "selection": {"rule": "first_verified", "scope": "selected", "candidate_ids": ["a"]},
        "members": [member]}
    result = endpoint_from_pipeline("f", pipeline)
    assert result["z"] == pytest.approx(.5)
    for key in ("warmup_cap_hit", "retained_cap_hit"):
        member["posterior"][key] = True
        assert endpoint_from_pipeline("f", pipeline)["status"] == "unavailable"
        member["posterior"][key] = False
    pipeline["selection"]["candidate_ids"] = ["b"]
    with pytest.raises(ValueError, match="predetermined"):
        endpoint_from_pipeline("f", pipeline)
