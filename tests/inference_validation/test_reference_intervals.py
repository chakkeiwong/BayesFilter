"""Numerical-reference agreement cannot become exact or sequential coverage."""
from copy import deepcopy
import math

import pytest

from bayesfilter.testing.inference_validation.catalog import get_target
from bayesfilter.testing.inference_validation.engines.pipeline import stopped_intervals
from bayesfilter.testing.inference_validation.references import analytic
from bayesfilter.testing.inference_validation.references.intervals import numerical_interval_agreement
from bayesfilter.testing.inference_validation.references.intervals import report_saved_intervals
from bayesfilter.testing.inference_validation.storage import write_json, file_hash


def metadata():
    return {"checked": True, "method": "independent_same_target_trapezoidal_grid",
        "summary_order": ["mean0", "mean1", "median0", "median1"],
        "summary": [.2, .3, .18, .28], "sensitivity": [.001]*4,
        "sensitivity_limits": [.002]*4, "edge_mass": 1.e-8,
        "expanded_edge_mass": 1.e-9, "edge_limit": 1.e-5}


def estimate(value=.2, se=.01, **kwargs):
    return {"name": "x", "kind": "mean", "estimate": value, "mcse": se,
            "valid": True, **kwargs}


@pytest.mark.parametrize("value,se,finding", [
    (.2, .01, "contains_sensitivity_interval"),
    (.4, .01, "disjoint_from_sensitivity_interval"),
    (.22, .01, "overlaps_sensitivity_interval"),
    (.2, 0., "overlaps_sensitivity_interval"),
])
def test_explicit_containment_disjoint_and_ambiguous_outcomes(value, se, finding):
    result = numerical_interval_agreement(estimate(value, se), ["x", "y"], metadata())
    assert result["available"] and result["finding"] == finding
    assert not result["exact_coverage_eligible"] and not result["integration_error_bound"]


@pytest.mark.parametrize("change", [
    {"checked": False}, {"checked": 1}, {"method": "unverified_draws"},
    {"summary_order": None}, {"summary_order": ["mean0", "mean0"]},
    {"summary": []}, {"summary": [float("nan")]*4},
    {"sensitivity": [-.1]*4}, {"sensitivity": [.003]*4},
    {"sensitivity_limits": [None]*4}, {"edge_mass": .1},
    {"edge_limit": float("inf")}, {"expanded_edge_mass": -1.},
])
def test_bad_reference_fails_closed(change):
    assert not numerical_interval_agreement(estimate(), ["x", "y"], {**metadata(), **change})["available"]


@pytest.mark.parametrize("change", [
    {"mcse": None}, {"mcse": -.1}, {"mcse": float("inf")},
    {"estimate": float("nan")}, {"valid": False}, {"kind": "variance"},
    {"kind": "quantile", "probability": .9}, {"kind": "quantile"},
    {"estimate": 1.e308, "mcse": 1.e308},
])
def test_bad_or_unsupported_quantity_fails_closed(change):
    assert not numerical_interval_agreement(estimate(**change), ["x", "y"], metadata())["available"]


def member(estimates):
    return {"posterior": {"retained_checks": [{"modern_rhat": {"precision": {"targets": estimates}}}],
        "passed": True, "warmup_cap_hit": False, "retained_cap_hit": False}}


def test_exact_gaussian_ssm_coverage_remains_exact():
    target, data = get_target("ssm_campaign_location"), [.1]*32
    truth = analytic.exact_functionals(target.target_id, {}, data)[("mean", 0)]
    result = stopped_intervals(member([estimate(truth, name="location")]), target, {}, data,
                               reference_metadata=metadata())
    row = result["quantities"][0]
    assert row["available"] and row["covered"] and "numerical_reference" not in row
    assert not result["sequential_coverage_established"]


@pytest.mark.parametrize("target", ["ssm_campaign_two_noises", "ssm_campaign_nonlinear"])
def test_public_assessment_reports_grid_uncertainty_separately(target):
    spec = get_target(target)
    m = member([estimate(.18, name=spec.parameters[0], kind="quantile", probability=.5)])
    original = deepcopy(m)
    result = stopped_intervals(m, spec, {}, [], reference_metadata=metadata())
    row = result["quantities"][0]
    assert not row["available"] and not row["covered"]
    assert row["numerical_reference"]["reference"] == .18
    assert row["numerical_reference"]["finding"] == "contains_sensitivity_interval"
    assert original == m


def test_multivariate_ssm_without_joint_oracle_remains_unavailable():
    spec = get_target("ssm_campaign_multivariate")
    result = stopped_intervals(member([estimate(name=spec.parameters[0])]), spec, {}, [],
        reference_metadata={"checked": False, "reason": "no joint posterior reference for K6"})
    row = result["quantities"][0]
    assert not row["available"] and not row["numerical_reference"]["available"]


def test_saved_report_preserves_original_assessment_and_unassessed_siblings(tmp_path, design):
    d = design("accuracy", "ssm_campaign_two_noises", "ordinary", replications=1)
    cell, output = tmp_path / "cell", tmp_path / "report.json"
    write_json(cell / "isolated_design.json", d.payload())
    spec = get_target(d.scenario.target)
    assessment = write_json(cell / "replication-0000/independent_assessment.json", {"members": [
        {"candidate_id": "chosen", "status": "assessed",
         "assessment": {"reference_metadata": metadata()},
         "member_record": member([estimate(name=spec.parameters[0])])},
        {"candidate_id": "sibling", "status": "unassessed_by_design"}]})
    before = file_hash(assessment)
    result = report_saved_intervals(cell, output)
    assert file_hash(assessment) == before and not result["new_sampling"]
    assert result["members"][1]["status"] == "unassessed_by_design"
    assert result["members"][0]["stopped_intervals"]["quantities"][0]["numerical_reference"]["available"]
    with pytest.raises(ValueError, match="already exists"):
        report_saved_intervals(cell, output)
