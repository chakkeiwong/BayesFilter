"""Public mixture call chain for global quantities; failures remain informative."""
import pytest

from bayesfilter.testing.inference_validation.designs import ScenarioSpec
from bayesfilter.testing.inference_validation.procedures import execute_pipeline


@pytest.mark.parametrize("start", ["single_mode", "mode_dispersed"])
def test_public_posterior_actually_evaluates_mode_indicator(design, tmp_path, start):
    d = design("stopping", "mixture", "prepared", replications=1, draws=64,
        scenario=ScenarioSpec("mixture", "prepared", start=start),
        step_size=1.3, l_grid=(2, 3), posterior_cap=128,
        options={"global_quantities": ["left_mode_probability"],
            "posterior_members": "selected", "member_rule": "first_verified",
            "acceptance_policy": {"practical_region": (.41, .99), "repair_region": (.405, .995)},
            "search": {"pilot_enabled": False, "refinement_rounds": 0,
                       "total_budget_units": 24, "repair_reserve_units": 4, "evidence_rungs": (1,)}})
    result = execute_pipeline(d, tmp_path)
    assert result["verified_candidate_ids"]
    member = next(m for m in result["members"] if m["status"] == "assessed")
    posterior = member["posterior"]
    assessments = [c["modern_rhat"]["assessment"] for c in posterior["warmup_checks"]]
    assessments += [c["assessment"] for c in posterior["retained_checks"]
                    if c.get("diagnostic_role") == "assessment"]
    assert assessments
    assert all("left_mode_probability" in a["quantity_names"] for a in assessments)
    assert member["warmup_exclusion_matches"]
    assert posterior["assessment_role"] == "posterior_only"
