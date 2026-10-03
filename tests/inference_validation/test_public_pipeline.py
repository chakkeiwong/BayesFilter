from pathlib import Path

import numpy as np
import pytest

from bayesfilter.testing.inference_validation.procedures import execute_pipeline
from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
from bayesfilter.testing.inference_validation.storage import read_json, read_tensor, file_hash


@pytest.fixture(scope="module")
def real_member(tmp_path_factory):
    from bayesfilter.testing.inference_validation.designs import ValidationDesign, ScenarioSpec
    d=ValidationDesign(design_id="debug-prepared",engine="accuracy",scenario=ScenarioSpec("gaussian","prepared"),
        replications=1,draws=64,seed=15481,budget_seconds=120,purpose="full numerical bridge regression",
        numerical_provenance="declared Gaussian hypotheses, native acceptance; no favorable diagnostic filtering",
        device="cpu_reference",l_grid=(2,3),step_size=1.3,posterior_cap=128,
        options={"acceptance_policy":{"practical_region":(.5,.9),"repair_region":(.45,.95)},
            "search":{"pilot_enabled":False,"refinement_rounds":0,"total_budget_units":20,
                           "repair_reserve_units":3,"evidence_rungs":(1,)}})
    root=tmp_path_factory.mktemp("public-procedure")
    result=execute_pipeline(d,root)
    return d,root,result


def test_real_tuning_all_verified_members_reach_posterior_assessment(real_member):
    _,root,result=real_member
    payload=read_json(result["tuning_path"])
    assert check_inventory(payload)["finding"]=="inventory_passed"
    assert result["verified_candidate_ids"]
    assert {m["candidate_id"] for m in result["members"]}==set(result["verified_candidate_ids"])
    for m in result["members"]:
        assert m["status"]=="assessed"
        assert m["warmup_exclusion_matches"]
        assert read_tensor(m["draws_path"]).shape[0]==m["posterior"]["retained_results_per_chain"]
        assert m["posterior"]["assessment_role"]=="posterior_only"


def test_reloading_complete_pipeline_does_not_append_draws(real_member):
    d,root,result=real_member
    hashes={m["candidate_id"]:file_hash(m["draws_path"]) for m in result["members"]}
    resumed=execute_pipeline(d,root)
    assert resumed["verified_candidate_ids"]==result["verified_candidate_ids"]
    assert {m["candidate_id"]:file_hash(m["draws_path"]) for m in resumed["members"]}==hashes


def test_interval_assessment_uses_actual_controller_quantities(real_member):
    from bayesfilter.testing.inference_validation.engines.pipeline import stopped_intervals
    from bayesfilter.testing.inference_validation.catalog import get_target
    _,_,result=real_member
    member=result["members"][0]
    assessed=stopped_intervals(member,get_target("gaussian"),{},None)
    assert not assessed["sequential_coverage_established"]
    if member["posterior"]["retained_checks"]:
        assert {r["kind"] for r in assessed["quantities"]}=={"mean","quantile"}
