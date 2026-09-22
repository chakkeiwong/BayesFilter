"""Public multi-model tuning -> reload -> cumulative posterior integration.

Tiny CPU reference allocations and deliberately broad acceptance screens test
pipeline mechanics, excluded warmup and cap reporting, not calibration.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
from bayesfilter.testing.inference_validation.designs import ValidationDesign, ScenarioSpec
from bayesfilter.testing.inference_validation.procedures import execute_pipeline
from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
from bayesfilter.testing.inference_validation.storage import read_json, read_tensor, file_hash


@pytest.mark.parametrize("target,epsilon", [("gaussian",1.3),("beta_binomial",.8),
    ("lgssm_location",.7),("banana",.5),("funnel_noncentered",1.2)])
def test_public_multimodel_candidates_reload_and_exclude_warmup(tmp_path,target,epsilon):
    design=ValidationDesign(design_id="integration-"+target,engine="accuracy",
        scenario=ScenarioSpec(target,"prepared"),replications=1,draws=64,seed=2026092081,
        budget_seconds=180,purpose="complete public integration mechanics, no posterior accuracy claim",
        numerical_provenance="M13 engineering fixture; wide acceptance screen and tiny posterior cap",
        device="cpu_reference",l_grid=(2,3),step_size=epsilon,posterior_cap=128,
        options={"acceptance_policy":{"practical_region":(.41,.99),"repair_region":(.405,.995)},
                 "search":{"pilot_enabled":False,"refinement_rounds":0,"total_budget_units":24,
                           "repair_reserve_units":4,"evidence_rungs":(1,)}})
    result=execute_pipeline(design,tmp_path)
    inventory=check_inventory(read_json(result["tuning_path"]))
    assert inventory["finding"]=="inventory_passed"
    assert result["verified_candidate_ids"], result
    assert {m["candidate_id"] for m in result["members"]}==set(result["verified_candidate_ids"])
    for member in result["members"]:
        assert member["status"]=="assessed"
        assert member["warmup_exclusion_matches"]
        assert read_tensor(member["draws_path"]).shape[0]==member["posterior"]["retained_results_per_chain"]
        assert member["posterior"]["assessment_role"]=="posterior_only"
    before={m["candidate_id"]:file_hash(m["draws_path"]) for m in result["members"]}
    resumed=execute_pipeline(design,tmp_path)
    assert before=={m["candidate_id"]:file_hash(m["draws_path"]) for m in resumed["members"]}
