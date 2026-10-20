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


def test_budget_stopped_posterior_resumes_exact_committed_chunks(real_member, tmp_path, monkeypatch):
    from bayesfilter.runtime.execution_budget import execution_budget, ExecutionBudgetExceeded
    from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint
    d, _, baseline = real_member
    allowed = True
    original = DurableTensorCheckpoint.run
    def stop_after_first_chunk(self, *args, **kwargs):
        nonlocal allowed
        value = original(self, *args, **kwargs)
        allowed = False
        return value
    monkeypatch.setattr(DurableTensorCheckpoint, "run", stop_after_first_chunk)
    with execution_budget(check=lambda: allowed), pytest.raises(ExecutionBudgetExceeded):
        execute_pipeline(d, tmp_path)
    assert list(tmp_path.glob("members/*/posterior_chunks/committed/*/bundle.json"))
    assert list(tmp_path.glob("members/*/budget_pauses/pause-*.json"))
    assert not list(tmp_path.glob("members/*/result.json"))
    assert not list(tmp_path.glob("members/*/draws.tensor"))
    monkeypatch.setattr(DurableTensorCheckpoint, "run", original)
    resumed = execute_pipeline(d, tmp_path)
    assert resumed["verified_candidate_ids"] == baseline["verified_candidate_ids"]
    for expected, actual in zip(baseline["members"], resumed["members"]):
        np.testing.assert_array_equal(read_tensor(expected["draws_path"]), read_tensor(actual["draws_path"]))
        np.testing.assert_array_equal(read_tensor(expected["warmup_path"]), read_tensor(actual["warmup_path"]))
