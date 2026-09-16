from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bayesfilter.testing.inference_validation.procedures import selected_member_ids, execute_pipeline
from bayesfilter.testing.inference_validation.engines.stopping import arm_quantities, summarize_pairs
from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications
from bayesfilter.testing.inference_validation.catalog import get_target
from bayesfilter.testing.inference_validation.storage import read_tensor, read_json


def test_selection_depends_on_tuning_id_and_keeps_scope_explicit(design):
    candidates=[SimpleNamespace(candidate_id=cid,leapfrog_steps=l) for cid,l in (("b",3),("a",5))]
    assert selected_member_ids(design(),candidates)==("b",)
    assert selected_member_ids(design(options={"member_rule":"first_verified"}),candidates)==("a",)


def test_design_rejects_threshold_override_and_undeclared_fixed_counts(design):
    with pytest.raises(ValueError,match="count controls"):
        design(options={"posterior_settings":{"warmup_rhat_max":8}})
    with pytest.raises(ValueError,match="fixed comparator"):
        design("stopping",route="prepared",options={"fixed_comparator":{"retained_results":100}})


def test_native_ordinary_search_passes_no_epsilon_override(design,tmp_path,monkeypatch):
    import bayesfilter.inference as public
    def intercept(**kwargs):
        assert kwargs["search_config"] is None
        assert kwargs["config"].mass_policy=="windowed_adaptive"
        raise RuntimeError("native dispatch checked")
    monkeypatch.setattr(public,"tune_hmc_kernel",intercept)
    d=design("sbc","normal_conjugate","ordinary",l_grid=(3,5,9,13,18,25),
             options={"native_search":True})
    with pytest.raises(RuntimeError,match="native dispatch checked"):
        execute_pipeline(d,tmp_path)
    with pytest.raises(ValueError,match="without a search override"):
        replace(d,options={"native_search":True,"search":{}})


def test_subset_completeness_does_not_claim_all_members(design):
    assessed={"candidate_id":"a","L":3,"assessment":{"finding":"within_descriptive_tolerance"},
              "warmup_exclusion_matches":True,"duplicate_chains":False,"stopped_intervals":{"quantities":[]}}
    other={"candidate_id":"b","L":5,"status":"unassessed_by_design"}
    result=summarize_replications(design("accuracy",route="prepared",replications=1,
        options={"posterior_members":"selected"}),[{"inventory":{"failures":[]},
        "members":[assessed,other],"tuning_completion":"complete"}])
    assert result["assessment_complete"]
    assert result["unassessed_by_design_members"]==1
    assert not result["all_verified_members_assessed"]


def test_missed_mode_reference_and_missing_comparator_denominator():
    draws=np.random.default_rng(76).normal(size=(128,4,2))
    draws[...,0]-=5.
    rows=arm_quantities(draws,get_target("mixture"),{},None,jit_compile=False)
    assert rows["left_mode_probability"]["error"] > .69
    result=summarize_pairs([{"stopped":rows}],2)
    assert result["quantities"]["left_mode_probability"]["arms"]["fixed"]["unavailable"]==2
    assert result["quantities"]["left_mode_probability"]["paired_replications"]==0


def test_entire_missing_stopping_cohort_keeps_declared_quantities(design):
    d=design("stopping","mixture","prepared",replications=3,
             options={"fixed_comparator":{"warmup_results":64,"retained_results":128}})
    result=summarize_replications(d,[])
    quantities=result["stopped_versus_fixed"]["quantities"]
    assert "left_mode_probability" in quantities
    assert "x:mean" in quantities
    assert all(q["arms"]["fixed"]["unavailable"]==3 for q in quantities.values())
    assert not result["comparison_complete"]


def test_real_simplex_member_and_fixed_comparator(design,tmp_path,monkeypatch):
    d=design("stopping","dirichlet","prepared",replications=1,draws=64,posterior_cap=256,
        step_size=.8,l_grid=(2,3),options={"member_rule":"first_verified","posterior_members":"selected",
        "acceptance_policy":{"practical_region":(.5,.9),"repair_region":(.45,.95)},
        "search":{"pilot_enabled":False,"refinement_rounds":0,"total_budget_units":40,"repair_reserve_units":8},
        "fixed_comparator":{"warmup_results":64,"retained_results":128}})
    output=execute_pipeline(d,tmp_path)
    assert output["verified_candidate_ids"]
    assert len(output["members"])==len(output["verified_candidate_ids"])
    members=[m for m in output["members"] if m["status"]=="assessed"]
    assert len(members)==1
    member=members[0]
    draws=read_tensor(member["fixed_comparator"]["draws_path"]).numpy()
    assert draws.shape==(128,4,3)
    np.testing.assert_allclose(draws.sum(-1),1.,atol=1e-12)
    assert read_tensor(member["draws_path"]).shape[-1]==3
    assert member["timing"]["controller_seconds"]>0
    for check in member["posterior"]["warmup_checks"]:
        assert "runtime" in check["health"]
    again=execute_pipeline(d,tmp_path)
    assert again["verified_candidate_ids"]==output["verified_candidate_ids"]
    assert read_json(tmp_path/"posterior_selection.json")["selected_candidate_ids"]==list(output["selection"]["candidate_ids"])
    from bayesfilter.testing.inference_validation import procedures
    def fail_fixed(*args,**kwargs):
        raise RuntimeError("injected comparator failure")
    monkeypatch.setattr(procedures,"run_fixed_comparator",fail_fixed)
    failed=execute_pipeline(d,tmp_path/"failed-comparator")
    member=next(m for m in failed["members"] if m["status"]=="assessed")
    assert member["fixed_comparator"]["status"]=="failed"
    assert "injected comparator failure" in member["fixed_comparator"]["reason"]
    assert read_tensor(member["draws_path"]).shape[-1]==3
