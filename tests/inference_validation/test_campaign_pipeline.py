from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from bayesfilter.testing.inference_validation.procedures import selected_member_ids, execute_pipeline
from bayesfilter.testing.inference_validation.engines.stopping import arm_quantities, summarize_pairs
from bayesfilter.testing.inference_validation.engines.pipeline import summarize_replications
from bayesfilter.testing.inference_validation.catalog import get_target
from bayesfilter.testing.inference_validation.storage import read_tensor, read_json


def test_global_mode_quantity_vetoes_false_precision(design):
    import tensorflow as tf
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy, assess_posterior
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, HMCPrecisionTarget
    from bayesfilter.testing.inference_validation.procedures import posterior_quantities
    d = design("stopping", "mixture", "prepared", options={"global_quantities":["left_mode_probability"]})
    identity, quantities = posterior_quantities(d)
    draws = tf.constant(np.random.default_rng(67).normal(size=(2048,4,2)) + [5.,0.], tf.float64)
    rhat = rank_normalized_split_rhat_summary(draws, rhat_max=1.01)
    assert rhat["passed"]
    policy = HMCPosteriorAssessmentPolicy(quantities_id=identity,
        precision=HMCPrecisionPolicy((HMCPrecisionTarget("left_mode_probability", mcse_absolute_max=.1),),
                                     method="lugsail", jit_compile=False))
    result = assess_posterior(draws, ("x","y"), policy=policy, stage="retained", rhat=rhat, quantities_fn=quantities)
    assert not result["passed"] and not result["precision"]["passed"]
    assert "left_mode_probability" in result["quantity_names"]


def test_mode_quantity_interval_uses_the_declared_mixture_law(design):
    from scipy.special import ndtr
    from bayesfilter.testing.inference_validation.engines.pipeline import stopped_intervals
    params = {"separation": 2., "weight": .25}
    truth = .25 * ndtr(2.) + .75 * ndtr(-2.)
    member = {"posterior": {"passed": True, "warmup_cap_hit": False, "retained_cap_hit": False,
        "retained_checks": [{"diagnostic_role": "assessment", "assessment": {"precision": {"targets": [
            {"name": "left_mode_probability", "kind": "mean", "estimate": truth,
             "mcse": .02, "valid": True}]}}}]}}
    stopped = stopped_intervals(member, get_target("mixture"), params, None)
    assert stopped["quantities"][0]["reference"] == pytest.approx(truth)
    assert stopped["quantities"][0]["covered"]
    d = design("stopping", "mixture", "prepared", replications=2,
        options={"global_quantities": ["left_mode_probability"]})
    row = {"candidate_id": "a", "L":3, "assessment":{"finding":"within_descriptive_tolerance"},
        "warmup_exclusion_matches":True, "duplicate_chains":False, "stopped_intervals": stopped}
    result = summarize_replications(d, [{"inventory":{"failures":[]}, "members":[row], "tuning_completion":"complete"}])
    counts = result["interval_coverage_at_stop"]["left_mode_probability:mean"]
    assert counts["covered"] == counts["available"] == counts["unavailable"] == 1


def test_missing_mode_posteriors_keep_the_mode_quantity_denominator(design):
    d = design("stopping", "mixture", "prepared", replications=2,
        options={"global_quantities": ["left_mode_probability"]})
    counts = summarize_replications(d, [])["interval_coverage_at_stop"]["left_mode_probability:mean"]
    assert counts["planned"] == counts["unavailable"] == 2
    assert counts["available"] == 0


def test_inventory_allows_terminal_shared_failure_observation():
    from bayesfilter.testing.inference_validation.engines.pipeline import check_inventory
    from tests.test_hmc_candidate_set_tuning import _scope, _config, _pass
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCTuningCandidateSetController
    from bayesfilter.inference.hmc_candidate_set_artifacts import candidate_set_result_payload
    def observe(work, candidate):
        return ({"decision":"failed", "evidence_validity":"shared_execution_invalid"}
            if work.stage == "verification" and candidate.leapfrog_steps == 5 else _pass(work, candidate))
    result = HMCTuningCandidateSetController(_scope(), _config()).run(observe)
    payload = candidate_set_result_payload(result)
    assert result.completion_status == "shared_invalidity"
    assert check_inventory(payload)["failures"] == []
    payload["observations"] = (*payload["observations"], payload["observations"][-1])
    assert "missing_or_duplicated_observation" in check_inventory(payload)["failures"]


def test_mode_dispersed_starts_require_chain_bank_preserving_route(design):
    from bayesfilter.testing.inference_validation.designs import ScenarioSpec
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.procedures import initial_starts
    with pytest.raises(ValueError, match="preserves"):
        ScenarioSpec("mixture", "ordinary", start="mode_dispersed")
    starts = initial_starts(ValidationTarget("mixture", jit_compile=False), "mode_dispersed").numpy()
    assert np.all(starts[:2,0] < -4.) and np.all(starts[2:,0] > 4.)


def test_selection_depends_on_tuning_id_and_keeps_scope_explicit(design):
    candidates=[SimpleNamespace(candidate_id=cid,leapfrog_steps=l) for cid,l in (("b",3),("a",5))]
    assert selected_member_ids(design(),candidates)==("b",)
    assert selected_member_ids(design(options={"member_rule":"first_verified"}),candidates)==("a",)


def test_design_rejects_threshold_override_and_undeclared_fixed_counts(design):
    with pytest.raises(ValueError,match="count controls"):
        design(options={"posterior_settings":{"warmup_rhat_max":8}})
    with pytest.raises(ValueError,match="fixed comparator"):
        design("stopping",route="prepared",options={"fixed_comparator":{"retained_results":100}})


@pytest.mark.parametrize('budget', [None, {'max_results_per_chain': 12000},
    {'max_results_per_chain': True, 'count_budget_reason': 'test'},
    {'max_results_per_chain': 12000, 'count_budget_reason': ' '},
    {'max_results_per_chain': 12000, 'count_budget_reason': 'test', 'retained_rhat_max': 2.}])
def test_design_rejects_invalid_posterior_count_budget(design, budget):
    with pytest.raises(ValueError, match='posterior_count_budget'):
        design(options={'posterior_count_budget': budget})


def test_design_count_budget_bounds_all_requested_counts(design):
    budget = {'max_results_per_chain': 12000, 'count_budget_reason': 'declared mechanics allocation'}
    with pytest.raises(ValueError, match='posterior_cap'):
        design(posterior_cap=12000)
    with pytest.raises(ValueError, match='posterior counts'):
        design(options={'posterior_count_budget': budget,
                        'posterior_settings': {'retained_max_results': 12001}})
    with pytest.raises(ValueError, match='fixed comparator'):
        design('stopping', route='prepared', options={'posterior_count_budget': budget,
            'fixed_comparator': {'warmup_results': 400, 'retained_results': 12001}})
    ordinary = design('stopping', route='prepared')
    extended = replace(ordinary, options={'posterior_count_budget': budget}, posterior_cap=12000)
    assert ordinary.identity != extended.identity


@pytest.mark.parametrize('method', [None, '', 'unknown', ['lugsail']])
def test_design_rejects_invalid_posterior_precision_method(design, method):
    with pytest.raises(ValueError, match='posterior_precision_method'):
        design(options={'posterior_precision_method': method})


def test_real_public_pipeline_uses_explicit_count_budget(design, tmp_path):
    d = design('stopping', 'gaussian', 'prepared', replications=1, posterior_cap=12000,
        step_size=1.3, l_grid=(3,), budget_seconds=180,
        options={'posterior_members': 'selected', 'member_rule': 'first_verified',
            'posterior_precision_method': 'autocorrelation',
            'posterior_count_budget': {'max_results_per_chain': 12000,
                'count_budget_reason': 'complete public-route long-count integration'},
            'acceptance_policy': {'practical_region': (.41,.99), 'repair_region': (.405,.995)},
            'search': {'pilot_enabled': False, 'refinement_rounds': 0,
                'total_budget_units': 24, 'repair_reserve_units': 4, 'evidence_rungs': (1,)},
            'posterior_settings': {'warmup_chunk_results': 6000, 'warmup_min_results': 12000,
                'warmup_check_window_results': 6000, 'warmup_max_results': 12000,
                'retained_chunk_results': 6000, 'retained_min_results': 12000, 'retained_max_results': 12000},
            'fixed_comparator': {'warmup_results': 6000, 'retained_results': 12000}})
    from bayesfilter.testing.inference_validation.engines.pipeline import run as run_validation_pipeline
    assessed = run_validation_pipeline(d, tmp_path)
    fit_root = tmp_path / 'replication-0000'
    result = read_json(fit_root / 'pipeline.json')
    assert result['verified_candidate_ids']
    member = next(m for m in result['members'] if m['status'] == 'assessed')
    posterior = member['posterior']
    assert posterior['config']['max_results_per_chain'] == 12000
    assert posterior['config']['assessment_policy']['precision']['method'] == 'autocorrelation'
    assert posterior['warmup_results_per_chain'] == posterior['retained_results_per_chain'] == 12000
    assert posterior['warmup_excluded_from_posterior'] and not posterior['hard_vetoes']
    assert read_tensor(member['fixed_comparator']['draws_path']).shape == (12000,4,2)
    assert member['fixed_comparator']['warmup_excluded_from_estimates']
    pair = assessed['replications'][0]['members'][0]['stopping_pair']
    assert pair['stopped']['x:mean']['method'] == pair['fixed']['x:mean']['method'] == 'autocorrelation'
    last = posterior['retained_checks'][-1]
    target = next(t for t in last[last['diagnostic_role']]['precision']['targets']
                  if t['name'] == 'x' and t['kind'] == 'mean')
    assert pair['stopped']['x:mean']['mcse'] == pytest.approx(target['mcse'], rel=1e-12)
    from bayesfilter.testing.inference_validation.storage import file_hash
    hashes = {k:file_hash(member[k]) for k in ('draws_path','warmup_path')}
    resumed = execute_pipeline(d, fit_root)
    again = next(m for m in resumed['members'] if m['status'] == 'assessed')
    assert hashes == {k:file_hash(again[k]) for k in hashes}


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
    assert result["requested_members"] == result["posterior_output_members"] == 1
    assert result["posterior_unavailable_members"] == 0
    assert result["all_members_without_posterior_output"] == 1
    assert not result["all_verified_members_assessed"]


def test_missing_requested_posterior_is_not_an_unassessed_sibling(design):
    requested = {"candidate_id": "a", "L": 3, "status": "unfunded"}
    sibling = {"candidate_id": "b", "L": 5, "status": "unassessed_by_design"}
    result = summarize_replications(design("accuracy", route="prepared", replications=1,
        options={"posterior_members": "selected"}), [{"inventory": {"failures": []},
        "members": [requested, sibling], "tuning_completion": "complete"}])
    assert result["verified_members"] == 2
    assert result["requested_members"] == result["posterior_unavailable_members"] == 1
    assert result["unassessed_by_design_members"] == 1
    assert result["all_members_without_posterior_output"] == 2
    assert result["posterior_output_members"] == 0
    assert not result["assessment_complete"]
    assert all(row["planned"] == row["unavailable"] == 1
               for row in result["interval_coverage_at_stop"].values())


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


def test_stopping_coverage_distinguishes_missing_from_uncovered_intervals():
    # One covered interval, one missed interval, and two unavailable fits.
    pairs = [{"stopped":{"x:mean":{"available":True,"covered":covered,"error":error}}}
             for covered,error in ((True,.01),(False,.3))]
    result = summarize_pairs(pairs,4,declared_names=("x:mean",))
    arms = result["quantities"]["x:mean"]["arms"]
    assert arms["stopped"]["covered"] == 1
    assert arms["stopped"]["available"] == arms["stopped"]["unavailable"] == 2
    assert arms["stopped"]["conditional_coverage_interval"] != arms["stopped"]["coverage_interval"]
    assert arms["fixed"]["conditional_coverage_interval"] is None


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
