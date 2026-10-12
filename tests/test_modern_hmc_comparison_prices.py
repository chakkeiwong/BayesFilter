"""Accounting/denominator regression tests; no sampler performance claims."""
import json

import pytest

from scripts.price_modern_hmc_frozen_design import cost_envelope, finalize, planned_rows


def design():
    return dict(schema="bayesfilter.modern_frozen_full_model_design.v1", target="NL-PP-SGQF",
        method="mams", seed=71410, max_repairs_per_family=1, repair_factor=1.5,
        execution=dict(num_warmup_steps=0, acceptance_policy=dict(
            trial_num_results=64, base_repetitions=32, max_repetitions=128, max_candidates=4)),
        search=dict(epsilon_by_l=[[2,[1.]], [5,[.5]]], max_candidates=4, evidence_rungs=[1,2,4]),
        posterior=dict(sequential=dict(warmup_min_results=2000, retained_min_results=1000,
            warmup_max_results=10000, retained_max_results=10000)))


def prices():
    return dict(roots=[dict(index=i, leapfrog_steps=l, epsilon=e, calls=[
        dict(status="returned", transitions=64, elapsed_seconds=64., hard_vetoes=[]),
        dict(status="returned", transitions=64, elapsed_seconds=32., hard_vetoes=[])])
        for i,(l,e) in enumerate(((2,1.), (5,.5)))])


def test_incremental_looks_fresh_stages_all_repairs_and_chains_accounted_once():
    result=cost_envelope(design(), prices(), load_multiplier=2, overhead_seconds=100, trial_overhead_seconds=0)
    assert result["stage_incremental_repetitions"] == [32,32,64]
    # ceil((L+1)*1.5) gives conservative child counts 5 and 9.
    rate=2*(1+6/3+1+10/6)
    assert result["qualification_seconds_by_final_rung"]["128"] == pytest.approx(rate*2*128*64)
    assert result["all_member_posterior_seconds"]["maximum"] == pytest.approx(rate*20000)
    assert result["maximum_count_scenario_seconds"] == pytest.approx(100+rate*(256*64+20000))
    assert result["reserved_member_count"] == 4
    assert result["probabilistic_upper_bound"] is False


def test_candidate_veto_still_consumes_complete_envelope():
    before=cost_envelope(design(), prices(), load_multiplier=1, overhead_seconds=0, trial_overhead_seconds=0)
    after=prices()
    after["roots"][0]["calls"][1]["hard_vetoes"]=["nonfinite"]
    result=cost_envelope(design(), after, load_multiplier=1, overhead_seconds=0, trial_overhead_seconds=0)
    assert result["maximum_count_scenario_seconds"] == before["maximum_count_scenario_seconds"]
    assert result["root_rates"][0]["numerical_veto"] is True


@pytest.mark.parametrize("mutation", ["omitted_root","missing_call","nan_price","wrong_pair",
    "missing_repairs","missing_error_allowance","duplicate_look","missing_posterior"])
def test_incomplete_or_unfunded_prices_cannot_suggest_launch(mutation):
    d,p=design(),prices()
    if mutation=="omitted_root": p["roots"].pop()
    elif mutation=="missing_call": p["roots"][0]["calls"].pop()
    elif mutation=="nan_price": p["roots"][0]["calls"][0]["elapsed_seconds"]=float("nan")
    elif mutation=="wrong_pair": p["roots"][0]["epsilon"] = 2.
    elif mutation=="missing_repairs": d["search"]["max_candidates"] = 2
    elif mutation=="missing_error_allowance": d["execution"]["acceptance_policy"]["max_candidates"]=2
    elif mutation=="duplicate_look": d["search"]["evidence_rungs"]=[1,2,4,8]
    elif mutation=="missing_posterior": d["posterior"]=None
    with pytest.raises(ValueError):
        cost_envelope(d,p,load_multiplier=1,overhead_seconds=0,trial_overhead_seconds=0)


def test_per_trial_overhead_charged_for_every_reserved_member_and_fresh_stage():
    base=cost_envelope(design(),prices(),load_multiplier=1,overhead_seconds=0,trial_overhead_seconds=0)
    plus=cost_envelope(design(),prices(),load_multiplier=1,overhead_seconds=0,trial_overhead_seconds=3)
    assert plus["maximum_count_scenario_seconds"]-base["maximum_count_scenario_seconds"] == 3*2*128*4


def test_declared_price_inventory_preserved_when_supervisor_interrupts(tmp_path):
    path=tmp_path/"design.json"
    path.write_text(json.dumps(design()))
    rows=planned_rows([path])
    assert len(rows[0]["roots"])==2
    rows[0]["status"]="running"
    rows[0]["roots"][0].update(status="priced",calls=prices()["roots"][0]["calls"])
    (tmp_path/"result.json").write_text(json.dumps(dict(status="running", rows=rows)))
    result=finalize(tmp_path,status="supervisor_budget_exhausted",elapsed=180.)
    assert result["rows"][0]["roots"][0]["status"]=="priced"
    assert result["rows"][0]["roots"][1]["status"]=="not_started_supervisor_budget_exhausted"
    assert result["supervisor_elapsed_seconds"]==180.
    with pytest.raises(ValueError,match="duplicate price cell"):
        planned_rows([path,path])
