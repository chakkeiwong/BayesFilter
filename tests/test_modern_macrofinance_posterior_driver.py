"""C3 campaign mechanics: identities, independent uncertainty and pricing."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('macrofinance_posterior_driver',
    Path(__file__).resolve().parents[1]/'scripts/diagnostics/run_modern_hmc_macrofinance_posterior.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def reference():
    return dict(quantities=list(driver.QUANTITIES),mean=[0.]*31,sd=[1.]*31,
                screen=dict(passed=True,tolerance=.02,uncertainty=[.001]*31))

def test_reference_declared_pass_cannot_hide_missing_or_large_uncertainty():
    assert driver.validate_reference(reference())['uncertainty']==[.001]*31
    for bad in (-1.,float('nan'),.03):
        ref=reference();ref['screen']['uncertainty'][0]=bad
        with pytest.raises(ValueError):driver.validate_reference(ref)
    ref=reference();ref['quantities'].reverse()
    with pytest.raises(ValueError,match='identity'):driver.validate_reference(ref)

def test_all_members_required_and_ordered_by_frozen_kernel():
    members={str(i):SimpleNamespace(num_leapfrog_steps=L,step_size=e)
             for i,(L,e) in enumerate(reversed(driver.PAIRS))}
    rows=driver.validate_pairs(members)
    assert tuple((m.num_leapfrog_steps,m.step_size) for _,m in rows)==driver.PAIRS
    members.pop('0')
    with pytest.raises(ValueError,match='seven'):driver.validate_pairs(members)

def test_all_posterior_and_pricing_streams_disjoint():
    used=[]
    for index in range(7):
        roots=driver.roots(index);used.extend(roots.values())
        for phase in ('warmup','retained'):
            first,second=roots[phase]
            used.extend((first,second+1009*(i+1)) for i in range(10))
    assert len(set(used))==len(used)

def test_quote_uses_actual_all_member_prices_and_preserves_budget_limit():
    price=dict(status='completed',stage='price',elapsed_seconds=170,rows=[
        dict(L=L,epsilon=e,status='priced',seconds=10.,transitions_per_chain=1000) for L,e in driver.PAIRS])
    quote=driver.posterior_quote(price,2320)
    assert quote['required_seconds']==2320
    assert quote['affordable']
    assert not driver.posterior_quote(price,2319)['affordable']
    price['rows'][0]['status']='numerical_health_failed'
    with pytest.raises(ValueError,match='seven'):driver.posterior_quote(price,10000)


def test_campaign_delegates_all_members_and_reference_to_shared_recovery(monkeypatch,tmp_path):
    from bayesfilter.testing.inference_validation import modern_full_model_consumer
    calls=[]
    def screen(values,reference,**kwargs):
        calls.append((values,reference,kwargs));return dict(passed=False)
    monkeypatch.setattr(modern_full_model_consumer,'reference_screen',screen)
    ordered=[(str(i),object()) for i in range(7)]
    previous=dict(rows=[dict(candidate_id=cid,status='posterior_not_started') for cid,_ in ordered])
    config=SimpleNamespace(assessment_policy=SimpleNamespace(precision='explicit-policy'))
    def run_cohort(**kwargs):
        assert kwargs['previous_report'] is previous
        assert [c['candidate_id'] for c in kwargs['candidates']]==[str(i) for i in range(7)]
        assert list(kwargs['members'])==[str(i) for i in range(7)]
        assert kwargs['names']==driver.QUANTITIES
        assert kwargs['factory'](ordered[6][1],6) is config
        assert not kwargs['reference_check']('retained-only',config)['passed']
        return dict(status='completed',delivered_count=0,rows=previous['rows'])
    result=driver.execute_posterior_cohort(ordered=ordered,output=tmp_path/'new',
        transform=None,reference=reference(),check_inputs=lambda:None,
        helper=SimpleNamespace(run_cohort=run_cohort),previous_report=previous,
        config_factory=lambda member,index:config)
    assert result['delivered_count']==0
    assert calls[0][0]=='retained-only'
    assert calls[0][2]==dict(mcse_multiplier=3.,normalized_tolerance=.1,precision_policy='explicit-policy')


def test_actual_shared_config_covers_every_quantity_and_derived_seed():
    from bayesfilter.inference.hmc_posterior_assessment import validate_sequential_seeds
    used=set()
    for index,(L,epsilon) in enumerate(driver.PAIRS):
        member=SimpleNamespace(step_size=epsilon,num_leapfrog_steps=L)
        config=driver.posterior_config(member,index)
        assert config.step_size==epsilon and config.num_leapfrog_steps==L
        assert tuple(t.name for t in config.assessment_policy.precision.targets)==driver.QUANTITIES
        assert config.assessment_policy.precision.method=='lugsail'
        assert config.jit_compile
        inventory=validate_sequential_seeds(config,forbidden=used)
        used.update(seed for _,_,seed in inventory);used.update(driver.roots(index).values())
        assert len(inventory)==20


def test_price_scope_rejects_changed_target_and_unverified_source(tmp_path):
    inputs=tmp_path/'inputs.json';inputs.write_text('target A')
    ref=tmp_path/'reference.json';ref.write_text('reference A')
    launch=dict(commit='frozen-commit',inputs=str(inputs))
    manifest=dict(stage='price',git_commit='frozen-commit',source_sha256={
        str(inputs):driver.digest(inputs),str(ref):driver.digest(ref)})
    price=dict(source_unchanged=True)
    driver.validate_price_scope(price,manifest,launch,ref)
    ref.write_text('different target reference')
    with pytest.raises(ValueError,match='scope'):driver.validate_price_scope(price,manifest,launch,ref)
    with pytest.raises(ValueError,match='integrity'):
        driver.validate_price_scope({},manifest,launch,ref)


def test_fixed_replication_uses_fresh_seeds_and_preserves_all_other_assessments():
    from dataclasses import asdict
    from bayesfilter.inference.hmc_posterior_assessment import validate_sequential_seeds
    old_seeds=set()
    for index,(L,epsilon) in enumerate(driver.PAIRS):
        member=SimpleNamespace(step_size=epsilon,num_leapfrog_steps=L)
        old=driver.posterior_config(member,index)
        old_seeds.update(seed for _,_,seed in validate_sequential_seeds(old))
        old_seeds.update(driver.roots(index).values())
    used=set(old_seeds)
    for index,(L,epsilon) in enumerate(driver.PAIRS):
        member=SimpleNamespace(step_size=epsilon,num_leapfrog_steps=L)
        old=driver.posterior_config(member,index)
        new=driver.posterior_config(member,index,True)
        assert new.retained_min_results==new.retained_max_results==10000
        differences={k for k,v in asdict(old).items() if v!=asdict(new)[k]}
        assert differences=={'warmup_seed','retained_seed','retained_min_results'}
        inventory=validate_sequential_seeds(new,forbidden=used)
        used.update(seed for _,_,seed in inventory);used.update(driver.roots(index,True).values())


def test_prefix_selection_uses_internal_assessment_not_fixed_count_or_reference_verdict():
    checks=[dict(completed_results_per_chain=n,health=dict(health_passed=True),
                 modern_rhat=dict(passed=passed),passed=False,reference=dict(passed=False))
            for n,passed in [(1000,False),(2000,True),(3000,True)]]
    result=dict(warmup_passed=True,retained_checks=checks)
    assert driver.earliest_internal_prefix(result)==2000
    checks[1]['reference']['passed']=True
    assert driver.earliest_internal_prefix(result)==2000
    checks[0]['health']['health_passed']=False
    assert driver.earliest_internal_prefix(result) is None
    result['warmup_passed']=False
    assert driver.earliest_internal_prefix(result) is None


def test_prefix_refusal_remains_unavailable_and_never_selects_the_terminal_by_default():
    result=dict(warmup_passed=True,retained_checks=[dict(completed_results_per_chain=10000,
        health=dict(health_passed=True),modern_rhat=dict(passed=False),passed=False)])
    assert driver.earliest_internal_prefix(result) is None


def test_fixed_comparison_keeps_failed_terminal_and_assesses_only_selected_prefix(tmp_path,monkeypatch):
    import tensorflow as tf
    from bayesfilter.testing.inference_validation import modern_full_model_consumer
    result=dict(warmup_passed=True,retained_results_per_chain=10000,hard_vetoes=[],
        retained_checks=[dict(completed_results_per_chain=2000,health=dict(health_passed=True),
                              modern_rhat=dict(passed=True),passed=False)])
    driver.write(tmp_path/'result.json',result)
    tf.io.write_file(str(tmp_path/'retained_raw.tensor'),
                     tf.io.serialize_tensor(tf.ones([10000,4,31],tf.float64)))
    row=dict(candidate_id='member',delivered=False,result_file=str(tmp_path/'result.json'),
             reference=dict(passed=False))
    seen=[]
    def screen(values,reference,**kwargs):
        seen.append(tuple(values.shape));return dict(passed=True)
    monkeypatch.setattr(modern_full_model_consumer,'reference_screen',screen)
    config=SimpleNamespace(assessment_policy=SimpleNamespace(precision='frozen'))
    report=driver.fixed_count_comparisons(dict(rows=[row]),[('member',object())],reference(),lambda m,i:config)
    assert seen==[(2000,4,31)]
    assert report[0]['early_prefix_reference']['passed']
    assert not report[0]['terminal_delivered'] and not report[0]['terminal_reference']['passed']
    assert not row['delivered'] and report[0]['fixed_count_reached']
