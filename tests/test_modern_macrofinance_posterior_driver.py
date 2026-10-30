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
