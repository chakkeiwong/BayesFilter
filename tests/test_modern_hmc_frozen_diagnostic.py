"""Frozen actual-consumer harness mechanics; no target-power claims."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
from scripts import run_modern_hmc_macrofinance_frozen_diagnostic as study


def test_shared_invalidity_is_preserved_before_audit_or_export(tmp_path, monkeypatch):
    """Mocked worker lifecycle; no GPU launch or scientific evidence."""
    import json
    import bayesfilter.inference as inference
    from bayesfilter.runtime import gpu_provenance
    from bayesfilter.testing.inference_validation import modern_bridge_audit
    from scripts import diagnose_modern_hmc_macrofinance_public as target
    source = tmp_path/'source.py'
    source.write_text('version = 1')
    inputs = tmp_path/'inputs.json'
    inputs.write_text('{}')
    campaign = tmp_path/'campaign'
    campaign.mkdir()
    (campaign/'frozen-cohort.json').write_text(json.dumps({
        'qualification_seed': [1, 2], 'reload_seed': [3, 4]}))
    (campaign/'proposal-design.json').write_text('{}')
    output = tmp_path/'run'
    output.mkdir()
    monkeypatch.setenv('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-mocked-no-device-work')
    monkeypatch.setattr(study, 'INPUTS', inputs)
    monkeypatch.setattr(study, 'source_paths', lambda _: ([source], None))
    monkeypatch.setattr(gpu_provenance, 'query_nvidia_smi_gpus', lambda: [])
    monkeypatch.setattr(gpu_provenance, 'selected_nvidia_gpu', lambda _: {'uuid': 'mock'})
    monkeypatch.setattr(target, 'build', lambda *a, **k: (object(), {}))
    fake_config = SimpleNamespace(payload=lambda: {})
    fake_binding = SimpleNamespace(config=SimpleNamespace(acceptance_policy=fake_config),
        initial_active_state=[], typed_adapter=object())
    monkeypatch.setattr(study, 'frozen_binding', lambda *a: fake_binding)
    monkeypatch.setattr(study, 'frozen_config', lambda *a: (fake_config, {}))
    primary = {'event': 'execution_failure', 'reason': 'execution source changed',
               'type': 'HMCSharedInvalidity'}

    def fail_tuning(**kwargs):
        source.write_text('version = 2')
        return SimpleNamespace(result=SimpleNamespace(completion_status='shared_invalidity',
            final_status='shared_invalidity', candidate_states={'old': 'verified'},
            verified_candidate_ids=[], candidates=[], repair_actions=[],
            accounting_events=[primary]))

    monkeypatch.setattr(inference, 'tune_hmc_kernel', fail_tuning)
    def forbidden(*a, **k):
        pytest.fail('shared invalidity must stop before audit/export')
    monkeypatch.setattr(modern_bridge_audit, 'audit_native_evidence', forbidden)
    monkeypatch.setattr(inference, 'export_hmc_candidate_retained_runners', forbidden)
    assert study.worker(SimpleNamespace(output=output, campaign=campaign,
                                       wall_seconds=1200, source=None)) == 1
    report = json.loads((output/'evidence/result.json').read_text())
    assert report['status'] == 'shared_invalidity'
    assert report['execution']['execution_failures'] == [primary]
    assert not report['execution']['verified']
    assert not report['execution']['all_verified_members_exported']
    assert not report['sources_unchanged']
    assert (output/'evidence/controller-outcome.json').is_file()


def cohort():
    return dict(epsilon_by_l=[(3,(.4,.8)),(5,(.4,.8)),(9,(.4,)),(13,(.4,)),(18,(.4,)),(25,(.2,.4))],
        controller_candidate_cap=9,statistical_family_cap=24,refinement_rounds=0,max_repairs_per_family=0,
        evidence_rungs=[1,2,4,8],qualification_seed=[20261025,6101])


def price_rows():
    roots=[];extra=[]
    for l,eps in study.pairs(cohort()):
        for e in eps:
            for repeat in range(2):
                r=dict(L=l,epsilon=e,repeat=repeat,num_results=64,chain_count=4,discarded_prefix=0,
                    transition_profile='hmc_uniform_duration',elapsed_seconds=float(l),health_failures=[],cold_call=False)
                (roots if e==.4 and l!=3 else extra).append(r)
    extra.append(dict(extra[0],cold_call=True,elapsed_seconds=1000.))
    return roots,extra


def test_complete_nominee_price_counts_compilation_once_and_keeps_bad_pair():
    a,b=price_rows();b[2]['health_failures']=['nonfinite_target_score']
    cost=study.cohort_price(a,b,cohort(),100000.)
    assert len(cost['pairs'])==9 and cost['statistical_family_cap']==24
    assert cost['native_maximum_seconds']==256*(3+3+5+5+9+13+18+25+25)
    assert cost['required_seconds']==1.5*cost['native_maximum_seconds']+1140
    assert cost['prior_price_vetoes'] and cost['affordable']
    assert not study.cohort_price(a,b,cohort(),100.)['affordable']


@pytest.mark.parametrize('failure',['missing','short','nonfinite','expansion'])
def test_invalid_cohort_or_incomplete_price_refuses_launch(failure):
    a,b=price_rows();c=cohort()
    if failure=='missing':b.pop(0)
    if failure=='short':a[0]['num_results']=8
    if failure=='nonfinite':a[0]['elapsed_seconds']=float('nan')
    if failure=='expansion':c['refinement_rounds']=1
    with pytest.raises(ValueError):study.cohort_price(a,b,c,100000.)


def test_actual_cohort_cap_preserves_statistical_multiplicity_and_maximum_work():
    from scripts.continue_modern_hmc_macrofinance_broad import policy
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    p=replace(policy(),base_repetitions=16,max_repetitions=128,max_candidates=24)
    bound=SimpleNamespace(config=SimpleNamespace(acceptance_policy=p,transition_profile=HMCTransitionProfile('hmc','uniform_duration')))
    cfg,quote=study.frozen_config(bound,cohort(),88000.)
    assert cfg.max_candidates==9 and cfg.replicated_acceptance_policy.max_candidates==24
    assert cfg.replicated_acceptance_policy.one_sided_alpha('measurement')==p.one_sided_alpha('measurement')
    assert cfg.max_gradient_work==2*128*64*4*sum(2*(l+1) for l,es in study.pairs(cohort()) for _ in es)
    assert cfg.evidence_rungs==(1,2,4,8) and cfg.refinement_rounds==0
    assert not cfg.explore_failed_intervals


def test_controller_keeps_failed_pair_and_every_healthy_member_without_expansion():
    from tests.test_hmc_candidate_set_tuning import _scope
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig,HMCTuningCandidateSetController
    c=cohort()
    cfg=HMCControllerConfig(primary_l_grid=tuple(l for l,_ in study.pairs(c)),epsilon_by_l=study.pairs(c),
        total_budget_units=50,repair_reserve_units=1,max_candidates=9,evidence_rungs=(1,),refinement_rounds=0)
    seen=[]
    def observe(work,candidate):
        seen.append((candidate.leapfrog_steps,candidate.epsilon,work.stage))
        if (candidate.leapfrog_steps,candidate.epsilon)==(5,.8):
            return dict(decision='repair_step_lower',acceptance=.01)
        return dict(decision='passed',acceptance=.7)
    result=HMCTuningCandidateSetController(replace(_scope(),max_repairs_per_family=0),cfg).run(observe)
    assert result.completion_status=='complete'
    assert len(result.candidates)==9 and len(result.verified_candidate_ids)==8
    assert not result.repair_actions
    assert {(l,e) for l,e,_ in seen}=={(l,e) for l,es in study.pairs(c) for e in es}


def test_source_drift_requires_exact_fresh_binding_reconciliation(tmp_path, monkeypatch):
    import hashlib, json
    root = tmp_path / 'root'
    scripts = root / 'scripts'
    scripts.mkdir(parents=True)
    for name in ('resume_modern_hmc_macrofinance_public.py',
                 'resume_modern_hmc_bridge.py',
                 'resume_modern_hmc_full_model_qualification.py'):
        (scripts / name).write_text(name)
    source = tmp_path / 'changed.py'
    source.write_text('before')
    prior = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_text('after')
    current = hashlib.sha256(source.read_bytes()).hexdigest()
    campaign = tmp_path / 'campaign'
    (campaign / 'price-01' / 'evidence').mkdir(parents=True)
    (campaign / 'frozen-cohort.json').write_text('{}')
    (campaign / 'proposal-design.json').write_text('{}')
    (campaign / 'price-01' / 'evidence' / 'manifest.json').write_text(
        json.dumps({'source_sha256': {str(source): prior}}))
    monkeypatch.setattr(study, 'ROOT', root)
    with pytest.raises(ValueError, match='reconciliation required'):
        study.source_paths(campaign)
    (campaign / 'source-reconciliation.json').write_text(json.dumps({
        'allowed_paths': [str(source)],
        'excluded_paths': [str(source)],
        'changed_paths': [{'path': str(source), 'prior_sha256': prior,
                           'current_sha256': current}],
        'fresh_binding_required': True,
        'old_numerical_evidence_reused': False,
    }))
    paths, reconciliation = study.source_paths(campaign)
    assert source not in paths and reconciliation['fresh_binding_required'] is True
    source.write_text('changed again while the target is running')
    paths, _ = study.source_paths(campaign)
    assert source not in paths
    (campaign / 'source-reconciliation.json').write_text(json.dumps({
        'allowed_paths': [str(source)], 'changed_paths': [],
        'excluded_paths': [],
        'fresh_binding_required': True, 'old_numerical_evidence_reused': False,
    }))
    with pytest.raises(ValueError, match='exactly cover'):
        study.source_paths(campaign)


def test_nine_pair_numerical_controller_exports_all_fresh_verified_members(tmp_path):
    import json
    from bayesfilter.inference import tune_hmc_kernel,export_hmc_candidate_retained_runners,load_hmc_candidate_retained_runners
    from tests.test_modern_hmc_pipeline import setup
    from tests.test_hmc_candidate_set_execution import make_binding
    base,_=setup()
    # Existing Gaussian mechanics floor .1 remains test-only. The actual model
    # uses .55; this integration checks scheduling, fresh evidence and reload.
    p=replace(base.config.acceptance_policy,max_candidates=24,max_repetitions=128)
    bound=make_binding(config=replace(base.config,acceptance_policy=p,use_xla=True,non_xla_reason=None),max_repairs_per_family=0)
    cfg,quote=study.frozen_config(bound,cohort(),120.)
    run=tune_hmc_kernel(adapter=bound._base_adapter,initial_position=bound.initial_active_state,
        config=cfg,candidate_set_adapter=bound.typed_adapter,output_dir=tmp_path/'tuning')
    assert run.result.completion_status=='complete'
    assert len(run.result.candidates)==9 and len(run.result.verified_candidate_ids)==9
    assert not run.result.repair_actions
    paths=export_hmc_candidate_retained_runners(candidate_set_result=run.result,retained_binding=bound,output_dir=tmp_path/'members')
    loaded=load_hmc_candidate_retained_runners(paths.values(),adapter=bound._base_adapter)
    assert set(loaded)==set(run.result.verified_candidate_ids)
    assert all(not m.claim_eligible and not m.posterior_convergence_authority for m in loaded.values())
    saved=json.loads((tmp_path/'tuning/tuning_checkpoint.json').read_text())['result']
    assert saved['config']['max_candidates']==9
    assert saved['config']['replicated_acceptance_policy']['max_candidates']==24
    seeds=[tuple(e['seed']) for e in run.result.accounting_events if e['event']=='numerical_chunk_charged']
    assert len(seeds)==len(set(seeds))
