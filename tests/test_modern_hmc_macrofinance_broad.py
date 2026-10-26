"""Broad-search engineering checks; CPU Gaussian tests are not consumer evidence."""
from dataclasses import replace
import json
from types import SimpleNamespace
import pytest
from scripts import continue_modern_hmc_macrofinance_broad as broad
from scripts import run_modern_hmc_filter_health_gpu as supervisor


def prices(seconds=10.):
    return [dict(L=length,repeat=repeat,num_results=64,elapsed_seconds=seconds,
        health_failures=[],epsilon=.4,transition_profile="hmc_uniform_duration",discarded_prefix=0,chain_count=4) for length in broad.L_GRID for repeat in (0,1)]


def test_full_horizon_price_counts_measurement_and_verification_once():
    report=broad.affordability(prices(),7000.)
    assert report['root_native_projection_seconds']==3840.
    assert report['conservative_root_allowance_seconds']==5760.
    assert report['available_tuning_seconds']==5980.
    assert report['affordable']
    assert not broad.affordability(prices(),6779.)['affordable']
    assert broad.affordability(prices(),6780.)['affordable']


@pytest.mark.parametrize('fault',['short','missing','duplicate','bad_health','nan','negative','epsilon','profile','prefix','chains'])
def test_price_rejects_short_missing_corrupt_or_unhealthy_calls(fault):
    rows=prices()
    if fault=='short':rows[0]['num_results']=8
    elif fault=='missing':rows.pop()
    elif fault=='duplicate':rows[-1]=dict(rows[0])
    elif fault=='bad_health':rows[0]['health_failures']=['nonfinite']
    elif fault=='nan':rows[0]['elapsed_seconds']=float('nan')
    elif fault=='negative':rows[0]['elapsed_seconds']=-1.
    elif fault=='epsilon':rows[0]['epsilon']=.1
    elif fault=='profile':rows[0]['transition_profile']='fixed_hmc'
    elif fault=='prefix':rows[0]['discarded_prefix']=16
    elif fault=='chains':rows[0]['chain_count']=1
    with pytest.raises(ValueError):broad.affordability(rows,10000.)


def test_controller_funds_all_roots_and_caps_statistical_family():
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    bound=SimpleNamespace(config=SimpleNamespace(acceptance_policy=broad.policy(),
        transition_profile=HMCTransitionProfile('hmc','uniform_duration')))
    cfg,quote=broad.controller_config(bound,10000.)
    assert cfg.primary_l_grid==(3,5,9,13,18,25)
    assert cfg.max_candidates==cfg.replicated_acceptance_policy.max_candidates==12
    assert cfg.epsilon_by_l==tuple((length,(.4,)) for length in broad.L_GRID)
    assert cfg.refinement_rounds==1 and cfg.epsilon_refinement_factors==(1.25,)
    assert cfg.max_gradient_work==quote['initial_cohort_mandatory_gradient_work']
    assert cfg.max_gradient_work==2*32*64*4*sum(2*(length+1) for length in broad.L_GRID+(25,)*6)
    assert cfg.replicated_acceptance_policy.qualification_mode=='lower_bound'


def test_real_broad_controller_keeps_all_parents_and_siblings(tmp_path,monkeypatch):
    from tests.test_modern_hmc_pipeline import setup
    from tests.test_hmc_candidate_set_execution import make_binding
    from bayesfilter.inference import load_hmc_candidate_retained_runners
    base,_=setup()
    # Deliberately permissive existing Gaussian mechanics policy, not study policy.
    execution=replace(base.config,acceptance_policy=replace(base.config.acceptance_policy,max_candidates=12))
    bound=make_binding(config=execution,max_repairs_per_family=1)
    # This integration fixture tests retention, not precision. The actual-policy
    # preflight refusal and boundary behavior have separate tests below.
    import scripts.diagnose_modern_hmc_evidence_design as design
    monkeypatch.setattr(design, 'resolution_screen', lambda policy: {'preferred_center_resolvable': True})
    result=broad.tune(bound._base_adapter,bound,tmp_path,120.)
    assert result['completion_status']=='complete'
    assert result['candidate_count']==12 and len(result['verified'])==12
    saved=json.loads((tmp_path/'tuning/tuning_checkpoint.json').read_text())['result']
    pairs={(c['leapfrog_steps'],c['epsilon']) for c in saved['candidates']}
    assert pairs=={(length,eps) for length in broad.L_GRID for eps in (.4,.5)}
    assert all(state=='verified' for state in result['candidate_states'].values())
    audit=result['native_audit']
    assert audit['returned_chunk_count']==audit['attempted_chunk_count']==384
    seeds=[tuple(row['seed']) for row in audit['native_call_audit']]
    assert len(seeds)==len(set(seeds))
    members=load_hmc_candidate_retained_runners(json.loads((tmp_path/'members.json').read_text()).values(),adapter=bound._base_adapter)
    assert set(members)==set(result['verified'])
    for member in members.values():
        assert not member.claim_eligible and not member.posterior_convergence_authority
    assert audit['rhat_role']=='reporting_only'


@pytest.mark.parametrize('task',['broad-price','broad-tune','broad-reload'])
def test_shared_supervisor_dispatches_broad_tasks_and_refreshes_terminal_progress(task,tmp_path,monkeypatch):
    from tests.test_modern_hmc_shared_gpu import gpu
    clock=[0.]
    monkeypatch.setattr(supervisor.time,'monotonic',lambda:clock[0])
    monkeypatch.setattr(supervisor.time,'sleep',lambda seconds:clock.__setitem__(0,clock[0]+seconds))
    monkeypatch.setattr(supervisor,'query_nvidia_smi_gpus',lambda:[gpu(0,0),gpu(1,20)])
    source=tmp_path/'source';source.mkdir()
    (source/'manifest.json').write_text(json.dumps(dict(environment=dict(
        CUDA_VISIBLE_DEVICES='GPU-1',TF_FORCE_GPU_ALLOW_GROWTH='true'),physical_gpu=dict(uuid='GPU-1'))))
    class Process:
        pid=456;returncode=None
        def poll(self):
            if clock[0]>=2:self.returncode=0
            return self.returncode
    def launch(command,**kwargs):
        assert command[1].endswith('continue_modern_hmc_macrofinance_broad.py')
        assert '--device' not in command
        assert command[command.index('--mode')+1]==task.removeprefix('broad-')
        assert kwargs['env']['CUDA_VISIBLE_DEVICES']==('GPU-1' if task=='broad-reload' else 'GPU-0')
        assert kwargs['env']['TF_FORCE_GPU_ALLOW_GROWTH']=='true'
        evidence=tmp_path/'run/evidence';evidence.mkdir()
        (evidence/'result.json').write_text(json.dumps(dict(status='completed',execution=dict(completion_status='complete'))))
        return Process()
    monkeypatch.setattr(supervisor.subprocess,'Popen',launch)
    monkeypatch.setattr(supervisor.os,'killpg',lambda *args:pytest.fail('healthy worker killed'))
    monkeypatch.setattr(supervisor,'checkpoint_progress',lambda output:dict(saved_trials=int(clock[0])))
    args=SimpleNamespace(output=tmp_path/'run',inputs=tmp_path/'inputs.json',source=source,
        task=task,max_wall_seconds=100.,worker_wall_seconds=80.)
    assert supervisor.supervise(args)==0
    manifest=json.loads((args.output/'manifest.json').read_text())
    assert manifest['plan']==broad.PLAN
    assert manifest['numerical_progress']['saved_trials']==2


def prospective():
    return dict(schema='bayesfilter.broad_prospective_design.v1',
        policy=replace(broad.policy(),base_repetitions=16,max_repetitions=128,max_candidates=24).payload(),
        evidence_rungs=[1,2,4,8],numeric_provenance='Test-only explicit hypotheses; no power claim',
        discarded_proposals=dict(epsilon_by_l=[(l,.4) for l in broad.L_GRID], transitions=8,
            attempts_per_l=4,factor=2.,preferred_region=(.65,.75),seed=(20261025,401),max_wall_seconds=120.))


def test_prospective_design_prices_all_candidates_and_preserves_evidence_growth():
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    raw=prospective();p,pilot,rungs,quote=broad.reviewed_design(raw)
    bound=SimpleNamespace(config=SimpleNamespace(acceptance_policy=p,
        transition_profile=HMCTransitionProfile('hmc','uniform_duration')))
    cfg,quote=broad.controller_config(bound,100000.,design=raw,
        epsilon_by_l=[(l,(.4,.2)) for l in broad.L_GRID])
    assert cfg.evidence_rungs==(1,2,4,8)
    assert cfg.max_gradient_work==2*sum(quote['maximum_stage_gradient_work_by_candidate'])
    assert cfg.epsilon_refinement_factors==(.5,2.) and cfg.explore_failed_intervals
    cost=broad.full_design_price(prices(),100000.,raw)
    assert cost['maximum_qualification_native_seconds']==2*24*128*10.
    assert not broad.full_design_price(prices(),5400.,raw)['full_search_funded']
    assert cost['full_search_funded']


def test_prospective_design_requires_every_L_and_capacity_for_all_nominees():
    raw=prospective();raw['discarded_proposals']['epsilon_by_l'].pop()
    with pytest.raises(ValueError,match='every broad L'):broad.reviewed_design(raw)
    raw=prospective();raw['discarded_proposals']['attempts_per_l']=5
    with pytest.raises(ValueError,match='every healthy proposal'):broad.reviewed_design(raw)


def test_fresh_cli_refuses_legacy_design_before_output_or_target(tmp_path):
    import subprocess,sys
    completed=subprocess.run([sys.executable,broad.__file__,'--mode','tune',
        '--inputs',str(tmp_path/'missing.json'),'--output',str(tmp_path/'no-output'),
        '--wall-seconds','60'],capture_output=True,text=True)
    assert completed.returncode==2 and 'requires --design' in completed.stderr
    assert not (tmp_path/'no-output').exists()


def test_price_source_requires_same_data_and_numerical_sources(tmp_path):
    inputs=tmp_path/'inputs.json';inputs.write_text('{}')
    code=tmp_path/'source.py';code.write_text('x=1')
    manifest=dict(inputs_sha256=broad.digest(inputs),source_sha256={str(code):broad.digest(code)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'result.json').write_text(json.dumps(dict(status='completed',execution=dict(rows=prices()))))
    assert broad.validated_price_source(tmp_path,{str(code):broad.digest(code)},inputs)==prices()
    code.write_text('x=2')
    with pytest.raises(ValueError,match='source changed'):
        broad.validated_price_source(tmp_path,{str(code):broad.digest(code)},inputs)
