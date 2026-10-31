"""Diagnostic fixtures for source-bound full80 glue, not posterior evidence."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','1')
import pytest

def test_npz_decoder(tmp_path):
    import numpy as np  # Independent archive-format fixture only.
    import tensorflow as tf
    from docs.benchmarks.full80_walnutpie_support import load_arrays
    p=tmp_path/'arrays.npz'
    np.savez(p,x=np.arange(12,dtype=np.float64).reshape(3,4),bad=np.array([object()]))
    tf.debugging.assert_equal(load_arrays(tf,p,['x'])['x'],tf.reshape(tf.range(12,dtype=tf.float64),[3,4]))
    with pytest.raises(ValueError,match='unsupported array storage'):
        load_arrays(tf,p,['bad'])

@pytest.mark.parametrize('order',[32,64])
def test_quadrature_matches_original_reference(order):
    import numpy as np  # Independent original setup reference.
    import tensorflow as tf
    from docs.benchmarks.full80_walnutpie_support import normal_quadrature
    nodes,weights=normal_quadrature(tf,order)
    x,w=np.polynomial.hermite.hermgauss(order)
    tf.debugging.assert_near(nodes,tf.constant(x*np.sqrt(2)),atol=5e-14,rtol=5e-14)
    tf.debugging.assert_near(weights,tf.constant(w/np.sqrt(np.pi)),atol=5e-14,rtol=5e-14)
    tf.debugging.assert_near(tf.reduce_sum(weights),tf.constant(1.,tf.float64),atol=5e-14,rtol=5e-14)

def test_explicit_score_and_identity_lifecycle():
    import tensorflow as tf
    from docs.benchmarks.full80_walnutpie_support import score_model,check_state
    from docs.benchmarks.walnutpie_gpu_tf import Config,make_supplied_mass_functions
    def evaluate(q):
        # Stop the original derivative to prove the explicit score is consumed.
        return tf.stop_gradient(-.5*tf.reduce_sum(q*q,-1)),-q,dict(
            status_code=tf.zeros([2],tf.int32),valid_pre_regularized_score=tf.ones([2],tf.bool))
    cfg=Config(2,3,warmup=3,draws=3,max_depth=2,max_halvings=2,adapt_mass=False)
    f=make_supplied_mass_functions(tf,score_model(tf,evaluate),cfg,adapt_step=False)
    q=tf.constant([[.1,.2,.3],[-.3,.4,.2]],tf.float64)
    state=f.initialize(q,tf.constant([[17,0],[29,0]]),tf.ones_like(q),tf.constant([.1,.1],tf.float64))
    tf.debugging.assert_equal(state.g,-q)
    warm=f.warmup(state);sample=f.sample(warm.state)
    check_state(tf,sample.state,evaluate)
    tf.debugging.assert_equal(sample.state.epsilon,warm.state.epsilon)
    tf.debugging.assert_equal(warm.state.inverse_mass,tf.ones_like(q))
    tf.debugging.assert_equal(sample.state.minimum,warm.state.minimum)

def test_invalid_status_does_not_become_valid_density():
    import tensorflow as tf
    from docs.benchmarks.full80_walnutpie_support import score_model
    from docs.benchmarks.walnutpie_gpu_tf import Config
    from docs.benchmarks.walnutpie_gpu_tf.core import Kernel
    def evaluate(q):
        return tf.zeros([2],tf.float64),tf.ones_like(q),dict(status_code=tf.constant([0,2]),
            valid_pre_regularized_score=tf.constant([True,False]))
    k=Kernel(tf,score_model(tf,evaluate),Config(2,3))
    q=tf.zeros([2,3],tf.float64);lp,g=k.value_gradient(q)
    tf.debugging.assert_equal(k.health(q,lp,g),[True,False])

def test_source_hash_mismatch_fails_before_import(tmp_path,monkeypatch):
    import json
    import docs.benchmarks.full80_walnutpie_support as support
    p=tmp_path/'checkpoint';p.write_text('changed')
    b=tmp_path/'binding.json';b.write_text(json.dumps(dict(checkpoint=str(p),checkpoint_sha256='incorrect')))
    monkeypatch.setattr(support,'BINDING',b)
    with pytest.raises(ValueError,match='changed bound input: checkpoint'):
        support.verify_binding()

def test_comparison_requires_valid_chains():
    import tensorflow as tf
    from docs.benchmarks.full80_walnutpie_report import compare
    ref=dict(passed=True,mean=tf.zeros([80],tf.float64),sd=tf.ones([80],tf.float64),
        reporting_mcse=tf.fill([80],tf.constant(.01,tf.float64)))
    candidate={**ref,'passed':False}
    assert compare(tf,ref,candidate,list(map(str,range(80))),warmup_ready=True)['qualified'] is False
    candidate={**ref,'mean':tf.tensor_scatter_nd_update(ref['mean'],[[0]],[1.])}
    result=compare(tf,ref,candidate,list(map(str,range(80))),warmup_ready=True)
    assert result['verdict']=='incompatible_at_declared_screen'
    assert result['flagged_parameters']==['0']
    assert compare(tf,ref,ref,list(map(str,range(80))),warmup_ready=True)['verdict']=='compatible_at_available_precision'
    broken={**ref,'reporting_mcse':tf.fill([80],tf.constant(float('nan'),tf.float64))}
    assert not compare(tf,ref,broken,list(map(str,range(80))),warmup_ready=True)['qualified']

def small_state(tf):
    from docs.benchmarks.walnutpie_gpu_tf import core,api
    cfg=core.Config(4,2,adapt_mass=False)
    kernel=core.Kernel(tf,lambda q:-.5*tf.reduce_sum(q*q,-1),cfg)
    state=api.initial_state(kernel,tf.zeros([4,2],tf.float64),
        tf.constant([[17+i,0] for i in range(4)]),mass=tf.ones([4,2],tf.float64),
        epsilon=tf.constant([1.,1.01,.99,1.02],tf.float64),adapt_step=False)
    return cfg,state

def test_author_monitor_does_not_average_or_mistake_cap_for_agreement():
    import tensorflow as tf
    from docs.benchmarks.full80_walnutpie_sampling_helpers import adaptation_decision
    _,state=small_state(tf)
    assert not adaptation_decision(tf,state._replace(iteration=tf.constant(45)))['freeze']
    result=adaptation_decision(tf,state._replace(iteration=tf.constant(50)))
    assert result['freeze'] and result['parameter_agreement']
    tf.debugging.assert_equal(state.epsilon,tf.constant([1.,1.01,.99,1.02],tf.float64))
    different=state._replace(iteration=tf.constant(1000),epsilon=tf.constant([1.,2.,3.,4.],tf.float64))
    result=adaptation_decision(tf,different)
    assert result['freeze'] and not result['parameter_agreement'] and result['reason']=='maximum'
    assert not result['posterior_stationarity_established']

def test_checkpoint_round_trip_and_corruption(tmp_path):
    import tensorflow as tf
    from docs.benchmarks.full80_walnutpie_sampling_helpers import checkpoint,restore_checkpoint
    from docs.benchmarks.walnutpie_gpu_tf.api import state_spec
    cfg,state=small_state(tf)
    records=checkpoint(tf,state,tmp_path,'state')
    restored=restore_checkpoint(tf,records,tmp_path,state_spec(tf,cfg))
    for a,b in zip(tf.nest.flatten(state),tf.nest.flatten(restored)):
        tf.debugging.assert_equal(a,b)
    (tmp_path/records[0]['path']).write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checkpoint hash'):
        restore_checkpoint(tf,records,tmp_path,state_spec(tf,cfg))

def test_controller_preserves_handoff_rejections_and_continues_after_failed_screen():
    import copy
    from types import SimpleNamespace
    import tensorflow as tf
    from docs.benchmarks.walnutpie_gpu_tf.api import Run
    from docs.benchmarks.full80_walnutpie_sampling import run_stream
    from docs.benchmarks.full80_walnutpie_sampling_helpers import adaptation_decision,POLICY
    _,initial=small_state(tf)
    policy={**POLICY,'retained_min':5,'retained_max':10,'assessment_stride':5}
    ctl=dict(phase='adapt',serial=0,adapt_count=0,retained_count=0,
        adaptation_complete=False,history=[],assessments=[])
    handoffs=[];archives=[];saved=[]
    def transition(s,adapt):
        handoffs.append((adapt,s))
        # All five outputs are rejections; they must still be archived/count.
        state=s._replace(iteration=s.iteration+5 if adapt else s.iteration,
            seed=s.seed+tf.constant([0,5]))
        q=tf.repeat(s.q[None],5,axis=0)
        return Run(state,q,tf.repeat(s.lp[None],5,0),*[tf.ones([5,4],tf.int32)]*3,
            tf.ones([5,4],tf.bool),tf.ones([5,4],tf.bool))
    def record(r,phase,seconds,c):
        archives.append((phase,r.draws))
        return dict(phase=phase,count=5,seconds=seconds)
    def assessment(c):return dict(passed=c['retained_count']==10)
    end,result=run_stream(tf,initial,ctl,
        functions=SimpleNamespace(warmup=lambda s:transition(s,True),sample=lambda s:transition(s,False)),
        policy=policy,adaptation_decision=lambda s:adaptation_decision(tf,s),
        validate=lambda s:tf.debugging.assert_equal(s.inverse_mass,tf.ones_like(s.inverse_mass)),
        record=record,assess=assessment,persist=lambda s,c:saved.append((s,copy.deepcopy(c))),
        budget=lambda c:None,forecast=lambda c:{},timed=lambda f,s:(f(s),1.),progress=lambda *a,**k:None)
    assert result['adapt_count']==50 and result['retained_count']==10
    assert [a['passed'] for a in result['assessments']]==[False,True]
    assert len(handoffs)==12 and result['phase']=='complete'
    tf.debugging.assert_equal(handoffs[10][1].seed,initial.seed+tf.constant([0,50]))
    tf.debugging.assert_equal(end.seed,initial.seed+tf.constant([0,60]))
    assert sum(int(x.shape[0]) for phase,x in archives if phase=='retained')==10
    tf.debugging.assert_equal(end.epsilon,initial.epsilon)

def test_resume_rejects_old_pilot_and_changed_source():
    from docs.benchmarks.full80_walnutpie_sampling import validate_resume
    current=dict(kind='full80_walnutpie_comparison',policy={},config={},binding_sha256='a',
        target_scope='b',source_hashes={'core':'c'})
    validate_resume(current,current)
    with pytest.raises(ValueError,match='engineering pilot'):
        validate_resume({**current,'kind':'pilot'},current)
    with pytest.raises(ValueError,match='source mismatch'):
        validate_resume({**current,'source_hashes':{'core':'different'}},current)

def test_forecast_uses_actual_counts_and_excludes_compilation():
    from docs.benchmarks.full80_walnutpie_sampling_helpers import timing_forecast
    history=[dict(phase='adapt',seconds=1000.,count=5,first_compiled_call=True),
        dict(phase='adapt',seconds=50.,count=5,first_compiled_call=False)]
    result=timing_forecast(history,'adapt',0,50)
    assert result['sample_seconds_per_transition']==10.
    assert not result['sampling_measured']
    history += [dict(phase='retained',seconds=25.,count=5,first_compiled_call=False)]
    result=timing_forecast(history,'retained',100,50)
    assert result['sampling_measured']
    assert result['next_assessment_seconds']==1.25*1300*5

def test_authorized_additional_seventy_hour_budget():
    from docs.benchmarks.full80_walnutpie_sampling import validate_allowance
    validate_allowance(251900,252000)
    validate_allowance(7200,252000)
    for wall,total in ((252001,252001),(252001,252000),(59,252000)):
        with pytest.raises(ValueError,match='70 hours'):
            validate_allowance(wall,total)
