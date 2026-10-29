"""Actual LEDH endpoint/force rejection mechanics; no posterior claims."""
from dataclasses import replace
import json,math,os,runpy
from pathlib import Path
import pytest
import tensorflow as tf
from bayesfilter.highdim.ledh_canonical_batch_fused_tf import (
    canonical_batch_fused_value_score,canonical_batch_fused_value_score_whileloop)
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce,FrozenTargetPotential,NeuralForceHMCConfig,
    PositionOnlyForceEvaluation,TargetPotentialEvaluation,neural_force_proposal,
    neural_force_hmc_transition,sample_neural_force_hmc,
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS)

fixture=runpy.run_path(str(Path(__file__).with_name('test_ledh_failure_contract.py')))['fixture']

def actual_target(entry,force_failure=False):
    model,inputs,controls=fixture(horizon=3)
    if force_failure:
        model=replace(model,transition_mean_tangent_fn=lambda theta,x,dx,dt:
            tf.fill(tf.shape(x),tf.constant(float('nan'),x.dtype)))
    @tf.function(input_signature=[tf.TensorSpec([2,1],tf.float64)],jit_compile=True)
    def evaluate(q):
        return entry(model,q,tf.ones([2,1,1],q.dtype),*inputs,**controls)
    def force(q):
        _,s,status=evaluate(q)
        return PositionOnlyForceEvaluation(-s,status['force_valid'])
    def potential(q):
        v,_,status=evaluate(q)
        return TargetPotentialEvaluation(-v,status)
    return evaluate,FrozenPositionOnlyForce(force,'test-real-ledh-force',
        semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS),\
        FrozenTargetPotential(potential,'test-real-ledh-target')

def preserve(name,payload):
    directory=os.environ.get('LEDH_PROGRAM_RESULT_DIR')
    if directory:
        def host(x):
            if tf.is_tensor(x):return host(x.numpy().tolist())
            if isinstance(x,dict):return {k:host(v) for k,v in x.items()}
            if isinstance(x,(list,tuple)):return [host(v) for v in x]
            if isinstance(x,float) and not math.isfinite(x):
                return {'__nonfinite_float__': 'NaN' if math.isnan(x) else ('Infinity' if x>0 else '-Infinity')}
            return x
        (Path(directory)/(name+'.json')).write_text(json.dumps(host(payload),indent=2,allow_nan=False)+'\n')

@pytest.mark.parametrize('entry',[canonical_batch_fused_value_score,canonical_batch_fused_value_score_whileloop])
def test_real_failed_endpoint_rejects_and_keeps_cache(entry):
    evaluate,force,target=actual_target(entry)
    q=tf.constant([[.001],[.7]],tf.float64)
    current=-evaluate(q)[0]
    config=NeuralForceHMCConfig(.02,1,(.4,))
    seed=tf.constant([7037,2],tf.int32)
    # Search is solely a deterministic mechanics-fixture construction, not tuning.
    for i in range(20):
        seed=tf.constant([7037,i],tf.int32)
        ps=tf.random.experimental.stateless_split(seed,2)[0]
        momentum=tf.random.stateless_normal([2,1],ps,dtype=tf.float64)/tf.sqrt(tf.constant(.4,tf.float64))
        if float(momentum[0,0]) < -.5:break
    result=neural_force_hmc_transition(q,current,force,target,config,seed)
    status=result.trace.endpoint_status
    assert status['numerical_failure'].numpy().tolist()==[True,False]
    assert status['outside_declared_support'].numpy().tolist()==[False,False]
    assert int(status['first_failure_time'][0])==0
    assert int(status['first_failure_stage'][0])==1
    assert not bool(result.trace.accepted[0])
    tf.debugging.assert_equal(result.position[0],q[0])
    tf.debugging.assert_equal(result.potential[0],current[0])
    assert bool(result.trace.finite_status[1])
    assert bool(result.trace.force_fallback[0])
    preserve('rejection-'+entry.__name__,dict(q=q,seed=seed,step_size=.02,inverse_mass=[.4],
        next_position=result.position,next_potential=result.potential,trace=result.trace._asdict()))

@pytest.mark.parametrize('inverse_mass',[1.,.4])
def test_real_finite_value_failed_force_is_reversible_and_can_accept(inverse_mass):
    evaluate,force,target=actual_target(canonical_batch_fused_value_score,force_failure=True)
    q=tf.constant([[.6],[.7]],tf.float64);p=tf.constant([[.01],[-.01]],tf.float64)
    config=NeuralForceHMCConfig(.001,2,(inverse_mass,))
    v,_,status=evaluate(q)
    assert bool(tf.reduce_all(status['value_valid'] & ~status['force_valid']))
    proposed=neural_force_proposal(q,p,force,config)
    reverse=neural_force_proposal(proposed.position,proposed.momentum,force,config)
    tf.debugging.assert_near(reverse.position,q,atol=1e-12,rtol=1e-12)
    tf.debugging.assert_near(reverse.momentum,p,atol=1e-12,rtol=1e-12)
    result=neural_force_hmc_transition(q,-v,force,target,config,[7042,2])
    assert bool(tf.reduce_all(result.trace.force_fallback))
    assert bool(tf.reduce_all(result.trace.accepted))
    independent_delta=(result.trace.final_potential-result.trace.initial_potential+
        .5*inverse_mass*tf.reduce_sum(result.trace.final_momentum**2-result.trace.initial_momentum**2,axis=-1))
    tf.debugging.assert_near(result.trace.log_acceptance_ratio,
        tf.minimum(tf.zeros_like(independent_delta),-independent_delta),atol=1e-12,rtol=1e-12)
    preserve('finite-force-extension-'+str(inverse_mass),dict(q=q,momentum=p,
        proposed_position=proposed.position,proposed_momentum=proposed.momentum,
        inverse_mass=[inverse_mass],step_size=.001,leapfrog=2,trace=result.trace._asdict()))

def test_chain_retains_numerical_status_and_invalid_initial_cache_raises():
    evaluate,force,target=actual_target(canonical_batch_fused_value_score)
    q=tf.constant([[.001],[.7]],tf.float64);v=-evaluate(q)[0]
    config=NeuralForceHMCConfig(.02,1,(.4,))
    chain=sample_neural_force_hmc(q,v,force,target,config,num_warmup=1,num_results=2,seed=[7042,3])
    assert chain.endpoint_status['numerical_failure'].shape==[3,2]
    tf.debugging.assert_all_finite(chain.potentials,'invalid endpoint corrupted retained cache')
    with pytest.raises(tf.errors.InvalidArgumentError):
        neural_force_hmc_transition(q,tf.constant([float('inf'),1.],tf.float64),force,target,config,[1,2])
