"""Public tuner status archives are engineering evidence, never admission."""
from dataclasses import replace
import tensorflow as tf
from bayesfilter.inference import (
    DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS,
    FourChainMeanBandAcceptancePolicy, FrozenPositionOnlyForce,
    FrozenTargetPotential, TensorFlowHMCKernelTuningConfig,
    bind_neural_force_hmc_tuning_runner, tune_hmc_kernel,
    load_tensorflow_hmc_tuning_result,
)
from bayesfilter.inference.neural_force_hmc import (
    PositionOnlyForceEvaluation, TargetPotentialEvaluation,
)
from bayesfilter.inference.hmc_tensorflow_tuning import ENDPOINT_STATUS_TRACE_FIELDS

class Adapter:
    target_scope = 'ledh-status-archive-test'
    def adapter_signature(self): return 'ledh-status-archive-adapter-v1'

def config():
    return TensorFlowHMCKernelTuningConfig(
        parameter_dimension=1, evidence_role='diagnostic_only',
        mass_window_results=(1,), step_adaptation_results=1,
        verification_results=1, max_leapfrog_steps=4, initial_step_size=.1,
        budget_provenance='tiny status archive regression',
        initial_step_size_provenance='convenience diagnostic',
        geometry_provenance='unit diagnostic scales',
        target_scope=Adapter.target_scope,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0.,1.), per_chain_band=(0.,1.)),
        target_accept_prob=.7, verification_repair_rounds=0,
        step_repair_factor=2., mass_shrinkage=.1, covariance_jitter=1e-9,
        eigenvalue_floor=1e-9, max_condition_number=1e8, seed=(20260917,31))

def binding(force, target):
    return bind_neural_force_hmc_tuning_runner(
        force=FrozenPositionOnlyForce(force, identity='status-force-v1',
            semantics=DETERMINISTIC_POSITION_ONLY_PROPOSAL_FIELD_SEMANTICS),
        target=FrozenTargetPotential(target, identity='status-target-v1'),
        target_scope=Adapter.target_scope)

def run(cfg, bound, out=None):
    return tune_hmc_kernel(adapter=Adapter(), initial_position=tf.zeros([1],tf.float64),
        parameter_scales=tf.ones([1],tf.float64), config=cfg,
        runner_binding=bound, output_dir=out)

def test_public_status_roundtrip_and_skipped_candidates(tmp_path):
    bound=binding(lambda q:q,lambda q:.5*tf.reduce_sum(q*q,axis=-1))
    result=run(config(),bound,tmp_path/'tuning')
    trace=result.endpoint_status_trace
    assert trace.shape==(3,4,len(ENDPOINT_STATUS_TRACE_FIELDS))
    tf.debugging.assert_equal(trace[:,:,0],tf.ones([3,4],tf.int32))
    tf.debugging.assert_equal(trace[:,:,10],tf.constant([[0]*4,[1]*4,[2]*4]))
    # Later trajectory candidates were skipped, so contribute no fictitious rows.
    assert float(tf.reduce_sum(result.candidate_summary[:,0]))==1
    loaded=load_tensorflow_hmc_tuning_result(result.artifact_manifest_path,
        runner_binding=bound,adapter=Adapter())
    tf.debugging.assert_equal(loaded.endpoint_status_trace,trace)
    assert not bool(loaded.handoff_eligible)

def test_public_status_keeps_support_numerical_and_force_distinct():
    def target(q):
        x=q[:,0];inside=tf.abs(x)<1e-12;support=x>=1e-12;numerical=x<=-1e-12
        zero=tf.zeros_like(x,tf.int32);absent=zero-1
        status=dict(value_valid=inside,force_valid=inside,
            outside_declared_support=support,numerical_failure=numerical,
            value_failure_code=tf.where(numerical,6,zero),
            force_failure_code=tf.where(inside,zero,3),
            first_failure_time=tf.where(inside,absent,2),
            first_failure_stage=tf.where(inside,absent,4),
            first_failure_particle=tf.where(inside,absent,0))
        return TargetPotentialEvaluation(tf.where(inside,tf.zeros_like(x),
            tf.constant(float('inf'),tf.float64)),status)
    def force(q):
        return PositionOnlyForceEvaluation(tf.zeros_like(q),tf.abs(q[:,0])<1e-12)
    result=run(replace(config(),max_leapfrog_steps=1),binding(force,target))
    trace=result.endpoint_status_trace
    assert bool(tf.reduce_any(trace[:,:,2]>0))
    assert bool(tf.reduce_any(trace[:,:,8]>0))
    tf.debugging.assert_equal(trace[:,:,2]+trace[:,:,8],tf.ones([3,4],tf.int32))
    tf.debugging.assert_equal(trace[:,:,9],tf.ones([3,4],tf.int32))
    tf.debugging.assert_equal(result.final_raw_state,tf.zeros([4,1],tf.float64))
    assert not bool(result.passed)

def test_public_status_retains_executed_repair():
    # Small steps have near-unit acceptance: this intentionally narrow diagnostic
    # band requests one larger-step retry; it is not a sampling criterion.
    cfg=replace(config(),max_leapfrog_steps=1,initial_step_size=.001,
        verification_repair_rounds=1,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=(0.,.01),per_chain_band=(0.,.01)))
    result=run(cfg,binding(lambda q:q,lambda q:.5*tf.reduce_sum(q*q,axis=-1)))
    trace=result.endpoint_status_trace
    assert trace.shape==(4,4,len(ENDPOINT_STATUS_TRACE_FIELDS))
    tf.debugging.assert_equal(trace[:,:,10],tf.constant([[0]*4,[1]*4,[2]*4,[2]*4]))
    tf.debugging.assert_equal(trace[:,:,12],tf.constant([[0]*4,[0]*4,[0]*4,[1]*4]))
