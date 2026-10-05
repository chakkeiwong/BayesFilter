"""Independent sequential-TFP oracle for trial batching; no tuning promotion."""
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference.hmc import FullChainHMCConfig, ReusableFullChainHMCRunner
from bayesfilter.inference.hmc_replicated_batch import ReplicatedTrialBatchRunner


def native_case(case):
    from tests.test_hmc_candidate_set_execution import make_binding, execution_config
    if case == 'gaussian':
        binding = make_binding(config=execution_config(use_xla=True,chain_mode='batched'))
        return binding._active_adapter, binding.initial_active_state, 1.3, 'candidate-bridge-test'
    from bayesfilter.testing.acceptance_release_validation import full_search_configuration
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.inference import (
        HMCCandidateExecutionConfig, PrecomputedMassArtifact, bind_hmc_candidate_set_execution)
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    config=full_search_configuration(case,seed=(20261002,2501),wall_seconds=1800)
    target=ValidationTarget(config['target'],config['parameters'],config['data'])
    starts=tf.constant(config['active_starts'],tf.float64)
    policy=HMCReplicatedAcceptancePolicy.from_payload(config['policy'])
    execution=HMCCandidateExecutionConfig(measurement_num_results=65,verification_num_results=65,
        num_warmup_steps=3,seed=(20261003,2900),acceptance_policy=policy,
        target_status_trace_policy='per_chain_step',chain_mode='batched',chunk_max_results=68)
    extra={}
    if case == 'funnel_residual':
        from bayesfilter.testing.inference_validation.funnel_maps import supplied_funnel_map
        extra['frozen_transport_payload']=supplied_funnel_map('residual')['transport_payload']
    else:
        factor=tf.linalg.diag(tf.constant(config['geometry']['scale'],tf.float64))
        extra['mass_artifact']=PrecomputedMassArtifact(position=config['geometry']['center'],factor=factor,
            covariance=tf.matmul(factor,factor,transpose_b=True),adapter_signature=target.adapter_signature(),
            position_role='native_batch_reference',covariance_source=config['provenance'])
    binding=bind_hmc_candidate_set_execution(adapter=target,initial_position=starts,
        start_coordinates='active',target_scope='inference_validation',config=execution,
        target_lineage={'model':target.target_id,'data':target.data,'parameters':target.parameters},
        source_paths=[str(Path(__file__))],scope_id='native-batch-parity',search_id=case,
        epsilon_domain=tuple(config['epsilon_domain']),repair_factor=1.3,max_repairs_per_family=0,**extra)
    return binding._active_adapter,starts,config['epsilon_by_l'][0][1][0],'inference_validation'


def kernel_config(epsilon,scope,steps=3):
    return FullChainHMCConfig(num_results=68,num_burnin_steps=0,step_size=epsilon,
        num_leapfrog_steps=steps,seed=(20261003,2900),use_xla=True,target_scope=scope,
        target_status_trace_policy='per_chain_step',capture_candidate_health=True)


@pytest.mark.parametrize('case',['gaussian','lgssm_qr','nonlinear','funnel_residual'])
@pytest.mark.parametrize('steps',[3,25])
def test_batched_trials_equal_individual_tfp_streams(case,steps):
    adapter,starts,epsilon,scope=native_case(case)
    config=kernel_config(epsilon,scope,steps)
    original=ReusableFullChainHMCRunner(adapter,starts,config,dynamic_num_leapfrog_steps=True)
    batched=ReplicatedTrialBatchRunner(adapter,starts,config,batch_size=2)
    seeds=tf.constant([[20261003,2901],[20261003,2902]],tf.int32)
    states=tf.stack([starts,starts+.01])  # Detect cross-trial state coupling.
    samples,trace,_=batched.run(states=states,seeds=seeds,step_size=epsilon,num_leapfrog_steps=steps)
    for index in range(2):
        ref=original.run(current_state=states[index],seed=seeds[index],step_size=epsilon,num_leapfrog_steps=steps)
        tf.debugging.assert_equal(samples[:,index],ref.samples)
        tf.nest.assert_same_structure(trace,ref.trace)
        for actual,expected in zip(tf.nest.flatten(trace),tf.nest.flatten(ref.trace)):
            tf.debugging.assert_equal(actual[:,index],expected)
    permuted,permuted_trace,_=batched.run(states=tf.reverse(states,[0]),seeds=tf.reverse(seeds,[0]),
                                         step_size=epsilon,num_leapfrog_steps=steps)
    tf.debugging.assert_equal(samples,tf.reverse(permuted,[1]))
    for actual,expected in zip(tf.nest.flatten(trace),tf.nest.flatten(permuted_trace)):
        tf.debugging.assert_equal(actual,tf.reverse(expected,[1]))
    assert batched._runner.experimental_get_tracing_count()==1


def test_duplicate_trial_seed_is_rejected_before_trace():
    adapter,starts,epsilon,scope=native_case('gaussian')
    runner=ReplicatedTrialBatchRunner(adapter,starts,kernel_config(epsilon,scope),batch_size=2)
    with pytest.raises(ValueError,match='share a root seed'):
        runner.run(states=tf.stack([starts,starts]),seeds=[[1,2],[1,2]],step_size=epsilon,num_leapfrog_steps=3)
    assert runner._runner.experimental_get_tracing_count()==0


@pytest.mark.parametrize('size',[0,33,True,2.5])
def test_unsupported_batch_size_is_rejected(size):
    adapter,starts,epsilon,scope=native_case('gaussian')
    with pytest.raises(ValueError,match='batch size'):
        ReplicatedTrialBatchRunner(adapter,starts,kernel_config(epsilon,scope),batch_size=size)


@pytest.mark.parametrize('case',['gaussian','lgssm_qr','nonlinear','funnel_residual'])
def test_maximum_batch_matches_all_32_original_streams(case):
    adapter,starts,epsilon,scope=native_case(case)
    config=kernel_config(epsilon,scope,25)
    reference=ReusableFullChainHMCRunner(adapter,starts,config,dynamic_num_leapfrog_steps=True)
    batch=ReplicatedTrialBatchRunner(adapter,starts,config,batch_size=32)
    seeds=tf.constant([[20261003,3000+i] for i in range(32)],tf.int32)
    states=tf.broadcast_to(starts,[32,*starts.shape])
    samples,trace,_=batch.run(states=states,seeds=seeds,step_size=epsilon,num_leapfrog_steps=25)
    # Every stream is checked; the largest batch does not inherit a tiny
    # two-trial fixture's numerical/shape qualification.
    for index in range(32):
        expected=reference.run(current_state=starts,seed=seeds[index],step_size=epsilon,num_leapfrog_steps=25)
        tf.debugging.assert_equal(samples[:,index],expected.samples)
        for actual,wanted in zip(tf.nest.flatten(trace),tf.nest.flatten(expected.trace)):
            tf.debugging.assert_equal(actual[:,index],wanted)
