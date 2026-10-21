"""CPU diagnostic exceptions: real native adapter and nonlinear pullback."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
from bayesfilter.inference.neural_force_hmc import (
    FrozenPositionOnlyForce, FrozenTargetPotential, bind_neural_force_hmc_tuning_runner,
    InvalidNeuralForceHMCConfiguration, PositionOnlyForceEvaluation)
from bayesfilter.testing.surrogate_hmc_sir_neutra import SIRNeuTraTarget, frozen_neutra_bindings


def test_frozen_map_force_and_independent_endpoint_match_derivatives():
    dtype = tf.float64
    transport = NeuTraTransport(NeuTraTransportConfig.hoffman_author_iaf(
        3, dtype="float64", conditional_scale_cap=2., seed=(11,23)))
    signature = "b"*64
    payload = transport.frozen_payload(target_signature=signature)
    raw_force = FrozenPositionOnlyForce(
        lambda q: PositionOnlyForceEvaluation(q+.4*q**3, tf.reduce_all(tf.abs(q)<30.,-1)), "aux")
    # Deliberately distinct endpoint: the connection must preserve its value.
    raw_target = FrozenTargetPotential(lambda q: tf.reduce_sum(q*q, -1), "independent")
    bound = frozen_neutra_bindings(payload, target_signature=signature,
        raw_force=raw_force, raw_target=raw_target, target_scope="sir-test")
    z = tf.constant([[.1,-.2,.3],[-.3,.2,-.1]],dtype)
    with tf.GradientTape() as tape:
        tape.watch(z)
        q, ld = transport.forward_and_logdet(z)
        aux = tf.reduce_sum(.5*q*q+.1*q**4,-1)-ld
    expected = tape.gradient(aux,z)
    actual = bound["force"].function(z)
    np.testing.assert_allclose(actual.force, expected, atol=2e-10,rtol=2e-10)
    assert bool(tf.reduce_all(actual.force_valid))
    for row in range(2):
        endpoint = bound["target"].function(z[row:row+1])
        np.testing.assert_allclose(endpoint.potential[0], tf.reduce_sum(q[row]**2)-ld[row],
                                   atol=2e-10,rtol=2e-10)
        assert bool(endpoint.status["value_valid"][0])
    assert bound["binding"].coordinate_scope == "transformed"
    assert bound["binding"].endpoint_target_includes_chart_log_jacobian
    assert not bound["transport"].trainable_variables
    factor = tf.linalg.diag(tf.constant([.8,1.1,1.3],dtype))
    center = tf.constant([.01,.02,-.03],dtype)
    adapter = SimpleNamespace(transform=SimpleNamespace(factor=factor,center=center),
        latent_to_position=lambda x: center+tf.linalg.matmul(x,factor,transpose_b=True))
    kernel = bound["binding"].build_tensor_kernel(adapter,
        step_size=.01, num_leapfrog_steps=2, target_scope=bound["binding"].target_scope,
        target_status_trace_policy="none", chain_execution_mode="tf_function", use_xla=True)
    mapped_z = adapter.latent_to_position(z)
    expected_force = tf.linalg.matmul(bound["force"].function(mapped_z).force,factor)
    np.testing.assert_allclose(kernel.force.function(z).force,expected_force,rtol=1e-12)
    np.testing.assert_allclose(kernel.target.function(z).potential,
                              bound["target"].function(mapped_z).potential,rtol=1e-12)
    @tf.function(input_signature=[tf.TensorSpec([2,3],dtype)],jit_compile=True)
    def transition(x):
        return kernel.one_step(x,kernel.bootstrap_results(x),seed=(21,31))
    state, result = transition(z)
    assert bool(tf.reduce_all(result.finite_status))
    assert not bool(tf.reduce_any(result.force_fallback))
    assert bool(tf.reduce_all(tf.math.is_finite(state)))
    with pytest.raises(InvalidNeuralForceHMCConfiguration):
        bind_neural_force_hmc_tuning_runner(force=bound["force"], target=bound["target"],
                                           target_scope="unbound")
    with pytest.raises(ValueError):
        frozen_neutra_bindings(payload,target_signature="c"*64,raw_force=raw_force,
                              raw_target=raw_target,target_scope="sir-test")


def test_matched_sir_adapter_real_native_program_and_prior(tmp_path):
    from bayesfilter.testing import surrogate_hmc_literature_priors as priors
    from bayesfilter.testing.surrogate_hmc_five_model import Case
    from bayesfilter.highdim.sqmc_nonlinear_tf import NonlinearSQMCSpec
    from bayesfilter.highdim.covariance_proposal_tf import ProposalControls
    dtype = tf.float32
    spec = NonlinearSQMCSpec("sir_d18")
    theta = spec.default_theta(dtype)
    initial = tf.random.stateless_normal([32,18],[21,1],dtype=dtype)
    noise = tf.random.stateless_normal([2,32,18],[21,2],dtype=dtype)
    uniform = tf.random.stateless_uniform([2,32,2],[21,3],dtype=dtype)
    model,_ = spec.model(theta,tf.zeros_like(theta))
    state = spec.initial_cloud(theta,initial,tf.zeros_like(theta))[0]
    obs=[]
    for _ in range(2):
        state=model.transition_mean_fn(theta,state)
        obs.append(tf.reduce_mean(model.observation_fn(state),0))
    data = tmp_path/"data.json"
    data.write_text("tiny independent diagnostic data")
    case=Case("sir_d18",spec,tf.stack(obs),theta,(initial,noise,uniform),
        ProposalControls(flow_steps=2,reset_steps=5),tf.constant([.2,.3,.5],dtype),
        data,prior_choice=priors.PRIOR_ID)
    adapter=SIRNeuTraTarget(case,2)
    q=tf.constant([[.001,.002,-.001],[-.002,-.001,.001]],dtype)
    result=adapter.evaluate(q)
    values,scores,valid=adapter.value_score(q)
    physical,jacobian=case.chart(q)
    raw=adapter.program(physical,case.observations,*case.fixed,case.beta)
    prior,dp=case.log_prior(q)
    np.testing.assert_allclose(values,raw["value"]+prior,rtol=2e-6)
    np.testing.assert_allclose(scores,raw["score"]*jacobian+dp,rtol=2e-6,atol=2e-6)
    np.testing.assert_array_equal(valid,[True,True])
    np.testing.assert_array_equal(result["steps_completed"],[2,2])
    assert "validity_trace" in result and "score_trace" in result
    assert adapter.target_signature != SIRNeuTraTarget(replace(case,
        beta=tf.constant([.3,.3,.4],dtype)),2).target_signature
    with pytest.raises(ValueError):
        SIRNeuTraTarget(case,1)
    from bayesfilter.testing.surrogate_hmc_sir_neutra import verify_sir_training_scope
    transport = NeuTraTransport(NeuTraTransportConfig.hoffman_author_iaf(
        3, dtype="float64", conditional_scale_cap=2., seed=(11,23)))
    payload = transport.frozen_payload(target_signature=adapter.target_signature)
    assert verify_sir_training_scope(case,payload,adapter.scope) == adapter.target_signature
    for changed in (replace(case, beta=tf.constant([.3,.3,.4],dtype)),
                    replace(case, prior_choice=priors.BENCHMARK_PRIOR_ID),
                    replace(case, observations=case.observations+.01),
                    replace(case, theta0=case.theta0+.01),
                    replace(case, fixed=(initial+1e-3,noise,uniform)),
                    replace(case, controls=replace(case.controls, flow_steps=3))):
        with pytest.raises(ValueError, match="training scope"):
            verify_sir_training_scope(changed,payload,adapter.scope)
    with pytest.raises(ValueError, match="scope"):
        verify_sir_training_scope(case,payload,None)
    bad_payload = {**payload, "config":{**payload["config"], "mask_policy":"legacy_degree_masks_v1"}}
    with pytest.raises(ValueError, match="canonical"):
        verify_sir_training_scope(case,bad_payload,adapter.scope)
