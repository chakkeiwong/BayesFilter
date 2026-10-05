"""Full-chain compilation diagnostics; four draws cannot establish convergence."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.mcmc.internal.leapfrog_integrator import SimpleLeapfrogIntegrator

from bayesfilter.inference.native_tfp_hmc import reviewed_independent_chain_target_fn
from bayesfilter.testing.inference_validation.ssm_campaign_profiles import PROFILES, generate_data
from bayesfilter.testing.inference_validation.targets import ValidationTarget


@pytest.mark.parametrize("name", list(PROFILES))
def test_actual_ssm_full_chain_xla_matches_graph(name):
    """Compile the entire supplied-score chain before declaring eligibility."""
    data = None if PROFILES[name].case == "K6" else generate_data(name, (20260926,71))
    outputs=[]
    for jit in (False,True):
        target=ValidationTarget(name,data=data,jit_compile=jit)
        initial=target._ssm.initial_starts()
        target_fn=reviewed_independent_chain_target_fn(target,chain_count=4,parameter_dim=target.parameter_dim)
        kernel=tfp.mcmc.HamiltonianMonteCarlo(target_fn,tf.constant(.001,tf.float64),2)
        def trace(state,result):
            status=target.target_status_telemetry(state)
            return (result.log_accept_ratio,result.accepted_results.target_log_prob,
                    status["status_code"],status["valid_pre_regularized_score"],status["floor_count_value"])
        @tf.function(input_signature=[tf.TensorSpec([4,target.parameter_dim],tf.float64),
                                      tf.TensorSpec([2],tf.int32)],autograph=False,jit_compile=jit)
        def sample(initial,seed):
            return tfp.mcmc.sample_chain(num_results=4,current_state=initial,kernel=kernel,
                trace_fn=trace,seed=seed,parallel_iterations=1)
        seed=tf.constant([20260929,71],tf.int32)
        result=sample(initial,seed)
        states,trace_values=result
        assert bool(tf.reduce_all(tf.math.is_finite(states)))
        assert bool(tf.reduce_all(tf.math.is_finite(trace_values[0])))
        assert bool(tf.reduce_all(tf.math.is_finite(trace_values[1])))
        assert bool(tf.reduce_all(trace_values[2]==0))
        assert bool(tf.reduce_all(trace_values[3]))
        assert bool(tf.reduce_all(trace_values[4]==0))
        sample(initial+tf.constant(.00001,tf.float64),seed)
        assert sample.experimental_get_tracing_count()==1
        # Graph and XLA random streams may differ for the same seed. Use
        # identical explicit momenta for a deterministic integrator comparison.
        integrator=SimpleLeapfrogIntegrator(target_fn,[tf.constant(.001,tf.float64)],2)
        @tf.function(input_signature=[tf.TensorSpec([4,target.parameter_dim],tf.float64),
                                      tf.TensorSpec([4,target.parameter_dim],tf.float64)],
                     autograph=False,jit_compile=jit)
        def proposal(state,momentum):
            return integrator([momentum],[state])
        momentum=tf.reshape(tf.linspace(tf.constant(-.4,tf.float64),tf.constant(.5,tf.float64),
                                        4*target.parameter_dim),initial.shape)
        outputs.append(tf.nest.map_structure(lambda t:t.numpy(),proposal(initial,momentum)))
    for graph,xla in zip(tf.nest.flatten(outputs[0]),tf.nest.flatten(outputs[1])):
        np.testing.assert_allclose(xla,graph,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize("name,parameters", [
    ("ssm_lgssm_qr", {}),
    ("ssm_nonlinear", {}),
    pytest.param("ssm_nonlinear", {
        "filter_id": "model-b-svd-cubature-deterministic-loglikelihood",
    }, id="ssm_nonlinear_cubature"),
])
def test_generic_ssm_full_chain_xla_matches_graph_with_explicit_momenta(name, parameters):
    """The generic validation wrappers must compile before receiving tuner authority."""
    target = ValidationTarget(name, parameters=parameters)
    width = target.parameter_dim
    initial = (tf.constant([[.5, -1.5]] * 4, tf.float64) if width == 2 else
               tf.constant([[.65, .4, .75]] * 4, tf.float64))
    momentum = tf.reshape(tf.linspace(tf.constant(-.25, tf.float64),
        tf.constant(.35, tf.float64), 4*width), [4,width])
    outputs = []
    for jit in (False, True):
        target_fn = reviewed_independent_chain_target_fn(target,chain_count=4,parameter_dim=width)
        integrator = SimpleLeapfrogIntegrator(target_fn,[tf.constant(.001,tf.float64)],2)
        @tf.function(input_signature=[tf.TensorSpec([4,width],tf.float64),
                                      tf.TensorSpec([4,width],tf.float64)],
                     autograph=False,jit_compile=jit)
        def proposal(state,momentum):
            return integrator([momentum],[state])
        outputs.append(tf.nest.map_structure(lambda t:t.numpy(),proposal(initial,momentum)))
        assert proposal.experimental_get_tracing_count() == 1
        kernel=tfp.mcmc.HamiltonianMonteCarlo(target_fn,tf.constant(.001,tf.float64),2)
        @tf.function(input_signature=[tf.TensorSpec([4,width],tf.float64),
                                      tf.TensorSpec([2],tf.int32)],
                     autograph=False,jit_compile=jit)
        def sample(state,seed):
            return tfp.mcmc.sample_chain(num_results=2,current_state=state,kernel=kernel,
                trace_fn=lambda _state,result: result.log_accept_ratio,
                seed=seed,parallel_iterations=1)
        states,ratios=sample(initial,tf.constant([20261002,714],tf.int32))
        assert bool(tf.reduce_all(tf.math.is_finite(states)))
        assert bool(tf.reduce_all(tf.math.is_finite(ratios)))
        assert sample.experimental_get_tracing_count()==1
    for graph,xla in zip(tf.nest.flatten(outputs[0]),tf.nest.flatten(outputs[1])):
        np.testing.assert_allclose(xla,graph,rtol=2e-5,atol=2e-5)


@pytest.mark.parametrize("name", list(PROFILES))
def test_campaign_xla_qualification_is_scoped_to_compiled_targets(name):
    from bayesfilter.inference.hmc_candidate_set_public import _preflight_exact_target
    data = None if PROFILES[name].case == "K6" else generate_data(name, (20260926,71))
    for jit in (False, True):
        target = ValidationTarget(name, data=data, jit_compile=jit)
        capability = target.value_score_capability()
        assert capability.full_chain_xla_diagnostic_ready is jit
        assert capability.target_scope == "inference_validation"
        if jit:
            _preflight_exact_target(target, use_xla=True, target_scope="inference_validation")
        else:
            with pytest.raises(ValueError, match="lacks full-chain XLA qualification"):
                _preflight_exact_target(target, use_xla=True, target_scope="inference_validation")


@pytest.mark.parametrize("name", ["ssm_campaign_location", "ssm_campaign_nonlinear"])
def test_public_prepared_tuner_executes_xla_measurement(name):
    from bayesfilter.inference import (
        HMCAcceptancePolicy, HMCCandidateExecutionConfig, HMCControllerConfig,
        PrecomputedMassArtifact, bind_hmc_candidate_set_execution, tune_hmc_kernel,
    )
    from bayesfilter.inference.hmc_candidate_set_execution import _tensor_from_payload
    target = ValidationTarget(name, data=generate_data(name, (20260926,71)), jit_compile=True)
    width = target.parameter_dim
    mass = PrecomputedMassArtifact(position=tf.zeros([width], tf.float64),
        factor=tf.eye(width, dtype=tf.float64), covariance=tf.eye(width, dtype=tf.float64),
        adapter_signature=target.adapter_signature(), position_role="reference_center",
        covariance_source="explicit full-chain compilation diagnostic")
    policy = HMCAcceptancePolicy()
    binding = bind_hmc_candidate_set_execution(adapter=target,
        initial_position=target._ssm.initial_starts(), target_scope="inference_validation",
        target_lineage={"model":name, "data":target.data, "prior":"declared campaign prior"},
        config=HMCCandidateExecutionConfig(measurement_num_results=policy.min_decisions_per_chain,
            verification_num_results=policy.min_decisions_per_chain, num_warmup_steps=0,
            seed=(20260929,71), acceptance_policy=policy, target_status_trace_policy="per_chain_step",
            use_xla=True, chain_mode="batched", reuse_leapfrog_graphs=True),
        source_paths=target._ssm.source_paths(), scope_id="ssm-xla-regression", search_id=name,
        epsilon_domain=(.0001,.01), repair_factor=1.1, max_repairs_per_family=0, mass_artifact=mass)
    run = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
        config=HMCControllerConfig(primary_l_grid=(2,), epsilon_by_l=((2,(.001,)),),
            total_budget_units=4, repair_reserve_units=1, evidence_rungs=(1,)),
        candidate_set_adapter=binding.typed_adapter)
    assert run.result.scope.use_xla
    assert binding._evidence
    # This short, deliberately tiny-epsilon chain is compilation evidence,
    # irrespective of acceptance qualification or posterior convergence.
    for evidence in binding._evidence.values():
        states = _tensor_from_payload(evidence["samples"])
        assert states.shape == (policy.min_decisions_per_chain,4,width)
        assert bool(tf.reduce_all(tf.math.is_finite(states)))
