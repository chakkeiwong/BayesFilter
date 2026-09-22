"""Tiny graph/health qualification; CPU deliberately hidden from accelerators."""
import json

import pytest
import tensorflow as tf

from tests.test_q20_master_integration import protocol
from tests.test_q20_production_repair import four_dimensional_bridge
from tests.test_hmc_candidate_set_execution import GaussianTarget, make_binding, execution_config
from bayesfilter.inference.fixed_transport_hmc_mechanics_tf import build_fixed_transport_one_step_transition
from bayesfilter.inference.q20_hmc_qualification import qualify_bridge, attach_qualification, QUALIFICATION_SCHEMA
from bayesfilter.inference.q20_production_config import digest


def test_finite_large_energy_is_reporting_only_and_bad_status_is_veto():
    target=GaussianTarget()
    primitive=build_fixed_transport_one_step_transition(target,state_shape=(4,2),step_size=50.,
        num_leapfrog_steps=3,use_xla=False,capture_health=True)
    result=primitive(tf.ones([4,2],tf.float64),tf.constant([45,16],tf.int32))
    assert bool(result[5])
    assert bool(tf.reduce_any(result[2] < -1000))
    class BadStatus(GaussianTarget):
        def target_status_telemetry(self,theta):
            result=super().target_status_telemetry(theta)
            result["valid_pre_regularized_score"]=tf.zeros(tf.shape(theta)[:-1],tf.bool)
            return result
    bad=build_fixed_transport_one_step_transition(BadStatus(),state_shape=(4,2),step_size=.1,
        num_leapfrog_steps=3,use_xla=False,capture_health=True)
    assert not bool(bad(tf.ones([4,2],tf.float64),tf.constant([45,16],tf.int32))[5])


def test_new_starts_preserve_geometry_domain_and_round_trip():
    from bayesfilter.inference.hmc_candidate_set_execution import bind_hmc_candidate_set_execution_new_starts
    from tests.test_hmc_candidate_set_execution import mass_for
    target=GaussianTarget()
    binding=make_binding(target=target,mass_artifact=mass_for(target,factor=[[2.,.3],[0.,.5]],center=[1.,-2.]),
                         epsilon_domain=(.01,.8))
    starts=tf.constant([[-3.,1.],[2.,3.],[4.,2.],[0.,-1.]],tf.float64)
    fresh=bind_hmc_candidate_set_execution_new_starts(binding=binding,initial_position=starts,
        config=execution_config(seed=(23,7)),scope_id="fresh",search_id="fresh")
    assert fresh.scope.epsilon_domain==binding.scope.epsilon_domain
    assert fresh._spec["layers"]==binding._spec["layers"]
    assert fresh.binding_hash!=binding.binding_hash
    tf.debugging.assert_near(fresh.position_samples(fresh.initial_active_state),starts,atol=1e-12)
    @tf.function(input_signature=(tf.TensorSpec([4,2],tf.float64),),jit_compile=False)
    def round_trip(x):
        return fresh.position_samples(fresh.active_positions(x))
    tf.debugging.assert_near(round_trip(starts),starts,atol=1e-12)


def test_enclosing_xla_and_actual_batched_runner_qualification(tmp_path):
    config=protocol()
    config["jit_compile"]=True
    config["training"]["betas"]=[0.,1.]
    bridge=four_dimensional_bridge(True)
    # Match the real q20 target's one-sided optional diagnostic inventory.
    original=bridge.component_target.batch_prior_likelihood_value_score_status
    def partial_conditioning(theta):
        a,b,c,d,status=original(theta)
        return a,b,c,d,{**status,"min_innovation_eigenvalue":tf.ones(tf.shape(theta)[:-1],tf.float64)}
    bridge.component_target.batch_prior_likelihood_value_score_status=partial_conditioning
    raw=bridge.fixed_beta_adapter(1.).log_prob_and_grad_status(tf.zeros([4,4],tf.float64))[2]
    assert "min_innovation_eigenvalue" in raw
    telemetry=bridge.fixed_beta_adapter(1.).target_status_telemetry(tf.zeros([4,4],tf.float64))
    assert "min_innovation_eigenvalue" not in telemetry
    assert "innovation_condition_estimate" not in telemetry
    qualify_bridge(config,bridge,tmp_path/"qualification")
    path=tmp_path/"qualification/result.json"
    receipt=json.loads(path.read_text())
    assert receipt["schema"] == QUALIFICATION_SCHEMA
    assert receipt["betas"]["1.0"]["public_batched_runner_passed"]
    assert receipt["betas"]["1.0"]["public_state_shape"] == [4, 4]
    assert receipt["betas"]["1.0"]["public_traces"] == 1
    qualified=attach_qualification(bridge,path,config)
    assert qualified.fixed_beta_adapter(1.).value_score_capability().full_chain_xla_diagnostic_ready
    with pytest.raises(ValueError,match="unqualified"):
        qualified.fixed_beta_adapter(.25)
    receipt.pop("checksum")
    receipt["schema"] = "bayesfilter.q20.bridge_hmc_qualification.v1"
    path.write_text(json.dumps({**receipt, "checksum": digest(receipt)}))
    with pytest.raises(ValueError, match="batched public-runner evidence"):
        attach_qualification(bridge, path, config)
