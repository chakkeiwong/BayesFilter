"""Discarded-pilot mechanics, explicitly not adaptation-quality evidence."""
import math

import pytest

from bayesfilter.inference.hmc_trajectory_proposals import (
    TrajectoryPilotConfig, propose_trajectory_candidates,
)
from bayesfilter.inference.hmc_transition_spec import FrozenTransitionSpec
from tests.test_modern_hmc_pipeline import setup


@pytest.mark.parametrize("criterion", ["snaper","chees_rate"])
def test_fixed_geometry_pilot_proposes_but_cannot_qualify(criterion):
    binding,_=setup()
    config=TrajectoryPilotConfig(criterion=criterion,adaptation_steps=16,
        initial_step_size=.2,target_accept_probability=.8,max_steps=16,initial_mala_steps=0,
        trajectory_learning_rate=.05,principal_component_ema_factor=8,state_ema_factor=8,
        epsilon_factors=(.8,1.),length_factors=(.8,1.),seed=(531,918))
    result=propose_trajectory_candidates(binding,config)
    assert not result["tuning_artifact_authority"]
    assert result["fresh_measurement_and_verification_required"]
    assert not binding._evidence
    assert result["adaptation_draws_discarded"]==17
    assert result["trace"]["metric_scale"]==[1.]*17
    assert result["candidates"]
    for row in result["candidates"]:
        candidate=FrozenTransitionSpec.from_payload(row)
        assert candidate.max_steps<=16
        assert math.isfinite(candidate.epsilon)
        assert candidate.profile.trajectory=="uniform_duration"


def test_mams_cannot_be_mislabeled_as_a_snaper_proposal():
    binding,_=setup(family="mams")
    config=TrajectoryPilotConfig(criterion="snaper",adaptation_steps=2,
        initial_step_size=.2,target_accept_probability=.8,max_steps=8,initial_mala_steps=0,
        trajectory_learning_rate=.05,principal_component_ema_factor=8,state_ema_factor=8,
        epsilon_factors=(1.,),length_factors=(1.,),seed=(51,19))
    with pytest.raises(ValueError,match="randomized-HMC"):
        propose_trajectory_candidates(binding,config)


def test_pilot_hypotheses_need_fresh_shared_controller_evidence(tmp_path):
    from dataclasses import replace
    from bayesfilter.inference import tune_hmc_kernel
    binding,search=setup()
    config=TrajectoryPilotConfig(criterion="snaper",adaptation_steps=16,
        initial_step_size=.2,target_accept_probability=.8,max_steps=16,initial_mala_steps=0,
        trajectory_learning_rate=.05,principal_component_ema_factor=8,state_ema_factor=8,
        epsilon_factors=(1.,),length_factors=(1.,),seed=(531,918))
    proposals=propose_trajectory_candidates(binding,config)
    pairs=proposals["epsilon_by_l"]
    search=replace(search,primary_l_grid=tuple(l for l,_ in pairs),epsilon_by_l=pairs)
    run=tune_hmc_kernel(adapter=binding._base_adapter,initial_position=binding.initial_active_state,
        config=search,candidate_set_adapter=binding.typed_adapter,output_dir=tmp_path)
    assert len(run.result.verified_candidate_ids)==len(proposals["candidates"])
    # A proposal is no substitute for either fresh stage, and its stream is
    # purpose-separated from all of the controller's evidence streams.
    assert {w.stage for w in run.result.work_items}=={"measurement","verification"}
    for evidence in binding._evidence.values():
        if "seed" in evidence:
            assert tuple(evidence["seed"])!=tuple(proposals["derived_seed"])


@pytest.mark.parametrize("criterion",["snaper","chees_rate"])
def test_source_objective_and_frozen_center_derivative_match_independent_formula(criterion):
    import numpy as np  # Independent diagnostic oracle only.
    import tensorflow as tf
    from tensorflow_probability.python.experimental.mcmc import gradient_based_trajectory_length_adaptation as source
    x=np.array([[-2.,.1],[-.5,-1.],[.2,.8],[1.5,2.]])
    y=np.array([[-1.2,-.2],[.1,-.7],[.4,1.4],[1.7,1.]])
    p=np.array([.6,.8]);tau=1.3
    xc=x-x.mean(axis=0);yc=y-y.mean(axis=0)
    # TFP stops gradients through the estimated center, following its
    # population-mean stand-in. This is not the total derivative of a sample
    # objective that continuously re-estimates its mean.
    if criterion=="snaper":
        delta=(yc@p)**2-(xc@p)**2
        expected=delta**2/tau
        expected_gradient=(4*delta*(yc@p)/tau)[:,None]*p
        fn=source.snaper_criterion
        extra={"direction":tf.constant(p,tf.float64)}
    else:
        delta=(yc*yc-xc*xc).sum(axis=-1)
        expected=.25*delta**2/tau
        expected_gradient=delta[:,None]*yc/tau
        fn=source.chees_rate_criterion;extra={}
    yt=tf.constant(y,tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(yt)
        actual=fn(tf.constant(x,tf.float64),yt,tf.ones([4],tf.float64),
            tf.constant(tau,tf.float64),**extra)
        total=tf.reduce_sum(actual)
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(tape.gradient(total,yt),expected_gradient,rtol=2e-14,atol=2e-14)
