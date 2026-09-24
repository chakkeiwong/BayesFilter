"""Independent checks of event-probability information, not binary quantiles."""
from dataclasses import replace
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy, assess_posterior
from bayesfilter.inference.hmc_posterior_diagnostics import _cross_chain_ess, _split_sample_major


def test_explicit_binary_event_ess_matches_direct_indicator_ess():
    draws=tf.random.stateless_normal([2048,4,1],(823,19),dtype=tf.float64)
    event=tf.cast(draws[...,0]>0.,tf.float64)
    policy=HMCPosteriorAssessmentPolicy(retained_bulk_ess_min=100.,retained_tail_ess_min=100.,
        quantities_id="sign-v1")
    args=dict(samples=draws,names=("x",),stage="retained",rhat={"passed":True,"rhat_threshold":1.05},
        quantities_fn=lambda x:{"sign":event})
    untyped=assess_posterior(policy=policy,**args)
    assert not untyped["information_passed"]
    assert untyped["tail_ess"][-1] is None
    typed=assess_posterior(policy=replace(policy,binary_quantity_names=("sign",)),**args)
    direct=_cross_chain_ess(_split_sample_major(tf.transpose(event[...,None],(1,0,2))))
    tf.debugging.assert_near(tf.constant(typed["binary_event_ess"]["sign"],tf.float64),direct[0],rtol=1e-10)
    assert typed["passed"] and typed["information_passed"]
    # This deterministic bank has exactly half ones. Folding at median .5
    # is constant despite healthy event mixing; preserve that raw diagnostic.
    assert float(tf.reduce_mean(event))==.5
    assert typed["modern_rhat"]["folded_rank_normalized_split_rhat"][-1] is None
    assert typed["modern_rhat"]["quantity_rhat_roles"][-1]=="binary_probability_rank_split"
    assert typed["tail_ess"][-1] is None  # Does not relabel event ESS as quantile ESS.
    assert typed["information_tail_floor_role"][-1]=="binary_event_probability_ess"


@pytest.mark.parametrize("value",[0.,1.])
def test_unobserved_binary_outcome_does_not_pass(value):
    draws=tf.random.stateless_normal([256,4,1],(45,81),dtype=tf.float64)
    policy=HMCPosteriorAssessmentPolicy(quantities_id="event-v1",binary_quantity_names=("event",))
    report=assess_posterior(draws,("x",),policy=policy,stage="retained",
        rhat={"passed":True,"rhat_threshold":1.05},
        quantities_fn=lambda x:{"event":tf.fill([256,4],tf.constant(value,tf.float64))})
    assert not report["passed"] and not report["information_passed"]


def test_binary_declaration_is_checked_against_actual_values():
    draws=tf.ones([16,4,1],tf.float64)*.5
    with pytest.raises(ValueError,match="other than 0/1"):
        assess_posterior(draws,("x",),policy=HMCPosteriorAssessmentPolicy(binary_quantity_names=("x",)),
            stage="retained",rhat={"passed":True})


def test_stuck_binary_chains_still_fail_probability_rhat():
    draws=tf.random.stateless_normal([256,4,1],(812,78),dtype=tf.float64)
    events=tf.broadcast_to(tf.constant([[0.,0.,1.,1.]],tf.float64),[256,4])
    policy=HMCPosteriorAssessmentPolicy(quantities_id="event-v1",binary_quantity_names=("event",),
        retained_bulk_ess_min=100.,retained_tail_ess_min=100.)
    report=assess_posterior(draws,("x",),policy=policy,stage="retained",
        rhat={"passed":True,"rhat_threshold":1.05},quantities_fn=lambda x:{"event":events})
    assert not report["passed"]
    assert not report["modern_rhat"]["passed"]
