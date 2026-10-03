"""Real TFP HMC -> shared health evaluator -> experimental uncertainty path."""
import json
import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing.acceptance_uncertainty_models import (
    MODEL_CASES, ResidualFunnelTarget, fixed_model_trace,
)
from bayesfilter.testing.acceptance_uncertainty_validation import experimental_policy
from bayesfilter.inference.hmc_acceptance_uncertainty import evaluate_acceptance_uncertainty
from bayesfilter.inference.hmc_verification import HMCAcceptancePolicy, evaluate_hmc_acceptance_evidence


@pytest.mark.parametrize("name,epsilon", MODEL_CASES)
def test_actual_model_trace_preserves_admission_and_checks_uncertainty(name, epsilon):
    trace = fixed_model_trace(name, epsilon)
    kwargs = dict(samples=trace["states"], log_accept_ratio=trace["log_accept_ratio"],
                  is_accepted=trace["is_accepted"])
    before = evaluate_hmc_acceptance_evidence(**kwargs, policy=HMCAcceptancePolicy()).payload()
    report = evaluate_acceptance_uncertainty(**kwargs, uncertainty_policy=experimental_policy(512), fixed_kernel=True)
    assert report["legacy_evidence"] == before
    assert not report["tuning_artifact_authority"]
    assert report["uncertainty"]["draws_per_chain"] == 512
    assert report["uncertainty"]["experimental_decision"] == "insufficient_information"
    assert np.isfinite(report["uncertainty"]["pooled_mean"])
    # Finite high acceptance does not itself qualify an SSM or a funnel.
    assert json.loads(json.dumps(report, allow_nan=False))["historical_decision_changed"] is False


def test_partial_funnel_density_includes_full_change_of_variables():
    target = ResidualFunnelTarget()
    q = tf.constant([[-2., .5, .1], [.2, -1., .7], [1.5, .4, -.3]], tf.float64)
    v = 3*q[:, 0]
    delta = .5*tf.tanh(q[:, 0]/2)
    x = tf.exp(v[:, None]/2+delta[:, None])*q[:, 1:]
    # Independent centered-density calculation, including triangular Jacobian.
    log_density = -.5*(v/3)**2 - tf.math.log(tf.constant(3., tf.float64))
    log_density += tf.reduce_sum(-.5*(x/tf.exp(v[:, None]/2))**2-v[:, None]/2, axis=1)
    log_jacobian = tf.math.log(tf.constant(3., tf.float64))+v+2*delta
    tf.debugging.assert_near(target.log_density(q), log_density+log_jacobian, atol=1e-12)
