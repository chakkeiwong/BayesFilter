"""Independent analytic oracles and false-reassurance controls; CPU reference."""
import math

import numpy as np  # Independent reference fixtures only.
import pytest
from scipy import stats
import tensorflow as tf

from bayesfilter.testing.inference_validation.mams_analytic_diagnostics import (
    ExactWhitenedFunnelTarget, analytic_quantities, analytic_reference_report)
from bayesfilter.testing.acceptance_uncertainty_models import ResidualFunnelTarget
from bayesfilter.testing.inference_validation.targets import ValidationTarget
from scripts.validate_mams_adaptation import fixture


@pytest.mark.parametrize("name", ["funnel_noncentered", "residual_funnel"])
def test_same_model_charts_density_and_full_jacobian(name):
    adapter, _, _, metadata = fixture(name, False)
    assert metadata["model_funnel_scale"] == 3.
    z = np.array([[-1.3, -.4, .7], [.7, 1.1, -1.4], [2., -.5, .9]])
    delta = .5*np.tanh(z[:, :1]/2) if name == "residual_funnel" else np.zeros((3, 1))
    v = 3*z[:, :1]
    expected = np.concatenate([v, np.exp(v/2+delta)*z[:, 1:]], axis=-1)
    np.testing.assert_allclose(adapter.to_model(tf.constant(z)), expected, rtol=2e-15)
    centered = stats.norm.logpdf(expected[:, 0], scale=3) + np.sum(
        stats.norm.logpdf(expected[:, 1:], scale=np.exp(v/2)), axis=-1)
    log_j = math.log(3)+v[:, 0]+2*delta[:, 0]
    numerical_log_j = []
    for row in z:
        h = 1e-5  # Centered FD O(h^2); separate from the density oracle.
        columns = [(adapter.to_model(tf.constant(row+h*np.eye(3)[i])).numpy()
                    - adapter.to_model(tf.constant(row-h*np.eye(3)[i])).numpy())/(2*h) for i in range(3)]
        numerical_log_j.append(np.linalg.slogdet(np.stack(columns, axis=1))[1])
    np.testing.assert_allclose(numerical_log_j, log_j, atol=2e-9, rtol=0)
    value = adapter.log_density(tf.constant(z)).numpy()
    offset = 1.5*math.log(2*math.pi) if name == "residual_funnel" else 0.
    np.testing.assert_allclose(value, centered+log_j+offset, atol=1e-13, rtol=1e-13)


def test_exact_and_partial_charts_give_identical_model_quantities():
    base = np.random.default_rng(72).normal(size=(32, 4, 3))
    partial = base.copy()
    partial[..., 1:] *= np.exp(-.5*np.tanh(base[..., :1]/2))
    exact = analytic_quantities("funnel_noncentered", ExactWhitenedFunnelTarget(jit_compile=False), use_xla=True)
    residual = analytic_quantities("residual_funnel", ResidualFunnelTarget(jit_compile=False), use_xla=True)
    left, right = exact.values(base), residual.values(partial)
    for spec in exact.specifications:
        if spec.coordinates == "model":
            np.testing.assert_allclose(left[spec.name], right[spec.name], rtol=1e-14, atol=1e-14)
    assert "model_x1_square" not in exact.precision_names
    assert "model_x1_square" in left
    assert "model_neck_mass" in exact.binary_names
    assert set(exact.posterior_quantities(base)) | set(exact.parameter_names) == set(exact.precision_names)
    exact.values(np.zeros((16, 2, 3)))
    assert exact.values.experimental_get_tracing_count() == 1


def test_student_tail_probability_and_finite_moment_roles():
    for df in (3., 5.):
        target = ValidationTarget("student_t", {"df": df}, jit_compile=False)
        contract = analytic_quantities("student_t", target, use_xla=False)
        specs = {s.name: s for s in contract.specifications}
        assert specs["q0_abs_gt_2"].expectation == pytest.approx(2*stats.t.sf(2, df), rel=1e-13)
        assert specs["q0_square"].expectation == df/(df-2)
        assert ("q0_square" in contract.precision_names) == (df > 4)
        assert "q0_fourth" not in contract.precision_names


def test_historical_unit_scale_chart_cannot_inherit_new_funnel_reference():
    old = ValidationTarget("funnel_noncentered", {"scale": 1.}, jit_compile=False)
    with pytest.raises(ValueError, match="scale-three chart"):
        analytic_quantities("funnel_noncentered", old)
    assert old.adapter_signature() != ExactWhitenedFunnelTarget().adapter_signature()


def test_truncated_tails_cannot_pass_as_precise_zero_probability():
    q = np.random.default_rng(530).standard_t(5, size=(2048, 4, 2))
    contract = analytic_quantities("student_t", ValidationTarget("student_t", {"df": 5.}), use_xla=False)
    report = analytic_reference_report(np.clip(q, -1.9, 1.9), contract, use_xla=False)
    tails = [r for r in report["rows"] if r["name"].endswith("abs_gt_2")]
    assert len(tails) == 2
    assert all(r["mcse"] is None and r["finding"] == "inconclusive" for r in tails)
    assert report["family_size"] == 6  # Unavailable tails remain in denominator.
    assert report["status"] != "no_discrepancy_detected"
    assert not report["accuracy_established"]


def test_wrong_model_chart_is_detected_despite_correct_latent_draws():
    class WrongMap(ExactWhitenedFunnelTarget):
        def to_model(self, q):
            return tf.concat([q[..., :1], tf.exp(q[..., :1]/2)*q[..., 1:]], -1)
    contract = analytic_quantities("funnel_noncentered", WrongMap(), use_xla=False)
    q = np.random.default_rng(540).normal(size=(2048, 4, 3))
    report = analytic_reference_report(q, contract, use_xla=False)
    rows = {r["name"]: r for r in report["rows"]}
    assert rows["model_v_square"]["finding"] == "discrepancy_detected"
    assert report["status"] == "discrepancy_detected"


@pytest.mark.parametrize("samples,reason", [
    (None, "no_retained_draws"), (np.zeros((0, 4, 2)), "insufficient_draws_or_chains"),
    (np.zeros((20, 1, 2)), "insufficient_draws_or_chains"),
    (np.full((20, 4, 2), np.nan), "nonfinite_quantities")])
def test_unavailable_reference_is_explicit(samples, reason):
    contract = analytic_quantities("gaussian", ValidationTarget("gaussian"), use_xla=False)
    report = analytic_reference_report(samples, contract, use_xla=False)
    assert report["reason"] == reason
    assert report["status"] in {"invalid", "inconclusive"}


def test_short_and_constant_chains_preserve_screen_denominator():
    contract = analytic_quantities("gaussian", ValidationTarget("gaussian"), use_xla=False)
    for q in (np.random.default_rng(1).normal(size=(16, 4, 2)), np.ones((1024, 4, 2))):
        report = analytic_reference_report(q, contract, use_xla=False)
        assert report["status"] == "inconclusive"
        assert len(report["rows"]) == report["family_size"] == 4
        assert all(r["mcse"] is None for r in report["rows"])


def test_state_space_has_no_fabricated_analytic_reference():
    # Shape stand-in only; no state-space numerical or accuracy claim.
    class ShapeOnly:
        parameter_dim = 2
    contract = analytic_quantities("LGSSM", ShapeOnly(), use_xla=False)
    report = analytic_reference_report(np.ones((32, 4, 2)), contract, use_xla=False)
    assert report["status"] == "reference_unavailable"
    assert not report["accuracy_established"]


def test_binary_quantity_policy_uses_probability_diagnostics():
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy, assess_posterior
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, HMCPrecisionTarget
    contract = analytic_quantities("student_t", ValidationTarget("student_t", {"df": 5.}), use_xla=False)
    policy = HMCPosteriorAssessmentPolicy(quantities_id=contract.payload()["quantities_id"],
        binary_quantity_names=contract.binary_names,
        precision=HMCPrecisionPolicy(tuple(HMCPrecisionTarget(n, mcse_sd_ratio_max=.1)
            for n in contract.precision_names), method="lugsail", jit_compile=False))
    q = tf.constant(np.random.default_rng(542).standard_t(5, size=(1024, 4, 2)))
    result = assess_posterior(q, contract.parameter_names, policy=policy, stage="retained",
        rhat={"rhat_threshold": 1.01}, quantities_fn=contract.posterior_quantities)
    assert set(result["quantity_names"]) == set(contract.precision_names)
    assert set(result["binary_event_ess"]) == set(contract.binary_names)
    assert len(result["precision"]["targets"]) == len(contract.precision_names)
    # No posterior-pass assertion: this is a callback/quantity-role test.
