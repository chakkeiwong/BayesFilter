"""Independent arithmetic and public-policy checks for the optional MCSE maximum."""
import json
import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_precision import (
    HMCPrecisionPolicy, HMCPrecisionTarget, _batch_maximum_program,
    mean_precision, precision_report,
)
from bayesfilter.inference.hmc_posterior_assessment import (
    HMCPosteriorAssessmentPolicy, assess_posterior,
)
from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary

METHOD = "lugsail_max_batch_means"


def batch_oracle(values, size):
    count = len(values) // size
    means = np.stack([values[i*size:(i+1)*size].mean(axis=0) for i in range(count)])
    return size * means.var(axis=0, ddof=1)


@pytest.mark.parametrize("rho", [0., .9, -.9])
@pytest.mark.parametrize("n,batch", [(240, 20), (247, 21)])
def test_compiled_maximum_matches_independent_per_chain_oracle(rho, n, batch):
    rng = np.random.default_rng(710)
    values = rng.normal(size=(n, 4, 3))
    for i in range(1, n):
        values[i] = rho * values[i-1] + math.sqrt(1-rho*rho) * values[i]
    plain = batch_oracle(values, batch)
    raw = 2*plain - batch_oracle(values, batch//3)
    selected = np.maximum(plain, raw)
    report = mean_precision(values, method=METHOD, batch_size=batch, min_batches=2)
    np.testing.assert_allclose(report["per_chain_long_run_variance"], selected, rtol=2e-12)
    np.testing.assert_allclose(report["per_chain_raw_lugsail_long_run_variance"], raw, atol=2e-12)
    np.testing.assert_array_equal(report["per_chain_batch_maximum_activated"], plain > raw)
    np.testing.assert_allclose(np.square(report["mcse"]), selected.sum(axis=0)/(16*n), rtol=2e-12)
    assert report["unused_terminal_draws"] == n % batch
    assert report["small_unused_terminal_draws"] == n % (batch//3)


def test_maximum_is_applied_before_pooling_and_exposes_negative_component():
    values = np.random.default_rng(712).normal(size=(240, 2, 1))
    # A small between-batch drift makes B positive while negative L survives.
    values[:, 0, 0] = np.tile(np.repeat([1., -1.], 10), 12) + .001*np.arange(240)
    values[:, 1, 0] = np.arange(240)
    original = mean_precision(values, method="lugsail", batch_size=20, min_batches=2)
    guarded = mean_precision(values, method=METHOD, batch_size=20, min_batches=2)
    raw = guarded["per_chain_raw_lugsail_long_run_variance"].numpy()
    plain = guarded["per_chain_plain_long_run_variance"].numpy()
    assert raw[0, 0] < 0 < plain[0, 0]
    assert not bool(original["valid"][0])
    assert bool(guarded["valid"][0])
    expected = np.maximum(plain, raw).sum()/(4*240)
    assert float(guarded["mcse"][0])**2 == pytest.approx(expected)
    assert expected > max(plain.sum(), raw.sum())/(4*240)


@pytest.mark.parametrize("plain,raw", [(float("nan"), 1.), (1., float("nan")),
    (float("inf"), 1.), (1., float("inf")), (1., -float("inf")), (0., 1.), (-1., 2.)])
def test_maximum_never_hides_nonfinite_or_nonpositive_plain(plain, raw):
    maximum, mask = _batch_maximum_program((1, 1), True)(
        tf.constant([[plain]], tf.float64), tf.constant([[raw]], tf.float64))
    assert math.isnan(float(maximum[0, 0]))
    assert not bool(mask[0, 0])


def test_no_change_when_lugsail_dominates_and_equality_does_not_activate():
    plain = tf.constant([[1., 2.]], tf.float64)
    raw = tf.constant([[3., 2.]], tf.float64)
    maximum, mask = _batch_maximum_program((1, 2), True)(plain, raw)
    np.testing.assert_array_equal(maximum, raw)
    assert not bool(tf.reduce_any(mask))


def test_public_policy_preserves_components_and_serializes_activation_booleans():
    values = tf.constant(np.random.default_rng(713).normal(size=(240, 4, 1)))
    policy = HMCPrecisionPolicy((HMCPrecisionTarget("x", mcse_sd_ratio_max=.2),),
        method=METHOD, batch_size=20, min_batches=2)
    assessment = assess_posterior(values, ("x",), stage="retained",
        policy=HMCPosteriorAssessmentPolicy(precision=policy),
        rhat=rank_normalized_split_rhat_summary(values, rhat_max=1.01))
    report = assessment["precision"]
    assert report == precision_report(values, ("x",), policy)
    payload = json.loads(json.dumps(report, allow_nan=False))
    assert payload["policy"]["method"] == METHOD
    metadata = payload["batch_metadata"]
    assert len(metadata["per_chain_raw_lugsail_long_run_variance"]) == 4
    assert all(type(row[0]) is bool for row in metadata["per_chain_batch_maximum_activated"])
    assert HMCPrecisionPolicy(policy.targets).method == "autocorrelation"


def test_underbatched_constant_and_nonfinite_samples_cannot_grant_precision():
    values = tf.ones((100, 4, 1), tf.float64)
    short = mean_precision(values, method=METHOD, batch_size=20)
    assert short["unavailable_reason"] == "insufficient_complete_batches"
    constant = mean_precision(values, method=METHOD, batch_size=10, min_batches=2)
    assert not bool(tf.reduce_any(constant["valid"]))
    invalid = np.ones((100, 4, 1))
    invalid[0, 0, 0] = np.nan
    with pytest.raises(tf.errors.InvalidArgumentError):
        mean_precision(invalid, method=METHOD)


def test_actual_sequential_controller_uses_optional_policy_and_excludes_warmup():
    from bayesfilter.inference.neutra_hmc import (
        SequentialExactTransitionConfig, run_sequential_exact_transition,
    )
    from bayesfilter.testing.inference_validation.engines.controller_stopping import GaussianAR1Transition
    transition = GaussianAR1Transition(0., jit_compile=True)
    precision = HMCPrecisionPolicy((HMCPrecisionTarget("x", mcse_absolute_max=.2),),
                                   method=METHOD, jit_compile=True)
    config = SequentialExactTransitionConfig(transition.signature, (731, 121), (812, 341),
        warmup_chunk_results=250, warmup_min_results=500, warmup_check_window_results=500,
        warmup_max_results=1000, retained_chunk_results=500, retained_min_results=500,
        retained_max_results=1000,
        assessment_policy=HMCPosteriorAssessmentPolicy(precision=precision))
    archives = []

    def archive(**entry):
        archives.append(entry)
        return {"stage": entry["stage"]}

    result = run_sequential_exact_transition(transition_program=transition,
        initial_transition_state=tf.zeros((4, 1), tf.float64), posterior_state_fn=lambda x: x,
        parameter_names=("x",), config=config, archive_callback=archive)
    assert result["warmup_passed"] and result["retained_passed"]
    assert result["warmup_excluded_from_posterior"]
    retained = tf.concat([a["posterior_samples"] for a in archives
                          if a["stage"] == "retained" and not a["cumulative"]], axis=0)
    tf.debugging.assert_equal(retained, result["private_retained_beta_one"])
    expected = precision_report(retained, ("x",), precision)
    last = result["retained_checks"][-1]
    actual = last[last["diagnostic_role"]]["precision"]
    assert actual == expected
    assert actual["policy"]["method"] == METHOD
    assert "per_chain_batch_maximum_activated" in actual["batch_metadata"]
