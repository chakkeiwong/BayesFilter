"""Independent arithmetic and actual-controller checks for AR(1) diagnostics."""
import math

import pytest
import tensorflow as tf

from bayesfilter.testing.inference_validation.engines.controller_stopping import (
    GaussianAR1Transition, fixed_mean_law, mean_interval,
)


@pytest.mark.parametrize("rho", [0., .8, -.4, .9999])
def test_fixed_mean_law_matches_full_conditional_covariance(rho):
    # Independent dense covariance calculation catches index and pooling errors.
    starts = [-8., 2., 4., 8.]
    times = range(4, 16)
    expected = sum(rho**abs(s-t) - rho**(s+t) for s in times for t in times) / (4 * 12**2)
    law = fixed_mean_law(rho, starts, warmup=3, draws=12)
    assert law["variance"] == pytest.approx(expected, rel=1e-10)
    assert law["mean"] == pytest.approx(sum(x * rho**t for x in starts for t in times) / 48)
    stationary = fixed_mean_law(rho, starts, warmup=3, draws=12, stationary_start=True)
    assert stationary["mean"] == 0
    assert stationary["variance"] == pytest.approx(
        sum(rho**abs(s-t) for s in times for t in times) / 576)


def test_exact_recurrence_matches_independently_expanded_innovations():
    transition = GaussianAR1Transition(.8, jit_compile=False)
    initial = tf.constant([[-8.], [-4.], [4.], [8.]], tf.float64)
    seed = tf.constant([123, 47], tf.int32)
    values = transition(initial, num_results=12, seed=seed, stage="fixed")
    noise = tf.random.stateless_normal((12, 4, 1), seed, dtype=tf.float64)
    expanded = tf.stack([.8**(t+1)*initial + math.sqrt(1-.8**2) * tf.add_n(
        [.8**(t-j)*noise[j] for j in range(t+1)]) for t in range(12)])
    tf.debugging.assert_near(values["posterior_samples"], expanded, atol=1e-12, rtol=1e-12)
    tf.debugging.assert_equal(values["final_transition_state"], values["posterior_samples"][-1])
    assert values["health"]["passed"]
    assert transition.program(12).experimental_get_tracing_count() == 1


def test_actual_controller_preserves_archives_and_excludes_warmup():
    from bayesfilter.inference.neutra_hmc import SequentialExactTransitionConfig, run_sequential_exact_transition
    transition = GaussianAR1Transition(0., jit_compile=False)
    archives = []
    def archive(**kwargs):
        archives.append(kwargs)
        return {"stage": kwargs["stage"]}
    config = SequentialExactTransitionConfig(transition.signature, (23, 51), (411, 31),
        warmup_chunk_results=250, warmup_min_results=500, warmup_check_window_results=500,
        warmup_max_results=1000, retained_chunk_results=500, retained_min_results=500,
        retained_max_results=1000)
    result = run_sequential_exact_transition(transition_program=transition,
        initial_transition_state=tf.zeros((4, 1), tf.float64), posterior_state_fn=lambda x:x,
        parameter_names=("x",), config=config, archive_callback=archive)
    assert result["warmup_passed"] and result["retained_passed"]
    streams = [a["seed"] for a in archives if not a["cumulative"]]
    assert len(streams) == len(set(streams))
    for stage in ("warmup", "retained"):
        values = tf.concat([a["posterior_samples"] for a in archives
                            if a["stage"] == stage and not a["cumulative"]], axis=0)
        tf.debugging.assert_equal(values, result[f"private_{stage}_beta_one"])
    assert result["warmup_excluded_from_posterior"]
    assert result["private_retained_beta_one"].shape[0] == result["retained_results_per_chain"]


def test_missing_interval_is_unavailable():
    row = mean_interval(tf.zeros((0, 4, 1), tf.float64), jit_compile=False)
    assert not row["available"] and not row["covered"]


@pytest.mark.parametrize("method", ["autocorrelation", "lugsail", "batch_means"])
def test_interval_uses_the_declared_precision_estimator(method):
    from bayesfilter.inference.hmc_precision import mean_precision
    values = GaussianAR1Transition(.8, jit_compile=False)(
        tf.zeros((4, 1), tf.float64), num_results=500, seed=(119, 931),
        stage="fixed")["posterior_samples"]
    row = mean_interval(values, jit_compile=False, method=method)
    expected = mean_precision(values, method=method, jit_compile=False)
    assert row["method"] == method + "_normal_95"
    assert row["estimator"] == expected["estimator"]
    assert row["mcse"] == pytest.approx(float(expected["mcse"][0]))
    assert row["interval"][1] - row["estimate"] == pytest.approx(1.959963984540054 * row["mcse"])
    missing = mean_interval(values[:0], jit_compile=False, method=method)
    assert missing["method"] == row["method"] and not missing["available"]


def test_invalid_interval_method_fails_even_when_draws_are_missing():
    with pytest.raises(ValueError, match="estimator"):
        mean_interval(tf.zeros((0, 4, 1), tf.float64), jit_compile=False, method="unknown")


def test_actual_controller_interval_matches_its_autocorrelation_policy():
    from bayesfilter.inference.neutra_hmc import SequentialExactTransitionConfig, run_sequential_exact_transition
    from bayesfilter.inference.hmc_precision import HMCPrecisionPolicy, HMCPrecisionTarget
    from bayesfilter.inference.hmc_posterior_assessment import HMCPosteriorAssessmentPolicy
    transition = GaussianAR1Transition(0., jit_compile=False)
    config = SequentialExactTransitionConfig(transition.signature, (731, 121), (812, 341),
        warmup_chunk_results=250, warmup_min_results=500, warmup_check_window_results=500,
        warmup_max_results=1000, retained_chunk_results=500, retained_min_results=500,
        retained_max_results=1000, assessment_policy=HMCPosteriorAssessmentPolicy(
            precision=HMCPrecisionPolicy((HMCPrecisionTarget('x', mcse_absolute_max=.15),),
                                         method='autocorrelation', jit_compile=False)))
    result = run_sequential_exact_transition(transition_program=transition,
        initial_transition_state=tf.zeros((4, 1), tf.float64), posterior_state_fn=lambda x:x,
        parameter_names=('x',), config=config)
    assert result['retained_checks']
    final = result['retained_checks'][-1]
    actual = final[final['diagnostic_role']]['precision']['targets'][0]
    interval = mean_interval(result['private_retained_beta_one'], method='autocorrelation', jit_compile=False)
    assert interval['estimator'] == actual['estimator']
    assert interval['estimate'] == actual['estimate']
    assert interval['mcse'] == actual['mcse']


@pytest.mark.parametrize("rho", [float("nan"), 1., -1., 1.1])
def test_invalid_transition_is_rejected(rho):
    with pytest.raises(ValueError):
        GaussianAR1Transition(rho)
