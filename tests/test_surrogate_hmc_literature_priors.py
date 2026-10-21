"""CPU-only independent density, derivative and consumer-wiring diagnostics."""
import math
from types import SimpleNamespace

import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.testing import surrogate_hmc_literature_priors as prior
from bayesfilter.testing import surrogate_hmc_five_model as run

D = tf.float64
tfd = tfp.distributions
CENTERS = {
    "lgssm": [.25, math.log(.18), math.log(.12), .04],
    "ksc": [1.5, 0.], "predator_prey": [.6, 114., 25., .3, .5, .5],
    "sir_d18": [0.] * 3, "range_bearing": [0.] * 2,
}


def normal(mu=0., sd=1.):
    return tfd.Normal(tf.constant(mu, D), tf.constant(sd, D))


def independent_density(name, q):
    """TFP distributions in physical coordinates plus independent chart AD.

    Autodiff is diagnostic-only here; the HMC prior force uses analytic scores.
    """
    theta, _ = prior.chart(name, q, tf.constant(CENTERS[name], D))
    scales = tf.constant(prior.CHART_SCALES[name], D)
    ig = tfd.InverseGamma(tf.constant(1., D), tf.constant(1., D))
    if name == "predator_prey":
        with tf.GradientTape() as tape:
            tape.watch(q)
            physical, _ = prior.chart(name, q, tf.constant(CENTERS[name], D))
            total = tf.reduce_sum(physical)
        jacobian = tape.gradient(total, q)
        density = tfd.Uniform(tf.constant(prior.PP_LOWER, D), tf.constant(prior.PP_UPPER, D))
        return tf.reduce_sum(density.log_prob(physical) + tf.math.log(jacobian))
    if name == "lgssm":
        phi = .55 * tf.tanh(theta[0])
        law = tfd.TruncatedNormal(tf.constant(0., D), tf.sqrt(tf.constant(.5, D)),
                                 tf.constant(-.55, D), tf.constant(.55, D))
        terms = [law.log_prob(phi) + tf.math.log(.55 * (1.-tf.tanh(theta[0])**2)),
                 ig.log_prob(tf.exp(theta[1])) + theta[1],
                 ig.log_prob(tf.exp(theta[2])) + theta[2], normal().log_prob(theta[3])]
    elif name == "ksc":
        gamma = normal().cdf(theta[0])
        beta_law = tfd.Beta(tf.constant(20., D), tf.constant(1.5, D))
        terms = [beta_law.log_prob((gamma+1.)/2.) - math.log(2.)
                 - tf.math.log(1.-beta_law.cdf(tf.constant(.5, D))) + normal().log_prob(theta[0]),
                 normal(0., math.sqrt(1.25)).log_prob(theta[1])]
    elif name == "sir_d18":
        beta, gamma = (10./9.)*tf.exp(theta[0]), .4*tf.exp(theta[1])
        terms = [normal(2., 1.).log_prob(beta) - normal(2., 1.).log_survival_function(0.) + tf.math.log(beta),
                 normal(.4, .5).log_prob(gamma) - normal(.4, .5).log_survival_function(0.) + tf.math.log(gamma),
                 ig.log_prob(tf.exp(2.*theta[2])) + math.log(2.) + 2.*theta[2]]
    else:
        terms = ig.log_prob(tf.exp(2.*theta)) + math.log(2.) + 2.*theta
    return tf.reduce_sum(terms) + tf.reduce_sum(tf.math.log(scales))


@pytest.mark.parametrize("name", prior.MODELS)
def test_normalized_physical_density_and_analytic_score(name):
    p = len(CENTERS[name])
    center = tf.constant(CENTERS[name], D)
    for offset in (-2., 0., 2.):
        q = tf.linspace(tf.constant(-.25+offset, D), tf.constant(.3+offset, D), p)
        value, score = prior.log_prob_and_score(name, q, center)
        with tf.GradientTape() as tape:
            tape.watch(q)
            independent = independent_density(name, q)
        tf.debugging.assert_near(value, independent, atol=2e-10, rtol=2e-10)
        tf.debugging.assert_near(score, tape.gradient(independent, q), atol=2e-9, rtol=2e-9)
        delta = tf.eye(p, dtype=D) * 1e-5
        plus = prior.log_prob_and_score(name, q[None]+delta, center)[0]
        minus = prior.log_prob_and_score(name, q[None]-delta, center)[0]
        tf.debugging.assert_near(score, (plus-minus)/2e-5, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("name", prior.MODELS)
def test_prior_and_chart_cpu_xla_batch_parity(name):
    p = len(CENTERS[name])
    center = tf.constant(CENTERS[name], D)
    @tf.function(input_signature=[tf.TensorSpec([None, p], D)], jit_compile=True, autograph=False)
    def compiled(q):
        value, score = prior.log_prob_and_score(name, q, center)
        theta, jac = prior.chart(name, q, center)
        return value, score, theta, jac
    rows = tf.constant([[offset]*p for offset in (-2., -.5, 0., .5, 2.)], D)
    actual = compiled(rows)
    for i, row in enumerate(rows):
        value, score = prior.log_prob_and_score(name, row, center)
        tf.debugging.assert_near(actual[0][i], value, atol=1e-10, rtol=1e-10)
        tf.debugging.assert_near(actual[1][i], score, atol=1e-10, rtol=1e-10)
    assert all(bool(tf.reduce_all(tf.math.is_finite(x))) for x in actual)
    compiled(rows[:2])
    assert compiled.experimental_get_tracing_count() == 1


def test_ksc_and_predator_transformed_tails_do_not_use_rounded_probabilities():
    for name in ("ksc", "predator_prey"):
        p = len(CENTERS[name])
        q = tf.constant([[-40.]*p, [40.]*p], D)
        value, score = prior.log_prob_and_score(name, q, tf.constant(CENTERS[name], D))
        assert bool(tf.reduce_all(tf.math.is_finite(value)))
        assert bool(tf.reduce_all(tf.math.is_finite(score)))


def make_case(name, tmp_path):
    data = tmp_path/(name+".json")
    data.write_text("test fixture")
    p = len(CENTERS[name])
    spec = SimpleNamespace(parameter_count=p, dimension=1, target_id="prior_wiring_fixture")
    return run.configure_prior(run.Case(name, spec, tf.zeros([1, 1], tf.float32),
        tf.constant(CENTERS[name], tf.float32), (), run.ProposalControls(),
        tf.constant([1/3]*3, tf.float32), data), prior.PRIOR_ID)


@pytest.mark.parametrize("name", prior.MODELS)
def test_actual_hmc_consumers_use_same_declared_prior(name, monkeypatch, tmp_path):
    case = make_case(name, tmp_path)
    # Deliberately different force likelihood and endpoint likelihood: the
    # shared term must be the prior, not the surrogate filter's reported value.
    def program(theta, *unused):
        return dict(score=-theta, value=tf.fill([2], tf.constant(987., tf.float32)),
                    valid=tf.ones([2], tf.bool), steps_completed=tf.ones([2], tf.int32))
    monkeypatch.setattr(run, "make_batched_score_filter", lambda *a, **k: program)
    monkeypatch.setattr(run, "reference", lambda case: (
        lambda theta: (-2.*tf.reduce_sum(theta**2), tf.constant(True)), "independent_test_reference"))
    bound = run.bindings(case)
    q = tf.constant([[s]*case.p for s in (-.4, 0., .4)], D)
    logp, score = case.log_prior(q)
    theta32, jac32 = case.chart(tf.cast(q, tf.float32))
    expected_force = -score + tf.cast(jac32*theta32, D)
    actual = bound["force"].function(q)
    tf.debugging.assert_near(actual.force, expected_force, atol=2e-4, rtol=2e-6)
    assert bool(tf.reduce_all(actual.force_valid))
    for i in range(3):
        scalar, valid = bound["scalar_force"](q[i])
        tf.debugging.assert_near(scalar, actual.force[i], atol=2e-4, rtol=2e-6)
        assert bool(valid)
    theta64, _ = case.chart(q)
    target = bound["target"].function(q)
    tf.debugging.assert_near(target.potential, 2.*tf.reduce_sum(theta64**2, axis=-1)-logp,
                             atol=1e-8, rtol=1e-12)
    benchmark = run.configure_prior(case, prior.BENCHMARK_PRIOR_ID)
    assert benchmark.scope()["prior"] != case.scope()["prior"]
    record = dict(status="FILTER_SCOPE_SELECTED_FOR_EXPLORATORY_HMC", scope=benchmark.scope(),
        selected=dict(valid=True, controls=run.asdict(case.controls), beta=[1/3]*3))
    record["selection_hash"] = run.hashed(record)
    selection = tmp_path/"old-selection.json"
    run.save(selection, record)
    with pytest.raises(ValueError, match="stale or mismatched"):
        run.load_selection(case, selection)


def test_master_admits_only_literature_prior_for_scientific_inference():
    from scripts import run_surrogate_hmc_five_model_master as master
    master.validate_stage_prior("inference", prior.PRIOR_ID)
    master.validate_stage_prior("mechanics", prior.PRIOR_ID)
    master.validate_stage_prior("mechanics", prior.BENCHMARK_PRIOR_ID)
    for choice in (None, prior.BENCHMARK_PRIOR_ID, "unknown"):
        with pytest.raises(ValueError):
            master.validate_stage_prior("inference", choice)


def test_predator_chart_rejects_invalid_center(tmp_path):
    case = make_case("predator_prey", tmp_path)
    case.theta0 = tf.constant([.6, 100., 25., .3, .5, .5], D)
    with pytest.raises(ValueError, match="inside the published prior box"):
        run.configure_prior(case, prior.PRIOR_ID)
