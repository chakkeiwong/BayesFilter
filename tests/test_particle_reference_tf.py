"""Small verification oracles for the value-only SIR reference; CPU-only."""
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from bayesfilter.nonlinear.particle_reference_tf import particle_reference

tfd = tfp.distributions
D = tf.float64


class Fixed:
    """Deterministic draw/log-density fixture, not a runtime distribution."""
    def __init__(self, draws, log_probs):
        self.draws = tf.constant(draws, D)
        self.log_probs = tf.constant(log_probs, D)

    def sample(self, num=None, seed=None):
        del num, seed
        return self.draws

    def log_prob(self, points):
        del points
        return self.log_probs


def call(prior, observation, *, dates=1, batch=1, transition=None, **kwargs):
    if transition is None:
        def transition(*_):
            raise AssertionError("one date must not consume a transition")
    return particle_reference(tf.zeros([dates, batch, 1], D), prior, transition, observation,
                              num_particles=2, seed=[20260919, 101], **kwargs)


def test_initial_proposal_keeps_likelihood_normalizing_constant():
    prior = Fixed([[[1.]], [[2.]]], np.log([[.8], [.2]]))
    proposal = Fixed([[[1.]], [[2.]]], np.log([[.5], [.5]]))
    obs = lambda t, x: Fixed([], np.log([[.25], [.75]]))
    expected = .5 * (1.6 * .25 + .4 * .75)
    result = call(prior, obs, initial_proposal=proposal, resample_ess_fraction=0)
    np.testing.assert_allclose(np.exp(result["log_likelihood"]), [expected], atol=1e-14)
    # A second unbalanced proposal sample is what distinguishes ordinary IS
    # from self-normalized IS. Here its average p/q is 1.6, not 1.
    prior = Fixed([[[1.]], [[1.]]], np.log([[.8], [.8]]))
    proposal = Fixed([[[1.]], [[1.]]], np.log([[.5], [.5]]))
    result = call(prior, obs, initial_proposal=proposal, resample_ess_fraction=0)
    np.testing.assert_allclose(np.exp(result["log_likelihood"]), [.8], atol=1e-14)


def test_transition_proposal_weights_and_no_resampling_carry():
    prior = Fixed([[[0.]], [[1.]]], [[0.], [0.]])
    transition = lambda t, x: Fixed([[[2.]], [[3.]]], np.log([[.8], [.8]]))
    proposal = lambda t, x: Fixed([[[2.]], [[3.]]], np.log([[.5], [.5]]))
    observation = lambda t, x: Fixed([], np.log([[.25], [.75]]))
    result = call(prior, observation, dates=2, transition=transition,
                  proposal_fn=proposal, resample_ess_fraction=0)
    # Initial .5; next weighted sum .25*(1.6*.25)+.75*(1.6*.75)=1.
    np.testing.assert_allclose(np.exp(result["log_likelihood"]), [.5], atol=1e-14)
    np.testing.assert_array_equal(result["history"]["resampled"], [[False], [False]])


def test_zero_first_weight_is_valid_and_all_zero_batch_is_rejected():
    prior = Fixed([[[0.], [0.]], [[1.], [1.]]], np.zeros((2, 2)))
    observation = lambda t, x: Fixed([], [[-np.inf, -np.inf], [0., -np.inf]])
    result = call(prior, observation, batch=2)
    np.testing.assert_array_equal(result["valid"], [True, False])
    np.testing.assert_allclose(result["log_likelihood"][0], np.log(.5))
    assert np.isneginf(result["log_likelihood"][1])
    np.testing.assert_array_equal(result["failure_date"], [-1, 0])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_weights_rejected(bad):
    prior = Fixed([[[0.]], [[1.]]], [[0.], [0.]])
    result = call(prior, lambda t, x: Fixed([], [[bad], [0.]]))
    assert not bool(result["valid"][0])


def test_initial_gamma_law_and_observation0_timing():
    prior = tfd.Independent(tfd.Gamma(tf.constant([[2.]], D), tf.constant([[3.]], D)), 1)
    # All observations have the same likelihood, so weighted initial moments
    # must reflect Gamma(2,3), not a transition or a Gaussian initial surrogate.
    observation = lambda t, x: tfd.Independent(tfd.Normal(tf.zeros_like(x), tf.ones_like(x)), 1)
    result = particle_reference(tf.zeros([1, 1, 1], D), prior,
        lambda *_: (_ for _ in ()).throw(AssertionError("unexpected transition")),
        observation, num_particles=32768, seed=[20260919, 111])
    np.testing.assert_allclose(result["history"]["mean"][0, 0, 0], 2/3, atol=.02)
    np.testing.assert_allclose(result["history"]["covariance"][0, 0, 0, 0], 2/9, atol=.015)
    assert np.all(result["particles"].numpy() > 0)


def gaussian_filter(seed, *, count=8192):
    y = tf.constant([.1, -.3, .2, .5], D)[:, None, None]
    prior = tfd.MultivariateNormalDiag(tf.zeros([1, 1], D), tf.ones([1, 1], D))
    transition = lambda t, x: tfd.MultivariateNormalDiag(.8*x, .3*tf.ones_like(x))
    observation = lambda t, x: tfd.MultivariateNormalDiag(x, .5*tf.ones_like(x))
    return particle_reference(y, prior, transition, observation, num_particles=count, seed=seed)


def test_gaussian_kalman_oracle_and_seed_replay():
    result = gaussian_filter(tf.constant([20260919, 112], tf.int32))
    repeated = gaussian_filter(tf.constant([20260919, 112], tf.int32))
    for a, b in zip(tf.nest.flatten(result), tf.nest.flatten(repeated), strict=True):
        np.testing.assert_array_equal(a, b)
    mean, variance, loglik = 0., 1., 0.
    means, variances = [], []
    for t, y in enumerate([.1, -.3, .2, .5]):
        if t:
            mean, variance = .8*mean, .64*variance+.09
        innovation = variance+.25
        loglik += -.5*(np.log(2*np.pi*innovation)+(y-mean)**2/innovation)
        mean += variance/innovation*(y-mean)
        variance *= .25/innovation
        means.append(mean)
        variances.append(variance)
    np.testing.assert_allclose(result["log_likelihood"], [loglik], atol=.10)
    np.testing.assert_allclose(result["history"]["mean"][:, 0, 0], means, atol=.03)
    np.testing.assert_allclose(result["history"]["covariance"][:, 0, 0, 0], variances, atol=.02)


def test_full_xla_replay_and_no_callbacks():
    compiled = tf.function(lambda seed: gaussian_filter(seed), autograph=False, jit_compile=True,
                          input_signature=[tf.TensorSpec([2], tf.int32)])
    seed = tf.constant([20260919, 113], tf.int32)
    result, repeated = compiled(seed), compiled(seed)
    for a, b in zip(tf.nest.flatten(result), tf.nest.flatten(repeated), strict=True):
        np.testing.assert_array_equal(a, b)
    assert bool(tf.reduce_all(result["valid"]))
    assert compiled.experimental_get_tracing_count() == 1
    graph = compiled.get_concrete_function().graph.as_graph_def()
    nodes = list(graph.node) + [n for f in graph.library.function for n in f.node_def]
    assert not any("PyFunc" in n.op for n in nodes)
    assert "HloModule" in compiled.experimental_get_compiler_ir(seed)(stage="hlo")


def test_resampling_records_the_weighted_cloud_and_parent_count():
    prior = Fixed([[[0.]], [[1.]]], [[0.], [0.]])
    result = call(prior, lambda t, x: Fixed([], [[-np.inf], [0.]]), resample_ess_fraction=1)
    np.testing.assert_array_equal(result["history"]["resampled"], [[True]])
    np.testing.assert_array_equal(result["history"]["unique_parent_count"], [[1]])
    np.testing.assert_array_equal(result["particles"], [[[1.]], [[1.]]])
    np.testing.assert_allclose(result["log_weights"], -np.log(2))
    np.testing.assert_array_equal(result["history"]["mean"], [[[1.]]])


def test_deterministic_proposal_path_plain_xla_parity():
    def evaluate(seed):
        prior = Fixed([[[1.]], [[1.]]], np.log([[.8], [.8]]))
        proposal = Fixed([[[1.]], [[1.]]], np.log([[.5], [.5]]))
        observation = lambda t, x: Fixed([], np.log([[.25], [.75]]))
        return particle_reference(tf.zeros([2, 1, 1], D), prior, lambda t, x: prior,
            observation, num_particles=2, seed=seed, initial_proposal=proposal,
            proposal_fn=lambda t, x: proposal, resample_ess_fraction=0)
    seed = tf.constant([20260919, 114], tf.int32)
    plain = tf.function(evaluate, autograph=False)(seed)
    compiled = tf.function(evaluate, autograph=False, jit_compile=True)(seed)
    for a, b in zip(tf.nest.flatten(plain), tf.nest.flatten(compiled), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-13)


@pytest.mark.parametrize("xla", [False, True])
def test_only_selected_batch_row_resamples_with_unequal_particle_count(xla):
    # N!=B is essential: N=B can silently apply the batch mask to particles.
    # Batch0 has a point mass at0; batch1's three particles keep equal weights.
    prior = Fixed([[[0.],[10.]],[[1.],[11.]],[[2.],[12.]]], np.zeros((3,2)))
    observation = lambda t,x:Fixed([],[[0.,0.],[-np.inf,0.],[-np.inf,0.]])

    def evaluate():
        return particle_reference(tf.zeros([1,2,1],D),prior,lambda t,x:prior,
            observation,num_particles=3,seed=[20260919,115],resample_ess_fraction=.9)

    result = tf.function(evaluate,autograph=False,jit_compile=xla)()
    np.testing.assert_array_equal(result["history"]["resampled"],[[True,False]])
    np.testing.assert_array_equal(result["history"]["unique_parent_count"],[[1,3]])
    np.testing.assert_array_equal(result["particles"][:,:,0],[[0.,10.],[0.,11.],[0.,12.]])
    np.testing.assert_allclose(result["log_likelihood"],[np.log(1/3),0.],atol=1e-14)
