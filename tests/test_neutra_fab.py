"""CPU-hidden independent mechanics/reference checks, not training evidence."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

from dataclasses import replace
import json
import math
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_transport import NeuTraTransport, NeuTraTransportConfig
from bayesfilter.inference.neutra_fab import (
    FABConfig, FABTrainer, fab_weighted_loss, intermediate_log_prob, replay_log_correction,
)


def config(**kwargs):
    base = FABConfig(16, 3, 3, 1, .2, .001, .9, .999, 1.e-8,
        0, 0, 1, 10., None, False, .65, 1.02, jit_compile=False)
    return replace(base, **kwargs)


def flow():
    return NeuTraTransport(NeuTraTransportConfig.hoffman_author_iaf(
        2, conditional_scale_cap=2., seed=(12, 31), dtype="float64"))


def normal(x):
    return -.5 * tf.reduce_sum(x*x, -1) - math.log(2*math.pi), -x, tf.reduce_all(tf.math.is_finite(x), -1)


def test_bridge_and_replay_algebra():
    q, p = tf.constant([-3., -5.], tf.float64), tf.constant([-1., -2.], tf.float64)
    tf.debugging.assert_near(intermediate_log_prob(q, p, 0.), q)
    tf.debugging.assert_near(intermediate_log_prob(q, p, .5), p)
    tf.debugging.assert_near(intermediate_log_prob(q, p, 1.), 2*p-q)
    q2 = q + 2
    q3 = q - .25
    tf.debugging.assert_near(replay_log_correction(q, q2) + replay_log_correction(q2, q3), q-q3)


def test_paper_loss_has_exact_detached_score_gradient_and_scale():
    mu = tf.Variable(.3, dtype=tf.float64)
    x = tf.constant([-1., 0., 1., 2.], tf.float64)
    with tf.GradientTape(persistent=True) as tape:
        log_q = -.5 * (x-mu)**2
        dependent_weights = mu*x
        loss = fab_weighted_loss(log_q, dependent_weights)
        corr = replay_log_correction(tf.zeros_like(log_q), log_q)
    expected = -tf.reduce_sum(tf.nn.softmax(dependent_weights) * (x-mu))
    tf.debugging.assert_near(tape.gradient(loss, mu), expected, atol=1.e-12)
    assert tape.gradient(corr, mu) is None


@pytest.mark.parametrize("transition", ["hmc", "metropolis"])
def test_ais_exact_constant_weights_for_matching_normal(transition):
    transport = flow()
    for v in transport.trainable_variables:
        v.assign(tf.zeros_like(v))
    trainer = FABTrainer(transport, normal, config(transition_operator=transition), target_signature="a"*64, seed=(4, 5))
    result = trainer.step(train=False)
    tf.debugging.assert_near(result["log_w"], tf.zeros([16], tf.float64), atol=1.e-12)
    assert float(result["ess_fraction"]) == pytest.approx(1., abs=1.e-12)
    assert bool(result["valid"])
    assert float(tf.reduce_min(result["movement"])) > 0.


def test_density_parameter_gradient_through_inverse_matches_finite_difference():
    transport = flow()
    x = tf.constant([[.2, -.4], [1., .8], [-.3, -.2]], tf.float64)
    var = transport.trainable_variables[-1]
    with tf.GradientTape() as tape:
        objective = tf.reduce_sum(transport.log_prob(x))
    derivative = tape.gradient(objective, var)
    initial = tf.identity(var)
    direction = tf.ones_like(var)
    h = 1.e-5
    var.assign(initial + h * direction)
    plus = tf.reduce_sum(transport.log_prob(x))
    var.assign(initial - h * direction)
    minus = tf.reduce_sum(transport.log_prob(x))
    var.assign(initial)
    tf.debugging.assert_near(tf.reduce_sum(derivative), (plus-minus)/(2*h), atol=1.e-7)


def test_ais_identity_transition_weights_match_shifted_gaussian_formula():
    transport = flow()
    for v in transport.trainable_variables:
        v.assign(tf.zeros_like(v))
    mu = tf.constant([.4, -.2], tf.float64)
    def shifted(x):
        value, score, valid = normal(x-mu)
        return value, score, valid
    trainer = FABTrainer(transport, shifted, config(initial_step_size=1.e-30),
        target_signature="c"*64, seed=(4, 5))
    x = tf.reshape(tf.linspace(tf.constant(-2., tf.float64), tf.constant(2., tf.float64), 32), [16, 2])
    result = trainer.ais(x, tf.constant([9, 4]), trainer.steps)
    expected = 2*tf.reduce_sum(x*mu, -1) - tf.reduce_sum(mu*mu)
    tf.debugging.assert_near(result["log_w"], expected, atol=1.e-12)


@pytest.mark.parametrize("transition", ["hmc", "metropolis"])
def test_fixed_kernel_ais_recovers_shifted_normal_normalizer_and_moment(transition):
    transport = NeuTraTransport(replace(flow().config, hidden_layers=(4, 4)))
    for var in transport.trainable_variables:
        var.assign(tf.zeros_like(var))
    mu = tf.constant([.3, -.2], tf.float64)
    trainer = FABTrainer(transport, lambda x: normal(x-mu),
        config(batch_size=4096, transition_operator=transition),
        target_signature="a"*64, seed=(41, 73))
    result = trainer.step(train=False)
    # Independent Gaussian integration: integral p^2/q = exp(||mu||^2),
    # and the normalized auxiliary law is N(2*mu, I). Fixed kernels only.
    weights = tf.exp(result["log_w"])
    exact_z = tf.exp(tf.reduce_sum(mu*mu))
    for values, expectation in [
        (weights, exact_z),
        (weights * result["x"][:, 0], exact_z * 2*mu[0]),
        (weights * result["x"][:, 1], exact_z * 2*mu[1]),
    ]:
        mean = tf.reduce_mean(values)
        se = tf.math.reduce_std(values) / tf.sqrt(tf.constant(4095., tf.float64))
        assert float(tf.abs(mean-expectation)) < 5*float(se)


def test_replay_distinct_indices_correction_and_resume():
    cfg = config(replay_capacity=128, replay_min_size=64, updates_per_pass=2)
    trainer = FABTrainer(flow(), normal, cfg, target_signature="b"*64, seed=(9, 8))
    for _ in range(5):  # Author fills floor(64/16)+1 batches, not four.
        assert not trainer.step()["updates"]
    selected = trainer.replay_sample(trainer.replay, tf.constant([6, 3]))
    assert int(tf.size(tf.unique(selected).y)) == 32
    assert int(tf.reduce_max(selected)) < 80
    before = json.loads(json.dumps(trainer.checkpoint(), allow_nan=False))
    a = trainer.step()
    assert len(a["updates"]) == 2
    fresh = FABTrainer(flow(), normal, cfg, target_signature="b"*64, seed=(9, 8))
    fresh.restore(before)
    b = fresh.step()
    tf.debugging.assert_near(a["x"], b["x"], atol=1.e-12)
    tf.debugging.assert_near(trainer.replay.log_q_old, fresh.replay.log_q_old, atol=1.e-12)
    for x, y in zip(trainer.variables, fresh.variables):
        tf.debugging.assert_near(x, y, atol=1.e-12)
    assert trainer.checkpoint()["checkpoint_hash"] == fresh.checkpoint()["checkpoint_hash"]


def test_nonfinite_initial_target_veto_preserves_map_and_optimizer():
    def invalid(x):
        value, score, valid = normal(x)
        return value, score, tf.zeros_like(valid)
    trainer = FABTrainer(flow(), invalid, config(), target_signature="a"*64, seed=(4, 5))
    before = trainer.checkpoint()
    with pytest.raises(ValueError, match="initial q draws"):
        trainer.step()
    assert before == trainer.checkpoint()


def test_replay_correction_clips_loss_but_preserves_uncapped_priority_adjustment():
    trainer = FABTrainer(flow(), normal, config(correction_clip=2.), target_signature="a"*64, seed=(1, 2))
    x = tf.ones([16, 2], tf.float64)
    old_q = trainer.transport.log_prob(x) + math.log(8.)
    before_q = trainer.transport.log_prob(x)
    result = trainer.update(x, tf.zeros([16], tf.float64), old_q, tf.constant(True))
    assert bool(result["valid"])
    assert float(result["correction_clipped_fraction"]) == 1.
    tf.debugging.assert_near(result["loss"], -2*tf.reduce_mean(before_q))
    tf.debugging.assert_near(result["log_w_adjustment"], tf.fill([16], tf.constant(math.log(8.), tf.float64)))


@pytest.mark.parametrize("transition", ["hmc", "metropolis"])
def test_xla_matches_graph_for_fixed_inputs_and_seed(transition):
    graph = FABTrainer(flow(), normal, config(transition_operator=transition), target_signature="a"*64, seed=(1, 2))
    xla = FABTrainer(flow(), normal, config(jit_compile=True, transition_operator=transition), target_signature="a"*64, seed=(1, 2))
    x = tf.reshape(tf.linspace(tf.constant(-1., tf.float64), tf.constant(1., tf.float64), 32), [16, 2])
    # RNG implementations can differ by backend. Compare deterministic update
    # mathematics on identical samples and weights; test compiled AIS separately.
    w = tf.linspace(tf.constant(-2., tf.float64), tf.constant(0., tf.float64), 16)
    a = graph.update(x, w, graph.transport.log_prob(x), tf.constant(False))
    b = xla.update(x, w, xla.transport.log_prob(x), tf.constant(False))
    tf.debugging.assert_near(a["loss"], b["loss"], atol=1.e-10)
    for u, v in zip(graph.variables, xla.variables):
        tf.debugging.assert_near(u, v, atol=1.e-10)
    assert bool(xla.step(train=False)["valid"])


def test_metropolis_adapts_once_per_temperature_after_all_mutations():
    cfg = config(transition_operator="metropolis", hmc_steps=4, adapt_step_size=True)
    trainer = FABTrainer(flow(), normal, cfg, target_signature="a"*64, seed=(4, 5))
    result = trainer.step(train=False)
    expected = tf.constant(cfg.initial_step_size, tf.float64) * tf.where(
        result["acceptance"] > cfg.target_acceptance,
        tf.constant(cfg.step_size_multiplier, tf.float64),
        tf.constant(1. / cfg.step_size_multiplier, tf.float64))
    tf.debugging.assert_near(result["steps"], expected, atol=1.e-12)


def test_invalid_config_and_target_restore_rejected():
    with pytest.raises(ValueError):
        config(batch_size=1)
    with pytest.raises(ValueError):
        config(replay_capacity=16, replay_min_size=16, updates_per_pass=2)
    trainer = FABTrainer(flow(), normal, config(), target_signature="a"*64, seed=(1, 2))
    other = FABTrainer(flow(), normal, config(), target_signature="b"*64, seed=(1, 2))
    with pytest.raises(ValueError, match="configuration/target"):
        other.restore(trainer.checkpoint())
