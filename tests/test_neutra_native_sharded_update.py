"""CPU-only diagnostic equivalence of complete canonical NeuTra updates."""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_transport import (
    NeuTraTransportConfig, NeuTraTransport, NeuTraOptimizerConfig,
    NeuTraTransportTrainer,
)


def target(q):
    # Coupling and quartic tails test more than a diagonal Gaussian gradient.
    a, b = q[:, 0], q[:, 1]
    v = -.5*(a*a + 2*b*b) - .1*(a-b)**4
    t = .4*(a-b)**3
    s = tf.stack([-a-t, -2*b+t], axis=-1)
    return v, s, tf.reduce_all(tf.abs(q) < 30, -1)


def trainer(dtype="float64", estimator="standard", shard=None, clip=None, target_dtype=None):
    cfg = NeuTraTransportConfig.hoffman_author_iaf(
        2, conditional_scale_cap=2., seed=(11, 23), dtype=dtype)
    opt = NeuTraOptimizerConfig(7, estimator, .001, .9, .999, 1e-7, clip,
                               target_dtype=target_dtype or dtype, shard_size=shard)
    return NeuTraTransportTrainer(NeuTraTransport(cfg), target, opt,
                                 target_signature="a"*64)


def assert_state_close(left, right, tolerance):
    for a, b in zip((*left.variables, *left.optimizer.variables),
                    (*right.variables, *right.optimizer.variables)):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("estimator", ["standard", "path"])
@pytest.mark.parametrize("shard,clip", [(2, None), (3, .05), (4, None)])
def test_complete_update_and_resume_equal_unsharded(dtype, estimator, shard, clip):
    whole, chunked = trainer(dtype, estimator, clip=clip), trainer(dtype, estimator, shard, clip)
    z = tf.random.stateless_normal([7, 2], [17, 29], dtype=tf.as_dtype(dtype))
    tol = 2e-5 if dtype == "float32" else 2e-10
    left, right = whole.evaluate(z), chunked.evaluate(z)
    assert bool(left["valid"]) and bool(right["valid"])
    for key in ("loss", "proposal_score"):
        np.testing.assert_allclose(left[key], right[key], rtol=tol, atol=tol)
    for a, b in zip(left["gradients"], right["gradients"]):
        np.testing.assert_allclose(a, b, rtol=tol, atol=tol)
    assert bool(whole.train_step(z)["valid"]) and bool(chunked.train_step(z)["valid"])
    assert int(chunked.optimizer.iterations) == 1
    assert_state_close(whole, chunked, tol)
    resumed = trainer(dtype, estimator, shard, clip)
    resumed.restore(chunked.checkpoint())
    z2 = tf.reverse(z, [0]) * tf.cast(.8, z.dtype)
    whole.train_step(z2)
    chunked.train_step(z2)
    resumed.train_step(z2)
    assert_state_close(whole, chunked, tol)
    assert_state_close(chunked, resumed, 0.)


def test_invalid_final_shard_rolls_back_entire_update():
    run = trainer(shard=3, clip=.05)
    z = tf.constant([[.2, .4]]*6 + [[100., 100.]], tf.float64)
    before = run.checkpoint()
    assert not bool(run.train_step(z)["valid"])
    assert run.checkpoint() == before


def test_target_receives_native_shards_and_padding_has_zero_weight():
    run = trainer(shard=3)
    seen = []
    def checked(q):
        seen.append(tuple(q.shape))
        return target(q)
    run.target_value_score = checked
    z = tf.constant([[.1*i, -.2*i] for i in range(7)], tf.float64)
    actual = run.evaluate(z)
    expected = trainer().evaluate(z)
    assert seen == [(3, 2)]
    np.testing.assert_allclose(actual["loss"], expected["loss"], rtol=1e-12)


@pytest.mark.parametrize("bad", [1, 8, 2.5])
def test_invalid_shard_configuration_rejected(bad):
    with pytest.raises(ValueError, match="shard_size"):
        trainer(shard=bad)


@pytest.mark.parametrize("dtype,target_dtype", [("float32", "float64"), ("float64", "float32")])
@pytest.mark.parametrize("estimator", ["standard", "path"])
def test_complete_update_with_distinct_flow_and_target_precision(dtype, target_dtype, estimator):
    whole = trainer(dtype, estimator, target_dtype=target_dtype)
    sharded = trainer(dtype, estimator, shard=3, target_dtype=target_dtype)
    z = tf.random.stateless_normal([7,2], [19,31], dtype=tf.as_dtype(dtype))
    left, right = whole.evaluate(z), sharded.evaluate(z)
    assert bool(left["valid"] & right["valid"])
    np.testing.assert_allclose(left["loss"], right["loss"], rtol=3e-5, atol=3e-5)
    for a, b in zip(left["gradients"], right["gradients"]):
        np.testing.assert_allclose(a, b, rtol=3e-5, atol=3e-5)
    assert bool(whole.train_step(z)["valid"] & sharded.train_step(z)["valid"])
    assert_state_close(whole, sharded, 3e-5)
