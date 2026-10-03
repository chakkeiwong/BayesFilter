"""Precision boundaries and independent references; tiny CPU/XLA fixtures."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_transport import (
    NeuTraTransport, NeuTraTransportConfig, NeuTraTransportTrainer, NeuTraOptimizerConfig,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact


def config(kind, dtype="float32"):
    return NeuTraTransportConfig(3, kind, (6, 6), 2, "elu", (247, 19), 2.,
        mixture_components=3, dtype=dtype, affine_center=(.2, -.1, .3), affine_scale=(.8, 1.2, 1.5))


def rows(dtype=tf.float32):
    return tf.constant([[.4, -.7, 1.1], [-1.3, .8, -.5], [2., 1.3, -.4], [.1, -.3, .6]], dtype)


def target(x):
    assert x.dtype == tf.float64
    # Coupled, nonstandard Gaussian: not an identity-map/zero-gradient check.
    matrix = tf.constant([[2., .2, -.1], [.2, 1.3, .15], [-.1, .15, .8]], tf.float64)
    score = -tf.matmul(x, matrix)
    return .5*tf.reduce_sum(x*score, axis=-1), score, tf.ones(tf.shape(x)[0], tf.bool)


def optimizer(estimator="standard"):
    return NeuTraOptimizerConfig(4, estimator, .001, .9, .999, 1.e-7, None)


@pytest.mark.parametrize("kind", ["iaf", "naf_dsf"])
@pytest.mark.parametrize("estimator", ["standard", "path"])
def test_fp32_flow_fp64_target_matches_identical_parameter_reference(kind, estimator):
    flow = NeuTraTransport(config(kind))
    reference = flow.as_dtype("float64", trainable=True)
    trainer = NeuTraTransportTrainer(flow, target, optimizer(estimator), target_signature="a"*64)
    ref = NeuTraTransportTrainer(reference, target, optimizer(estimator), target_signature="a"*64)
    actual, expected = trainer.evaluate(rows()), ref.evaluate(tf.cast(rows(), tf.float64))
    assert bool(actual["valid"])
    assert actual["loss"].dtype == tf.float64
    assert all(v.dtype == tf.float32 for v in flow.trainable_variables)
    assert all(g.dtype == tf.float32 for g in actual["gradients"])
    np.testing.assert_allclose(actual["loss"], expected["loss"], atol=2.e-5, rtol=2.e-5)
    g = np.concatenate([x.numpy().ravel() for x in actual["gradients"]])
    r = np.concatenate([x.numpy().ravel() for x in expected["gradients"]])
    assert np.linalg.norm(g-r)/(1.+np.linalg.norm(r)) < 2.e-5
    result = trainer.train_step(rows())
    assert bool(result["valid"]) and int(result["iteration"]) == 1
    assert all(tf.as_dtype(v.dtype) == tf.float32 for v in trainer.optimizer.variables
               if tf.as_dtype(v.dtype).is_floating)


@pytest.mark.parametrize("kind", ["iaf", "naf_dsf"])
def test_fp32_tail_inverse_and_implicit_input_derivative(kind):
    flow = NeuTraTransport(config(kind))
    x = tf.constant([[-30., .2, 18.], [20., -25., -.3], [0., 0., 0.], [1., -1., 1.]], tf.float32)
    @tf.function(input_signature=[tf.TensorSpec([4, 3], tf.float32)], jit_compile=True)
    def check(z):
        y, ld = flow.forward_and_logdet(z)
        with tf.GradientTape() as tape:
            tape.watch(y)
            inverse, inverse_ld = flow.inverse_and_forward_logdet(y)
        dy = tape.gradient(inverse, y)
        return y, ld, inverse, inverse_ld, dy
    y, ld, recovered, inverse_ld, derivative = check(x)
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in (y, ld, recovered, inverse_ld, derivative))
    np.testing.assert_allclose(recovered, x, atol=2.e-4, rtol=2.e-5)
    np.testing.assert_allclose(ld, inverse_ld, atol=2.e-4, rtol=2.e-5)
    reference = flow.as_dtype("float64")
    y64 = tf.cast(y, tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(y64)
        inverse64, _ = reference.inverse_and_forward_logdet(y64)
    np.testing.assert_allclose(derivative, tape.gradient(inverse64, y64), atol=2.e-4, rtol=2.e-4)


@pytest.mark.parametrize("kind", ["iaf", "naf_dsf"])
def test_fp32_checkpoint_resume_and_explicit_fp64_export(kind):
    cfg = config(kind)
    first = NeuTraTransportTrainer(NeuTraTransport(cfg), target, optimizer(), target_signature="a"*64)
    assert bool(first.train_step(rows())["valid"])
    checkpoint = json.loads(json.dumps(first.checkpoint()))
    resumed = NeuTraTransportTrainer(NeuTraTransport(cfg), target, optimizer(), target_signature="a"*64)
    resumed.restore(checkpoint)
    first.train_step(rows()); resumed.train_step(rows())
    assert first.checkpoint() == resumed.checkpoint()
    frozen = first.transport.as_dtype("float64").frozen_payload(target_signature="a"*64)
    assert frozen["precision_conversion"]["source_dtype"] == "float32"
    loaded = load_frozen_neutra_artifact(frozen, expected_target_signature="a"*64).transport
    assert loaded.dtype == tf.float64 and not loaded.trainable_variables
    expected = first.transport.as_dtype("float64").forward_batch(tf.cast(rows(), tf.float64))
    np.testing.assert_array_equal(loaded.forward_batch(tf.cast(rows(), tf.float64)), expected)
    with pytest.raises(ValueError, match="configuration"):
        wrong = NeuTraTransportTrainer(NeuTraTransport(config(kind, "float64")), target,
                                      optimizer(), target_signature="a"*64)
        wrong.restore(checkpoint)


def test_historical_configured_checkpoint_without_precision_fields_resumes():
    cfg = replace(config("iaf", "float64"), affine_center=(), affine_scale=())
    trainer = NeuTraTransportTrainer(NeuTraTransport(cfg), target, optimizer(), target_signature="a"*64)
    trainer.train_step(rows(tf.float64))
    checkpoint = trainer.checkpoint()
    for key in ("dtype", "affine_center", "affine_scale"):
        checkpoint["transport_config"].pop(key)
    checkpoint["optimizer_config"].pop("target_dtype")
    checkpoint.pop("checkpoint_hash")
    checkpoint["checkpoint_hash"] = hashlib.sha256(json.dumps(checkpoint, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    restored = NeuTraTransportTrainer(NeuTraTransport(cfg), target, optimizer(), target_signature="a"*64)
    restored.restore(checkpoint)
    assert restored.checkpoint() == trainer.checkpoint()


def test_precision_and_affine_configuration_reject_invalid_values():
    for changes in ({"dtype": "float16"}, {"affine_scale": (1., 0., 1.)},
                    {"affine_center": (0.,)}, {"inverse_atol": 0.}):
        with pytest.raises(ValueError):
            replace(config("iaf"), **changes)
    assert config("naf_dsf").inverse_atol == 16.*2.**-23
    assert config("naf_dsf", "float64").inverse_atol == 1.e-11


def test_configured_fp32_naf_weighted_consumer_and_validation():
    from bayesfilter.inference.neutra_weighted_training import WeightedForwardKLNeuTraTrainer, WeightedNeuTraConfig
    flow = NeuTraTransport(config("naf_dsf"))
    trainer = WeightedForwardKLNeuTraTrainer(WeightedNeuTraConfig(dimension=3), transport=flow)
    result = trainer.train_step(rows(), tf.zeros([4], tf.float32))
    assert int(result.step) == 1
    validation = trainer.validation_batch(rows(), tf.zeros([4], tf.float32))
    assert all(bool(tf.reduce_all(tf.math.is_finite(v))) for v in vars(validation).values())


def test_fp32_finalization_verifies_the_exported_fp64_map():
    class Bridge:
        parameter_dim = 3
        def value_score_status(self, x, beta):
            value, score, valid = target(x)
            return value, score, {"bridge_valid": valid}
    trainer = NeuTraTransportTrainer(NeuTraTransport(config("iaf")), target,
                                    optimizer(), target_signature="a"*64)
    trainer.train_step(rows())
    final = trainer.finalize(Bridge(), diagnostic_seed=[491, 23])
    assert final["post_training"]["rows"] == 1000
    assert final["post_training"]["valid_rows"] == 1000
    assert final["post_training"]["training_dtype"] == "float32"
    assert final["frozen_transport"]["config"]["dtype"] == "float64"
    assert final["frozen_transport"]["precision_conversion"]["requires_fresh_evaluation"]
    assert not final["training_quality_established"]


@pytest.mark.parametrize("estimator", ["standard", "path"])
def test_target_can_also_explicitly_use_fp32(estimator):
    def target32(x):
        assert x.dtype == tf.float32
        return -.5*tf.reduce_sum(x*x, axis=1), -x, tf.ones([4], tf.bool)
    trainer = NeuTraTransportTrainer(NeuTraTransport(config("iaf")), target32,
        replace(optimizer(estimator), target_dtype="float32"), target_signature="a"*64)
    result = trainer.train_step(rows())
    assert bool(result["valid"]) and result["loss"].dtype == tf.float32
