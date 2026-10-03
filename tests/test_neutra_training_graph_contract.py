"""CPU reference checks for fixed training graphs and real batch accounting."""
import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_training import NeuTraReverseKLTrainer, NeuTraTrainerConfig


class GaussianTarget:
    def __init__(self):
        self.calls = 0

    def batch_value_and_score(self, theta):
        self.calls += 1
        return -.5 * tf.reduce_sum(theta * theta, axis=-1), -theta


def trainer():
    return NeuTraReverseKLTrainer(GaussianTarget(), NeuTraTrainerConfig(
        dimension=2, family="affine_diag", jit_compile=False))


@pytest.mark.parametrize("mode", ["direct", "external", "padded_singleton", "scalar_chunks"])
def test_singleton_updates_are_rejected_before_any_state_change(mode):
    fit = trainer()
    before = fit.state_payload()
    z = tf.ones((1, 2), tf.float64)
    with pytest.raises(ValueError, match="at least two"):
        if mode == "direct":
            fit.train_step(z)
        elif mode == "external":
            fit.train_step_with_external_value_score(z, tf.ones((1,), tf.float64), z)
        elif mode == "padded_singleton":
            fit.train_step_with_external_value_score_chunks(
                (tf.repeat(z, 4, axis=0),), (tf.ones((4,), tf.float64),),
                (tf.repeat(z, 4, axis=0),), (1,))
        else:
            fit.train_step_with_external_value_score_chunks(
                (z, z), (tf.ones((1,), tf.float64),) * 2, (z, z), (1, 1))
    assert fit.state_payload() == before
    assert fit.target.calls == 0


def test_cache_eviction_preserves_updates_and_signatures_are_static():
    cached, reference = trainer(), trainer()
    for count in (2, 3, 4, 5, 6, 2, 6):
        z = tf.reshape(tf.range(count * 2, dtype=tf.float64), (count, 2)) / 10
        actual = cached.train_step(z)
        expected = reference._train_step_impl(z)
        np.testing.assert_allclose(actual.loss, expected[0], rtol=1e-12, atol=1e-12)
        for left, right in zip(cached.variables, reference.variables):
            np.testing.assert_allclose(left, right, rtol=1e-12, atol=1e-12)
    assert int(cached.step) == int(reference.step) == 7
    programs = cached._compiled_train_step.programs
    assert len(programs) == 4
    for signature, program in programs.items():
        assert program.input_signature == signature
        assert all(s.shape.is_fully_defined() for s in signature)
        assert program.experimental_get_tracing_count() == 1


def test_static_targets_and_single_row_validation_remain_supported():
    class StaticTarget:
        def batch_value_and_score(self, theta):
            assert theta.shape[0] is not None
            return tf.zeros((theta.shape[0],), tf.float64), tf.zeros_like(theta)
    fit = NeuTraReverseKLTrainer(StaticTarget(), NeuTraTrainerConfig(
        dimension=2, family="affine_diag", jit_compile=False))
    fit.train_step(tf.ones((3, 2), tf.float64))
    fit.train_step(tf.ones((4, 2), tf.float64))
    result = fit.validation_batch(tf.ones((1, 2), tf.float64))
    assert tuple(result.theta.shape) == (1, 2)


def test_external_and_chunked_programs_have_explicit_signatures():
    fit = trainer()
    z = tf.ones((2, 2), tf.float64)
    value = -tf.ones((2,), tf.float64)
    fit.train_step_with_external_value_score(z, value, -z)
    fit.train_step_with_external_value_score_chunks((z, z), (value, value), (-z, -z), (2, 1))
    for cache in (fit._compiled_external_train_step, fit._compiled_external_gradients):
        for signature, program in cache.programs.items():
            assert program.input_signature == signature
            assert all(s.shape.is_fully_defined() for s in signature)
