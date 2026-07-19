from __future__ import annotations

import copy
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact
from bayesfilter.inference.neutra_training import (
    NEUTRA_TRAINING_NONCLAIMS,
    NeuTraReverseKLTrainer,
    NeuTraTrainerConfig,
    NeuTraTrainingError,
)


TARGET_SIGNATURE = "1" * 64


class DiagonalGaussianTarget:
    def __init__(self, mean=(0.0, 0.0), variance=(1.0, 1.0)):
        self.mean = tf.constant(mean, dtype=tf.float64)
        self.variance = tf.constant(variance, dtype=tf.float64)

    def batch_value_and_score(self, theta):
        delta = theta - self.mean
        value = -0.5 * tf.reduce_sum(tf.square(delta) / self.variance, axis=-1)
        return value, -delta / self.variance


class CorrelatedGaussianTarget:
    def __init__(self):
        self.mean = tf.constant([0.3, -0.2], dtype=tf.float64)
        self.precision = tf.constant([[1.5, -0.4], [-0.4, 0.9]], dtype=tf.float64)

    def batch_value_and_score(self, theta):
        delta = theta - self.mean
        score = -tf.matmul(delta, self.precision)
        value = 0.5 * tf.reduce_sum(delta * score, axis=-1)
        return value, score


class CurvedRidgeTarget:
    def batch_value_and_score(self, theta):
        x = theta[..., 0]
        y = theta[..., 1]
        residual = y - 0.35 * tf.square(x)
        value = -0.5 * tf.square(x / 1.2) - 0.5 * tf.square(residual / 0.7)
        score_y = -residual / (0.7**2)
        score_x = -x / (1.2**2) - score_y * 0.7 * x
        return value, tf.stack((score_x, score_y), axis=-1)


def _config(**overrides):
    values = {
        "dimension": 2,
        "family": "dense_iaf",
        "hidden_layers": (3,),
        "activation": "tanh",
        "initialization_seed": (20260714, 2101),
        "learning_rate": 0.01,
        "jit_compile": False,
    }
    values.update(overrides)
    return NeuTraTrainerConfig(**values)


def _base_rows():
    return tf.constant(
        [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]],
        dtype=tf.float64,
    )


def _gradient_values(trainer, z):
    result, gradients = trainer.loss_and_gradients(z)
    return result, tuple(gradient.numpy() for gradient in gradients)


def test_affine_reverse_kl_gradient_has_analytic_sign_and_reduction() -> None:
    trainer = NeuTraReverseKLTrainer(
        DiagonalGaussianTarget(),
        _config(family="affine_diag"),
    )
    shift = np.array([0.4, -0.3], dtype=np.float64)
    scale = np.array([1.2, 0.8], dtype=np.float64)
    trainer.variables[0].assign(shift)
    trainer.variables[1].assign(np.log(scale))

    result, gradients = _gradient_values(trainer, _base_rows())
    np.testing.assert_allclose(gradients[0], shift, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gradients[1], np.square(scale) - 1.0, rtol=1e-12, atol=1e-12
    )
    assert math.isfinite(float(result.loss.numpy()))
    assert "training loss is not a transport promotion criterion" in NEUTRA_TRAINING_NONCLAIMS


def test_reversed_target_score_fails_the_analytic_gradient_contract() -> None:
    class ReversedScoreGaussian(DiagonalGaussianTarget):
        def batch_value_and_score(self, theta):
            value, score = super().batch_value_and_score(theta)
            return value, -score

    trainer = NeuTraReverseKLTrainer(
        ReversedScoreGaussian(),
        _config(family="affine_diag"),
    )
    shift = np.array([0.4, -0.3], dtype=np.float64)
    trainer.variables[0].assign(shift)
    _, gradients = _gradient_values(trainer, _base_rows())

    np.testing.assert_allclose(gradients[0], -shift, rtol=1e-12, atol=1e-12)
    assert not np.allclose(gradients[0], shift, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("family", ("affine_diag", "dense_iaf"))
def test_batch_duplication_and_permutation_leave_mean_gradient_unchanged(family) -> None:
    trainer = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), _config(family=family))
    z = tf.constant([[0.2, -0.4], [0.1, 0.3], [-0.8, 0.5]], dtype=tf.float64)
    _, baseline = _gradient_values(trainer, z)
    _, duplicated = _gradient_values(trainer, tf.repeat(z, repeats=3, axis=0))
    _, permuted = _gradient_values(trainer, tf.gather(z, [2, 0, 1]))
    for expected, repeated, shuffled in zip(baseline, duplicated, permuted):
        np.testing.assert_allclose(repeated, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(shuffled, expected, rtol=2e-12, atol=2e-12)


def test_dense_iaf_manual_target_score_gradient_matches_debug_full_autodiff() -> None:
    target = CurvedRidgeTarget()
    trainer = NeuTraReverseKLTrainer(target, _config())
    z = tf.constant([[0.2, -0.4], [0.1, 0.3], [-0.8, 0.5]], dtype=tf.float64)
    result, gradients = trainer.loss_and_gradients(z)

    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(trainer.variables)
        theta, logdet = trainer.forward_and_logdet(z)
        target_value, _ = target.batch_value_and_score(theta)
        debug_loss = tf.reduce_mean(-target_value - logdet)
    debug_gradients = tape.gradient(debug_loss, trainer.variables)

    np.testing.assert_allclose(
        result.loss.numpy(), debug_loss.numpy(), rtol=2e-12, atol=2e-12
    )
    for actual, expected in zip(gradients, debug_gradients):
        np.testing.assert_allclose(
            actual.numpy(), expected.numpy(), rtol=2e-12, atol=2e-12
        )


def test_train_step_updates_variables_and_clips_global_norm() -> None:
    trainer = NeuTraReverseKLTrainer(
        CorrelatedGaussianTarget(),
        _config(gradient_clip_norm=1.0e-4),
    )
    before = tuple(variable.numpy().copy() for variable in trainer.variables)
    result = trainer.train_step(_base_rows())

    assert int(result.step.numpy()) == 1
    assert bool(result.clipping_applied.numpy()) is True
    assert float(result.clipped_gradient_norm.numpy()) <= 1.000001e-4
    assert any(
        not np.array_equal(old, variable.numpy())
        for old, variable in zip(before, trainer.variables)
    )


def test_state_restore_replays_next_update_exactly() -> None:
    config = _config()
    first = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), config)
    first.train_step(_base_rows())
    state = first.state_payload()
    z_next = tf.constant([[0.3, -0.6], [-0.2, 0.9]], dtype=tf.float64)
    expected_result = first.train_step(z_next)
    expected_variables = tuple(variable.numpy().copy() for variable in first.variables)

    resumed = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), config)
    resumed.restore_state(state)
    actual_result = resumed.train_step(z_next)

    np.testing.assert_array_equal(actual_result.loss.numpy(), expected_result.loss.numpy())
    np.testing.assert_array_equal(
        actual_result.gradient_norm.numpy(), expected_result.gradient_norm.numpy()
    )
    for expected, actual in zip(expected_variables, resumed.variables):
        np.testing.assert_array_equal(actual.numpy(), expected)

    tampered = copy.deepcopy(state)
    tampered["step"] = int(tampered["step"]) + 1
    with pytest.raises(NeuTraTrainingError, match="state_hash mismatch"):
        resumed.restore_state(tampered)


def test_frozen_snapshot_replays_trainable_transport() -> None:
    trainer = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), _config())
    trainer.train_step(_base_rows())
    z = tf.constant([[0.2, -0.4], [2.5, -3.0]], dtype=tf.float64)
    expected_theta, expected_logdet = trainer.forward_and_logdet(z)
    payload = trainer.frozen_transport_payload(
        transport_id="phase3-dense-iaf-fixture",
        target_signature=TARGET_SIGNATURE,
    )
    artifact = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=TARGET_SIGNATURE,
    )

    np.testing.assert_array_equal(
        artifact.transport.forward_batch(z).numpy(), expected_theta.numpy()
    )
    np.testing.assert_array_equal(
        artifact.transport.log_abs_det_jacobian_batch(z).numpy(),
        expected_logdet.numpy(),
    )
    assert artifact.manifest.training_state_hash == trainer.state_payload()["state_hash"]


def test_stateless_base_sampling_is_replayable_and_role_separated() -> None:
    trainer = NeuTraReverseKLTrainer(DiagonalGaussianTarget(), _config())
    first = trainer.sample_base(batch_size=4, seed=(20260714, 2101))
    replay = trainer.sample_base(batch_size=4, seed=(20260714, 2101))
    validation = trainer.sample_base(batch_size=4, seed=(20260714, 2201))
    np.testing.assert_array_equal(first.numpy(), replay.numpy())
    assert not np.array_equal(first.numpy(), validation.numpy())


def test_validation_batch_is_deterministic_shaped_and_nonupdating() -> None:
    trainer = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), _config())
    z = trainer.sample_base(batch_size=4, seed=(20260714, 2201))
    before = trainer.state_payload()
    first = trainer.validation_batch(z)
    second = trainer.validation_batch(z)
    after = trainer.state_payload()

    assert first.per_sample_loss.shape == (4,)
    assert first.target_value.shape == (4,)
    assert first.theta.shape == (4, 2)
    assert first.logdet.shape == (4,)
    assert first.scale_log.shape == (4, 2)
    for left, right in zip(first.__dict__.values(), second.__dict__.values()):
        np.testing.assert_array_equal(left.numpy(), right.numpy())
    assert after == before


def test_nonfinite_target_and_invalid_target_shapes_fail_closed() -> None:
    class NonfiniteTarget:
        def batch_value_and_score(self, theta):
            return tf.fill(theta.shape[:-1], tf.constant(float("inf"), tf.float64)), theta

    class WrongShapeTarget:
        def batch_value_and_score(self, theta):
            return tf.zeros((1,), tf.float64), tf.zeros_like(theta)

    with pytest.raises(tf.errors.InvalidArgumentError, match="target value"):
        NeuTraReverseKLTrainer(NonfiniteTarget(), _config()).loss_and_gradients(
            _base_rows()
        )
    with pytest.raises(NeuTraTrainingError, match="target value shape mismatch"):
        NeuTraReverseKLTrainer(WrongShapeTarget(), _config()).loss_and_gradients(
            _base_rows()
        )


def test_config_rejects_invalid_training_contracts() -> None:
    with pytest.raises(ValueError, match="family"):
        _config(family="real_nvp")
    with pytest.raises(ValueError, match="hidden"):
        _config(hidden_layers=())
    with pytest.raises(ValueError, match="learning_rate"):
        _config(learning_rate=0.0)
    with pytest.raises(ValueError, match="paper_piecewise"):
        _config(learning_rate_schedule="paper_piecewise")
    assert NeuTraTrainerConfig(dimension=2).jit_compile is True
