from __future__ import annotations

import copy
import math
import os
import warnings

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import (
    PURE_BOUNDED_NEUTRA_FAMILY,
    PURE_PAPER_NEUTRA_FAMILY,
    pure_bounded_neutra_config,
    pure_paper_neutra_config as public_pure_paper_neutra_config,
)
from bayesfilter.inference.neutra_artifacts import (
    InvalidNeuTraArtifact,
    finalize_dense_iaf_neutra_artifact_payload,
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (
    NEUTRA_TRAINING_NONCLAIMS,
    NeuTraReverseKLTrainer,
    NeuTraTrainerConfig,
    NeuTraTrainingError,
    preflight_neutra_affine_chart,
    pure_paper_neutra_config,
    ssl_lstm_pure_neutra_config,
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


def test_composed_training_warns_for_identity_chart() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: TARGET_SIGNATURE
        config = type(
            "Config",
            (),
            {
                "prior_center": tf.zeros(2, tf.float64),
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                },
            },
        )()

    config = NeuTraTrainerConfig(
        dimension=2,
        family="ssl_lstm_tuned_capacity_dense_iaf",
        hidden_layers=(32, 32),
        activation="elu",
        initialization_seed=(1, 2),
        learning_rate=1.0e-3,
        learning_rate_schedule="adaptive_constant",
        epsilon=1.0e-7,
        initialization_scale=0.02,
        gradient_clip_norm=10.0,
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=(0.0, 0.0),
        target_parameter_names=("x", "y"),
        target_chart="identity",
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature=TARGET_SIGNATURE,
        jit_compile=False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        NeuTraReverseKLTrainer(Target(), config)
    assert any("appropriately scaled" in str(item.message) for item in caught)


def test_composed_training_unspecified_chart_is_nonbreaking_and_warns() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: TARGET_SIGNATURE
        config = type(
            "Config",
            (),
            {
                "prior_center": tf.zeros(2, tf.float64),
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                },
            },
        )()

    config = NeuTraTrainerConfig(
        dimension=2,
        family="ssl_lstm_tuned_capacity_dense_iaf",
        hidden_layers=(32, 32),
        activation="elu",
        initialization_seed=(1, 2),
        learning_rate=1.0e-3,
        learning_rate_schedule="adaptive_constant",
        epsilon=1.0e-7,
        initialization_scale=0.02,
        gradient_clip_norm=10.0,
        gradient_clip_mode="per_variable",
        stages=3,
        fixed_translation=(0.0, 0.0),
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature=TARGET_SIGNATURE,
        jit_compile=False,
    )
    assert config.target_chart == "unspecified"
    with pytest.warns(RuntimeWarning, match="appropriately scaled"):
        NeuTraReverseKLTrainer(Target(), config)


def test_affine_chart_preflight_checks_value_score_and_jacobian() -> None:
    class Physical(DiagonalGaussianTarget):
        pass

    physical = Physical(mean=(0.3, -0.2), variance=(0.25, 4.0))
    factor = tf.constant([[0.5, 0.0], [0.0, 2.0]], tf.float64)
    center = tf.constant([0.3, -0.2], tf.float64)
    z = tf.constant([[0.1, -0.4], [1.2, 0.7]], tf.float64)

    class Transformed:
        def batch_value_and_score(self, latent):
            theta = center[None, :] + tf.matmul(latent, factor, transpose_b=True)
            value, score = physical.batch_value_and_score(theta)
            return value + tf.linalg.slogdet(factor)[1], tf.matmul(score, factor)

    report = preflight_neutra_affine_chart(
        chart_name="test",
        center=center,
        factor=factor,
        latent=z,
        physical_target=physical,
        transformed_target=Transformed(),
    )
    assert report.passed
    assert report.value_max_abs_residual == 0.0
    assert report.score_max_abs_residual == 0.0


def test_affine_chart_preflight_rejects_singular_factor() -> None:
    with pytest.raises(ValueError, match="preflight failed"):
        preflight_neutra_affine_chart(
            chart_name="singular",
            center=tf.zeros(2, tf.float64),
            factor=tf.constant([[1.0, 0.0], [0.0, 0.0]], tf.float64),
            latent=tf.ones((2, 2), tf.float64),
        )

    report = preflight_neutra_affine_chart(
        chart_name="singular",
        center=tf.zeros(2, tf.float64),
        factor=tf.constant([[1.0, 0.0], [0.0, 0.0]], tf.float64),
        latent=tf.ones((2, 2), tf.float64),
        strict=False,
    )
    assert not report.passed
    assert not report.finite


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


def test_validation_exposes_scale_logits_and_hidden_preactivations() -> None:
    target = CorrelatedGaussianTarget()
    trainer = NeuTraReverseKLTrainer(
        target,
        _config(
            dimension=2,
            family="dense_iaf",
            hidden_layers=(3, 4),
            activation="elu",
            target_signature=TARGET_SIGNATURE,
        ),
    )
    z = tf.constant([[0.2, -0.4], [0.1, 0.3]], dtype=tf.float64)
    validation = trainer.validation_batch(z)
    assert tuple(validation.scale_logits.shape) == (2, 2)
    assert tuple(validation.hidden_preactivations.shape) == (2, 2, 4)
    np.testing.assert_allclose(
        validation.scale_log.numpy(),
        np.tanh(validation.scale_logits.numpy()),
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.all(np.isfinite(validation.hidden_preactivations.numpy()))

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


@pytest.mark.parametrize("family", ("affine_diag", "dense_iaf"))
def test_generic_train_step_rejects_nonfinite_without_mutating_state(family) -> None:
    class NonfiniteTarget:
        def batch_value_and_score(self, theta):
            value = tf.fill(theta.shape[:-1], tf.constant(float("nan"), tf.float64))
            return value, tf.fill(tf.shape(theta), tf.constant(float("nan"), tf.float64))

    trainer = NeuTraReverseKLTrainer(
        NonfiniteTarget(), _config(family=family, jit_compile=True)
    )
    before = trainer.state_payload()
    with pytest.raises(NeuTraTrainingError, match="rejected nonfinite"):
        trainer.train_step(_base_rows())
    assert trainer.state_payload() == before


def test_chunked_external_update_matches_full_external_update() -> None:
    config = _config(family="affine_diag", gradient_clip_norm=10.0)
    full = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), config)
    chunked = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), config)
    chunked.restore_state(full.state_payload())
    z = tf.constant(
        [[0.2, -0.4], [0.1, 0.3], [-0.8, 0.5]], dtype=tf.float64
    )
    theta, _ = full.forward_and_logdet(z)
    target_value, target_score = full.target.batch_value_and_score(theta)
    expected = full.train_step_with_external_value_score(z, target_value, target_score)
    actual = chunked.train_step_with_external_value_score_chunks(
        (z[:2], tf.concat((z[2:3], z[2:3]), axis=0)),
        (target_value[:2], tf.concat((target_value[2:3], target_value[2:3]), axis=0)),
        (target_score[:2], tf.concat((target_score[2:3], target_score[2:3]), axis=0)),
        (2, 1),
    )
    np.testing.assert_allclose(actual.loss.numpy(), expected.loss.numpy(), rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(
        actual.gradient_norm.numpy(), expected.gradient_norm.numpy(), rtol=1e-7, atol=1e-8
    )
    np.testing.assert_allclose(
        actual.surrogate.numpy(), expected.surrogate.numpy(), rtol=1e-7, atol=1e-8
    )
    np.testing.assert_allclose(
        actual.target_value_mean.numpy(), expected.target_value_mean.numpy(), rtol=1e-7, atol=1e-8
    )
    assert int(actual.step.numpy()) == int(expected.step.numpy()) == 1
    for left, right in zip(chunked.variables, full.variables):
        np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=1e-7, atol=1e-8)
    for left, right in zip(chunked.first_moments, full.first_moments):
        np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=1e-7, atol=1e-8)
    for left, right in zip(chunked.second_moments, full.second_moments):
        np.testing.assert_allclose(left.numpy(), right.numpy(), rtol=1e-7, atol=1e-8)


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


@pytest.mark.parametrize("family", ("affine_diag", "dense_iaf"))
def test_generic_learning_rate_state_controls_next_update_and_replays(family) -> None:
    config = _config(family=family, learning_rate=1.0e-2)
    reference = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), config)
    reduced = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), config)
    reduced.set_learning_rate(2.0e-3)
    assert float(reduced.learning_rate_at(0).numpy()) == pytest.approx(2.0e-3)
    assert reduced.state_payload()["effective_learning_rate"] == pytest.approx(2.0e-3)
    assert reduced.state_payload()["state_hash"] != reference.state_payload()["state_hash"]
    z = _base_rows()
    reference.train_step(z)
    reduced.train_step(z)
    assert not all(
        np.array_equal(left.numpy(), right.numpy())
        for left, right in zip(reference.variables, reduced.variables)
    )

    state = reduced.state_payload()
    expected = reduced.train_step(z)
    expected_variables = tuple(variable.numpy().copy() for variable in reduced.variables)
    resumed = NeuTraReverseKLTrainer(CorrelatedGaussianTarget(), config)
    resumed.restore_state(state)
    assert float(resumed.learning_rate_at(1).numpy()) == pytest.approx(2.0e-3)
    actual = resumed.train_step(z)
    np.testing.assert_array_equal(actual.loss.numpy(), expected.loss.numpy())
    for expected_value, actual_variable in zip(expected_variables, resumed.variables):
        np.testing.assert_array_equal(actual_variable.numpy(), expected_value)


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


def test_pure_composed_iaf_has_no_affine_component_and_restores_learning_rate() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: "2" * 64
        config = type(
            "Config",
            (),
            {
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                }
            },
        )()

    config = ssl_lstm_pure_neutra_config(
        dimension=2,
        initial_output_shift=(0.3, -0.2),
        initial_output_scale_log=(-1.2, -0.8),
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature="2" * 64,
        s_max=2.0,
        jit_compile=False,
    )
    trainer = NeuTraReverseKLTrainer(Target(), config)
    np.testing.assert_allclose(trainer.variables[5].numpy(), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(trainer.variables[11].numpy(), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        trainer.variables[17].numpy()[2:],
        [0.3, -0.2],
        rtol=0.0,
        atol=0.0,
    )
    zero = tf.zeros((1, 2), tf.float64)
    theta, _ = trainer.forward_and_logdet(zero)
    np.testing.assert_allclose(theta.numpy(), [[0.3, -0.2]], rtol=0.0, atol=1e-14)
    trainer.set_learning_rate(5.0e-4)
    state = trainer.state_payload()
    restored = NeuTraReverseKLTrainer(Target(), config)
    restored.restore_state(state)
    assert float(restored.learning_rate_at(0).numpy()) == pytest.approx(5.0e-4)
    payload = restored.frozen_transport_payload(
        transport_id="pure-composed-test",
        target_signature=TARGET_SIGNATURE,
    )
    assert payload["component_order"] == [
        "dense_iaf_00",
        "mixing_reverse_00",
        "dense_iaf_01",
        "mixing_reverse_01",
        "dense_iaf_02",
    ]
    assert all(component["kind"] not in {"affine", "affine_dense"} for component in payload["components"])
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=TARGET_SIGNATURE)
    z = tf.constant([[0.2, -0.4], [1.0, -1.0]], tf.float64)
    np.testing.assert_allclose(
        loaded.transport.inverse_theta_to_z_batch(
            loaded.transport.forward_z_to_theta_batch(z)
        ).numpy(),
        z.numpy(),
        rtol=0.0,
        atol=2e-14,
    )


def test_pure_composed_iaf_rejects_fixed_chart_fields() -> None:
    values = dict(
        dimension=2,
        initial_output_shift=(0.3, -0.2),
        initial_output_scale_log=(-1.2, -0.8),
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature="2" * 64,
        s_max=2.0,
        jit_compile=False,
    )
    config = ssl_lstm_pure_neutra_config(**values)
    with pytest.raises(ValueError, match="forbids fixed affine"):
        NeuTraTrainerConfig(**{
            **config.__dict__,
            "fixed_translation": (0.0, 0.0),
        })


def test_pure_paper_recipe_uses_piecewise_lr_no_clipping_and_all_kernel_init() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: "2" * 64
        config = type(
            "Config",
            (),
            {
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                }
            },
        )()

    config = ssl_lstm_pure_neutra_config(
        dimension=2,
        initial_output_shift=(0.3, -0.2),
        initial_output_scale_log=(-1.2, -0.8),
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature="2" * 64,
        s_max=2.0,
        learning_rate=1.0e-2,
        learning_rate_schedule="paper_piecewise",
        gradient_clip_mode="none",
        kernel_initialization="paper_variance_scaling",
        scale_transform="identity",
        epsilon=1.0e-8,
        jit_compile=False,
    )
    trainer = NeuTraReverseKLTrainer(Target(), config)
    assert "paper_piecewise_boundaries" not in config.manifest_payload()
    output_kernels = (trainer.variables[4], trainer.variables[10], trainer.variables[16])
    assert all(bool(tf.reduce_any(tf.not_equal(value, 0.0)).numpy()) for value in output_kernels)
    assert float(trainer.learning_rate_at(0).numpy()) == pytest.approx(1.0e-2)
    assert float(trainer.learning_rate_at(999).numpy()) == pytest.approx(1.0e-2)
    assert float(trainer.learning_rate_at(1000).numpy()) == pytest.approx(1.0e-3)
    assert float(trainer.learning_rate_at(3999).numpy()) == pytest.approx(1.0e-3)
    assert float(trainer.learning_rate_at(4000).numpy()) == pytest.approx(1.0e-4)
    result = trainer.train_step(_base_rows())
    assert not bool(result.clipping_applied.numpy())
    np.testing.assert_allclose(
        result.clipped_gradient_norm.numpy(),
        result.gradient_norm.numpy(),
        rtol=0.0,
        atol=0.0,
    )
    state = trainer.state_payload()
    restored = NeuTraReverseKLTrainer(Target(), config)
    restored.restore_state(state)
    assert restored.state_payload()["state_hash"] == state["state_hash"]
    payload = restored.frozen_transport_payload(
        transport_id="pure-paper-recipe-test",
        target_signature=TARGET_SIGNATURE,
    )
    assert all(
        component.get("scale_transform") == "identity"
        for component in payload["components"]
        if component["kind"] == "dense_autoregressive_iaf"
    )
    loaded = load_frozen_neutra_artifact(
        payload, expected_target_signature=TARGET_SIGNATURE
    )
    z = tf.constant([[0.2, -0.4], [1.0, -1.0]], tf.float64)
    np.testing.assert_allclose(
        loaded.transport.inverse_theta_to_z_batch(
            loaded.transport.forward_z_to_theta_batch(z)
        ).numpy(),
        z.numpy(),
        rtol=0.0,
        atol=2e-12,
    )
    with pytest.raises(NeuTraTrainingError, match="paper_piecewise"):
        restored.set_learning_rate(1.0e-3)


def test_pure_paper_recipe_accepts_manifest_bound_schedule_boundaries() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: "2" * 64
        config = type(
            "Config",
            (),
            {
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                }
            },
        )()

    values = dict(
        dimension=2,
        initial_output_shift=(0.3, -0.2),
        initial_output_scale_log=(-1.2, -0.8),
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature="2" * 64,
        s_max=2.0,
        learning_rate=1.0e-2,
        learning_rate_schedule="paper_piecewise",
        paper_piecewise_boundaries=(99, 399),
        gradient_clip_mode="none",
        kernel_initialization="paper_variance_scaling",
        scale_transform="identity",
        epsilon=1.0e-8,
        jit_compile=False,
    )
    config = ssl_lstm_pure_neutra_config(**values)
    trainer = NeuTraReverseKLTrainer(Target(), config)

    assert config.manifest_payload()["paper_piecewise_boundaries"] == [99, 399]
    assert float(trainer.learning_rate_at(0).numpy()) == pytest.approx(1.0e-2)
    assert float(trainer.learning_rate_at(99).numpy()) == pytest.approx(1.0e-2)
    assert float(trainer.learning_rate_at(100).numpy()) == pytest.approx(1.0e-3)
    assert float(trainer.learning_rate_at(399).numpy()) == pytest.approx(1.0e-3)
    assert float(trainer.learning_rate_at(400).numpy()) == pytest.approx(1.0e-4)

    state = trainer.state_payload()
    restored = NeuTraReverseKLTrainer(Target(), config)
    restored.restore_state(state)
    assert restored.state_payload()["state_hash"] == state["state_hash"]

    extended_config = NeuTraTrainerConfig(
        **{**config.__dict__, "paper_piecewise_boundaries": (9, 19, 29)}
    )
    extended = NeuTraReverseKLTrainer(Target(), extended_config)
    assert extended_config.manifest_payload()["paper_piecewise_boundaries"] == [
        9,
        19,
        29,
    ]
    for iteration, expected_rate in (
        (9, 1.0e-2),
        (10, 1.0e-3),
        (20, 1.0e-4),
        (30, 1.0e-5),
    ):
        assert float(extended.learning_rate_at(iteration).numpy()) == pytest.approx(
            expected_rate
        )

    for invalid in ((99, 99), (100, 99), (-1,), (99.0, 399), (True, 399)):
        with pytest.raises(ValueError, match="paper_piecewise_boundaries"):
            NeuTraTrainerConfig(
                **{**config.__dict__, "paper_piecewise_boundaries": invalid}
            )
    with pytest.raises(ValueError, match="requires paper_piecewise"):
        NeuTraTrainerConfig(
            **{
                **config.__dict__,
                "learning_rate_schedule": "adaptive_constant",
            }
        )


def test_restore_accepts_only_history_preserving_paper_schedule_extension() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: "2" * 64
        config = type(
            "Config",
            (),
            {
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                }
            },
        )()

    base_values = dict(
        dimension=2,
        initial_output_shift=(0.3, -0.2),
        initial_output_scale_log=(-1.2, -0.8),
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature="2" * 64,
        s_max=2.0,
        learning_rate=1.0e-2,
        learning_rate_schedule="paper_piecewise",
        paper_piecewise_boundaries=(99, 399),
        gradient_clip_mode="none",
        kernel_initialization="paper_variance_scaling",
        scale_transform="identity",
        epsilon=1.0e-8,
        jit_compile=False,
    )
    saved_trainer = NeuTraReverseKLTrainer(
        Target(), ssl_lstm_pure_neutra_config(**base_values)
    )
    saved_trainer.step.assign(500)
    assert saved_trainer.optimizer is not None
    saved_trainer.optimizer.iterations.assign(500)
    state = saved_trainer.state_payload()

    prospective_config = ssl_lstm_pure_neutra_config(
        **{**base_values, "paper_piecewise_boundaries": (99, 399, 499)}
    )
    prospective = NeuTraReverseKLTrainer(Target(), prospective_config)
    with pytest.raises(NeuTraTrainingError, match="config mismatch"):
        prospective.restore_state(state)
    prospective.restore_state(state, allow_paper_schedule_extension=True)
    assert int(prospective.step.numpy()) == 500
    assert int(prospective.optimizer.iterations.numpy()) == 500
    for restored_value, saved_value in zip(
        prospective.variables, saved_trainer.variables
    ):
        np.testing.assert_array_equal(restored_value.numpy(), saved_value.numpy())
    for restored_value, saved_value in zip(
        prospective.optimizer.variables, saved_trainer.optimizer.variables
    ):
        np.testing.assert_array_equal(restored_value.numpy(), saved_value.numpy())
    assert float(prospective.learning_rate_at(499).numpy()) == pytest.approx(1.0e-4)
    assert float(prospective.learning_rate_at(500).numpy()) == pytest.approx(1.0e-5)

    rejected_configs = (
        {**base_values, "paper_piecewise_boundaries": (99, 399, 498)},
        {**base_values, "paper_piecewise_boundaries": (99, 499, 599)},
        {**base_values, "paper_piecewise_boundaries": (99,)},
        {
            **base_values,
            "paper_piecewise_boundaries": (99, 399, 499),
            "initialization_seed": (7, 8),
        },
    )
    for rejected_values in rejected_configs:
        rejected = NeuTraReverseKLTrainer(
            Target(), ssl_lstm_pure_neutra_config(**rejected_values)
        )
        with pytest.raises(NeuTraTrainingError, match="config mismatch"):
            rejected.restore_state(state, allow_paper_schedule_extension=True)


def test_generic_pure_paper_family_uses_dimension_width_and_no_affine() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: "2" * 64
        config = type(
            "Config",
            (),
            {
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                }
            },
        )()

    config = pure_paper_neutra_config(
        dimension=2,
        initial_output_shift=(0.3, -0.2),
        initial_output_scale_log=(-1.2, -0.8),
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature="2" * 64,
        jit_compile=False,
    )
    assert public_pure_paper_neutra_config is pure_paper_neutra_config
    assert config.family == PURE_PAPER_NEUTRA_FAMILY
    trainer = NeuTraReverseKLTrainer(Target(), config)
    assert config.hidden_layers == (2, 2)
    assert config.gradient_clip_mode == "none"
    assert config.scale_transform == "identity"
    assert float(trainer.learning_rate_at(0).numpy()) == pytest.approx(1.0e-2)
    with pytest.raises(ValueError, match="preset mismatch"):
        NeuTraTrainerConfig(
            **{
                **config.__dict__,
                "learning_rate_schedule": "adaptive_constant",
                "learning_rate": 1.0e-3,
                "gradient_clip_mode": "per_variable",
                "kernel_initialization": "legacy_zero_output",
                "scale_transform": "bounded_tanh",
            }
        )
    payload = trainer.frozen_transport_payload(
        transport_id="generic-pure-paper-test",
        target_signature=TARGET_SIGNATURE,
    )
    assert payload["procedure"] == "bayesfilter_pure_paper_dense_iaf_v1"
    assert [component["kind"] for component in payload["components"]] == [
        "dense_autoregressive_iaf",
        "mixing_linear",
        "dense_autoregressive_iaf",
        "mixing_linear",
        "dense_autoregressive_iaf",
    ]
    assert not payload["fixed_translation"]
    assert not payload["fixed_output_scale"]
    assert not payload["fixed_output_factor"]
    for field, value in (
        ("fixed_translation", [0.0, 0.0]),
        ("fixed_output_scale", [1.0, 1.0]),
        ("fixed_output_factor", [[1.0, 0.0], [0.0, 1.0]]),
    ):
        tampered = dict(payload)
        tampered[field] = value
        tampered = finalize_dense_iaf_neutra_artifact_payload(tampered)
        with pytest.raises(InvalidNeuTraArtifact, match="must not contain fixed affine"):
            load_frozen_neutra_artifact(
                tampered,
                expected_target_signature=TARGET_SIGNATURE,
            )

    bounded_scale = copy.deepcopy(payload)
    dense_component = next(
        component
        for component in bounded_scale["components"]
        if component["kind"] == "dense_autoregressive_iaf"
    )
    dense_component["scale_transform"] = "bounded_tanh"
    dense_component["s_max"] = 5.0
    bounded_scale = finalize_dense_iaf_neutra_artifact_payload(bounded_scale)
    with pytest.raises(InvalidNeuTraArtifact, match="scale transform mismatch"):
        load_frozen_neutra_artifact(
            bounded_scale,
            expected_target_signature=TARGET_SIGNATURE,
        )

    before = tuple(variable.numpy().copy() for variable in trainer.variables)
    initial_final_bias = before[-1].copy()
    trainer.train_step(_base_rows())
    assert all(variable.trainable for variable in trainer.variables)
    assert any(
        not np.array_equal(old, variable.numpy())
        for old, variable in zip(before, trainer.variables)
    )
    assert not np.array_equal(initial_final_bias, trainer.variables[-1].numpy())
    loaded = load_frozen_neutra_artifact(
        payload,
        expected_target_signature=TARGET_SIGNATURE,
    )
    z = tf.constant([[0.2, -0.4], [1.0, -1.0]], tf.float64)
    np.testing.assert_allclose(
        loaded.transport.inverse_theta_to_z_batch(
            loaded.transport.forward_z_to_theta_batch(z)
        ).numpy(),
        z.numpy(),
        rtol=0.0,
        atol=2e-12,
    )


def test_pure_bounded_family_contract_and_scale_envelope() -> None:
    class Target(DiagonalGaussianTarget):
        parameter_dim = 2
        parameter_names = ("x", "y")
        target_signature = lambda self: TARGET_SIGNATURE
        adapter_signature = lambda self: "2" * 64
        config = type(
            "Config",
            (),
            {
                "signature_payload": lambda self: {
                    "parameter_transform": {
                        "orientation": "identity",
                        "inverse_orientation": "identity",
                    }
                }
            },
        )()

    initial_scale = (math.log(0.002), math.log(0.004))
    config = pure_bounded_neutra_config(
        dimension=2,
        initial_output_shift=(0.3, -0.2),
        initial_output_scale_log=initial_scale,
        target_parameter_names=("x", "y"),
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature="2" * 64,
        s_max=6.5,
        jit_compile=False,
    )
    assert config.family == PURE_BOUNDED_NEUTRA_FAMILY
    assert config.hidden_layers == (2, 2)
    assert config.scale_transform == "bounded_tanh"
    assert config.manifest_payload()["scale_transform"] == "bounded_tanh"
    with pytest.raises(ValueError, match="s_max must exceed"):
        pure_bounded_neutra_config(
            dimension=2,
            initial_output_shift=(0.3, -0.2),
            initial_output_scale_log=initial_scale,
            target_parameter_names=("x", "y"),
            target_signature=TARGET_SIGNATURE,
            target_adapter_signature="2" * 64,
            s_max=6.0,
            jit_compile=False,
        )

    trainer = NeuTraReverseKLTrainer(Target(), config)
    z = tf.constant(
        [[-100.0, 100.0], [-10.0, 10.0], [0.0, 0.0], [10.0, -10.0]],
        tf.float64,
    )
    scale_log = trainer.transport.scale_log(z)
    assert bool(tf.reduce_all(tf.math.is_finite(scale_log)).numpy())
    assert float(tf.reduce_max(tf.abs(scale_log)).numpy()) <= config.s_max + 2.0e-12
    assert bool(tf.reduce_all(tf.math.is_finite(tf.exp(scale_log))).numpy())
    theta, logdet = trainer.forward_and_logdet(z)
    assert bool(tf.reduce_all(tf.math.is_finite(theta)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(logdet)).numpy())

    @tf.function(jit_compile=True)
    def compiled(values):
        output, output_logdet = trainer.forward_and_logdet(values)
        return output, output_logdet, trainer.transport.scale_log(values)

    try:
        compiled_theta, compiled_logdet, compiled_scale_log = compiled(z)
    except (tf.errors.InvalidArgumentError, tf.errors.UnimplementedError) as exc:
        pytest.fail(f"bounded pure NeuTra XLA contract failed: {exc}")
    np.testing.assert_allclose(compiled_theta.numpy(), theta.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(compiled_logdet.numpy(), logdet.numpy(), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        compiled_scale_log.numpy(), scale_log.numpy(), rtol=1e-10, atol=1e-10
    )

    payload = trainer.frozen_transport_payload(
        transport_id="pure-bounded-contract-test",
        target_signature=TARGET_SIGNATURE,
    )
    assert payload["procedure"] == "bayesfilter_pure_bounded_dense_iaf_v1"
    dense_components = [
        component
        for component in payload["components"]
        if component["kind"] == "dense_autoregressive_iaf"
    ]
    assert len(dense_components) == 3
    assert all(component["scale_transform"] == "bounded_tanh" for component in dense_components)
    assert all(component["s_max"] == pytest.approx(6.5) for component in dense_components)
    assert all(component["kind"] not in {"affine", "affine_dense"} for component in payload["components"])
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=TARGET_SIGNATURE)
    np.testing.assert_allclose(
        loaded.transport.inverse_theta_to_z_batch(
            loaded.transport.forward_z_to_theta_batch(z)
        ).numpy(),
        z.numpy(),
        rtol=0.0,
        atol=3.0e-10,
    )
    target_score = -theta
    expected_score = loaded.transport.pullback_score_batch(z, target_score)
    expected_score += loaded.transport.log_abs_det_jacobian_score_batch(z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        tape_theta, tape_logdet = loaded.transport.forward_z_to_theta_batch(z), loaded.transport.log_abs_det_jacobian_batch(z)
        transformed = -0.5 * tf.reduce_sum(tf.square(tape_theta), axis=-1) + tape_logdet
    direct_score = tape.gradient(transformed, z, output_gradients=tf.ones_like(transformed))
    np.testing.assert_allclose(expected_score.numpy(), direct_score.numpy(), rtol=2e-9, atol=2e-9)

    missing_scale_transform = copy.deepcopy(payload)
    for component in missing_scale_transform["components"]:
        if component["kind"] == "dense_autoregressive_iaf":
            component.pop("scale_transform", None)
    missing_scale_transform = finalize_dense_iaf_neutra_artifact_payload(missing_scale_transform)
    with pytest.raises(InvalidNeuTraArtifact, match="declaration missing"):
        load_frozen_neutra_artifact(
            missing_scale_transform,
            expected_target_signature=TARGET_SIGNATURE,
        )

    bad_initialization = copy.deepcopy(payload)
    bad_initialization["initial_output_scale_log"] = [-7.0, -7.0]
    bad_initialization = finalize_dense_iaf_neutra_artifact_payload(bad_initialization)
    with pytest.raises(InvalidNeuTraArtifact, match="represent initialization"):
        load_frozen_neutra_artifact(
            bad_initialization,
            expected_target_signature=TARGET_SIGNATURE,
        )

    affine_tamper = copy.deepcopy(payload)
    affine_tamper["fixed_translation"] = [0.0, 0.0]
    affine_tamper = finalize_dense_iaf_neutra_artifact_payload(affine_tamper)
    with pytest.raises(InvalidNeuTraArtifact, match="must not contain fixed affine"):
        load_frozen_neutra_artifact(affine_tamper, expected_target_signature=TARGET_SIGNATURE)
