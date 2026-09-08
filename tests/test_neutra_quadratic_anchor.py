from __future__ import annotations

import copy
import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_artifacts import (
    InvalidNeuTraArtifact,
    finalize_dense_iaf_neutra_artifact_payload,
    load_frozen_neutra_artifact,
)
from bayesfilter.inference.neutra_training import (
    QUADRATIC_ANCHOR_INITIALIZATION_MODE,
    NeuTraReverseKLTrainer,
    pure_paper_neutra_config,
)


TARGET_SIGNATURE = "1" * 64
ADAPTER_SIGNATURE = "2" * 64
ESTIMATOR_SIGNATURE = "3" * 64


class _GaussianTarget:
    parameter_dim = 3
    parameter_names = ("x", "y", "q")
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

    def target_signature(self):
        return TARGET_SIGNATURE

    def adapter_signature(self):
        return ADAPTER_SIGNATURE

    def batch_value_and_score(self, theta):
        center = tf.constant([0.1, -0.4, 0.2], tf.float64)
        delta = theta - center
        return -0.5 * tf.reduce_sum(tf.square(delta), axis=-1), -delta


_L = ((1.2, 0.0, 0.0), (0.3, 0.8, 0.0), (-0.2, 0.1, 1.5))
_MU = (0.4, -0.2, 0.7)
_LOG_DIAG = tuple(math.log(_L[i][i]) for i in range(3))
_Z = tf.constant([[0.2, -0.4, 1.1], [-1.0, 0.5, 0.3]], tf.float64)


def _config(*, scale_transform="identity", release_steps=2, jit_compile=False):
    return pure_paper_neutra_config(
        dimension=3,
        initial_output_shift=_MU,
        initial_output_scale_log=_LOG_DIAG,
        target_parameter_names=_GaussianTarget.parameter_names,
        target_signature=TARGET_SIGNATURE,
        target_adapter_signature=ADAPTER_SIGNATURE,
        initialization_mode=QUADRATIC_ANCHOR_INITIALIZATION_MODE,
        initial_anchor_factor=_L,
        anchor_release_steps=release_steps,
        anchor_estimator_signature=ESTIMATOR_SIGNATURE,
        scale_transform=scale_transform,
        s_max=2.0 if scale_transform == "dsge_bounded_tanh" else 1.0,
        jit_compile=jit_compile,
    )


def _anchor_state(trainer):
    return tuple(
        tuple(variable.numpy().copy() for variable in trainer.variables[index:index + 1])
        for index in trainer._anchor_variable_indices
    )


def test_quadratic_anchor_direct_log_is_exact_and_roundtrips():
    trainer = NeuTraReverseKLTrainer(_GaussianTarget(), _config())
    theta, logdet = trainer.forward_and_logdet(_Z)
    expected = tf.matmul(_Z, tf.transpose(tf.constant(_L, tf.float64))) + tf.constant(
        _MU, tf.float64
    )
    np.testing.assert_allclose(theta.numpy(), expected.numpy(), rtol=0.0, atol=1.0e-13)
    np.testing.assert_allclose(
        logdet.numpy(), np.sum(_LOG_DIAG), rtol=0.0, atol=1.0e-13
    )
    payload = trainer.frozen_transport_payload(
        transport_id="quadratic-anchor-test", target_signature=TARGET_SIGNATURE
    )
    assert payload["procedure"] == "bayesfilter_pure_paper_dense_iaf_quadratic_anchor_v1"
    assert payload["initialization_mode"] == QUADRATIC_ANCHOR_INITIALIZATION_MODE
    assert not any(
        component["kind"] in {"affine", "affine_dense"}
        for component in payload["components"]
    )
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=TARGET_SIGNATURE)
    np.testing.assert_allclose(
        loaded.transport.forward_z_to_theta_batch(_Z).numpy(),
        expected.numpy(),
        rtol=0.0,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        loaded.transport.inverse_theta_to_z_batch(expected).numpy(),
        _Z.numpy(),
        rtol=0.0,
        atol=1.0e-12,
    )


def test_quadratic_anchor_dsge_transform_uses_exact_unscaled_tanh():
    trainer = NeuTraReverseKLTrainer(
        _GaussianTarget(), _config(scale_transform="dsge_bounded_tanh")
    )
    theta, logdet = trainer.forward_and_logdet(_Z)
    expected = tf.matmul(_Z, tf.transpose(tf.constant(_L, tf.float64))) + tf.constant(
        _MU, tf.float64
    )
    np.testing.assert_allclose(theta.numpy(), expected.numpy(), rtol=0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        logdet.numpy(), np.sum(_LOG_DIAG), rtol=0.0, atol=2.0e-12
    )
    dense = [
        component
        for component in trainer.frozen_transport_payload(
            transport_id="quadratic-anchor-dsge", target_signature=TARGET_SIGNATURE
        )["components"]
        if component["kind"] == "dense_autoregressive_iaf"
    ]
    assert all(component["scale_transform"] == "dsge_bounded_tanh" for component in dense)
    assert np.max(np.abs(trainer.transport.scale_log(_Z).numpy())) < 2.0


def test_quadratic_anchor_release_masks_direct_external_and_chunk_routes():
    target = _GaussianTarget()
    for route in ("direct", "external", "chunks"):
        trainer = NeuTraReverseKLTrainer(target, _config(release_steps=1))
        before = tuple(variable.numpy().copy() for variable in trainer.variables)
        before_m = tuple(variable.numpy().copy() for variable in trainer.optimizer._momentums)
        before_v = tuple(variable.numpy().copy() for variable in trainer.optimizer._velocities)
        theta, _ = trainer.forward_and_logdet(_Z)
        values, scores = target.batch_value_and_score(theta)

        def update():
            if route == "direct":
                return trainer.train_step(_Z)
            if route == "external":
                return trainer.train_step_with_external_value_score(_Z, values, scores)
            return trainer.train_step_with_external_value_score_chunks(
                (_Z[:1], _Z[1:]),
                (values[:1], values[1:]),
                (scores[:1], scores[1:]),
                (1, 1),
            )

        update()  # current step 0: anchor remains frozen
        for index in trainer._anchor_variable_indices:
            np.testing.assert_array_equal(before[index], trainer.variables[index].numpy())
        for index in trainer._anchor_variable_indices:
            # Keras Adam slots must not decay while the anchor is held.
            np.testing.assert_array_equal(before_m[index], trainer.optimizer._momentums[index].numpy())
            np.testing.assert_array_equal(before_v[index], trainer.optimizer._velocities[index].numpy())
        assert any(
            not np.array_equal(before[index], trainer.variables[index].numpy())
            for index in range(len(trainer.variables))
            if index not in trainer._anchor_variable_indices
        )

        anchor_after_warmup = tuple(
            trainer.variables[index].numpy().copy() for index in trainer._anchor_variable_indices
        )
        update()  # current step 1: first released update
        assert any(
            not np.array_equal(old, trainer.variables[index].numpy())
            for old, index in zip(anchor_after_warmup, trainer._anchor_variable_indices)
        )


def test_quadratic_anchor_artifact_score_chain_and_hash_rejection():
    trainer = NeuTraReverseKLTrainer(_GaussianTarget(), _config(release_steps=0))
    payload = trainer.frozen_transport_payload(
        transport_id="quadratic-anchor-score", target_signature=TARGET_SIGNATURE
    )
    loaded = load_frozen_neutra_artifact(payload, expected_target_signature=TARGET_SIGNATURE)
    theta = loaded.transport.forward_z_to_theta_batch(_Z)
    target_score = -theta
    expected = loaded.transport.pullback_score_batch(_Z, target_score)
    expected += loaded.transport.log_abs_det_jacobian_score_batch(_Z)
    with tf.GradientTape() as tape:
        tape.watch(_Z)
        mapped = loaded.transport.forward_z_to_theta_batch(_Z)
        transformed = -0.5 * tf.reduce_sum(tf.square(mapped), axis=-1)
    direct = tape.gradient(transformed, _Z, output_gradients=tf.ones_like(transformed))
    np.testing.assert_allclose(expected.numpy(), direct.numpy(), rtol=0.0, atol=2.0e-11)

    tampered = copy.deepcopy(payload)
    tampered["anchor_factor_orientation"] = "column_lower_cholesky"
    tampered = finalize_dense_iaf_neutra_artifact_payload(tampered)
    with pytest.raises(InvalidNeuTraArtifact, match="factor orientation"):
        load_frozen_neutra_artifact(tampered, expected_target_signature=TARGET_SIGNATURE)


def test_quadratic_anchor_xla_train_step():
    trainer = NeuTraReverseKLTrainer(
        _GaussianTarget(), _config(release_steps=1, jit_compile=True)
    )
    first = trainer.train_step(_Z)
    second = trainer.train_step(_Z)
    assert int(first.step.numpy()) == 1
    assert int(second.step.numpy()) == 2
