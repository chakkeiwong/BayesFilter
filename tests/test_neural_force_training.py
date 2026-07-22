"""Reference tests for graph-native scalar residual-force training."""

from __future__ import annotations

import inspect

import pytest
import tensorflow as tf

from bayesfilter.inference.neural_force_training import (
    NeuralForceTrainingError,
    ScalarResidualForceTrainingConfig,
    ScalarResidualPotentialNetwork,
    evaluate_scalar_force,
    load_frozen_scalar_residual_force,
    train_scalar_residual_force,
)


TARGET = "a" * 64
TRANSPORT = "b" * 64


def _value_force(kind: str, positions: tf.Tensor):
    positions = tf.convert_to_tensor(positions, tf.float64)
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(positions)
        x = positions[:, 0]
        y = positions[:, 1]
        if kind == "gaussian":
            value = 0.5 * tf.square(x / 1.4) + 0.5 * tf.square(y / 0.7)
        elif kind == "banana":
            curved = y + 0.25 * (tf.square(x) - 1.0)
            value = 0.5 * tf.square(x) + 0.5 * tf.square(curved / 0.7)
        elif kind == "funnel":
            value = 0.5 * tf.square(x / 2.0) + 0.5 * tf.square(y) * tf.exp(-x) + 0.5 * x
        else:
            raise ValueError(kind)
    force = tape.gradient(value, positions, output_gradients=tf.ones_like(value))
    return value, force


def _data(kind: str):
    train = tf.random.stateless_uniform(
        [256, 2], (2026, 10), minval=-1.5, maxval=1.5, dtype=tf.float64
    )
    heldout = tf.random.stateless_uniform(
        [96, 2], (2026, 11), minval=-1.8, maxval=1.8, dtype=tf.float64
    )
    train_value, train_force = _value_force(kind, train)
    heldout_value, heldout_force = _value_force(kind, heldout)
    return train, train_value, train_force, heldout, heldout_value, heldout_force


def _config(tmp_path, *, name: str, steps: int = 120):
    return ScalarResidualForceTrainingConfig(
        target_signature=TARGET,
        transport_signature=TRANSPORT,
        dimension=2,
        hidden_layers=(8, 8),
        output_dir=tmp_path / name,
        seed=(20260717, 52000),
        steps=steps,
        batch_size=32,
        learning_rate=5.0e-3,
        heartbeat_every=20,
        device="/CPU:0",
        require_gpu=False,
    )


@pytest.mark.parametrize("kind", ["gaussian", "banana", "funnel"])
def test_scalar_force_training_analytic_fixtures(tmp_path, kind):
    data = _data(kind)
    target_scale = tf.maximum(tf.math.reduce_std(data[2], axis=0), 1.0e-6)
    zero_residual_rmse = tf.sqrt(
        tf.reduce_mean(tf.square((data[3] - data[5]) / target_scale))
    )
    result = train_scalar_residual_force(
        train_positions=data[0],
        train_potentials=data[1],
        train_forces=data[2],
        heldout_positions=data[3],
        heldout_potentials=data[4],
        heldout_forces=data[5],
        config=_config(tmp_path, name=kind),
    )
    assert result.metrics["heldout"]["predictions_all_finite"] is True
    assert result.metrics["heldout"]["standardized_force_rmse"] < float(
        zero_residual_rmse
    )
    assert result.runtime_metadata["jit_compile"] is True
    assert result.runtime_metadata["training_control_flow"] == "tf_while_loop"
    assert result.runtime_metadata["sample_axis_python_loop_used"] is False


def test_exported_force_is_gradient_of_exported_scalar_and_reload_matches(tmp_path):
    data = _data("banana")
    config = _config(tmp_path, name="gradient-reload", steps=40)
    result = train_scalar_residual_force(
        train_positions=data[0],
        train_potentials=data[1],
        train_forces=data[2],
        heldout_positions=data[3],
        heldout_potentials=data[4],
        heldout_forces=data[5],
        config=config,
    )
    points = data[3][:12]
    with tf.GradientTape() as tape:
        tape.watch(points)
        potential = result.frozen.potential(points)
    direct_gradient = tape.gradient(
        potential, points, output_gradients=tf.ones_like(potential)
    )
    tf.debugging.assert_near(result.frozen.force(points), direct_gradient, atol=2e-10, rtol=2e-10)

    payload = __import__("json").loads(result.artifact_path.read_text(encoding="utf-8"))
    loaded = load_frozen_scalar_residual_force(
        payload,
        expected_target_signature=TARGET,
        expected_transport_signature=TRANSPORT,
    )
    tf.debugging.assert_equal(loaded.potential(points), result.frozen.potential(points))
    tf.debugging.assert_equal(loaded.force(points), result.frozen.force(points))


def test_centered_potential_metric_is_offset_invariant(tmp_path):
    del tmp_path
    data = _data("gaussian")
    network = ScalarResidualPotentialNetwork(
        dimension=2,
        hidden_layers=(4,),
        position_mean=tf.zeros([2], tf.float64),
        position_scale=tf.ones([2], tf.float64),
        seed=(1, 2),
        trainable=False,
    )
    from bayesfilter.inference.neural_force_training import FrozenScalarResidualForce

    frozen = FrozenScalarResidualForce(network, TARGET, TRANSPORT, "c" * 64, "d" * 64)
    common = dict(
        frozen=frozen,
        positions=data[3],
        target_forces=data[5],
        force_scale=tf.math.reduce_std(data[2], axis=0),
        potential_scale=tf.math.reduce_std(data[1]),
    )
    original = evaluate_scalar_force(target_potentials=data[4], **common)
    shifted = evaluate_scalar_force(target_potentials=data[4] + 173.0, **common)
    assert original["standardized_force_rmse"] == shifted["standardized_force_rmse"]
    assert original["centered_standardized_potential_rmse"] == pytest.approx(
        shifted["centered_standardized_potential_rmse"], abs=1e-12
    )


def test_frozen_loader_rejects_target_transport_and_tensor_substitution(tmp_path):
    data = _data("gaussian")
    result = train_scalar_residual_force(
        train_positions=data[0],
        train_potentials=data[1],
        train_forces=data[2],
        heldout_positions=data[3],
        heldout_potentials=data[4],
        heldout_forces=data[5],
        config=_config(tmp_path, name="substitution", steps=5),
    )
    import json

    payload = json.loads(result.artifact_path.read_text(encoding="utf-8"))
    with pytest.raises(NeuralForceTrainingError, match="target signature"):
        load_frozen_scalar_residual_force(
            payload, expected_target_signature="e" * 64, expected_transport_signature=TRANSPORT
        )
    with pytest.raises(NeuralForceTrainingError, match="transport signature"):
        load_frozen_scalar_residual_force(
            payload, expected_target_signature=TARGET, expected_transport_signature="e" * 64
        )
    payload["network"]["weights"][0][0][0] += 0.1
    with pytest.raises(NeuralForceTrainingError, match="artifact signature"):
        load_frozen_scalar_residual_force(
            payload, expected_target_signature=TARGET, expected_transport_signature=TRANSPORT
        )


def test_training_source_has_no_numpy_or_python_step_loop():
    import bayesfilter.inference.neural_force_training as module

    source = inspect.getsource(module)
    assert "import numpy" not in source
    assert "tf.while_loop" in source
    assert "numpy_function" not in source
    assert "py_function" not in source
