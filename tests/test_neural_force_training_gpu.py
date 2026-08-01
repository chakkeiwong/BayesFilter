"""Trusted GPU/XLA canary for scalar residual-force training."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)

from bayesfilter.inference.neural_force_training import (  # noqa: E402
    ScalarResidualForceTrainingConfig,
    train_scalar_residual_force,
)


def test_scalar_force_training_is_batched_gpu_xla_and_reloadable(tmp_path):
    train = tf.random.stateless_normal([128, 2], (91, 1), dtype=tf.float64)
    heldout = tf.random.stateless_normal([48, 2], (91, 2), dtype=tf.float64)

    def values_forces(position):
        with tf.GradientTape() as tape:
            tape.watch(position)
            value = 0.5 * tf.reduce_sum(tf.square(position), axis=-1)
            value += 0.1 * tf.pow(position[:, 0], 2) * position[:, 1]
        return value, tape.gradient(
            value, position, output_gradients=tf.ones_like(value)
        )

    train_value, train_force = values_forces(train)
    heldout_value, heldout_force = values_forces(heldout)
    result = train_scalar_residual_force(
        train_positions=train,
        train_potentials=train_value,
        train_forces=train_force,
        heldout_positions=heldout,
        heldout_potentials=heldout_value,
        heldout_forces=heldout_force,
        config=ScalarResidualForceTrainingConfig(
            target_signature="a" * 64,
            transport_signature="b" * 64,
            dimension=2,
            hidden_layers=(4, 4),
            output_dir=tmp_path / "gpu-xla-training",
            seed=(20260717, 99),
            steps=12,
            batch_size=32,
            learning_rate=5.0e-3,
            heartbeat_every=3,
            device="/GPU:0",
            require_gpu=True,
        ),
    )
    assert result.metrics["heldout"]["predictions_all_finite"] is True
    assert result.runtime_metadata["jit_compile"] is True
    assert result.runtime_metadata["training_control_flow"] == "tf_while_loop"
    assert all("GPU" in value.upper() for value in result.runtime_metadata["variable_devices"])
    assert bool(GPU_POLICY["all_physical_devices_memory_growth"])
