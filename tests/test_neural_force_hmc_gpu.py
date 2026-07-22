"""Trusted GPU/XLA canary for the corrected neural-force HMC kernel."""

from __future__ import annotations

import inspect

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


GPU_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)

from bayesfilter.inference.neural_force_hmc import (  # noqa: E402
    FrozenPositionOnlyForce,
    FrozenTargetPotential,
    NeuralForceHMCConfig,
    neural_force_hmc_transition,
    sample_neural_force_hmc,
)


def test_neural_force_hmc_gpu_xla_canary_and_memory_growth():
    config = NeuralForceHMCConfig(0.12, 6, (1.0, 0.5), dtype="float64")
    force = FrozenPositionOnlyForce(
        lambda position: position,
        identity="gpu-xla-gaussian-force",
    )
    target = FrozenTargetPotential(
        lambda position: 0.5 * tf.reduce_sum(tf.square(position), axis=-1),
        identity="gpu-xla-gaussian-target",
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def run(position, potential, seed):
        return neural_force_hmc_transition(
            position, potential, force, target, config, seed
        )

    position = tf.constant([[0.2, -0.4], [1.0, 0.3], [-0.5, 0.8]], tf.float64)
    potential = 0.5 * tf.reduce_sum(tf.square(position), axis=-1)
    with tf.device("/GPU:0"):
        first = run(position, potential, tf.constant([2026, 717], tf.int32))
        replay = run(position, potential, tf.constant([2026, 717], tf.int32))
    tf.debugging.assert_all_finite(first.position, "GPU/XLA position")
    tf.debugging.assert_all_finite(first.trace.delta_h, "GPU/XLA delta H")
    tf.debugging.assert_equal(first.position, replay.position)
    tf.debugging.assert_equal(first.trace.delta_h, replay.trace.delta_h)
    assert bool(GPU_POLICY["all_physical_devices_memory_growth"])
    assert first.position.device.endswith("GPU:0")


def test_sample_loop_compiles_with_xla_on_gpu():
    config = NeuralForceHMCConfig(0.1, 3, (1.0,), dtype="float64")
    force = FrozenPositionOnlyForce(lambda position: position, "gpu-chain-force")
    target = FrozenTargetPotential(
        lambda position: 0.5 * tf.reduce_sum(tf.square(position), axis=-1),
        "gpu-chain-target",
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def run(position, potential, seed):
        return sample_neural_force_hmc(
            position,
            potential,
            force,
            target,
            config,
            num_warmup=4,
            num_results=8,
            seed=seed,
        )

    initial = tf.zeros([4, 1], tf.float64)
    with tf.device("/GPU:0"):
        chain = run(initial, tf.zeros([4], tf.float64), tf.constant([33, 44], tf.int32))
    assert chain.positions.shape == (12, 4, 1)
    tf.debugging.assert_equal(tf.reduce_all(chain.finite_status), True)
    assert chain.positions.device.endswith("GPU:0")
    source = inspect.getsource(sample_neural_force_hmc)
    assert "tf.while_loop" in source
    assert "numpy" not in source
