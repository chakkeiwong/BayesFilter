from __future__ import annotations

import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.hmc_coordinates import (
    AffineCoordinateTransform,
    WarmupTrajectoryPolicy,
)
from bayesfilter.inference.hmc_tuning import WindowedMassAdaptationConfig
from bayesfilter.inference.hmc_warmup import run_operational_windowed_warmup
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth


class _XLAWarmupGaussian:
    parameter_dim = 2

    def log_prob_and_grad(self, theta: tf.Tensor):
        value = tf.convert_to_tensor(theta, dtype=tf.float64)
        precision = tf.constant([[1.0, 0.2], [0.2, 2.0]], dtype=tf.float64)
        score = -tf.linalg.matvec(precision, value)
        return -0.5 * tf.reduce_sum(value * -score, axis=-1), score

    def adapter_signature(self) -> str:
        return "operational-warmup-xla-gaussian-v1"


def test_operational_warmup_executes_under_trusted_xla() -> None:
    if not tf.config.list_physical_devices("GPU"):
        pytest.skip("trusted GPU is required for operational warmup XLA validation")
    policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    assert policy["configured_before_logical_device_initialization"] is True
    transform = AffineCoordinateTransform(
        center=np.zeros(2),
        factor=np.eye(2),
        covariance_signature="operational-warmup-xla-identity",
    )
    result = run_operational_windowed_warmup(
        adapter=_XLAWarmupGaussian(),
        initial_transform=transform,
        initial_canonical_theta=np.array([0.2, -0.1]),
        initial_step_size=0.2,
        trajectory_policy=WarmupTrajectoryPolicy(2, 8),
        config=WindowedMassAdaptationConfig(
            warmup_steps=20,
            initial_buffer=2,
            final_buffer=8,
            first_window_size=10,
            min_window_samples=2,
        ),
        target_accept_prob=0.70,
        seed=(20260721, 701),
        target_scope="operational_warmup_xla_gaussian",
        chain_execution_mode="tf_function",
        jit_compile=True,
    )
    assert result.final_kernel_state.epsilon is not None
    assert len(result.windows) == 3
    assert all(np.isfinite(window.mean_acceptance_probability) for window in result.windows)
    assert all(window.state_map_residual <= 1.0e-10 for window in result.windows)
    assert result.every_update_used_by_later_transition is True
