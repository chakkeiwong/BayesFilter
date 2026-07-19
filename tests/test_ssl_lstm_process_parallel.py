"""Focused contracts for the SSL-LSTM process-parallel training boundary."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.inference.cpu_value_score_pool import (
    CPUValueScorePool,
    CPUValueScorePoolConfig,
)
from bayesfilter.inference.neutra_training import (
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (
    PRIOR_CENTER,
    complexity_posterior_target,
)


def _pool_config(q: int = 1, worker_count: int = 2) -> CPUValueScorePoolConfig:
    return CPUValueScorePoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_target_tf:"
            "complexity_target_worker_factory"
        ),
        worker_config={"q": q},
        dimension=4,
        worker_count=worker_count,
    )


def test_process_pool_matches_scalar_eager_target_and_records_cpu_workers():
    target = complexity_posterior_target(1, jit_compile=False)
    rows = np.asarray(
        [
            [0.35, -0.08, 0.65, 0.05],
            [0.37, -0.06, 0.63, 0.07],
        ],
        dtype=np.float64,
    )
    expected = [target.eager_value_and_score(row) for row in rows]
    expected_values = np.asarray([float(value.numpy()) for value, _ in expected])
    expected_scores = np.asarray([score.numpy() for _, score in expected])
    with CPUValueScorePool(_pool_config()) as pool:
        values, scores, metadata = pool.evaluate(rows, request_id="test-request")
    np.testing.assert_allclose(values, expected_values, rtol=1.0e-9, atol=1.0e-9)
    np.testing.assert_allclose(scores, expected_scores, rtol=1.0e-8, atol=1.0e-8)
    assert metadata["request_id"] == "test-request"
    assert metadata["backend"] == (
        "persistent_cpu_worker_value_score_custom_gradient_bridge"
    )
    assert metadata["worker_pids"]
    assert all(record["cuda_visible_devices"] == "-1" for record in metadata["worker_metadata"])
    assert all(record["tensorflow_gpu_devices"] == [] for record in metadata["worker_metadata"])


def test_external_score_bridge_updates_transport_without_target_batch_call():
    target = complexity_posterior_target(1, jit_compile=False)
    config = ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=tuple(float(value) for value in PRIOR_CENTER.numpy()),
        target_parameter_names=target.parameter_names,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=4.0e-4,
        initialization_scale=0.01,
        gradient_clip_norm=10.0,
        initialization_seed=(20260719, 9111),
        jit_compile=False,
    )
    trainer = NeuTraReverseKLTrainer(target, config)
    z = tf.constant([[0.0, 0.0, 0.0, 0.0], [0.1, -0.1, 0.05, 0.02]], tf.float64)
    theta, _ = trainer.forward_and_logdet(z)
    values, scores = target.batch_value_and_score(theta)
    before = int(trainer.step.numpy())
    result = trainer.train_step_with_external_value_score(z, values, scores)
    assert int(trainer.step.numpy()) == before + 1
    assert bool(tf.reduce_all(tf.math.is_finite(result.loss)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(result.gradient_norm)).numpy())
