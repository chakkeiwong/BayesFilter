"""Reference checks for the compiled CPU worker's complete target."""

import json

import pytest
import tensorflow as tf

from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import (
    batch_native_complexity_posterior_target,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import complexity_posterior_target


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("backend", ["tensorflow_eigh", "tensorflow_eigh_strict_factor_cached"])
def test_batch_worker_target_preserves_valid_scalar_rows(jit, backend):
    rows = tf.constant([[0.35, -0.08, 0.65, 0.05], [0.37, -0.06, 0.63, 0.07]], tf.float64)
    scalar = complexity_posterior_target(1, jit_compile=False)
    target = batch_native_complexity_posterior_target(
        1, jit_compile=jit, principal_sqrt_backend=backend
    )
    call = tf.function(
        target.neutra_batch_log_prob_and_grad_status,
        input_signature=[tf.TensorSpec([2, 4], tf.float64)],
        jit_compile=jit,
    )
    value, score, status = call(rows)
    diagnostics = {key: tensor.numpy().tolist() for key, tensor in status.items()}
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"])), json.dumps(diagnostics)
    reference = [scalar.eager_value_and_score(row) for row in tf.unstack(rows)]
    tf.debugging.assert_near(value, tf.stack([item[0] for item in reference]), atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(score, tf.stack([item[1] for item in reference]), atol=1e-9, rtol=1e-9)
