from __future__ import annotations

import tensorflow as tf

from bayesfilter.testing.zhao_cui_actual_sv_neutra_target_tf import (
    SCORE_BACKEND_ID,
    make_actual_sv_zc_neutra_adapter,
)


def test_svx_same_program_adapter_exposes_admitted_manual_score_metadata() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()

    capability = adapter.value_score_capability()
    assert adapter.runtime_autodiff_for_hmc is False
    assert adapter.score_backend_id == SCORE_BACKEND_ID
    assert (
        capability.runtime_backend
        == "tensorflow_batched_fixed_adjacent_squared_tt_actual_sv_same_program_manual_score"
    )
    assert capability.xla_hmc_ready is True
    assert capability.full_chain_xla_diagnostic_ready is True
    assert capability.target_scope == "SVX-ZC-T10-d10-r2-o25-center-frozen-ukf-v1"


def test_svx_same_program_adapter_batch_wiring_is_finite_and_has_required_status() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    theta = tf.constant(
        [
            [0.2533471031357998, -0.4054651081081643],
            [0.2000000000000000, -0.35667494393873245],
            [0.3000000000000000, -0.4307829160924542],
        ],
        dtype=tf.float64,
    )

    value, score, status = adapter.neutra_batch_log_prob_and_grad_status(theta)

    assert value.shape == (3,)
    assert score.shape == (3, 2)
    assert bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())
    assert set(status) >= {
        "status_code",
        "valid_pre_regularized_score",
        "floor_count_value",
        "min_innovation_eigenvalue",
        "innovation_condition_estimate",
    }
    assert bool(tf.reduce_all(tf.equal(status["status_code"], 0)).numpy())
    assert bool(tf.reduce_all(status["valid_pre_regularized_score"]).numpy())


def test_svx_same_program_adapter_score_matches_own_fd() -> None:
    adapter = make_actual_sv_zc_neutra_adapter()
    theta = tf.constant([[0.6, 0.4]], dtype=tf.float64)

    value, score, _status = adapter.neutra_batch_log_prob_and_grad_status(theta)

    epsilon = tf.constant(1.0e-6, tf.float64)
    columns = []
    for index in range(2):
        basis = tf.one_hot(index, 2, dtype=tf.float64)[None, :]
        plus = adapter.log_prob(theta + epsilon * basis)[0]
        minus = adapter.log_prob(theta - epsilon * basis)[0]
        columns.append((plus - minus) / (2.0 * epsilon))
    finite_difference = tf.stack(columns, axis=0)[None, :]

    assert bool(tf.math.is_finite(value[0]).numpy())
    tf.debugging.assert_near(score, finite_difference, atol=2e-6, rtol=2e-6)
