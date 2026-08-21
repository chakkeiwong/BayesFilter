"""Focused tests for TensorFlow NeuTra curriculum training primitives."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.inference.neutra_curriculum_training import (
    train_neutra_curriculum_protocol,
    tune_neutra_curriculum_probe,
)
from bayesfilter.inference.neutra_staged_training import (
    dense_iaf_five_stage_variable_groups,
)
from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)


def _transport() -> WeightedDenseIAFTransport:
    return WeightedDenseIAFTransport(
        WeightedNeuTraConfig(
            dimension=3,
            hidden_layers=(6, 6),
            stages=2,
            activation="tanh",
            s_max=2.0,
            stage_s_max=(2.0, 0.5),
            stage_unbounded_scale_linear=(True, False),
            initialization_scale=0.02,
            initialization_seed=(20260815, 7101),
            jit_compile=True,
        )
    )


def _latent(update: int) -> tf.Tensor:
    return tf.random.stateless_normal((32, 3), seed=(20260815, 7200 + update), dtype=tf.float64)


def _loss(transport: WeightedDenseIAFTransport) -> tf.Tensor:
    latent = tf.random.stateless_normal((128, 3), seed=(20260815, 7301), dtype=tf.float64)
    physical, logdet = transport.forward_and_logdet(latent)
    return tf.reduce_mean(0.5 * tf.reduce_sum(tf.square(physical - 0.2), axis=1) - logdet)


def test_probe_tunes_rates_from_same_incoming_state_and_restores_selected_state() -> None:
    transport = _transport()
    result = tune_neutra_curriculum_probe(
        transport=transport,
        target_log_prob_fn=lambda rows: -0.5 * tf.reduce_sum(tf.square(rows - 0.2), axis=1),
        variable_groups=dense_iaf_five_stage_variable_groups(transport),
        active_groups=("affine_location",),
        learning_rates=(5.0e-4, 1.0e-3),
        updates=2,
        latent_batch_fn=_latent,
        selection_loss_fn=_loss,
    )
    assert result.tuning_optimizer_updates == 4
    assert len(result.candidates) == 2
    assert result.selected_loss == min(item.terminal_loss for item in result.candidates)
    for variable, expected in zip(transport.trainable_variables, result.selected_state, strict=True):
        tf.debugging.assert_equal(variable, expected)


def test_protocol_consumes_exact_budget_and_preserves_global_step_ranges() -> None:
    transport = _transport()
    result = train_neutra_curriculum_protocol(
        transport=transport,
        target_log_prob_fn=lambda rows: -0.5 * tf.reduce_sum(tf.square(rows - 0.2), axis=1),
        variable_groups=dense_iaf_five_stage_variable_groups(transport),
        sequence=("affine_location", "stage_0_residual"),
        learning_rate=1.0e-3,
        total_updates=7,
        warmup_updates_per_group=2,
        latent_batch_fn=_latent,
        selection_loss_fn=_loss,
    )
    assert result.executed_updates == 7
    assert [(phase.first_global_update, phase.last_global_update) for phase in result.phases] == [
        (1, 2),
        (3, 4),
        (5, 7),
    ]
    assert result.phases[-1].name == "joint"


def test_cold_protocol_is_one_joint_phase_with_exact_budget() -> None:
    transport = _transport()
    result = train_neutra_curriculum_protocol(
        transport=transport,
        target_log_prob_fn=lambda rows: -0.5 * tf.reduce_sum(tf.square(rows), axis=1),
        variable_groups=dense_iaf_five_stage_variable_groups(transport),
        sequence=(),
        learning_rate=5.0e-4,
        total_updates=3,
        warmup_updates_per_group=1,
        latent_batch_fn=_latent,
        selection_loss_fn=_loss,
    )
    assert result.executed_updates == 3
    assert len(result.phases) == 1
    assert result.phases[0].active_groups == tuple(
        group.name for group in dense_iaf_five_stage_variable_groups(transport)
    )
