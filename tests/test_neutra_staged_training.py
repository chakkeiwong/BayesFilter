"""Focused mechanics tests for generic five-stage NeuTra training."""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.neutra_staged_training import (
    NeuTraAdaptiveStagePolicy,
    NeuTraFiveStageSpec,
    NeuTraStageSpec,
    NeuTraStagedTrainingError,
    NeuTraVariableGroup,
    NeuTraVariablePart,
    dense_iaf_five_stage_spec,
    dense_iaf_five_stage_variable_groups,
    neutra_full_variable_masks,
    train_neutra_five_stage,
)
from bayesfilter.inference.neutra_weighted_training import (
    WeightedDenseIAFTransport,
    WeightedNeuTraConfig,
)


DTYPE = tf.float64


def _transport(*, stages: int = 2) -> WeightedDenseIAFTransport:
    return WeightedDenseIAFTransport(
        WeightedNeuTraConfig(
            dimension=3,
            hidden_layers=(6, 6),
            stages=stages,
            activation="tanh",
            s_max=2.0,
            stage_s_max=(2.0,) + (0.5,) * (stages - 1),
            stage_unbounded_scale_linear=(True,) + (False,) * (stages - 1),
            initialization_scale=0.02,
            initialization_seed=(20260815, 3101),
            learning_rate=1.0e-3,
            gradient_clip_norm=10.0,
            jit_compile=True,
        )
    )


def _selection(transport: WeightedDenseIAFTransport) -> tf.Tensor:
    latent = tf.random.stateless_normal((256, 3), seed=(20260815, 3102), dtype=DTYPE)
    physical, logdet = transport.forward_and_logdet(latent)
    target = -0.5 * tf.reduce_sum(tf.square(physical - 0.25), axis=1)
    return tf.reduce_mean(-target - logdet)


def _batch(_phase: str, update: int, candidate: int) -> tf.Tensor:
    return tf.random.stateless_normal(
        (32, 3),
        seed=(20260815 + candidate, 3200 + update),
        dtype=DTYPE,
    )


def _tiny_spec(stages: int = 2) -> NeuTraFiveStageSpec:
    return dense_iaf_five_stage_spec(
        stages=stages,
        learning_rates=(1.0e-3,),
        affine_updates=2,
        simple_updates=2,
        progressive_updates=2,
        joint_updates=2,
        checkpoint_every=1,
    )


def test_dense_group_adapter_partitions_every_parameter_without_overlap() -> None:
    transport = _transport(stages=3)
    groups = dense_iaf_five_stage_variable_groups(transport)
    assert [group.name for group in groups] == [
        "affine_location",
        "simple_linear_scale",
        "stage_0_residual",
        "stage_1",
        "stage_2",
    ]
    spec = dense_iaf_five_stage_spec(stages=3, learning_rates=(1.0e-3,))
    assert [phase.stage for phase in spec.optimizer_phases()] == [1, 2, 3, 3, 3, 4]
    assert spec.joint.active_groups == tuple(group.name for group in groups)


def test_full_variable_masks_cover_only_the_requested_groups() -> None:
    transport = _transport(stages=1)
    groups = dense_iaf_five_stage_variable_groups(transport)
    masks = neutra_full_variable_masks(
        transport=transport,
        variable_groups=groups,
        active_groups=("affine_location",),
    )
    assert len(masks) == len(transport.trainable_variables)
    output_bias_index = len(transport.stages[0].trainable_variables) - 2
    for index, mask in enumerate(masks):
        if index == output_bias_index:
            tf.debugging.assert_equal(mask[:3], tf.zeros((3,), DTYPE))
            tf.debugging.assert_equal(mask[3:], tf.ones((3,), DTYPE))
        else:
            tf.debugging.assert_equal(mask, tf.zeros_like(mask))


def test_affine_stage_changes_only_shift_half_of_first_output_bias() -> None:
    transport = _transport(stages=1)
    initial = tuple(tf.identity(variable) for variable in transport.trainable_variables)
    result = train_neutra_five_stage(
        transport=transport,
        target_log_prob_fn=lambda rows: -0.5
        * tf.reduce_sum(tf.square(rows - 0.25), axis=1),
        variable_groups=dense_iaf_five_stage_variable_groups(transport),
        spec=_tiny_spec(stages=1),
        latent_batch_fn=_batch,
        selection_loss_fn=_selection,
        validation_fn=lambda _transport: {"passed": True},
        jit_compile=True,
    )
    stage_one_state = result.stages[0].candidates[0].terminal_state
    output_bias_index = len(transport.stages[0].trainable_variables) - 2
    for index, (before, after) in enumerate(zip(initial, stage_one_state, strict=True)):
        if index == output_bias_index:
            tf.debugging.assert_equal(before[:3], after[:3])
            assert bool(tf.reduce_any(tf.not_equal(before[3:], after[3:])).numpy())
        else:
            tf.debugging.assert_equal(before, after)


def test_each_learning_rate_candidate_starts_from_identical_incoming_state() -> None:
    transport = _transport(stages=1)
    groups = dense_iaf_five_stage_variable_groups(transport)
    one_update = NeuTraStageSpec(
        "affine_location", 1, ("affine_location",), 1, (5.0e-4, 1.0e-3), 1
    )
    spec = NeuTraFiveStageSpec(
        affine=one_update,
        simple=NeuTraStageSpec(
            "simple_linear_scale",
            2,
            ("affine_location", "simple_linear_scale"),
            1,
            (1.0e-3,),
            1,
        ),
        progressive=(
            NeuTraStageSpec(
                "progressive_stage_0",
                3,
                ("affine_location", "simple_linear_scale", "stage_0_residual"),
                1,
                (1.0e-3,),
                1,
            ),
        ),
        joint=NeuTraStageSpec(
            "joint_fine_tune",
            4,
            ("affine_location", "simple_linear_scale", "stage_0_residual"),
            1,
            (1.0e-3,),
            1,
        ),
    )
    shared_batch = tf.random.stateless_normal((32, 3), seed=(20260815, 3301), dtype=DTYPE)
    result = train_neutra_five_stage(
        transport=transport,
        target_log_prob_fn=lambda rows: -0.5
        * tf.reduce_sum(tf.square(rows - 0.25), axis=1),
        variable_groups=groups,
        spec=spec,
        latent_batch_fn=lambda _phase, _update, _candidate: shared_batch,
        selection_loss_fn=_selection,
        validation_fn=lambda _transport: {"passed": True},
    )
    candidates = result.stages[0].candidates
    assert len(candidates) == 2
    assert candidates[0].selected_update == 1
    assert candidates[1].selected_update == 1
    initial = tuple(tf.identity(variable) for variable in _transport(stages=1).trainable_variables)
    for before, left, right in zip(
        initial,
        candidates[0].terminal_state,
        candidates[1].terminal_state,
        strict=True,
    ):
        left_delta = left - before
        right_delta = right - before
        tf.debugging.assert_near(
            right_delta,
            tf.constant(2.0, DTYPE) * left_delta,
            atol=2.0e-10,
            rtol=2.0e-5,
        )


def test_validation_cannot_mutate_transport() -> None:
    transport = _transport(stages=1)

    def mutate(active: WeightedDenseIAFTransport):
        active.trainable_variables[0].assign_add(tf.ones_like(active.trainable_variables[0]))
        return {"passed": False}

    with pytest.raises(tf.errors.InvalidArgumentError, match="validation must not mutate"):
        train_neutra_five_stage(
            transport=transport,
            target_log_prob_fn=lambda rows: -0.5 * tf.reduce_sum(tf.square(rows), axis=1),
            variable_groups=dense_iaf_five_stage_variable_groups(transport),
            spec=_tiny_spec(stages=1),
            latent_batch_fn=_batch,
            selection_loss_fn=_selection,
            validation_fn=mutate,
        )


def test_overlap_and_incomplete_joint_coverage_fail_closed() -> None:
    transport = _transport(stages=1)
    variable = transport.trainable_variables[0]
    with pytest.raises(ValueError, match="must not overlap"):
        train_neutra_five_stage(
            transport=transport,
            target_log_prob_fn=lambda rows: -0.5 * tf.reduce_sum(tf.square(rows), axis=1),
            variable_groups=(
                NeuTraVariableGroup("left", (NeuTraVariablePart(variable),)),
                NeuTraVariableGroup("right", (NeuTraVariablePart(variable),)),
            ),
            spec=NeuTraFiveStageSpec(
                affine=NeuTraStageSpec("a", 1, ("left",), 1, (1.0e-3,), 1),
                simple=NeuTraStageSpec("b", 2, ("left",), 1, (1.0e-3,), 1),
                progressive=(
                    NeuTraStageSpec("c", 3, ("left",), 1, (1.0e-3,), 1),
                ),
                joint=NeuTraStageSpec("d", 4, ("left",), 1, (1.0e-3,), 1),
            ),
            latent_batch_fn=_batch,
            selection_loss_fn=_selection,
            validation_fn=lambda _transport: {"passed": True},
        )


def test_nonfinite_target_fails_closed() -> None:
    transport = _transport(stages=1)
    with pytest.raises(NeuTraStagedTrainingError, match="nonfinite update"):
        train_neutra_five_stage(
            transport=transport,
            target_log_prob_fn=lambda rows: tf.fill(
                (tf.shape(rows)[0],), tf.constant(float("nan"), DTYPE)
            ),
            variable_groups=dense_iaf_five_stage_variable_groups(transport),
            spec=_tiny_spec(stages=1),
            latent_batch_fn=_batch,
            selection_loss_fn=lambda _transport: tf.constant(0.0, DTYPE),
            validation_fn=lambda _transport: {"passed": False},
        )


def test_adaptive_joint_phase_reduces_learning_rate_then_stops_on_plateau() -> None:
    transport = _transport(stages=1)
    spec = dense_iaf_five_stage_spec(
        stages=1,
        learning_rates=(1.0e-3,),
        affine_updates=1,
        simple_updates=1,
        progressive_updates=1,
        joint_updates=5,
        checkpoint_every=1,
        joint_adaptive_policy=NeuTraAdaptiveStagePolicy(
            minimum_updates=1,
            patience_checkpoints=1,
            minimum_improvement=0.0,
            learning_rate_reduction_factor=0.5,
            maximum_learning_rate_reductions=1,
        ),
    )
    result = train_neutra_five_stage(
        transport=transport,
        target_log_prob_fn=lambda rows: -0.5 * tf.reduce_sum(tf.square(rows), axis=1),
        variable_groups=dense_iaf_five_stage_variable_groups(transport),
        spec=spec,
        latent_batch_fn=_batch,
        selection_loss_fn=lambda _transport: tf.constant(0.0, DTYPE),
        validation_fn=lambda _transport: {"passed": False},
    )
    candidate = result.stages[-1].candidates[0]
    assert candidate.executed_updates == 2
    assert candidate.learning_rate_reductions == 1
    assert candidate.stop_reason == "plateau_after_maximum_lr_reductions"
    assert candidate.selected_update == 0
    assert [row[0] for row in candidate.checkpoint_history] == [1, 2]


def test_carry_selected_restores_optimizer_state_for_every_phase_candidate() -> None:
    transport = _transport(stages=1)
    spec = dense_iaf_five_stage_spec(
        stages=1,
        learning_rates=(5.0e-4, 1.0e-3),
        affine_updates=2,
        simple_updates=2,
        progressive_updates=2,
        joint_updates=2,
        checkpoint_every=1,
    )
    result = train_neutra_five_stage(
        transport=transport,
        target_log_prob_fn=lambda rows: -0.5
        * tf.reduce_sum(tf.square(rows - 0.25), axis=1),
        variable_groups=dense_iaf_five_stage_variable_groups(transport),
        spec=spec,
        latent_batch_fn=_batch,
        selection_loss_fn=_selection,
        validation_fn=lambda _transport: {"passed": True},
        optimizer_state_policy="carry_selected",
    )
    assert result.optimizer_state_policy == "carry_selected"
    assert result.stages[0].incoming_optimizer_iterations == 0
    for previous, current in zip(result.stages[:-1], result.stages[1:], strict=True):
        assert current.incoming_optimizer_iterations == previous.selected_optimizer_iterations
        for candidate in current.candidates:
            terminal_iteration = int(candidate.terminal_optimizer_state[0].numpy())
            assert terminal_iteration == (
                current.incoming_optimizer_iterations + candidate.executed_updates
            )


def test_carry_selected_requires_cumulative_variable_masks() -> None:
    transport = _transport(stages=1)
    groups = dense_iaf_five_stage_variable_groups(transport)
    spec = NeuTraFiveStageSpec(
        affine=NeuTraStageSpec(
            "affine_location", 1, ("affine_location",), 1, (1.0e-3,), 1
        ),
        simple=NeuTraStageSpec(
            "simple_linear_scale", 2, ("simple_linear_scale",), 1, (1.0e-3,), 1
        ),
        progressive=(
            NeuTraStageSpec(
                "progressive_stage_0",
                3,
                ("affine_location", "simple_linear_scale", "stage_0_residual"),
                1,
                (1.0e-3,),
                1,
            ),
        ),
        joint=NeuTraStageSpec(
            "joint_fine_tune",
            4,
            ("affine_location", "simple_linear_scale", "stage_0_residual"),
            1,
            (1.0e-3,),
            1,
        ),
    )
    with pytest.raises(ValueError, match="cumulative active variable masks"):
        train_neutra_five_stage(
            transport=transport,
            target_log_prob_fn=lambda rows: -0.5 * tf.reduce_sum(tf.square(rows), axis=1),
            variable_groups=groups,
            spec=spec,
            latent_batch_fn=_batch,
            selection_loss_fn=_selection,
            validation_fn=lambda _transport: {"passed": True},
            optimizer_state_policy="carry_selected",
        )
