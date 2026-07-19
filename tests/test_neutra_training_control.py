from __future__ import annotations

import copy

import pytest

from bayesfilter.inference.neutra_training_control import (
    NeuTraPlateauConfig,
    NeuTraPlateauController,
    NeuTraPlateauError,
    joint_training_checkpoint_payload,
    paired_one_sided_upper_bound,
    validate_joint_training_checkpoint,
)


STATE_A = "a" * 64
STATE_B = "b" * 64
STATE_C = "c" * 64


def _config(**changes):
    values = {
        "validation_check_every": 100,
        "patience_steps": 500,
        "max_steps": 2000,
        "initial_learning_rate": 1.0e-3,
        "learning_rate_factor": 0.5,
        "minimum_learning_rate_fraction": 1.0 / 16.0,
        "absolute_min_delta": 0.0,
        "one_sided_critical_value": 1.0,
        "saturation_max": 0.05,
    }
    values.update(changes)
    return NeuTraPlateauConfig(**values)


def _observe(
    controller,
    step,
    losses=(1.0, 2.0, 3.0, 4.0),
    state=STATE_A,
    **diagnostics,
):
    return controller.observe(
        step=step,
        per_sample_loss=losses,
        saturation_fraction=0.0,
        trainer_state_hash=state,
        **diagnostics,
    )


def test_paired_upper_bound_detects_meaningful_improvement() -> None:
    mean, upper = paired_one_sided_upper_bound(
        (1.0, 2.0, 3.0, 4.0),
        (0.5, 1.5, 2.5, 3.5),
        critical_value=1.0,
    )
    assert mean == pytest.approx(-0.5)
    assert upper == pytest.approx(-0.5)


def test_exact_n_reduction_and_2n_stop_without_resetting_patience() -> None:
    controller = NeuTraPlateauController(_config())
    assert _observe(controller, 0).kind == "initialize_best"
    for step in (100, 200, 300, 400):
        assert _observe(controller, step).kind == "continue"
    reduction = _observe(controller, 500)
    assert reduction.kind == "reduce_learning_rate"
    assert reduction.current_learning_rate == pytest.approx(5.0e-4)
    assert reduction.steps_since_best == 500
    for step in (600, 700, 800, 900):
        assert _observe(controller, step).kind == "continue"
    stopped = _observe(controller, 1000)
    assert stopped.should_stop is True
    assert stopped.stop_reason == "plateau_after_lr_repair"
    assert controller.learning_rate_reductions == 1


def test_improvement_after_reduction_starts_a_new_plateau() -> None:
    controller = NeuTraPlateauController(_config())
    _observe(controller, 0)
    for step in (100, 200, 300, 400, 500):
        action = _observe(controller, step)
    assert action.kind == "reduce_learning_rate"
    improved = _observe(
        controller,
        600,
        losses=(0.5, 1.5, 2.5, 3.5),
        state=STATE_B,
    )
    assert improved.kind == "improved"
    assert improved.best_step == 600
    assert improved.steps_since_best == 0
    for step in (700, 800, 900, 1000, 1100):
        action = _observe(
            controller,
            step,
            losses=(0.5, 1.5, 2.5, 3.5),
            state=STATE_B,
        )
    assert action.kind == "reduce_learning_rate"
    assert action.current_learning_rate == pytest.approx(2.5e-4)
    assert controller.learning_rate_reductions == 2


def test_insignificant_scalar_decrease_does_not_reset_patience() -> None:
    controller = NeuTraPlateauController(_config(absolute_min_delta=0.05))
    _observe(controller, 0)
    action = _observe(
        controller,
        100,
        losses=(0.99, 1.99, 2.99, 3.99),
        state=STATE_B,
    )
    assert action.meaningful_improvement is False
    assert action.best_step == 0
    assert action.steps_since_best == 100


def test_saturation_stops_and_cannot_replace_best() -> None:
    controller = NeuTraPlateauController(_config())
    _observe(controller, 0)
    action = controller.observe(
        step=100,
        per_sample_loss=(0.0, 0.0, 0.0, 0.0),
        saturation_fraction=0.051,
        trainer_state_hash=STATE_B,
    )
    assert action.stop_reason == "scale_saturation_above_cap"
    assert controller.best_trainer_state_hash == STATE_A
    stopped_state = controller.state_payload()
    resumed = NeuTraPlateauController(_config())
    resumed.restore_state(stopped_state)
    assert resumed.state_payload() == stopped_state


def test_lower_loss_support_invalid_checkpoint_cannot_become_best() -> None:
    controller = NeuTraPlateauController(_config())
    initial = _observe(controller, 0)
    rejected = _observe(
        controller,
        100,
        losses=(0.0, 0.0, 0.0, 0.0),
        state=STATE_B,
        moderate_shell_max_inverse_radius=5.53,
    )
    assert initial.checkpoint_eligible is True
    assert rejected.meaningful_improvement is False
    assert rejected.checkpoint_eligible is False
    assert rejected.checkpoint_eligibility_vetoes == (
        "moderate_shell_missing_support",
    )
    assert controller.best_step == 0
    assert controller.best_trainer_state_hash == STATE_A


def test_later_support_valid_checkpoint_can_initialize_export_candidate() -> None:
    controller = NeuTraPlateauController(_config())
    rejected = _observe(
        controller,
        0,
        moderate_shell_max_inverse_radius=5.53,
    )
    admitted = _observe(
        controller,
        100,
        losses=(2.0, 3.0, 4.0, 5.0),
        state=STATE_B,
        moderate_shell_max_inverse_radius=3.23,
    )
    assert rejected.kind == "checkpoint_ineligible"
    assert rejected.best_step is None
    assert admitted.kind == "initialize_best"
    assert admitted.meaningful_improvement is True
    assert admitted.best_step == 100
    assert controller.best_trainer_state_hash == STATE_B
    assert controller.steps_since_best == 0


def test_resume_preserves_support_admissibility_and_best_selection() -> None:
    uninterrupted = NeuTraPlateauController(_config())
    _observe(
        uninterrupted,
        0,
        moderate_shell_max_inverse_radius=5.53,
    )
    _observe(
        uninterrupted,
        100,
        losses=(0.0, 0.0, 0.0, 0.0),
        state=STATE_B,
        moderate_shell_max_inverse_radius=5.10,
    )
    checkpoint = uninterrupted.state_payload()
    resumed = NeuTraPlateauController(_config())
    resumed.restore_state(checkpoint)
    left = _observe(
        uninterrupted,
        200,
        losses=(0.5, 1.5, 2.5, 3.5),
        state=STATE_C,
        roundtrip_max_abs=2.0e-15,
        moderate_shell_max_inverse_radius=3.20,
    )
    right = _observe(
        resumed,
        200,
        losses=(0.5, 1.5, 2.5, 3.5),
        state=STATE_C,
        roundtrip_max_abs=2.0e-15,
        moderate_shell_max_inverse_radius=3.20,
    )
    assert left == right
    assert resumed.state_payload() == uninterrupted.state_payload()
    assert resumed.best_step == 200
    assert resumed.best_checkpoint_diagnostics == {
        "all_finite": True,
        "saturation_fraction": 0.0,
        "roundtrip_max_abs": 2.0e-15,
        "moderate_shell_max_inverse_radius": 3.20,
    }


def test_maximum_step_and_learning_rate_floor_are_explicit() -> None:
    maxed = NeuTraPlateauController(_config(max_steps=1000))
    _observe(maxed, 0)
    for step in range(100, 1001, 100):
        action = _observe(
            maxed,
            step,
            losses=tuple(value - step / 10000.0 for value in (1.0, 2.0, 3.0, 4.0)),
            state=STATE_B,
        )
    assert action.stop_reason == "maximum_steps_reached"

    boundary = NeuTraPlateauController(_config(max_steps=1000, patience_steps=500))
    _observe(boundary, 0)
    for step in range(100, 1001, 100):
        action = _observe(boundary, step)
    assert action.stop_reason == "maximum_steps_reached"
    assert boundary.learning_rate_reductions == 1

    floored = NeuTraPlateauController(
        _config(minimum_learning_rate_fraction=0.75)
    )
    _observe(floored, 0)
    for step in (100, 200, 300, 400, 500):
        action = _observe(floored, step)
    assert action.kind == "minimum_learning_rate_reached"
    assert action.current_learning_rate == pytest.approx(1.0e-3)
    assert floored.minimum_learning_rate_reached is True


def test_controller_resume_matches_uninterrupted_transitions() -> None:
    uninterrupted = NeuTraPlateauController(_config())
    _observe(uninterrupted, 0)
    for step in (100, 200, 300, 400, 500, 600, 700):
        _observe(uninterrupted, step)
    checkpoint = uninterrupted.state_payload()

    resumed = NeuTraPlateauController(_config())
    resumed.restore_state(checkpoint)
    for step in (800, 900, 1000):
        left = _observe(uninterrupted, step)
        right = _observe(resumed, step)
        assert left == right
    assert resumed.state_payload() == uninterrupted.state_payload()


def test_joint_checkpoint_binds_best_state_and_detects_tampering() -> None:
    current = {"state_hash": STATE_C, "config": {"family": "fixture"}}
    best = {"state_hash": STATE_A, "config": {"family": "fixture"}}
    controller = {"best_trainer_state_hash": STATE_A}
    payload = joint_training_checkpoint_payload(
        trainer_state=current,
        controller_state=controller,
        best_trainer_state=best,
    )
    validate_joint_training_checkpoint(payload)

    mismatched = copy.deepcopy(best)
    mismatched["state_hash"] = STATE_B
    with pytest.raises(NeuTraPlateauError, match="best trainer/controller"):
        joint_training_checkpoint_payload(
            trainer_state=current,
            controller_state=controller,
            best_trainer_state=mismatched,
        )
    tampered = copy.deepcopy(payload)
    tampered["trainer_state"]["state_hash"] = STATE_B
    with pytest.raises(NeuTraPlateauError, match="joint checkpoint hash"):
        validate_joint_training_checkpoint(tampered)


def test_invalid_or_partial_controller_state_fails_without_mutation() -> None:
    controller = NeuTraPlateauController(_config())
    _observe(controller, 0)
    before = controller.state_payload()
    tampered = copy.deepcopy(before)
    tampered["best_step"] = 100
    with pytest.raises(NeuTraPlateauError, match="state_hash"):
        controller.restore_state(tampered)
    assert controller.state_payload() == before
