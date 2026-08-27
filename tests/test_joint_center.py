from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference.joint_center import (
    JOINT_CENTER_NONCLAIMS,
    JointCenterLocatorConfig,
    JointCenterStagedConfig,
    _standardized_objective_and_gradient,
    locate_joint_center,
    locate_joint_center_staged,
)


def test_joint_center_api_is_exported_from_inference_package() -> None:
    from bayesfilter import inference

    assert inference.JointCenterLocatorConfig is JointCenterLocatorConfig
    assert inference.locate_joint_center is locate_joint_center
    assert inference.JointCenterStagedConfig is JointCenterStagedConfig
    assert inference.locate_joint_center_staged is locate_joint_center_staged


def _quadratic_target(
    precision: np.ndarray, mode: np.ndarray
):
    precision_tf = tf.constant(precision, tf.float64)
    mode_tf = tf.constant(mode, tf.float64)

    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1])
        displacement = position - mode_tf
        score = -tf.linalg.matvec(precision_tf, displacement)
        value = -0.5 * tf.tensordot(
            displacement, tf.linalg.matvec(precision_tf, displacement), 1
        )
        return value, score

    return target


def test_config_defaults_to_xla_and_validates_positive_fields() -> None:
    config = JointCenterLocatorConfig()
    assert config.jit_compile is True
    assert config.num_correction_pairs == 10
    assert config.max_iterations == 30
    assert config.max_line_search_iterations == 20
    assert config.gradient_tolerance == 0.02
    assert config.max_objective_evaluations == 601
    assert config.parallel_iterations == 1
    assert config.max_wall_seconds is None
    with pytest.raises(ValueError, match="max_iterations"):
        JointCenterLocatorConfig(max_iterations=0)
    with pytest.raises(ValueError, match="gradient_tolerance"):
        JointCenterLocatorConfig(gradient_tolerance=-1.0)
    with pytest.raises(ValueError, match="max_objective_evaluations"):
        JointCenterLocatorConfig(max_objective_evaluations=0)
    with pytest.raises(ValueError, match="requires jit_compile=False"):
        JointCenterLocatorConfig(max_wall_seconds=1.0)


def test_initial_position_and_scale_require_rank_one_vectors() -> None:
    target = _quadratic_target(np.eye(2), np.zeros(2))
    with pytest.raises(ValueError, match="initial_position must be a rank-1 vector"):
        locate_joint_center(target, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="scale must be a rank-1 vector"):
        locate_joint_center(target, np.zeros(2), scale=np.ones((1, 2)))


def test_nonidentity_standardized_chain_rule_matches_autodiff() -> None:
    initial = tf.constant([0.3, -0.2], tf.float64)
    scale = tf.constant([0.5, 2.0], tf.float64)
    z = tf.constant([0.4, -0.1], tf.float64)
    target = _quadratic_target(
        np.array([[2.0, 0.3], [0.3, 1.2]]), np.array([0.1, 0.25])
    )
    objective, gradient = _standardized_objective_and_gradient(
        target, initial, scale, z
    )
    with tf.GradientTape() as tape:
        tape.watch(z)
        theta = initial + scale * z
        value, _ = target(theta)
        reference_objective = -value
    reference_gradient = tape.gradient(reference_objective, z)
    np.testing.assert_allclose(objective.numpy(), reference_objective.numpy())
    np.testing.assert_allclose(gradient.numpy(), reference_gradient.numpy())


@pytest.mark.parametrize("jit_compile", [False, True])
def test_rotated_quadratic_recovers_mode_with_exact_accounting(
    jit_compile: bool,
) -> None:
    precision = np.array([[2.0, 0.4], [0.4, 1.3]])
    mode = np.array([0.2, -0.15])
    result = locate_joint_center(
        _quadratic_target(precision, mode),
        np.array([-0.4, 0.3]),
        scale=np.array([0.5, 2.0]),
        config=JointCenterLocatorConfig(
            gradient_tolerance=1.0e-9,
            max_iterations=30,
            max_objective_evaluations=100,
            jit_compile=jit_compile,
        ),
    )
    assert result.status == "converged"
    assert result.endpoint_accepted is True
    assert result.optimizer_converged is True
    assert result.optimizer_failed is False
    assert result.cap_exhausted is False
    assert result.wall_time_exhausted is False
    assert result.reported_objective_evaluations == result.callback_attempts
    assert result.callback_attempts == result.optimizer_target_rows
    assert result.physical_target_rows == result.optimizer_target_rows + 2
    np.testing.assert_allclose(result.endpoint_position, mode, atol=1.0e-7)
    assert result.best_evaluated_position is not None
    np.testing.assert_allclose(result.best_evaluated_position, mode, atol=1.0e-7)
    assert result.best_evaluated_objective is not None
    assert result.best_evaluated_objective >= result.endpoint_objective - 1.0e-12
    assert result.endpoint_score_max_abs <= 1.0e-7


def test_default_xla_and_non_xla_endpoints_match() -> None:
    precision = np.array([[1.7, -0.2], [-0.2, 0.9]])
    mode = np.array([0.25, -0.4])
    target = _quadratic_target(precision, mode)
    common = dict(
        gradient_tolerance=1.0e-9,
        max_iterations=30,
        max_objective_evaluations=100,
    )
    graph = locate_joint_center(
        target,
        np.zeros(2),
        scale=np.array([0.25, 3.0]),
        config=JointCenterLocatorConfig(jit_compile=False, **common),
    )
    xla = locate_joint_center(
        target,
        np.zeros(2),
        scale=np.array([0.25, 3.0]),
        config=JointCenterLocatorConfig(jit_compile=True, **common),
    )
    assert graph.status == xla.status == "converged"
    np.testing.assert_allclose(graph.endpoint_position, xla.endpoint_position)
    np.testing.assert_allclose(
        graph.best_evaluated_position, xla.best_evaluated_position
    )
    assert graph.best_evaluated_objective == pytest.approx(
        xla.best_evaluated_objective, abs=1.0e-12
    )
    assert graph.best_evaluated_source == xla.best_evaluated_source
    assert graph.reported_objective_evaluations == xla.reported_objective_evaluations
    assert graph.optimizer_target_rows == xla.optimizer_target_rows


def test_hard_cap_guards_attempt_602_before_target_evaluation() -> None:
    result = locate_joint_center(
        _quadratic_target(np.eye(2), np.array([1.0, -1.0])),
        np.zeros(2),
        config=JointCenterLocatorConfig(
            gradient_tolerance=1.0e-12,
            max_iterations=10,
            max_line_search_iterations=10,
            max_objective_evaluations=1,
            jit_compile=False,
        ),
    )
    assert result.status == "evaluation_cap_exhausted"
    assert result.endpoint_accepted is False
    assert result.cap_exhausted is True
    assert result.callback_attempts > 1
    assert result.reported_objective_evaluations == result.callback_attempts
    assert result.optimizer_target_rows == 1
    assert result.physical_target_rows == 3


def test_non_xla_wall_guard_returns_typed_timeout_without_target_overrun() -> None:
    result = locate_joint_center(
        _quadratic_target(np.eye(2), np.array([1.0, -1.0])),
        np.zeros(2),
        config=JointCenterLocatorConfig(
            max_wall_seconds=1.0e-12,
            jit_compile=False,
        ),
    )
    assert result.status == "wall_time_exhausted"
    assert result.endpoint_accepted is False
    assert result.wall_time_exhausted is True
    assert result.cap_exhausted is False
    assert result.optimizer_target_rows == 0
    assert result.callback_attempts > result.optimizer_target_rows
    assert result.reported_objective_evaluations == result.callback_attempts
    assert result.physical_target_rows == 2


def test_nonfinite_initial_target_fails_closed_before_optimizer() -> None:
    def target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1])
        return tf.constant(np.nan, tf.float64), tf.zeros_like(position)

    result = locate_joint_center(target, np.zeros(2))
    assert result.status == "initial_target_invalid"
    assert result.endpoint_accepted is False
    assert result.reported_objective_evaluations == 0
    assert result.callback_attempts == 0
    assert result.optimizer_target_rows == 0
    assert result.physical_target_rows == 1


def test_optimizer_exception_falls_back_without_endpoint_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import joint_center

    def raising_optimizer(*args, **kwargs):
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(
        joint_center.tfp.optimizer, "lbfgs_minimize", raising_optimizer
    )
    result = locate_joint_center(
        _quadratic_target(np.eye(2), np.ones(2)),
        np.zeros(2),
        config=JointCenterLocatorConfig(jit_compile=False),
    )
    assert result.status == "optimizer_exception"
    assert result.endpoint_accepted is False
    assert result.exception_type == "RuntimeError"
    np.testing.assert_array_equal(result.endpoint_position, np.zeros(2))


def test_failed_optimizer_result_is_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import joint_center

    def failed_optimizer(function, initial_position, **kwargs):
        from tensorflow_probability.python.optimizer.lbfgs import (
            LBfgsOptimizerResults,
        )

        value, gradient = function(initial_position)
        return LBfgsOptimizerResults(
            converged=tf.constant(False),
            failed=tf.constant(True),
            num_iterations=tf.constant(1),
            num_objective_evaluations=tf.constant(1),
            position=initial_position,
            objective_value=value,
            objective_gradient=gradient,
            position_deltas=tf.zeros([10, 2], tf.float64),
            gradient_deltas=tf.zeros([10, 2], tf.float64),
        )

    monkeypatch.setattr(
        joint_center.tfp.optimizer, "lbfgs_minimize", failed_optimizer
    )
    result = locate_joint_center(
        _quadratic_target(np.eye(2), np.ones(2)),
        np.zeros(2),
        config=JointCenterLocatorConfig(jit_compile=False),
    )
    assert result.status == "optimizer_failed"
    assert result.endpoint_accepted is False


def test_best_exact_internal_callback_point_survives_lower_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import joint_center
    from tensorflow_probability.python.optimizer.lbfgs import (
        LBfgsOptimizerResults,
    )

    def endpoint_after_better_interior(function, initial_position, **kwargs):
        function(initial_position)
        function(tf.ones_like(initial_position))
        endpoint = tf.fill(tf.shape(initial_position), tf.constant(2.0, tf.float64))
        value, gradient = function(endpoint)
        return LBfgsOptimizerResults(
            converged=tf.constant(True),
            failed=tf.constant(False),
            num_iterations=tf.constant(1),
            num_objective_evaluations=tf.constant(3),
            position=endpoint,
            objective_value=value,
            objective_gradient=gradient,
            position_deltas=tf.zeros([10, 1], tf.float64),
            gradient_deltas=tf.zeros([10, 1], tf.float64),
        )

    monkeypatch.setattr(
        joint_center.tfp.optimizer,
        "lbfgs_minimize",
        endpoint_after_better_interior,
    )
    result = locate_joint_center(
        _quadratic_target(np.eye(1), np.ones(1)),
        np.zeros(1),
        config=JointCenterLocatorConfig(jit_compile=False),
    )

    np.testing.assert_array_equal(result.endpoint_position, np.array([2.0]))
    np.testing.assert_array_equal(result.best_evaluated_position, np.array([1.0]))
    np.testing.assert_array_equal(result.best_evaluated_score, np.array([0.0]))
    assert result.best_evaluated_objective == pytest.approx(0.0)
    assert result.best_evaluated_source == "optimizer_callback"
    assert result.best_evaluated_callback_index == 1
    assert result.best_evaluated_is_endpoint is False
    assert result.payload()["best_evaluated_source"] == "optimizer_callback"


def test_converged_finite_sentinel_zero_score_is_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import joint_center
    from tensorflow_probability.python.optimizer.lbfgs import (
        LBfgsOptimizerResults,
    )

    def finite_reject_target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1])
        valid = tf.reduce_all(tf.abs(position) <= 1.0)
        valid_value = -0.5 * tf.reduce_sum(tf.square(position - 0.5))
        return tf.cond(
            valid,
            lambda: (valid_value, -(position - 0.5)),
            lambda: (tf.constant(-1.0e30, tf.float64), tf.zeros_like(position)),
        )

    def false_converged(function, initial_position, **kwargs):
        value, gradient = function(initial_position)
        return LBfgsOptimizerResults(
            converged=tf.constant(True),
            failed=tf.constant(False),
            num_iterations=tf.constant(1),
            num_objective_evaluations=tf.constant(1),
            position=tf.fill(tf.shape(initial_position), tf.constant(10.0, tf.float64)),
            objective_value=value,
            objective_gradient=gradient,
            position_deltas=tf.zeros([10, 2], tf.float64),
            gradient_deltas=tf.zeros([10, 2], tf.float64),
        )

    monkeypatch.setattr(
        joint_center.tfp.optimizer, "lbfgs_minimize", false_converged
    )
    result = locate_joint_center(
        finite_reject_target,
        np.zeros(2),
        config=JointCenterLocatorConfig(jit_compile=False),
    )
    assert result.optimizer_converged is True
    assert result.status == "endpoint_objective_decrease"
    assert result.endpoint_accepted is False
    np.testing.assert_array_equal(result.endpoint_position, np.full(2, 10.0))
    assert result.endpoint_objective == pytest.approx(-1.0e30)
    assert result.payload()["objective_nondecreasing"] is False


def test_result_payload_is_array_free_and_has_no_geometry_authority() -> None:
    result = locate_joint_center(
        _quadratic_target(np.eye(2), np.array([0.1, -0.2])),
        np.zeros(2),
        config=JointCenterLocatorConfig(jit_compile=False),
    )
    payload = result.payload()
    assert tuple(payload["nonclaims"]) == JOINT_CENTER_NONCLAIMS
    assert "endpoint_position" not in payload
    assert "initial_position" not in payload
    forbidden = {
        "precision",
        "covariance",
        "mass_matrix",
        "inverse_hessian",
        "position_deltas",
        "gradient_deltas",
        "step_size",
        "num_leapfrog_steps",
    }
    assert forbidden.isdisjoint(payload)
    assert result.endpoint_position.flags.writeable is False
    assert result.initial_position.flags.writeable is False
    assert result.endpoint_score.flags.writeable is False
    assert result.initial_score.flags.writeable is False


def test_staged_config_requires_strictly_larger_total_budget() -> None:
    config = JointCenterStagedConfig()
    assert config.checkpoint_iterations == 30
    assert config.total_iterations == 60
    assert config.max_objective_evaluations == 600
    with pytest.raises(ValueError, match="total_iterations"):
        JointCenterStagedConfig(checkpoint_iterations=30, total_iterations=30)


def test_staged_checkpoint_is_private_immutable_and_validated_once() -> None:
    calls = []

    def validator(checkpoint) -> bool:
        calls.append(checkpoint)
        return True

    result = locate_joint_center_staged(
        _quadratic_target(np.array([[2.0, 0.3], [0.3, 1.0]]), np.array([0.4, -0.2])),
        np.zeros(2),
        checkpoint_validator=validator,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=10,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
        ),
    )
    assert len(calls) == 1
    assert result.checkpoint_validator_calls == 1
    assert result.checkpoint_validated is True
    assert result.continuation_started is True
    assert result.checkpoint.position.flags.writeable is False
    assert result.checkpoint.score.flags.writeable is False
    checkpoint_payload = result.checkpoint.payload()
    assert "position" not in checkpoint_payload
    assert "score" not in checkpoint_payload
    public = result.payload()
    assert "position_deltas" not in str(public)
    assert "gradient_deltas" not in str(public)
    assert "inverse_hessian" not in str(public)
    assert "mass_matrix" not in str(public)


def test_staged_locator_retains_best_internal_callback_across_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import joint_center
    from tensorflow_probability.python.optimizer.lbfgs import (
        LBfgsOptimizerResults,
    )

    calls = 0

    def endpoint_after_better_interior(
        function,
        initial_position,
        previous_optimizer_results=None,
        **_kwargs,
    ):
        nonlocal calls
        calls += 1
        if previous_optimizer_results is None:
            function(initial_position)
            function(tf.ones_like(initial_position))
            endpoint = tf.fill(
                tf.shape(initial_position), tf.constant(2.0, tf.float64)
            )
            value, gradient = function(endpoint)
            evaluations = tf.constant(3)
            iterations = tf.constant(1)
        else:
            endpoint = previous_optimizer_results.position
            value = previous_optimizer_results.objective_value
            gradient = previous_optimizer_results.objective_gradient
            evaluations = previous_optimizer_results.num_objective_evaluations
            iterations = tf.constant(2)
        return LBfgsOptimizerResults(
            converged=tf.constant(True),
            failed=tf.constant(False),
            num_iterations=iterations,
            num_objective_evaluations=evaluations,
            position=endpoint,
            objective_value=value,
            objective_gradient=gradient,
            position_deltas=tf.zeros([10, 1], tf.float64),
            gradient_deltas=tf.zeros([10, 1], tf.float64),
        )

    monkeypatch.setattr(
        joint_center.tfp.optimizer,
        "lbfgs_minimize",
        endpoint_after_better_interior,
    )
    result = locate_joint_center_staged(
        _quadratic_target(np.eye(1), np.ones(1)),
        np.zeros(1),
        checkpoint_validator=lambda checkpoint: True,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=2,
            jit_compile=False,
        ),
    )

    assert calls == 2
    np.testing.assert_array_equal(result.endpoint_position, np.array([2.0]))
    np.testing.assert_array_equal(result.best_evaluated_position, np.array([1.0]))
    np.testing.assert_array_equal(result.best_evaluated_score, np.array([0.0]))
    assert result.best_evaluated_objective == pytest.approx(0.0)
    assert result.best_evaluated_source == "optimizer_callback"
    assert result.best_evaluated_callback_index == 1
    assert result.best_evaluated_is_endpoint is False


def test_checkpoint_rejection_prevents_continuation_target_calls() -> None:
    result = locate_joint_center_staged(
        _quadratic_target(np.eye(2), np.array([0.5, -0.25])),
        np.zeros(2),
        checkpoint_validator=lambda checkpoint: False,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=10,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
        ),
    )
    assert result.status == "checkpoint_rejected"
    assert result.checkpoint_validated is False
    assert result.checkpoint_validator_calls == 1
    assert result.continuation_started is False
    assert result.optimizer_target_rows == result.checkpoint.optimizer_target_rows
    assert result.physical_target_rows == result.optimizer_target_rows + 2


def test_checkpoint_validator_exception_prevents_continuation() -> None:
    def raising_validator(checkpoint) -> bool:
        raise RuntimeError("synthetic checkpoint failure")

    result = locate_joint_center_staged(
        _quadratic_target(np.array([[4.0, 0.7], [0.7, 0.5]]), np.array([1.0, -1.0])),
        np.zeros(2),
        checkpoint_validator=raising_validator,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=10,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
        ),
    )
    assert result.status == "checkpoint_validator_exception"
    assert result.checkpoint_validator_calls == 1
    assert result.continuation_started is False
    assert result.exception_type == "RuntimeError"
    assert result.optimizer_target_rows == result.checkpoint.optimizer_target_rows


def test_same_state_continuation_matches_one_shot_optimizer() -> None:
    target = _quadratic_target(
        np.array([[4.0, 0.7, 0.0], [0.7, 1.5, 0.2], [0.0, 0.2, 0.3]]),
        np.array([0.6, -0.4, 0.25]),
    )
    initial = np.array([-0.8, 0.5, -0.2])
    scale = np.array([0.5, 2.0, 1.5])
    staged = locate_joint_center_staged(
        target,
        initial,
        scale=scale,
        checkpoint_validator=lambda checkpoint: True,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=10,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
        ),
    )
    one_shot = locate_joint_center(
        target,
        initial,
        scale=scale,
        config=JointCenterLocatorConfig(
            max_iterations=10,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
        ),
    )
    np.testing.assert_allclose(
        staged.endpoint_position, one_shot.endpoint_position, rtol=0.0, atol=1.0e-12
    )
    np.testing.assert_allclose(
        staged.endpoint_score, one_shot.endpoint_score, rtol=0.0, atol=1.0e-12
    )
    assert staged.endpoint_objective == pytest.approx(
        one_shot.endpoint_objective, abs=1.0e-12
    )
    assert staged.reported_objective_evaluations == (
        one_shot.reported_objective_evaluations
    )
    assert staged.callback_attempts == one_shot.callback_attempts
    assert staged.optimizer_target_rows == one_shot.optimizer_target_rows
    assert staged.physical_target_rows == staged.optimizer_target_rows + 3


def test_staged_global_cap_can_fire_only_after_checkpoint() -> None:
    target = _quadratic_target(
        np.array([[4.0, 0.7], [0.7, 0.5]]), np.array([1.0, -1.0])
    )
    first_stage = locate_joint_center(
        target,
        np.zeros(2),
        config=JointCenterLocatorConfig(
            max_iterations=1,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
        ),
    )
    staged = locate_joint_center_staged(
        target,
        np.zeros(2),
        checkpoint_validator=lambda checkpoint: True,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=10,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=first_stage.optimizer_target_rows,
            jit_compile=False,
        ),
    )
    assert staged.checkpoint_validated is True
    assert staged.continuation_started is True
    assert staged.status == "evaluation_cap_exhausted"
    assert staged.cap_exhausted is True
    assert staged.optimizer_target_rows == first_stage.optimizer_target_rows
    assert staged.callback_attempts > staged.optimizer_target_rows
    assert staged.physical_target_rows == staged.optimizer_target_rows + 3


def test_staged_global_wall_guard_can_fire_only_after_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import joint_center

    after_checkpoint = False

    def fake_monotonic() -> float:
        return 10.0 if after_checkpoint else 0.0

    def validator(checkpoint) -> bool:
        nonlocal after_checkpoint
        after_checkpoint = True
        return True

    monkeypatch.setattr(joint_center.time, "monotonic", fake_monotonic)
    staged = locate_joint_center_staged(
        _quadratic_target(
            np.array([[4.0, 0.7], [0.7, 0.5]]), np.array([1.0, -1.0])
        ),
        np.zeros(2),
        checkpoint_validator=validator,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=10,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
            max_wall_seconds=1.0,
        ),
    )
    assert staged.checkpoint_validated is True
    assert staged.continuation_started is True
    assert staged.status == "wall_time_exhausted"
    assert staged.wall_time_exhausted is True
    assert staged.callback_attempts > staged.optimizer_target_rows


def test_staged_finite_sentinel_endpoint_is_not_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from bayesfilter.inference import joint_center
    from tensorflow_probability.python.optimizer.lbfgs import (
        LBfgsOptimizerResults,
    )

    def finite_reject_target(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        position = tf.reshape(tf.convert_to_tensor(theta, tf.float64), [-1])
        valid = tf.reduce_all(tf.abs(position) <= 1.0)
        value = -0.5 * tf.reduce_sum(tf.square(position - 0.5))
        return tf.cond(
            valid,
            lambda: (value, -(position - 0.5)),
            lambda: (tf.constant(-1.0e30, tf.float64), tf.zeros_like(position)),
        )

    original = joint_center.tfp.optimizer.lbfgs_minimize
    calls = 0

    def staged_optimizer(function, initial_position, previous_optimizer_results=None, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(
                function,
                initial_position,
                previous_optimizer_results=previous_optimizer_results,
                **kwargs,
            )
        previous = previous_optimizer_results
        return LBfgsOptimizerResults(
            converged=tf.constant(True),
            failed=tf.constant(False),
            num_iterations=tf.constant(2),
            num_objective_evaluations=previous.num_objective_evaluations,
            position=tf.fill(tf.shape(previous.position), tf.constant(10.0, tf.float64)),
            objective_value=previous.objective_value,
            objective_gradient=previous.objective_gradient,
            position_deltas=previous.position_deltas,
            gradient_deltas=previous.gradient_deltas,
        )

    monkeypatch.setattr(
        joint_center.tfp.optimizer, "lbfgs_minimize", staged_optimizer
    )
    result = locate_joint_center_staged(
        finite_reject_target,
        np.zeros(2),
        checkpoint_validator=lambda checkpoint: True,
        config=JointCenterStagedConfig(
            checkpoint_iterations=1,
            total_iterations=2,
            gradient_tolerance=1.0e-12,
            max_objective_evaluations=100,
            jit_compile=False,
        ),
    )
    assert result.status == "endpoint_objective_decrease"
    assert result.endpoint_accepted is False
    assert result.endpoint_objective == pytest.approx(-1.0e30)
