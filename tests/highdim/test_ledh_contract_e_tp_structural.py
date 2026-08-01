from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import ledh_contract_e_tp_structural_tf as structural_tp
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp
from bayesfilter.structural import StatePartition, StructuralFilterConfig
from bayesfilter.structural_tf import TFStructuralStateSpace


DTYPE = tf.float64
THETA = tf.constant([0.70, 0.25, 0.55, 0.80], DTYPE)
PARENTS = tf.constant(
    [[-0.8, 0.2], [-0.1, -0.3], [0.6, 0.4], [1.0, -0.2]], DTYPE
)
INNOVATIONS = tf.constant([[-1.2], [0.0], [1.2]], DTYPE)
PARENT_WEIGHTS = tf.constant([0.15, 0.35, 0.30, 0.20], DTYPE)
INNOVATION_WEIGHTS = tf.constant([0.25, 0.50, 0.25], DTYPE)


def _transition(previous: tf.Tensor, innovation: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    rho, sigma, alpha, beta = tf.unstack(theta)
    stochastic = rho * previous[:, 0] + sigma * innovation[:, 0]
    deterministic = alpha * previous[:, 1] + beta * tf.math.tanh(stochastic)
    return tf.stack([stochastic, deterministic], axis=1)


def _residual(
    previous: tf.Tensor,
    innovation: tf.Tensor,
    next_state: tf.Tensor,
    theta: tf.Tensor,
) -> tf.Tensor:
    del innovation
    _, _, alpha, beta = tf.unstack(theta)
    return (
        next_state[:, 1]
        - alpha * previous[:, 1]
        - beta * tf.math.tanh(next_state[:, 0])
    )[:, None]


def _model(theta: tf.Tensor = THETA) -> TFStructuralStateSpace:
    partition = StatePartition(
        state_names=("m", "k"),
        stochastic_indices=(0,),
        deterministic_indices=(1,),
        innovation_dim=1,
    )
    return TFStructuralStateSpace(
        partition=partition,
        config=StructuralFilterConfig(
            integration_space="innovation", deterministic_completion="required"
        ),
        initial_mean=tf.zeros([2], DTYPE),
        initial_covariance=tf.eye(2, dtype=DTYPE),
        innovation_covariance=tf.eye(1, dtype=DTYPE),
        observation_covariance=tf.eye(1, dtype=DTYPE),
        transition_fn=lambda previous, innovation: _transition(previous, innovation, theta),
        observation_fn=lambda state: (state[:, 0] + state[:, 1])[:, None],
        deterministic_residual_fn=lambda previous, innovation, state: _residual(
            previous, innovation, state, theta
        ),
        name="contract_e_tp_structural_singular_fixture",
    )


def _teacher(theta: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    parent_count = tf.shape(PARENTS)[0]
    innovation_count = tf.shape(INNOVATIONS)[0]
    previous = tf.repeat(PARENTS, innovation_count, axis=0)
    innovation = tf.tile(INNOVATIONS, [parent_count, 1])
    points = _transition(previous, innovation, theta)
    weights = tf.reshape(
        PARENT_WEIGHTS[:, None] * INNOVATION_WEIGHTS[None, :], [-1]
    )
    stochastic, deterministic = tf.unstack(points, axis=1)
    features = tf.stack(
        [
            tf.ones_like(stochastic),
            stochastic,
            tf.square(stochastic),
            stochastic + deterministic,
        ],
        axis=0,
    )
    return points, tf.math.log(weights), features


ACTIVE = tf.constant([1, 4, 6, 11], tf.int32)


def _row_scale() -> tf.Tensor:
    points, log_weights, features = _teacher(THETA)
    reduced = tp._dense_teacher_reduce_core(log_weights, features)
    return tf.maximum(
        tf.reduce_max(tf.abs(features), axis=1), tf.abs(reduced["target"])
    )


def test_structural_teacher_uses_innovation_space_and_exact_completion() -> None:
    model = _model()
    teacher = structural_tp.structural_parent_innovation_teacher_tf(
        model, PARENTS, INNOVATIONS
    )
    assert model.partition.stochastic_indices == (0,)
    assert model.partition.deterministic_indices == (1,)
    assert int(teacher["teacher_count"].numpy()) == 12
    assert bool(teacher["integration_space_innovation"].numpy())
    assert bool(teacher["deterministic_completion_required"].numpy())
    np.testing.assert_allclose(teacher["deterministic_residual"], 0.0, atol=2e-16)


def test_structural_completion_value_and_total_tangent_are_exact() -> None:
    repeated = tf.repeat(PARENTS, tf.shape(INNOVATIONS)[0], axis=0)
    innovations = tf.tile(INNOVATIONS, [tf.shape(PARENTS)[0], 1])
    result = structural_tp.structural_residual_jacobian_tf(
        _model(), repeated, innovations, THETA, _transition, _residual
    )
    np.testing.assert_allclose(result["residual"], 0.0, atol=2e-16)
    np.testing.assert_allclose(result["residual_jacobian"], 0.0, atol=4e-16)


def test_projection_selects_only_teacher_support_and_preserves_structural_features() -> None:
    points, log_weights, features = _teacher(THETA)
    result = tp._contract_e_tp_dense_square_forward_core(
        points, log_weights, features, ACTIVE, _row_scale()
    )
    np.testing.assert_allclose(
        result["student_points"], tf.gather(points, ACTIVE), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(result["matched_target"], result["target"], atol=2e-15)
    repeated = tf.gather(
        tf.repeat(PARENTS, tf.shape(INNOVATIONS)[0], axis=0), ACTIVE
    )
    innovations = tf.gather(
        tf.tile(INNOVATIONS, [tf.shape(PARENTS)[0], 1]), ACTIVE
    )
    residual = _residual(repeated, innovations, result["student_points"], THETA)
    np.testing.assert_allclose(residual, 0.0, atol=2e-16)
    assert float(result["minimum_weight"].numpy()) > 0.0


def test_structural_projection_tangent_matches_teacher_and_same_scalar_fd() -> None:
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(THETA)
        points, log_weights, features = _teacher(THETA)
        projected = tp._contract_e_tp_dense_square_forward_core(
            points, log_weights, features, ACTIVE, _row_scale()
        )
        teacher_target = tp._dense_teacher_reduce_core(log_weights, features)["target"]
        scalar = tf.math.log(projected["matched_target"][-1] + 2.0)
    projected_jacobian = tape.jacobian(projected["matched_target"], THETA)
    teacher_jacobian = tape.jacobian(teacher_target, THETA)
    score = tape.gradient(scalar, THETA)
    np.testing.assert_allclose(projected_jacobian, teacher_jacobian, rtol=2e-13, atol=2e-14)
    step = 1.0e-5
    finite_difference = []
    for index in range(4):
        direction = np.zeros(4)
        direction[index] = step

        def evaluate(theta: tf.Tensor) -> tf.Tensor:
            current_points, current_log_weights, current_features = _teacher(theta)
            current = tp._contract_e_tp_dense_square_forward_core(
                current_points,
                current_log_weights,
                current_features,
                ACTIVE,
                _row_scale(),
            )
            return tf.math.log(current["matched_target"][-1] + 2.0)

        finite_difference.append(
            float(
                (
                    evaluate(tf.constant(THETA.numpy() + direction, DTYPE))
                    - evaluate(tf.constant(THETA.numpy() - direction, DTYPE))
                ).numpy()
                / (2.0 * step)
            )
        )
    np.testing.assert_allclose(score, finite_difference, rtol=2e-8, atol=2e-10)


def test_hidden_full_state_structural_route_is_rejected() -> None:
    mixed = _model()
    full_state = TFStructuralStateSpace(
        partition=mixed.partition,
        config=StructuralFilterConfig(
            integration_space="full_state",
            deterministic_completion="approximate",
            approximation_label="artificial_full_state_noise_negative_control",
            allow_full_state_for_mixed=True,
        ),
        initial_mean=mixed.initial_mean,
        initial_covariance=mixed.initial_covariance,
        innovation_covariance=mixed.innovation_covariance,
        observation_covariance=mixed.observation_covariance,
        transition_fn=mixed.transition_fn,
        observation_fn=mixed.observation_fn,
        deterministic_residual_fn=mixed.deterministic_residual_fn,
    )
    with pytest.raises(ValueError, match="innovation integration"):
        structural_tp.structural_parent_innovation_teacher_tf(
            full_state, PARENTS, INNOVATIONS
        )


def test_owned_structural_tp_module_has_no_numpy_or_jitter() -> None:
    source = Path(structural_tp.__file__).read_text(encoding="utf-8")
    for forbidden in ("import numpy", "from numpy", ".numpy(", "jitter", "cholesky"):
        assert forbidden not in source.lower()
