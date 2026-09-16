"""Structural teacher construction for experimental Contract E--TP."""

from __future__ import annotations

import tensorflow as tf

from bayesfilter.structural_tf import TFStructuralStateSpace
from bayesfilter.highdim import ledh_contract_e_tp_tf as tp


DTYPE = tf.float64
STRUCTURAL_FIXTURE_PARAMETER_COUNT = 4
STRUCTURAL_FIXTURE_STATE_DIM = 2
STRUCTURAL_FIXTURE_FEATURE_COUNT = 4
STRUCTURAL_FIXTURE_VALUE_OPERATION_COUNT = 16
STRUCTURAL_FIXTURE_TANGENT_OPERATION_COUNT = 64


def _gamma(operation_count: int) -> tf.Tensor:
    unit_roundoff = tf.constant(0.5 * 2.220446049250313e-16, DTYPE)
    count = tf.cast(operation_count, DTYPE)
    return count * unit_roundoff / (1.0 - count * unit_roundoff)


def _poison_invalid(value: tf.Tensor, valid: tf.Tensor) -> tf.Tensor:
    value = tf.convert_to_tensor(value)
    return tf.where(valid, value, tf.fill(tf.shape(value), tf.cast(float("nan"), value.dtype)))


def structural_fixture_transition_components_tf(
    previous: tf.Tensor, innovation: tf.Tensor, theta: tf.Tensor
) -> dict[str, tf.Tensor]:
    """Evaluate the frozen two-state structural fixture transition."""

    rho, sigma, alpha, beta = tf.unstack(tf.convert_to_tensor(theta, DTYPE))
    previous = tf.convert_to_tensor(previous, DTYPE)
    innovation = tf.convert_to_tensor(innovation, DTYPE)
    stochastic = rho * previous[:, 0] + sigma * innovation[:, 0]
    tanh_value = tf.math.tanh(stochastic)
    deterministic = alpha * previous[:, 1] + beta * tanh_value
    return {
        "candidates": tf.stack([stochastic, deterministic], axis=1),
        "stochastic_preactivation": stochastic,
        "tanh_value": tanh_value,
    }


def structural_fixture_features_tf(points: tf.Tensor) -> tf.Tensor:
    """Return the four frozen structural fixture features."""

    points = tf.convert_to_tensor(points, DTYPE)
    stochastic, deterministic = tf.unstack(points, axis=1)
    return tf.stack(
        [
            tf.ones_like(stochastic),
            stochastic,
            tf.square(stochastic),
            stochastic + deterministic,
        ],
        axis=0,
    )


def structural_fixture_recursive_core_tf(
    theta: tf.Tensor,
    initial_parents: tf.Tensor,
    initial_weights: tf.Tensor,
    innovations: tf.Tensor,
    innovation_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    support_perturbation: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Run the fixed-index structural fixture with one functional time loop."""

    theta = tf.reshape(
        tf.convert_to_tensor(theta, DTYPE), [STRUCTURAL_FIXTURE_PARAMETER_COUNT]
    )
    parents = tf.ensure_shape(
        tf.convert_to_tensor(initial_parents, DTYPE), [None, STRUCTURAL_FIXTURE_STATE_DIM]
    )
    parent_weights = tf.ensure_shape(
        tf.convert_to_tensor(initial_weights, DTYPE), [None]
    )
    innovations = tf.ensure_shape(tf.convert_to_tensor(innovations, DTYPE), [None, 1])
    innovation_weights = tf.ensure_shape(
        tf.convert_to_tensor(innovation_weights, DTYPE), [None]
    )
    active_indices = tf.convert_to_tensor(active_indices, tf.int32)
    row_scales = tf.convert_to_tensor(row_scales, DTYPE)
    support_perturbation = tf.reshape(
        tf.convert_to_tensor(support_perturbation, DTYPE), []
    )
    time_steps = active_indices.shape[0]
    parent_count = parents.shape[0]
    innovation_count = innovations.shape[0]
    if time_steps is None or time_steps < 1:
        raise ValueError("structural fixture requires a static positive horizon")
    if parent_count is None or innovation_count is None:
        raise ValueError("structural fixture requires static parent/innovation counts")
    if active_indices.shape != (time_steps, STRUCTURAL_FIXTURE_FEATURE_COUNT):
        raise ValueError("active_indices has the wrong structural fixture shape")
    if row_scales.shape != (time_steps, STRUCTURAL_FIXTURE_FEATURE_COUNT):
        raise ValueError("row_scales has the wrong structural fixture shape")
    teacher_count = parent_count * innovation_count

    parent_weights = parent_weights / tf.reduce_sum(parent_weights)
    innovation_weights = innovation_weights / tf.reduce_sum(innovation_weights)
    parameter_basis = tf.eye(STRUCTURAL_FIXTURE_PARAMETER_COUNT, dtype=DTYPE)
    parent_tangents0 = tf.zeros(
        [parent_count, STRUCTURAL_FIXTURE_STATE_DIM, STRUCTURAL_FIXTURE_PARAMETER_COUNT],
        DTYPE,
    )
    residual_history0 = tf.zeros([time_steps, teacher_count, 1], DTYPE)
    residual_tangent_history0 = tf.zeros(
        [time_steps, teacher_count, 1, STRUCTURAL_FIXTURE_PARAMETER_COUNT], DTYPE
    )
    expanded_tangent_history0 = tf.zeros(
        [time_steps, teacher_count, STRUCTURAL_FIXTURE_PARAMETER_COUNT], DTYPE
    )
    tangent_bound_history0 = tf.zeros(
        [time_steps, teacher_count, STRUCTURAL_FIXTURE_PARAMETER_COUNT], DTYPE
    )
    value_bound_history0 = tf.zeros([time_steps, teacher_count, 1], DTYPE)
    parent_history0 = tf.zeros(
        [time_steps, teacher_count, STRUCTURAL_FIXTURE_STATE_DIM], DTYPE
    )
    candidate_history0 = tf.zeros(
        [time_steps, teacher_count, STRUCTURAL_FIXTURE_STATE_DIM], DTYPE
    )
    tanh_history0 = tf.zeros([time_steps, teacher_count], DTYPE)
    increment_history0 = tf.zeros([time_steps], DTYPE)
    minimum_weight_history0 = tf.zeros([time_steps], DTYPE)
    feature_residual_history0 = tf.zeros(
        [time_steps, STRUCTURAL_FIXTURE_FEATURE_COUNT], DTYPE
    )
    valid_history0 = tf.zeros([time_steps], tf.bool)
    kernel_match_history0 = tf.zeros([time_steps], tf.bool)

    def cond(
        index: tf.Tensor,
        _parents: tf.Tensor,
        _parent_tangents: tf.Tensor,
        _weights: tf.Tensor,
        _total: tf.Tensor,
        _valid: tf.Tensor,
        *_history: tf.Tensor,
    ) -> tf.Tensor:
        del _parents, _parent_tangents, _weights, _total, _valid, _history
        return index < time_steps

    def body(
        index: tf.Tensor,
        current_parents: tf.Tensor,
        current_parent_tangents: tf.Tensor,
        current_weights: tf.Tensor,
        total: tf.Tensor,
        prior_valid: tf.Tensor,
        residual_history: tf.Tensor,
        residual_tangent_history: tf.Tensor,
        expanded_tangent_history: tf.Tensor,
        tangent_bound_history: tf.Tensor,
        value_bound_history: tf.Tensor,
        parent_history: tf.Tensor,
        candidate_history: tf.Tensor,
        tanh_history: tf.Tensor,
        increment_history: tf.Tensor,
        minimum_weight_history: tf.Tensor,
        feature_residual_history: tf.Tensor,
        valid_history: tf.Tensor,
        kernel_match_history: tf.Tensor,
    ) -> tuple[tf.Tensor, ...]:
        repeated_parents = tf.reshape(
            tf.broadcast_to(
                current_parents[:, None, :],
                [parent_count, innovation_count, STRUCTURAL_FIXTURE_STATE_DIM],
            ),
            [teacher_count, STRUCTURAL_FIXTURE_STATE_DIM],
        )
        tiled_innovations = tf.reshape(
            tf.broadcast_to(
                innovations[None, :, :], [parent_count, innovation_count, 1]
            ),
            [teacher_count, 1],
        )
        repeated_parent_tangents = tf.reshape(
            tf.broadcast_to(
                current_parent_tangents[:, None, :, :],
                [
                    parent_count,
                    innovation_count,
                    STRUCTURAL_FIXTURE_STATE_DIM,
                    STRUCTURAL_FIXTURE_PARAMETER_COUNT,
                ],
            ),
            [
                teacher_count,
                STRUCTURAL_FIXTURE_STATE_DIM,
                STRUCTURAL_FIXTURE_PARAMETER_COUNT,
            ],
        )
        components = structural_fixture_transition_components_tf(
            repeated_parents, tiled_innovations, theta
        )
        tanh_repeat = tf.math.tanh(components["stochastic_preactivation"])
        kernel_match = tf.reduce_all(
            tf.equal(
                tf.bitcast(components["tanh_value"], tf.int64),
                tf.bitcast(tanh_repeat, tf.int64),
            )
        )
        inject = tf.where(
            tf.equal(index, time_steps - 1),
            support_perturbation,
            tf.constant(0.0, DTYPE),
        )
        candidates = components["candidates"] + tf.stack(
            [
                tf.zeros([teacher_count], DTYPE),
                tf.fill([teacher_count], inject),
            ],
            axis=1,
        )
        rho, _sigma, alpha, beta = tf.unstack(theta)
        stochastic_tangent = (
            rho * repeated_parent_tangents[:, 0, :]
            + repeated_parents[:, 0, None] * parameter_basis[0][None, :]
            + tiled_innovations[:, 0, None] * parameter_basis[1][None, :]
        )
        tanh_tangent = (
            1.0 - tf.square(tanh_repeat)
        )[:, None] * stochastic_tangent
        deterministic_tangent = (
            alpha * repeated_parent_tangents[:, 1, :]
            + repeated_parents[:, 1, None] * parameter_basis[2][None, :]
            + beta * tanh_tangent
            + tanh_repeat[:, None] * parameter_basis[3][None, :]
        )
        candidate_tangent = tf.stack(
            [stochastic_tangent, deterministic_tangent], axis=1
        )
        residual = (
            candidates[:, 1]
            - alpha * repeated_parents[:, 1]
            - beta * tanh_repeat
        )[:, None]
        term_candidate = candidate_tangent[:, 1, :]
        term_direct_alpha = (
            repeated_parents[:, 1, None] * parameter_basis[2][None, :]
        )
        term_parent = alpha * repeated_parent_tangents[:, 1, :]
        term_direct_beta = tanh_repeat[:, None] * parameter_basis[3][None, :]
        term_tanh = beta * tanh_tangent
        expanded_tangent = (
            term_candidate
            - term_direct_alpha
            - term_parent
            - term_direct_beta
            - term_tanh
        )
        residual_tangent = expanded_tangent[:, None, :]
        tangent_scale = tf.maximum(
            tf.ones_like(expanded_tangent),
            tf.abs(term_candidate)
            + tf.abs(term_direct_alpha)
            + tf.abs(term_parent)
            + tf.abs(term_direct_beta)
            + tf.abs(term_tanh),
        )
        tangent_bound = (
            _gamma(STRUCTURAL_FIXTURE_TANGENT_OPERATION_COUNT) * tangent_scale
        )
        value_scale = tf.maximum(
            tf.ones([teacher_count], DTYPE),
            tf.abs(candidates[:, 1])
            + tf.abs(alpha * repeated_parents[:, 1])
            + tf.abs(beta * tanh_repeat),
        )[:, None]
        value_bound = _gamma(STRUCTURAL_FIXTURE_VALUE_OPERATION_COUNT) * value_scale
        tangent_valid = tf.reduce_all(
            tf.abs(residual_tangent[:, 0, :]) <= tangent_bound
        )
        support_valid = (
            tf.reduce_all(tf.abs(residual) <= value_bound)
            & tangent_valid
            & kernel_match
        )
        teacher_weights = tf.reshape(
            current_weights[:, None] * innovation_weights[None, :], [-1]
        )
        features = structural_fixture_features_tf(candidates)
        projection = tp._contract_e_tp_dense_square_forward_core(
            candidates,
            tf.math.log(teacher_weights),
            features,
            active_indices[index],
            row_scales[index],
        )
        increment = tf.math.log1p(tf.reduce_sum(tf.square(projection["matched_target"])))
        step_valid = (
            prior_valid
            & support_valid
            & projection["valid_chart"]
            & tf.math.is_finite(increment)
        )
        next_parents = _poison_invalid(projection["student_points"], step_valid)
        next_parent_tangents = _poison_invalid(
            tf.gather(candidate_tangent, active_indices[index], axis=0), step_valid
        )
        next_weights = _poison_invalid(projection["student_weights"], step_valid)
        next_total = _poison_invalid(total + increment, step_valid)
        return (
            index + 1,
            next_parents,
            next_parent_tangents,
            next_weights,
            next_total,
            step_valid,
            tf.tensor_scatter_nd_update(residual_history, [[index]], [residual]),
            tf.tensor_scatter_nd_update(
                residual_tangent_history, [[index]], [residual_tangent]
            ),
            tf.tensor_scatter_nd_update(
                expanded_tangent_history, [[index]], [expanded_tangent]
            ),
            tf.tensor_scatter_nd_update(
                tangent_bound_history, [[index]], [tangent_bound]
            ),
            tf.tensor_scatter_nd_update(
                value_bound_history, [[index]], [value_bound]
            ),
            tf.tensor_scatter_nd_update(
                parent_history, [[index]], [repeated_parents]
            ),
            tf.tensor_scatter_nd_update(candidate_history, [[index]], [candidates]),
            tf.tensor_scatter_nd_update(tanh_history, [[index]], [tanh_repeat]),
            tf.tensor_scatter_nd_update(increment_history, [[index]], [increment]),
            tf.tensor_scatter_nd_update(
                minimum_weight_history, [[index]], [projection["minimum_weight"]]
            ),
            tf.tensor_scatter_nd_update(
                feature_residual_history, [[index]], [projection["feature_residual"]]
            ),
            tf.tensor_scatter_nd_update(valid_history, [[index]], [step_valid]),
            tf.tensor_scatter_nd_update(
                kernel_match_history, [[index]], [kernel_match]
            ),
        )

    loop_result = tf.while_loop(
        cond,
        body,
        (
            tf.constant(0, tf.int32),
            parents,
            parent_tangents0,
            parent_weights,
            tf.constant(0.0, DTYPE),
            tf.constant(True),
            residual_history0,
            residual_tangent_history0,
            expanded_tangent_history0,
            tangent_bound_history0,
            value_bound_history0,
            parent_history0,
            candidate_history0,
            tanh_history0,
            increment_history0,
            minimum_weight_history0,
            feature_residual_history0,
            valid_history0,
            kernel_match_history0,
        ),
        parallel_iterations=1,
        maximum_iterations=time_steps,
    )
    return {
        "objective": loop_result[4],
        "valid": loop_result[5],
        "final_parents": loop_result[1],
        "final_parent_tangents": loop_result[2],
        "final_weights": loop_result[3],
        "residual_history": loop_result[6],
        "residual_jacobian": loop_result[7],
        "expanded_tangent": loop_result[8],
        "tangent_bound": loop_result[9],
        "value_bound_history": loop_result[10],
        "repeated_parent_history": loop_result[11],
        "candidate_history": loop_result[12],
        "tanh_history": loop_result[13],
        "increment_history": loop_result[14],
        "minimum_weight_history": loop_result[15],
        "feature_residual_history": loop_result[16],
        "valid_history": loop_result[17],
        "kernel_match_history": loop_result[18],
    }


def make_structural_fixture_recursive_tf(
    initial_parents: tf.Tensor,
    initial_weights: tf.Tensor,
    innovations: tf.Tensor,
    innovation_weights: tf.Tensor,
    active_indices: tf.Tensor,
    row_scales: tf.Tensor,
    *,
    jit_compile: bool = True,
):
    """Bind one fixed structural fixture into an XLA-default value/score graph."""

    initial_parents = tf.convert_to_tensor(initial_parents, DTYPE)
    initial_weights = tf.convert_to_tensor(initial_weights, DTYPE)
    innovations = tf.convert_to_tensor(innovations, DTYPE)
    innovation_weights = tf.convert_to_tensor(innovation_weights, DTYPE)
    active_indices = tf.convert_to_tensor(active_indices, tf.int32)
    row_scales = tf.convert_to_tensor(row_scales, DTYPE)

    @tf.function(
        input_signature=[
            tf.TensorSpec([STRUCTURAL_FIXTURE_PARAMETER_COUNT], DTYPE),
            tf.TensorSpec([], DTYPE),
        ],
        jit_compile=jit_compile,
        reduce_retracing=True,
    )
    def evaluate(theta: tf.Tensor, support_perturbation: tf.Tensor) -> dict[str, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(theta)
            result = structural_fixture_recursive_core_tf(
                theta,
                initial_parents,
                initial_weights,
                innovations,
                innovation_weights,
                active_indices,
                row_scales,
                support_perturbation,
            )
        score = tape.gradient(result["objective"], theta)
        tangent_values = result["residual_jacobian"][:, :, 0, :]
        tangent_valid = tf.reduce_all(
            tf.abs(tangent_values) <= result["tangent_bound"]
        )
        expansion_valid = tf.reduce_all(
            tf.abs(tangent_values - result["expanded_tangent"])
            <= result["tangent_bound"]
        )
        valid = result["valid"] & tangent_valid & expansion_valid
        return {
            **result,
            "valid": valid,
            "objective": _poison_invalid(result["objective"], valid),
            "score": _poison_invalid(score, valid),
            "final_parents": _poison_invalid(result["final_parents"], valid),
            "final_weights": _poison_invalid(result["final_weights"], valid),
            "tangent_valid": tangent_valid,
            "expansion_valid": expansion_valid,
            "value_operation_count": tf.constant(
                STRUCTURAL_FIXTURE_VALUE_OPERATION_COUNT, tf.int32
            ),
            "tangent_operation_count": tf.constant(
                STRUCTURAL_FIXTURE_TANGENT_OPERATION_COUNT, tf.int32
            ),
        }

    return evaluate


def structural_parent_innovation_teacher_tf(
    model: TFStructuralStateSpace,
    parents: tf.Tensor,
    innovations: tf.Tensor,
) -> dict[str, tf.Tensor]:
    """Push a parent-by-innovation rule through the declared structural map."""

    if model.config.integration_space != "innovation":
        raise ValueError("Contract E--TP structural teacher requires innovation integration")
    if model.partition.deterministic_dim and model.config.deterministic_completion != "required":
        raise ValueError("mixed structural teacher requires deterministic completion")
    parents = tf.convert_to_tensor(parents, tf.float64)
    innovations = tf.convert_to_tensor(innovations, tf.float64)
    tf.debugging.assert_rank(parents, 2)
    tf.debugging.assert_rank(innovations, 2)
    tf.debugging.assert_equal(tf.shape(parents)[1], model.partition.state_dim)
    tf.debugging.assert_equal(tf.shape(innovations)[1], model.partition.innovation_dim)
    parent_count = tf.shape(parents)[0]
    innovation_count = tf.shape(innovations)[0]
    repeated_parents = tf.repeat(parents, innovation_count, axis=0)
    tiled_innovations = tf.tile(innovations, [parent_count, 1])
    candidates = model.transition(repeated_parents, tiled_innovations)
    residual = model.deterministic_residual(
        repeated_parents, tiled_innovations, candidates
    )
    return {
        "parents": repeated_parents,
        "innovations": tiled_innovations,
        "candidates": candidates,
        "deterministic_residual": residual,
        "parent_count": parent_count,
        "innovation_count": innovation_count,
        "teacher_count": parent_count * innovation_count,
        "integration_space_innovation": tf.constant(True),
        "deterministic_completion_required": tf.constant(
            model.config.deterministic_completion == "required"
        ),
    }


def structural_residual_jacobian_tf(
    model: TFStructuralStateSpace,
    parents: tf.Tensor,
    innovations: tf.Tensor,
    theta: tf.Tensor,
    parameterized_transition,
    parameterized_residual,
) -> dict[str, tf.Tensor]:
    """Check completion value and total tangent for a parameterized fixture."""

    with tf.GradientTape() as tape:
        tape.watch(theta)
        candidates = parameterized_transition(parents, innovations, theta)
        residual = parameterized_residual(parents, innovations, candidates, theta)
    return {
        "candidates": candidates,
        "residual": residual,
        "residual_jacobian": tape.jacobian(residual, theta),
    }


__all__ = [
    "STRUCTURAL_FIXTURE_FEATURE_COUNT",
    "STRUCTURAL_FIXTURE_PARAMETER_COUNT",
    "STRUCTURAL_FIXTURE_STATE_DIM",
    "STRUCTURAL_FIXTURE_TANGENT_OPERATION_COUNT",
    "STRUCTURAL_FIXTURE_VALUE_OPERATION_COUNT",
    "make_structural_fixture_recursive_tf",
    "structural_parent_innovation_teacher_tf",
    "structural_fixture_features_tf",
    "structural_fixture_recursive_core_tf",
    "structural_fixture_transition_components_tf",
    "structural_residual_jacobian_tf",
]
