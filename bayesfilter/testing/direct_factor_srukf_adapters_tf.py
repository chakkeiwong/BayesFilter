"""Frozen model adapters for direct-factor SR-UKF certification tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import tensorflow as tf

from bayesfilter.nonlinear.factor_srukf_compat import (
    covariance_to_cholesky_factor_contract,
)
from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
    TFFactorSRUKFObservationGeometry,
)
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    load_deterministic_lgssm_exact_target,
)
from bayesfilter.testing.multidim_triangular_lgssm_batched_tf import (
    materialize_lower_triangular_lgssm_batch,
)


_PI = tf.constant(3.141592653589793238462643383279502884, tf.float64)
_RANGE_EPS = tf.constant(1.0e-12, tf.float64)


@dataclass(frozen=True)
class FrozenFactorSRUKFAdapter:
    """A factor contract plus its frozen observations and identity metadata."""

    model: TFFactorSRUKFModel
    derivatives: TFFactorSRUKFDerivatives
    observations: tf.Tensor
    theta: tf.Tensor
    metadata: Mapping[str, Any]


def wrap_angle_tf(value: tf.Tensor) -> tf.Tensor:
    return tf.math.floormod(tf.convert_to_tensor(value, tf.float64) + _PI, 2.0 * _PI) - _PI


def circular_range_bearing_geometry(
    *, branch_margin_floor: float = 1.0e-8
) -> TFFactorSRUKFObservationGeometry:
    """Return the fixed-branch circular observation geometry."""

    def mean_fn(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        range_mean = tf.einsum("r,br->b", weights, points[:, :, 0])
        sine = tf.einsum("r,br->b", weights, tf.sin(points[:, :, 1]))
        cosine = tf.einsum("r,br->b", weights, tf.cos(points[:, :, 1]))
        return tf.stack([range_mean, tf.atan2(sine, cosine)], axis=1)

    def mean_derivative_fn(
        points: tf.Tensor, d_points: tf.Tensor, weights: tf.Tensor
    ) -> tf.Tensor:
        d_range = tf.einsum("r,bpr->bp", weights, d_points[:, :, :, 0])
        sine = tf.einsum("r,br->b", weights, tf.sin(points[:, :, 1]))
        cosine = tf.einsum("r,br->b", weights, tf.cos(points[:, :, 1]))
        d_sine = tf.einsum(
            "r,br,bpr->bp",
            weights,
            tf.cos(points[:, :, 1]),
            d_points[:, :, :, 1],
        )
        d_cosine = -tf.einsum(
            "r,br,bpr->bp",
            weights,
            tf.sin(points[:, :, 1]),
            d_points[:, :, :, 1],
        )
        d_angle = (
            cosine[:, None] * d_sine - sine[:, None] * d_cosine
        ) / (tf.square(sine) + tf.square(cosine))[:, None]
        return tf.stack([d_range, d_angle], axis=2)

    def residual_fn(predicted: tf.Tensor, observed: tf.Tensor) -> tf.Tensor:
        raw = tf.convert_to_tensor(observed, tf.float64) - tf.convert_to_tensor(
            predicted, tf.float64
        )
        return tf.concat([raw[..., :1], wrap_angle_tf(raw[..., 1:2])], axis=-1)

    def residual_derivative_fn(
        predicted: tf.Tensor,
        observed: tf.Tensor,
        d_predicted: tf.Tensor,
        d_observed: tf.Tensor,
    ) -> tf.Tensor:
        del predicted, observed
        return tf.convert_to_tensor(d_observed, tf.float64) - tf.convert_to_tensor(
            d_predicted, tf.float64
        )

    def residual_branch_margin_fn(
        predicted: tf.Tensor, observed: tf.Tensor
    ) -> tf.Tensor:
        raw_angle = tf.convert_to_tensor(observed, tf.float64)[..., 1] - tf.convert_to_tensor(
            predicted, tf.float64
        )[..., 1]
        wrapped = wrap_angle_tf(raw_angle)
        return _PI - tf.abs(wrapped)

    def mean_branch_margin_fn(points: tf.Tensor, weights: tf.Tensor) -> tf.Tensor:
        sine = tf.einsum("r,br->b", weights, tf.sin(points[:, :, 1]))
        cosine = tf.einsum("r,br->b", weights, tf.cos(points[:, :, 1]))
        return tf.sqrt(tf.square(sine) + tf.square(cosine))

    return TFFactorSRUKFObservationGeometry(
        mean_fn=mean_fn,
        mean_derivative_fn=mean_derivative_fn,
        residual_fn=residual_fn,
        residual_derivative_fn=residual_derivative_fn,
        residual_branch_margin_fn=residual_branch_margin_fn,
        mean_branch_margin_fn=mean_branch_margin_fn,
        branch_margin_floor=branch_margin_floor,
        name="range_bearing_circular_fixed_branch_v1",
    )


def range_bearing_observation_tf(states: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(states, tf.float64)
    px = values[..., 0]
    py = values[..., 1]
    return tf.stack(
        [tf.sqrt(tf.square(px) + tf.square(py) + _RANGE_EPS), tf.atan2(py, px)],
        axis=-1,
    )


def range_bearing_observation_jacobian_tf(states: tf.Tensor) -> tf.Tensor:
    values = tf.convert_to_tensor(states, tf.float64)
    px = values[..., 0]
    py = values[..., 1]
    radius = tf.sqrt(tf.square(px) + tf.square(py) + _RANGE_EPS)
    radius_squared = tf.square(px) + tf.square(py)
    zero = tf.zeros_like(px)
    return tf.stack(
        [
            tf.stack([px / radius, py / radius, zero, zero], axis=-1),
            tf.stack([-py / radius_squared, px / radius_squared, zero, zero], axis=-1),
        ],
        axis=-2,
    )


def build_common_v2_lgssm_factor_adapter(theta: Any) -> FrozenFactorSRUKFAdapter:
    from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
        _common_lgssm_v2_spec,
    )

    values = _rank2_theta(theta, 2)
    spec = _common_lgssm_v2_spec()
    batch_size = int(values.shape[0])
    a_scale = values[:, 0]
    r_scale = values[:, 1]
    tf.debugging.assert_positive(r_scale, "observation covariance scale must be positive")
    a0 = tf.convert_to_tensor(spec.parameters["A"], tf.float64)
    c0 = tf.convert_to_tensor(spec.parameters["C"], tf.float64)
    q0 = tf.convert_to_tensor(spec.parameters["Q"], tf.float64)
    r0 = tf.convert_to_tensor(spec.parameters["R"], tf.float64)
    p0 = tf.convert_to_tensor(spec.parameters["P0"], tf.float64)
    m0 = tf.convert_to_tensor(spec.parameters["m0"], tf.float64)
    transition_matrix = a_scale[:, None, None] * a0[None, :, :]
    observation_covariance = r_scale[:, None, None] * r0[None, :, :]
    zero_covariance_derivative = tf.zeros([batch_size, 2, 2, 2], tf.float64)
    initial_factor, d_initial_factor = covariance_to_cholesky_factor_contract(
        tf.broadcast_to(p0[None, :, :], [batch_size, 2, 2]),
        zero_covariance_derivative,
        name="common_v2_lgssm_initial_covariance",
    )
    process_factor, d_process_factor = covariance_to_cholesky_factor_contract(
        tf.broadcast_to(q0[None, :, :], [batch_size, 2, 2]),
        zero_covariance_derivative,
        name="common_v2_lgssm_process_covariance",
    )
    d_observation_covariance = tf.stack(
        [
            tf.zeros([batch_size, 1, 1], tf.float64),
            tf.broadcast_to(r0[None, :, :], [batch_size, 1, 1]),
        ],
        axis=1,
    )
    observation_factor, d_observation_factor = covariance_to_cholesky_factor_contract(
        observation_covariance,
        d_observation_covariance,
        name="common_v2_lgssm_observation_covariance",
    )

    def transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        return tf.einsum("bij,brj->bri", transition_matrix, previous) + process

    def transition_state_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        return tf.broadcast_to(
            transition_matrix[:, None, :, :],
            [batch_size, tf.shape(previous)[1], 2, 2],
        )

    def transition_process_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del previous
        return tf.broadcast_to(
            tf.eye(2, batch_shape=[batch_size, 1], dtype=tf.float64),
            [batch_size, tf.shape(process)[1], 2, 2],
        )

    def d_transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        first = tf.einsum("ij,brj->bri", a0, previous)
        return tf.stack([first, tf.zeros_like(first)], axis=1)

    def observe(states: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,brj->bri", c0, states)

    def observation_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(
            c0[None, None, :, :], [batch_size, tf.shape(states)[1], 1, 2]
        )

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros([batch_size, 2, tf.shape(states)[1], 1], tf.float64)

    return FrozenFactorSRUKFAdapter(
        model=TFFactorSRUKFModel(
            initial_mean=tf.broadcast_to(m0[None, :], [batch_size, 2]),
            initial_factor=initial_factor,
            process_factor=process_factor,
            observation_factor=observation_factor,
            transition_fn=transition,
            observation_fn=observe,
            name="common_v2_lgssm_direct_factor",
        ),
        derivatives=TFFactorSRUKFDerivatives(
            d_initial_mean=tf.zeros([batch_size, 2, 2], tf.float64),
            d_initial_factor=d_initial_factor,
            d_process_factor=d_process_factor,
            d_observation_factor=d_observation_factor,
            transition_state_jacobian_fn=transition_state_jacobian,
            transition_process_jacobian_fn=transition_process_jacobian,
            d_transition_fn=d_transition,
            observation_state_jacobian_fn=observation_jacobian,
            d_observation_fn=d_observation,
            name="common_v2_lgssm_physical_scale_derivatives",
        ),
        observations=tf.broadcast_to(
            tf.convert_to_tensor(spec.observations, tf.float64)[None, :, :],
            [batch_size, int(spec.observations.shape[0]), 1],
        ),
        theta=values,
        metadata={
            "model_id": spec.model_id,
            "parameter_names": ("transition_matrix_scale", "observation_noise_scale"),
            "parameter_coordinate": "physical covariance scale",
            "fixture_checksum": spec.checksum(),
        },
    )


def build_common_v2_range_bearing_factor_adapter(theta: Any) -> FrozenFactorSRUKFAdapter:
    from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
        _common_range_bearing_v2_spec,
    )

    values = _rank2_theta(theta, 2)
    tf.debugging.assert_positive(values, "range/bearing standard deviations must be positive")
    spec = _common_range_bearing_v2_spec()
    batch_size = int(values.shape[0])
    a = tf.convert_to_tensor(spec.parameters["A"], tf.float64)
    q = tf.convert_to_tensor(spec.parameters["Q"], tf.float64)
    p0 = tf.convert_to_tensor(spec.parameters["P0"], tf.float64)
    m0 = tf.convert_to_tensor(spec.parameters["m0"], tf.float64)
    zero_covariance_derivative = tf.zeros([batch_size, 2, 4, 4], tf.float64)
    initial_factor, d_initial_factor = covariance_to_cholesky_factor_contract(
        tf.broadcast_to(p0[None, :, :], [batch_size, 4, 4]),
        zero_covariance_derivative,
        name="common_v2_range_bearing_initial_covariance",
    )
    process_factor, d_process_factor = covariance_to_cholesky_factor_contract(
        tf.broadcast_to(q[None, :, :], [batch_size, 4, 4]),
        zero_covariance_derivative,
        name="common_v2_range_bearing_process_covariance",
    )
    observation_factor = tf.linalg.diag(values)
    d_observation_factor = tf.broadcast_to(
        tf.linalg.diag(tf.eye(2, dtype=tf.float64))[None, :, :, :],
        [batch_size, 2, 2, 2],
    )

    def transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        return tf.einsum("ij,brj->bri", a, previous) + process

    def transition_state_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        return tf.broadcast_to(a[None, None, :, :], [batch_size, tf.shape(previous)[1], 4, 4])

    def transition_process_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del previous
        return tf.broadcast_to(tf.eye(4, dtype=tf.float64)[None, None, :, :], [batch_size, tf.shape(process)[1], 4, 4])

    def d_transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        return tf.zeros([batch_size, 2, tf.shape(previous)[1], 4], tf.float64)

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros([batch_size, 2, tf.shape(states)[1], 2], tf.float64)

    return FrozenFactorSRUKFAdapter(
        model=TFFactorSRUKFModel(
            initial_mean=tf.broadcast_to(m0[None, :], [batch_size, 4]),
            initial_factor=initial_factor,
            process_factor=process_factor,
            observation_factor=observation_factor,
            transition_fn=transition,
            observation_fn=range_bearing_observation_tf,
            observation_geometry=circular_range_bearing_geometry(),
            name="common_v2_range_bearing_direct_factor",
        ),
        derivatives=TFFactorSRUKFDerivatives(
            d_initial_mean=tf.zeros([batch_size, 2, 4], tf.float64),
            d_initial_factor=d_initial_factor,
            d_process_factor=d_process_factor,
            d_observation_factor=d_observation_factor,
            transition_state_jacobian_fn=transition_state_jacobian,
            transition_process_jacobian_fn=transition_process_jacobian,
            d_transition_fn=d_transition,
            observation_state_jacobian_fn=range_bearing_observation_jacobian_tf,
            d_observation_fn=d_observation,
            name="common_v2_range_bearing_physical_scale_derivatives",
        ),
        observations=tf.broadcast_to(
            tf.convert_to_tensor(spec.observations, tf.float64)[None, :, :],
            [batch_size, int(spec.observations.shape[0]), 2],
        ),
        theta=values,
        metadata={
            "model_id": spec.model_id,
            "parameter_names": ("sigma_range", "sigma_bearing"),
            "parameter_coordinate": "physical standard deviation",
            "fixture_checksum": spec.checksum(),
            "observation_geometry": "range_bearing_circular_fixed_branch_v1",
        },
    )


def predator_prey_rk4_value_state_parameter_jacobians(
    model: Any, theta: tf.Tensor, previous_points: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Propagate physical-coordinate RK4 value, state, and parameter Jacobians."""

    parameters = _rank2_theta(theta, 6)
    points = tf.convert_to_tensor(previous_points, tf.float64)
    batch_size = int(parameters.shape[0])
    point_count = tf.shape(points)[1]
    state = points
    state_jacobian = tf.broadcast_to(
        tf.eye(2, dtype=tf.float64)[None, None, :, :],
        [batch_size, point_count, 2, 2],
    )
    parameter_jacobian = tf.zeros([batch_size, 6, point_count, 2], tf.float64)

    def rhs(current: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        r, capacity, half_sat, s_rate, u_rate, v_rate = tf.unstack(parameters, axis=1)
        prey = current[..., 0]
        predator = current[..., 1]
        denominator = half_sat[:, None] + prey
        interaction = prey * predator / denominator
        logistic = prey * (1.0 - prey / capacity[:, None])
        value = tf.stack(
            [
                r[:, None] * logistic - s_rate[:, None] * interaction,
                u_rate[:, None] * interaction - v_rate[:, None] * predator,
            ],
            axis=-1,
        )
        d_interaction_prey = predator * half_sat[:, None] / tf.square(denominator)
        d_interaction_predator = prey / denominator
        state_partial = tf.stack(
            [
                tf.stack(
                    [r[:, None] * (1.0 - 2.0 * prey / capacity[:, None]) - s_rate[:, None] * d_interaction_prey, -s_rate[:, None] * d_interaction_predator],
                    axis=-1,
                ),
                tf.stack(
                    [u_rate[:, None] * d_interaction_prey, u_rate[:, None] * d_interaction_predator - v_rate[:, None]],
                    axis=-1,
                ),
            ],
            axis=-2,
        )
        zero = tf.zeros_like(prey)
        interaction_a = -prey * predator / tf.square(denominator)
        parameter_partial = tf.stack(
            [
                tf.stack([logistic, zero], axis=-1),
                tf.stack([r[:, None] * tf.square(prey) / tf.square(capacity)[:, None], zero], axis=-1),
                tf.stack([-s_rate[:, None] * interaction_a, u_rate[:, None] * interaction_a], axis=-1),
                tf.stack([-interaction, zero], axis=-1),
                tf.stack([zero, interaction], axis=-1),
                tf.stack([zero, -predator], axis=-1),
            ],
            axis=1,
        )
        return value, state_partial, parameter_partial

    def stage(
        current: tf.Tensor, d_state: tf.Tensor, d_parameter: tf.Tensor
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        value, state_partial, parameter_partial = rhs(current)
        return (
            value,
            tf.einsum("brij,brjk->brik", state_partial, d_state),
            tf.einsum("brij,bprj->bpri", state_partial, d_parameter)
            + parameter_partial,
        )

    step = tf.convert_to_tensor(model.rk4_internal_step, tf.float64)
    for _ in range(int(model.manifest_payload()["rk4_substeps"])):
        k1, a1, b1 = stage(state, state_jacobian, parameter_jacobian)
        k2, a2, b2 = stage(state + 0.5 * step * k1, state_jacobian + 0.5 * step * a1, parameter_jacobian + 0.5 * step * b1)
        k3, a3, b3 = stage(state + 0.5 * step * k2, state_jacobian + 0.5 * step * a2, parameter_jacobian + 0.5 * step * b2)
        k4, a4, b4 = stage(state + step * k3, state_jacobian + step * a3, parameter_jacobian + step * b3)
        scale = step / tf.constant(6.0, tf.float64)
        state = state + scale * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state_jacobian = state_jacobian + scale * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        parameter_jacobian = parameter_jacobian + scale * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
    return state, state_jacobian, parameter_jacobian


def build_common_v2_predator_prey_factor_adapter(theta_r: Any) -> FrozenFactorSRUKFAdapter:
    from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
        _common_predator_prey_v2_spec,
        bayesfilter_model_for_spec_v2,
    )

    r_values = _rank2_theta(theta_r, 1)
    spec = _common_predator_prey_v2_spec()
    source_model = bayesfilter_model_for_spec_v2(spec)
    batch_size = int(r_values.shape[0])
    base_parameters = tf.convert_to_tensor(spec.theta, tf.float64)
    full_theta = tf.concat(
        [r_values, tf.broadcast_to(base_parameters[None, 1:], [batch_size, 5])], axis=1
    )
    zero_covariance_derivative = tf.zeros([batch_size, 1, 2, 2], tf.float64)
    initial_factor, d_initial_factor = covariance_to_cholesky_factor_contract(
        tf.broadcast_to(source_model.initial_covariance[None, :, :], [batch_size, 2, 2]),
        zero_covariance_derivative,
        name="common_v2_predator_prey_initial_covariance",
    )
    process_factor, d_process_factor = covariance_to_cholesky_factor_contract(
        tf.broadcast_to(source_model.process_covariance[None, :, :], [batch_size, 2, 2]),
        zero_covariance_derivative,
        name="common_v2_predator_prey_process_covariance",
    )
    observation_factor, d_observation_factor = covariance_to_cholesky_factor_contract(
        tf.broadcast_to(source_model.observation_covariance[None, :, :], [batch_size, 2, 2]),
        zero_covariance_derivative,
        name="common_v2_predator_prey_observation_covariance",
    )

    def transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        value, _state_jacobian, _parameter_jacobian = predator_prey_rk4_value_state_parameter_jacobians(source_model, full_theta, previous)
        return value + process

    def transition_state_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        _value, state_jacobian, _parameter_jacobian = predator_prey_rk4_value_state_parameter_jacobians(source_model, full_theta, previous)
        return state_jacobian

    def transition_process_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del previous
        return tf.broadcast_to(tf.eye(2, dtype=tf.float64)[None, None, :, :], [batch_size, tf.shape(process)[1], 2, 2])

    def d_transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        _value, _state_jacobian, parameter_jacobian = predator_prey_rk4_value_state_parameter_jacobians(source_model, full_theta, previous)
        return parameter_jacobian[:, :1, :, :]

    def observation_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(tf.eye(2, dtype=tf.float64)[None, None, :, :], [batch_size, tf.shape(states)[1], 2, 2])

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return tf.zeros([batch_size, 1, tf.shape(states)[1], 2], tf.float64)

    return FrozenFactorSRUKFAdapter(
        model=TFFactorSRUKFModel(
            initial_mean=tf.broadcast_to(source_model.initial_mean[None, :], [batch_size, 2]),
            initial_factor=initial_factor,
            process_factor=process_factor,
            observation_factor=observation_factor,
            transition_fn=transition,
            observation_fn=tf.identity,
            name="common_v2_predator_prey_direct_factor",
        ),
        derivatives=TFFactorSRUKFDerivatives(
            d_initial_mean=tf.zeros([batch_size, 1, 2], tf.float64),
            d_initial_factor=d_initial_factor,
            d_process_factor=d_process_factor,
            d_observation_factor=d_observation_factor,
            transition_state_jacobian_fn=transition_state_jacobian,
            transition_process_jacobian_fn=transition_process_jacobian,
            d_transition_fn=d_transition,
            observation_state_jacobian_fn=observation_jacobian,
            d_observation_fn=d_observation,
            name="common_v2_predator_prey_physical_r_derivative",
        ),
        observations=tf.broadcast_to(
            tf.convert_to_tensor(spec.observations, tf.float64)[None, :, :],
            [batch_size, int(spec.observations.shape[0]), 2],
        ),
        theta=r_values,
        metadata={
            "model_id": spec.model_id,
            "parameter_names": ("r",),
            "parameter_coordinate": "physical theta=(r,K,a,s,u,v), r only",
            "fixture_checksum": spec.checksum(),
            "horizon": int(spec.observations.shape[0]),
        },
    )


def build_lgssm_exact_factor_adapter(theta: Any) -> FrozenFactorSRUKFAdapter:
    values = _rank2_theta(theta, 18)
    bundle = load_deterministic_lgssm_exact_target()
    materialized = materialize_lower_triangular_lgssm_batch(values, bundle.contract)
    batch_size = int(values.shape[0])
    observation_covariance = materialized.observation_covariance + tf.constant(1.0e-9, tf.float64) * tf.eye(4, batch_shape=[batch_size], dtype=tf.float64)
    initial_factor, d_initial_factor = covariance_to_cholesky_factor_contract(materialized.initial_covariance, materialized.d_initial_covariance, name="lgssm_exact_initial_covariance")
    process_factor, d_process_factor = covariance_to_cholesky_factor_contract(materialized.transition_covariance, materialized.d_transition_covariance, name="lgssm_exact_process_covariance")
    observation_factor, d_observation_factor = covariance_to_cholesky_factor_contract(observation_covariance, materialized.d_observation_covariance, name="lgssm_exact_observation_covariance_with_fixed_jitter")

    def transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        return materialized.transition_offset[:, None, :] + tf.einsum("bij,brj->bri", materialized.transition_matrix, previous) + process

    def transition_state_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        return tf.broadcast_to(materialized.transition_matrix[:, None, :, :], [batch_size, tf.shape(previous)[1], 4, 4])

    def transition_process_jacobian(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del previous
        return tf.broadcast_to(tf.eye(4, dtype=tf.float64)[None, None, :, :], [batch_size, tf.shape(process)[1], 4, 4])

    def d_transition(previous: tf.Tensor, process: tf.Tensor) -> tf.Tensor:
        del process
        return materialized.d_transition_offset[:, :, None, :] + tf.einsum("bpij,brj->bpri", materialized.d_transition_matrix, previous)

    def observe(states: tf.Tensor) -> tf.Tensor:
        return materialized.observation_offset[:, None, :] + tf.einsum("bij,brj->bri", materialized.observation_matrix, states)

    def observation_jacobian(states: tf.Tensor) -> tf.Tensor:
        return tf.broadcast_to(materialized.observation_matrix[:, None, :, :], [batch_size, tf.shape(states)[1], 4, 4])

    def d_observation(states: tf.Tensor) -> tf.Tensor:
        return materialized.d_observation_offset[:, :, None, :] + tf.einsum("bpij,brj->bpri", materialized.d_observation_matrix, states)

    observations = tf.convert_to_tensor(bundle.fixture["observations"], tf.float64)
    return FrozenFactorSRUKFAdapter(
        model=TFFactorSRUKFModel(initial_mean=materialized.initial_mean, initial_factor=initial_factor, process_factor=process_factor, observation_factor=observation_factor, transition_fn=transition, observation_fn=observe, name="lgssm_exact_direct_factor"),
        derivatives=TFFactorSRUKFDerivatives(d_initial_mean=materialized.d_initial_mean, d_initial_factor=d_initial_factor, d_process_factor=d_process_factor, d_observation_factor=d_observation_factor, transition_state_jacobian_fn=transition_state_jacobian, transition_process_jacobian_fn=transition_process_jacobian, d_transition_fn=d_transition, observation_state_jacobian_fn=observation_jacobian, d_observation_fn=d_observation, name="lgssm_exact_raw18_derivatives"),
        observations=tf.broadcast_to(observations[None, :, :], [batch_size, 120, 4]),
        theta=values,
        metadata={"model_id": "LGSSM-EXACT", "parameter_names": bundle.parameter_names, "parameter_coordinate": "persisted raw 18-dimensional coordinate", "target_signature": bundle.target_signature, "fixed_innovation_jitter": 1.0e-9},
    )


def _rank2_theta(theta: Any, parameter_dim: int) -> tf.Tensor:
    values = tf.convert_to_tensor(theta, tf.float64)
    if values.shape.rank == 1:
        values = values[None, :]
    if values.shape.rank != 2 or values.shape[0] is None or values.shape[1] != parameter_dim:
        raise ValueError(f"theta must have shape [B,{parameter_dim}]")
    tf.debugging.assert_all_finite(values, "theta contains NaN or Inf")
    return values


__all__ = [
    "FrozenFactorSRUKFAdapter",
    "build_common_v2_lgssm_factor_adapter",
    "build_common_v2_predator_prey_factor_adapter",
    "build_common_v2_range_bearing_factor_adapter",
    "build_lgssm_exact_factor_adapter",
    "circular_range_bearing_geometry",
    "predator_prey_rk4_value_state_parameter_jacobians",
    "range_bearing_observation_jacobian_tf",
    "range_bearing_observation_tf",
    "wrap_angle_tf",
]
