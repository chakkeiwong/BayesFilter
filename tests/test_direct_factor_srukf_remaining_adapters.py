"""Certification tests for the four remaining direct-factor SR-UKF adapters."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.linear import block_qr_conditional_tf, stack_qr_tf
from bayesfilter.linear.batched_kalman_svd_derivatives_tf import (
    tf_batched_svd_linear_gaussian_score_first_order_graph_status,
)
from bayesfilter.nonlinear import factor_srukf_tf
from bayesfilter.nonlinear.factor_srukf_tf import (
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
    tf_factor_srukf_value_and_score,
)
from bayesfilter.testing.direct_factor_srukf_adapters_tf import (
    build_common_v2_lgssm_factor_adapter,
    build_common_v2_predator_prey_factor_adapter,
    build_common_v2_range_bearing_factor_adapter,
    build_lgssm_exact_factor_adapter,
    circular_range_bearing_geometry,
    predator_prey_rk4_value_state_parameter_jacobians,
)
from bayesfilter.testing.deterministic_lgssm_exact_target_tf import (
    load_deterministic_lgssm_exact_target,
)
from bayesfilter.testing.multidim_triangular_lgssm_tf import (
    lower_triangular_lgssm_log_prob_score_status,
)


DTYPE = tf.float64


def _evaluate(builder, theta: tf.Tensor, *, jit_compile: bool = False):
    adapter = builder(theta)
    return adapter, tf_factor_srukf_value_and_score(
        adapter.observations,
        adapter.model,
        adapter.derivatives,
        jit_compile=jit_compile,
    )


def _centered_difference(builder, theta: tf.Tensor, step: float) -> tf.Tensor:
    values = tf.convert_to_tensor(theta, DTYPE)
    columns = []
    for index in range(int(values.shape[1])):
        direction = tf.one_hot(index, int(values.shape[1]), dtype=DTYPE)[None, :]
        plus_adapter = builder(values + step * direction)
        minus_adapter = builder(values - step * direction)
        plus = tf_factor_srukf_value_and_score(
            plus_adapter.observations,
            plus_adapter.model,
            plus_adapter.derivatives,
            jit_compile=False,
        ).log_likelihood[0]
        minus = tf_factor_srukf_value_and_score(
            minus_adapter.observations,
            minus_adapter.model,
            minus_adapter.derivatives,
            jit_compile=False,
        ).log_likelihood[0]
        columns.append((plus - minus) / (2.0 * step))
    return tf.stack(columns)


def _assert_result_health(result) -> None:
    tf.debugging.assert_all_finite(result.log_likelihood, "value")
    tf.debugging.assert_all_finite(result.score, "score")
    tf.debugging.assert_positive(result.diagnostics["minimum_qr_pivot"])
    tf.debugging.assert_all_finite(
        result.diagnostics["maximum_factor_reconstruction_residual"],
        "factor residual",
    )
    tf.debugging.assert_all_finite(
        result.diagnostics["maximum_derivative_reconstruction_residual"],
        "derivative residual",
    )


def _common_v2_lgssm_svd_authority(theta: tf.Tensor):
    from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
        _common_lgssm_v2_spec,
    )

    spec = _common_lgssm_v2_spec()
    values = tf.convert_to_tensor(theta, DTYPE)
    batch_size = int(values.shape[0])
    a0 = tf.convert_to_tensor(spec.parameters["A"], DTYPE)
    c0 = tf.convert_to_tensor(spec.parameters["C"], DTYPE)
    q0 = tf.convert_to_tensor(spec.parameters["Q"], DTYPE)
    r0 = tf.convert_to_tensor(spec.parameters["R"], DTYPE)
    p0 = tf.convert_to_tensor(spec.parameters["P0"], DTYPE)
    transition = values[:, 0, None, None] * a0[None, :, :]
    observation_covariance = values[:, 1, None, None] * r0[None, :, :]
    zeros_vector = tf.zeros([batch_size, 2], DTYPE)
    zeros_dvector = tf.zeros([batch_size, 2, 2], DTYPE)
    zeros_dmatrix = tf.zeros([batch_size, 2, 2, 2], DTYPE)
    d_transition = tf.stack(
        [
            tf.broadcast_to(a0[None, :, :], [batch_size, 2, 2]),
            tf.zeros([batch_size, 2, 2], DTYPE),
        ],
        axis=1,
    )
    d_observation_covariance = tf.stack(
        [
            tf.zeros([batch_size, 1, 1], DTYPE),
            tf.broadcast_to(r0[None, :, :], [batch_size, 1, 1]),
        ],
        axis=1,
    )
    return tf_batched_svd_linear_gaussian_score_first_order_graph_status(
        spec.observations,
        transition_offset=zeros_vector,
        transition_matrix=transition,
        transition_covariance=tf.broadcast_to(q0[None, :, :], [batch_size, 2, 2]),
        observation_offset=tf.zeros([batch_size, 1], DTYPE),
        observation_matrix=tf.broadcast_to(c0[None, :, :], [batch_size, 1, 2]),
        observation_covariance=observation_covariance,
        initial_state_mean=tf.broadcast_to(spec.parameters["m0"][None, :], [batch_size, 2]),
        initial_state_covariance=tf.broadcast_to(p0[None, :, :], [batch_size, 2, 2]),
        d_initial_state_mean=zeros_dvector,
        d_initial_state_covariance=zeros_dmatrix,
        d_transition_offset=zeros_dvector,
        d_transition_matrix=d_transition,
        d_transition_covariance=zeros_dmatrix,
        d_observation_offset=tf.zeros([batch_size, 2, 1], DTYPE),
        d_observation_matrix=tf.zeros([batch_size, 2, 1, 2], DTYPE),
        d_observation_covariance=d_observation_covariance,
        jitter=tf.constant(0.0, DTYPE),
        singular_floor=tf.constant(1.0e-12, DTYPE),
    )


def test_common_v2_lgssm_matches_exact_svd_authority_and_finite_difference() -> None:
    theta = tf.constant([[1.0, 1.0]], DTYPE)
    _adapter, result = _evaluate(build_common_v2_lgssm_factor_adapter, theta)
    authority = _common_v2_lgssm_svd_authority(theta)
    _assert_result_health(result)
    tf.debugging.assert_equal(authority.valid_pre_regularized_score, [True])
    tf.debugging.assert_near(result.log_likelihood, authority.log_likelihood, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(result.score, authority.score, atol=1e-9, rtol=1e-9)
    for step in (1.0e-5, 5.0e-6):
        tf.debugging.assert_near(
            result.score[0],
            _centered_difference(build_common_v2_lgssm_factor_adapter, theta, step),
            atol=2e-8,
            rtol=2e-8,
        )


def test_circular_geometry_mean_and_derivative_cross_angle_cut() -> None:
    geometry = circular_range_bearing_geometry()
    points = tf.constant([[[2.0, 3.13], [4.0, -3.13]]], DTYPE)
    weights = tf.constant([0.5, 0.5], DTYPE)
    d_points = tf.constant([[[[1.0, 0.4], [3.0, -0.2]]]], DTYPE)
    mean = geometry.mean_fn(points, weights)
    derivative = geometry.mean_derivative_fn(points, d_points, weights)
    tf.debugging.assert_near(mean[:, 0], [3.0], atol=1e-14)
    tf.debugging.assert_near(tf.abs(mean[:, 1]), [3.141592653589793], atol=1e-12)
    step = tf.constant(1.0e-6, DTYPE)
    plus_mean = geometry.mean_fn(points + step * d_points[:, 0], weights)
    minus_mean = geometry.mean_fn(points - step * d_points[:, 0], weights)
    finite_difference = geometry.residual_fn(minus_mean, plus_mean) / (2.0 * step)
    tf.debugging.assert_near(derivative[:, 0], finite_difference, atol=2e-8, rtol=2e-8)
    tf.debugging.assert_greater(
        geometry.mean_branch_margin_fn(points, weights),
        tf.constant([0.99], DTYPE),
    )


def test_circular_geometry_residual_wrap_and_branch_margin() -> None:
    geometry = circular_range_bearing_geometry()
    predicted = tf.constant([[1.0, 3.13], [1.0, -3.13], [1.0, 0.0]], DTYPE)
    observed = tf.constant([[1.1, -3.13], [0.9, 3.13], [1.0, 3.1415926535]], DTYPE)
    residual = geometry.residual_fn(predicted, observed)
    tf.debugging.assert_near(residual[:2, 1], [0.023185307179586, -0.023185307179586], atol=1e-12)
    margin = geometry.residual_branch_margin_fn(predicted, observed)
    tf.debugging.assert_greater(margin[:2], tf.constant([3.0, 3.0], DTYPE))
    tf.debugging.assert_less(margin[2], tf.constant(1.0e-8, DTYPE))


def test_circular_geometry_zero_resultant_is_score_inadmissible() -> None:
    geometry = circular_range_bearing_geometry()
    points = tf.constant([[[1.0, 0.0], [1.0, 3.141592653589793]]], DTYPE)
    weights = tf.constant([0.5, 0.5], DTYPE)
    margin = geometry.mean_branch_margin_fn(points, weights)
    tf.debugging.assert_less(margin, tf.constant([1.0e-12], DTYPE))
    tf.debugging.assert_less(
        margin, tf.constant([geometry.branch_margin_floor], DTYPE)
    )


def _near_cut_toy_model():
    geometry = circular_range_bearing_geometry()
    model = TFFactorSRUKFModel(
        initial_mean=tf.zeros([1, 2], DTYPE),
        initial_factor=0.01 * tf.eye(2, batch_shape=[1], dtype=DTYPE),
        process_factor=0.01 * tf.eye(2, batch_shape=[1], dtype=DTYPE),
        observation_factor=0.1 * tf.eye(2, batch_shape=[1], dtype=DTYPE),
        transition_fn=lambda previous, process: previous + process,
        observation_fn=tf.identity,
        observation_geometry=geometry,
        name="near_cut_toy",
    )

    def identity_jac(points, process=None):
        del process
        return tf.broadcast_to(tf.eye(2, dtype=DTYPE)[None, None, :, :], [1, tf.shape(points)[1], 2, 2])

    derivatives = TFFactorSRUKFDerivatives(
        d_initial_mean=tf.zeros([1, 1, 2], DTYPE),
        d_initial_factor=tf.zeros([1, 1, 2, 2], DTYPE),
        d_process_factor=tf.zeros([1, 1, 2, 2], DTYPE),
        d_observation_factor=tf.zeros([1, 1, 2, 2], DTYPE),
        transition_state_jacobian_fn=identity_jac,
        transition_process_jacobian_fn=identity_jac,
        d_transition_fn=lambda previous, process: tf.zeros([1, 1, tf.shape(previous)[1], 2], DTYPE),
        observation_state_jacobian_fn=identity_jac,
        d_observation_fn=lambda states: tf.zeros([1, 1, tf.shape(states)[1], 2], DTYPE),
    )
    return model, derivatives


def test_near_angle_cut_preserves_value_but_rejects_score() -> None:
    model, derivatives = _near_cut_toy_model()
    observation = tf.constant([[[0.0, 3.141592653589793 - 1.0e-10]]], DTYPE)
    eager = tf_factor_srukf_value_and_score(
        observation, model, derivatives, jit_compile=False
    )
    xla = tf_factor_srukf_value_and_score(
        observation, model, derivatives, jit_compile=True
    )
    for result in (eager, xla):
        tf.debugging.assert_all_finite(result.log_likelihood, "fixed value")
        assert bool(tf.reduce_any(tf.math.is_nan(result.score)).numpy())
        tf.debugging.assert_equal(
            result.diagnostics["observation_geometry_score_valid"], [False]
        )
        tf.debugging.assert_equal(result.diagnostics["invalid_transition_count"], [1])
        tf.debugging.assert_equal(result.diagnostics["classified_invalid_count"], [1])
        tf.debugging.assert_equal(result.diagnostics["invalid_count"], [1])
        tf.debugging.assert_equal(result.diagnostics["roundoff_repair_count"], [0])
        tf.debugging.assert_equal(result.diagnostics["row_class_code"], [2])
        tf.debugging.assert_equal(result.diagnostics["status_code"], [2])
        tf.debugging.assert_equal(
            result.diagnostics["valid_pre_regularized_score"], [False]
        )
        tf.debugging.assert_equal(result.diagnostics["output_finite"], [False])
        tf.debugging.assert_equal(result.diagnostics["nonfinite_output"], [True])
    tf.debugging.assert_near(eager.log_likelihood, xla.log_likelihood)
    for field in (
        "invalid_transition_count",
        "classified_invalid_count",
        "roundoff_repair_count",
        "row_class_code",
        "valid_pre_regularized_score",
        "output_finite",
        "nonfinite_output",
    ):
        tf.debugging.assert_equal(eager.diagnostics[field], xla.diagnostics[field])


def test_range_bearing_score_finite_difference_step_halving_and_xla() -> None:
    theta = tf.constant([[0.12, 0.04]], DTYPE)
    adapter, eager = _evaluate(build_common_v2_range_bearing_factor_adapter, theta)
    _assert_result_health(eager)
    tf.debugging.assert_greater(
        eager.diagnostics["minimum_observation_geometry_branch_margin"],
        tf.constant([0.9], DTYPE),
    )
    fd_coarse = _centered_difference(build_common_v2_range_bearing_factor_adapter, theta, 1.0e-5)
    fd_fine = _centered_difference(build_common_v2_range_bearing_factor_adapter, theta, 5.0e-6)
    tf.debugging.assert_near(eager.score[0], fd_coarse, atol=3e-5, rtol=3e-5)
    tf.debugging.assert_near(eager.score[0], fd_fine, atol=8e-6, rtol=8e-6)
    xla = tf_factor_srukf_value_and_score(adapter.observations, adapter.model, adapter.derivatives, jit_compile=True)
    tf.debugging.assert_near(eager.log_likelihood, xla.log_likelihood, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(eager.score, xla.score, atol=1e-10, rtol=1e-10)


def test_common_v2_predator_prey_binds_physical_r_and_source_rk4() -> None:
    from experiments.dpf_implementation.tf_tfp.fixtures.common_model_suite_tf import (
        _common_predator_prey_v2_spec,
        bayesfilter_model_for_spec_v2,
    )

    spec = _common_predator_prey_v2_spec()
    source_model = bayesfilter_model_for_spec_v2(spec)
    points = tf.constant([[[50.0, 5.0], [49.0, 5.2]]], DTYPE)
    value, _state_jacobian, parameter_jacobian = predator_prey_rk4_value_state_parameter_jacobians(source_model, spec.theta[None, :], points)
    source_value, source_parameter_jacobian = source_model.transition_mean_parameter_jacobian(spec.theta, points[0])
    tf.debugging.assert_near(value[0], source_value, atol=2e-12, rtol=2e-12)
    tf.debugging.assert_near(parameter_jacobian[0], source_parameter_jacobian, atol=2e-12, rtol=2e-12)
    theta = tf.constant([[0.6]], DTYPE)
    adapter, eager = _evaluate(build_common_v2_predator_prey_factor_adapter, theta)
    assert adapter.metadata["horizon"] == 3
    assert tuple(adapter.observations.shape) == (1, 3, 2)
    _assert_result_health(eager)
    for step, tolerance in ((1e-5, 5e-8), (5e-6, 2e-8)):
        tf.debugging.assert_near(eager.score[0], _centered_difference(build_common_v2_predator_prey_factor_adapter, theta, step), atol=tolerance, rtol=tolerance)


def test_lgssm_exact_full_score_authority_prior_separation_and_sampled_fd() -> None:
    bundle = load_deterministic_lgssm_exact_target()
    theta = bundle.raw_truth[None, :]
    _adapter, result = _evaluate(build_lgssm_exact_factor_adapter, theta)
    posterior_value, posterior_score, likelihood_value, likelihood_score, status = lower_triangular_lgssm_log_prob_score_status(bundle.raw_truth, tf.constant(bundle.fixture["observations"], DTYPE), bundle.contract)
    del posterior_value, posterior_score
    tf.debugging.assert_equal(status["valid_pre_regularized_score"], True)
    _assert_result_health(result)
    tf.debugging.assert_near(result.log_likelihood[0], likelihood_value, atol=1e-10, rtol=1e-10)
    tf.debugging.assert_near(result.score[0], likelihood_score, atol=1e-9, rtol=1e-9)
    for index in (0, 4, 10, 14):
        direction = tf.one_hot(index, 18, dtype=DTYPE)[None, :]
        step = tf.constant(5.0e-6, DTYPE)
        plus_adapter = build_lgssm_exact_factor_adapter(theta + step * direction)
        minus_adapter = build_lgssm_exact_factor_adapter(theta - step * direction)
        plus = tf_factor_srukf_value_and_score(plus_adapter.observations, plus_adapter.model, plus_adapter.derivatives, jit_compile=False).log_likelihood[0]
        minus = tf_factor_srukf_value_and_score(minus_adapter.observations, minus_adapter.model, minus_adapter.derivatives, jit_compile=False).log_likelihood[0]
        tf.debugging.assert_near(result.score[0, index], (plus - minus) / (2.0 * step), atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize(
    ("builder", "theta"),
    [
        (build_common_v2_lgssm_factor_adapter, [[1.0, 1.0]]),
        (build_common_v2_predator_prey_factor_adapter, [[0.6]]),
        (build_lgssm_exact_factor_adapter, None),
    ],
)
def test_remaining_adapters_eager_xla_parity(builder, theta) -> None:
    if theta is None:
        theta_tensor = load_deterministic_lgssm_exact_target().raw_truth[None, :]
    else:
        theta_tensor = tf.constant(theta, DTYPE)
    adapter, eager = _evaluate(builder, theta_tensor)
    xla = tf_factor_srukf_value_and_score(adapter.observations, adapter.model, adapter.derivatives, jit_compile=True)
    tf.debugging.assert_near(eager.log_likelihood, xla.log_likelihood, atol=1e-9, rtol=1e-9)
    tf.debugging.assert_near(eager.score, xla.score, atol=1e-9, rtol=1e-9)


def test_temporal_direct_factor_sources_do_not_decompose_covariance() -> None:
    sources = "\n".join(
        [
            inspect.getsource(factor_srukf_tf._one_step),
            inspect.getsource(stack_qr_tf.batched_stack_qr_lower),
            inspect.getsource(block_qr_conditional_tf.batched_block_qr_conditional),
        ]
    ).lower()
    for forbidden in ("cholesky", "svd(", "eigh(", "eigvalsh("):
        assert forbidden not in sources


def test_closure_harness_route_guard_and_numerical_gates_fail_closed() -> None:
    import importlib.util

    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_direct_factor_srukf_remaining_adapter_closure_20260817.py"
    )
    spec = importlib.util.spec_from_file_location("remaining_adapter_closure_harness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._route_guard()["status"] == "passed"
    base = {
        "authority_value_delta": 0.0,
        "authority_score_delta": 0.0,
        "finite_difference_delta_5e_6": 0.0,
        "eager_xla_value_delta": 0.0,
        "eager_xla_score_delta": 0.0,
        "minimum_qr_pivot": 1.0,
        "minimum_observation_geometry_branch_margin": 1.0,
        "maximum_factor_reconstruction_residual": 0.0,
        "maximum_derivative_reconstruction_residual": 0.0,
    }
    rows = [
        {**base, "model_id": "lgssm_2d_h25_rich"},
        {**base, "model_id": "range_bearing_4d_h20_rich"},
        {**base, "model_id": "predator_prey_rk4"},
        {**base, "model_id": "LGSSM-EXACT"},
    ]
    assert module._numerical_gates(rows)["status"] == "passed"
    rows[1]["finite_difference_delta_5e_6"] = 1.0
    assert module._numerical_gates(rows)["status"] == "failed"
