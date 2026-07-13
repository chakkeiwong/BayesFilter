from __future__ import annotations

import math

import pytest
import tensorflow as tf

from bayesfilter.testing.scalar_lgssm_forecast_oracle import (
    ANALYTIC_STATUS_DEGENERATE_LOG_VARIANCE,
    ANALYTIC_STATUS_VALID,
    DEFAULT_QUANTILE_PROBABILITIES,
    FORECAST_HORIZON,
    ScalarLGSSMInnovationBank,
    ScalarLGSSMParameters,
    analytic_scalar_lgssm_forecast,
    make_scalar_lgssm_innovation_bank,
    scalar_lgssm_analytic_compiled_program,
    scalar_lgssm_simulation_compiled_program,
    simulate_scalar_lgssm_forecast,
)


def _f(hex_value: str) -> tf.Tensor:
    return tf.constant(float.fromhex(hex_value), tf.float64)


def _fixture_parameters() -> ScalarLGSSMParameters:
    return ScalarLGSSMParameters(
        transition_coefficient=_f("0x1.70a3d70a3d70ap-1"),
        transition_offset=_f("-0x1.47ae147ae147bp-4"),
        observation_coefficient=_f("0x1.2666666666666p+0"),
        observation_offset=_f("0x1.47ae147ae147bp-5"),
        terminal_mean=_f("0x1.3333333333333p-2"),
        terminal_variance=_f("0x1.47ae147ae147bp-3"),
        process_variance=_f("0x1.70a3d70a3d70ap-4"),
        observation_variance=_f("0x1.47ae147ae147bp-4"),
    )


def _parameters(**overrides: float) -> ScalarLGSSMParameters:
    values = {
        "transition_coefficient": 0.7,
        "transition_offset": -0.08,
        "observation_coefficient": 1.15,
        "observation_offset": 0.04,
        "terminal_mean": 0.3,
        "terminal_variance": 0.16,
        "process_variance": 0.09,
        "observation_variance": 0.08,
    }
    values.update(overrides)
    return ScalarLGSSMParameters(
        **{name: tf.constant(value, tf.float64) for name, value in values.items()}
    )


def _default_probabilities() -> tf.Tensor:
    return tf.constant(DEFAULT_QUANTILE_PROBABILITIES, tf.float64)


def test_fixture_analytic_formula_matches_manual_first_two_horizons() -> None:
    parameters = _fixture_parameters()
    result = analytic_scalar_lgssm_forecast(
        parameters, quantile_probabilities=_default_probabilities(), jit_compile=False
    )
    a = float(parameters.transition_coefficient)
    b = float(parameters.transition_offset)
    c = float(parameters.observation_coefficient)
    terminal_mean = float(parameters.terminal_mean)
    terminal_variance = float(parameters.terminal_variance)
    q = float(parameters.process_variance)
    r = float(parameters.observation_variance)
    state_mean_1 = a * terminal_mean + b
    state_mean_2 = a * state_mean_1 + b
    state_variance_1 = a * a * terminal_variance + q
    state_variance_2 = a**4 * terminal_variance + q * (a * a + 1.0)
    state_covariance_12 = a**3 * terminal_variance + q * a
    tf.debugging.assert_near(
        result.state_mean[:2], tf.constant([state_mean_1, state_mean_2], tf.float64)
    )
    tf.debugging.assert_near(result.state_covariance[0, 0], state_variance_1)
    tf.debugging.assert_near(result.state_covariance[1, 1], state_variance_2)
    tf.debugging.assert_near(result.state_covariance[0, 1], state_covariance_12)
    tf.debugging.assert_near(
        result.observation_covariance[0, 1], c * c * state_covariance_12
    )
    tf.debugging.assert_near(
        result.observation_variance[0], c * c * state_variance_1 + r
    )


def test_default_python_quantile_tuple_is_canonical_float64() -> None:
    result = analytic_scalar_lgssm_forecast(_fixture_parameters(), jit_compile=False)
    assert result.quantile_probabilities.dtype == tf.float64
    tf.debugging.assert_equal(result.quantile_probabilities, _default_probabilities())
    assert bool(result.log_variance_valid)
    assert result.status.numpy().decode("ascii") == ANALYTIC_STATUS_VALID


@pytest.mark.parametrize("coefficient", [1.0, -0.65])
def test_direct_finite_sums_handle_unit_and_negative_transition(coefficient: float) -> None:
    parameters = _parameters(transition_coefficient=coefficient)
    result = analytic_scalar_lgssm_forecast(
        parameters, quantile_probabilities=_default_probabilities(), jit_compile=False
    )
    means = []
    current = float(parameters.terminal_mean)
    for _ in range(FORECAST_HORIZON):
        current = coefficient * current + float(parameters.transition_offset)
        means.append(current)
    tf.debugging.assert_near(result.state_mean, tf.constant(means, tf.float64))
    tf.debugging.assert_near(
        result.observation_covariance, tf.transpose(result.observation_covariance)
    )
    assert float(result.state_symmetry_residual) <= float(result.state_psd_tolerance)
    assert float(result.observation_symmetry_residual) <= float(
        result.observation_psd_tolerance
    )
    assert float(result.minimum_state_covariance_eigenvalue) >= -float(
        result.state_psd_tolerance
    )
    assert float(result.minimum_observation_covariance_eigenvalue) >= -float(
        result.observation_psd_tolerance
    )


def test_gaussian_moments_quantiles_and_covariance_are_exact_identities() -> None:
    result = analytic_scalar_lgssm_forecast(
        _fixture_parameters(),
        quantile_probabilities=_default_probabilities(),
        jit_compile=False,
    )
    tf.debugging.assert_equal(
        result.observation_third_central_moment, tf.zeros([FORECAST_HORIZON], tf.float64)
    )
    tf.debugging.assert_near(
        result.observation_fourth_central_moment,
        3.0 * tf.square(result.observation_variance),
    )
    tf.debugging.assert_near(
        result.observation_log_variance, tf.math.log(result.observation_variance)
    )
    tf.debugging.assert_near(
        result.observation_quantiles[:, 2], result.observation_mean, atol=1.0e-14
    )
    tf.debugging.assert_equal(result.quantile_probabilities, _default_probabilities())


def test_zero_noise_and_zero_terminal_variance_give_degenerate_law() -> None:
    parameters = _parameters(
        terminal_variance=0.0,
        process_variance=0.0,
        observation_variance=0.0,
    )
    result = analytic_scalar_lgssm_forecast(
        parameters, quantile_probabilities=_default_probabilities(), jit_compile=False
    )
    tf.debugging.assert_equal(result.observation_variance, tf.zeros([10], tf.float64))
    assert bool(tf.reduce_all(tf.math.is_inf(result.observation_log_variance)))
    tf.debugging.assert_equal(
        result.observation_quantiles,
        tf.broadcast_to(result.observation_mean[:, tf.newaxis], [10, 5]),
    )
    tf.debugging.assert_equal(result.degenerate_variance_mask, tf.ones([10], tf.bool))
    assert not bool(result.log_variance_valid)
    assert (
        result.status.numpy().decode("ascii")
        == ANALYTIC_STATUS_DEGENERATE_LOG_VARIANCE
    )
    tf.debugging.assert_equal(
        result.minimum_state_covariance_eigenvalue, tf.constant(0.0, tf.float64)
    )
    tf.debugging.assert_equal(
        result.minimum_observation_covariance_eigenvalue,
        tf.constant(0.0, tf.float64),
    )


def test_state_psd_diagnostic_is_independent_when_observation_coefficient_is_zero() -> None:
    result = analytic_scalar_lgssm_forecast(
        _parameters(observation_coefficient=0.0, observation_variance=0.2),
        jit_compile=False,
    )
    assert float(result.minimum_state_covariance_eigenvalue) >= -float(
        result.state_psd_tolerance
    )
    assert float(result.minimum_observation_covariance_eigenvalue) > 0.0
    assert float(result.state_symmetry_residual) <= float(result.state_psd_tolerance)
    assert float(result.observation_symmetry_residual) <= float(
        result.observation_psd_tolerance
    )


@pytest.mark.parametrize(
    "name,value,error",
    [
        ("terminal_variance", -0.1, ValueError),
        ("process_variance", -0.1, ValueError),
        ("observation_variance", -0.1, ValueError),
        ("transition_coefficient", math.inf, ValueError),
    ],
)
def test_invalid_parameter_values_fail_closed(name: str, value: float, error: type[Exception]) -> None:
    with pytest.raises(error):
        _parameters(**{name: value})


def test_float32_parameters_fail_closed() -> None:
    values = {name: tf.constant(0.1, tf.float64) for name in (
        "transition_coefficient",
        "transition_offset",
        "observation_coefficient",
        "observation_offset",
        "terminal_mean",
        "terminal_variance",
        "process_variance",
        "observation_variance",
    )}
    values["transition_coefficient"] = tf.constant(0.1, tf.float32)
    with pytest.raises(TypeError, match="float64"):
        ScalarLGSSMParameters(**values)


@pytest.mark.parametrize(
    "probabilities,error",
    [
        (tf.constant([0.0, 0.5], tf.float64), ValueError),
        (tf.constant([0.5, 0.5], tf.float64), ValueError),
        (tf.constant([0.5, float("nan")], tf.float64), ValueError),
        (tf.constant([0.25, 0.75], tf.float32), TypeError),
    ],
)
def test_invalid_quantile_probabilities_fail_closed(
    probabilities: tf.Tensor, error: type[Exception]
) -> None:
    with pytest.raises(error):
        analytic_scalar_lgssm_forecast(
            _fixture_parameters(),
            quantile_probabilities=probabilities,
            jit_compile=False,
        )


def test_innovation_bank_replays_and_families_are_disjoint() -> None:
    kwargs = dict(
        chain_count=4,
        draw_count=16,
        forecast_replication_count=3,
        seed=tf.constant([20260713, 1303], tf.int32),
        arm_id=1,
    )
    left = make_scalar_lgssm_innovation_bank(**kwargs)
    right = make_scalar_lgssm_innovation_bank(**kwargs)
    tf.debugging.assert_equal(left.terminal_standard_normal, right.terminal_standard_normal)
    tf.debugging.assert_equal(left.process_standard_normal, right.process_standard_normal)
    tf.debugging.assert_equal(left.observation_standard_normal, right.observation_standard_normal)
    assert not bool(
        tf.reduce_all(left.process_standard_normal == left.observation_standard_normal)
    )
    changed_arm = make_scalar_lgssm_innovation_bank(**{**kwargs, "arm_id": 2})
    assert not bool(
        tf.reduce_all(left.terminal_standard_normal == changed_arm.terminal_standard_normal)
    )


def test_direct_simulation_matches_hand_equations_and_preserves_axes() -> None:
    parameters = _parameters()
    terminal = tf.reshape(tf.range(6, dtype=tf.float64) / 10.0, [1, 2, 3])
    process = tf.reshape(tf.range(60, dtype=tf.float64) / 100.0, [1, 2, 3, 10])
    observation = -tf.reshape(
        tf.range(60, dtype=tf.float64) / 200.0, [1, 2, 3, 10]
    )
    bank = ScalarLGSSMInnovationBank(
        terminal_standard_normal=terminal,
        process_standard_normal=process,
        observation_standard_normal=observation,
        root_seed=tf.constant([11, 29], tf.int32),
        arm_id=3,
    )
    result = simulate_scalar_lgssm_forecast(parameters, bank, jit_compile=False)
    expected_terminal = parameters.terminal_mean + tf.sqrt(parameters.terminal_variance) * terminal
    expected_process_0 = tf.sqrt(parameters.process_variance) * process[..., 0]
    expected_state_0 = (
        parameters.transition_coefficient * expected_terminal
        + parameters.transition_offset
        + expected_process_0
    )
    expected_observation_noise_0 = (
        tf.sqrt(parameters.observation_variance) * observation[..., 0]
    )
    expected_observation_0 = (
        parameters.observation_coefficient * expected_state_0
        + parameters.observation_offset
        + expected_observation_noise_0
    )
    tf.debugging.assert_near(result.terminal_states, expected_terminal)
    tf.debugging.assert_near(result.process_innovations[..., 0], expected_process_0)
    tf.debugging.assert_near(result.states[..., 0], expected_state_0)
    tf.debugging.assert_near(
        result.observation_innovations[..., 0], expected_observation_noise_0
    )
    tf.debugging.assert_near(result.observations[..., 0], expected_observation_0)
    assert result.observations.shape == (1, 2, 3, 10)


def test_invalid_materialized_banks_fail_closed() -> None:
    valid = make_scalar_lgssm_innovation_bank(
        chain_count=1,
        draw_count=2,
        forecast_replication_count=3,
        seed=tf.constant([1, 2], tf.int32),
        arm_id=1,
    )
    invalid = ScalarLGSSMInnovationBank(
        terminal_standard_normal=valid.terminal_standard_normal,
        process_standard_normal=tf.tensor_scatter_nd_update(
            valid.process_standard_normal,
            tf.constant([[0, 0, 0, 0]], tf.int32),
            tf.constant([float("nan")], tf.float64),
        ),
        observation_standard_normal=valid.observation_standard_normal,
        root_seed=valid.root_seed,
        arm_id=valid.arm_id,
    )
    with pytest.raises(ValueError, match="finite"):
        simulate_scalar_lgssm_forecast(_fixture_parameters(), invalid, jit_compile=False)


def test_wrong_horizon_and_seed_contracts_fail_closed() -> None:
    with pytest.raises(ValueError, match="frozen at 10"):
        analytic_scalar_lgssm_forecast(
            _fixture_parameters(),
            horizon=9,
            quantile_probabilities=_default_probabilities(),
            jit_compile=False,
        )
    with pytest.raises(TypeError, match="int32"):
        make_scalar_lgssm_innovation_bank(
            chain_count=1,
            draw_count=1,
            forecast_replication_count=1,
            seed=tf.constant([1, 2], tf.int64),
            arm_id=1,
        )


def test_xla_analytic_and_simulation_match_eager_and_reuse_traces() -> None:
    parameters = _fixture_parameters()
    probabilities = _default_probabilities()
    eager = analytic_scalar_lgssm_forecast(
        parameters, quantile_probabilities=probabilities, jit_compile=False
    )
    compiled = analytic_scalar_lgssm_forecast(
        parameters, quantile_probabilities=probabilities, jit_compile=True
    )
    tf.debugging.assert_near(compiled.observation_mean, eager.observation_mean)
    tf.debugging.assert_near(compiled.observation_covariance, eager.observation_covariance)
    tf.debugging.assert_near(compiled.observation_quantiles, eager.observation_quantiles)
    analytic_program = scalar_lgssm_analytic_compiled_program(5)
    assert len(analytic_program._list_all_concrete_functions_for_serialization()) == 1

    bank = make_scalar_lgssm_innovation_bank(
        chain_count=1,
        draw_count=2,
        forecast_replication_count=3,
        seed=tf.constant([20260713, 1303], tf.int32),
        arm_id=1,
    )
    eager_paths = simulate_scalar_lgssm_forecast(parameters, bank, jit_compile=False)
    compiled_paths = simulate_scalar_lgssm_forecast(parameters, bank, jit_compile=True)
    tf.debugging.assert_near(compiled_paths.states, eager_paths.states)
    tf.debugging.assert_near(compiled_paths.observations, eager_paths.observations)
    simulation_program = scalar_lgssm_simulation_compiled_program(1, 2, 3)
    assert len(simulation_program._list_all_concrete_functions_for_serialization()) == 1
