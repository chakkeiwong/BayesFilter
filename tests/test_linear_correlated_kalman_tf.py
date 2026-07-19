from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import math

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear import (
    tf_correlated_kalman_filter,
    tf_correlated_kalman_filter_batched_time_varying,
    tf_correlated_kalman_log_likelihood,
    tf_correlated_kalman_log_likelihood_batched_time_varying,
    tf_masked_correlated_kalman_filter,
    tf_masked_correlated_kalman_filter_batched_time_varying,
    tf_masked_correlated_kalman_log_likelihood,
    tf_masked_correlated_kalman_log_likelihood_batched_time_varying,
)
from bayesfilter.linear.kalman_tf import (
    tf_kalman_filter,
    tf_masked_kalman_filter,
)


def _primitive_noise_blocks(scale: float = 1.0) -> tuple[np.ndarray, ...]:
    state_factor = scale * np.array([[0.25, 0.00, 0.10], [0.05, 0.18, -0.04]])
    shared_measurement_factor = np.array(
        [[0.08, -0.03, 0.02], [-0.02, 0.05, 0.04]]
    )
    independent_measurement_factor = np.diag([0.28, 0.22])
    process_covariance = state_factor @ state_factor.T
    cross_covariance = state_factor @ shared_measurement_factor.T
    measurement_covariance = (
        shared_measurement_factor @ shared_measurement_factor.T
        + independent_measurement_factor @ independent_measurement_factor.T
    )
    joint = np.block(
        [
            [process_covariance, cross_covariance],
            [cross_covariance.T, measurement_covariance],
        ]
    )
    assert np.linalg.eigvalsh(joint).min() > -1.0e-14
    return process_covariance, cross_covariance, measurement_covariance, joint


def _fixture(scale: float = 1.0) -> dict[str, np.ndarray]:
    process_covariance, cross_covariance, measurement_covariance, _ = (
        _primitive_noise_blocks(scale)
    )
    time_dim = 4
    transition_matrix = np.stack(
        [
            np.array([[0.72 + 0.02 * t, 0.06], [-0.03, 0.63 - 0.01 * t]])
            for t in range(time_dim)
        ]
    )
    observation_matrix = np.stack(
        [
            np.array([[1.0, 0.18 + 0.02 * t], [-0.12, 0.85]])
            for t in range(time_dim)
        ]
    )
    return {
        "observations": np.array(
            [[0.21, -0.08], [0.14, 0.03], [0.07, -0.02], [0.18, 0.05]]
        ),
        "transition_offset": np.array(
            [[0.01, -0.02], [0.00, -0.01], [0.02, 0.00], [0.01, 0.01]]
        ),
        "transition_matrix": transition_matrix,
        "transition_covariance": np.repeat(
            process_covariance[np.newaxis],
            time_dim,
            axis=0,
        ),
        "observation_offset": np.array(
            [[0.02, -0.01], [0.01, 0.00], [0.00, -0.02], [0.02, 0.01]]
        ),
        "observation_matrix": observation_matrix,
        "observation_covariance": np.repeat(
            measurement_covariance[np.newaxis],
            time_dim,
            axis=0,
        ),
        "state_measurement_cross_covariance": np.repeat(
            cross_covariance[np.newaxis],
            time_dim,
            axis=0,
        ),
        "initial_state_mean": np.array([0.05, -0.04]),
        "initial_state_covariance": np.array([[0.30, 0.04], [0.04, 0.24]]),
    }


def _numpy_filter(
    fixture: dict[str, np.ndarray],
    mask: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    mean = fixture["initial_state_mean"].copy()
    covariance = fixture["initial_state_covariance"].copy()
    log_likelihood = 0.0
    means = []
    covariances = []
    for time_index, raw_row in enumerate(fixture["observations"]):
        transition = fixture["transition_matrix"][time_index]
        mean = fixture["transition_offset"][time_index] + transition @ mean
        covariance = (
            transition @ covariance @ transition.T
            + fixture["transition_covariance"][time_index]
        )
        covariance = 0.5 * (covariance + covariance.T)

        observation = fixture["observation_matrix"][time_index]
        measurement_covariance = fixture["observation_covariance"][time_index]
        cross_covariance = fixture["state_measurement_cross_covariance"][time_index]
        row = np.where(np.isfinite(raw_row), raw_row, 0.0)
        if mask is None:
            active = np.arange(row.size)
        else:
            active = np.flatnonzero(mask[time_index])
        if active.size:
            observation_active = observation[active]
            cross_active = cross_covariance[:, active]
            measurement_active = measurement_covariance[np.ix_(active, active)]
            innovation = row[active] - (
                fixture["observation_offset"][time_index, active]
                + observation_active @ mean
            )
            innovation_covariance = (
                observation_active @ covariance @ observation_active.T
                + observation_active @ cross_active
                + cross_active.T @ observation_active.T
                + measurement_active
            )
            state_innovation_covariance = covariance @ observation_active.T + cross_active
            sign, log_det = np.linalg.slogdet(innovation_covariance)
            assert sign > 0
            solve = np.linalg.solve(innovation_covariance, innovation)
            log_likelihood -= 0.5 * (
                active.size * math.log(2.0 * math.pi)
                + log_det
                + innovation @ solve
            )
            gain = np.linalg.solve(
                innovation_covariance,
                state_innovation_covariance.T,
            ).T
            mean = mean + gain @ innovation
            covariance = covariance - gain @ state_innovation_covariance.T
            covariance = 0.5 * (covariance + covariance.T)
        means.append(mean.copy())
        covariances.append(covariance.copy())
    return log_likelihood, np.stack(means), np.stack(covariances)


def _to_tf(fixture: dict[str, np.ndarray]) -> dict[str, tf.Tensor]:
    return {
        name: tf.convert_to_tensor(value, dtype=tf.float64)
        for name, value in fixture.items()
    }


def _batched_fixture(scales: tuple[float, ...] = (0.85, 1.0, 1.15)) -> dict[str, tf.Tensor]:
    fixtures = [_fixture(scale) for scale in scales]
    shared_observations = tf.constant(fixtures[0]["observations"], dtype=tf.float64)
    return {
        "observations": shared_observations,
        **{
            name: tf.constant(
                np.stack([fixture[name] for fixture in fixtures]),
                dtype=tf.float64,
            )
            for name in fixtures[0]
            if name != "observations"
        },
    }


def test_scalar_correlated_filter_matches_numpy_multistep_oracle() -> None:
    fixture = _fixture()
    expected_value, expected_means, expected_covariances = _numpy_filter(fixture)

    value, means, covariances = tf_correlated_kalman_filter(**_to_tf(fixture))

    np.testing.assert_allclose(value.numpy(), expected_value, atol=1.0e-11)
    np.testing.assert_allclose(means.numpy(), expected_means, atol=1.0e-11)
    np.testing.assert_allclose(covariances.numpy(), expected_covariances, atol=1.0e-11)
    assert value.shape == tf.TensorShape([])
    assert means.shape == tf.TensorShape([4, 2])
    assert covariances.shape == tf.TensorShape([4, 2, 2])


def test_one_step_update_matches_direct_joint_gaussian_conditioning() -> None:
    fixture = _fixture()
    one_step = {name: value[:1] for name, value in fixture.items() if value.ndim > 1}
    one_step["initial_state_mean"] = fixture["initial_state_mean"]
    one_step["initial_state_covariance"] = fixture["initial_state_covariance"]
    expected_value, expected_means, expected_covariances = _numpy_filter(one_step)

    value, means, covariances = tf_correlated_kalman_filter(**_to_tf(one_step))

    np.testing.assert_allclose(value.numpy(), expected_value, atol=1.0e-12)
    np.testing.assert_allclose(means.numpy(), expected_means, atol=1.0e-12)
    np.testing.assert_allclose(covariances.numpy(), expected_covariances, atol=1.0e-12)


def test_s_zero_matches_existing_dense_covariance_filter() -> None:
    fixture = _to_tf(_fixture())
    fixture["state_measurement_cross_covariance"] = tf.zeros_like(
        fixture["state_measurement_cross_covariance"]
    )
    correlated = tf_correlated_kalman_filter(**fixture)
    independent = tf_kalman_filter(
        observations=fixture["observations"],
        transition_offset=fixture["transition_offset"],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_offset=fixture["observation_offset"],
        observation_matrix=fixture["observation_matrix"],
        observation_covariance=fixture["observation_covariance"],
        initial_state_mean=fixture["initial_state_mean"],
        initial_state_covariance=fixture["initial_state_covariance"],
        return_filtered=True,
    )

    for actual, expected in zip(correlated, independent, strict=True):
        assert expected is not None
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1.0e-11)


def test_masked_nan_rows_match_compact_numpy_oracle_and_finite_dummy() -> None:
    fixture = _fixture()
    mask = np.array(
        [[True, True], [True, False], [False, False], [False, True]],
        dtype=bool,
    )
    fixture["observations"] = fixture["observations"].copy()
    fixture["observations"][~mask] = np.nan
    expected_value, expected_means, expected_covariances = _numpy_filter(fixture, mask)
    tensor_fixture = _to_tf(fixture)

    value, means, covariances = tf_masked_correlated_kalman_filter(
        **tensor_fixture,
        observation_mask=tf.constant(mask),
    )
    finite_fixture = dict(tensor_fixture)
    finite_fixture["observations"] = tf.where(
        mask,
        tensor_fixture["observations"],
        tf.constant(123.0, dtype=tf.float64),
    )
    finite_value = tf_masked_correlated_kalman_log_likelihood(
        **finite_fixture,
        observation_mask=tf.constant(mask),
    )

    np.testing.assert_allclose(value.numpy(), expected_value, atol=1.0e-11)
    np.testing.assert_allclose(means.numpy(), expected_means, atol=1.0e-11)
    np.testing.assert_allclose(covariances.numpy(), expected_covariances, atol=1.0e-11)
    np.testing.assert_allclose(value.numpy(), finite_value.numpy(), atol=1.0e-12)


def test_s_zero_matches_existing_masked_filter_for_finite_rows() -> None:
    fixture = _to_tf(_fixture())
    mask = tf.constant(
        [[True, True], [True, False], [False, False], [False, True]],
        dtype=tf.bool,
    )
    fixture["state_measurement_cross_covariance"] = tf.zeros_like(
        fixture["state_measurement_cross_covariance"]
    )
    correlated = tf_masked_correlated_kalman_filter(
        **fixture,
        observation_mask=mask,
    )
    independent = tf_masked_kalman_filter(
        observations=fixture["observations"],
        transition_offset=fixture["transition_offset"],
        transition_matrix=fixture["transition_matrix"],
        transition_covariance=fixture["transition_covariance"],
        observation_offset=fixture["observation_offset"],
        observation_matrix=fixture["observation_matrix"],
        observation_covariance=fixture["observation_covariance"],
        initial_state_mean=fixture["initial_state_mean"],
        initial_state_covariance=fixture["initial_state_covariance"],
        observation_mask=mask,
        return_filtered=True,
    )

    for actual, expected in zip(correlated, independent, strict=True):
        assert expected is not None
        np.testing.assert_allclose(actual.numpy(), expected.numpy(), atol=1.0e-11)


def test_observed_nan_fails_closed_but_masked_nan_is_accepted() -> None:
    fixture = _fixture()
    fixture["observations"] = fixture["observations"].copy()
    fixture["observations"][1, 1] = np.nan
    tensor_fixture = _to_tf(fixture)
    accepting_mask = tf.constant(
        [[True, True], [True, False], [True, True], [True, True]],
        dtype=tf.bool,
    )
    rejecting_mask = tf.ones([4, 2], dtype=tf.bool)

    accepted = tf_masked_correlated_kalman_log_likelihood(
        **tensor_fixture,
        observation_mask=accepting_mask,
    )
    assert np.isfinite(accepted.numpy())
    with pytest.raises(tf.errors.InvalidArgumentError, match="Observed entries"):
        tf_masked_correlated_kalman_log_likelihood(
            **tensor_fixture,
            observation_mask=rejecting_mask,
        )


def test_exact_singular_and_zero_process_covariance_are_supported() -> None:
    fixture = _fixture()
    direction = np.array([[0.17], [-0.06]])
    measurement_shared = np.array([[0.04], [-0.02]])
    singular_q = direction @ direction.T
    singular_s = direction @ measurement_shared.T
    independent_r = np.diag([0.20**2, 0.24**2])
    singular_r = measurement_shared @ measurement_shared.T + independent_r
    fixture["transition_covariance"][:] = singular_q
    fixture["state_measurement_cross_covariance"][:] = singular_s
    fixture["observation_covariance"][:] = singular_r
    singular_value = tf_correlated_kalman_log_likelihood(**_to_tf(fixture))

    zero_fixture = _fixture()
    zero_fixture["transition_covariance"][:] = 0.0
    zero_fixture["state_measurement_cross_covariance"][:] = 0.0
    zero_value = tf_correlated_kalman_log_likelihood(**_to_tf(zero_fixture))

    assert np.isfinite(singular_value.numpy())
    assert np.isfinite(zero_value.numpy())
    assert np.linalg.matrix_rank(singular_q) == 1
    assert np.count_nonzero(zero_fixture["transition_covariance"]) == 0


def test_batched_time_varying_rows_match_scalar_and_have_frozen_shapes() -> None:
    batched = _batched_fixture()
    value, means, covariances = tf_correlated_kalman_filter_batched_time_varying(
        **batched
    )
    scalar_results = []
    for batch_index in range(3):
        scalar_results.append(
            tf_correlated_kalman_filter(
                observations=batched["observations"],
                **{
                    name: tensor[batch_index]
                    for name, tensor in batched.items()
                    if name != "observations"
                },
            )
        )

    np.testing.assert_allclose(
        value.numpy(),
        tf.stack([result[0] for result in scalar_results]).numpy(),
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        means.numpy(),
        tf.stack([result[1] for result in scalar_results]).numpy(),
        atol=1.0e-11,
    )
    np.testing.assert_allclose(
        covariances.numpy(),
        tf.stack([result[2] for result in scalar_results]).numpy(),
        atol=1.0e-11,
    )
    assert value.shape == tf.TensorShape([3])
    assert means.shape == tf.TensorShape([3, 4, 2])
    assert covariances.shape == tf.TensorShape([3, 4, 2, 2])


def test_batched_masked_nan_singleton_and_permutation_contracts() -> None:
    batched = _batched_fixture()
    mask = tf.constant(
        [[True, True], [False, True], [False, False], [True, False]],
        dtype=tf.bool,
    )
    batched["observations"] = tf.where(
        mask,
        batched["observations"],
        tf.constant(np.nan, dtype=tf.float64),
    )
    base = tf_masked_correlated_kalman_filter_batched_time_varying(
        **batched,
        observation_mask=mask,
    )
    permutation = tf.constant([2, 0, 1], dtype=tf.int32)
    permuted_inputs = {
        name: tensor if name == "observations" else tf.gather(tensor, permutation)
        for name, tensor in batched.items()
    }
    permuted = tf_masked_correlated_kalman_filter_batched_time_varying(
        **permuted_inputs,
        observation_mask=mask,
    )
    singleton_inputs = {
        name: tensor if name == "observations" else tensor[:1]
        for name, tensor in batched.items()
    }
    singleton = tf_masked_correlated_kalman_filter_batched_time_varying(
        **singleton_inputs,
        observation_mask=mask,
    )

    for base_tensor, permuted_tensor in zip(base, permuted, strict=True):
        np.testing.assert_allclose(
            permuted_tensor.numpy(),
            tf.gather(base_tensor, permutation).numpy(),
            atol=1.0e-11,
        )
    for base_tensor, singleton_tensor in zip(base, singleton, strict=True):
        np.testing.assert_allclose(
            singleton_tensor.numpy(),
            base_tensor[:1].numpy(),
            atol=1.0e-11,
        )


def test_batched_api_rejects_per_batch_data_and_implicit_model_broadcasting() -> None:
    batched = _batched_fixture()
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="rank"):
        tf_correlated_kalman_log_likelihood_batched_time_varying(
            **{
                **batched,
                "observations": tf.stack([batched["observations"]] * 3),
            }
        )
    with pytest.raises((ValueError, tf.errors.InvalidArgumentError), match="rank"):
        tf_correlated_kalman_log_likelihood_batched_time_varying(
            **{
                **batched,
                "transition_matrix": batched["transition_matrix"][:, 0],
            }
        )


def test_log_likelihood_only_routes_match_filter_values() -> None:
    scalar = _to_tf(_fixture())
    scalar_filter = tf_correlated_kalman_filter(**scalar)[0]
    scalar_value = tf_correlated_kalman_log_likelihood(**scalar)
    batched = _batched_fixture()
    batch_filter = tf_correlated_kalman_filter_batched_time_varying(**batched)[0]
    batch_value = tf_correlated_kalman_log_likelihood_batched_time_varying(**batched)

    np.testing.assert_allclose(scalar_value.numpy(), scalar_filter.numpy(), atol=1.0e-12)
    np.testing.assert_allclose(batch_value.numpy(), batch_filter.numpy(), atol=1.0e-12)


def _parameterized_correlated_value(alpha: tf.Tensor) -> tf.Tensor:
    alpha = tf.convert_to_tensor(alpha, dtype=tf.float64)
    time_dim = 3
    sigma_u = tf.constant(0.24, dtype=tf.float64)
    sigma_eta = tf.constant(0.30, dtype=tf.float64)
    process_variance = sigma_u**2
    cross_covariance = sigma_u * alpha
    measurement_variance = alpha**2 + sigma_eta**2
    return tf_correlated_kalman_log_likelihood(
        observations=tf.constant([[0.12], [-0.04], [0.09]], dtype=tf.float64),
        transition_offset=tf.zeros([time_dim, 1], dtype=tf.float64),
        transition_matrix=tf.fill([time_dim, 1, 1], tf.constant(0.78, tf.float64)),
        transition_covariance=tf.fill([time_dim, 1, 1], process_variance),
        observation_offset=tf.zeros([time_dim, 1], dtype=tf.float64),
        observation_matrix=tf.ones([time_dim, 1, 1], dtype=tf.float64),
        observation_covariance=tf.fill([time_dim, 1, 1], measurement_variance),
        state_measurement_cross_covariance=tf.fill(
            [time_dim, 1, 1],
            cross_covariance,
        ),
        initial_state_mean=tf.constant([0.03], dtype=tf.float64),
        initial_state_covariance=tf.constant([[0.20]], dtype=tf.float64),
    )


def test_autodiff_gradient_matches_central_difference_for_correlated_parameter() -> None:
    alpha = tf.constant(0.07, dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(alpha)
        value = _parameterized_correlated_value(alpha)
    gradient = tape.gradient(value, alpha)
    step = tf.constant(1.0e-6, dtype=tf.float64)
    central = (
        _parameterized_correlated_value(alpha + step)
        - _parameterized_correlated_value(alpha - step)
    ) / (2.0 * step)

    assert gradient is not None
    assert np.isfinite(value.numpy())
    np.testing.assert_allclose(gradient.numpy(), central.numpy(), rtol=2.0e-7, atol=2.0e-8)


def test_target_only_cpu_xla_value_and_gradient_match_non_xla() -> None:
    alpha = tf.constant(0.07, dtype=tf.float64)

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled(value: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            tape.watch(value)
            target = _parameterized_correlated_value(value)
        gradient = tape.gradient(target, value)
        assert gradient is not None
        return target, gradient

    with tf.GradientTape() as tape:
        tape.watch(alpha)
        expected_value = _parameterized_correlated_value(alpha)
    expected_gradient = tape.gradient(expected_value, alpha)
    xla_value, xla_gradient = compiled(alpha)

    assert expected_gradient is not None
    np.testing.assert_allclose(xla_value.numpy(), expected_value.numpy(), atol=1.0e-11)
    np.testing.assert_allclose(
        xla_gradient.numpy(),
        expected_gradient.numpy(),
        rtol=1.0e-10,
        atol=1.0e-11,
    )
    assert len(compiled._list_all_concrete_functions_for_serialization()) == 1


def test_masked_batched_log_likelihood_matches_filter_value() -> None:
    batched = _batched_fixture()
    mask = tf.constant(
        [[True, True], [True, False], [False, False], [False, True]],
        dtype=tf.bool,
    )
    batched["observations"] = tf.where(
        mask,
        batched["observations"],
        tf.constant(np.nan, dtype=tf.float64),
    )
    filter_value = tf_masked_correlated_kalman_filter_batched_time_varying(
        **batched,
        observation_mask=mask,
    )[0]
    compact_value = tf_masked_correlated_kalman_log_likelihood_batched_time_varying(
        **batched,
        observation_mask=mask,
    )

    np.testing.assert_allclose(compact_value.numpy(), filter_value.numpy(), atol=1.0e-12)
