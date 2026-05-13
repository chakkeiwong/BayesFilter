import numpy as np
import tensorflow as tf

from bayesfilter.testing import (
    make_affine_gaussian_structural_oracle_tf,
    make_nonlinear_accumulation_model_tf,
    make_univariate_nonlinear_growth_model_tf,
)


def test_model_a_affine_oracle_metadata_and_deterministic_residual() -> None:
    model = make_affine_gaussian_structural_oracle_tf()
    previous = tf.constant([[0.3, -0.1]], dtype=tf.float64)
    innovation = tf.constant([[0.2]], dtype=tf.float64)
    next_state = model.transition(previous, innovation)
    residual = model.deterministic_residual(previous, innovation, next_state)

    assert model.name == "model_a_affine_gaussian_structural_oracle"
    assert model.partition.state_names == ("m", "lag_m")
    assert model.partition.stochastic_indices == (0,)
    assert model.partition.deterministic_indices == (1,)
    assert model.is_affine
    np.testing.assert_allclose(residual.numpy(), [[0.0]], atol=1e-14)
    np.testing.assert_allclose(next_state.numpy()[:, 1], previous.numpy()[:, 0], atol=1e-14)


def test_model_b_nonlinear_accumulation_law_and_residual() -> None:
    model = make_nonlinear_accumulation_model_tf()
    previous = tf.constant([[0.2, -0.3]], dtype=tf.float64)
    innovation = tf.constant([[0.4]], dtype=tf.float64)
    next_state = model.transition(previous, innovation)
    observation = model.observe(next_state)
    residual = model.deterministic_residual(previous, innovation, next_state)

    expected_m = 0.70 * 0.2 + 0.25 * 0.4
    expected_k = 0.55 * (-0.3) + 0.80 * np.tanh(expected_m)
    np.testing.assert_allclose(next_state.numpy(), [[expected_m, expected_k]], atol=1e-14)
    np.testing.assert_allclose(observation.numpy(), [[expected_m + expected_k]], atol=1e-14)
    np.testing.assert_allclose(residual.numpy(), [[0.0]], atol=1e-14)
    assert not model.is_affine


def test_model_c_autonomous_growth_law_and_phase_residual() -> None:
    model = make_univariate_nonlinear_growth_model_tf()
    previous = tf.constant([[0.2, 1.0]], dtype=tf.float64)
    innovation = tf.constant([[0.4]], dtype=tf.float64)
    next_state = model.transition(previous, innovation)
    observation = model.observe(next_state)
    residual = model.deterministic_residual(previous, innovation, next_state)

    expected_x = 0.5 * 0.2 + 25.0 * 0.2 / (1.0 + 0.2**2) + 8.0 * np.cos(1.2) + 0.4
    expected_tau = 2.0
    np.testing.assert_allclose(next_state.numpy(), [[expected_x, expected_tau]], atol=1e-14)
    np.testing.assert_allclose(observation.numpy(), [[expected_x**2 / 20.0]], atol=1e-14)
    np.testing.assert_allclose(residual.numpy(), [[0.0]], atol=1e-14)
    assert model.config.approximation_label == "autonomous_phase_nonlinear_growth_testing_fixture"
