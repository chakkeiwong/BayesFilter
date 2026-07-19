from __future__ import annotations

import ast
import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest
import tensorflow as tf

import bayesfilter
import bayesfilter.nonlinear as nonlinear
import bayesfilter.nonlinear.batched_svd_sigma_point_tf as production_module
from bayesfilter.nonlinear import (
    tf_batched_svd_sigma_point_value_and_score_custom_gradient,
)
from bayesfilter.nonlinear.experimental_batched_svd_sigma_point_tf import (
    TFBatchedStructuralStateSpace,
    tf_batched_svd_sigma_point_value_and_score,
)
from docs.benchmarks.benchmark_experimental_batched_svd_sigma_point_cpu_gpu import (
    _batched_model_and_derivatives,
    _stable_fixture,
    _to_tensors,
)


def _fixture(*, batch_size: int = 2, time_steps: int = 3) -> dict[str, tf.Tensor]:
    return _to_tensors(
        _stable_fixture(
            batch_size=batch_size,
            time_steps=time_steps,
            state_dim=2,
            obs_dim=2,
            parameter_dim=2,
        )
    )


def _theta_for_fixture(tensors: dict[str, tf.Tensor]) -> tf.Tensor:
    batch_size = int(tensors["initial_mean"].shape[0])
    parameter_dim = int(tensors["d_initial_mean"].shape[1])
    return tf.zeros([batch_size, parameter_dim], dtype=tf.float64)


def _model_derivatives(tensors: dict[str, tf.Tensor]):
    return _batched_model_and_derivatives(tensors)


def test_batched_svd_custom_gradient_public_imports() -> None:
    assert hasattr(
        nonlinear,
        "tf_batched_svd_sigma_point_value_and_score_custom_gradient",
    )
    assert hasattr(
        bayesfilter,
        "tf_batched_svd_sigma_point_value_and_score_custom_gradient",
    )
    assert (
        bayesfilter.tf_batched_svd_sigma_point_value_and_score_custom_gradient
        is tf_batched_svd_sigma_point_value_and_score_custom_gradient
    )
    assert (
        nonlinear.tf_batched_svd_sigma_point_value_and_score_custom_gradient
        is tf_batched_svd_sigma_point_value_and_score_custom_gradient
    )


def test_batched_svd_custom_gradient_source_contract() -> None:
    source = inspect.getsource(
        production_module.tf_batched_svd_sigma_point_value_and_score_custom_gradient
    )
    tree = ast.parse(source)

    assert "tf.custom_gradient" in source
    assert "tf_batched_svd_sigma_point_value_and_score" in source
    assert "np." not in source
    assert not any(
        isinstance(node, ast.Name) and node.id in {"np", "numpy"}
        for node in ast.walk(tree)
    )
    assert "import numpy" not in inspect.getsource(production_module)
    assert not any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))


def test_batched_svd_custom_gradient_matches_experimental_principal_sqrt() -> None:
    tensors = _fixture()
    model, derivatives = _model_derivatives(tensors)
    theta = _theta_for_fixture(tensors)

    value, score, diagnostics = (
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            theta,
            tensors["observations"],
            model,
            derivatives,
            spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        )
    )
    expected_value, expected_score, expected_diagnostics = (
        tf_batched_svd_sigma_point_value_and_score(
            tensors["observations"],
            model,
            derivatives,
            backend="tf_principal_sqrt_ukf",
            spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        )
    )

    np.testing.assert_allclose(value.numpy(), expected_value.numpy(), atol=1.0e-8)
    np.testing.assert_allclose(
        score.numpy(),
        expected_score.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert diagnostics["backend"].numpy() == expected_diagnostics["backend"].numpy()
    assert diagnostics["filter_gradient_api"].numpy() == (
        b"bayesfilter.nonlinear."
        b"tf_batched_svd_sigma_point_value_and_score_custom_gradient"
    )
    assert diagnostics["time_recursion"].numpy() == b"tf.while_loop"
    assert bool(diagnostics["filter_autodiff_allowed_for_hmc"].numpy()) is False


def test_batched_svd_custom_gradient_tape_uses_returned_score() -> None:
    tensors = _fixture()
    model, derivatives = _model_derivatives(tensors)
    theta = tf.Variable(_theta_for_fixture(tensors))
    weights = tf.constant([1.25, -0.5], dtype=tf.float64)

    with tf.GradientTape() as tape:
        value, score, _diagnostics = (
            tf_batched_svd_sigma_point_value_and_score_custom_gradient(
                theta,
                tensors["observations"],
                model,
                derivatives,
                spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            )
        )
        objective = tf.reduce_sum(weights * value)
    gradient = tape.gradient(objective, theta)

    np.testing.assert_allclose(
        gradient.numpy(),
        (weights[:, tf.newaxis] * score).numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_batched_svd_custom_gradient_supports_row_aligned_observations() -> None:
    tensors = _fixture(batch_size=2, time_steps=3)
    model, derivatives = _model_derivatives(tensors)
    theta = tf.Variable(_theta_for_fixture(tensors))
    observations = tf.stack(
        (tensors["observations"], tensors["observations"] + 0.025), axis=0
    )
    weights = tf.constant([1.25, -0.5], dtype=tf.float64)

    with tf.GradientTape() as tape:
        value, score, diagnostics = (
            tf_batched_svd_sigma_point_value_and_score_custom_gradient(
                theta,
                observations,
                model,
                derivatives,
                spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            )
        )
        objective = tf.reduce_sum(weights * value)
    gradient = tape.gradient(objective, theta)

    np.testing.assert_allclose(
        gradient.numpy(),
        (weights[:, tf.newaxis] * score).numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
    assert diagnostics["batching_contract"].numpy() == (
        b"batch_over_parameter_proposals_and_row_aligned_observations"
    )


def test_batched_svd_custom_gradient_blocks_model_autodiff_leakage() -> None:
    tensors = _fixture()
    model, derivatives = _model_derivatives(tensors)
    theta = tf.Variable(_theta_for_fixture(tensors))

    with tf.GradientTape() as tape:
        leaked_model = TFBatchedStructuralStateSpace(
            initial_mean=model.initial_mean + 0.1 * theta,
            initial_covariance=model.initial_covariance,
            innovation_covariance=model.innovation_covariance,
            observation_covariance=model.observation_covariance,
            transition_fn=model.transition_fn,
            observation_fn=model.observation_fn,
            deterministic_residual_fn=model.deterministic_residual_fn,
            name=model.name,
        )
        value, score, _diagnostics = (
            tf_batched_svd_sigma_point_value_and_score_custom_gradient(
                theta,
                tensors["observations"],
                leaked_model,
                derivatives,
                spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            )
        )
        objective = tf.reduce_sum(value)
    gradient = tape.gradient(objective, theta)

    np.testing.assert_allclose(
        gradient.numpy(),
        score.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )


def test_batched_svd_custom_gradient_rejects_non_authorized_backend() -> None:
    tensors = _fixture()
    model, derivatives = _model_derivatives(tensors)
    theta = _theta_for_fixture(tensors)

    with pytest.raises(ValueError, match="requires backend='tf_principal_sqrt_ukf'"):
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            theta,
            tensors["observations"],
            model,
            derivatives,
            backend="tf_svd_ukf",
        )


def test_batched_svd_custom_gradient_cpu_xla_smoke() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    tensors = _fixture()
    model, derivatives = _model_derivatives(tensors)
    theta = _theta_for_fixture(tensors)
    eager_value, eager_score, _diagnostics = (
        tf_batched_svd_sigma_point_value_and_score_custom_gradient(
            theta,
            tensors["observations"],
            model,
            derivatives,
            spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
        )
    )

    @tf.function(jit_compile=True, reduce_retracing=True)
    def compiled_call(theta_input: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        value, score, _diagnostics = (
            tf_batched_svd_sigma_point_value_and_score_custom_gradient(
                theta_input,
                tensors["observations"],
                model,
                derivatives,
                spectral_gap_tolerance=tf.constant(1.0e-10, dtype=tf.float64),
            )
        )
        return value, score

    xla_value, xla_score = compiled_call(theta)

    np.testing.assert_allclose(xla_value.numpy(), eager_value.numpy(), atol=1.0e-8)
    np.testing.assert_allclose(
        xla_score.numpy(),
        eager_score.numpy(),
        rtol=1.0e-7,
        atol=1.0e-7,
    )
