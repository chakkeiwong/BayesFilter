from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.diagnostics import TFFilterDiagnostics, TFRegularizationDiagnostics
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)
from bayesfilter.results_tf import TFFilterDerivativeResult, TFFilterValueResult
from bayesfilter.structural import FilterRunMetadata


ROOT = Path(__file__).resolve().parents[1]


def _metadata():
    return FilterRunMetadata(
        filter_name="tf_fixture",
        partition=None,
        integration_space="full_state",
        deterministic_completion="none",
        differentiability_status="value_only",
        compiled_status="eager_tf",
    )


def _tf_model():
    return TFLinearGaussianStateSpace(
        initial_mean=tf.zeros([2], dtype=tf.float64),
        initial_covariance=tf.eye(2, dtype=tf.float64),
        transition_offset=tf.zeros([2], dtype=tf.float64),
        transition_matrix=tf.eye(2, dtype=tf.float64),
        transition_covariance=tf.eye(2, dtype=tf.float64),
        observation_offset=tf.zeros([1], dtype=tf.float64),
        observation_matrix=tf.ones([1, 2], dtype=tf.float64),
        observation_covariance=tf.eye(1, dtype=tf.float64),
    )


def test_tf_linear_state_space_preserves_tensors_and_shapes():
    model = _tf_model()

    assert isinstance(model.initial_mean, tf.Tensor)
    assert model.state_dim == 2
    assert model.observation_dim == 1


def test_tf_linear_state_space_rejects_bad_static_shape():
    with pytest.raises(ValueError, match="transition_matrix"):
        TFLinearGaussianStateSpace(
            initial_mean=tf.zeros([2], dtype=tf.float64),
            initial_covariance=tf.eye(2, dtype=tf.float64),
            transition_offset=tf.zeros([2], dtype=tf.float64),
            transition_matrix=tf.zeros([3, 2], dtype=tf.float64),
            transition_covariance=tf.eye(2, dtype=tf.float64),
            observation_offset=tf.zeros([1], dtype=tf.float64),
            observation_matrix=tf.ones([1, 2], dtype=tf.float64),
            observation_covariance=tf.eye(1, dtype=tf.float64),
        )


def test_tf_derivative_contract_validates_shapes():
    p, n, m = 2, 3, 1
    derivatives = TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=tf.zeros([p, n], dtype=tf.float64),
        d_initial_covariance=tf.zeros([p, n, n], dtype=tf.float64),
        d_transition_offset=tf.zeros([p, n], dtype=tf.float64),
        d_transition_matrix=tf.zeros([p, n, n], dtype=tf.float64),
        d_transition_covariance=tf.zeros([p, n, n], dtype=tf.float64),
        d_observation_offset=tf.zeros([p, m], dtype=tf.float64),
        d_observation_matrix=tf.zeros([p, m, n], dtype=tf.float64),
        d_observation_covariance=tf.zeros([p, m, m], dtype=tf.float64),
        d2_initial_mean=tf.zeros([p, p, n], dtype=tf.float64),
        d2_initial_covariance=tf.zeros([p, p, n, n], dtype=tf.float64),
        d2_transition_offset=tf.zeros([p, p, n], dtype=tf.float64),
        d2_transition_matrix=tf.zeros([p, p, n, n], dtype=tf.float64),
        d2_transition_covariance=tf.zeros([p, p, n, n], dtype=tf.float64),
        d2_observation_offset=tf.zeros([p, p, m], dtype=tf.float64),
        d2_observation_matrix=tf.zeros([p, p, m, n], dtype=tf.float64),
        d2_observation_covariance=tf.zeros([p, p, m, m], dtype=tf.float64),
    )

    assert derivatives.parameter_dim == p
    assert derivatives.state_dim == n
    assert derivatives.observation_dim == m


def test_tf_result_containers_preserve_tensor_values():
    diagnostics = TFFilterDiagnostics(
        backend="tf_fixture",
        mask_convention="static_dummy_row",
        regularization=TFRegularizationDiagnostics(
            jitter=tf.constant(1e-9, dtype=tf.float64),
            derivative_target="implemented_regularized_law",
        ),
    )
    value = TFFilterValueResult(
        log_likelihood=tf.constant(1.5, dtype=tf.float64),
        filtered_means=tf.zeros([1, 2], dtype=tf.float64),
        filtered_covariances=tf.zeros([1, 2, 2], dtype=tf.float64),
        metadata=_metadata(),
        diagnostics=diagnostics,
    )
    derivative = TFFilterDerivativeResult(
        log_likelihood=value.log_likelihood,
        score=tf.zeros([2], dtype=tf.float64),
        hessian=tf.zeros([2, 2], dtype=tf.float64),
        metadata=_metadata(),
        diagnostics={"backend": "tf_fixture"},
        trace=({"step": tf.constant(0, dtype=tf.int32)},),
    )

    assert isinstance(value.log_likelihood, tf.Tensor)
    assert isinstance(value.filtered_means, tf.Tensor)
    assert value.diagnostics.regularization.derivative_target == "implemented_regularized_law"
    assert isinstance(derivative.score, tf.Tensor)
    with pytest.raises(TypeError):
        derivative.diagnostics["backend"] = "other"


def test_production_tf_contract_modules_do_not_import_numpy_or_call_dot_numpy():
    module_paths = [
        ROOT / "bayesfilter" / "diagnostics.py",
        ROOT / "bayesfilter" / "results_tf.py",
        ROOT / "bayesfilter" / "linear" / "types_tf.py",
    ]

    for path in module_paths:
        text = path.read_text(encoding="utf-8")
        assert "import numpy" not in text
        assert "from numpy" not in text
        assert ".numpy(" not in text
