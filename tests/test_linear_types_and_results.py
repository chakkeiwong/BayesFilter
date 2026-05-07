from types import MappingProxyType

import numpy as np
import pytest

from bayesfilter import (
    FilterDerivativeResult,
    FilterRunMetadata,
    FilterValueResult,
    LinearGaussianStateSpace,
    LinearGaussianStateSpaceDerivatives,
)


def _metadata():
    return FilterRunMetadata(
        filter_name="fixture",
        partition=None,
        integration_space="full_state",
        deterministic_completion="none",
    )


def test_linear_state_space_derivative_shape_validation():
    p, n, m = 2, 3, 4
    derivatives = LinearGaussianStateSpaceDerivatives(
        d_initial_mean=np.zeros((p, n)),
        d_initial_covariance=np.zeros((p, n, n)),
        d_transition_offset=np.zeros((p, n)),
        d_transition_matrix=np.zeros((p, n, n)),
        d_transition_covariance=np.zeros((p, n, n)),
        d_observation_offset=np.zeros((p, m)),
        d_observation_matrix=np.zeros((p, m, n)),
        d_observation_covariance=np.zeros((p, m, m)),
        d2_initial_mean=np.zeros((p, p, n)),
        d2_initial_covariance=np.zeros((p, p, n, n)),
        d2_transition_offset=np.zeros((p, p, n)),
        d2_transition_matrix=np.zeros((p, p, n, n)),
        d2_transition_covariance=np.zeros((p, p, n, n)),
        d2_observation_offset=np.zeros((p, p, m)),
        d2_observation_matrix=np.zeros((p, p, m, n)),
        d2_observation_covariance=np.zeros((p, p, m, m)),
    )

    assert derivatives.parameter_dim == p
    assert derivatives.state_dim == n
    assert derivatives.observation_dim == m


def test_linear_state_space_rejects_bad_shapes():
    with pytest.raises(ValueError, match="expected"):
        LinearGaussianStateSpace(
            initial_mean=np.zeros(2),
            initial_covariance=np.eye(2),
            transition_offset=np.zeros(2),
            transition_matrix=np.zeros((3, 2)),
            transition_covariance=np.eye(2),
            observation_offset=np.zeros(1),
            observation_matrix=np.zeros((1, 2)),
            observation_covariance=np.eye(1),
        )


def test_filter_results_freeze_arrays_and_diagnostics():
    result = FilterValueResult(
        log_likelihood=1,
        filtered_means=np.zeros((1, 2)),
        filtered_covariances=np.zeros((1, 2, 2)),
        metadata=_metadata(),
        diagnostics={"backend": "fixture", "values": np.array([1.0])},
    )

    assert isinstance(result.diagnostics, MappingProxyType)
    with pytest.raises(ValueError):
        result.filtered_means[0, 0] = 1.0
    with pytest.raises(TypeError):
        result.diagnostics["backend"] = "other"


def test_filter_derivative_result_freezes_score_hessian_and_trace():
    result = FilterDerivativeResult(
        log_likelihood=0.0,
        score=np.zeros(2),
        hessian=np.zeros((2, 2)),
        metadata=_metadata(),
        trace=({"grad": np.ones(2)},),
    )

    with pytest.raises(ValueError):
        result.score[0] = 2.0
    with pytest.raises(ValueError):
        result.hessian[0, 0] = 2.0
    assert isinstance(result.trace[0], MappingProxyType)
