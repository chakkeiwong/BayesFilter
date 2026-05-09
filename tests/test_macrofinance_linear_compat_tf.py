from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.linear.kalman_qr_derivatives_tf import (
    tf_qr_linear_gaussian_score_hessian,
)
from bayesfilter.linear.kalman_qr_tf import tf_qr_linear_gaussian_log_likelihood
from bayesfilter.linear.types_tf import (
    TFLinearGaussianStateSpace,
    TFLinearGaussianStateSpaceDerivatives,
)


MACROFINANCE_ROOT = Path("/home/chakwong/MacroFinance")


def _macrofinance_module(name: str):
    if not MACROFINANCE_ROOT.exists():
        pytest.skip("MacroFinance checkout is not available")
    root = str(MACROFINANCE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module(name)


def _to_tf_model(model) -> TFLinearGaussianStateSpace:
    return TFLinearGaussianStateSpace(
        initial_mean=tf.convert_to_tensor(model.initial_mean, dtype=tf.float64),
        initial_covariance=tf.convert_to_tensor(model.initial_covariance, dtype=tf.float64),
        transition_offset=tf.convert_to_tensor(model.transition_offset, dtype=tf.float64),
        transition_matrix=tf.convert_to_tensor(model.transition_matrix, dtype=tf.float64),
        transition_covariance=tf.convert_to_tensor(
            model.transition_covariance,
            dtype=tf.float64,
        ),
        observation_offset=tf.convert_to_tensor(model.observation_offset, dtype=tf.float64),
        observation_matrix=tf.convert_to_tensor(model.observation_matrix, dtype=tf.float64),
        observation_covariance=tf.convert_to_tensor(
            model.observation_covariance,
            dtype=tf.float64,
        ),
    )


def _to_tf_derivatives(derivatives) -> TFLinearGaussianStateSpaceDerivatives:
    return TFLinearGaussianStateSpaceDerivatives(
        d_initial_mean=tf.convert_to_tensor(derivatives.d_initial_mean, dtype=tf.float64),
        d_initial_covariance=tf.convert_to_tensor(
            derivatives.d_initial_covariance,
            dtype=tf.float64,
        ),
        d_transition_offset=tf.convert_to_tensor(
            derivatives.d_transition_offset,
            dtype=tf.float64,
        ),
        d_transition_matrix=tf.convert_to_tensor(
            derivatives.d_transition_matrix,
            dtype=tf.float64,
        ),
        d_transition_covariance=tf.convert_to_tensor(
            derivatives.d_transition_covariance,
            dtype=tf.float64,
        ),
        d_observation_offset=tf.convert_to_tensor(
            derivatives.d_observation_offset,
            dtype=tf.float64,
        ),
        d_observation_matrix=tf.convert_to_tensor(
            derivatives.d_observation_matrix,
            dtype=tf.float64,
        ),
        d_observation_covariance=tf.convert_to_tensor(
            derivatives.d_observation_covariance,
            dtype=tf.float64,
        ),
        d2_initial_mean=tf.convert_to_tensor(
            derivatives.d2_initial_mean,
            dtype=tf.float64,
        ),
        d2_initial_covariance=tf.convert_to_tensor(
            derivatives.d2_initial_covariance,
            dtype=tf.float64,
        ),
        d2_transition_offset=tf.convert_to_tensor(
            derivatives.d2_transition_offset,
            dtype=tf.float64,
        ),
        d2_transition_matrix=tf.convert_to_tensor(
            derivatives.d2_transition_matrix,
            dtype=tf.float64,
        ),
        d2_transition_covariance=tf.convert_to_tensor(
            derivatives.d2_transition_covariance,
            dtype=tf.float64,
        ),
        d2_observation_offset=tf.convert_to_tensor(
            derivatives.d2_observation_offset,
            dtype=tf.float64,
        ),
        d2_observation_matrix=tf.convert_to_tensor(
            derivatives.d2_observation_matrix,
            dtype=tf.float64,
        ),
        d2_observation_covariance=tf.convert_to_tensor(
            derivatives.d2_observation_covariance,
            dtype=tf.float64,
        ),
    )


def _fixture():
    helpers = _macrofinance_module("tests.test_generic_lgssm_autodiff_validation")
    base_case = helpers._make_seeded_stable_lgssm_case(state_dim=2, seed=902)
    params = helpers.SMALL_DIM_PARAMETER_POINTS[1]
    case = helpers._case_with_params(base_case, params)
    model, derivatives = helpers._build_state_space_with_derivatives(case, case.params)
    return helpers, case, model, derivatives


def _sparse_mask(observations: np.ndarray) -> np.ndarray:
    mask = np.ones_like(observations, dtype=bool)
    mask[1, 0] = False
    mask[2, :] = False
    return mask


def test_macrofinance_static_fixture_uses_time_invariant_derivative_contract() -> None:
    _helpers, _case, model, derivatives = _fixture()

    assert np.asarray(model.transition_matrix).ndim == 2
    assert np.asarray(model.transition_covariance).ndim == 2
    assert np.asarray(model.observation_matrix).ndim == 2
    assert np.asarray(model.observation_covariance).ndim == 2
    assert np.asarray(derivatives.d_transition_matrix).ndim == 3
    assert np.asarray(derivatives.d_observation_covariance).ndim == 3
    assert np.asarray(derivatives.d2_transition_matrix).ndim == 4
    assert np.asarray(derivatives.d2_observation_covariance).ndim == 4


def test_bayesfilter_qr_value_matches_macrofinance_static_dense_and_masked() -> None:
    helpers, case, model, _derivatives = _fixture()
    mf_qr = _macrofinance_module("filters.tf_qr_sqrt_differentiated_kalman")
    bf_model = _to_tf_model(model)
    observations = tf.convert_to_tensor(case.observations, dtype=tf.float64)
    mask = tf.convert_to_tensor(_sparse_mask(case.observations), dtype=tf.bool)
    jitter = tf.constant(helpers.JITTER, dtype=tf.float64)

    macrofinance_dense = mf_qr.tf_qr_sqrt_kalman_loglik(
        case.observations,
        model,
        jitter=helpers.JITTER,
    )
    bayesfilter_dense = tf_qr_linear_gaussian_log_likelihood(
        observations,
        bf_model,
        backend="tf_qr",
        jitter=jitter,
    )
    macrofinance_masked = mf_qr.tf_qr_sqrt_masked_kalman_loglik(
        case.observations,
        model,
        _sparse_mask(case.observations),
        jitter=helpers.JITTER,
    )
    bayesfilter_masked = tf_qr_linear_gaussian_log_likelihood(
        observations,
        bf_model,
        backend="tf_masked_qr",
        observation_mask=mask,
        jitter=jitter,
    )

    np.testing.assert_allclose(
        bayesfilter_dense.log_likelihood.numpy(),
        macrofinance_dense.numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        bayesfilter_masked.log_likelihood.numpy(),
        macrofinance_masked.numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    assert bayesfilter_dense.metadata.filter_name == "tf_qr_sqrt_kalman"
    assert bayesfilter_masked.metadata.filter_name == "tf_qr_sqrt_masked_kalman"
    assert bayesfilter_masked.diagnostics.mask_convention == "static_dummy_row"


def test_bayesfilter_qr_score_hessian_matches_macrofinance_dense_backend() -> None:
    helpers, case, model, derivatives = _fixture()
    mf_qr = _macrofinance_module("filters.tf_qr_sqrt_differentiated_kalman")
    bf_model = _to_tf_model(model)
    bf_derivatives = _to_tf_derivatives(derivatives)

    macrofinance_loglik, macrofinance_score, macrofinance_hessian = (
        mf_qr.tf_qr_sqrt_differentiated_kalman_loglik(
            case.observations,
            model,
            derivatives,
            jitter=helpers.JITTER,
        )
    )
    bayesfilter = tf_qr_linear_gaussian_score_hessian(
        tf.convert_to_tensor(case.observations, dtype=tf.float64),
        bf_model,
        bf_derivatives,
        backend="tf_qr_sqrt",
        jitter=tf.constant(helpers.JITTER, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        bayesfilter.log_likelihood.numpy(),
        macrofinance_loglik.numpy(),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        bayesfilter.score.numpy(),
        macrofinance_score.numpy(),
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        bayesfilter.hessian.numpy(),
        macrofinance_hessian.numpy(),
        rtol=1e-8,
        atol=1e-8,
    )
    assert bayesfilter.metadata.filter_name == "tf_qr_sqrt_differentiated_kalman"


def test_bayesfilter_masked_qr_score_hessian_matches_macrofinance_sparse_oracle() -> None:
    helpers, case, model, derivatives = _fixture()
    mf_masked_qr = _macrofinance_module("filters.masked_qr_sqrt_differentiated_kalman")
    bf_model = _to_tf_model(model)
    bf_derivatives = _to_tf_derivatives(derivatives)
    mask = _sparse_mask(case.observations)

    macrofinance_loglik, macrofinance_score, macrofinance_hessian = (
        mf_masked_qr.masked_qr_sqrt_differentiated_kalman_loglik(
            case.observations,
            model,
            derivatives,
            mask,
            jitter=helpers.JITTER,
        )
    )
    bayesfilter = tf_qr_linear_gaussian_score_hessian(
        tf.convert_to_tensor(case.observations, dtype=tf.float64),
        bf_model,
        bf_derivatives,
        backend="tf_masked_qr_sqrt",
        observation_mask=tf.convert_to_tensor(mask, dtype=tf.bool),
        jitter=tf.constant(helpers.JITTER, dtype=tf.float64),
    )

    np.testing.assert_allclose(
        bayesfilter.log_likelihood.numpy(),
        macrofinance_loglik,
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        bayesfilter.score.numpy(),
        macrofinance_score,
        rtol=1e-8,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        bayesfilter.hessian.numpy(),
        macrofinance_hessian,
        rtol=1e-7,
        atol=1e-6,
    )
    assert bayesfilter.metadata.filter_name == "tf_qr_sqrt_masked_differentiated_kalman"
    assert bayesfilter.diagnostics.mask_convention == "static_dummy_row"
