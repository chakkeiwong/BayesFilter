from __future__ import annotations

import pytest
import tensorflow as tf

from bayesfilter import (
    DEFAULT_SRUKF_BACKEND,
    HISTORICAL_PRINCIPAL_SQRT_SRUKF_BACKEND,
    TFFactorSRUKFDerivatives,
    TFFactorSRUKFModel,
    default_srukf_backend,
    resolve_srukf_backend,
    srukf_backend_metadata,
    srukf_backend_status,
    tf_default_srukf_value_and_score,
)


def _factor_contract() -> tuple[TFFactorSRUKFModel, TFFactorSRUKFDerivatives]:
    model = TFFactorSRUKFModel(
        tf.constant([[0.0]], tf.float64),
        tf.constant([[[0.5]]], tf.float64),
        tf.constant([[[0.1]]], tf.float64),
        tf.constant([[[0.2]]], tf.float64),
        lambda state, innovation: state + innovation,
        lambda state: state,
    )
    derivatives = TFFactorSRUKFDerivatives(
        tf.zeros([1, 1, 1], tf.float64),
        tf.zeros([1, 1, 1, 1], tf.float64),
        tf.zeros([1, 1, 1, 1], tf.float64),
        tf.zeros([1, 1, 1, 1], tf.float64),
        lambda state, innovation: tf.ones([1, tf.shape(state)[1], 1, 1], tf.float64),
        lambda state, innovation: tf.ones([1, tf.shape(state)[1], 1, 1], tf.float64),
        lambda state, innovation: tf.zeros([1, 1, tf.shape(state)[1], 1], tf.float64),
        lambda state: tf.ones([1, tf.shape(state)[1], 1, 1], tf.float64),
        lambda state: tf.zeros([1, 1, tf.shape(state)[1], 1], tf.float64),
    )
    return model, derivatives


def test_direct_factor_is_repository_default_and_principal_root_is_historical() -> None:
    assert DEFAULT_SRUKF_BACKEND == "direct_factor_srukf"
    assert default_srukf_backend() == DEFAULT_SRUKF_BACKEND
    assert resolve_srukf_backend() == DEFAULT_SRUKF_BACKEND
    assert srukf_backend_status(DEFAULT_SRUKF_BACKEND) == "default"
    assert srukf_backend_status(HISTORICAL_PRINCIPAL_SQRT_SRUKF_BACKEND) == (
        "historical_reference"
    )
    assert srukf_backend_metadata(DEFAULT_SRUKF_BACKEND)["backend_contract"] == (
        "TFFactorSRUKFModel"
    )


def test_unknown_backend_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown SR-UKF backend"):
        resolve_srukf_backend("not-a-srukf-backend")


def test_default_route_accepts_only_factor_contract() -> None:
    model, derivatives = _factor_contract()
    result = tf_default_srukf_value_and_score(
        tf.constant([[[0.1], [0.2]]], tf.float64),
        model,
        derivatives,
        jit_compile=False,
    )
    assert result.diagnostics["backend"].numpy() == b"direct_factor_srukf"
    assert result.diagnostics["backend_status"].numpy() == b"default"

    with pytest.raises(TypeError, match="requires TFFactorSRUKFModel"):
        tf_default_srukf_value_and_score(
            tf.constant([[[0.1]]], tf.float64),
            object(),
            derivatives,
            jit_compile=False,
        )
