from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.nonlinear.factor_srukf_compat import covariance_model_to_factor_contract
from bayesfilter.nonlinear.factor_srukf_tf import TFFactorSRUKFDerivatives, TFFactorSRUKFModel


class _Legacy:
    initial_mean = tf.constant([[0.1]], tf.float64)
    initial_covariance = tf.constant([[[0.49]]], tf.float64)
    innovation_covariance = tf.constant([[[0.04]]], tf.float64)
    observation_covariance = tf.constant([[[0.09]]], tf.float64)
    transition_fn = staticmethod(lambda x, q: x + q)
    observation_fn = staticmethod(lambda x: x)


class _LegacyDerivatives:
    d_initial_mean = tf.zeros([1, 1, 1], tf.float64)
    d_initial_covariance = tf.constant([[[[0.14]]]], tf.float64)
    d_innovation_covariance = tf.constant([[[[0.08]]]], tf.float64)
    d_observation_covariance = tf.constant([[[[0.06]]]], tf.float64)
    transition_state_jacobian_fn = staticmethod(lambda x, q: tf.ones([1, tf.shape(x)[1], 1, 1], tf.float64))
    transition_innovation_jacobian_fn = staticmethod(lambda x, q: tf.ones([1, tf.shape(x)[1], 1, 1], tf.float64))
    d_transition_fn = staticmethod(lambda x, q: tf.zeros([1, 1, tf.shape(x)[1], 1], tf.float64))
    observation_state_jacobian_fn = staticmethod(lambda x: tf.ones([1, tf.shape(x)[1], 1, 1], tf.float64))
    d_observation_fn = staticmethod(lambda x: tf.zeros([1, 1, tf.shape(x)[1], 1], tf.float64))


def test_compatibility_conversion_is_one_time_factor_contract() -> None:
    model, derivatives = covariance_model_to_factor_contract(_Legacy(), _LegacyDerivatives())
    np.testing.assert_allclose(model.initial_factor, [[[0.7]]])
    np.testing.assert_allclose(model.process_factor, [[[0.2]]])
    np.testing.assert_allclose(model.observation_factor, [[[0.3]]])
    np.testing.assert_allclose(derivatives.d_initial_factor, [[[[0.1]]]])
    np.testing.assert_allclose(derivatives.d_process_factor, [[[[0.2]]]])
    np.testing.assert_allclose(derivatives.d_observation_factor, [[[[0.1]]]])
