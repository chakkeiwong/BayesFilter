from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.ledh_younis_kdm_lgssm_reference_tf import (
    scalar_lgssm_value_and_score,
)


def test_scalar_lgssm_reference_score_matches_value_finite_difference():
    dtype = tf.float64
    observations = tf.constant([0.3, -0.8, 0.4, 1.1], dtype)
    theta = tf.constant([0.72], dtype)
    epsilon = 1.0e-6
    value, score = scalar_lgssm_value_and_score(theta, observations)
    plus, _ = scalar_lgssm_value_and_score(
        theta + tf.constant([epsilon], dtype), observations
    )
    minus, _ = scalar_lgssm_value_and_score(
        theta - tf.constant([epsilon], dtype), observations
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert np.isfinite(float(value.numpy()))
    assert np.isfinite(float(score.numpy()))
    np.testing.assert_allclose(
        score.numpy(), finite_difference.numpy(), rtol=2.0e-8, atol=2.0e-9
    )


def test_scalar_lgssm_reference_rejects_bad_shapes():
    with np.testing.assert_raises(ValueError):
        scalar_lgssm_value_and_score(
            tf.constant([0.7, 0.1], tf.float64),
            tf.constant([0.2], tf.float64),
        )
