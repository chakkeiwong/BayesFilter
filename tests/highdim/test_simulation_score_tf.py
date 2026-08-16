from __future__ import annotations

import pytest
import tensorflow as tf

from bayesfilter.highdim.simulation_score_tf import (
    fisher_identity_simulation_score,
)


def test_fisher_identity_estimate_uses_normalized_likelihood_weights() -> None:
    result = fisher_identity_simulation_score(
        tf.constant([0.0, 0.0], tf.float64),
        tf.constant([[1.0, 2.0], [3.0, 4.0]], tf.float64),
    )
    tf.debugging.assert_near(result.log_marginal, tf.constant(0.0, tf.float64))
    tf.debugging.assert_near(result.score, tf.constant([2.0, 3.0], tf.float64))
    tf.debugging.assert_near(result.effective_sample_fraction, tf.constant(1.0, tf.float64))
    assert bool(result.finite.numpy())
    assert not bool(result.collapsed.numpy())


def test_fisher_identity_reports_weight_collapse_without_hiding_the_estimate() -> None:
    result = fisher_identity_simulation_score(
        tf.constant([0.0, -100.0, -100.0], tf.float64),
        tf.constant([[7.0], [0.0], [0.0]], tf.float64),
        minimum_effective_sample_fraction=0.5,
    )
    assert bool(result.finite.numpy())
    assert bool(result.collapsed.numpy())
    assert float(result.maximum_normalized_weight.numpy()) > 0.99
    tf.debugging.assert_near(result.score, tf.constant([7.0], tf.float64), atol=1e-12)


@pytest.mark.parametrize(
    "likelihood,score",
    [
        (tf.constant([[0.0]], tf.float64), tf.constant([1.0], tf.float64)),
        (tf.constant([0.0, 1.0], tf.float64), tf.constant([[1.0]], tf.float64)),
    ],
)
def test_fisher_identity_rejects_invalid_shapes(likelihood: tf.Tensor, score: tf.Tensor) -> None:
    with pytest.raises(ValueError):
        fisher_identity_simulation_score(likelihood, score)


def test_fisher_identity_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        fisher_identity_simulation_score(
            tf.constant([0.0], tf.float64),
            tf.constant([[0.0]], tf.float64),
            minimum_effective_sample_fraction=1.1,
        )
