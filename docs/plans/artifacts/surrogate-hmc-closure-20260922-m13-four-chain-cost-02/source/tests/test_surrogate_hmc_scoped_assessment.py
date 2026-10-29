"""Fail-closed and uncertainty-design checks for prospective retained diagnostics."""
import pytest
import tensorflow as tf

from bayesfilter.testing.surrogate_hmc_scoped_assessment import assess, diagnostic_calculator
from bayesfilter.testing.surrogate_hmc_ar2_sampling import precision_design


def test_constant_chains_cannot_establish_equivalence_even_with_equal_means():
    points = tf.zeros([4, 1000, 2], tf.float64)
    values = tf.ones([4, 1000, 13], tf.float64)
    result = assess(points, values, tf.ones([13], tf.float64), tf.zeros([13], tf.float64),
                    tf.ones([13], tf.float64), seed=[12, 34])
    assert result["status"] == "INCONCLUSIVE"
    assert result["qualified"] is False
    assert result["owner_acceptance"] == "NOT_ASSIGNED"
    assert diagnostic_calculator(1000) is diagnostic_calculator(1000)


def test_reference_uncertainty_consumes_margin_and_changes_precision_requirement():
    variance = tf.ones([13], tf.float64)
    margin = tf.ones([13], tf.float64)*.1
    clean = precision_design(variance, margin, tf.zeros([13], tf.float64))
    noisy = precision_design(variance, margin, margin*.5)
    tf.debugging.assert_near(tf.constant(noisy["required_effective_draws_total"]),
                             4*tf.constant(clean["required_effective_draws_total"]))
    with pytest.raises(tf.errors.InvalidArgumentError):
        precision_design(variance, margin, margin)
