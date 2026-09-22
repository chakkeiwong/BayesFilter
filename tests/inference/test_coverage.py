"""Phase 0 Task 0.3 smoke tests for joint credible-region coverage.

Calibration is the point of these tests: a correct sampler must be covered by
the joint region at close to the nominal rate, and the conjunction of marginals
must visibly fail at the 0.95**P rate that motivated choosing the joint region.
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import tensorflow as tf

from bayesfilter.inference.coverage import (
    joint_mahalanobis_coverage,
    marginal_coverage_diagnostic,
)

DTYPE = tf.float64


def _gaussian_samples(num_samples, scales, seed, mean=None):
    """Draw independent Gaussian samples with the given per-coordinate scales."""
    generator = tf.random.Generator.from_seed(seed)
    scales = tf.constant(scales, DTYPE)
    draws = generator.normal([num_samples, int(scales.shape[0])], dtype=DTYPE)
    draws = draws * scales
    if mean is not None:
        draws = draws + tf.constant(mean, DTYPE)
    return draws


def test_centered_truth_is_covered():
    """True theta at the posterior center must be inside the region."""
    samples = _gaussian_samples(4000, [1.0, 2.0, 3.0], seed=11)
    result = joint_mahalanobis_coverage(samples, tf.zeros([3], DTYPE))

    assert result["covers"] is True
    assert result["p_value"] > 0.05
    assert result["parameter_dim"] == 3
    assert result["num_samples"] == 4000


def test_far_truth_is_not_covered():
    """True theta many posterior SDs away must fall outside the region."""
    samples = _gaussian_samples(4000, [1.0, 1.0, 1.0], seed=12)
    far = tf.constant([10.0, 10.0, 10.0], DTYPE)
    result = joint_mahalanobis_coverage(samples, far)

    assert result["covers"] is False
    assert result["mahalanobis_squared"] > result["threshold"]
    assert result["p_value"] < 1.0e-6


def test_threshold_matches_chi_squared_quantile():
    """The region threshold is the chi-squared (1-alpha) quantile at P df."""
    samples = _gaussian_samples(2000, [1.0] * 5, seed=13)
    result = joint_mahalanobis_coverage(samples, tf.zeros([5], DTYPE), alpha=0.05)
    # chi2(df=5) 95th percentile = 11.0705
    assert result["threshold"] == pytest.approx(11.0705, rel=1.0e-3)


def test_joint_region_is_calibrated_across_replications():
    """
    The calibration check that justifies the Option A decision.

    A correct sampler should be covered at ~1-alpha. Anything far from nominal
    means the statistic or the threshold is wrong.
    """
    covered = 0
    replications = 200
    for replication in range(replications):
        samples = _gaussian_samples(600, [1.0, 2.0, 0.5], seed=5000 + replication)
        result = joint_mahalanobis_coverage(samples, tf.zeros([3], DTYPE))
        covered += int(result["covers"])

    rate = covered / replications
    # Binomial SE at p=0.95, n=200 is ~0.015; allow a wide band so the test
    # is not itself flaky, but tight enough to catch a miscalibrated region.
    assert 0.88 <= rate <= 1.0, f"joint coverage rate {rate:.3f} off nominal 0.95"


def test_marginal_conjunction_is_weaker_than_joint_region():
    """
    Records the defect that motivated Option A.

    P(all P marginals cover) = 0.95**P for a correct sampler, so at P=5 a
    correct sampler fails the conjunction about a quarter of the time.
    """
    samples = _gaussian_samples(2000, [1.0] * 5, seed=14)
    diagnostic = marginal_coverage_diagnostic(samples, tf.zeros([5], DTYPE))

    assert diagnostic["parameter_dim"] == 5
    assert diagnostic["probability_all_covered_if_correct"] == pytest.approx(
        0.95**5, rel=1e-9
    )
    assert diagnostic["probability_all_covered_if_correct"] < 0.78
    assert len(diagnostic["covered_per_parameter"]) == 5


def test_marginal_diagnostic_localizes_a_single_bad_coordinate():
    """One shifted coordinate should show up as exactly one marginal miss."""
    samples = _gaussian_samples(3000, [1.0, 1.0, 1.0], seed=15)
    truth = tf.constant([0.0, 8.0, 0.0], DTYPE)
    diagnostic = marginal_coverage_diagnostic(samples, truth)

    assert diagnostic["covered_per_parameter"] == [True, False, True]
    assert diagnostic["count_covered"] == 2
    assert diagnostic["all_marginals_covered"] is False


def test_correlated_posterior_is_handled():
    """A strongly correlated posterior must still admit a valid region."""
    base = _gaussian_samples(3000, [1.0, 1.0], seed=16)
    # Induce correlation: second coordinate tracks the first.
    correlated = tf.stack(
        [base[:, 0], 0.95 * base[:, 0] + 0.05 * base[:, 1]], axis=1
    )
    result = joint_mahalanobis_coverage(correlated, tf.zeros([2], DTYPE))

    assert result["covers"] is True
    assert result["mahalanobis_squared"] >= 0.0


def test_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="rank 2"):
        joint_mahalanobis_coverage(tf.zeros([10], DTYPE), tf.zeros([1], DTYPE))


def test_rejects_dimension_mismatch():
    samples = _gaussian_samples(100, [1.0, 1.0, 1.0], seed=17)
    with pytest.raises(ValueError, match="does not match"):
        joint_mahalanobis_coverage(samples, tf.zeros([2], DTYPE))


def test_rejects_too_few_samples():
    samples = _gaussian_samples(3, [1.0, 1.0, 1.0, 1.0, 1.0], seed=18)
    with pytest.raises(ValueError, match="num_samples > parameter_dim"):
        joint_mahalanobis_coverage(samples, tf.zeros([5], DTYPE))


def test_rejects_non_finite_samples():
    samples = _gaussian_samples(100, [1.0, 1.0], seed=19)
    poisoned = tf.tensor_scatter_nd_update(
        samples, [[0, 0]], tf.constant([float("nan")], DTYPE)
    )
    with pytest.raises(ValueError, match="non-finite"):
        joint_mahalanobis_coverage(poisoned, tf.zeros([2], DTYPE))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
