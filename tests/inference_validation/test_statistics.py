import numpy as np
import pytest

from bayesfilter.testing.inference_validation.engines.statistics import (
    randomized_rank, rank_uniform_test, two_sample_test, binomial_interval, accuracy_assessment,
)


def test_randomized_ties_cover_every_ordinal():
    rng = np.random.default_rng(89)
    ranks = [randomized_rank(0, np.zeros(7), rng) for _ in range(2048)]
    assert set(ranks) == set(range(8))
    assert all(175 < count < 330 for count in np.bincount(ranks))
    assert randomized_rank(2, [0, 1, 3], rng) == 2
    with pytest.raises(ValueError, match="finite"):
        randomized_rank(np.nan, [0], rng)


def test_multinomial_null_plus_one_and_multiplicity():
    kwargs = dict(null_draws=1999, seed=14, alpha=.05, multiplicity=4)
    uniform = rank_uniform_test(list(range(8))*32, 7, **kwargs)
    assert uniform["p_value"] == 1.
    skew = rank_uniform_test([0]*256, 7, **kwargs)
    assert skew["p_value"] == 1/2000
    assert skew["threshold"] == .05/4
    assert not skew["accuracy_established"]


def test_tied_permutation_ks_and_strong_location_defect():
    kwargs = dict(seed=17, permutations=399, alpha=.05)
    assert two_sample_test([0, 0, 1], [0, 0, 1], **kwargs)["p_value"] == 1.
    result = two_sample_test(np.zeros(32), np.ones(32), **kwargs)
    assert result["finding"] == "discrepancy_detected"
    assert result["p_value"] >= 1/400


def test_clopper_pearson_endpoints_and_empty_draws():
    assert binomial_interval(0, 10)[0] == 0.
    assert binomial_interval(10, 10)[1] == 1.
    assert binomial_interval(0, 10)[1] == pytest.approx(1-.025**.1)
    r = accuracy_assessment(np.zeros((0, 4, 2)), np.zeros((10, 2)), tolerance=.1)
    assert r["finding"] == "unavailable"


def test_correlated_external_reference_does_not_receive_iid_standard_error():
    rng = np.random.default_rng(971)
    result = accuracy_assessment(rng.normal(size=(64,4,1)), rng.normal(size=(100,1)),
                                 tolerance=.25, reference_iid=False)
    bounded, mean, _ = result["quantities"]
    assert bounded["standard_error"] is None
    assert mean["reference_mean_se"] is None
    assert not result["reference_iid"]
