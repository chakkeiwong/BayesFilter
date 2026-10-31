"""Independent diagnostic checks of reference quadrature and fitted proposals."""
import copy

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from bayesfilter.testing.inference_validation.references import gaussian_mixture_reference as reference


def proposal(dimension):
    rng = np.random.default_rng(851)
    centers = rng.normal(size=(8, dimension))
    centers[0] = 0
    factors = np.array([np.linalg.cholesky((a := rng.normal(size=(dimension, dimension)))
                                         @ a.T/dimension+np.eye(dimension)) for _ in range(8)])
    factors[0] = np.eye(dimension)
    return dict(centers=centers.tolist(), factors=factors.tolist(), weights=[1/8]*8)


@pytest.mark.parametrize('dimension', [2, 6])
def test_full_mixture_density_normalizer_and_means_against_scipy(dimension):
    design = proposal(dimension)
    centers, factors = np.array(design['centers']), np.array(design['factors'])
    components = [multivariate_normal(mean=c, cov=f@f.T) for c, f in zip(centers, factors)]
    def independent_log_density(q):
        return logsumexp(np.array([component.logpdf(q) for component in components]), axis=0)-np.log(8)
    q, weights, receipt = reference.weighted_draws(independent_log_density, design, power=12, seed=308)
    np.testing.assert_allclose(weights, 1/len(q), rtol=1e-12)
    assert abs(receipt['log_normalizer']) < 1e-12
    np.testing.assert_allclose(weights@q, centers.mean(axis=0), atol=.02)
    np.testing.assert_array_equal(q, reference.draw_mixture(design, power=12, seed=308))
    assert not np.array_equal(q, reference.draw_mixture(design, power=12, seed=316))


def test_reference_uses_distinct_scrambles_and_records_uncertainty():
    design = proposal(6)
    center, factor = np.array(design['centers']), np.array(design['factors'])
    components = [multivariate_normal(mean=c, cov=f@f.T) for c, f in zip(center, factor)]
    def log_target(q):
        return logsumexp([component.logpdf(q) for component in components], axis=0)-np.log(8)
    result = reference.reference_rung(log_target, lambda q: q, design, power=8, seed=501, replications=4)
    seeds = [seed for row in result['rows'] for seed in row['component_seeds']]
    assert len(set(seeds)) == 32
    assert np.all(np.array(result['pointwise_half_width']) > 0)
    assert not result['rigorous_integration_error_bound']


def test_fit_preserves_prior_and_resolves_separated_six_dimensional_clouds():
    rng = np.random.default_rng(91)
    q = rng.normal(size=(1024, 6))+np.repeat([-3., 3.], 512)[:, None]
    result = reference.fit_weighted_proposal(q, np.linspace(1, 2, len(q)))
    centers, factors = reference.validate_proposal(result)
    np.testing.assert_array_equal(centers[0], np.zeros(6))
    np.testing.assert_array_equal(factors[0], np.eye(6))
    assert centers[1:, 0].min() < -2 and centers[1:, 0].max() > 2
    assert np.all(np.linalg.eigvalsh(factors@factors.transpose(0, 2, 1)) > 0)


@pytest.mark.parametrize('defect', ['weights', 'prior', 'dimension', 'nonfinite', 'singular'])
def test_invalid_proposal_fails_closed(defect):
    design = copy.deepcopy(proposal(6))
    if defect == 'weights':
        design['weights'] = [.25]*4+[0.]*4
    elif defect == 'prior':
        design['centers'][0][0] = .01
    elif defect == 'dimension':
        design['factors'][1] = [[1.]]
    elif defect == 'nonfinite':
        design['centers'][2][1] = float('nan')
    else:
        design['factors'][2] = np.zeros((6, 6)).tolist()
    with pytest.raises(ValueError):
        reference.draw_mixture(design, power=5, seed=701)


def test_target_failure_is_not_discarded_or_zero_weighted():
    def log_target(q):
        values = -.5*np.sum(q*q, axis=1)
        values[0] = -np.inf
        return values
    with pytest.raises(ValueError, match='independent target'):
        reference.weighted_draws(log_target, proposal(6), power=5, seed=31)
    with pytest.raises(np.linalg.LinAlgError):
        reference.fit_weighted_proposal(np.ones((20, 6)), np.ones(20))
