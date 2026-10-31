"""Independent diagnostic importance proposal; never used by a sampler.

The equal mixture always includes the source-coordinate N(0,I) prior. Proposal
learning is discarded and held separate from frozen scrambled-QMC estimation.
NumPy/SciPy are intentionally confined to this independent reference.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc, t

from .modern_structural import log_posterior, quantities


def component_log_densities(q, centers, factors):
    values = []
    for center, factor in zip(centers, factors):
        residual = np.linalg.solve(factor, (q-center).T).T
        sign, logdet = np.linalg.slogdet(factor)
        if sign == 0 or not np.isfinite(logdet):
            raise ValueError('singular proposal factor')
        values.append(-.5*np.sum(residual**2+math.log(2*math.pi), axis=1)-logdet)
    return np.stack(values, axis=1)


def mixture_log_density(q, centers, factors):
    return logsumexp(component_log_densities(q, centers, factors), axis=1)-math.log(len(centers))


def draw_mixture(proposal, power, seed):
    centers, factors = np.array(proposal['centers']), np.array(proposal['factors'])
    if centers.shape != (8,5) or factors.shape != (8,5,5):
        raise ValueError('eight five-dimensional Gaussian components required')
    if not np.array_equal(centers[0], np.zeros(5)) or not np.array_equal(factors[0], np.eye(5)):
        raise ValueError('the prior component must remain fixed')
    if type(power) is not int or not 3 <= power <= 30:
        raise ValueError('equal component stratification needs Sobol power >=3')
    draws = []
    for index, (center, factor) in enumerate(zip(centers, factors)):
        u = qmc.Sobol(5, scramble=True, seed=seed+index, bits=32).random_base2(power-3)
        if np.any((u <= 0) | (u >= 1)):
            raise ValueError('Sobol endpoint requires a recorded new scramble')
        draws.append(center+ndtri(u)@factor.T)
    return np.concatenate(draws)


def weighted_draws(observations, proposal, *, power, seed):
    q = draw_mixture(proposal, power, seed)
    logweight = np.concatenate([log_posterior(q[i:i+1024], observations)
                               for i in range(0,len(q),1024)])
    logweight -= mixture_log_density(q, proposal['centers'], proposal['factors'])
    normalizer = logsumexp(logweight)
    weights = np.exp(logweight-normalizer)
    if not np.isfinite(weights).all() or not np.isfinite(normalizer):
        raise ValueError('nonfinite independent integration weights')
    receipt = dict(points=len(q), seed=seed, weight_ess=float(1/np.sum(weights**2)),
                   max_weight=float(weights.max()), log_normalizer=float(normalizer-math.log(len(q))),
                   component_posterior_weights=weights.reshape(8,-1).sum(axis=1).tolist())
    return q, weights, receipt


def fit_weighted_proposal(q, weights):
    """Regularized weighted EM for seven equal components, plus the fixed prior.

    The .1 global covariance contribution regularizes only the proposal. A
    failed positive-definiteness check rejects the reference instead of changing
    the filter likelihood or the importance weights.
    """
    q, weights = np.asarray(q), np.asarray(weights)
    if q.ndim != 2 or q.shape[1] != 5 or weights.shape != (len(q),):
        raise ValueError('weighted five-dimensional points required')
    if not np.isfinite(q).all() or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('invalid fitting weights')
    weights = weights/weights.sum()
    mean = weights@q
    residual = q-mean
    covariance = (residual.T*weights)@residual
    np.linalg.cholesky(covariance)
    _, directions = np.linalg.eigh(covariance)
    order = np.argsort(residual@directions[:,-1])
    ordered_cdf = np.cumsum(weights[order])
    selected = order[np.searchsorted(ordered_cdf, (np.arange(7)+.5)/7)]
    centers = q[selected].copy()
    factors = np.repeat(np.linalg.cholesky(covariance)[None,:,:], 7, axis=0)
    objectives = []
    for _ in range(12):
        logdensities = component_log_densities(q, centers, factors)
        logsum = logsumexp(logdensities, axis=1)
        objectives.append(float(weights@(logsum-math.log(7))))
        responsibility = np.exp(logdensities-logsum[:,None])*weights[:,None]
        for component in range(7):
            mass = responsibility[:,component].sum()
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError('empty proposal component')
            local_weights = responsibility[:,component]/mass
            centers[component] = local_weights@q
            centered = q-centers[component]
            local_covariance = (centered.T*local_weights)@centered
            factors[component] = np.linalg.cholesky(.9*local_covariance+.1*covariance)
    return dict(centers=np.concatenate((np.zeros((1,5)),centers)).tolist(),
                factors=np.concatenate((np.eye(5)[None,:,:],factors)).tolist(),
                weights=[1/8]*8, covariance_shrinkage=.1, em_updates=12,
                weighted_proposal_log_density=objectives,
                evidence_role='independent_reference_proposal_only')


def train_proposal(observations, center, factor, *, on_round=None):
    proposal = dict(centers=[np.zeros(5).tolist()]+[np.asarray(center).tolist()]*7,
                    factors=[np.eye(5).tolist()]+[(2*np.asarray(factor)).tolist()]*7,
                    weights=[1/8]*8)
    rows = []
    for iteration in range(2):
        q, weights, receipt = weighted_draws(observations, proposal, power=15,
                                            seed=20261108+10000*iteration)
        proposal = fit_weighted_proposal(q, weights)
        receipt.update(round=iteration, proposal=proposal)
        rows.append(receipt)
        if on_round is not None:
            on_round(receipt)
    return proposal, rows


def reference_rung(observations, proposal, *, power, replications=8, seed=30361008,
                   on_replication=None):
    if type(replications) is not int or replications < 2:
        raise ValueError('at least two scrambles required')
    rows = []
    for replication in range(replications):
        q, weights, receipt = weighted_draws(observations, proposal, power=power,
                                            seed=seed+8*replication)
        values = quantities(q)
        mean = weights@values
        sd = np.sqrt(weights@((values-mean)**2))
        receipt.update(replication=replication, mean=mean.tolist(), sd=sd.tolist())
        rows.append(receipt)
        if on_replication is not None:
            on_replication(receipt)
    means = np.array([row['mean'] for row in rows])
    se = means.std(axis=0,ddof=1)/math.sqrt(replications)
    return dict(power=power, points_per_replication=2**power, replications=replications,
                rows=rows, mean=means.mean(axis=0).tolist(),
                sd=np.sqrt(np.mean(np.array([row['sd'] for row in rows])**2,axis=0)).tolist(),
                randomization_mcse=se.tolist(), pointwise_half_width=(t.ppf(.975,replications-1)*se).tolist(),
                numerical_target='full_five_probit_principal_root_UKF_posterior',
                rigorous_integration_error_bound=False, exact_nonlinear_posterior=False)
