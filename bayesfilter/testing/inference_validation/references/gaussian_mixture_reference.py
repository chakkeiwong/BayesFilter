"""Independent diagnostic importance integration with a frozen Gaussian mixture.

This NumPy/SciPy reference never supplies a sampler or tuning decision. Equal
component stratification is weighted by the *full* mixture density. Proposal
training and final integration use separate caller-declared random streams.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import qmc, t

from .modern_structural_mixture import component_log_densities, mixture_log_density


def validate_proposal(proposal):
    centers = np.asarray(proposal['centers'], dtype=float)
    factors = np.asarray(proposal['factors'], dtype=float)
    if centers.ndim != 2 or centers.shape[0] != 8 or centers.shape[1] < 1:
        raise ValueError('eight nonempty Gaussian components required')
    dimension = centers.shape[1]
    if factors.shape != (8, dimension, dimension):
        raise ValueError('proposal dimension mismatch')
    if not np.isfinite(centers).all() or not np.isfinite(factors).all():
        raise ValueError('nonfinite proposal')
    if not np.array_equal(np.asarray(proposal.get('weights', [1/8]*8)), np.full(8, 1/8)):
        raise ValueError('this stratification requires eight equal weights')
    if (not np.array_equal(centers[0], np.zeros(dimension))
            or not np.array_equal(factors[0], np.eye(dimension))):
        raise ValueError('the fixed standard-normal prior component is required')
    signs, logdets = np.linalg.slogdet(factors)
    if np.any(signs == 0) or not np.isfinite(logdets).all():
        raise ValueError('singular proposal factor')
    return centers, factors


def draw_mixture(proposal, *, power, seed):
    centers, factors = validate_proposal(proposal)
    if type(power) is not int or not 3 <= power <= 30:
        raise ValueError('equal component stratification requires power >= 3')
    dimension = centers.shape[1]
    draws = []
    for index, (center, factor) in enumerate(zip(centers, factors)):
        uniform = qmc.Sobol(dimension, scramble=True, seed=seed+index,
                           bits=32).random_base2(power-3)
        if np.any((uniform <= 0) | (uniform >= 1)):
            raise ValueError('Sobol endpoint requires a recorded new scramble')
        draws.append(center+ndtri(uniform)@factor.T)
    return np.concatenate(draws)


def weighted_draws(log_target, proposal, *, power, seed):
    q = draw_mixture(proposal, power=power, seed=seed)
    values = np.concatenate([np.asarray(log_target(q[first:first+1024]))
                             for first in range(0, len(q), 1024)])
    if values.shape != (len(q),) or not np.isfinite(values).all():
        raise ValueError('nonfinite or incorrectly shaped independent target')
    logweight = values-mixture_log_density(q, proposal['centers'], proposal['factors'])
    normalizer = logsumexp(logweight)
    weights = np.exp(logweight-normalizer)
    if not np.isfinite(normalizer) or not np.isfinite(weights).all():
        raise ValueError('nonfinite importance weights')
    return q, weights, dict(
        points=len(q), seed=seed, component_seeds=list(range(seed, seed+8)),
        weight_ess=float(1/np.sum(weights**2)), max_weight=float(weights.max()),
        log_normalizer=float(normalizer-math.log(len(q))),
        component_posterior_weights=weights.reshape(8, -1).sum(axis=1).tolist())


def fit_weighted_proposal(q, weights):
    """Fit seven equal Gaussian components; keep an eighth fixed prior component.

    Twelve EM updates and 10% covariance shrinkage are explicit proposal
    hypotheses of the N5 reference experiment, not likelihood regularization.
    """
    q, weights = np.asarray(q, dtype=float), np.asarray(weights, dtype=float)
    if q.ndim != 2 or q.shape[1] < 1 or weights.shape != (len(q),):
        raise ValueError('weighted multidimensional points required')
    if (not np.isfinite(q).all() or not np.isfinite(weights).all()
            or np.any(weights < 0) or not 0 < weights.sum() < math.inf):
        raise ValueError('invalid fitting weights')
    weights = weights/weights.sum()
    dimension = q.shape[1]
    mean = weights@q
    residual = q-mean
    covariance = (residual.T*weights)@residual
    global_factor = np.linalg.cholesky(covariance)
    _, directions = np.linalg.eigh(covariance)
    order = np.argsort(residual@directions[:, -1])
    cdf = np.cumsum(weights[order])
    selected = order[np.searchsorted(cdf, (np.arange(7)+.5)/7)]
    centers = q[selected].copy()
    factors = np.repeat(global_factor[None, :, :], 7, axis=0)
    objectives = []
    for _ in range(12):
        densities = component_log_densities(q, centers, factors)
        total = logsumexp(densities, axis=1)
        objectives.append(float(weights@(total-math.log(7))))
        responsibilities = np.exp(densities-total[:, None])*weights[:, None]
        for index in range(7):
            mass = responsibilities[:, index].sum()
            if not np.isfinite(mass) or mass <= 0:
                raise ValueError('empty proposal component')
            component_weights = responsibilities[:, index]/mass
            centers[index] = component_weights@q
            centered = q-centers[index]
            local_covariance = (centered.T*component_weights)@centered
            factors[index] = np.linalg.cholesky(.9*local_covariance+.1*covariance)
    proposal = dict(
        centers=np.concatenate((np.zeros((1, dimension)), centers)).tolist(),
        factors=np.concatenate((np.eye(dimension)[None, :, :], factors)).tolist(),
        weights=[1/8]*8, covariance_shrinkage=.1, em_updates=12,
        weighted_proposal_log_density=objectives,
        evidence_role='independent_reference_proposal_only')
    validate_proposal(proposal)
    return proposal


def train_proposal(log_target, center, factor, *, seed, on_round=None):
    center, factor = np.asarray(center), np.asarray(factor)
    if center.ndim != 1 or factor.shape != (len(center), len(center)):
        raise ValueError('local center/factor dimensions differ')
    proposal = dict(centers=[np.zeros(len(center)).tolist()]+[center.tolist()]*7,
                    factors=[np.eye(len(center)).tolist()]+[(2*factor).tolist()]*7,
                    weights=[1/8]*8)
    rows = []
    for iteration in range(2):
        q, weights, receipt = weighted_draws(log_target, proposal, power=15,
                                            seed=seed+10000*iteration)
        proposal = fit_weighted_proposal(q, weights)
        receipt.update(round=iteration, proposal=proposal)
        rows.append(receipt)
        if on_round is not None:
            on_round(receipt)
    return proposal, rows


def reference_rung(log_target, quantities, proposal, *, power, seed,
                   replications=8, on_replication=None):
    if type(replications) is not int or replications < 2:
        raise ValueError('at least two independent scrambles required')
    rows = []
    for replication in range(replications):
        q, weights, receipt = weighted_draws(log_target, proposal, power=power,
                                            seed=seed+8*replication)
        values = np.asarray(quantities(q))
        if values.ndim != 2 or values.shape[0] != len(q) or not np.isfinite(values).all():
            raise ValueError('nonfinite or incorrectly shaped reference quantities')
        mean = weights@values
        receipt.update(replication=replication, mean=mean.tolist(),
                       sd=np.sqrt(weights@((values-mean)**2)).tolist())
        rows.append(receipt)
        if on_replication is not None:
            on_replication(receipt)
    means = np.array([row['mean'] for row in rows])
    se = means.std(axis=0, ddof=1)/math.sqrt(replications)
    return dict(
        power=power, points_per_replication=2**power, replications=replications,
        rows=rows, mean=means.mean(axis=0).tolist(),
        sd=np.sqrt(np.mean(np.array([row['sd'] for row in rows])**2, axis=0)).tolist(),
        randomization_mcse=se.tolist(), pointwise_half_width=(t.ppf(.975, replications-1)*se).tolist(),
        rigorous_integration_error_bound=False, exact_nonlinear_posterior=False)
