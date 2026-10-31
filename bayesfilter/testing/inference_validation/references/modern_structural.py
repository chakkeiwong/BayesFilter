"""Independent NumPy/SciPy diagnostic of the full five-parameter structural UKF.

Never import this module from a sampler, tuning or preparation execution path.
The principal-root sigma-point likelihood is the numerical posterior target,
not the exact nonlinear-generative likelihood.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.special import expit, logsumexp, ndtr, ndtri
from scipy.stats import qmc, t

LOWER = np.array([.05, .05, .05, .02, .02])
WIDTH = np.array([.98, 1.25, .98, 1., 1.]) - LOWER
QUANTITY_NAMES = tuple(f'{prefix}_{name}' for prefix in ('physical', 'source_square', 'source_tail')
                       for name in ('rho', 'sigma', 'phi', 'gamma', 'R'))


def principal_root_2d(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[-2:] != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError('finite two-dimensional covariance required')
    determinant = matrix[..., 0, 0]*matrix[..., 1, 1] - matrix[..., 0, 1]*matrix[..., 1, 0]
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    if np.any(determinant <= 0) or np.any(trace <= 0):
        raise ValueError('nonpositive covariance in independent reference')
    rootdet = np.sqrt(determinant)
    return (matrix + rootdet[..., None, None]*np.eye(2)) / np.sqrt(trace+2*rootdet)[..., None, None]


def physical(q):
    return LOWER + WIDTH*ndtr(q)


def quantities(q):
    q = np.asarray(q)
    return np.concatenate((physical(q), q*q, expit(2*(q-1))), axis=-1)


def log_likelihood(q, observations):
    q = np.asarray(q, dtype=float)
    y = np.asarray(observations, dtype=float).reshape(-1)
    if q.ndim != 2 or q.shape[1] != 5 or not np.isfinite(q).all() or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError('finite [batch,5] source coordinates and full observations required')
    batch = len(q)
    rho, sigma, phi, gamma, noise = physical(q).T
    variance = .13+noise
    cross = np.array([.04, .09])
    mean = cross[None, :]*y[0]/variance[:, None]
    covariance = np.diag(cross)[None, :, :] - np.outer(cross, cross)[None, :, :]/variance[:, None, None]
    total = -.5*(np.log(2*math.pi*variance) + y[0]**2/variance)
    offsets = np.concatenate((np.zeros((1,3)), np.sqrt(3)*np.eye(3), -np.sqrt(3)*np.eye(3)))
    mean_weights = np.array([0.] + [1/6]*6)
    covariance_weights = mean_weights.copy(); covariance_weights[0]=2.
    for observation in y[1:]:
        factor = np.zeros((batch,3,3))
        factor[:,:2,:2] = principal_root_2d(covariance)
        factor[:,2,2] = 1.
        points = np.einsum('ra,bda->brd',offsets,factor)
        points[:,:,:2] += mean[:,None,:]
        m = rho[:,None]*points[:,:,0] + sigma[:,None]*points[:,:,2]
        k = phi[:,None]*points[:,:,1] + gamma[:,None]*m*m
        predicted = np.stack((m,k),-1)
        predicted_mean = np.einsum('r,brd->bd',mean_weights,predicted)
        centered = predicted-predicted_mean[:,None,:]
        predicted_covariance = np.einsum('r,bri,brj->bij',covariance_weights,centered,centered)
        observation_centered = centered.sum(axis=-1)
        innovation_variance = np.einsum('r,br,br->b',covariance_weights,observation_centered,observation_centered)+noise
        if not np.isfinite(innovation_variance).all() or np.any(innovation_variance<=0):
            raise ValueError('invalid innovation covariance in independent reference')
        cross_covariance = np.einsum('r,bri,br->bi',covariance_weights,centered,observation_centered)
        residual = observation-predicted_mean.sum(axis=-1)
        total -= .5*(np.log(2*math.pi*innovation_variance)+residual**2/innovation_variance)
        mean = predicted_mean+cross_covariance*residual[:,None]/innovation_variance[:,None]
        covariance = predicted_covariance - np.einsum('bi,bj->bij',cross_covariance,cross_covariance)/innovation_variance[:,None,None]
        covariance = (covariance+covariance.swapaxes(-2,-1))/2
    if not np.isfinite(total).all():
        raise ValueError('independent likelihood returned nonfinite values')
    return total


def log_posterior(q, observations):
    q = np.asarray(q)
    return log_likelihood(q, observations) - .5*np.sum(q*q+math.log(2*math.pi),axis=1)


def mixture_log_density(q, center, factor):
    q = np.asarray(q)
    residual = np.linalg.solve(factor,(q-center).T).T
    prior = -.5*np.sum(q*q+math.log(2*math.pi),axis=1)
    local = -.5*np.sum(residual*residual+math.log(2*math.pi),axis=1)-np.linalg.slogdet(factor)[1]
    return logsumexp(np.stack((prior,local)),axis=0)-math.log(2)


def assess_refinement(previous, current, *, tolerance=.002):
    """Empirical refinement only; neither exact error bounds nor simultaneous CIs."""
    if previous is None:
        return dict(status='reference_limited', reason='two_orders_required')
    if current['power'] <= previous['power'] or not 0 < tolerance < 1:
        raise ValueError('increasing orders and a positive relative tolerance required')
    mean = np.asarray(current['mean'])
    old_mean = np.asarray(previous['mean'])
    sd = np.asarray(current['sd'])
    half_width = np.asarray(current['pointwise_half_width'])
    if any(value.shape != (15,) for value in (mean, old_mean, sd, half_width)):
        raise ValueError('all declared quantities must be present')
    if not all(np.isfinite(value).all() for value in (mean, old_mean, sd, half_width)) or np.any(sd <= 0) or np.any(half_width < 0):
        raise ValueError('invalid reference uncertainty')
    change = np.abs(mean-old_mean)/sd
    uncertainty = half_width/sd
    passed = bool(np.all(change <= tolerance) and np.all(uncertainty <= tolerance))
    return dict(status='empirically_refined' if passed else 'reference_limited',
        relative_tolerance=tolerance, mean_change_in_sd=change.tolist(),
        pointwise_half_width_in_sd=uncertainty.tolist(),
        quantity_names=QUANTITY_NAMES, rigorous_integration_error_bound=False,
        simultaneous_confidence_guarantee=False)


def reference_rung(observations, center, factor, *, power, replications=8, seed=20261008,
                   on_replication=None):
    if type(power) is not int or not 2 <= power <= 30 or type(replications) is not int or replications < 2:
        raise ValueError('Sobol power in [2,30] and at least two independent scrambles required')
    center=np.asarray(center); factor=np.asarray(factor)*2
    if center.shape != (5,) or factor.shape != (5,5) or not np.isfinite(center).all() or not np.isfinite(factor).all() or np.linalg.slogdet(factor)[0] == 0:
        raise ValueError('finite five-coordinate center and nonsingular factor required')
    rows=[]
    for replication in range(replications):
        components=[]
        for component in range(2):
            uniform=qmc.Sobol(5,scramble=True,seed=seed+2*replication+component,bits=32).random_base2(power-1)
            if np.any((uniform<=0)|(uniform>=1)):
                raise ValueError('Sobol endpoint needs a separately recorded scramble')
            normal=ndtri(uniform)
            components.append(normal if component==0 else center+normal@factor.T)
        q=np.concatenate(components)
        values=[]
        for first in range(0,len(q),1024):
            values.append(log_posterior(q[first:first+1024],observations))
        logweight=np.concatenate(values)-mixture_log_density(q,center,factor)
        logsum=logsumexp(logweight)
        weights=np.exp(logweight-logsum)
        estimates=quantities(q)
        mean=weights@estimates
        variance=weights@((estimates-mean)**2)
        rows.append(dict(replication=replication,mean=mean.tolist(),sd=np.sqrt(variance).tolist(),
            log_normalizer=float(logsum-math.log(len(q))),
            weight_ess=float(1/np.sum(weights*weights)),max_weight=float(weights.max()),
            prior_component_posterior_weight=float(weights[:len(q)//2].sum())))
        if on_replication is not None:
            on_replication(rows[-1])
    means=np.array([row['mean'] for row in rows]);sds=np.array([row['sd'] for row in rows])
    se=means.std(axis=0,ddof=1)/math.sqrt(replications)
    return dict(power=power,points_per_replication=2**power,replications=replications,rows=rows,
        mean=means.mean(axis=0).tolist(),sd=np.sqrt(np.mean(sds*sds,axis=0)).tolist(),
        randomization_mcse=se.tolist(),pointwise_half_width=(t.ppf(.975,replications-1)*se).tolist(),
        numerical_target='full_five_probit_principal_root_UKF_posterior',
        rigorous_integration_error_bound=False,exact_nonlinear_posterior=False)
