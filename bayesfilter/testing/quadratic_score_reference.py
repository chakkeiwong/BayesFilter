"""Diagnostic local-regression score estimates; NumPy is reference-only here.

This module has no runtime/admission role. Numerical fitting cannot establish
that input log likelihoods are accurate. Correlated likelihood errors across
nearby points require independent whole-proposal replicates for uncertainty;
ordinary least-squares standard errors would give misleading precision.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class QuadraticScoreEstimate:
    intercept: float
    score: np.ndarray
    hessian: np.ndarray
    coefficients: np.ndarray
    condition_number: float
    residual_rms: float
    residual_max: float
    design_rank: int


def quadratic_design(z):
    """a+b'z+1/2*z'Cz: diagonal terms have factor 1/2, cross terms do not."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError('design points must be a finite matrix')
    _, d = z.shape
    columns = [np.ones(len(z)), *z.T]
    columns.extend(0.5*z[:,j]**2 for j in range(d))
    columns.extend(z[:,j]*z[:,k] for j in range(d) for k in range(j+1,d))
    return np.column_stack(columns)


def fit_quadratic_score(z, log_likelihood_differences, scales, radius, *, max_condition=1e8):
    """Return score in theta coordinates for theta=theta0+radius*diag(scales)*z.

    No implicit ridge, clipping, weighting, coordinate selection or fallback.
    max_condition is an engineering rejection threshold, not an accuracy claim.
    """
    z = np.asarray(z, dtype=np.float64)
    design = quadratic_design(z)
    values = np.asarray(log_likelihood_differences, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    d = z.shape[1]
    if values.shape != (len(z),) or scales.shape != (d,):
        raise ValueError('incompatible value/scale dimensions')
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(scales)) or np.any(scales<=0):
        raise ValueError('finite values and positive scales required')
    if not np.isfinite(radius) or radius<=0 or max_condition<=1:
        raise ValueError('positive radius and meaningful condition limit required')
    if design.shape[0] < design.shape[1]:
        raise ValueError('underdetermined quadratic design')
    coef, _, rank, singular = np.linalg.lstsq(design, values, rcond=None)
    condition = singular[0]/singular[-1] if singular[-1]>0 else np.inf
    if rank != design.shape[1] or not np.isfinite(condition) or condition>max_condition:
        raise ValueError('rank-deficient or ill-conditioned quadratic design')
    hz = np.zeros((d,d))
    hz[np.diag_indices(d)] = coef[1+d:1+2*d]
    offset = 1+2*d
    for j in range(d):
        for k in range(j+1,d):
            hz[j,k] = hz[k,j] = coef[offset]
            offset += 1
    steps = radius*scales
    residual = values-design@coef
    return QuadraticScoreEstimate(float(coef[0]),coef[1:1+d]/steps,
        hz/np.outer(steps,steps),coef,float(condition),
        float(np.sqrt(np.mean(residual**2))),float(np.max(np.abs(residual))),int(rank))


def log_weight_likelihood_ratios(log_weights, baseline_log_weights):
    """Common-proposal log likelihood ratios, retaining zero-weight paths.

    Rows index parameter points and columns index the same paths. A negative
    infinite log weight is a valid zero weight. NaN, positive infinity and an
    all-zero importance sample are invalid. Direct weights allow a path that
    has zero baseline weight to have positive weight at a perturbed parameter.
    """
    values = np.asarray(log_weights, dtype=np.float64)
    base = np.asarray(baseline_log_weights, dtype=np.float64)
    if values.ndim != 2 or base.shape != (values.shape[1],) or not base.size:
        raise ValueError('matching nonempty path log weights required')
    if any(np.any(np.isnan(a)) or np.any(np.isposinf(a)) for a in (values, base)):
        raise ValueError('NaN or positive infinite log weight')
    if not np.any(np.isfinite(base)) or np.any(~np.any(np.isfinite(values), axis=1)):
        raise ValueError('all-zero importance sample')
    base_shift = np.max(base)
    base_relative_log_sum = np.log(np.exp(base-base_shift).sum())
    maximum = np.max(values, axis=1, keepdims=True)
    relative = np.exp(values-maximum)
    total = relative.sum(axis=1)
    ratio = (maximum[:, 0]-base_shift) + np.log(total)-base_relative_log_sum
    ess = total**2 / np.sum(relative**2, axis=1)
    return ratio, ess


def log_likelihood_ratios(log_joint_differences, baseline_log_weights):
    """Convenience form when every joint-density difference is finite.

    For negative infinite joint densities use log_weight_likelihood_ratios
    directly, avoiding an undefined subtraction of two negative infinities.
    """
    delta = np.asarray(log_joint_differences, dtype=np.float64)
    base = np.asarray(baseline_log_weights, dtype=np.float64)
    if delta.ndim != 2 or base.shape != (delta.shape[1],) or not np.all(np.isfinite(delta)):
        raise ValueError('finite matching path log-density differences required')
    return log_weight_likelihood_ratios(delta+base[None, :], base)


def delete_group_log_ratios(log_weights, baseline_log_weights, groups):
    """Ratios after deleting each equal group of common iid path draws.

    Direct complement sums avoid log(1-mass) cancellation for dominant groups.
    Returns [parameter point, deleted group]. Undefined deletions raise; callers
    must flag unavailable jackknife uncertainty without altering the full fit.
    """
    values = np.asarray(log_weights, dtype=np.float64)
    base = np.asarray(baseline_log_weights, dtype=np.float64)
    log_weight_likelihood_ratios(values, base)  # Validate the full calculation.
    if groups < 2 or base.size % groups or groups > base.size:
        raise ValueError('at least two equal nonempty path groups required')
    group_size = base.size // groups
    results = []
    for group in range(groups):
        keep = np.ones(base.size, dtype=bool)
        keep[group*group_size:(group+1)*group_size] = False
        result, _ = log_weight_likelihood_ratios(values[:, keep], base[keep])
        results.append(result)
    return np.column_stack(results)


def equal_group_jackknife_se(deleted_estimates):
    """Delete-group jackknife standard error; does not remove numerical bias."""
    estimates = np.asarray(deleted_estimates, dtype=np.float64)
    if estimates.ndim != 2 or len(estimates) < 2 or not np.all(np.isfinite(estimates)):
        raise ValueError('finite estimates from at least two deleted groups required')
    centered = estimates - estimates.mean(axis=0)
    return np.sqrt((len(estimates)-1)/len(estimates) * np.sum(centered**2, axis=0))
