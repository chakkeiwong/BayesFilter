"""Independent diagnostic SSM oracles; never imported by inference targets.

NumPy/SciPy is used only for reference filtering, integration and post-run
assessment. Model metadata is shared; no TensorFlow filtering code is reused.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
import numpy as np
from scipy import linalg, special

from ..ssm_campaign_profiles import get_profile


def model_coordinates(target, q):
    p = get_profile(target)
    q = np.asarray(q)
    if p.family == "persistence_noise":
        return np.stack((p.persistence_cap*np.tanh(q[..., 0]), np.exp(q[..., 1])), -1)
    if p.family == "two_noises":
        return np.exp(q)
    return q


def scalar_parameters(target, q):
    p = get_profile(target)
    q = np.asarray(q, dtype=float)
    shape = q.shape[:-1]
    rho = np.full(shape, p.rho)
    process = np.full(shape, p.process_sd**2)
    obs = np.full(shape, p.observation_sd**2)
    location = np.zeros(shape)
    if p.family == "location":
        location = q[..., 0]
    elif p.family == "persistence_noise":
        rho, obs = p.persistence_cap*np.tanh(q[..., 0]), np.exp(2*q[..., 1])
    elif p.family == "two_noises":
        process, obs = np.exp(2*q[..., 0]), np.exp(2*q[..., 1])
    return rho, process, obs, location


def scalar_likelihood(target, q, data):
    rho, process, obs, location = scalar_parameters(target, q)
    mean = np.zeros_like(rho)
    covariance = process/(1-rho**2)
    total = np.zeros_like(rho)
    for y in data:
        mean = rho*mean
        covariance = rho*rho*covariance+process
        innovation = y-location-mean
        variance = covariance+obs
        total -= 0.5*(np.log(2*math.pi*variance)+innovation**2/variance)
        gain = covariance/variance
        mean += gain*innovation
        covariance *= 1-gain
    return total


def dense_scalar_likelihood(target, q, data):
    """Observation-marginal Gaussian; independent check on innovation timing."""
    rho, process, obs, location = scalar_parameters(target, np.asarray(q))
    n = len(data)
    lags = np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    covariance = process/(1-rho**2)*rho**lags + np.eye(n)*obs
    residual = np.asarray(data)-location
    factor = linalg.cholesky(covariance, lower=True)
    standardized = linalg.solve_triangular(factor, residual, lower=True)
    return -.5*(standardized@standardized+n*math.log(2*math.pi))-np.log(np.diag(factor)).sum()


def nonlinear_likelihood(q, data):
    """Augmented eigen UKF, alpha=1,beta=2,kappa=0, same declared approximation.

    The seven points integrate the previous 2-D Gaussian and one innovation.
    Symmetric point pairs make eigenvector signs/order irrelevant for this rule.
    """
    q = np.asarray(q, dtype=float)
    shape = q.shape[:-1]
    flat = q.reshape(-1, 2)
    b = len(flat)
    mean = np.zeros((b, 2))
    covariance = np.broadcast_to(np.diag([.25, .20]), (b, 2, 2)).copy()
    offsets = np.concatenate((np.zeros((1, 3)), np.eye(3)*np.sqrt(3), -np.eye(3)*np.sqrt(3)))
    wm = np.array([0.] + [1/6]*6)
    wc = wm.copy(); wc[0] = 2.
    total = np.zeros(b)
    for y in data:
        aug_cov = np.zeros((b, 3, 3)); aug_cov[:, :2, :2] = covariance; aug_cov[:, 2, 2] = 1.
        eigenvalues, vectors = np.linalg.eigh(aug_cov)
        if np.any(eigenvalues <= 0):
            raise ValueError("independent UKF reference encountered nonpositive covariance")
        factor = vectors*np.sqrt(eigenvalues)[:, None, :]
        points = np.einsum("ra,bda->brd", offsets, factor)
        points[:, :, :2] += mean[:, None, :]
        m = flat[:, :1]*points[:, :, 0] + .25*points[:, :, 2]
        k = .55*points[:, :, 1] + flat[:, 1:]*np.tanh(m)
        predicted = np.stack((m, k), -1)
        predicted_mean = np.einsum("r,brn->bn", wm, predicted)
        centered = predicted-predicted_mean[:, None, :]
        pcov = np.einsum("r,bri,brj->bij", wc, centered, centered)
        observed = predicted.sum(-1)
        obs_mean = observed@wm
        deviations = observed-obs_mean[:, None]
        variance = np.sum(wc*deviations**2, axis=1)+.30**2
        cross = np.einsum("r,bri,br->bi", wc, centered, deviations)
        gain = cross/variance[:, None]
        residual = y-obs_mean
        total -= .5*(np.log(2*math.pi*variance)+residual**2/variance)
        mean = predicted_mean+gain*residual[:, None]
        covariance = pcov-np.einsum("bi,bj,b->bij", gain, gain, variance)
        covariance = (covariance+covariance.swapaxes(-1, -2))/2
    return total.reshape(shape)


@lru_cache(maxsize=1)
def multivariate_definition():
    root = Path(__file__).resolve().parents[4]
    config = json.loads((root/"docs/benchmarks/configs/multidim_lgssm_full_estimation_rerun_2026_07_13.json").read_text())
    contract = json.loads((root/config["source_contract"]["path"]).read_text())
    fixture = json.loads((root/"docs/benchmarks/artifacts/multidim_lgssm_full_estimation_rerun_2026_07_13/fixture_T120_seed20260709_301.json").read_text())
    return contract, fixture


def multivariate_log_density(q, data=None):
    contract, fixture = multivariate_definition()
    data = np.asarray(fixture["observations"] if data is None else data)
    center = np.asarray(fixture["raw_truth"])
    scale = np.array([.5]*4+[.6]*6+[.35]*8)
    q = np.asarray(q); results = []
    for row in q.reshape(-1, 18):
        transition = np.diag(contract["transform"]["rho_max"]*np.tanh(row[:4]))
        transition[np.tril_indices(4, -1)] = contract["transform"]["lower_scale"]*np.tanh(row[4:10])
        process, observation = np.diag(np.exp(2*row[10:14])), np.diag(np.exp(2*row[14:18]))
        mean = np.zeros(4)
        covariance = linalg.solve_discrete_lyapunov(transition, process)
        total = -.5*np.sum(((row-center)/scale)**2)
        for y in data:
            mean = transition@mean
            covariance = transition@covariance@transition.T+process
            innovation = y-mean
            var = covariance+observation
            factor = linalg.cho_factor(var, lower=True)
            solved = linalg.cho_solve(factor, innovation)
            total -= .5*(4*math.log(2*math.pi)+2*np.log(np.diag(factor[0])).sum()+innovation@solved)
            gain = linalg.cho_solve(factor, covariance).T
            mean += gain@innovation
            covariance -= gain@covariance
            covariance = (covariance+covariance.T)/2
        results.append(total)
    return np.array(results).reshape(q.shape[:-1])


def log_density(target, q, data):
    p = get_profile(target)
    q = np.asarray(q, dtype=float)
    if p.family == "multivariate":
        return multivariate_log_density(q, data)
    if data is None or len(data) != p.horizon:
        raise ValueError("reference requires the frozen campaign dataset")
    prior = -.5*np.sum(((q-np.array(p.prior_mean))/np.array(p.prior_scale))**2, -1)
    return prior + (nonlinear_likelihood(q, data) if p.family == "nonlinear" else scalar_likelihood(target, q, data))


def location_posterior(data):
    target = "ssm_campaign_location"
    p = get_profile(target)
    rho, process, obs = p.rho, p.process_sd**2, p.observation_sd**2
    n = len(data); lag = np.abs(np.arange(n)[:, None]-np.arange(n)[None, :])
    covariance = process/(1-rho**2)*rho**lag + obs*np.eye(n)
    precision_one = linalg.solve(covariance, np.ones(n), assume_a="pos")
    variance = 1/(1/p.prior_scale[0]**2+precision_one.sum())
    return variance*(precision_one@np.asarray(data)), variance


def _grid(target, data, radius, resolution):
    p = get_profile(target)
    center, scale = np.array(p.prior_mean), np.array(p.prior_scale)
    axes = [np.linspace(m-radius*s, m+radius*s, resolution) for m,s in zip(center, scale)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    density = log_density(target, grid, data)
    trapezoid = np.ones(resolution); trapezoid[[0,-1]] = .5
    log_weight = density+np.log(trapezoid)[:, None]+np.log(trapezoid)[None, :]
    weight = np.exp(log_weight-special.logsumexp(log_weight))
    model = model_coordinates(target, grid)
    mean = np.sum(weight[..., None]*model, axis=(0,1))
    bounded = np.sum(weight[..., None]*np.arctan(model), axis=(0,1))
    medians = []
    for i in range(2):
        marginal = weight.sum(axis=1-i)
        # Cell-centered interpolation reduces grid quantile discretization bias.
        raw_median = np.interp(.5, np.cumsum(marginal)-.5*marginal, axes[i])
        raw = center.copy(); raw[i] = raw_median
        medians.append(model_coordinates(target, raw)[i])
    edge = float(weight[[0,-1], :].sum()+weight[1:-1, [0,-1]].sum())
    return grid, weight, np.concatenate((mean, medians, bounded)), edge


@lru_cache(maxsize=12)
def _checked_grid(target, data, resolution, sensitivity):
    if resolution < 41 or resolution % 2 != 1:
        raise ValueError("reference resolution must be odd and >=41")
    coarse = (resolution+1)//2
    if coarse % 2 == 0: coarse += 1
    grid, weights, summary, edge = _grid(target, data, 6., resolution)
    _, _, coarse_summary, _ = _grid(target, data, 6., coarse)
    expanded_resolution = 2*round((resolution-1)*4/3/2)+1
    _, _, expanded_summary, expanded_edge = _grid(target, data, 8., expanded_resolution)
    delta = np.maximum(abs(summary-coarse_summary), abs(summary-expanded_summary))
    limits = np.array([*sensitivity, *sensitivity, .001, .001])
    valid = bool(np.all(delta <= limits) and max(edge, expanded_edge) <= 1.e-5)
    metadata = {"method":"independent_same_target_trapezoidal_grid", "checked":valid,
        "resolution":resolution, "coarse_resolution":coarse,
        "expanded_resolution":expanded_resolution, "prior_sd_radius":6., "expanded_radius":8.,
        "summary_order":["mean0", "mean1", "median0", "median1", "atan0", "atan1"],
        "summary":summary.tolist(), "sensitivity":delta.tolist(), "sensitivity_limits":limits.tolist(),
        "edge_mass":edge, "expanded_edge_mass":expanded_edge, "edge_limit":1.e-5,
        "integration_error_bound":False, "accuracy_established":False}
    return grid.reshape(-1,2), weights.ravel(), metadata


def posterior_reference(target, count, seed, data, *, resolution=161, sensitivity=None):
    p = get_profile(target)
    if p.reference == "unavailable":
        return None, {"checked":False, "reason":"no joint posterior reference for K6"}
    if p.reference == "analytic":
        mean, variance = location_posterior(data)
        raw = np.random.default_rng(seed).normal(mean, np.sqrt(variance), (count,1))
        return raw, {"checked":True, "method":"independent_dense_gaussian_conjugacy",
                     "mean":mean, "variance":variance, "integration_error_bound":True}
    sensitivity = tuple(sensitivity or (.002, .002))
    grid, weights, metadata = _checked_grid(target, tuple(data), resolution, sensitivity)
    if not metadata["checked"]:
        return None, metadata
    rows = np.random.default_rng(seed).choice(len(weights), count, p=weights)
    return model_coordinates(target, grid[rows]), metadata


def nonlinear_latent_reference(q, data, order):
    """Tiny true-model likelihood via Gauss-Hermite over initial state/noise.

    This is separate approximation-error evidence, never a sampler reference
    for the sigma-point target. Only up to two observations are budgeted.
    """
    if not 1 <= len(data) <= 2 or not 3 <= order <= 21:
        raise ValueError("latent reference requires T<=2 and order in [3,21]")
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    dimension = len(data)+2
    points = np.stack(np.meshgrid(*([nodes*np.sqrt(2)]*dimension), indexing="ij"), -1).reshape(-1, dimension)
    logweights = sum(x.ravel() for x in np.meshgrid(*([np.log(weights/np.sqrt(math.pi))]*dimension), indexing="ij"))
    m, k = .5*points[:, 0], np.sqrt(.2)*points[:, 1]
    loglike = np.zeros(len(points))
    rho, beta = q
    for i, observation in enumerate(data):
        m = rho*m+.25*points[:, i+2]
        k = .55*k+beta*np.tanh(m)
        loglike -= .5*(math.log(2*math.pi*.09)+(observation-m-k)**2/.09)
    return float(special.logsumexp(logweights+loglike))


def sanity_comparators(target, data):
    """Prior and local-Gaussian diagnostics, never inputs to tuning or starts."""
    from scipy import optimize
    p = get_profile(target)
    if p.reference == "unavailable":
        return {"status":"not_evaluated", "reason":"K6 has no checked joint posterior oracle"}
    center, scale = np.array(p.prior_mean), np.array(p.prior_scale)
    rng = np.random.default_rng(20260926)
    prior = model_coordinates(target, rng.normal(size=(16384,len(center)))*scale+center)
    result = {"role":"descriptive comparators only; no tuning information",
              "prior_mean":prior.mean(0).tolist(), "prior_median":np.median(prior,axis=0).tolist()}
    fit = optimize.minimize(lambda q:-float(log_density(target,q,data)), center, method="BFGS")
    mode = fit.x
    h = 1e-4
    eye = np.eye(len(mode))*h
    f = lambda q:-float(log_density(target,q,data))
    hessian = np.array([[(f(mode+a+b)-f(mode+a-b)-f(mode-a+b)+f(mode-a-b))/(4*h*h)
                        for a in eye] for b in eye])
    eigenvalues = np.linalg.eigvalsh(hessian)
    result.update(local_mode_raw=mode.tolist(), optimizer_success=bool(fit.success),
                  optimizer_message=str(fit.message), hessian_eigenvalues=eigenvalues.tolist(),
                  hessian_fd_step=h, gaussian_draw_count=16384)
    if fit.success and np.all(eigenvalues > 0):
        draws = rng.multivariate_normal(mode,np.linalg.inv(hessian),size=16384)
        model = model_coordinates(target,draws)
        result.update(laplace_mean=model.mean(0).tolist(),laplace_median=np.median(model,axis=0).tolist(),
                      laplace_status="local_positive_hessian_approximation")
    else:
        result["laplace_status"] = "unavailable_optimizer_or_hessian"
    if p.reference == "grid":
        raw, weights, _, _ = _grid(target,data,6.,161)
        result["coarse_posterior_grid_mode_raw"] = raw.reshape(-1,2)[weights.argmax()].tolist()
    return result
