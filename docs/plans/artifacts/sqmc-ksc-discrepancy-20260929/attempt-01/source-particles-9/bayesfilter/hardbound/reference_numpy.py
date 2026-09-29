"""Independent NumPy reference implementation (diagnostic only).

Re-implements the yield maps and observation log density of
``bayesfilter.hardbound`` without importing the TF modules' internals, per
the master program's diagnostic-module allowance. Used exclusively by
tests and grid-posterior references; never on a runtime path.
"""

from __future__ import annotations

import numpy as np

MATURITIES = np.array([0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
FX_MATURITY = 1.0
LOG2PI = float(np.log(2.0 * np.pi))


def gauss_legendre_unit(order: int):
    x, w = np.polynomial.legendre.leggauss(int(order))
    return 0.5 * (x + 1.0), 0.5 * w


def dns_loadings(s: np.ndarray, decay: float) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    e = np.exp(-decay * s)
    return np.stack([np.ones_like(s), e, decay * s * e], axis=-1)


def yield_curve(factors, maturities, decay, lower_bound, alpha, bound_map,
                order=40):
    factors = np.asarray(factors, dtype=float)
    maturities = np.atleast_1d(np.asarray(maturities, dtype=float))
    nodes, weights = gauss_legendre_unit(order)
    horizons = maturities[:, None] * nodes[None, :]          # [M, K]
    loadings = dns_loadings(horizons, decay)                 # [M, K, 3]
    forwards = np.einsum("mki,...i->...mk", loadings, factors)
    if bound_map == "hard":
        bounded = np.maximum(lower_bound, forwards)
    elif bound_map == "softplus":
        z = (forwards - lower_bound) / alpha
        bounded = lower_bound + alpha * np.logaddexp(0.0, z)
    else:
        raise ValueError(bound_map)
    return np.einsum("k,...mk->...m", weights, bounded)


def observation_mean(states, target_id, fix):
    states = np.asarray(states, dtype=float)
    bound_map = {"mf_s1_k40_softplus": "softplus",
                 "mf_c1_k40_hardmax": "hard"}[target_id]
    xd, xf, xb = states[..., 0:3], states[..., 3:6], states[..., 6:8]
    yd = yield_curve(xd, MATURITIES, fix.decay_d, fix.lower_bound_d,
                     fix.alpha_d, bound_map, fix.quadrature_order)
    yf = yield_curve(xf, MATURITIES, fix.decay_f, fix.lower_bound_f,
                     fix.alpha_f, bound_map, fix.quadrature_order)
    yd_fx = yield_curve(xd, FX_MATURITY, fix.decay_d, fix.lower_bound_d,
                        fix.alpha_d, bound_map, fix.quadrature_order)[..., 0]
    yf_fx = yield_curve(xf, FX_MATURITY, fix.decay_f, fix.lower_bound_f,
                        fix.alpha_f, bound_map, fix.quadrature_order)[..., 0]
    fx = FX_MATURITY * (yd_fx - yf_fx + xb[..., 0] + xb[..., 1])
    return np.concatenate([yd, yf, fx[..., None]], axis=-1)


def noise_scales_vector(noise_scales):
    s = np.asarray(noise_scales, dtype=float)
    return np.concatenate(
        [np.repeat(s[..., 0:1], 6, -1), np.repeat(s[..., 1:2], 6, -1),
         s[..., 2:3]], axis=-1)


def observation_log_density(y, states, noise_scales, target_id, fix):
    y = np.asarray(y, dtype=float)
    mean = observation_mean(states, target_id, fix)
    scales = noise_scales_vector(noise_scales)
    z = (y - mean) / scales
    return np.sum(-0.5 * z * z - np.log(scales) - 0.5 * LOG2PI, axis=-1)
