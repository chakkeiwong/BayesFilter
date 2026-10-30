"""Independent NumPy/SciPy diagnostic for the frozen Phase9 OU posterior.

Never imported by a sampler, tuner, runtime target or admission implementation.
The chart follows MacroFinance's Phase9 source; matrix exponentiation, direct
Lyapunov solution and covariance Kalman recursion independently check its
factor-QR value program. Integration coordinates are z=theta/0.35, not theta.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.linalg import expm
from scipy.special import expit

PRIOR_SCALE = .35
DELTA = 1. / 12.
NYQUIST_FREQUENCY = (1. - 1.e-3) * math.pi / DELTA
R_LOG_CONDITION_LIMIT = math.log(50.) - 1.e-6
STATE_MEAN = np.array([.35, -.22])
OBSERVATION_OFFSET = np.array([.12, -.07])
PARAMETER_NAMES = (
    "S_log_diag_G", "S_log_diag_Pi", "S_lower_Pi_G", "omega_unconstrained",
    "Q_log_cholesky_G", "Q_lower_Pi_G", "Q_log_cholesky_Pi",
    "R_log_sd_mean", "R_log_sd_contrast",
)
QUANTITY_NAMES = (*PARAMETER_NAMES,
    *("prior_square_" + name for name in PARAMETER_NAMES),
    *("smooth_upper_tail_" + name for name in PARAMETER_NAMES),
    "state_sd_G", "state_sd_Pi", "acf1_G", "acf1_Pi")


def _points(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 9 or not len(value) or not np.isfinite(value).all():
        raise ValueError("finite nonempty [batch,9] chart coordinates required")
    return value


def _observations(value):
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError("finite nonempty [time,2] observations required")
    return value


def decode_chart(theta):
    theta = _points(theta)
    batch = len(theta)
    s = np.zeros((batch, 2, 2))
    s[:, 0, 0], s[:, 1, 1], s[:, 1, 0] = np.exp(theta[:, 0]), np.exp(theta[:, 1]), theta[:, 2]
    k = s @ s.swapaxes(-1, -2)
    omega = NYQUIST_FREQUENCY * np.tanh(theta[:, 3])
    k[:, 0, 1] += omega
    k[:, 1, 0] -= omega
    transition = expm(-DELTA * k)
    factor = np.zeros_like(s)
    factor[:, 0, 0], factor[:, 1, 0], factor[:, 1, 1] = np.exp(theta[:, 4]), theta[:, 5], np.exp(theta[:, 6])
    innovation = factor @ factor.swapaxes(-1, -2)
    contrast = .25 * R_LOG_CONDITION_LIMIT * np.tanh(theta[:, 8])
    measurement = np.zeros_like(s)
    measurement[:, 0, 0] = np.exp(2. * (theta[:, 7] + contrast))
    measurement[:, 1, 1] = np.exp(2. * (theta[:, 7] - contrast))
    operator = np.eye(4)[None] - np.einsum("bik,bjl->bijkl", transition, transition).reshape(batch, 4, 4)
    stationary = np.linalg.solve(operator, innovation.reshape(batch, 4, 1)).reshape(batch, 2, 2)
    stationary = .5 * (stationary + stationary.swapaxes(-1, -2))
    arrays = transition, innovation, measurement, stationary
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("nonfinite independent chart")
    if np.any(np.linalg.eigvalsh(stationary) <= 0):
        raise ValueError("nonpositive independent stationary covariance")
    residual = stationary - transition @ stationary @ transition.swapaxes(-1, -2) - innovation
    scale = np.maximum(np.max(np.abs(stationary), axis=(1, 2)) + np.max(np.abs(innovation), axis=(1, 2)), np.finfo(float).tiny)
    if np.any(np.max(np.abs(residual), axis=(1, 2)) / scale > 1.e-10):
        raise ValueError("independent Lyapunov residual exceeds target tolerance")
    return arrays


def covariance_log_likelihood(observations, transition, innovation, measurement,
                              stationary, *, initial_mean=None, initial_covariance=None):
    """Prediction-first covariance recursion, including all Gaussian constants."""
    y = _observations(observations)
    matrices = [np.asarray(a, dtype=np.float64) for a in (transition, innovation, measurement, stationary)]
    if (any(a.ndim != 3 or a.shape[1:] != (2, 2) for a in matrices)
            or len({a.shape for a in matrices}) != 1 or not len(matrices[0])
            or not all(np.isfinite(a).all() for a in matrices)):
        raise ValueError("matching finite [batch,2,2] model matrices required")
    transition, innovation, measurement, stationary = matrices
    batch = len(transition)
    mean = np.broadcast_to(STATE_MEAN if initial_mean is None else initial_mean, (batch, 2)).copy()
    covariance = np.broadcast_to(stationary if initial_covariance is None else initial_covariance, (batch, 2, 2)).copy()
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("finite initial state required")
    offset = STATE_MEAN - (transition @ STATE_MEAN[:, None]).squeeze(-1)
    result = np.zeros(batch)
    for observation in y:
        predicted_mean = offset + (transition @ mean[..., None]).squeeze(-1)
        predicted_covariance = transition @ covariance @ transition.swapaxes(-1, -2) + innovation
        variance = predicted_covariance + measurement
        cholesky = np.linalg.cholesky(variance)
        residual = observation - OBSERVATION_OFFSET - predicted_mean
        whitened = np.linalg.solve(cholesky, residual[..., None]).squeeze(-1)
        result -= .5 * (2 * math.log(2 * math.pi) + 2 * np.log(np.diagonal(cholesky, axis1=-2, axis2=-1)).sum(-1) + (whitened**2).sum(-1))
        gain = np.linalg.solve(variance, predicted_covariance.swapaxes(-1, -2)).swapaxes(-1, -2)
        mean = predicted_mean + (gain @ residual[..., None]).squeeze(-1)
        covariance = predicted_covariance - gain @ predicted_covariance
        covariance = .5 * (covariance + covariance.swapaxes(-1, -2))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite independent covariance likelihood")
    return result


def log_likelihood(theta, observations):
    return covariance_log_likelihood(observations, *decode_chart(theta))


def log_density(theta, observations):
    """Same unnormalized chart density as the public adapter."""
    theta = _points(theta)
    return log_likelihood(theta, observations) - .5 * np.sum((theta / PRIOR_SCALE)**2, axis=1)


def log_posterior_standardized(z, observations):
    """Normalized-prior density in z; constant Jacobian handled exactly."""
    z = _points(z)
    return log_likelihood(PRIOR_SCALE * z, observations) - .5 * np.sum(z*z + math.log(2*math.pi), axis=1)


def quantities_standardized(z):
    z = _points(z)
    theta = PRIOR_SCALE * z
    transition, _, _, stationary = decode_chart(theta)
    lag = transition @ stationary
    state_sd = np.sqrt(np.diagonal(stationary, axis1=-2, axis2=-1))
    acf = np.diagonal(lag, axis1=-2, axis2=-1) / state_sd**2
    return np.concatenate((theta, z*z, expit(2*(z-1)), state_sd, acf), axis=1)
