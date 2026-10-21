"""Joint non-centered log posterior for the hard-bound model (eq. 56 chart).

Unknowns: theta (9 = 6 theta_bar components + 3 log noise scales) and raw
latents z = (x0_raw [8], eta_raw [T, 8]), all standard normal a priori.
States are reconstructed deterministically; the observation density is the
S1 or C1 map. Also provides the tiny K=1 gate model of the master program.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from bayesfilter.hardbound.dns_curve_tf import DTYPE, yield_curve
from bayesfilter.hardbound import model_tf

_LOG2PI = tf.constant(1.8378770664093453, DTYPE)

# Priors (master program Sec. 2, frozen).
PRIOR_THETA_MEAN = tf.constant(
    [0.02, -0.01, 0.005, 0.015, -0.008, 0.004], DTYPE)
PRIOR_THETA_SD = tf.constant([0.02] * 6, DTYPE)
PRIOR_LOG_NOISE_MEAN = tf.constant([-7.600902459542082] * 3, DTYPE)  # log 5e-4
PRIOR_LOG_NOISE_SD = tf.constant([0.5] * 3, DTYPE)


def _std_normal_logpdf_sum(x):
    return tf.reduce_sum(-0.5 * x * x - 0.5 * _LOG2PI)


def states_from_raws(theta_bar, x0_raw, eta_raw,
                     fix: model_tf.HardBoundFixture = model_tf.FIXTURE):
    """Deterministic state reconstruction in the non-centered chart."""
    phi = tf.constant(fix.phi_diag, DTYPE)
    q_sd = tf.sqrt(tf.constant(fix.q_diag, DTYPE))
    p0_sd = tf.sqrt(tf.constant(fix.p0_diag, DTYPE))
    theta_ext = tf.concat([theta_bar, tf.zeros([2], DTYPE)], axis=0)
    x0 = theta_ext + p0_sd * x0_raw
    etas = q_sd * eta_raw  # [T, 8]

    # Unrolled recursion (T is small and static). tf.scan is avoided
    # deliberately: its gradient under NUTS's graph tracing produced
    # IndexedSlices that TFP's kernel bootstrap cannot handle.
    horizon = eta_raw.shape[0]
    states = []
    x = x0
    for t in range(horizon):
        x = theta_ext + phi * (x - theta_ext) + etas[t]
        states.append(x)
    return tf.stack(states, axis=0)  # [T, 8]


def joint_log_prob(y, theta, x0_raw, eta_raw, target_id,
                   fix: model_tf.HardBoundFixture = model_tf.FIXTURE):
    """log p(theta, x0_raw, eta_raw | y) up to a constant."""
    theta = tf.convert_to_tensor(theta, DTYPE)
    theta_bar = theta[:6]
    log_noise = theta[6:9]
    noise_scales = tf.exp(log_noise)
    lp = _std_normal_logpdf_sum(x0_raw) + _std_normal_logpdf_sum(eta_raw)
    lp += tf.reduce_sum(
        -0.5 * ((theta_bar - PRIOR_THETA_MEAN) / PRIOR_THETA_SD) ** 2
        - tf.math.log(PRIOR_THETA_SD) - 0.5 * _LOG2PI)
    lp += tf.reduce_sum(
        -0.5 * ((log_noise - PRIOR_LOG_NOISE_MEAN) / PRIOR_LOG_NOISE_SD) ** 2
        - tf.math.log(PRIOR_LOG_NOISE_SD) - 0.5 * _LOG2PI)
    states = states_from_raws(theta_bar, x0_raw, eta_raw, fix)
    lp += tf.reduce_sum(model_tf.observation_log_density(
        y, states, noise_scales, target_id, fix))
    return lp


# ---------------------------------------------------------------------------
# K=1 gate model (master program Phase 2, binding specification):
# 1-D level factor, T=4, K=1, one maturity, hard bound at 0.
# Parameters: level mean mu_L (prior N(0.005, 0.02^2)) and log noise scale
# (prior N(log 5e-4, 0.5^2)). Latents: 5 raws.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateModel:
    phi: float = 0.9
    q_sd: float = 2e-3
    p0_sd: float = 5e-3
    maturity: float = 1.0
    lower_bound: float = 0.0
    horizon: int = 4
    prior_mu_mean: float = 0.005
    prior_mu_sd: float = 0.02
    prior_lognoise_mean: float = -7.600902459542082
    prior_lognoise_sd: float = 0.5


GATE = GateModel()


def gate_states_from_raws(mu, raws, gm: GateModel = GATE):
    x0 = mu + gm.p0_sd * raws[0]
    xs = []
    x = x0
    for t in range(gm.horizon):
        x = mu + gm.phi * (x - mu) + gm.q_sd * raws[1 + t]
        xs.append(x)
    return tf.stack(xs)


def gate_observation_mean(states, gm: GateModel = GATE):
    # level-only factor: forward f(s; x) = x (constant loading 1), so the
    # K=1 Gauss-Legendre average of max(0, x) at any maturity is max(0, x).
    return tf.maximum(tf.constant(gm.lower_bound, DTYPE), states)


def gate_joint_log_prob(y, params, raws, gm: GateModel = GATE):
    """log p(mu, log_sd, raws | y) up to a constant. params: [2], raws: [5]."""
    mu, log_sd = params[0], params[1]
    sd = tf.exp(log_sd)
    lp = _std_normal_logpdf_sum(raws)
    lp += (-0.5 * ((mu - gm.prior_mu_mean) / gm.prior_mu_sd) ** 2
           - tf.math.log(tf.constant(gm.prior_mu_sd, DTYPE)) - 0.5 * _LOG2PI)
    lp += (-0.5 * ((log_sd - gm.prior_lognoise_mean)
                   / gm.prior_lognoise_sd) ** 2
           - tf.math.log(tf.constant(gm.prior_lognoise_sd, DTYPE))
           - 0.5 * _LOG2PI)
    states = gate_states_from_raws(mu, raws, gm)
    mean = gate_observation_mean(states, gm)
    z = (tf.convert_to_tensor(y, DTYPE) - mean) / sd
    lp += tf.reduce_sum(-0.5 * z * z - log_sd - 0.5 * _LOG2PI)
    return lp


# ---------------------------------------------------------------------------
# Natively batched targets for multi-chain HMC. No pfor/vectorized_map is
# used anywhere in this package (repository governance: pfor requires prior
# written approval); batching is explicit tensor broadcasting instead.
# ---------------------------------------------------------------------------

def joint_log_prob_batched(y, theta, x0_raw, eta_raw, target_id,
                           fix: model_tf.HardBoundFixture = model_tf.FIXTURE):
    """Chain-batched joint log prob.

    theta: [C, 9]; x0_raw: [C, 8]; eta_raw: [C, T, 8]; y: [T, 13].
    Returns [C].
    """
    theta = tf.convert_to_tensor(theta, DTYPE)
    theta_bar = theta[:, :6]
    log_noise = theta[:, 6:9]
    noise_scales = tf.exp(log_noise)

    lp = tf.reduce_sum(-0.5 * x0_raw * x0_raw - 0.5 * _LOG2PI, axis=-1)
    lp += tf.reduce_sum(-0.5 * eta_raw * eta_raw - 0.5 * _LOG2PI,
                        axis=[-2, -1])
    lp += tf.reduce_sum(
        -0.5 * ((theta_bar - PRIOR_THETA_MEAN) / PRIOR_THETA_SD) ** 2
        - tf.math.log(PRIOR_THETA_SD) - 0.5 * _LOG2PI, axis=-1)
    lp += tf.reduce_sum(
        -0.5 * ((log_noise - PRIOR_LOG_NOISE_MEAN) / PRIOR_LOG_NOISE_SD) ** 2
        - tf.math.log(PRIOR_LOG_NOISE_SD) - 0.5 * _LOG2PI, axis=-1)

    phi = tf.constant(fix.phi_diag, DTYPE)
    q_sd = tf.sqrt(tf.constant(fix.q_diag, DTYPE))
    p0_sd = tf.sqrt(tf.constant(fix.p0_diag, DTYPE))
    theta_ext = tf.concat(
        [theta_bar, tf.zeros_like(theta_bar[:, :2])], axis=-1)  # [C, 8]
    x = theta_ext + p0_sd * x0_raw
    etas = q_sd * eta_raw  # [C, T, 8]
    horizon = eta_raw.shape[1]
    states = []
    for t in range(horizon):
        x = theta_ext + phi * (x - theta_ext) + etas[:, t]
        states.append(x)
    states = tf.stack(states, axis=1)  # [C, T, 8]

    mean = model_tf.observation_mean(states, target_id, fix)  # [C, T, 13]
    scales = model_tf.noise_scales_vector(noise_scales)  # [C, 13]
    z = (y[None, :, :] - mean) / scales[:, None, :]
    lp += tf.reduce_sum(
        -0.5 * z * z - tf.math.log(scales[:, None, :]) - 0.5 * _LOG2PI,
        axis=[-2, -1])
    return lp


def gate_joint_log_prob_batched(y, params, raws, gm: GateModel = GATE):
    """Chain-batched gate-model log prob. params: [C, 2]; raws: [C, 5]."""
    params = tf.convert_to_tensor(params, DTYPE)
    raws = tf.convert_to_tensor(raws, DTYPE)
    mu = params[:, 0]
    log_sd = params[:, 1]
    sd = tf.exp(log_sd)
    lp = tf.reduce_sum(-0.5 * raws * raws - 0.5 * _LOG2PI, axis=-1)
    lp += (-0.5 * ((mu - gm.prior_mu_mean) / gm.prior_mu_sd) ** 2
           - tf.math.log(tf.constant(gm.prior_mu_sd, DTYPE)) - 0.5 * _LOG2PI)
    lp += (-0.5 * ((log_sd - gm.prior_lognoise_mean)
                   / gm.prior_lognoise_sd) ** 2
           - tf.math.log(tf.constant(gm.prior_lognoise_sd, DTYPE))
           - 0.5 * _LOG2PI)
    x = mu + gm.p0_sd * raws[:, 0]
    y = tf.convert_to_tensor(y, DTYPE)
    for t in range(gm.horizon):
        x = mu + gm.phi * (x - mu) + gm.q_sd * raws[:, 1 + t]
        mean = tf.maximum(tf.constant(gm.lower_bound, DTYPE), x)
        z = (y[t] - mean) / sd
        lp += -0.5 * z * z - log_sd - 0.5 * _LOG2PI
    return lp
