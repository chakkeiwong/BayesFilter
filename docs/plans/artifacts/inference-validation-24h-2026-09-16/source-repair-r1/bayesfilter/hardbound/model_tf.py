"""The two-country hard-bound working example: constants, simulation,
observation density (survey Sec. 13.1; master program Sec. 2, frozen).

Targets: ``mf_s1_k40_softplus`` (softplus map) and ``mf_c1_k40_hardmax``
(hard max). All runtime tensors float64 TF.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tensorflow as tf

from bayesfilter.hardbound.dns_curve_tf import DTYPE, yield_curve

TARGET_S1 = "mf_s1_k40_softplus"
TARGET_C1 = "mf_c1_k40_hardmax"
_BOUND_MAP = {TARGET_S1: "softplus", TARGET_C1: "hard"}

MATURITIES = (0.25, 1.0, 2.0, 5.0, 10.0, 30.0)
FX_MATURITY = 1.0


@dataclass(frozen=True)
class HardBoundFixture:
    """Frozen working-example constants (master program Sec. 2)."""

    decay_d: float = 0.65
    decay_f: float = 0.45
    lower_bound_d: float = 0.0
    lower_bound_f: float = -0.005
    alpha_d: float = 1.5e-3
    alpha_f: float = 1.0e-3
    quadrature_order: int = 40
    phi_diag: tuple = (0.95, 0.90, 0.85, 0.95, 0.90, 0.85, 0.98, 0.98)
    q_diag: tuple = (2e-6, 3e-6, 4e-6, 2e-6, 3e-6, 4e-6, 0.5e-6, 0.5e-6)
    p0_diag: tuple = (1e-5, 1.5e-5, 2e-5, 1e-5, 1.5e-5, 2e-5, 0.25e-5, 0.25e-5)
    theta_bar_truth: tuple = (0.02, -0.01, 0.005, 0.015, -0.008, 0.004)
    noise_scale_truth: tuple = (5e-4, 5e-4, 5e-4)  # dom yields, for yields, FX
    maturities: tuple = MATURITIES
    fx_maturity: float = FX_MATURITY


FIXTURE = HardBoundFixture()


def _country_yields(fix, factors, country, target_id):
    if country == "d":
        decay, ell, alpha = fix.decay_d, fix.lower_bound_d, fix.alpha_d
    else:
        decay, ell, alpha = fix.decay_f, fix.lower_bound_f, fix.alpha_f
    return yield_curve(
        factors,
        tf.constant(fix.maturities, DTYPE),
        decay,
        ell,
        alpha,
        _BOUND_MAP[target_id],
        fix.quadrature_order,
    )


def observation_mean(states: tf.Tensor, target_id: str,
                     fix: HardBoundFixture = FIXTURE) -> tf.Tensor:
    """Mean observation h(x): 6 domestic yields, 6 foreign yields, 1 FX row.

    ``states``: [..., 8] -> returns [..., 13].
    """
    states = tf.convert_to_tensor(states, DTYPE)
    xd, xf, xb = states[..., 0:3], states[..., 3:6], states[..., 6:8]
    yd = _country_yields(fix, xd, "d", target_id)  # [..., 6]
    yf = _country_yields(fix, xf, "f", target_id)  # [..., 6]
    # FX at tau_fx: tau * (y_d(tau) - y_f(tau) + b), unit basis loadings.
    tau = tf.constant(fix.fx_maturity, DTYPE)
    yd_fx = yield_curve(xd, tau[None], fix.decay_d, fix.lower_bound_d,
                        fix.alpha_d, _BOUND_MAP[target_id],
                        fix.quadrature_order)[..., 0]
    yf_fx = yield_curve(xf, tau[None], fix.decay_f, fix.lower_bound_f,
                        fix.alpha_f, _BOUND_MAP[target_id],
                        fix.quadrature_order)[..., 0]
    basis = xb[..., 0] + xb[..., 1]
    fx = tau * (yd_fx - yf_fx + basis)
    return tf.concat([yd, yf, fx[..., None]], axis=-1)


def noise_scales_vector(noise_scales: tf.Tensor) -> tf.Tensor:
    """Expand 3 grouped scales to the 13 observation rows.

    Implemented as a dense matmul rather than ``tf.repeat`` because the
    gradient of ``tf.repeat`` is an ``IndexedSlices``, which TFP's NUTS
    internals cannot multiply during kernel bootstrap.
    """
    noise_scales = tf.convert_to_tensor(noise_scales, DTYPE)
    expand = tf.constant(
        [[1.0] * 6 + [0.0] * 7,
         [0.0] * 6 + [1.0] * 6 + [0.0],
         [0.0] * 12 + [1.0]], DTYPE)  # [3, 13]
    return tf.linalg.matvec(expand, noise_scales, transpose_a=True)


def observation_log_density(y: tf.Tensor, states: tf.Tensor,
                            noise_scales: tf.Tensor, target_id: str,
                            fix: HardBoundFixture = FIXTURE) -> tf.Tensor:
    """Sum over rows of the diagonal-Gaussian observation log density.

    ``y``, mean: [..., 13]; ``noise_scales``: [..., 3]; returns [...].
    """
    y = tf.convert_to_tensor(y, DTYPE)
    mean = observation_mean(states, target_id, fix)
    scales = noise_scales_vector(noise_scales)
    z = (y - mean) / scales
    log2pi = tf.constant(1.8378770664093453, DTYPE)
    return tf.reduce_sum(
        -0.5 * z * z - tf.math.log(scales) - 0.5 * log2pi, axis=-1
    )


def simulate(theta_bar: tf.Tensor, noise_scales: tf.Tensor, horizon: int,
             seed: int, target_id: str,
             fix: HardBoundFixture = FIXTURE) -> dict:
    """Simulate states and observations from the model at given parameters."""
    gen = tf.random.Generator.from_seed(seed)
    phi = tf.constant(fix.phi_diag, DTYPE)
    q_sd = tf.sqrt(tf.constant(fix.q_diag, DTYPE))
    p0_sd = tf.sqrt(tf.constant(fix.p0_diag, DTYPE))
    theta_ext = tf.concat(
        [tf.convert_to_tensor(theta_bar, DTYPE), tf.zeros([2], DTYPE)], axis=0
    )
    x = theta_ext + p0_sd * gen.normal([8], dtype=DTYPE)
    states = []
    for _ in range(horizon):
        x = theta_ext + phi * (x - theta_ext) + q_sd * gen.normal([8], dtype=DTYPE)
        states.append(x)
    states = tf.stack(states, axis=0)  # [T, 8]
    mean = observation_mean(states, target_id, fix)  # [T, 13]
    scales = noise_scales_vector(tf.convert_to_tensor(noise_scales, DTYPE))
    y = mean + scales * gen.normal(tf.shape(mean), dtype=DTYPE)
    return {"states": states, "observations": y, "means": mean}
