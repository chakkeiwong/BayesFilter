"""Geweke joint-distribution and simulation-based-calibration harnesses.

Master program Phase 3. Both harnesses treat the sampler as a black box on
the C1 target; neither needs an analytic posterior. Runtime TF float64.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.hardbound import joint_target_tf as jt
from bayesfilter.hardbound import model_tf
from bayesfilter.hardbound.dns_curve_tf import DTYPE

FIX = model_tf.FIXTURE


def _prior_sample(rng):
    theta_bar = (jt.PRIOR_THETA_MEAN.numpy()
                 + jt.PRIOR_THETA_SD.numpy() * rng.standard_normal(6))
    log_noise = (jt.PRIOR_LOG_NOISE_MEAN.numpy()
                 + jt.PRIOR_LOG_NOISE_SD.numpy() * rng.standard_normal(3))
    return np.concatenate([theta_bar, log_noise])


def _simulate_y(theta, x0_raw, eta_raw, target_id):
    states = jt.states_from_raws(
        tf.constant(theta[:6], DTYPE),
        tf.constant(x0_raw, DTYPE),
        tf.constant(eta_raw, DTYPE))
    mean = model_tf.observation_mean(states, target_id)
    scales = model_tf.noise_scales_vector(
        tf.exp(tf.constant(theta[6:9], DTYPE)))
    return states, mean, scales


_TRANSITION_CACHE: dict = {}


def _make_transition_fn(target_id, horizon, num_steps):
    """One traced NUTS transition fn per (target, horizon, steps): avoids
    per-iteration retracing in the successive-conditional loop."""
    key = (target_id, horizon, num_steps)
    if key in _TRANSITION_CACHE:
        return _TRANSITION_CACHE[key]

    @tf.function(input_signature=[
        tf.TensorSpec([horizon, 13], DTYPE),
        tf.TensorSpec([9], DTYPE),
        tf.TensorSpec([8], DTYPE),
        tf.TensorSpec([horizon, 8], DTYPE),
        tf.TensorSpec([2], tf.int32),
    ])
    def transition(y, theta0, x0_raw0, eta_raw0, seed):
        def lp(theta, x0_raw, eta_raw):
            return jt.joint_log_prob(y, theta, x0_raw, eta_raw, target_id)

        # Amendment A3: fixed-trajectory HMC replaces NUTS suite-wide. The
        # trajectory length stands in for the retired `max_tree_depth=8`
        # ceiling, which bounded NUTS at 2**8 = 256 leapfrog steps.
        kernel = tfp.mcmc.HamiltonianMonteCarlo(
            lp, step_size=tf.constant(2e-3, DTYPE), num_leapfrog_steps=50)
        out = tfp.mcmc.sample_chain(
            num_results=1, num_burnin_steps=num_steps - 1,
            current_state=[theta0, x0_raw0, eta_raw0],
            kernel=kernel, seed=seed, trace_fn=None)
        return out[0][-1], out[1][-1], out[2][-1]

    _TRANSITION_CACHE[key] = transition
    return transition


def _posterior_transition(y, theta0, x0_raw0, eta_raw0, target_id,
                          num_leapfrog_results=25, seed=0):
    """A few NUTS transitions leaving p(theta, z | y) invariant."""
    fn = _make_transition_fn(target_id, eta_raw0.shape[0],
                             num_leapfrog_results)
    out = fn(tf.constant(y, DTYPE), tf.constant(theta0, DTYPE),
             tf.constant(x0_raw0, DTYPE), tf.constant(eta_raw0, DTYPE),
             tf.constant([seed % (2**31 - 1), (seed + 7) % (2**31 - 1)],
                         tf.int32))
    return [o.numpy() for o in out]


@dataclass
class GewekeResult:
    z_scores: np.ndarray
    names: list


def geweke_test(target_id="mf_c1_k40_hardmax", horizon=8, n_mc=4000,
                n_sc=4000, thin=1, seed=20260821, transitions_per_step=25):
    """Geweke (2004) marginal-conditional vs successive-conditional.

    Test functions: 9 parameters, their squares, binding fraction of the
    simulated observation means, and mean FX row: 20 functionals.
    Returns z-scores (MC vs SC mean differences / pooled se with SC
    autocorrelation-adjusted variance via batch means).
    """
    rng = np.random.default_rng(seed)

    def functionals(theta, mean_obs):
        bind_frac_d = float(np.mean(
            mean_obs[:, :6] <= FIX.lower_bound_d + 1e-12))
        bind_frac_f = float(np.mean(
            mean_obs[:, 6:12] <= FIX.lower_bound_f + 1e-12))
        return np.concatenate([theta, theta**2,
                               [bind_frac_d, bind_frac_f]])

    # marginal-conditional: iid draws from the joint
    mc = []
    for _ in range(n_mc):
        theta = _prior_sample(rng)
        x0_raw = rng.standard_normal(8)
        eta_raw = rng.standard_normal((horizon, 8))
        _, mean, _ = _simulate_y(theta, x0_raw, eta_raw, target_id)
        mc.append(functionals(theta, mean.numpy()))
    mc = np.array(mc)

    # successive-conditional: alternate posterior transition and data refresh
    theta = _prior_sample(rng)
    x0_raw = rng.standard_normal(8)
    eta_raw = rng.standard_normal((horizon, 8))
    sc = []
    for i in range(n_sc):
        states, mean, scales = _simulate_y(theta, x0_raw, eta_raw, target_id)
        y = (mean + scales * tf.constant(
            rng.standard_normal(mean.shape), DTYPE)).numpy()
        theta, x0_raw, eta_raw = _posterior_transition(
            y, theta, x0_raw, eta_raw, target_id,
            num_leapfrog_results=transitions_per_step, seed=seed + i)
        _, mean2, _ = _simulate_y(theta, x0_raw, eta_raw, target_id)
        sc.append(functionals(theta, mean2.numpy()))
    sc = np.array(sc)[::thin]

    # batch-means variance for the autocorrelated SC chain
    def batch_var(a, n_batch=20):
        n_batch = max(2, min(n_batch, len(a) // 2))
        m = len(a) // n_batch
        bm = a[: m * n_batch].reshape(n_batch, m, -1).mean(axis=1)
        return bm.var(axis=0, ddof=1) / n_batch

    var_mc = mc.var(axis=0, ddof=1) / len(mc)
    var_sc = batch_var(sc)
    z = (mc.mean(0) - sc.mean(0)) / np.sqrt(var_mc + var_sc)
    names = ([f"theta_{i}" for i in range(9)]
             + [f"theta2_{i}" for i in range(9)]
             + ["bind_frac_d", "bind_frac_f"])
    return GewekeResult(z_scores=z, names=names)


def sbc(target_id="mf_c1_k40_hardmax", horizon=20, n_reps=200,
        n_posterior=100, warmup=600, seed=20260821):
    """Simulation-based calibration ranks for the 9 parameters."""
    from bayesfilter.hardbound.hmc_runner import NutsConfig, run_nuts

    rng = np.random.default_rng(seed)
    ranks = np.zeros((n_reps, 9), dtype=int)
    for r in range(n_reps):
        theta_true = _prior_sample(rng)
        x0_raw = rng.standard_normal(8)
        eta_raw = rng.standard_normal((horizon, 8))
        _, mean, scales = _simulate_y(theta_true, x0_raw, eta_raw, target_id)
        y = (mean + scales * tf.constant(
            rng.standard_normal(mean.shape), DTYPE)).numpy()
        y_tf = tf.constant(y, DTYPE)

        def lp(theta, x0r, etar):
            return jt.joint_log_prob_batched(y_tf, theta, x0r, etar,
                                             target_id)

        init = [
            tf.constant(theta_true[None, :]
                        + 0.3 * np.concatenate(
                            [jt.PRIOR_THETA_SD.numpy(),
                             jt.PRIOR_LOG_NOISE_SD.numpy()])[None, :]
                        * rng.standard_normal((1, 9)), DTYPE),
            tf.constant(rng.standard_normal((1, 8)) * 0.5, DTYPE),
            tf.constant(rng.standard_normal((1, horizon, 8)) * 0.5, DTYPE),
        ]
        out = run_nuts(lp, init, NutsConfig(
            num_chains=1, num_warmup=warmup, num_samples=n_posterior,
            seed=seed + 13 * r))
        draws = out["samples"][0].numpy()[:, 0, :]  # [n_posterior, 9]
        ranks[r] = (draws < theta_true[None, :]).sum(axis=0)
    return ranks
