"""Phase 2 gates G2.0--G2.3 of the hard-bound master program."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.hardbound import gate_grid_reference as gref
from bayesfilter.hardbound import joint_target_tf as jt
from bayesfilter.hardbound import model_tf
from bayesfilter.hardbound.hmc_runner import NutsConfig, run_nuts

RNG = np.random.default_rng(20260821)
GM = jt.GATE


def _simulate_gate(seed=20260821, hard=True, mu=None, log_sd=None):
    rng = np.random.default_rng(seed)
    mu = GM.prior_mu_mean if mu is None else mu
    log_sd = GM.prior_lognoise_mean if log_sd is None else log_sd
    raws = rng.normal(size=1 + GM.horizon)
    states = gref._gate_states(mu, raws[None, :], GM)[0]
    mean = np.maximum(GM.lower_bound, states) if hard else states
    return mean + np.exp(log_sd) * rng.normal(size=GM.horizon), states


def _affine_gate_marginal_from_tf_target(y, mu, log_sd, newton_steps=25):
    """Exact Gaussian integral of the TF joint target over the latents.

    For the bound-removed (affine) gate model the joint log prob is exactly
    quadratic in the 5 latent raws, so p(y | params) equals
    exp(lp(z*)) * (2 pi)^{5/2} / sqrt(det H), with z* the latent mode and
    H the negative latent Hessian: exact, no quadrature. Because the TF
    target hard-codes the max, we shift the model so the bound never binds
    (all forwards above the bound) instead of editing the target: we pass
    states guaranteed positive by construction of the (mu, y) draw.
    """
    params = tf.constant([mu, log_sd], tf.float64)
    y_tf = tf.constant(y, tf.float64)
    z = tf.zeros([5], tf.float64)
    for _ in range(newton_steps):
        with tf.GradientTape() as t2:
            t2.watch(z)
            with tf.GradientTape() as t1:
                t1.watch(z)
                lp = jt.gate_joint_log_prob(y_tf, params, z)
            g = t1.gradient(lp, z)
        H = t2.jacobian(g, z)
        z = z - tf.linalg.solve(H, -g[:, None])[:, 0] * -1.0
    with tf.GradientTape() as t2:
        t2.watch(z)
        with tf.GradientTape() as t1:
            t1.watch(z)
            lp = t1_lp = jt.gate_joint_log_prob(y_tf, params, z)
        g = t1.gradient(lp, z)
    H = -t2.jacobian(g, z)
    _, logdet = np.linalg.slogdet(H.numpy())
    lp0 = float(lp.numpy())
    # subtract the parameter-prior terms to isolate log p(y | params)
    prior = (-0.5 * ((mu - GM.prior_mu_mean) / GM.prior_mu_sd) ** 2
             - np.log(GM.prior_mu_sd) - 0.5 * gref.LOG2PI)
    prior += (-0.5 * ((log_sd - GM.prior_lognoise_mean)
                      / GM.prior_lognoise_sd) ** 2
              - np.log(GM.prior_lognoise_sd) - 0.5 * gref.LOG2PI)
    return lp0 - prior + 2.5 * gref.LOG2PI - 0.5 * logdet


def test_g2_0_kalman_tie_out_affine_variant():
    # Bound-removed variant: two independent exact computations of
    # p(y | params) must agree with the exact Kalman answer:
    # (a) the dense grid filter (validates the diagnostic reference), and
    # (b) the exact Gaussian integral of the TF joint target (validates
    #     the target's likelihood content), realized by keeping every
    #     state far above the bound so max() is affine along the path.
    for k in range(20):
        rng = np.random.default_rng(1000 + k)
        # keep the state path safely positive so the hard max is inactive:
        mu = 0.15 + 0.02 * rng.normal()
        log_sd = GM.prior_lognoise_mean + 0.5 * rng.normal()
        y, _ = _simulate_gate(seed=2000 + k, hard=False, mu=mu, log_sd=log_sd)
        kal = gref.gate_kalman_loglik(y, mu, log_sd, GM)
        grid = gref.gate_gridfilter_loglik(y, mu, log_sd, GM,
                                           hard_bound=False, n_grid=4000)
        assert abs(grid - kal) / abs(kal) < 1e-6, (k, grid, kal)
        tfm = _affine_gate_marginal_from_tf_target(y, mu, log_sd)
        assert abs(tfm - kal) / abs(kal) < 1e-6, (k, tfm, kal)


def test_g2_1_gradient_and_value_continuity():
    y, _ = _simulate_gate()
    y_tf = tf.constant(y, tf.float64)
    params = tf.constant([0.004, GM.prior_lognoise_mean + 0.1], tf.float64)
    raws = tf.constant(RNG.normal(size=5), tf.float64)

    # finite-difference gradient check away from kinks
    with tf.GradientTape() as tape:
        tape.watch([params, raws])
        lp = jt.gate_joint_log_prob(y_tf, params, raws)
    grads = tape.gradient(lp, [params, raws])
    eps = 1e-6
    for var_idx, var in enumerate([params, raws]):
        v = var.numpy()
        for i in range(v.shape[0]):
            vp, vm = v.copy(), v.copy()
            vp[i] += eps
            vm[i] -= eps
            args_p = [params.numpy(), raws.numpy()]
            args_m = [params.numpy(), raws.numpy()]
            args_p[var_idx] = vp
            args_m[var_idx] = vm
            fd = (jt.gate_joint_log_prob(
                    y_tf, tf.constant(args_p[0]), tf.constant(args_p[1]))
                  - jt.gate_joint_log_prob(
                    y_tf, tf.constant(args_m[0]), tf.constant(args_m[1]))
                  ).numpy() / (2 * eps)
            an = grads[var_idx].numpy()[i]
            assert abs(fd - an) < 1e-4 * max(1.0, abs(an)), (var_idx, i, fd, an)

    # Value continuity across a node-binding boundary (survey Sec. 15).
    # With raws = 0 and mu = delta, every state sits at delta, so delta = 0
    # is exactly the kink. The log density has a large but finite one-sided
    # slope there (order sd^-2), so a flat tolerance conflates slope with
    # discontinuity. The discriminating statistic is the symmetric second
    # difference lp(+d) - 2 lp(0) + lp(-d): for a continuous value with a
    # gradient kink it scales like (jump in slope) * d and vanishes as
    # d -> 0; a genuine value jump leaves it constant.
    raws0 = tf.constant(np.zeros(5), tf.float64)

    def lp_at(delta):
        params_c = tf.constant([delta, GM.prior_lognoise_mean], tf.float64)
        return float(jt.gate_joint_log_prob(y_tf, params_c, raws0).numpy())

    second_diffs = {}
    for d in (1e-6, 1e-8, 1e-10):
        second_diffs[d] = abs(lp_at(d) - 2.0 * lp_at(0.0) + lp_at(-d))
    # linear decay with d (allowing float rounding floor at tiny d)
    assert second_diffs[1e-8] < max(1e-1 * second_diffs[1e-6], 1e-7)
    assert second_diffs[1e-10] < max(1e-1 * second_diffs[1e-8], 1e-7)


def test_g2_1_full_model_gradient_smoke():
    fix = model_tf.FIXTURE
    sim = model_tf.simulate(
        tf.constant(fix.theta_bar_truth, tf.float64),
        tf.constant(fix.noise_scale_truth, tf.float64),
        horizon=8, seed=20260821, target_id="mf_c1_k40_hardmax")
    theta = tf.constant(
        list(fix.theta_bar_truth) + [np.log(5e-4)] * 3, tf.float64)
    x0_raw = tf.constant(RNG.normal(size=8), tf.float64)
    eta_raw = tf.constant(RNG.normal(size=(8, 8)), tf.float64)
    with tf.GradientTape() as tape:
        tape.watch([theta, x0_raw, eta_raw])
        lp = jt.joint_log_prob(sim["observations"], theta, x0_raw, eta_raw,
                               "mf_c1_k40_hardmax")
    grads = tape.gradient(lp, [theta, x0_raw, eta_raw])
    for g in grads:
        assert g is not None and bool(tf.reduce_all(tf.math.is_finite(g)))


@pytest.mark.hmc
def test_g2_2_gate_model_grid_agreement():
    y, _ = _simulate_gate(mu=-0.002)  # bind roughly half the periods
    y_tf = tf.constant(y, tf.float64)

    nc = 4
    init = [
        tf.constant(
            np.array([0.005, GM.prior_lognoise_mean])[None, :]
            + np.array([0.01, 0.3])[None, :] * RNG.normal(size=(nc, 2)),
            tf.float64),
        tf.constant(RNG.normal(size=(nc, 5)), tf.float64),
    ]

    def lp(params, raws):
        return jt.gate_joint_log_prob_batched(y_tf, params, raws)

    out = run_nuts(lp, init, NutsConfig(num_chains=nc, num_warmup=6000,
                                        num_samples=8000, seed=20260821,
                                        initial_step_size=2e-3,
                                        target_accept=0.99,
                                        max_tree_depth=12))
    n_total = nc * 8000
    # Amended per master-program risk table (2026-08-21): the kink target
    # makes occasional NUTS divergences unavoidable at any practical step
    # size; the exactness evidence is the grid agreement below, and the
    # sampler-health gate is a divergence rate bound, not zero.
    assert out["divergences"] <= 0.001 * n_total, out["divergences"]
    rhat_params = out["rhat"][0].numpy()
    assert np.all(rhat_params < 1.02), rhat_params
    ess = out["ess"][0].numpy()
    assert np.all(ess > 250), ess

    draws = out["samples"][0].numpy().reshape(-1, 2)  # [C*S, 2]

    mu_grid = np.linspace(-0.06, 0.06, 40)
    ls_grid = np.linspace(GM.prior_lognoise_mean - 1.6,
                          GM.prior_lognoise_mean + 1.6, 40)
    post = gref.gate_grid_posterior(y, GM, mu_grid, ls_grid)
    for dim, grid, marg in ((0, mu_grid, post.sum(1)), (1, ls_grid,
                                                        post.sum(0))):
        g_mean = float(np.sum(grid * marg))
        g_sd = float(np.sqrt(np.sum((grid - g_mean) ** 2 * marg)))
        h_mean = draws[:, dim].mean()
        h_sd = draws[:, dim].std()
        n_eff = min(float(ess[dim]), draws.shape[0])
        mc_se = h_sd / np.sqrt(n_eff)
        assert abs(h_mean - g_mean) < 3 * mc_se + 1e-12, (dim, h_mean, g_mean)
        assert abs(h_sd - g_sd) / g_sd < 0.1, (dim, h_sd, g_sd)
        # Wasserstein-1 between HMC empirical marginal and grid marginal
        cdf_grid = np.cumsum(marg)
        hmc_cdf = np.searchsorted(np.sort(draws[:, dim]), grid,
                                  side="right") / draws.shape[0]
        w1 = np.trapz(np.abs(hmc_cdf - cdf_grid), grid)
        assert w1 < 0.05 * g_sd, (dim, w1, g_sd)


@pytest.mark.hmc
def test_g2_3_full_c1_fixture_recovery():
    fix = model_tf.FIXTURE
    T = 40
    sim = model_tf.simulate(
        tf.constant(fix.theta_bar_truth, tf.float64),
        tf.constant(fix.noise_scale_truth, tf.float64),
        horizon=T, seed=20260821, target_id="mf_c1_k40_hardmax")
    y_tf = sim["observations"]
    truth = np.array(list(fix.theta_bar_truth) + [np.log(5e-4)] * 3)

    nc = 4
    init = [
        tf.constant(truth[None, :] + RNG.normal(size=(nc, 9)) * 0.05
                    * np.abs(truth).clip(min=0.01), tf.float64),
        tf.constant(RNG.normal(size=(nc, 8)) * 0.1, tf.float64),
        tf.constant(RNG.normal(size=(nc, T, 8)) * 0.1, tf.float64),
    ]

    def lp(theta, x0_raw, eta_raw):
        return jt.joint_log_prob_batched(y_tf, theta, x0_raw, eta_raw,
                                         "mf_c1_k40_hardmax")

    out = run_nuts(lp, init, NutsConfig(num_chains=nc, num_warmup=2000,
                                        num_samples=1000, seed=20260822,
                                        initial_step_size=5e-3,
                                        target_accept=0.98))
    n_total = 4 * 1000
    assert out["divergences"] <= 0.001 * n_total, out["divergences"]
    rhat = out["rhat"][0].numpy()
    assert np.all(rhat < 1.02), rhat
    draws = out["samples"][0].numpy().reshape(-1, 9)
    post_mean = draws.mean(0)
    post_sd = draws.std(0)
    assert np.all(np.abs(post_mean - truth) < 3 * post_sd), (
        post_mean, truth, post_sd)
