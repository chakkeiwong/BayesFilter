"""Phase 2 gates G2.0--G2.3 of the hard-bound master program."""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

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


def test_gate_gridfilter_batched_parity():
    """Batched grid filter matches scalar version."""
    y, _ = _simulate_gate(seed=20260823, mu=0.01, log_sd=-7.5)
    mu_test = 0.008
    logsd_grid = np.linspace(-8.0, -7.0, 10)
    # Scalar version
    scalar_ll = np.array([gref.gate_gridfilter_loglik(
        y, mu_test, ls, GM) for ls in logsd_grid])
    # Batched version
    batched_ll = gref.gate_gridfilter_loglik_batched(
        y, mu_test, logsd_grid, GM)
    np.testing.assert_allclose(batched_ll, scalar_ll, rtol=0, atol=1e-10)


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
                                        num_leapfrog_steps=100))
    n_total = nc * 8000
    # Amended per master-program risk table (2026-08-21): the kink target
    # makes occasional NUTS divergences unavoidable at any practical step
    # size; the exactness evidence is the grid agreement below, and the
    # sampler-health gate is a divergence rate bound, not zero.
    assert out["divergences"] <= 0.001 * n_total, out["divergences"]
    rhat_params = out["rhat"][0].numpy()
    assert np.all(rhat_params < 1.01), rhat_params
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
        cdf_grid = np.cumsum(marg) - 0.5 * marg
        hmc_cdf = np.searchsorted(np.sort(draws[:, dim]), grid,
                                  side="right") / draws.shape[0]
        w1 = np.trapz(np.abs(hmc_cdf - cdf_grid), grid)
        tol = max(0.10 * g_sd, 1.5 * g_sd / np.sqrt(n_eff))
        assert w1 < tol, (dim, w1, tol, g_sd, n_eff)


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
    # Non-centred parameter chart (master program Amendment A2): theta_raw is
    # O(1) under a standard normal prior. In the natural chart the
    # per-coordinate posterior sd spans 1.9e4 (theta_bar levels ~3e-5 against
    # latent shock raws ~5.8e-1), which no single step size can mix; the
    # rescale removes 50x of that and diagonal mass matrix adaptation absorbs
    # the residual 3.9e2.
    raw_truth = jt.raw_from_theta(tf.constant(truth[None, :], tf.float64))
    init = [
        tf.constant(raw_truth.numpy() + RNG.normal(size=(nc, 9)) * 0.05,
                    tf.float64),
        tf.constant(RNG.normal(size=(nc, 8)) * 0.1, tf.float64),
        tf.constant(RNG.normal(size=(nc, T, 8)) * 0.1, tf.float64),
    ]

    def lp(theta_raw, x0_raw, eta_raw):
        return jt.joint_log_prob_raw_batched(y_tf, theta_raw, x0_raw, eta_raw,
                                             "mf_c1_k40_hardmax")

    # Budget: 1600 mass matrix estimation steps (0.8*2000) proved thin for a
    # 337-dim diagonal -- attempt 3 reached max R-hat 1.073, mixing but not
    # converged. 4000 warmup gives 3200 estimation steps and 3000 draws
    # reduces finite-sample R-hat noise. The 1.01 threshold matches the
    # repository standard (bayesfilter.inference.staged_fixed_kernel_hmc
    # retained_rhat_threshold and the lgssm_neutra suite RHAT_MAX).
    # Amendment A3: diagonal adaptation reduced max R-hat from 60.5 to 1.048
    # but stalled. Six of nine coordinates have low ESS and R-hat above the
    # threshold,
    # indicating off-diagonal posterior correlation structure that the diagonal
    # preconditioner cannot capture. Switch to dense (full covariance) mass
    # matrix adaptation.
    # Amendment A4: block-dense single-freeze adaptation stalled in turn at max
    # R-hat 1.083. This route is windowed full-joint adaptation -- one metric
    # over all 337 coordinates, rebuilt at each doubling-window boundary with
    # the off-diagonals shrunk toward the empirical diagonal. See
    # docs/plans/g2_3_windowed_dense_mass_convergence.md.
    # Repair cycle 1 after the L=32 gate attempt at warmup 4000 / draws 3000
    # returned max R-hat 1.0163 -- inside [1.01, 1.02), so it passed master
    # program line 191 as written and failed this test's stricter 1.01.  Budget
    # is raised at an UNCHANGED threshold; the bound is not smoothed and L is not
    # retuned on the failed gate data, both of which the master program forbids.
    # Two pieces of evidence motivate budget specifically: (1) the slow-window
    # condition numbers were still descending when warmup ended (1.44e3 -> 8.75e2
    # at the last two windows), so the metric had not converged; (2) theta8's ESS
    # was 360 against ~1400 for the other eight, making its R-hat the noisiest of
    # the nine.  More warmup buys metric refinement, more draws buy R-hat
    # precision.  This budget is declared once and reported whatever it returns.
    num_warmup = 8000
    ns = 8000
    # Master program Amendment A3 retired NUTS for fixed-trajectory HMC, which
    # has no tree doubling and so needs an explicit trajectory length; A3
    # requires that length be chosen by a manual ladder for G2.3 rather than
    # inherited.  Selection is recorded in
    # docs/plans/hardbound-g2-3-leapfrog-ladder-2026-09-01.md (plan),
    # ...-screen-2026-09-01.json (per-rung data), and
    # ...-result-2026-09-01.md (result).  Over L in {8,16,32,64,128} at
    # warmup/draws 1000/1000, L=32 was the only rung under the 1.2 eligibility
    # bar (max R-hat 1.087) and also carried the highest min(ESS)/gradient, by
    # 4.3x.  The response is NOT monotone in L: L=64 collapsed to max R-hat
    # 4.847 while L=128 recovered to 1.228, with the damage confined to the
    # three log-noise-scale coordinates.  One seed per rung, so L=32 is a screen
    # nomination that then passed this gate, not a statistically established
    # optimum, and no L between 32 and 128 was tried.
    out = run_nuts(lp, init, NutsConfig(num_chains=nc, num_warmup=num_warmup,
                                        num_samples=ns, seed=20260822,
                                        initial_step_size=1e-2,
                                        target_accept=0.70,
                                        num_leapfrog_steps=32,
                                        dense_mass_windowed=True))
    n_total = nc * ns
    rhat = out["rhat"][0].numpy()
    ess = out["ess"][0].numpy()
    # The assertion below is scoped to `[0]` == theta_raw, 9 of the 337 sampled
    # coordinates.  The runner also returns R-hat/ESS for x0_raw (8) and
    # eta_raw (40x8 = 320); those 328 were computed and discarded here, so no
    # G2.3 run has ever checked whether the latent block converged.  A
    # parameter block can mix while the latent path does not, and the parameter
    # posterior is a marginal of the joint -- so latent non-convergence would
    # invalidate the parameter numbers rather than being a separate concern.
    #
    # Reported, NOT asserted: the three G2.3 criteria are frozen by the master
    # program and widening the R-hat assertion to all 337 coordinates would
    # change the gate contract, which needs an amendment.  Reporting first
    # establishes what such an amendment would cost.
    for _blk, _nm in ((1, "x0_raw"), (2, "eta_raw")):
        _r = out["rhat"][_blk].numpy().reshape(-1)
        _e = out["ess"][_blk].numpy().reshape(-1)
        print(f"latent {_nm:8s} n={_r.size:4d}  max R-hat {_r.max():.4f}"
              f"  n>=1.01 {int((_r >= 1.01).sum()):4d}"
              f"  min ESS {_e.min():8.1f}  median ESS {np.median(_e):8.1f}")
    # Print before asserting: a pass otherwise discards the numbers the result
    # note needs, and a 2h run should not have to be repeated to recover them.
    print(f"\nmax R-hat {rhat.max():.4f}  min ESS {ess.min():.1f}")
    print(f"per-theta R-hat {np.array2string(rhat, precision=4)}")
    print(f"per-theta ESS   {np.array2string(ess, precision=1)}")
    print(f"sampling divergences {out['divergences']} / {n_total}"
          f"  warmup divergences {out['warmup_divergences']}")
    for w in out["window_diagnostics"]:
        if w["update_mass"]:
            print(f"  window {w['index']} [{w['start']}:{w['end']}] "
                  f"cond {w['condition_number']:.3e} "
                  f"pooled {w['pooled_draws']} step {w['step_size_after']:.3e}")
    # Acceptance against A3's target 0.70, band (0.65, 0.75).  The runner
    # computes both of these and this call site previously discarded them, which
    # left the band unverified on the runs that decide the gate.  The band
    # applies to the WARMUP rate: that is the adapter's own achieved rate
    # against its target.  Sampling acceptance sits structurally higher because
    # the windowed handoff freezes `exp(log_averaging_step)`, the smoothed
    # dual-averaged step size, which is more conservative than the instantaneous
    # value.  Reported, not asserted -- acceptance is not one of the three G2.3
    # criteria, and a band assertion here would fire for handoff reasons rather
    # than sampler-health reasons.
    print(f"sampling acceptance {float(tf.reduce_mean(tf.cast(out['sampling_is_accepted'], tf.float64))):.4f}"
          f"  warmup acceptance (final window) "
          f"{out['window_diagnostics'][-1]['acceptance']:.4f}"
          f"  [A3 target 0.70, band (0.65, 0.75) applies to warmup]")
    # R-hat on prefixes of the retained draws.  This is the diagnostic that
    # separates the two readings of a near-threshold max R-hat: a descending
    # sequence means finite-sample noise or slow mixing that more draws fix,
    # while a flat sequence means the chains disagree about the distribution and
    # no additional draws will help.  Observability only -- not asserted.
    raw = out["samples"][0].numpy()          # [ns, nc, 9]
    for frac in (0.25, 0.5, 0.75, 1.0):
        cut = max(2, int(ns * frac))
        pref = tfp.mcmc.potential_scale_reduction(
            tf.constant(raw[:cut], tf.float64), split_chains=True).numpy()
        print(f"  prefix {frac:4.2f} ({cut:5d} draws)  max R-hat {pref.max():.4f}"
              f"  theta8 R-hat {pref[8]:.4f}")
    # Map draws back to the natural chart before comparing against truth.  Done
    # before the R-hat assertion so a near-threshold R-hat failure still reports
    # whether the fixture was recovered, which is the scientific question; the
    # earlier ordering discarded it on every failing run.
    draws = jt.theta_from_raw(raw.reshape(-1, 9)).numpy()
    post_mean = draws.mean(0)
    post_sd = draws.std(0)
    print(f"post_mean {np.array2string(post_mean, precision=6)}")
    print(f"truth     {np.array2string(truth, precision=6)}")
    print(f"|err|/sd  {np.array2string(np.abs(post_mean - truth) / post_sd, precision=3)}")
    # Prior -> posterior contraction, the diagnostic that separates "the
    # likelihood identified this" from "the prior was already centred on the
    # truth".  PRIOR_PARAM_MEAN equals FIXTURE.theta_bar_truth exactly, and
    # PRIOR_LOG_NOISE_MEAN equals log(noise_scale_truth), so the prior is
    # centred ON the value being recovered.  Coverage of truth by a credible
    # interval is therefore NOT by itself evidence about the likelihood -- a
    # sampler that ignored the data entirely would also cover.  What separates
    # the two is the sd ratio: 1.0 means the posterior IS the prior, large
    # means the data dominates.  Reported, not asserted.
    prior_sd = jt.PRIOR_PARAM_SD.numpy()
    print(f"prior_sd  {np.array2string(prior_sd, precision=4)}")
    print(f"sd_ratio  {np.array2string(prior_sd / post_sd, precision=2)}"
          f"   (prior sd / post sd; 1.0 = data added nothing)")
    # 95% equal-tailed credible interval, and whether it covers the simulation
    # truth coordinate by coordinate.  The third gate criterion below is a 3-sd
    # screen on the posterior MEAN, which is a weaker and differently shaped
    # check than interval coverage: it is symmetric by construction and says
    # nothing about either tail.  Coverage is the question a reader actually
    # asks of a posterior, and the draws it needs are already in hand, so
    # reporting it costs nothing beyond two percentiles.
    #
    # Reported, NOT asserted.  The three G2.3 criteria are frozen by the master
    # program; adding a fourth would change the gate contract and needs an
    # amendment, not a test edit.
    #
    # This is NOT a calibration result.  One fixture at one simulation seed
    # gives each parameter a single Bernoulli trial, and the nine are dependent
    # through the shared 40x13 observation array, so "k of 9 covered" has no
    # useful sampling distribution here.  Calibration would need simulation-
    # based calibration over many prior draws, which this gate does not attempt.
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    covered = (truth >= lo) & (truth <= hi)
    # Where truth sits inside the interval, in units of the interval half-width:
    # 0 at the centre, +-1 exactly at an edge.  Magnitudes near 1 are borderline,
    # because the quantiles themselves carry Monte Carlo error at min ESS ~849.
    pos = (truth - 0.5 * (lo + hi)) / (0.5 * (hi - lo))
    print(f"ci95_lo   {np.array2string(lo, precision=6)}")
    print(f"ci95_hi   {np.array2string(hi, precision=6)}")
    print(f"covered   {np.array2string(covered)}  ({int(covered.sum())}/9)")
    print(f"pos_in_ci {np.array2string(pos, precision=3)}")
    assert out["divergences"] <= 0.001 * n_total, out["divergences"]
    # ESS reported alongside R-hat: distinguishes "needs more draws" from
    # "a coordinate block still will not move".
    assert np.all(rhat < 1.01), (rhat, ess)
    assert np.all(np.abs(post_mean - truth) < 3 * post_sd), (
        post_mean, truth, post_sd)
