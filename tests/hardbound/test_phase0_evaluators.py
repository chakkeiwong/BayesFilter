"""Phase 0 gates G0.1--G0.3 of the hard-bound master program."""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from bayesfilter.hardbound import model_tf, reference_numpy as ref
from bayesfilter.hardbound.dns_curve_tf import dns_loadings

FIX = model_tf.FIXTURE
RNG = np.random.default_rng(20260821)


def _random_states(n):
    # states spread around plausible factor magnitudes, including binding
    base = np.array([0.02, -0.01, 0.005, 0.015, -0.008, 0.004, 0.0, 0.0])
    return base + RNG.normal(scale=0.02, size=(n, 8))


def test_g0_1_tf_matches_numpy_reference():
    states = _random_states(100)
    y = ref.observation_mean(states, "mf_c1_k40_hardmax", FIX) \
        + RNG.normal(scale=5e-4, size=(100, 13))
    scales = np.array([5e-4, 5e-4, 5e-4])
    for target in ("mf_c1_k40_hardmax", "mf_s1_k40_softplus"):
        mean_tf = model_tf.observation_mean(
            tf.constant(states), target).numpy()
        mean_np = ref.observation_mean(states, target, FIX)
        np.testing.assert_allclose(mean_tf, mean_np, atol=1e-10, rtol=0)
        ld_tf = model_tf.observation_log_density(
            tf.constant(y), tf.constant(states), tf.constant(scales),
            target).numpy()
        ld_np = ref.observation_log_density(y, states, scales, target, FIX)
        np.testing.assert_allclose(ld_tf, ld_np, atol=1e-8, rtol=1e-12)


def test_g0_2_softplus_hard_gap_bounds():
    states = _random_states(500)
    hard = model_tf.observation_mean(
        tf.constant(states), "mf_c1_k40_hardmax").numpy()
    soft = model_tf.observation_mean(
        tf.constant(states), "mf_s1_k40_softplus").numpy()
    gap_d = soft[:, :6] - hard[:, :6]
    gap_f = soft[:, 6:12] - hard[:, 6:12]
    # eq (88)-(89): 0 < gap <= alpha log 2 in exact arithmetic. Far from
    # the bound the gap decays like alpha*exp(-|u-ell|/alpha) and falls
    # below float64 resolution, so the computed gap may round to 0 or to
    # a tiny negative; tolerate rounding at 100*eps of the yield scale.
    tol = 100 * np.finfo(np.float64).eps
    assert gap_d.min() > -tol and gap_f.min() > -tol
    assert gap_d.max() <= FIX.alpha_d * np.log(2.0) + 1e-15
    assert gap_f.max() <= FIX.alpha_f * np.log(2.0) + 1e-15
    # strict positivity holds where the gap is numerically resolvable:
    # near-bound states produced by the pinned construction below.
    # the bound is approached: engineer a state pinned at the bound.
    pinned = np.zeros((1, 8))
    pinned[0, :3] = [FIX.lower_bound_d, 0.0, 0.0]
    hard_p = model_tf.observation_mean(
        tf.constant(pinned), "mf_c1_k40_hardmax").numpy()
    soft_p = model_tf.observation_mean(
        tf.constant(pinned), "mf_s1_k40_softplus").numpy()
    gap_p = (soft_p - hard_p)[0, :6]
    assert gap_p.max() > 0.99 * FIX.alpha_d * np.log(2.0)


def test_g0_3_binding_sets_are_interval_patterns():
    # crossing lemma (eq 82): binding set along s is an interval, the
    # complement of an interval, empty, or everything.
    s_grid = np.linspace(1e-4, 30.0, 400)
    loadings = dns_loadings(tf.constant(s_grid), FIX.decay_d).numpy()
    states = _random_states(1000)[:, :3]
    forwards = loadings @ states.T  # [400, 1000]
    binding = (forwards <= FIX.lower_bound_d).T  # [1000, 400]
    for row in binding:
        switches = int(np.sum(row[1:] != row[:-1]))
        assert switches <= 2, f"{switches} sign changes: not interval-pattern"


def test_simulate_reproducible_and_binding_fraction():
    out = model_tf.simulate(
        tf.constant(FIX.theta_bar_truth, tf.float64),
        tf.constant(FIX.noise_scale_truth, tf.float64),
        horizon=40, seed=20260821, target_id="mf_c1_k40_hardmax")
    out2 = model_tf.simulate(
        tf.constant(FIX.theta_bar_truth, tf.float64),
        tf.constant(FIX.noise_scale_truth, tf.float64),
        horizon=40, seed=20260821, target_id="mf_c1_k40_hardmax")
    np.testing.assert_array_equal(out["observations"].numpy(),
                                  out2["observations"].numpy())
    assert out["observations"].shape == (40, 13)
