"""Phase 2 mechanics checks for windowed full-joint dense mass adaptation.

Smoke and mechanics scope only, per
`docs/plans/g2_3_windowed_dense_mass_convergence.md`.  These checks establish
that the window loop runs, stays finite, produces the documented output
structure, and reconstructs the target correctly.  They deliberately assert
nothing about convergence: the budgets here are far too small, and R-hat from a
200-step warmup carries no information about the G2.3 promotion criterion.
"""

from __future__ import annotations

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.hardbound import joint_target_tf as jt
from bayesfilter.hardbound import model_tf
from bayesfilter.hardbound.hmc_runner import NutsConfig, run_nuts
from bayesfilter.hardbound.windowed_dense_mass_adaptation import (
    _make_flat_target,
    _merge_truncated_tail_window,
    _shrink_toward_diagonal,
    _static_part_sizes,
)
from bayesfilter.inference.hmc_tuning import (
    WindowedMassAdaptationConfig,
    build_windowed_warmup_schedule,
)

RNG = np.random.default_rng(20260826)
T = 40
TARGET_ID = "mf_c1_k40_hardmax"


def _g2_3_target(num_chains):
    """The G2.3 target and an initialization, matching the gate test setup."""
    fix = model_tf.FIXTURE
    sim = model_tf.simulate(
        tf.constant(fix.theta_bar_truth, tf.float64),
        tf.constant(fix.noise_scale_truth, tf.float64),
        horizon=T, seed=20260821, target_id=TARGET_ID)
    y_tf = sim["observations"]
    truth = np.array(list(fix.theta_bar_truth) + [np.log(5e-4)] * 3)
    raw_truth = jt.raw_from_theta(tf.constant(truth[None, :], tf.float64))

    init = [
        tf.constant(raw_truth.numpy() + RNG.normal(size=(num_chains, 9)) * 0.05,
                    tf.float64),
        tf.constant(RNG.normal(size=(num_chains, 8)) * 0.1, tf.float64),
        tf.constant(RNG.normal(size=(num_chains, T, 8)) * 0.1, tf.float64),
    ]

    def lp(theta_raw, x0_raw, eta_raw):
        return jt.joint_log_prob_raw_batched(
            y_tf, theta_raw, x0_raw, eta_raw, TARGET_ID)

    return lp, init


def test_flat_target_matches_part_target():
    """Flattening is a relabelling, so the density must be unchanged.

    This is the check that the full-joint route computes the same target as the
    block route rather than a differently-conditioned one.  A closed-form
    identity, so it is a hard veto: any mismatch means the concatenate/split
    round trip is wrong and every downstream number is meaningless.
    """
    lp, init = _g2_3_target(num_chains=3)
    shapes, sizes = _static_part_sizes(init)
    assert sizes == [9, 8, T * 8]

    flat_target = _make_flat_target(lp, shapes, sizes)
    flat = tf.concat(
        [tf.reshape(p, [3, s]) for p, s in zip(init, sizes)], axis=1)
    assert int(flat.shape[1]) == 337

    expected = lp(*init).numpy()
    actual = flat_target(flat).numpy()
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_shrinkage_preserves_marginal_variances():
    """Diagonal-target shrinkage must move only the off-diagonal entries.

    The route relies on this to stay usable in the raw chart, where marginal
    variances span orders of magnitude and an identity target would swamp the
    small-variance coordinates.
    """
    dim = 12
    a = RNG.normal(size=(dim, dim))
    cov = tf.constant(a @ a.T + np.eye(dim), tf.float64)
    scale = tf.constant(np.diag(10.0 ** RNG.uniform(-4, 4, size=dim)),
                        tf.float64)
    cov = scale @ cov @ scale

    shrunk = _shrink_toward_diagonal(cov, 0.1)

    np.testing.assert_allclose(
        tf.linalg.diag_part(shrunk).numpy(),
        tf.linalg.diag_part(cov).numpy(), rtol=1e-12)
    off = cov - tf.linalg.diag(tf.linalg.diag_part(cov))
    np.testing.assert_allclose(
        (shrunk - tf.linalg.diag(tf.linalg.diag_part(shrunk))).numpy(),
        (0.9 * off).numpy(), rtol=1e-12)
    # Shrinkage must not destroy positive definiteness.
    assert tf.reduce_min(tf.linalg.eigvalsh(shrunk)).numpy() > 0.0


def test_tail_window_merge_at_g2_3_budget():
    """The frozen metric must not come from a truncated stub window.

    At the G2.3 budget the shared builder emits `..., 800, 1600, 700`.  The
    sampling phase freezes the last slow window's metric, so an un-merged
    schedule would hand the sampler a covariance built from 700 steps when the
    preceding window already had 1600.  Cheap structural check; no sampling.
    """
    raw = build_windowed_warmup_schedule(WindowedMassAdaptationConfig(
        warmup_steps=4000, initial_buffer=75, final_buffer=50,
        first_window_size=25))
    raw_slow = [w.end - w.start for w in raw if w.kind == "slow"]
    assert raw_slow == [25, 50, 100, 200, 400, 800, 1600, 700], raw_slow

    merged = _merge_truncated_tail_window(raw)
    slow = [w.end - w.start for w in merged if w.kind == "slow"]
    assert slow == [25, 50, 100, 200, 400, 800, 2300], slow

    # Still a contiguous cover of the full warmup, with indices renumbered.
    assert merged[0].start == 0
    assert merged[-1].end == 4000
    for prev, nxt in zip(merged, merged[1:]):
        assert prev.end == nxt.start
    assert [w.index for w in merged] == list(range(len(merged)))
    # The frozen metric now uses at least as many steps as any earlier window.
    assert slow[-1] >= max(slow[:-1])


def test_tail_window_merge_is_a_noop_on_clean_schedules():
    """A schedule whose last slow window is already full must be untouched."""
    clean = build_windowed_warmup_schedule(WindowedMassAdaptationConfig(
        warmup_steps=200, initial_buffer=75, final_buffer=50,
        first_window_size=25))
    assert [w.end - w.start for w in clean if w.kind == "slow"] == [25, 50]
    # 50 >= 25, so nothing is merged.
    assert _merge_truncated_tail_window(clean) == clean


@pytest.mark.hmc
def test_windowed_route_shapes_and_finiteness():
    """Window loop runs end to end and returns the documented structure.

    Budget is a smoke budget.  Hard vetoes: crash, non-finite draws, wrong
    output structure.  Everything numeric here is explanatory only.
    """
    lp, init = _g2_3_target(num_chains=2)

    out = run_nuts(lp, init, NutsConfig(
        num_chains=2, num_warmup=200, num_samples=50, seed=20260826,
        initial_step_size=1e-2, target_accept=0.70,
        dense_mass_windowed=True))

    assert len(out["samples"]) == 3
    for drawn, part in zip(out["samples"], init):
        assert tuple(drawn.shape) == (50, 2) + tuple(part.shape[1:])
        assert bool(tf.reduce_all(tf.math.is_finite(drawn)).numpy())

    # R-hat/ESS must come back on the coordinate shape the gate test indexes.
    assert tuple(out["rhat"][0].shape) == (9,)
    assert tuple(out["ess"][0].shape) == (9,)

    windows = out["window_diagnostics"]
    assert [w["kind"] for w in windows][0] == "initial_fast"
    assert [w["kind"] for w in windows][-1] == "final_fast"
    assert not windows[0]["update_mass"]
    assert not windows[-1]["update_mass"]
    # Contiguous cover of the whole warmup.
    assert windows[0]["start"] == 0
    assert windows[-1]["end"] == 200
    for prev, nxt in zip(windows, windows[1:]):
        assert prev["end"] == nxt["start"]

    updated = [w for w in windows if w["update_mass"]]
    assert updated, "no slow window rebuilt the metric"
    for w in updated:
        assert np.isfinite(w["condition_number"])
        assert w["min_eigenvalue"] > 0.0

    print("windows:", [(w["kind"], w["start"], w["end"]) for w in windows])
    print("sampling divergences:", out["divergences"],
          "warmup divergences:", out["warmup_divergences"])
    print("final step size:", float(out["final_step_size"].numpy()))
    print("condition numbers:", [w["condition_number"] for w in updated])


@pytest.mark.hmc
def test_windowed_route_doubling_schedule():
    """Slow windows double until the final buffer, over a longer warmup.

    Verifies the schedule mechanics that the 200-step smoke budget is too short
    to exercise: the initial buffer is protected, slow-window lengths double,
    and the metric is rebuilt once per slow window.
    """
    lp, init = _g2_3_target(num_chains=1)

    out = run_nuts(lp, init, NutsConfig(
        num_chains=1, num_warmup=1000, num_samples=10, seed=20260826,
        initial_step_size=1e-2, target_accept=0.70,
        dense_mass_windowed=True))

    windows = out["window_diagnostics"]
    assert windows[0]["end"] == 75, "initial buffer not 75 steps"
    assert windows[-1]["start"] == 950, "final buffer not 50 steps"

    slow = [w for w in windows if w["kind"] == "slow"]
    lengths = [w["end"] - w["start"] for w in slow]
    # Doubling, with the truncated tail merged into the last full window:
    # the raw builder gives [25, 50, 100, 200, 400, 100] over this span.
    assert lengths == [25, 50, 100, 200, 500], lengths
    assert lengths[-1] >= max(lengths[:-1]), (
        "the frozen metric must not come from a truncated window")
    assert all(w["update_mass"] for w in slow)
    assert all(bool(np.isfinite(w["condition_number"])) for w in slow)

    for drawn in out["samples"]:
        assert bool(tf.reduce_all(tf.math.is_finite(drawn)).numpy())

    print("slow window lengths:", lengths)
    print("condition numbers:", [w["condition_number"] for w in slow])
    print("step sizes:", [w["step_size_after"] for w in windows])
