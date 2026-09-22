"""G2.3 fixed-trajectory HMC leapfrog ladder (master program Amendment A3).

A3 requires an explicit `num_leapfrog_steps` for G2.3 "selected via manual
tuning ladder". This module is that ladder. It is a NOMINATION screen, not a
gate: it ranks trajectory lengths by mixing per unit of work and asserts only
the true vetoes. The promotion criterion for the nominated L remains the G2.3
gate in `test_phase2_joint_hmc.py`.

Plan: docs/plans/hardbound-g2-3-leapfrog-ladder-2026-09-01.md
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.hardbound import model_tf, joint_target_tf
from bayesfilter.hardbound.windowed_dense_mass_adaptation import (
    run_windowed_dense_mass_adaptation,
)

jt = joint_target_tf

# Frozen across rungs so that L is the only varying control. Provenance and
# failure modes for each are tabulated in section 4 of the plan; they are
# NUTS-era warm starts, not validated defaults for this kernel.
SEED = 20260822
NUM_CHAINS = 4
SCREEN_WARMUP = 1000
SCREEN_SAMPLES = 1000
INITIAL_STEP_SIZE = 1e-2
TARGET_ACCEPT = 0.70

# Amendment A3 acceptance band, applied to the warmup phase, which is where dual
# averaging is actually driving toward the target. In-band acceptance is NOT
# evidence of good mixing -- that inference is the G2.3 error A3 corrects.
ACCEPT_BAND = (0.65, 0.75)

# The sampling phase is NOT expected to land in ACCEPT_BAND, and a band veto on
# it would fire for reasons unrelated to L. The handoff freezes
# `exp(log_averaging_step)`, the smoothed dual-averaged step size, which is more
# conservative than the instantaneous value; a smaller step buys higher
# acceptance. Measured on a 3-d Gaussian smoke target: final warmup window 0.60,
# sampling 0.86, against target 0.70. The offset is a property of the windowed
# handoff and applies at every rung, so it cannot discriminate between rungs.
# What is vetoed instead is genuine adapter failure, which this band is wide
# enough to catch and narrow enough to mean something.
PATHOLOGY_BAND = (0.30, 0.98)

# Eligibility bar for nomination. Loose on purpose: it only certifies that the
# chain moved at all, so that a rung which random-walks is excluded from the
# ESS-per-gradient ranking rather than winning it on a technicality.
ELIGIBILITY_RHAT = 1.2

LADDER = [8, 16, 32, 64, 128]

_RESULTS_PATH = pathlib.Path(
    "docs/plans/hardbound-g2-3-leapfrog-ladder-screen-2026-09-01.json")


def _g2_3_target(num_chains):
    """The G2.3 fixture: T=40, C1 hard-max target, non-centred raw chart.

    The *target* matches the gate call site in `test_phase2_joint_hmc.py`
    exactly: same fixture, horizon, simulation seed, `target_id`, and log-prob.
    The initialisation *distribution* also matches (same `raw_truth` centre,
    same 0.05/0.1/0.1 scales), but the realised draw does not.  The gate uses a
    module-level `np.random.default_rng(20260821)` whose state at its call site
    depends on which earlier tests in that file already drew from it; this uses
    a fresh, independent `RandomState`.  So a rung here is a valid draw from the
    gate's setup, not a replay of the gate's starting point -- relevant because
    R-hat at finite warmup depends on init dispersion.  Ladder comparisons are
    still internally consistent: every rung shares this one starting point.
    """
    rng = np.random.RandomState(20260826)
    fix = model_tf.FIXTURE
    horizon = 40
    sim = model_tf.simulate(
        tf.constant(fix.theta_bar_truth, tf.float64),
        tf.constant(fix.noise_scale_truth, tf.float64),
        horizon=horizon, seed=20260821, target_id="mf_c1_k40_hardmax")
    y_tf = sim["observations"]
    truth = np.array(list(fix.theta_bar_truth) + [np.log(5e-4)] * 3)

    raw_truth = jt.raw_from_theta(tf.constant(truth[None, :], tf.float64))
    init = [
        tf.constant(raw_truth.numpy() + rng.normal(size=(num_chains, 9)) * 0.05,
                    tf.float64),
        tf.constant(rng.normal(size=(num_chains, 8)) * 0.1, tf.float64),
        tf.constant(rng.normal(size=(num_chains, horizon, 8)) * 0.1,
                    tf.float64),
    ]

    def lp(theta_raw, x0_raw, eta_raw):
        return jt.joint_log_prob_raw_batched(y_tf, theta_raw, x0_raw, eta_raw,
                                             "mf_c1_k40_hardmax")

    return lp, init


@pytest.mark.hmc
@pytest.mark.extended
@pytest.mark.parametrize("num_leapfrog_steps", LADDER)
def test_g2_3_leapfrog_ladder_rung(num_leapfrog_steps):
    """One ladder rung. Records diagnostics; asserts only the true vetoes."""
    lp, init = _g2_3_target(NUM_CHAINS)

    started = time.time()
    out = run_windowed_dense_mass_adaptation(
        target_log_prob_fn=lp,
        initial_states=init,
        num_warmup_steps=SCREEN_WARMUP,
        num_samples=SCREEN_SAMPLES,
        initial_step_size=INITIAL_STEP_SIZE,
        target_accept_prob=TARGET_ACCEPT,
        seed=SEED,
        num_leapfrog_steps=num_leapfrog_steps)
    wall = time.time() - started

    rhat = out["rhat"][0].numpy()
    ess = out["ess"][0].numpy()
    accept = float(tf.reduce_mean(
        tf.cast(out["sampling_is_accepted"], tf.float64)).numpy())
    # Warmup acceptance in the last window is where the A3 band applies: it is
    # the adapter's own achieved rate against its 0.70 target, before the
    # smoothed-step-size handoff shifts the sampling rate upward.
    warmup_accept = float(out["window_diagnostics"][-1]["acceptance"])
    n_total = NUM_CHAINS * SCREEN_SAMPLES

    # Mixing per unit of work. Ranking on ESS alone would trivially favour the
    # largest L, since each iteration buys L gradients worth of trajectory.
    gradients = NUM_CHAINS * (SCREEN_WARMUP + SCREEN_SAMPLES) * num_leapfrog_steps
    ess_per_grad = float(ess.min()) / gradients

    conds = [w["condition_number"] for w in out["window_diagnostics"]
             if w["update_mass"]]
    pooled = [w["pooled_draws"] for w in out["window_diagnostics"]
              if w["update_mass"]]

    print(f"\n{'=' * 72}")
    print(f"L={num_leapfrog_steps}  step={float(out['final_step_size']):.4e}  "
          f"L*eps={num_leapfrog_steps * float(out['final_step_size']):.4f}  "
          f"wall={wall:.1f}s")
    print(f"acceptance: warmup(final window) {warmup_accept:.4f} "
          f"[A3 band {ACCEPT_BAND}]   sampling {accept:.4f} "
          f"[offset expected, see PATHOLOGY_BAND]")
    print(f"max R-hat {rhat.max():.4f}   min ESS {ess.min():.1f}   "
          f"min ESS/grad {ess_per_grad:.3e}")
    print(f"per-theta R-hat {np.array2string(rhat, precision=4)}")
    print(f"per-theta ESS   {np.array2string(ess, precision=1)}")
    print(f"divergences sampling {out['divergences']}/{n_total}  "
          f"warmup {out['warmup_divergences']}")
    print(f"slow-window condition numbers "
          f"{['%.2e' % c for c in conds]}")
    print(f"slow-window pooled draws {pooled}")
    print(f"{'=' * 72}")

    record = {
        "num_leapfrog_steps": num_leapfrog_steps,
        "max_rhat": float(rhat.max()),
        "rhat": [float(r) for r in rhat],
        "min_ess": float(ess.min()),
        "ess": [float(e) for e in ess],
        "min_ess_per_gradient": ess_per_grad,
        "gradient_evaluations": int(gradients),
        "sampling_acceptance": accept,
        "warmup_final_window_acceptance": warmup_accept,
        "warmup_window_acceptance": [
            float(w["acceptance"]) for w in out["window_diagnostics"]],
        "final_step_size": float(out["final_step_size"]),
        "integration_time": num_leapfrog_steps * float(out["final_step_size"]),
        "sampling_divergences": int(out["divergences"]),
        "warmup_divergences": int(out["warmup_divergences"]),
        "slow_window_condition_numbers": [float(c) for c in conds],
        "slow_window_pooled_draws": [int(p) for p in pooled],
        "wall_seconds": wall,
        "eligible": bool(rhat.max() < ELIGIBILITY_RHAT),
        "warmup_acceptance_in_a3_band": bool(
            ACCEPT_BAND[0] <= warmup_accept <= ACCEPT_BAND[1]),
    }
    _append_screen_record(record)

    # --- vetoes only; R-hat is deliberately NOT asserted here -------------
    # A nomination screen that failed on the criterion it exists to inform
    # would produce no information. R-hat is recorded and ranked instead.
    for part in out["samples"]:
        assert bool(tf.reduce_all(tf.math.is_finite(part)).numpy()), (
            f"nonfinite draws at L={num_leapfrog_steps}")
    assert out["divergences"] <= 0.001 * n_total, (
        num_leapfrog_steps, out["divergences"])
    # Adapter-failure veto, not an A3-band check. A rung outside this wide band
    # has a sampler that either barely moves or barely integrates, so its R-hat
    # and ESS describe something other than fixed-trajectory HMC at target 0.70.
    # The A3 band itself is recorded per rung as
    # `warmup_acceptance_in_a3_band` and read in the result note; asserting it
    # here would veto rungs for the windowed handoff's known step-size offset
    # rather than for anything about L.
    assert PATHOLOGY_BAND[0] <= accept <= PATHOLOGY_BAND[1], (
        f"L={num_leapfrog_steps} sampling acceptance {accept:.4f} outside "
        f"pathology band {PATHOLOGY_BAND}: the adapter did not produce a "
        f"usable step size, so this rung's diagnostics describe a different "
        f"sampler")


def _append_screen_record(record):
    """Accumulate rung records so the ladder survives a partial run.

    Host-side artifact writing, outside any TF kernel: diagnostic serialization
    per the CLAUDE.md host-boundary rule.
    """
    existing = []
    if _RESULTS_PATH.exists():
        existing = json.loads(_RESULTS_PATH.read_text())
    existing = [r for r in existing
                if r["num_leapfrog_steps"] != record["num_leapfrog_steps"]]
    existing.append(record)
    existing.sort(key=lambda r: r["num_leapfrog_steps"])
    _RESULTS_PATH.write_text(json.dumps(existing, indent=2) + "\n")
