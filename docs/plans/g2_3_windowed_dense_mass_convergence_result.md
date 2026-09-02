# G2.3 Windowed Dense Mass Convergence — Result Note

Plan: [g2_3_windowed_dense_mass_convergence.md](g2_3_windowed_dense_mass_convergence.md)
Date: 2026-08-30
Status: primary criterion met on one seed; ranking claims not supported

## Outcome

`test_g2_3_full_c1_fixture_recovery` passes with windowed full-joint dense mass
adaptation. The test's own assertions are the criterion, and all three held:
sampling divergences within `0.001 * 12000`, `max R-hat < 1.02` across all nine
theta coordinates, and posterior means within `3 * post_sd` of the fixture
truth.

Run: 4 chains, 4000 warmup, 3000 draws, 337 dimensions, seed 20260822, CPU,
XLA on, float64, wall time 2 h 11 m 08 s (7867.72 s).

This is the first route to clear the threshold. Diagonal adaptation reached max
R-hat 1.048 and block-dense single-freeze reached 1.083, both against the
unchanged 1.02 threshold.

## Decision Table

| Field | Status |
|---|---|
| Decision | Windowed full-joint route clears the G2.3 gate; keep `dense_mass_windowed=True` on this test |
| Primary criterion | Met — `max R-hat < 1.02` on all 9 theta coordinates |
| Veto diagnostics | None fired — sampling divergences within budget, all draws finite, no Cholesky failure, posterior-vs-truth check passed |
| Main uncertainty | One seed. λ=0.1 and the diagonal shrinkage target are uncalibrated for this target; the tail-window merge is justified by argument, not by an ablation |
| Next justified action | If this route is to become a default rather than a passing test, run a multi-seed replication and a λ sweep |
| Not concluded | That windowed full-joint adaptation is superior to the diagonal or block-dense routes; that λ=0.1 or the diagonal target is optimal; that the merge is what produced the pass; that G2.3's posterior is correct |

## Inference-Status Table

| Row | Content |
|---|---|
| Hard veto screen | Supported. Divergence budget, finiteness, metric validity, and the posterior-vs-truth check all pass. These are pass/fail conditions and the run's status on them is established |
| Statistically supported ranking | None. One seed per route and no uncertainty analysis. The three routes are not ranked by this evidence |
| Descriptive-only differences | The 1.048 / 1.083 / pass progression across diagonal, block-dense, and windowed full-joint routes. Suggestive of the intended mechanism, not a measured effect. Window condition numbers likewise |
| Default-readiness | Not established. Passing this test authorizes the flag on this test, not a repository default. A default change needs multi-seed evidence |
| Next evidence needed | Multi-seed replication (≥4 seeds) for any ranking claim; λ ∈ {0.05, 0.1, 0.2, 0.3} sweep to calibrate the shrinkage; a merge-off arm to attribute the pass to Amendment 2 |

## What Changed From The Approved Plan

Two amendments, both recorded in the plan and both found by reading the shared
interfaces rather than by a failed run.

**Amendment 1 — shrinkage target.** The plan specified `(1-λ)·cov + λ·I`. G2.3
is sampled in the raw chart where marginal sd spans about 1.9e4, so an identity
target is negligible against the large coordinates and dominant over the small
ones; at λ=0.1 it would have replaced the small marginals with values orders of
magnitude too large. The implementation uses `(1-λ)·cov + λ·diag(cov)`, which
moves only off-diagonals, preserves every marginal variance exactly, and is
invariant to per-coordinate rescaling. This also targets where the small-sample
problem actually is: marginals are estimated from all pooled draws, the 56,953
covariance entries are not.

**Amendment 2 — truncated tail window.** `build_windowed_warmup_schedule`
emits the leftover slow span as a short final window, giving
`[25,50,100,200,400,800,1600,700]` at this budget. The sampling phase freezes
the last slow window's metric, so the approved plan would have frozen a
covariance built from 700 steps right after a window with 1600.
`_merge_truncated_tail_window` extends the last full window instead, giving
`[...,800,2300]`. The shared builder is unmodified; this is hardbound-route
post-processing.

## Verification

Six focused checks, all passing — see the Phase 2 Outcome section of the plan
for the full table. Two are closed-form and carry the most weight:

- `test_flat_target_matches_part_target` — the flatten/split round trip
  reproduces the block target exactly (0 atol, 0 rtol). This is what rules out
  "the full-joint route converged because it sampled a different density."
- `test_shrinkage_preserves_marginal_variances` — diagonal-target shrinkage
  moves only off-diagonals and preserves positive definiteness under a 1e8
  variance spread.

Regression: `tests/hardbound/` non-hmc suite.

## Post-Run Red Team

**Strongest alternative explanation.** The windowed schedule interleaves eight
metric rebuilds into the same 4000-step warmup, and the merge gives the final
metric 2,300 steps of draws. The pass may be attributable to the schedule's
effective warmup quality rather than to full-joint versus block-diagonal
structure. Nothing here isolates the full-joint metric as the operative change,
because both amendments and the schedule change moved together.

**What would overturn the conclusion.** A seed that fails at the same
configuration; or a diagonal/block-dense arm that also passes once given the
same windowed schedule, which would show the metric structure was never the
binding constraint.

**Weakest part of the evidence.** Single seed at 2 h 11 m per run, and no
ablation separating Amendment 1, Amendment 2, the windowed schedule, and the
full-joint metric. The plan's own pre-mortem flagged uncalibrated λ as the
weakest point and that remains true — λ=0.1 was never tested against
alternatives on this target.

**Failure classification if a later seed fails.** That would be a
tuning-candidate failure, not a route or harness failure: the closed-form
checks would still hold, so the correct response is a λ sweep and multi-seed
characterization rather than abandoning the route.
