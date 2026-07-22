# Phase 8 Owner-Amendment Proposal Review

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Reviewed file:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase8-owner-decision-amendment-proposal-2026-07-14.md`

Reviewer: fresh bounded Codex substitute after the one-path Claude review
returned no substantive output while a tiny Claude health probe passed

## Iteration 1

`VERDICT: REVISE`

Seven material defects were found:

1. an FD-step-scaled diagnostic was mislabeled as HMC energy error and was too
   permissive;
2. the frozen leaderboard seeds were also proposed for tuning;
3. the candidate graph allowed result-dependent combinations and lacked exact
   comparator edges;
4. `N=32` did not exercise the then-proposed `256/512` chunks;
5. the seed-level estimands and fixed-data sampling unit were incomplete;
6. the plug-in formula did not establish the claimed `80%` power; and
7. proposal approval ambiguously authorized a runtime without an exact command,
   node count, callable count, or aggregate budget.

## Iteration 2

`VERDICT: REVISE`

The first repairs closed the original seven findings. Four executable-definition
defects remained: missing candidate-edge formulas, a missing simultaneous value-
interval decision, an unspecified Philox sampling assumption, and a malformed
sign-reversal rule.

## Iteration 3

`VERDICT: REVISE`

The edge, sampling, and sign definitions were repaired. Three final consistency
defects remained: strict versus non-strict value containment, stale conflicting
sign text, and an overclaim that a common global oracle scale prevents a strong
coordinate from masking weak-coordinate relative error.

## Final Repair And Verdict

The proposal now:

- uses strict value-interval containment;
- defines sign reversal from the deterministic Kalman component and the implied
  Contract E mean-gradient interval;
- labels the gradient metric as a center-local, global-oracle-scale first-order
  contribution metric and exposes its weak-coordinate limitation;
- freezes exact staged nodes/edges, disjoint lower-rung/calibration/audit seeds,
  seed-level estimands, interval assumptions, node cap, and no-expansion rules;
- removes the unsupported power claim; and
- authorizes no command before a separately reviewed exact harness subplan.

No material findings remain.

`VERDICT: AGREE`
