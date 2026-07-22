# Phase 5 Fixture Branch-Stability Repair

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `V2_FROZEN_BEFORE_V2_CANDIDATE_OUTPUT`

## Failure

The original fixture passed the exact same-core manual-JVP gate and every chart
check, but its same-callable CPU-XLA FD artifact failed the predeclared branch
identity check. At the active reset `batch=1,time=1`, the largest two scaled
geometry entries at center were approximately `0.6327348891` and
`0.6320082916`, a margin of only `7.27e-4`. The largest frozen FD step is
`1/128 = 7.8125e-3`, and several endpoints changed the geometry maximum index.

This is a fixture failure for branch-stable FD evidence. It is not a failure of
the exact analytic JVP, not a chart failure, and not evidence against Contract
E. The failed artifact remains preserved at
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase5/cpu-xla-same-callable-certificate-v1.json`.

## Repair Contract

Before evaluating any repaired-fixture objective or score, branch-only screens
tested positive dyadic perturbations to
`transition_noise[1][1][3][2]`, originally `1/8`. The selection criteria were:

- every declared active branch mask identical at center and all 30 frozen FD
  endpoints;
- every chart valid at center and all endpoints; and
- choose the smallest tested positive dyadic perturbation that passes.

The tested deltas, in order, were `1/64`, `1/32`, and `1/16`. All passed. The
first therefore binds, changing the prepared noise entry to `9/64`. No
candidate objective, likelihood, score, or FD derivative was read during this
selection.

The repaired fixture is frozen at
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-tiny-fixture-freeze-v2-2026-07-14.json`.
The original fixture is not modified or reinterpreted.

## Nonclaims

This repair establishes only that the tiny FD fixture is not knowingly placed
across a declared nonsmooth branch at the frozen steps. It does not establish a
general branch margin, derivative error bound, Kalman equivalence, production
schedule/chunk/ridge adequacy, HMC readiness, or leaderboard readiness.
