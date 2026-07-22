# Phase 5 Subplan Review, Iteration 1

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Reviewer: fresh bounded Codex substitute reviewer

Reviewed path:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase5-canonical-graph-subplan-2026-07-14.md`

## Material Findings

1. The finite LGSSM scalar was under-specified: parameter ordering/transforms,
   time-zero convention, logit signs, batch aggregation, and exact post-reset
   uniform log weights were not frozen.
2. No non-arbitrary derivative pass criterion prevented an inconclusive local
   comparison from advancing.
3. The identity-basis `p` direction axis and score aggregation were not
   operationally specified.
4. The initialization test incorrectly implied that every LGSSM parameter must
   affect stationary initialization.

`VERDICT: REVISE`

## Visible Repair

The same subplan now freezes:

- the physical five-parameter LGSSM, matrices, transition-first zero-based time
  convention, corrected-logit equation/signs, sum-over-time and mean-over-batch
  objective, and exact `-log(N)` post-reset weights;
- a solve-based, no-jitter/no-explicit-inverse linear-Gaussian LEDH map and
  explicit transport-geometry/epsilon-start branch semantics;
- the final `p=5` direction axis through every tangent state and exact score
  aggregation;
- the exact stationary-initialization dependency map; and
- exact outer-wiring microcertificates plus a predeclared `0 ULP`
  same-primal-core analytic-JVP versus forward-autodiff gate.

An inconclusive or nonzero-ULP derivative result now blocks canonical gradient
certification. The heuristic FD screen cannot override that block.
