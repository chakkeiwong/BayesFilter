# Contract E--TP Phase 5 Progressive Generalized-SV Repair Result

metadata_date: 2026-07-15
status: NEGATIVE_RESULT_FEATURE_FAMILY_INSUFFICIENT

## Question And Verdict

Do multiple bounded target-continuation marks and a distribution-aware fixed
overcomplete chart repair the generalized-SV `T=10` score instability?

No. The implementation is internally correct relative to its finite scalar:
same-scalar FD passes, all admitted charts are strictly positive and full rank,
and feature residuals are at roundoff. The quantity computed remains different
from the dense target filter score, especially in the persistence direction.

## Repair Loop

1. Progressive marks `(1,4,9)` with a square-equivalent chart retained exact
   feature constraints but left a `24.0%` worst score gap.
2. Max-min overcomplete references worsened downstream recursion even with only
   the old full-prefix feature (`82.7%` gap). Exact feature matching does not
   determine unmatched downstream functionals.
3. Pure weighted-quantile/Voronoi references at 8, 12, and 16 anchors failed
   strict positivity; no clipping or weight floor was introduced.
4. Convex chi-square and KL information projections were attempted as offline
   preparation repairs. The former reached a boundary optimum; the latter did
   not meet the exact equality gate. Neither failed artifact was admitted.
5. A constructive positive-basis plus quantile analytic-center reference
   passed all chart gates. Its worst score gaps were `13.7%` at teacher order 25
   and `9.53%` at order 41 with eight quantile anchors.
6. Increasing only quantile-anchor capacity to 12 at order 41 worsened the gap
   to `32.9%`.

## Root-Cause Classification

The rejected hypotheses are derivative wiring, target/proposal conflation,
late-step-only error, and too few overcomplete anchors. Per-time score
increments first show sustained `gamma` drift at increments 3--5 and later
partial cancellation. The remaining cause is feature-family insufficiency:
the finite continuation marks constrain selected future likelihood integrals,
but they do not determine the recursively required distribution or all of its
parameter tangents.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| reject progressive generalized-SV family | adjacent orders retain material score gaps; capacity increase worsens | engineering vetoes pass | no cross-method equivalence margin | close row at `T=10` and continue other rows | not rejection of shared scalar core |
| preserve order-41 eight-anchor result as diagnostic | smallest observed gap in repaired family | refinement sensitivity blocks promotion | center-only | retain artifact for future distributional-basis work | no default/full-horizon claim |
| do not run generalized `T=100` | short-prefix repair did not converge | continuation gate for this row | materially different feature needed | Phase 6 may certify comparator separately but paired row is negative | not a campaign stop |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | admitted finite programs pass FD/chart/reference validity |
| Statistically supported ranking | none |
| Descriptive-only differences | all score/value gaps |
| Default readiness | false |
| Next evidence | new distributional feature family with predeclared basis/refinement |

## Post-Run Red Team

The strongest alternative explanation is remaining teacher/grid resolution.
Order 25 to 41 improves the eight-anchor score, but the 12-anchor refinement
worsens sharply and the square baseline was already order-unstable. Therefore
the current evidence cannot select a converged numerical result. A future
candidate would overturn this conclusion only by passing adjacent teacher,
continuation-grid, and capacity refinements with stable per-time score
increments.
