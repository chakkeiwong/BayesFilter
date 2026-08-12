# SVX-ZC Capacity Tuning Reset Memo

Date: 2026-08-01

## Active state

The value-only capacity grid is valid. The active interpretation uses the
user-intended three-significant-digit rule: the first two significant digits
must agree; the third may change.

Active derived artifact:
`docs/plans/artifacts/bayesfilter-svx-zc-capacity-tuning-20260801/reinterpretation-attempt01/result.json`

Active result:
`docs/plans/bayesfilter-svx-zc-capacity-self-convergence-tuning-result-2026-08-01.md`

## Current nomination

The smallest center-point nominee is degree 10, rank 2, order 25. Degree 10
rank 4 and degree 10 rank 6 also pass the prefix rule but have higher stored
coefficient counts.

The source value grid is the 17-cell artifact under `attempt03`. The active
reinterpretation made zero filter, score, or dense-reference calls.

## What is complete

- Generic significant-digit-prefix comparison utility.
- Tests: 18 focused tests pass.
- Value-only SVX-ZC grid: 17 valid cells.
- Rank refinements: all 12 available comparisons prefix-stable.
- Degree-10-to-degree-12 comparisons: all 4 prefix-stable.
- Read-only reinterpretation with no scientific rerun.

## What remains

Final capacity promotion is not complete. Quadrature confirmation and the four
frozen parameter-neighborhood validation points were deferred because the
original strict interpretation found no nominee. They now need to be run for:

- base `(degree=10, rank=2, order=25)`;
- degree neighbor `(12,2,25)`;
- rank neighbor `(10,4,25)`;
- order confirmations for `(10,2,29)` and `(10,2,33)`;
- the same required comparisons at the four frozen validation points.

These are new value evaluations under a new bounded continuation plan. Do not
run score or HMC tuning until the value checks pass.

## Historical artifacts

The original `attempt01`-`attempt03` source artifacts preserve the former
absolute-place interpretation and implementation history. They remain useful
for provenance but are not the active scientific decision under the corrected
rule.

## Nonclaims

There is no final capacity promotion, exact-likelihood accuracy, score
accuracy, HMC validity, posterior agreement, GPU/XLA readiness, production
readiness, or cross-scope transfer claim.
