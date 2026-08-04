# SVX-ZC Monograph Fixed-Branch Admission Result

Date: 2026-07-31
Plan: `docs/plans/bayesfilter-svx-zc-monograph-fixed-branch-admission-plan-2026-07-31.md`

## Decision

`SVX-ZC` remains blocked for NeuTra, but the blocker is numerical fixed-branch
admission, not a requirement to reproduce the author implementation. Under
`docs/main.tex`, the route is correctly described as a BayesFilter fixed-branch
TT/KR approximation with `extension_or_invention` classification.

The fresh CPU ladder wrote
`docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt03/result.json`.
No rank passed all hard vetoes.

| Rank | Hard result | Max residual | Max condition | Dense value gap/observation | Same-scalar FD |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | blocked rank-saturation | 0.0663071 | 1.00 | 0.136531 | pass |
| 2 | blocked rank-saturation | 0.0566693 | 4.28 | 0.103106 | pass |
| 4 | blocked rank-saturation | 0.0564391 | 7.82e3 | 0.100878 | pass |
| 6 | blocked rank-saturation | 0.0564383 | 2.69e4 | 0.100853 | pass |

The rank-saturation veto was declared as residual `<=1e-8`; every candidate
missed it by roughly four to seven orders of magnitude. Positivity, finite
value/score, affine coordinate/Jacobian consistency, carried marginal closure,
condition ceilings, branch identity, decreasing FD windows, and the absence of
a retained tensor-product grid passed. Dense-reference gaps are descriptive
quality evidence, not a source-route veto or a claim of exactness.

Attempt 02 is retained as a harness failure because it double-transformed the
observations for the dense reference. Attempt 03 corrected that input while
preserving the candidate route and scope.

## Registry Change

`bayesfilter/testing/neutra_model_registry_tf.py` now records:

```text
SVX-ZC -> TARGET_BLOCKED_FILTER_ADMISSION
reason: monograph fixed-branch TT/KR candidate fails rank-saturation residual gate
reentry: fixed-branch numerical admission
```

The old source-route mismatch text remains in historical plans and artifacts;
it is not the active interpretation.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Failed rank-saturation residual for ranks 1, 2, 4, and 6 |
| Statistically supported ranking | None; no stochastic ranking or uncertainty analysis |
| Descriptive-only differences | Dense gaps, residuals, condition numbers, and route values |
| Default readiness | Not assessed |
| Next evidence needed | A target-specific fixed-branch repair that lowers residual while preserving all other gates |

## Nonclaims

No author-source-faithfulness, exact filtering, posterior correctness, NeuTra
training quality, HMC convergence, superiority, leaderboard readiness, or
default-readiness claim is made.

## Post-Run Red Team

The strongest alternative explanation is insufficient fixed rank/basis/fit
capacity rather than a wrong monograph route. The near-plateau from ranks 2 to
6 and the large residual support that explanation, but do not prove it. The
next discriminating artifact is a bounded degree/quadrature/rank repair ladder
with the same data and independent reference; relaxing the residual veto after
seeing these results would invalidate the admission claim.
