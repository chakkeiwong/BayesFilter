# SVX-ZC Score Capacity Validation Reset Memo

Date: 2026-08-02

The active question is whether the score is internally consistent with the
selected finite likelihood and whether it is sensitive to nearby degree, rank,
and quadrature settings. The value-only capacity artifact nominated
`(degree=10, rank=2, order=25)` but did not test score behavior.

The next run is governed by:
`docs/plans/bayesfilter-svx-zc-score-capacity-validation-plan-2026-08-02.md`.

Five frozen score cells are planned: `(10,2,25)`, `(12,2,25)`, `(10,4,25)`,
`(10,2,29)`, and `(10,2,33)`. Each uses the same transformed SV data, UKF
initialization, total autodiff score, fixed-branch finite differences, and
`h=(1e-2,3e-3,1e-3,3e-4)` ladder.

The finite-difference policy can establish derivative consistency for the
tested finite programs. It cannot establish exact score correctness or infer
score convergence from value-prefix stability. Cross-capacity score movement
is descriptive until a separate score-capacity criterion is reviewed.

## Terminal run state

The five-cell run completed in
`docs/plans/artifacts/bayesfilter-svx-zc-score-capacity-validation-20260802/attempt01/`.
The center nominee `(10,2,25)`, rank neighbor `(10,4,25)`, and order-29 cell
passed same-scalar derivative consistency. Order 33 was blocked by one changed
`log_scale_shift_index` branch under `log_beta + 0.01`; degree 12 was blocked
by multiple perturbation branch changes and no valid `log_beta` finite-
difference window. Degree 12 nevertheless shows large descriptive score
movement relative to the nominee, so the value-prefix nomination must not be
treated as score convergence.
