# BayesFilter V1 P1 Derivative-Validation Matrix Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P1 / R2 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Motivation

The nonlinear model suite now has Models A-C, analytic first-order derivative
providers for Models B-C, and a structural fixed-support score branch for
default Model C.  The evidence is real, but it is spread across older plans and
tests.  Before wider branch grids, HMC, GPU/XLA, or external integration, V1
needs a single matrix that says exactly which model/backend derivative claims
are certified, diagnostic, deferred, or blocked.

This phase is a validation and evidence-indexing phase.  It is not a request to
implement nonlinear Hessians.

## Scope

Models:
- Model A: affine Gaussian structural oracle;
- Model B: nonlinear accumulation;
- Model C: autonomous nonlinear growth, both smooth-phase control and default
  structural fixed-support law where relevant.

Backends:
- SVD cubature;
- SVD-UKF;
- SVD-CUT4.

Derivative objects:
- value likelihood;
- analytic first-order score;
- score branch and diagnostics;
- Hessian status.

## Hypotheses

H-P1.1:
Every Model A-C/backend cell already has enough value evidence to classify its
implemented value likelihood status.

H-P1.2:
Every Model A-C/backend cell has enough score evidence, or a clear blocker, to
avoid an unknown score-status cell.

H-P1.3:
Hessian status can be recorded without implementing nonlinear Hessians:
`not_required_for_v1_score_first`, `testing_oracle_only`, `linear_qr_only`, or
`requires_named_consumer`.

H-P1.4:
Default Model C score certification is valid only for the structural
fixed-support branch with `allow_fixed_null_support=True`; the old collapsed
smooth branch remains a blocker.

## Required Matrix Columns

For each model/backend row, record:
- model name and fixture constructor;
- backend name;
- value status and evidence;
- score status and evidence;
- score branch label;
- derivative provider;
- finite-difference or exact reference target;
- structural-null diagnostics, if applicable;
- active-floor and weak-gap status;
- compiled/eager parity status, if available;
- Hessian status;
- public-claim status: certified, diagnostic, deferred, or blocked.

## Execution Steps

1. Inventory existing tests and result artifacts.
2. Build the matrix from existing evidence first.
3. Identify missing cells, if any.
4. Add only lightweight tests needed to close explicit missing cells.
5. Run focused nonlinear derivative tests.
6. Run the focused V1 regression if code changes.
7. Write a result artifact with the matrix and interpretation.
8. Update the V1 reset memo with the phase result.

## Primary Gate

P1 passes only if:
- no Model A-C/backend cell has unknown value or score status;
- every certified score cell cites finite-difference, exact affine, or
  documented oracle evidence;
- every Hessian cell has explicit status without implying production support;
- default Model C structural score rows report the structural fixed-support
  branch and the relevant null diagnostics.

## Veto Diagnostics

Stop and ask for direction if:
- a score row is certified without reference evidence;
- default Model C is certified without `allow_fixed_null_support=True`;
- a testing-only autodiff oracle is promoted to production API;
- nonlinear Hessian implementation becomes necessary before a consumer is
  named;
- evidence requires editing MacroFinance, DSGE, Chapter 18b, structural plans,
  or the shared monograph reset memo;
- production NumPy dependency would be introduced.

## Expected Artifacts

```text
docs/plans/bayesfilter-v1-p1-derivative-validation-matrix-result-2026-05-14.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional only if new code/tests are required:

```text
tests/test_nonlinear_sigma_point_scores_tf.py
tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py
```

## Continuation Rule

Continue to P2 only if P1 passes and the result identifies at least one
practical score-certified nonlinear target for branch-grid diagnostics.
