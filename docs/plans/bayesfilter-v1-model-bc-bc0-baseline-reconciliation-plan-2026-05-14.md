# BayesFilter V1 Model B/C BC0 Baseline Reconciliation Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC0 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Convert the completed V1 Model B/C evidence into one current matrix before
running wider tests.  This is an evidence-indexing phase, not a new numerical
experiment.

This plan is a planning artifact.  During this review pass it should not update
the reset memo or rewrite prior conclusions.  Reset-memo updates occur only
when BC0 is executed.

## Scope

Models:
- Model B: smooth nonlinear accumulation.
- Model C: autonomous nonlinear growth with default structural fixed-support
  score handling.

Filters:
- SVD cubature.
- SVD-UKF.
- SVD-CUT4.

Evidence columns:
- value status;
- score status;
- branch label;
- fixed-support and structural-null diagnostics;
- HMC status;
- GPU/XLA status;
- reference status;
- Hessian status;
- current claim label: `certified`, `diagnostic`, `blocked`, or `deferred`.

## Entry Gate

BC0 may start after the V1 master execution summary exists:

```text
docs/plans/bayesfilter-v1-master-program-execution-summary-2026-05-14.md
```

## Execution Steps

1. Read P1-P8 result artifacts from the completed V1 master program.
2. Read current nonlinear B/C tests and benchmark output schemas.
3. Populate one row for each model/filter cell.
4. Mark every missing evidence item as `blocked`, `deferred`, or
   `diagnostic`; do not leave unknown status cells.
5. Verify that default Model C score rows cite the structural fixed-support
   branch with `allow_fixed_null_support=True`.
6. Write the BC0 result matrix and update the V1 reset memo.

## Primary Gate

BC0 passes only if no B/C/filter cell has unknown status.

## Veto Diagnostics

Stop and ask for direction if:
- default Model C score status omits the structural fixed-support branch;
- dense one-step projection is described as exact full nonlinear likelihood;
- a tiny HMC smoke is described as convergence evidence;
- a tiny GPU/XLA shape is described as broad speedup evidence;
- nonlinear Hessians are promoted without a named consumer.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc0-baseline-matrix-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

## What This Phase Will Not Conclude

- No new B/C numerical stability claim.
- No HMC convergence claim.
- No GPU performance claim.
- No exact nonlinear likelihood claim for Models B-C.

## Continuation Rule

Continue to BC1 only if BC0 passes and the result identifies the exact
currently accepted boxes, branches, and blocked/deferred cells.
