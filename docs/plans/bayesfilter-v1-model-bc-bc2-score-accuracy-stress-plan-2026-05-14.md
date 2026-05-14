# BayesFilter V1 Model B/C BC2 Score Accuracy Stress Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC2 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Quantify analytic score accuracy for Models B-C across the stable BC1 boxes.

## Entry Gate

BC2 may start only after BC1 identifies stable or explicitly narrowed score
boxes for the target model/filter cells.

## Evidence Contract

Question:
- Do analytic scores agree with centered finite differences for the same
  implemented value filter across the BC1 stable boxes?

Baseline:
- BC1 stable/narrowed boxes and current analytic score tests.

Primary criterion:
- Residuals are below model/filter-specific tolerances on the declared stable
  boxes, or rows are blocked with exact failure labels.

Veto diagnostics:
- finite differences cross active branch changes;
- compiled/eager mismatch is ignored for compiled paths;
- score status is promoted when only value status passed.

What will not be concluded:
- HMC convergence, Hessian correctness, or exact full nonlinear likelihood.

Artifact:
- Score residual table and BC2 result file.

## Predeclared Tolerance Rule

Before running BC2, write a tolerance table in the execution note or result
draft.  For each model/filter row, the table must state:
- absolute residual tolerance;
- relative residual tolerance;
- finite-difference step ladder;
- parameter scaling rule;
- source of the tolerance: P1/P2 baseline, machine precision, local curvature,
  or prior accepted finite-difference residual.

Maximum absolute residual and maximum relative residual are pass/fail metrics.
Step-sensitivity plots, per-parameter residual distributions, and runtime are
explanatory unless they are explicitly predeclared as veto diagnostics.

If the tolerance table cannot be justified before seeing new residuals, stop
and write a BC2 tolerance subplan instead of running the stress tests.

## Execution Steps

1. For each BC1 stable cell, choose centered finite-difference step ladders.
2. Compare analytic score to finite differences for deterministic and seeded
   rows.
3. Exclude or label rows where branch changes make finite differences
   invalid.
4. Check eager/compiled parity where practical and relevant.
5. Record max absolute residual, relative residual, finite-difference step,
   branch label, support residuals, and structural-null diagnostics.
6. Write the BC2 result artifact and update the V1 reset memo.

## Primary Gate

BC2 passes only if every tested row either satisfies its tolerance under stable
branch diagnostics or receives an exact blocker label.

## Veto Diagnostics

Stop and ask for direction if:
- branch changes are included in finite-difference residuals without labels;
- Model C structural fixed-support diagnostics are omitted;
- compiled/eager mismatch is treated as harmless without explanation;
- finite-difference tolerances are loosened after seeing failures without a
  stated numerical reason.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc2-score-stress-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional, only if needed:

```text
tests/test_nonlinear_sigma_point_scores_tf.py
docs/benchmarks/bayesfilter-v1-model-bc-score-stress-*.json
```

## Continuation Rule

Continue to BC3 only for cells whose value and score status is stable or whose
blocker is explicit.  Continue to BC5 only for cells that also pass BC3.
