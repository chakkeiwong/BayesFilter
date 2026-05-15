# BayesFilter V1 Model B/C BC1 Wider Branch Boxes Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC1 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Test whether Model B/C value and score branch claims survive bounded parameter
boxes wider than the first-rung P2/P3 evidence.

## Entry Gate

BC1 may start only after BC0 passes and records the current B/C/filter evidence
matrix.

## Evidence Contract

Question:
- Which Model B/C/filter cells remain stable over wider bounded parameter
  boxes, and which cells need narrowed boxes or blocker labels?

Baseline:
- BC0 matrix and existing P1-P3 nonlinear branch/value/score evidence.

Primary criterion:
- Every model/filter cell has either a stable box with passing diagnostics or a
  narrowed/blocked box with exact row-level failure labels.

Veto diagnostics:
- active floors, weak gaps, support residuals, structural-null residuals, or
  nonfinite rows are hidden by aggregation;
- default Model C runs without `allow_fixed_null_support=True`;
- failures lack row-level parameter, seed, filter, and branch labels.

What will not be concluded:
- HMC readiness, convergence, exact likelihood quality, or GPU speedup.

Artifact:
- BC1 result file plus any lightweight branch-diagnostic test/artifact updates.

## Parameter Boxes

Model B:
\[
  \rho\in[0.55,0.85],\quad
  \sigma\in[0.15,0.40],\quad
  \beta\in[0.45,1.10].
\]

Model C:
\[
  \sigma_u\in[0.60,1.40],\quad
  \sigma_y\in[0.60,1.40],\quad
  P_{0,x}\in[0.10,0.50].
\]

## Execution Steps

1. Define deterministic grid rows and seeded random box rows.
2. Run value and score branch diagnostics for all three filters.
3. For default Model C, require structural fixed-support evaluation and
   record structural-null residuals.
4. Record branch counts, active floors, weak gaps, nonfinite rows,
   deterministic residuals, support residuals, structural-null residuals, and
   failure labels.
5. If a full box fails, narrow the box only with explicit row-level rationale.
6. Write the BC1 result artifact and update the V1 reset memo.

## Stop And Narrowing Rules

- Predeclare deterministic grid rows and seeded random rows before execution.
- Allow at most one planned narrowed-box proposal per model/filter after the
  full box is evaluated.
- The narrowed box must retain a named scientific or use-case rationale and
  must not be selected solely to remove failing rows.
- If the narrowed box still fails a veto diagnostic, classify the cell as
  `blocked` and stop rather than narrowing again.
- Any additional exploratory rows after a blocker must be labeled
  `diagnostic_only` and must not be used for promotion in BC1.

## Primary Gate

BC1 passes only if each model/filter has a documented stable box or a narrowed
box/blocker with exact diagnostic labels.

## Veto Diagnostics

Stop and ask for direction if:
- Model C score evaluation omits `allow_fixed_null_support=True`;
- failures are aggregated without row-level metadata;
- regularization hides active-floor behavior;
- a branch-grid pass is described as HMC readiness.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc1-branch-boxes-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional, only if needed:

```text
tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py
docs/benchmarks/bayesfilter-v1-model-bc-branch-boxes-*.json
```

## Continuation Rule

Continue to BC2 only for cells with stable or explicitly narrowed score boxes.
Do not run score stress tests across unresolved branch transitions.
