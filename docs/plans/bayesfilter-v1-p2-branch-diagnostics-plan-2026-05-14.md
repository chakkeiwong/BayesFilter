# BayesFilter V1 P2 Branch Diagnostics Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P2 / R3 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Entry Gate

P2 may start only after P1 produces a derivative-validation matrix with no
unknown score-status cells for Models A-C across SVD cubature, SVD-UKF, and
SVD-CUT4.

## Motivation

Score parity at a few points is not enough for HMC or benchmark promotion.
The next question is whether practical parameter boxes remain on a valid
branch: finite scores, no active floors, no weak active spectral gaps, and no
moving structural-null blocks.

## Scope

Targets:
- Model B nonlinear accumulation;
- smooth-phase Model C as a comparison/control;
- default Model C under `allow_fixed_null_support=True`.

Backends:
- SVD cubature;
- SVD-UKF;
- SVD-CUT4.

## Hypotheses

H-P2.1:
Model B has a practical parameter box where all selected backends have finite
value and score with no active floors or weak active gaps.

H-P2.2:
Default Model C has a reportable structural fixed-support branch box, or a
clear blocker label explaining why it should not proceed to HMC/benchmark
promotion.

H-P2.3:
Branch summaries can report failure labels without hiding active-floor,
weak-gap, nonfinite, or structural-null failures.

## Required Diagnostics

Each grid summary must report:
- total points and ok fraction;
- backend and model;
- parameter ranges;
- active floor count;
- weak spectral gap count;
- nonfinite value/score count;
- structural null count and residuals where applicable;
- fixed-null derivative residual where applicable;
- deterministic residual;
- support residual;
- representative failure labels.

## Primary Gate

P2 passes if:
- at least one practical Model B box passes for all selected backends;
- default Model C has either a passing structural branch box or a documented
  blocker;
- failure labels remain visible and are not converted into success.

## Veto Diagnostics

Stop and ask for direction if:
- default Model C is tested without the structural fixed-support option;
- active floors or weak gaps are treated as acceptable score rows;
- branch summaries omit failure labels;
- the phase requires GPU, HMC, Hessian, MacroFinance, or DSGE work.

## Expected Artifacts

```text
docs/plans/bayesfilter-v1-p2-branch-diagnostics-result-2026-05-14.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional if the current diagnostic helper is insufficient:

```text
bayesfilter/testing/nonlinear_diagnostics_tf.py
tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py
```

## Continuation Rule

Continue to P3 only if P2 identifies a benchmarkable nonlinear score branch
box and records any Model C limitations explicitly.
