# BayesFilter V1 P4 Nonlinear HMC Target Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P4 / R5 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Entry Gate

P4 may start only after:
- P1 has certified score status for the target backend/model cells;
- P2 has identified a stable branch box;
- P3 has benchmark metadata for the intended target.

## Motivation

HMC readiness is target-specific.  A filter with a score is not automatically
an HMC-ready posterior.  This phase chooses one nonlinear target and runs at
most a tiny CPU smoke after value, score, branch, and compiled/eager gates are
met.

## Default Target

Model B is the default first nonlinear HMC candidate because it is smooth and
does not require the structural fixed-support branch.  Default Model C may be
considered only if P2 shows a stable structural branch box.

## Primary Gate

P4 passes if:
- a named target has finite value and score on the target box;
- branch diagnostics remain stable on the target box;
- compiled/eager parity is checked if compiled execution is used;
- any tiny HMC smoke reports finite chains and explicit diagnostics;
- convergence is not claimed unless convergence diagnostics are actually run.

## Veto Diagnostics

Stop and ask for direction if:
- HMC starts before finite-score and branch gates pass;
- default Model C is selected before structural branch stability is shown;
- GPU is used without escalated permissions;
- tiny smoke output is described as convergence certification;
- Hessian or mass-matrix claims are introduced without their own gate.

## Expected Artifacts

```text
docs/plans/bayesfilter-v1-p4-nonlinear-hmc-target-result-2026-05-14.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional code/tests must be opt-in and must not slow default CI.

## Continuation Rule

Continue to P5 only if P4 identifies a real need for Hessians or curvature.
Otherwise keep Hessian work deferred and continue only to optional GPU/XLA or
reference phases when justified.
