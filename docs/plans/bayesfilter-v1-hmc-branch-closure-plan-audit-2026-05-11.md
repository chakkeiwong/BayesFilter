# Audit: BayesFilter v1 HMC And Branch-diagnostic Closure Plan

## Date

2026-05-11

## Scope

This audit reviews:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-plan-2026-05-11.md
```

It is written from the perspective of another developer before execution.  The
governing reset memo remains:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

## Decision

Approved for scoped execution after tightening.

The plan is complete enough to run automatically because it has explicit
primary criteria, veto diagnostics, lane boundaries, and phase ordering.
Execution must stay in the BayesFilter v1 external-compatibility lane.

## Required Tightening

1. Do not add a public QR score-only API in this pass.
   The current QR derivative implementation propagates first- and second-order
   state, covariance, factor, innovation, gain, and update derivatives together.
   A public score-only analytic API would be a real refactor, not a small
   phase-local change.

2. Use QR value plus TensorFlow autodiff gradient as the first HMC sampler path.
   The analytic QR score/Hessian should validate parity and curvature, but the
   sampler log-prob must not force Hessian materialization.

3. Require benchmark rows to materialize the tensors they claim to measure.
   A row that returns only `log_likelihood` from the full derivative wrapper can
   understate the practical cost of using score and Hessian.

4. Keep SVD-CUT branch-frequency evidence diagnostic-only.
   A smooth tiny-box sweep is not enough to promote SVD-CUT HMC.  It can only
   justify a future target-specific SVD-CUT HMC plan.

5. Keep GPU/XLA claims narrow.
   Non-escalated GPU failures are sandbox evidence only, and no QR derivative
   XLA/GPU run should occur unless CPU diagnostics identify a worthwhile shape.

6. Do not stage out-of-lane files.
   The shared monograph reset memo, structural SVD/SGU files, MacroFinance,
   DSGE, `Zone.Identifier` sidecars, and local images are not part of this lane.

## Missing-point Check

The plan covers:

- lane and worktree audit;
- QR derivative memory diagnosis;
- first fixed LGSSM HMC target contract;
- sampler smoke without convergence claim;
- SVD-CUT branch-frequency diagnostics;
- GPU/XLA decision review;
- CI/runtime tier review;
- result artifact, reset-memo update, source-map update, and commit boundary.

No additional blocker was found.

## Execution Guardrails

Proceed phase by phase only while:

- all edits are BayesFilter-local v1-lane files;
- primary criteria pass;
- veto diagnostics remain inactive;
- HMC diagnostics stay target-specific;
- SVD-CUT HMC remains blocked unless a separate target-specific plan is written;
- optional MacroFinance and DSGE evidence stays read-only and out of this pass.

## Audit Result

The plan may proceed automatically with the tightening above.  If execution
would require external client edits, shared reset-memo staging, or a production
DSGE/MacroFinance dependency, stop and ask for direction.
