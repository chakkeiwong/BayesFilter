# Audit: BayesFilter v1 HMC Readiness And Diagnostic Gap Closure Plan

## Date

2026-05-11

## Scope

This audit reviews:

```text
docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-plan-2026-05-11.md
```

The governing reset memo is:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

## Decision

Approved for scoped execution after tightening.

The plan has sufficient lane boundaries, hypotheses, primary criteria, veto
diagnostics, and artifact requirements to run automatically phase by phase.

## Required Tightening

1. Private dense QR score-only diagnostics are allowed in this phase because
   the plan needs to separate first-order derivative cost from Hessian
   materialization.

2. Public score-only API promotion is out of scope.  Dense and masked
   score-only semantics should be reviewed together in a later API-freeze pass.

3. The HMC target may use QR value plus TensorFlow autodiff as the sampler
   gradient path.  Full QR score/Hessian remains the parity and curvature
   diagnostic.

4. Benchmark rows must materialize the tensors they claim to measure:
   score-only rows must materialize log likelihood and score, while
   score/Hessian rows must materialize log likelihood, score, and Hessian.

5. SVD-CUT branch-frequency evidence remains diagnostic-only.  A smooth tiny
   parameter box can motivate a later SVD-CUT HMC plan, but it cannot promote
   SVD-CUT HMC readiness in this pass.

6. GPU/XLA derivative work remains deferred unless CPU evidence identifies a
   matching shape worth testing with escalated GPU visibility.

7. MacroFinance and DSGE remain read-only external targets.  This pass must not
   edit or import external project internals in production code.

## Missing-point Check

The tightened plan covers:

- lane and dirty-worktree audit;
- private dense QR score-only diagnostic cost isolation;
- score-only versus score/Hessian cost decomposition;
- state/observation score-envelope diagnostics;
- first target-specific QR HMC smoke;
- SVD-CUT branch-frequency diagnostics;
- CI tier containment;
- reset memo, source map, result artifact, and scoped commit boundary.

No additional blocker was found.

## Audit Result

Proceed automatically while primary criteria pass and veto diagnostics remain
inactive.  Stop if execution requires MacroFinance/DSGE edits, shared
monograph or structural-lane staging, public score-only promotion, broad HMC
claims, or unapproved GPU/CUDA execution.
