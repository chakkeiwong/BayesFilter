# BayesFilter V1 P5 Hessian Consumer Assessment Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P5 / R6 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Entry Gate

P5 starts only if a concrete nonlinear Hessian consumer is named, or P4 shows
that score-only workflows are insufficient.

## Motivation

The derivative-validation matrix must record Hessian status, but production
nonlinear Hessian implementation is not a default V1 requirement.  This phase
prevents accidental drift from "we can derive Hessians" to "we should implement
all nonlinear Hessians now."

## Candidate Consumers

Acceptable consumers:
- Newton or trust-region optimization;
- Laplace approximation;
- Riemannian or curvature-aware HMC;
- observed-information diagnostics.

Non-consumers:
- general curiosity;
- HMC with ordinary score-only dynamics;
- benchmark decoration without a downstream use.

## Primary Gate

P5 passes if:
- a consumer is named and an implementation/test plan is justified; or
- no consumer is named and nonlinear Hessians remain explicitly deferred.

## Veto Diagnostics

Stop and ask for direction if:
- nonlinear Hessian code starts before a consumer is named;
- testing-only autodiff is exposed as production API;
- SVD/eigen Hessian claims ignore active floors, weak gaps, structural null
  contracts, or second-derivative provider requirements;
- memory and tensor-shape costs are not stated.

## Expected Artifacts

```text
docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-result-2026-05-14.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Only if a consumer is accepted:

```text
docs/plans/bayesfilter-v1-p5-hessian-implementation-subplan-*.md
```
