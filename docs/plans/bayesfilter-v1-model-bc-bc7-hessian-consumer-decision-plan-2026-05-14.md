# BayesFilter V1 Model B/C BC7 Hessian Consumer Decision Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC7 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Decide whether any BC1-BC6 result names a concrete nonlinear Hessian consumer.

## Entry Gate

BC7 may start after BC1-BC6 are complete, blocked, or explicitly deferred.

## Evidence Contract

Question:
- Does V1 now have a named consumer that justifies nonlinear Hessian design?

Baseline:
- P5 Hessian consumer assessment and BC1-BC6 artifacts.

Primary criterion:
- Hessian status is explicit and non-promotional.

Veto diagnostics:
- Hessian implementation starts before a consumer is named;
- testing-only autodiff oracle is exposed as production API;
- second-order SVD branch issues are ignored.

What will not be concluded:
- Nonlinear Hessian correctness or production readiness.

Artifact:
- BC7 decision/result file and optional Hessian implementation subplan only if
  justified.

## Candidate Consumers

Name a consumer only if there is an immediate plan or claim requiring it:
- Newton or trust-region optimization;
- Laplace approximation;
- observed information;
- Riemannian or curvature-aware HMC;
- curvature diagnostics with explicit public/user value.

## Execution Steps

1. Review BC1-BC6 for any named Hessian consumer.
2. If no consumer exists, keep nonlinear Hessians deferred.
3. If a consumer exists, write a separate implementation plan with tensor
   shape, memory, branch, and second-derivative provider contracts.
4. Keep testing-only autodiff oracles in the testing namespace.
5. Write the BC7 result artifact and update the V1 reset memo.

## Primary Gate

BC7 passes if every Model B/C/filter Hessian status is explicit:
`deferred_no_consumer`, `testing_oracle_only`, `linear_qr_only`,
`requires_named_consumer`, or a consumer-specific subplan path.

## Veto Diagnostics

Stop and ask for direction if:
- production Hessian code is proposed inside BC7;
- a testing oracle is described as a production API;
- Hessian status is inferred from first-order score success;
- HMC/NUTS diagnostics are used to imply Hessian readiness.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc7-hessian-consumer-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Optional only if justified:

```text
docs/plans/bayesfilter-v1-model-bc-hessian-implementation-subplan-2026-05-14.md
```

## Continuation Rule

Continue to BC8 when Hessian status is explicit.  If a consumer is named, stop
for human direction before implementing nonlinear Hessians.
