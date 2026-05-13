# BayesFilter V1 P5 Hessian Consumer Assessment Result

## Date

2026-05-14

## Governing Plan

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
docs/plans/bayesfilter-v1-p5-hessian-consumer-assessment-plan-2026-05-14.md
```

## Phase Scope

P5 closes R6 by deciding whether BayesFilter V1 needs production nonlinear
Hessians for SVD cubature, SVD-UKF, or SVD-CUT4 now.  This phase is a consumer
gate.  It does not implement nonlinear Hessian code unless a concrete consumer
is named or P4 shows score-only workflows are insufficient.

## Plan Tightening

No plan edit was needed.  The P5 entry gate is already precise:

- start implementation only if a concrete nonlinear Hessian consumer is named;
- otherwise keep nonlinear Hessians explicitly deferred.

The only tightening applied in this result is interpretive: ordinary HMC is not
a Hessian consumer because its transition dynamics need a value and score, not
a full observed-information matrix.

## Independent Audit

Audit question:

```text
Does any current V1 requirement justify production nonlinear Hessians?
```

Audit findings:

- P1 records Hessian status for every nonlinear model/backend cell and marks
  production nonlinear Hessians as deferred.
- P2 and P3 certify nonlinear value/score branch behavior at their stated
  local diagnostic scope, not second-order behavior.
- P4 selected Model B with SVD-CUT4 analytic score as the first nonlinear HMC
  target and completed only a tiny CPU smoke.  That smoke did not require a
  Hessian and did not claim convergence.
- No current V1 plan names Newton optimization, trust-region optimization,
  Laplace approximation, Riemannian HMC, curvature-aware HMC, or nonlinear
  observed-information diagnostics as an immediate consumer.
- The only nonlinear Hessian-like SVD-CUT4 path remains
  `bayesfilter.testing.tf_svd_cut_autodiff_oracle`, which is testing-only and
  must not be promoted to production API.
- Production nonlinear derivative results already return `hessian=None` with
  an explicit Hessian-deferred status.

Veto diagnostics checked:

| Veto | Result |
| --- | --- |
| Hessian code starts before a consumer is named | clear; no Hessian code added |
| Testing-only autodiff exposed as production API | clear; oracle remains in testing namespace |
| SVD/eigen Hessian claims ignore branch gaps or structural null contracts | clear; no such claims added |
| Memory and tensor-shape costs omitted for an implementation plan | not applicable; no implementation plan is justified |

## Decision

Nonlinear Hessians remain deferred for BayesFilter V1 production.

No P5 Hessian implementation subplan is opened.  The V1 production contract for
nonlinear SVD sigma-point filters remains:

- value: implemented and tested at documented model/backend scope;
- score: implemented and tested at documented branch scope;
- Hessian: explicitly deferred unless a future phase names a concrete consumer.

## Validation

Documentation-only checks:

```bash
git diff --check
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml')); print('source_map ok')"
```

Result:

- whitespace check passes;
- source-map YAML parses;
- no production code or tests are changed by P5.

## Gate Result

P5 primary gate passes by the deferral branch:

```text
No concrete nonlinear Hessian consumer is named, and Hessian work remains
explicitly deferred.
```

## Next Phase Justification

P6 is still justified only as an optional diagnostic phase.  If executed, GPU
or CUDA visibility commands must use escalated sandbox permissions, and any
speed claim must be limited to the tested shapes.  P6 must not change
production behavior.
