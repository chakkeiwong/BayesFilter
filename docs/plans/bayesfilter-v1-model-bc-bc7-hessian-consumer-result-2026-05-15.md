# BayesFilter V1 Model B/C BC7 Hessian Consumer Result

## Date

2026-05-15

## Controlling Documents

Master program:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

Phase plan:

```text
docs/plans/bayesfilter-v1-model-bc-bc7-hessian-consumer-decision-plan-2026-05-14.md
```

## Plan Tightening And Drift Check

No plan change was required.  BC7 is a decision phase only.  It must not start
nonlinear Hessian implementation unless BC1-BC6 named a concrete consumer.

No drift found:

- no nonlinear Hessian production code was written;
- testing-only autodiff oracles remain testing-only;
- first-order score success was not interpreted as Hessian readiness;
- HMC and GPU diagnostics were not used to imply Hessian readiness.

## Independent Plan Audit

BC1-BC6 were reviewed for candidate Hessian consumers:

- Newton or trust-region optimization: not named by any BC phase;
- Laplace approximation: not named by any BC phase;
- observed information: not named by any BC phase;
- Riemannian or curvature-aware HMC: not named by BC5;
- curvature diagnostics with explicit public/user value: not named by BC6.

Existing linear QR Hessian work and testing-only Hessian oracles are useful
background evidence, but they are outside the nonlinear Model B/C Hessian
implementation decision.

## Decision Matrix

| Model | Filter | Hessian status | Rationale |
| --- | --- | --- | --- |
| Model B nonlinear accumulation | SVD cubature | `deferred_no_consumer` | BC1-BC6 require first-order score and value diagnostics only. |
| Model B nonlinear accumulation | SVD-UKF | `deferred_no_consumer` | No Newton, Laplace, observed-information, or curvature-HMC consumer was named. |
| Model B nonlinear accumulation | SVD-CUT4 | `deferred_no_consumer` | BC5 HMC readiness used analytic scores only and made no curvature claim. |
| Model C autonomous nonlinear growth | SVD cubature | `deferred_no_consumer` | Structural fixed-support first-order score is sufficient for current claims. |
| Model C autonomous nonlinear growth | SVD-UKF | `deferred_no_consumer` | The `T=32` moving structural-null blocker is a first-order branch boundary, not a Hessian consumer. |
| Model C autonomous nonlinear growth | SVD-CUT4 | `deferred_no_consumer` | No second-order consumer or public claim was named. |

## Veto Audit

No veto diagnostic fired:

- no Hessian code was proposed or implemented;
- no testing-only autodiff oracle was promoted to production API;
- Hessian status was not inferred from first-order score success;
- HMC readiness candidates were not promoted to Hessian readiness.

## Interpretation

BC7 passes.  Nonlinear Hessians should remain deferred until a concrete
consumer exists.  If a future phase asks for Newton optimization, Laplace
approximation, observed information, Riemannian HMC, or a curvature diagnostic,
that phase should first write a separate implementation plan covering tensor
shapes, memory scaling, second-derivative providers, and SVD branch contracts.

## Continuation Decision

BC8 is justified.  The final consolidation should report nonlinear Hessians as
`deferred_no_consumer` for all Model B/C/filter cells.
