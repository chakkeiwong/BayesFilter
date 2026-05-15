# BayesFilter V1 Model B/C BC4 Reference Decision Result

## Date

2026-05-15

## Governing Plan

```text
docs/plans/bayesfilter-v1-model-bc-bc4-reference-decision-plan-2026-05-14.md
```

## Phase Intent

BC4 decides whether the current Model B/C claims require stronger approximation
references than dense one-step projection and finite-difference diagnostics.

## Plan Tightening

No plan rewrite was required.  BC4 is kept as a decision phase: it does not add
dense multi-step quadrature, high-particle SMC, or new reference dependencies
unless a current claim needs them.

## Independent Audit

As a second-developer audit, BC4 does not conflate comparator types:

- BC1 branch stability is compared against branch diagnostics, not exact
  nonlinear likelihood;
- BC2 score accuracy is compared against centered finite differences of the
  implemented value filter;
- BC3 horizon/noise robustness is a finite value/score/branch envelope;
- dense one-step projection remains diagnostic and is not used as full
  likelihood truth.

## Required Decision Table

| Claim | Current comparator/reference basis | Comparator type | Why sufficient | Out of scope |
| --- | --- | --- | --- | --- |
| BC1 Model B branch-box stability for all three filters | Row-level value and analytic-score branch diagnostics over predeclared deterministic and seeded parameter rows | `diagnostic_only` | The claim is about finite implemented values, finite scores, branch labels, and structural residuals, so exact likelihood is not needed | Full nonlinear likelihood accuracy, HMC convergence, GPU speedup |
| BC1 Model C structural fixed-support branch-box stability for all three filters | Row-level value and analytic-score branch diagnostics with `allow_fixed_null_support=True` | `diagnostic_only` | The claim is a branch-contract claim for structural fixed support; the relevant evidence is structural-null and fixed-null residual metadata | Alternative moving-null semantics, exact nonlinear likelihood |
| BC2 Model B analytic score accuracy | Centered finite differences of the same implemented value filter on BC1 rows | `deterministic_approximation` | Score correctness is tested against deterministic finite differences and branch stability; exact likelihood is not the comparator | Hessian correctness, posterior recovery |
| BC2 Model C analytic score accuracy | Centered finite differences of the same implemented structural fixed-support value filter on BC1 rows | `deterministic_approximation` | The derivative claim is local to the implemented structural fixed-support branch and does not require a full nonlinear reference | Moving-null support derivative theory |
| BC3 Model B horizon/noise envelope | Finite value, finite score, branch diagnostics, support residuals, and deterministic residuals over predeclared panels | `diagnostic_only` | The claim is stability over horizon/noise ladders, not approximation error to an exact likelihood | Exact filter quality, posterior calibration |
| BC3 Model C horizon/noise envelope and SVD-UKF blocker | Same diagnostics, plus exact `blocked_moving_structural_null` labels for selected SVD-UKF rows | `diagnostic_only` | The current claim is an envelope and blocker classification; exact references would not remove the structural-support branch blocker | Redesigning structural support semantics |
| HMC readiness candidate selection for BC5 | BC1-BC3 value/score/branch gates plus future HMC diagnostics | `diagnostic_only` | HMC target readiness first needs finite target and gradient diagnostics; exact nonlinear likelihood would be a separate posterior-recovery claim | HMC convergence and posterior-recovery certification |

## Decision

No stronger reference artifact is required for the current BC1-BC3 claims.
Exact full nonlinear likelihood for Models B-C remains deferred.  A bounded
multi-step quadrature or seeded high-particle SMC reference should be opened
only if a future phase asks for approximation-quality ranking, posterior
recovery, or filter-error calibration rather than value/score/branch
stability.

## Gate Result

BC4 primary gate passes.  Reference status is explicit and non-promotional:

- dense one-step projection is diagnostic only;
- finite differences are deterministic score diagnostics only;
- no Monte Carlo reference is introduced;
- no production dependency is added.

## Veto Diagnostics

| Veto | Status |
| --- | --- |
| Reference work begins without a named current claim | Clear |
| Monte Carlo uncertainty omitted | Clear; no Monte Carlo reference used |
| Exact/deterministic/Monte Carlo labels conflated | Clear |
| Reference dependency enters production imports | Clear |

## Interpretation

The Model B/C thorough-testing campaign can continue without exact nonlinear
references because the active claims are branch, score, and robustness claims.
The correct unresolved hypothesis is not "exact reference missing"; it is
whether future HMC or approximation-quality claims will need a stronger
reference.

## Continuation Decision

BC5 is justified for targets whose BC1-BC3 gates passed.  BC5 must keep HMC
results at diagnostic/candidate/blocker scope unless convergence and posterior
recovery criteria are predeclared and passed.
