# Independent Audit: Phase 3 QR Linear Derivative Port

Date: 2026-05-09

Auditor stance: treat the QR/square-root Phase 3 plan as if another developer
proposed it.  The objective is to decide whether the next BayesFilter
implementation phase is complete enough to execute without silently mixing
testing references, production filters, and MacroFinance application logic.

## Verdict

Proceed, but split Phase 3 into small gates:

1. QR and Cholesky factor derivative identities.
2. QR/square-root value and masked value recursions.
3. QR/square-root analytic score/Hessian recursion.

This order is required because a score/Hessian mismatch is otherwise hard to
localize: it can come from the QR derivative algebra, the square-root Kalman
recursion, the likelihood derivatives, or the state/covariance derivative
updates.

## Required Corrections Before Coding

- Update the consolidation map so it no longer says to choose dense or
  solve-form derivatives as the production route.
- Treat the already ported covariance-form and solve-form derivatives as
  `bayesfilter.testing` references only.
- Keep production QR modules TensorFlow/TensorFlow Probability only: no NumPy
  import and no `.numpy()` conversion.
- Move any eager trace conversion into testing/debug helpers, not into
  production filter functions.

## Execution Gates

### Phase 3A: Factor Identities

Required operations:
- port positive-diagonal thin QR convention;
- port first- and second-order QR factor derivative helpers;
- port Cholesky factor derivative helpers if the QR backend uses them for
  initial or innovation factors;
- add reconstruction tests for matrix, covariance, first derivatives, and
  second derivatives.

Stop if:
- first- or second-order reconstruction errors exceed tight tolerances;
- signs/pivots are not deterministic;
- helper code needs NumPy in production modules.

### Phase 3B: QR Value And Masked Value

Required operations:
- port dense QR/square-root Kalman value recursion;
- port masked QR/square-root value recursion;
- preserve the static dummy-row mask convention;
- compare dense QR value against existing Cholesky TF value;
- compare all-true masked QR against dense QR;
- verify all-missing periods contribute zero measurement likelihood.

Stop if:
- QR value disagrees with Cholesky on well-conditioned cases;
- masked QR fails all-missing semantics;
- production trace/debug handling requires eager `.numpy()`.

### Phase 3C: QR Score And Hessian

Required operations:
- port QR/square-root analytic score/Hessian recursion;
- compare against `bayesfilter.testing` solve/covariance references on small
  fixtures;
- compare against TensorFlow autodiff where the fixture is smooth and small;
- check Hessian symmetry;
- check graph reuse for static shapes.

Stop if:
- QR derivative errors cannot be localized to a factor identity, value
  recursion, or testing-reference mismatch;
- derivative behavior near singular pivots cannot be labeled cleanly.

## Non-Goals

- Do not port SVD value or SVD derivatives in this phase.
- Do not port nonlinear UKF, SVD sigma-point, or CUT4-G in this phase.
- Do not edit MacroFinance or `/home/chakwong/python`.
- Do not promote testing references into public production exports.

## Audit Outcome

Phase 3 is justified if executed in the gated order above.  Continue
automatically from 3A to 3B to 3C only when the previous gate passes and the
source hygiene rule remains satisfied.
