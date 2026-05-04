# Audit: Phase 8 MacroFinance adapter pilot

## Scope

This audit reviews
`docs/plans/bayesfilter-phase8-macrofinance-adapter-plan-2026-05-04.md`
from the perspective of a second developer before executing the Phase 8
implementation pass.

Audited repositories:

- BayesFilter: `/home/chakwong/BayesFilter`
- MacroFinance: `/home/chakwong/MacroFinance`

MacroFinance commit audited:

- `e23c31e Document one-country HMC remaining test gates`

## Findings

### Latest-code gate

The plan correctly requires a latest-code gate.  The gate has been satisfied for
this pass by pulling MacroFinance over SSH and refreshing `origin/main`.

Remaining risk:

- The configured MacroFinance `origin` remains HTTPS.  Future automated pulls in
  this environment should use SSH or change the remote URL.

Decision:

- Not a blocker for this pass.

### First adapter slice

The plan correctly makes the first BayesFilter implementation slice value-only.
The safest first target is a MacroFinance-shaped `LinearGaussianStateSpace`
object converted into BayesFilter's exact covariance-form Kalman object.

Required guardrails:

- Do not import MacroFinance at `bayesfilter` package import time.
- Do not copy AFNS, Riccati, cross-currency, production-panel, or
  identification construction code.
- Do not copy MacroFinance Kalman or differentiated-Kalman recursions.
- Preserve initial mean, initial covariance, observation jitter, and missing
  observation mask behavior.

Decision:

- Step 8.2 is justified.

### Derivative bridge

The plan correctly defers derivative bridging until after value parity.  The
updated MacroFinance checkout adds an important detail: one-country analytical
HMC now separates value-plus-score from value-plus-score-plus-Hessian workloads.
BayesFilter should preserve this split.

Required guardrails:

- A first-order bridge must not require Hessian work.
- A second-order bridge may request Hessian work explicitly.
- Result metadata must include parameter names, derivative order, backend
  provenance, jitter, and shape information.
- Derivative recursions remain delegated to MacroFinance during this phase.

Decision:

- Step 8.3 is justified after Step 8.2 passes.

### HMC-readiness gate

The plan correctly avoids adding a BayesFilter sampler in Phase 8.  MacroFinance
has smoke and target-XLA chain-policy references, but those are not convergence
evidence for BayesFilter.

Required guardrails:

- Only test finite conformance operations: value, score, negative Hessian, and
  metadata.
- Label the result as HMC-ready-smoke or HMC-contract-ready, not converged.
- Do not assert posterior convergence diagnostics in BayesFilter Phase 8.

Decision:

- Step 8.4 is justified after Step 8.3 passes.

### Large-scale and cross-currency deferral

The plan correctly defers large-scale and cross-currency providers.  Those
providers contain masking, sparse derivative, production-readiness,
identification, and cross-currency structural coverage policies that are broader
than the minimal adapter boundary.

Required guardrails:

- Record follow-on hypotheses and candidate tests.
- Do not let these richer providers block the one-country adapter pilot.

Decision:

- Step 8.5 is justified as an audit note, not an implementation slice.

## Missing Points Added By Audit

The execution should add the following details beyond the original plan:

1. A durable BayesFilter adapter result type for value likelihoods.
2. A durable BayesFilter derivative result type with optional Hessian.
3. Pure BayesFilter tests for adapter behavior without MacroFinance import.
4. Optional integration tests that skip if `/home/chakwong/MacroFinance` is not
   available.
5. Explicit tests that first-order derivative bridging does not request
   Hessian output.
6. Reset-memo updates after each phase, including whether the next phase remains
   justified.

## Final Audit Decision

The plan has no blocking issue after the additions above.  Execute all Phase 8
steps in order using the required cycle:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```
