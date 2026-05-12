# Audit: Phase E2 PF-PF and Jacobian/log-det enrichment plan

## Date

2026-05-12

## Scope

This audit reviews:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e2-pfpf-jacobian-plan-2026-05-11.md`

The goal is to determine whether E2 is strong enough to deepen the PF-PF chapter
without drifting into a merely slightly longer explanation of formulas that are
already present.

## Assessment

Assessment: **proceed with one tightening**.

The E2 plan is already strong.  It asks the right mathematical questions and it
correctly centers the change-of-variables story, corrected weights, and
Jacobian/log-determinant machinery.  The remaining weakness is that it should be
more explicit about the difference between:

- exact correctness of the change-of-variables identity itself,
- and practical approximation error introduced by numerical flow integration,
  finite particles, or Jacobian evaluation.

## Tightening required

Add one explicit required output:

- a short table separating:
  - mathematically exact identities,
  - approximation sources,
  - and implementation observables when those approximations fail.

Suggested form:

| Component | Exact identity? | Approximation source | Observable failure symptom |
| --- | --- | --- | --- |

## Audit disposition

Proceed after adding the table above.

## Next phase justified?

Yes, once the distinction between exact identity and numerical approximation is
made explicit in the E2 plan.
