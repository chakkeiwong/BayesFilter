# Audit: Phase E4 learned/amortized OT enrichment plan

## Date

2026-05-12

## Scope

This audit reviews:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e4-learned-ot-plan-2026-05-11.md`

The goal is to determine whether E4 is strong enough to deepen the learned OT
chapter without drifting toward an implementation summary or a polished ML note.

## Assessment

Assessment: **proceed with one tightening**.

The plan is directionally strong and already benefits from the residual/target-
shift tightening added earlier.  The remaining weakness is that it should
explicitly require a separation between:

- map-level residuals,
- chapter-level interpretation of posterior-level consequences,
- and what would still need experimental evidence before posterior claims become
  stronger.

Without this, the chapter may still become mathematically better but too quick in
translating residual metrics into posterior claims.

## Tightening required

Add one explicit required subsection or table:

| Residual quantity | What it measures directly | What it does not by itself prove | What evidence would be needed next |
| --- | --- | --- | --- |

## Audit disposition

Proceed after adding the table above.

## Next phase justified?

Yes, once the plan explicitly prevents residual metrics from being overread as
posterior guarantees.
