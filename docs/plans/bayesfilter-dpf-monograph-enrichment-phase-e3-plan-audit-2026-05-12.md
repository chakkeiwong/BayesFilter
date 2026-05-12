# Audit: Phase E3 differentiable resampling and OT enrichment plan

## Date

2026-05-12

## Scope

This audit reviews:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e3-resampling-ot-plan-2026-05-11.md`

The goal is to determine whether E3 is strong enough to drive a genuinely deeper
resampling/OT chapter rather than a somewhat cleaner summary of the same
material.

## Assessment

Assessment: **proceed with one tightening**.

The E3 plan is already one of the stronger subplans.  It centers the
bias-versus-differentiability trade-off correctly and demands stronger OT and
Sinkhorn treatment.  The main additional tightening needed is to require an
explicit distinction among:

- exact categorical resampling law,
- exact OT problem,
- entropic OT relaxation,
- finite-iteration Sinkhorn approximation.

Without that distinction, even a stronger chapter may still slide over a crucial
boundary between mathematical object and numerical approximation.

## Tightening required

Add one explicit required table:

| Layer | Mathematical object | Exact or approximate? | Source of approximation |
| --- | --- | --- | --- |

This should sit alongside the existing comparison tables and force the chapter to
separate the OT problem itself from the finite numerical solver used to compute
it.

## Audit disposition

Proceed after adding the table above.

## Next phase justified?

Yes, once the plan explicitly distinguishes the OT object from the solver-level
approximation.
