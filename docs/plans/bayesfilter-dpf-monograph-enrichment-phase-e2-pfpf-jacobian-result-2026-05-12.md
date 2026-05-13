# Phase E2 result: PF-PF and Jacobian/log-det enrichment

## Date

2026-05-12

## Purpose

This note records the PF-PF and Jacobian/log-determinant enrichment phase of the
DPF monograph round.

## Plan tightening before execution

Pretending to be another developer, the E2 subplan was audited before
execution.  The audit concluded that the plan should explicitly separate:
- mathematically exact identities,
- approximation sources,
- and observable failure symptoms.

That tightening was added to the E2 plan before execution began.

Plan audit note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e2-plan-audit-2026-05-12.md`

## Execution

### 1. Target chapter inspected and deepened

Target chapter:
- `docs/chapters/ch19c_dpf_implementation_literature.tex`

Direct enrichment performed:
- strengthened the paragraph explaining that the exact change-of-variables
  identity is not itself the approximation;
- strengthened the Jacobian/log-det discussion so it now links directly to
  observable implementation failures;
- strengthened the “what proposal correction restores, and what it does not”
  section so that the exact identity / approximate method distinction is now
  explicit rather than implicit.

### 2. Source and coverage lane used

- inspected CIP LEDH/PF-PF material;
- used the student report parse for topic-ordering and coverage orientation on
  PF-PF / Jacobian / consistency topics.

### 3. Exact-identity versus approximation clarification achieved

The chapter now separates:
- the exact change-of-variables identity for the transformed proposal density;
- the approximation introduced by the choice of flow family;
- the approximation introduced by numerical integration of the flow;
- the approximation introduced by the finite-particle system.

This is the main conceptual tightening of the phase.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- local text audit confirmed that the chapter now contains stronger language on:
  - exact identity versus approximation,
  - Jacobian/log-det debugging relevance,
  - and what correction restores versus what remains approximate.

## Audit

### Primary criterion

Satisfied.

The chapter now makes the proposal-correction story mathematically explicit
enough that a reader is much less likely to confuse raw flow transport with a
proposal-corrected particle method.

### Veto diagnostics

- **change-of-variables only sketched**: cleared.
- **sounds like PF-PF makes nonlinear filtering exact in general**: cleared.
- **practical Jacobian/log-det burden still unclear**: cleared.

## Interpretation

This phase materially improves the usefulness of the PF-PF chapter for both
implementation and later HMC interpretation.  The most important gain is the
explicit separation between:
- what is mathematically exact in the correction formula, and
- what remains approximate because of closure, numerical integration, and
  finite-particle effects.

That distinction is exactly the kind of thing an implementation/debugging agent
needs in order to reason correctly about failures.

## Next phase justified?

Yes.

Phase E3 is now justified because the resampling/OT chapter can now build on a
clear particle-filter baseline, a clearer flow chapter, and a now more explicit
PF-PF correction chapter.
