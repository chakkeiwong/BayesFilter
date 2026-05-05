# Audit: BayesFilter structural SVD 12-phase implementation plan

## Date

2026-05-05

## Plan Reviewed

- `docs/plans/bayesfilter-structural-svd-12-phase-implementation-plan-2026-05-05.md`

## Audit Stance

Pretending to be another developer, I approve the plan as a detailed
implementation roadmap with one strict interpretation: this session may execute
only documentation, audit, reset-memo, and commit phases.  Backend
implementation must begin in a later coding session at Phase 1, the
mathematical source audit.

## Completeness Check

The plan covers the twelve gaps requested by the user:

1. mathematical source audit;
2. code reuse and migration audit;
3. BayesFilter structural sigma-point core;
4. exact Kalman and degenerate linear spine;
5. generic structural fixtures;
6. MacroFinance adapter and analytic derivative spine;
7. DSGE adapter integration;
8. model-specific DSGE completion evidence;
9. derivative and Hessian safety gate;
10. JIT and static-shape production gate;
11. HMC validation ladder;
12. documentation, provenance, and release gate.

The plan correctly orders the work:

```text
math/source audit
  -> code audit
  -> core filters and exact baselines
  -> generic fixtures
  -> MacroFinance and DSGE adapters
  -> model-specific residuals
  -> derivative/JIT gates
  -> HMC ladder
  -> release documentation
```

## Strengths

- It keeps exact collapsed LGSSM filtering separate from nonlinear structural
  filtering.
- It requires deterministic completion pointwise for mixed structural models.
- It preserves explicit approximation labels for mixed full-state nonlinear
  sigma-point paths.
- It does not treat SmallNK as sufficient evidence for Rotemberg, SGU, EZ, or
  NAWM-scale models.
- It blocks HMC until value, residual, derivative, and JIT gates pass.
- It treats result metadata as part of correctness.
- It keeps DSGE and MacroFinance economics in the client repositories.

## Risks and Required Interpretations

1. **Phase 1 is not optional**

   A later coding agent must write the mathematical source audit before
   implementing backend changes.  Chat-level discussion is not enough.

2. **MacroFinance derivative migration is high risk**

   Existing analytic derivatives may be valuable, but they must be audited for
   notation, assumptions, and numerical agreement before migration.

3. **Rotemberg and SGU bridges are not final evidence**

   The DSGE client already exposes first-order completion bridges.  Those are
   adapter readiness evidence, not nonlinear structural correctness evidence.

4. **SVD/eigen derivatives require stress gates**

   The plan correctly refuses to promote SVD/eigen-gradient HMC merely because
   the value path runs.

5. **BayesFilter dirty state must be respected**

   Current untracked items include a Julier PDF, templates, and multiple
   planning artifacts.  The executing agent must stage explicit paths only.

6. **This session must not implement code**

   The user reserved this session for documentation.  Therefore executing
   "each phase" in this session means executing the plan-writing, audit,
   reset-memo, validation, and commit phases, then stopping at the first
   implementation gate.

## Missing Points Found

No blocking omissions were found.  Two clarifications should guide execution:

- The plan should be treated as BayesFilter-owned even though DSGE and
  MacroFinance tests appear in later phases.
- Phase 12 source-map updates should occur only when the plan/audit/result docs
  are committed or when monograph text changes.  Do not edit source maps for
  uncommitted PDFs or unrelated external assets.

## Decision

Proceed with the documentation-only execution:

1. update the BayesFilter reset memo;
2. validate markdown/hygiene;
3. commit scoped planning docs and the previously created handoff plan if it is
   intended to be visible to the next coding agent;
4. do not implement backend code in this session.

The next coding session should begin at Phase 1 of the plan.
