# Phase E5 result: DPF-specific HMC enrichment

## Date

2026-05-12

## Purpose

This note records the DPF-specific HMC enrichment phase of the monograph round.

## Plan tightening before execution

Pretending to be another developer, the E5 subplan was audited before
execution.  The audit concluded that the plan should explicitly require a
rung-by-rung promotion-evidence ladder, not just a rung-by-rung target-status
description.

That tightening was added to the E5 plan before execution began.

Plan audit note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e5-plan-audit-2026-05-12.md`

## Execution

### 1. Target chapter deepened

Target chapter:
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

Direct enrichment performed:
- strengthened the recommendation section so that each DPF rung is read not only
  by its current HMC interpretation, but also by what evidence would be needed
  to promote it;
- strengthened the closing interpretation so that future promotion is explicitly
  rung-specific:
  - raw flow requires stronger exactness or controlled-approximation
    justification;
  - PF-PF requires Monte Carlo, Jacobian, and compiled-path evidence;
  - relaxed resampling requires explicit target-definition and posterior-
    sensitivity evidence;
  - learned OT requires a residual-to-posterior error story strong enough to
    justify downstream use.

### 2. Main enrichment gains

The chapter now does more than classify target types.  It also functions as a
forward-looking experimental and implementation guide by telling the reader what
would have to be shown next to upgrade a given rung.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, and no LaTeX errors.  The only match
  to the scan pattern was the package name `rerunfilecheck`, not a rerun
  diagnostic.
- local text audit confirmed stronger rung-promotion language and a clearer link
  between target interpretation and next-step evidence.

## Audit

### Primary criterion

Satisfied.

The chapter is now detailed enough that a future experimenter can use it not
only to classify a DPF-HMC path by target type, but also to decide what evidence
would be required to make a stronger claim.

### Veto diagnostics

- **differentiability treated as near-validity**: cleared.
- **DSGE/MacroFinance still too slogan-like**: sufficiently improved for this
  phase.
- **surrogate-HMC / HNN comparison too superficial**: improved enough to clear
  for this enrichment step, though it remains a strong candidate for later deeper
  comparative work.

## Interpretation

This phase turns the DPF-specific HMC chapter into a much better bridge between
mathematical interpretation and future experimentation.  It now says not only
what each rung currently means, but also what kind of evidence would have to be
produced before a stronger HMC claim becomes legitimate.

## Next phase justified?

Yes.

Phase E6 is now justified because the DPF block is rich enough that a
literature-to-debugging crosswalk can be built without relying on vague chapter
summaries.
