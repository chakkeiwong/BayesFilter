# Phase E3 result: differentiable resampling and OT enrichment

## Date

2026-05-12

## Purpose

This note records the differentiable-resampling and OT enrichment phase of the
DPF monograph round.

## Plan tightening before execution

Pretending to be another developer, the E3 subplan was audited before
execution.  The audit concluded that the plan should explicitly separate:

- exact categorical resampling law;
- the exact unregularized OT problem where that comparison is made;
- the entropically regularized OT problem actually used for smoothness;
- the finite-iteration Sinkhorn approximation used in code.

That tightening was added to the E3 plan before execution began.

Plan audit note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e3-plan-audit-2026-05-12.md`

## Execution

### 0. Build-integration repair

Before continuing to E4, the active files were re-audited against the plan.
That audit found that the target chapter was not yet included in `docs/main.tex`
and therefore previous build success did not prove that E3 compiled.  It also
found a latent LaTeX defect in the exact/relaxed/solver approximation list.

Repairs performed:
- added `\input{chapters/ch32_diff_resampling_neural_ot}` to `docs/main.tex`
  between the PF-PF chapter and the learned-OT chapter;
- repaired the malformed approximation-hierarchy enumerate block;
- added `placeins` and a chapter-local `\FloatBarrier` so E3 tables do not
  float into the learned-OT chapter;
- set `hypertexnames=false` to eliminate duplicate Hyperref destination
  warnings from deferred chapter-local tables.

### 1. Target chapter deepened

Target chapter:
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`

Direct enrichment performed:
- strengthened the transport/projection discussion so that the mathematical
  transport object and the numerical solver approximation are no longer blurred;
- added the unregularized OT coupling object explicitly as a linear-programming
  baseline;
- added a local soft-resampling nonlinear test-function bias expansion;
- strengthened the barycentric-map discussion so that the OT-relaxed target and
  the finite Sinkhorn approximation are clearly separated;
- strengthened the bias-versus-differentiability discussion so that the chapter
  now explicitly distinguishes:
  1. exact categorical resampling;
  2. exact unregularized OT interpretation;
  3. entropically regularized OT;
  4. finite-iteration Sinkhorn approximation.
- added source-backed tables for:
  - OT object versus solver approximation;
  - resampling family comparison;
  - implementation debug map;
  - source claim / local derivation / debugging use.

### 2. Main enrichment gains

The chapter now makes it much harder to conflate:
- a transport formulation of equal-weight resampling,
- the relaxed OT problem actually chosen for differentiability,
- and the numerical procedure used to approximate that relaxed problem.

This is the key conceptual tightening required by the enrichment round.

## Tests

- `latexmk -C main.tex` was run once before the corrected build to clear stale
  LaTeX artifacts.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed from
  `docs/` after the E3 chapter was wired into `main.tex`.
- `git diff --check` passed.
- focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, and no LaTeX errors after the final
  E3 repair build.
- local text audit confirmed stronger chapter-level presence of:
  - categorical resampling law,
  - unregularized OT,
  - entropic OT,
  - Sinkhorn approximation,
  - explicit bias language,
  - and implementation-debug mappings.

## Audit

### Primary criterion

Satisfied.

The chapter now makes the bias-versus-differentiability trade-off much more
explicit, both mathematically and in a way that is useful for later
implementation and HMC interpretation.

### Veto diagnostics

- **OT too intuitive / not precise enough**: cleared.
- **soft-resampling bias still only verbal**: improved enough to clear for this
  enrichment phase.
- **regularization and finite-iteration issues disconnected from target
  interpretation**: cleared.

## Interpretation

This phase now gives the resampling chapter the missing middle layer between raw
method description and implementation-useful target analysis.  The reader can now
trace a future issue not just to “OT resampling,” but to one of several distinct
layers:
- categorical law,
- OT formulation,
- entropic relaxation,
- finite Sinkhorn approximation.

That is exactly the kind of separation needed if the monograph is going to serve
as a debugging reference rather than just a conceptual survey.

The earlier E3 result note should be read with this correction: the build claim
was not sufficient until `ch32_diff_resampling_neural_ot.tex` was actually
included in `docs/main.tex`.  The corrected phase now compiles the intended
chapter.

## Next phase justified?

Yes.

Phase E4 is now justified because learned/amortized OT can now be deepened on
top of a much clearer OT baseline and a more explicit approximation hierarchy.
