# Phase R5 result: learned/amortized OT and implementation-mathematics rewrite

## Date

2026-05-10

## Purpose

This note records the fifth reader-facing rewrite phase of the DPF monograph
rebuild.

## Target file

- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

## Plan

Rewrite the old HMC-assessment draft in substance into a mathematically
controlled chapter on learned/amortized OT and the implementation-facing
mathematics of treating it as an approximation to the OT baseline.

## Execution

Rewrote `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
substantially.

The rewritten chapter now includes:

1. the hierarchy of approximations:
   exact resampling -> OT relaxation -> learned OT approximation;
2. the teacher-versus-learned map distinction;
3. a map-level approximation residual and its interpretation;
4. training-distribution dependence and extrapolation risks;
5. implementation-facing mathematical consequences for runtime,
   deterministic evaluation, state dimension, and architecture dependence;
6. an explicit chapter-boundary section deferring the final HMC verdict to the
   later assessment chapter.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- The wording scan for approximation-status language was reviewed and accepted
  because the flagged phrases appear only in explicit approximation-hierarchy or
  negative boundary statements.

## Audit

### Primary criterion

Satisfied.

The rewritten chapter now makes the teacher-versus-learned map distinction
mathematically explicit and prevents learned OT from being read as a free
acceleration with no target consequences.

### Veto diagnostics

- **Learned-OT-equals-OT veto**: cleared.
- **Residual-only-operational veto**: cleared.  Approximation residuals are now
  interpreted mathematically as part of a layered target-shift story.
- **Premature HMC-verdict veto**: cleared.  The final HMC judgment is explicitly
  deferred to the later assessment chapter.

## Interpretation

This phase is important because it stabilizes the approximation hierarchy of the
whole DPF block.  The monograph can now say clearly that learned or amortized OT
is not merely an implementation trick; it is a further approximation to an
already relaxed transport map.  That makes the later HMC analysis sharper,
because the chain of approximations is now explicit rather than implicit.

## Tidy-up

No phase-local formatting or citation blocker remains.

## Next phase justified?

Yes.

The next rewrite phase is justified because the final remaining reader-facing
problem is now the HMC-target and structural-model suitability chapter, which can
refer back to a fully explicit ladder:

1. classical PF baseline;
2. particle-flow transport;
3. PF-PF proposal correction;
4. differentiable resampling and OT relaxation;
5. learned/amortized OT approximation.

## Recommended next phase

Proceed to the final HMC-target correctness and nonlinear DSGE/MacroFinance
suitability chapter, making the rung-by-rung target interpretation the center of
the exposition.
