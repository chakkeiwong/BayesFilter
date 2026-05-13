# Phase R4 result: differentiable resampling and optimal transport rewrite

## Date

2026-05-10

## Purpose

This note records the fourth reader-facing rewrite phase of the DPF monograph
rebuild.

## Target file

- `docs/chapters/ch32_diff_resampling_neural_ot.tex`

## Plan

Rewrite the old mixed resampling/neural-OT draft into a mathematically serious
resampling and OT chapter covering the resampling bottleneck, pathwise
nondifferentiability of standard resampling, soft resampling, transport /
projection language, entropic OT, Sinkhorn structure, barycentric projection,
and the bias-versus-differentiability trade-off.

## Execution

Rewrote `docs/chapters/ch32_diff_resampling_neural_ot.tex` substantially.

The rewritten chapter now includes:

1. the weighted-to-equal-weight resampling bottleneck stated at the level of
   empirical measures;
2. standard resampling as a discontinuous categorical map;
3. soft resampling as a smooth relaxation;
4. equal-weight resampling viewed as a transport problem;
5. entropic OT resampling with the primal formulation;
6. the Sinkhorn scaling form and barycentric projection;
7. an explicit comparison of standard, soft, and OT resampling in terms of
   differentiability and bias;
8. an explicit chapter-boundary section deferring learned/amortized OT to the
   next stage.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed and
  reported all targets up to date.
- `git diff --check` passed.
- The wording scan for high-stakes target-status language was reviewed and
  accepted because the flagged statements are boundary statements or explicit
  trade-off descriptions rather than overclaims.

## Audit

### Primary criterion

Satisfied.

The chapter now makes the differentiability-versus-bias problem mathematically
explicit instead of treating differentiable resampling as a vague engineering
upgrade.

### Veto diagnostics

- **OT-equals-multinomial veto**: cleared.  The chapter now states explicitly
  that OT resampling is a relaxed transport construction rather than identical to
  categorical resampling.
- **Bias-source vagueness veto**: cleared.  Soft resampling and entropic OT are
  both described as relaxed constructions whose modifications to the resampling
  map are part of the mathematical story.
- **Learned-OT-domination veto**: cleared.  The chapter defers learned/amortized
  OT rather than letting it dominate the OT baseline.

## Interpretation

This phase gives the rebuilt DPF block the missing mathematical middle layer
between PF-PF correction and the later learned/surrogate and HMC chapters.  The
main conceptual gain is that the monograph now states explicitly that
pathwise differentiability is obtained by changing the resampling map, and hence
by changing the mathematical object being differentiated.

That statement is essential for the later HMC-suitability analysis.  Without it,
one would again risk conflating smooth gradients with exact-target inference.

## Tidy-up

No phase-local formatting or citation blocker remains.

## Next phase justified?

Yes.

The next rewrite phase is justified because the learned/amortized OT layer can
now be written as a further approximation on top of a clear OT baseline, and the
HMC-assessment chapter can later refer back to an explicit resampling-status
hierarchy.

## Recommended next phase

Proceed to the learned/amortized OT and implementation-mathematics chapter,
keeping the teacher-versus-learned map distinction explicit and treating
approximation residuals as part of the target-status story.
