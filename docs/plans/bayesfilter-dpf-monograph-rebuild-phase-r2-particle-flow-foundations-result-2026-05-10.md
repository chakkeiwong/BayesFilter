# Phase R2 result: particle-flow foundations rewrite

## Date

2026-05-10

## Purpose

This note records the second reader-facing rewrite phase of the DPF monograph
rebuild.

## Target file

- `docs/chapters/ch19b_dpf_literature_survey.tex`

## Plan

Rewrite the old broad survey-style DPF chapter into a mathematically serious
particle-flow foundations chapter covering the transport motivation, homotopy
path, continuity equation, EDH under Gaussian closure, the exact
linear-Gaussian recovery benchmark, LEDH, and stiffness/discretization issues.

## Execution

Rewrote `docs/chapters/ch19b_dpf_literature_survey.tex` substantially.

The rewritten chapter now includes:

1. the conceptual move from discrete reweighting/resampling to continuous
   transport;
2. the pseudo-time homotopy density with explicit normalizing constant;
3. the continuity equation and the corresponding flow PDE;
4. EDH under Gaussian closure, including the homotopy covariance family and the
   affine EDH ODE coefficients;
5. the exact linear-Gaussian recovery statement as the main special-case
   benchmark;
6. LEDH via particle-specific local linearization and local precision matrices;
7. stiffness and discretization as mathematical and numerical concerns;
8. an explicit chapter-boundary section stating what is not yet claimed.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- Wording scan for exactness/high-stakes language was reviewed.  Uses of
  `exact` were judged acceptable because they appear only in:
  - the exact continuity/conservation statement,
  - the exact linear-Gaussian recovery benchmark,
  - and explicit negative/qualifying boundary statements.

## Audit

### Primary criterion

Satisfied.

The chapter now reads as a theory chapter rather than as a broad literature memo.
It defines the mathematical transport problem clearly and states where the
Gaussian-closure or local-linearization approximation enters.

### Veto diagnostics

- **Survey drift veto**: cleared.  The chapter is no longer organized as a broad
  commentary on the literature.
- **EDH/LEDH distinction veto**: cleared.  Global EDH and local LEDH are now
  mathematically separated.
- **Exact-versus-approximate-status veto**: cleared for this phase.  The
  linear-Gaussian recovery statement is isolated as the exact special case and
  the nonlinear flow constructions are described as closure- or
  linearization-dependent.

### Remaining wider-block issue

The broader DPF block still contains later chapters whose exposition remains
below the final standard.  That does not block R2, because this phase's own
mathematical criterion has been met and its compile/audit path succeeded.

## Interpretation

This phase establishes the correct mathematical bridge between the classical
particle-filter baseline and the later PF-PF/resampling chapters.  The main
improvement is that the monograph now states exactly what the particle flow is:
not yet a corrected filtering target, but a transport construction governed by a
homotopy and continuity equation, with exactness only in the linear-Gaussian
special case.

That separation is crucial for the later PF-PF chapter.  Without it, proposal
correction, differentiable resampling, and HMC suitability would again be built
on an ambiguous mathematical foundation.

## Tidy-up

No phase-local formatting or citation blocker remains for this rewrite.

## Next phase justified?

Yes.

The next rewrite phase is justified because the particle-flow foundations chapter
is now strong enough to support a mathematically serious PF-PF / proposal-
correction rewrite.

## Recommended next phase

Proceed to the PF-PF / proposal-correction chapter, separating:
- flow-as-transport,
- flow-as-proposal,
- corrected importance weights,
- and Jacobian/log-determinant evolution.
