# Phase R1 result: particle-filter foundations rewrite

## Date

2026-05-10

## Purpose

This note records the first actual reader-facing rewrite phase of the DPF
monograph rebuild.

## Target file

- `docs/chapters/ch19_particle_filters.tex`

## Plan

Rewrite the old short gateway chapter into a mathematically serious
particle-filter foundations chapter that establishes the exact nonlinear
filtering recursion, the empirical particle approximation, the SIS/SIR
construction, the bootstrap PF likelihood estimator, its estimator-status, and
the role of ESS/degeneracy.

## Execution

Rewrote `docs/chapters/ch19_particle_filters.tex` substantially.

The rewritten chapter now includes:

1. nonlinear state-space model and exact filtering recursion;
2. predictive and filtering laws in BayesFilter notation;
3. marginal-likelihood factorization;
4. empirical filtering measures and test-function approximation;
5. sequential importance sampling with explicit trajectory/proposal factoring;
6. sequential importance resampling and the bootstrap proposal choice;
7. the bootstrap particle-filter likelihood estimator;
8. a proposition-level statement of its status as an unbiased estimator of the
   marginal likelihood under standard assumptions;
9. ESS and degeneracy discussion as the mathematical baseline motivating later
   DPF chapters;
10. an explicit chapter-boundary section stating what is not yet claimed.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed with
  the document up to date after the bibliography/key cleanup.
- `git diff --check` identified only a trailing blank-line issue in
  `docs/references.bib`, which remains to be tidied before commit.
- targeted wording scan on the rewritten chapter found uses of `unbiased` and
  `HMC-ready`; these were reviewed and found acceptable because:
  - `unbiased` appears only in a scoped proposition-level estimator-status
    statement;
  - `HMC-ready` appears only in the explicit negative boundary statement saying
    the chapter does *not* yet claim such a backend.

## Audit

### Primary criterion

Satisfied.

The rewritten chapter is materially stronger than the earlier gateway version and
now reads as a mathematical baseline chapter rather than as a high-level program
memo.

### Veto diagnostics

- **Commentary drift**: cleared for this chapter.  The prose is now centered on
  definitions, recursions, estimator status, and degeneracy.
- **Likelihood-status support**: cleared for the bootstrap PF statement.  The
  chapter now phrases the result carefully and ties it to standard assumptions.
- **Notation / citation failure**: partially cleared.  The particle-filter
  chapter itself now compiles cleanly enough, but the wider DPF block still has
  unresolved exposition-quality issues in later draft chapters and some residual
  bibliography cleanup work remains outside this chapter.

## Interpretation

This phase established the correct mathematical starting point for the rebuilt
DPF block.  The most important improvement is conceptual: the DPF discussion now
starts from a clearly defined exact filtering recursion and a clearly identified
classical Monte Carlo estimator, rather than from vague prose about particle
methods.

That matters for every later phase.  Particle flow, PF-PF, differentiable
resampling, OT relaxations, and learned transport operators now have a concrete
baseline against which they can be interpreted.

## Tidy-up

Remaining local tidy item from this phase:
- remove the trailing blank line at the end of `docs/references.bib` before the
  next commit.

## Next phase justified?

Yes, with a qualification.

The next rewrite phase is justified because the particle-filter baseline chapter
now exists in a satisfactory mathematical form.  However, the later DPF draft
chapters still remain below the final exposition standard, so the next phase
must continue the rebuild rather than attempt minor patching.

## Recommended next phase

Proceed to the next reader-facing rewrite phase on **particle-flow foundations**
(EDH / LEDH / homotopy / continuity-equation structure), keeping PF-PF
proposal correction separate unless the derivation shows that one combined
chapter remains readable.
