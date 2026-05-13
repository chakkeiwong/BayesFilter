# Phase R3 result: PF-PF and proposal-correction rewrite

## Date

2026-05-10

## Purpose

This note records the third reader-facing rewrite phase of the DPF monograph
rebuild.

## Target file

- `docs/chapters/ch19c_dpf_implementation_literature.tex`

## Plan

Rewrite the old implementation-literature draft into a mathematically serious
PF-PF / proposal-correction chapter covering flow-as-proposal, transformed
proposal densities, corrected importance weights, Jacobian/log-determinant
evolution, and the exact status of what the correction restores.

## Execution

Rewrote `docs/chapters/ch19c_dpf_implementation_literature.tex` substantially.

The rewritten chapter now includes:

1. the need for proposal correction outside exact special cases;
2. the flow map as a change of variables;
3. transformed proposal density under the flow map;
4. corrected PF-PF importance weights;
5. the distinction between EDH/PF and LEDH/PF at the level of map construction;
6. Jacobian-matrix and log-determinant evolution identities;
7. a rigorous discussion of what proposal correction restores and what still
   remains approximate;
8. an explicit chapter-boundary section stating what is not yet claimed.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- Wording scan for status language was reviewed.  Uses of `exact`,
  `restores`, and `HMC target` were judged acceptable because:
  - `exact` appears only in explicitly limited contexts (special cases or
    negative statements);
  - `restores` refers specifically to the proposal-to-target density ratio, not
    to exact nonlinear filtering in general;
  - `HMC target` appears only in negative or future-facing boundary statements.

## Audit

### Primary criterion

Satisfied.

The rewritten chapter now states clearly that the flow map is not by itself the
final corrected object and that proposal correction is the mathematical bridge
between transport approximation and a cleaner likelihood interpretation.

### Veto diagnostics

- **Change-of-variables clarity veto**: cleared.
- **Raw-flow versus corrected-PF-PF blur veto**: cleared.
- **Status ambiguity veto**: cleared for this phase.  The chapter now states
  explicitly what correction restores and what remains approximate due to
  closure, discretization, or finite-particle effects.

### Remaining wider-block issue

Later chapters in the DPF block still remain below the final standard, but that
no longer blocks this phase.  The present chapter now supplies the necessary
proposal-correction layer for the later resampling and HMC-assessment work.

## Interpretation

This phase closes the most important mathematical gap left by the raw-flow
chapter.  The DPF exposition now distinguishes clearly between:

- a flow used as a transport approximation, and
- a flow used as a proposal inside an importance-corrected particle method.

That distinction is essential for the later HMC discussion.  Without it, the
monograph would still risk confusing a geometrically appealing transport with a
well-defined target construction.

## Tidy-up

No phase-local formatting blocker remains.

## Next phase justified?

Yes.

The next rewrite phase is justified because the resampling chapter can now be
written on top of a clear baseline:

1. classical particle filtering;
2. particle-flow transport;
3. proposal-corrected PF-PF.

## Recommended next phase

Proceed to the differentiable-resampling and OT chapter, making the
bias-versus-differentiability trade-off explicit and keeping learned/amortized
OT clearly downstream of the OT relaxation itself.
