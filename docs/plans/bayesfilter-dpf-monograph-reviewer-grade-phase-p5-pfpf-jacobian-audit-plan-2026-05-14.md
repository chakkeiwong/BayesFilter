# Phase P5 plan: PF-PF proposal correction and Jacobian audit

## Date

2026-05-14

## Target chapter

- `docs/chapters/ch19c_dpf_implementation_literature.tex`

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, P3, and passed P4 result.
- P2 established bibliography-spine support only for PF-PF sources; do not claim
  ResearchAssistant-reviewed PF-PF support unless a later artifact records it.
- Allowed write set: this target chapter and the P5 result artifact.  Touch
  shared files only if the result records a necessary citation/label reason.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Make PF-PF defensible as an importance-sampling construction based on a
transported proposal density, not as an informal claim that particle flow
improves particles.

## Required implementation instructions

1. Add a notation/object inventory:
   - pre-flow particle;
   - post-flow particle;
   - ancestor state;
   - pre-flow proposal density;
   - flow map;
   - inverse map;
   - Jacobian matrix;
   - post-flow proposal density;
   - intended filtering target;
   - corrected weight.
2. Expand one-step proposal correction:
   - define conditioning on ancestor and observation;
   - state bijection/differentiability assumptions;
   - derive change-of-variables step by step;
   - derive corrected weight for transition-prior proposal.
3. Add generic proposal version:
   - show how the formula changes when pre-flow proposal is not the transition
     prior;
   - state why this matters for implementation extensions.
4. Expand trajectory-level interpretation:
   - distinguish one-step correction from full filtering likelihood;
   - state what correction restores and what finite-particle uncertainty
     remains.
5. Expand EDH/PF and LEDH/PF subsections:
   - global affine map case;
   - particle-local map case;
   - Jacobian/log-det burden in each.
6. Expand log-determinant derivation:
   - Jacobian ODE;
   - Jacobi formula;
   - trace identity;
   - affine simplification;
   - sign convention for forward versus inverse map.
7. Add implementation-audit diagnostics:
   - synthetic affine map density check;
   - finite-difference determinant check;
   - AD determinant check in small dimension;
   - corrected/uncorrected weight comparison;
   - log-weight normalization check;
   - value-gradient consistency on fixed seeds.

## Required source use

- PF-PF / invertible particle-flow sources.
- SMC proposal and importance-weight sources.
- Flow chapter formulas from P4.
- Because P2 found no local ResearchAssistant summaries, source use is
  bibliography-spine provenance unless reviewed source evidence is added later.
  Corrected-weight and log-determinant claims must be locally derived.

## Mathematical audit rules

- Every density ratio must specify numerator target and denominator proposal.
- Every determinant must specify whether it is for the forward or inverse map.
- Do not use `corrected` without stating the corrected object.
- Do not imply that proposal correction removes finite-particle variance.

## Required local tests/checks

- Search for `correct` and verify object specificity.
- Search for determinant expressions and verify sign/inverse convention.
- Use MathDevMCP or manual proof obligations for:
  - change-of-variables formula;
  - corrected weight derivation;
  - Jacobian ODE;
  - log-det trace identity.
- Create a P5 result artifact with an actual completion date in the filename.
- Record a derivation-obligation table and mark each obligation as
  `MathDevMCP`, `manual`, or `blocked`, with reasons for any manual fallback.
- Run the established LaTeX build and record DPF-local warnings.

## Veto diagnostics

The phase fails if:

- PF-PF remains a conceptual bridge instead of a derivation;
- determinant sign convention is ambiguous;
- trajectory-level status is unclear;
- HMC relevance is asserted before finite-particle and numerical caveats;
- diagnostics are not tied to equations.
- the phase edits or stages student-baseline files;
- bibliography-spine source support is described as ResearchAssistant-reviewed;
- MathDevMCP/manual derivation-obligation evidence is omitted;
- build status is omitted.

## Exit gate

The chapter is ready only if a skeptical reviewer can audit PF-PF as a
proposal-density correction and identify exactly what remains approximate.
