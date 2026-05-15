# Phase P4 plan: EDH and LEDH particle-flow derivation expansion

## Date

2026-05-14

## Target chapter

- `docs/chapters/ch19b_dpf_literature_survey.tex`

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, and passed P3 result.
- P2 established bibliography-spine support only for particle-flow sources; do
  not claim ResearchAssistant-reviewed EDH/LEDH support unless a later artifact
  records it.
- Allowed write set: this target chapter and the P4 result artifact.  Touch
  shared files only if the result records a necessary citation/label reason.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Rewrite the particle-flow chapter as a derivation-led monograph chapter.  The
reader must see exactly how the homotopy, continuity equation, Gaussian closure,
EDH affine field, linear-Gaussian recovery, and LEDH local linearization fit
together.

## Required implementation instructions

1. Add a notation/object inventory:
   - predictive law;
   - filtering law;
   - homotopy density;
   - normalizer;
   - pseudo-time;
   - velocity field;
   - flow map;
   - affine coefficients;
   - Gaussian closure;
   - local Jacobian and local information vector.
2. Expand homotopy derivation:
   - define normalized and unnormalized paths;
   - derive endpoint identities;
   - differentiate the log path;
   - explain the role of the normalizer.
3. Expand transport derivation:
   - state regularity assumptions;
   - derive the continuity equation;
   - derive the log-density continuity equation;
   - explain non-uniqueness of the velocity field.
4. Derive EDH under Gaussian closure:
   - write Gaussian log density under homotopy;
   - derive precision and covariance evolution;
   - match affine velocity terms;
   - derive $A(\lambda)$ and $b(\lambda)$ in BayesFilter notation;
   - state exactly where closure enters.
5. Expand linear-Gaussian recovery:
   - derive posterior covariance endpoint;
   - derive posterior mean endpoint;
   - connect to Kalman update;
   - define this as the exact implementation benchmark.
6. Derive LEDH:
   - local observation linearization;
   - local precision;
   - local information vector;
   - particle-local affine flow;
   - approximation status.
7. Add stiffness and discretization detail:
   - eigenvalue spread;
   - observation-noise sensitivity;
   - explicit integrator stability;
   - local stiffness variation in LEDH.
8. Add claim-status and limitations tables.

## Required source use

- Particle-flow source papers for EDH/LEDH.
- PF-PF sources for how flow later becomes a proposal.
- Numerical ODE/stiffness references only if used for substantive claims.
- Because P2 found no local ResearchAssistant summaries, these sources are
  bibliography-spine provenance unless reviewed source evidence is added later.
  EDH/LEDH formulas must be derived locally or weakened.

## Mathematical audit rules

- Do not call EDH exact outside the linear-Gaussian/Gaussian-closure regime.
- Do not present $A(\lambda)$ and $b(\lambda)$ as unexplained formulas.
- Do not hide the normalizer in the homotopy.
- Do not treat LEDH as a minor implementation variant; it changes the
  approximation structure.

## Required local tests/checks

- Search for `exact` and verify every instance states the regime.
- Search for `Gaussian closure` and verify the approximation point is local.
- Confirm every EDH/LEDH formula has symbol definitions.
- Use MathDevMCP or manual proof obligations for:
  - homotopy derivative;
  - covariance derivative;
  - affine-flow matching;
  - local linearization information vector.
- Create a P4 result artifact with an actual completion date in the filename.
- Record a derivation-obligation table and mark each obligation as
  `MathDevMCP`, `manual`, or `blocked`, with reasons for any manual fallback.
- Run the established LaTeX build and record DPF-local warnings.

## Veto diagnostics

The phase fails if:

- EDH remains formula-led;
- linear-Gaussian recovery lacks mean and covariance detail;
- LEDH local linearization is not derived;
- stiffness remains a casual warning;
- raw flow can still be misread as a corrected posterior sampler.
- the phase edits or stages student-baseline files;
- bibliography-spine source support is described as ResearchAssistant-reviewed;
- MathDevMCP/manual derivation-obligation evidence is omitted;
- build status is omitted.

## Exit gate

The chapter is ready only if a reviewer can reproduce the EDH/LEDH construction
and state what is exact, approximate, and numerical.
