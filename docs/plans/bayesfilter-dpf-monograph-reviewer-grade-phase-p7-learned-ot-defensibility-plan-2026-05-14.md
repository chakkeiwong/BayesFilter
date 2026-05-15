# Phase P7 plan: learned and amortized OT defensibility expansion

## Date

2026-05-14

## Target chapter

- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, P3, P4, P5, and passed P6 result.
- P2 established bibliography-spine support only for EOT, computational OT, and
  set-architecture sources; do not claim ResearchAssistant-reviewed learned-OT
  support unless a later artifact records it.
- Allowed write set: this target chapter and the P7 result artifact.  Touch
  shared files only if the result records a necessary citation/label reason.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Make learned OT impossible to read as a speed trick that preserves target
correctness automatically.  The chapter must present learned transport as a
student approximation to a teacher map, with residuals and evidence
requirements.

## Required implementation instructions

1. Add a notation/object inventory:
   - teacher map;
   - exact OT teacher;
   - EOT teacher;
   - finite Sinkhorn teacher;
   - barycentric teacher output;
   - learned student map;
   - training distribution;
   - loss function;
   - permutation action;
   - residuals.
2. Define teacher variants explicitly:
   - unregularized OT;
   - EOT;
   - finite Sinkhorn;
   - teacher actually used for training.
3. Define student map:
   - inputs;
   - output;
   - parameter vector;
   - dependence on particle count, dimension, epsilon, and architecture.
4. Expand training objective:
   - supervised teacher/student objective;
   - empirical training distribution;
   - what objective does and does not prove.
5. Expand symmetry discussion:
   - permutation equivariance for output clouds;
   - permutation invariance for scalar summaries;
   - why symmetry is necessary but not target-sufficient.
6. Expand residual hierarchy:
   - teacher-object error;
   - finite-solver error;
   - supervised learning error;
   - distribution shift;
   - filtering-summary error;
   - posterior error;
   - HMC value-gradient error.
7. Add failure regimes:
   - sharp weights;
   - high dimension;
   - particle-count shift;
   - epsilon shift;
   - structural-model regions;
   - learned-map collapse.
8. Add evidence ladder for banking use:
   - teacher residual tests;
   - stress tests;
   - posterior comparison;
   - gradient checks;
   - governance limitations.

## Required source use

- EOT teacher source.
- Computational OT/Sinkhorn source.
- DeepSets and set-transformer/permutation architecture sources.
- Student reports only as coverage and implementation-warning inputs.
- Because P2 found no local ResearchAssistant summaries, source use is
  bibliography-spine provenance unless reviewed source evidence is added later.
  Learned-map claims must be presented as surrogate/evidence-gated claims.

## Mathematical audit rules

- Do not say learned OT approximates categorical resampling unless the complete
  approximation chain is stated.
- Do not equate small map residual with small posterior error.
- Do not present runtime gains as target correctness evidence.
- Every learned-map claim must state the training-distribution scope.

## Required local tests/checks

- Search for `guarantee`, `works`, `valid`, `correct`, and `target`.
- Confirm every residual has a direct object and a "does not prove" statement.
- Confirm symmetry equations are dimensionally meaningful.
- Use MathDevMCP or manual checks for teacher/student residual definitions and
  permutation-equivariance equations.
- Create a P7 result artifact with an actual completion date in the filename.
- Record a derivation-obligation table and mark each obligation as
  `MathDevMCP`, `manual`, or `blocked`, with reasons for any manual fallback.
- Run the established LaTeX build and record DPF-local warnings.

## Veto diagnostics

The phase fails if:

- the teacher actually being approximated is ambiguous;
- training distribution is vague;
- learned residuals are not connected to filtering and HMC quantities;
- banking credibility is argued from speed or smoothness;
- architecture constraints are treated as software-only details.
- the phase edits or stages student-baseline files;
- bibliography-spine source support is described as ResearchAssistant-reviewed;
- MathDevMCP/manual derivation-obligation evidence is omitted;
- build status is omitted.

## Exit gate

The chapter is ready only if learned OT is clearly a layered surrogate with
explicit residual, extrapolation, and target-status risks.
