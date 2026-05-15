# Phase P6 plan: differentiable resampling and OT/Sinkhorn expansion

## Date

2026-05-14

## Target chapter

- `docs/chapters/ch32_diff_resampling_neural_ot.tex`

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, passed P3 result, and either passed P4/P5
  results produced after P3 or a P3/P4/P5 reconciliation note confirming that
  the provisional P4/P5 results remain valid.
- P2 established bibliography-spine support only for differentiable resampling,
  OT, EOT, and Sinkhorn sources; do not claim ResearchAssistant-reviewed support
  unless a later artifact records it.
- Allowed write set: this target chapter and the P6 result artifact.  Touch
  shared files only if the result records a necessary citation/label reason.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.
- If this target chapter already has uncommitted edits when P6 starts, audit the
  existing diff first.  Record whether the partial diff is adopted, repaired, or
  superseded.  Do not treat a partial chapter diff as a completed P6 result.

## Purpose

Make the mathematical cost of differentiability explicit.  The chapter must
prevent any reader from confusing categorical resampling, unregularized OT,
entropic OT, finite Sinkhorn, and autodiff-through-solver paths.

## Required implementation instructions

1. Add a notation/object inventory:
   - weighted empirical measure;
   - equal-weight empirical measure;
   - ancestor index map;
   - soft-resampling interpolation;
   - transport coupling;
   - OT feasible set;
   - cost matrix;
   - entropy convention;
   - EOT primal;
   - Sinkhorn scaling vectors;
   - barycentric projection;
   - finite-iteration solver path.
2. Expand categorical resampling:
   - define ancestor selection through cumulative weights;
   - add a simple one-dimensional discontinuity example;
   - distinguish pathwise derivative from score-function alternatives.
3. Expand soft resampling:
   - derive preserved mean;
   - expand nonlinear test-function bias;
   - state which function classes are preserved and which are shifted;
   - connect shrinkage to cloud diversity.
4. Expand unregularized OT:
   - define coupling constraints;
   - explain equal-weight target class;
   - state why this is a transport projection, not categorical resampling.
5. Expand EOT/Sinkhorn:
   - define entropy sign convention;
   - derive first-order conditions;
   - derive Gibbs kernel and scaling form;
   - derive row/column scaling updates;
   - state uniqueness/strict convexity assumptions conservatively.
6. Expand barycentric projection:
   - distinguish coupling from deterministic output cloud;
   - define map status and differentiability assumptions.
7. Add numerical-analysis section:
   - epsilon limit and regularization bias;
   - kernel underflow;
   - log-domain stabilization;
   - finite iteration residual;
   - unrolled versus implicit differentiation;
   - memory and runtime scaling.
8. Add claim-status table for categorical, OT, EOT, finite Sinkhorn, and
   solver-differentiated objectives.

## Required source use

- Differentiable-resampling papers.
- OT monographs/texts for coupling formulation.
- Sinkhorn and computational OT sources.
- Stabilized Sinkhorn sources if stabilization claims are made.
- Because P2 found no local ResearchAssistant summaries, source use is
  bibliography-spine provenance unless reviewed source evidence is added later.
  OT/EOT/Sinkhorn formulas must be derived locally or weakened.

## Mathematical audit rules

- Do not describe soft or OT resampling as exact categorical resampling.
- Do not call Sinkhorn a black-box differentiable layer.
- Every EOT equation must state entropy convention and marginal constraints.
- Every finite-solver claim must distinguish mathematical problem from
  implementation path.

## Required local tests/checks

- Search for `exact` and verify exact-for-what.
- Search for `optimal` and verify objective and regularization.
- Confirm all OT symbols have definitions.
- Use MathDevMCP or manual proof obligations for:
  - soft-resampling test-function bias;
  - EOT scaling form;
  - Sinkhorn marginal equations;
  - barycentric map dimensions.
- Create a P6 result artifact with an actual completion date in the filename.
- Record a derivation-obligation table and mark each obligation as
  `MathDevMCP`, `manual`, or `blocked`, with reasons for any manual fallback.
- Run the established LaTeX build and record DPF-local warnings.
- Inspect table/PDF readability explicitly; P0 already identified DPF-local
  table width risk in this chapter.
- Record the disposition of any pre-existing `ch32` partial diff.

## Veto diagnostics

The phase fails if:

- differentiability appears free;
- OT and categorical resampling remain blurred;
- finite Sinkhorn is not separated from EOT;
- barycentric projection is not explained;
- numerical stabilization is mentioned without mathematical context.
- the phase edits or stages student-baseline files;
- bibliography-spine source support is described as ResearchAssistant-reviewed;
- MathDevMCP/manual derivation-obligation evidence is omitted;
- build or table-readability status is omitted.
- P3/P4/P5 reconciliation is missing while provisional P4/P5 results predate
  the completed P3 gate.
- pre-existing `ch32` partial edits are neither audited nor given a disposition.

## Exit gate

The chapter is ready only if the reader can state the exact object whose
gradient is computed for every resampling variant.
