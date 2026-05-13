# Phase R4 plan: differentiable resampling and optimal transport rewrite

## Date

2026-05-10

## Purpose

This is the fourth reader-facing rewrite phase of the DPF monograph rebuild.
Its purpose is to write the mathematically rigorous differentiable-resampling
and optimal-transport chapter that follows the PF-PF proposal-correction
chapter.

## Scope

This phase covers the resampling bottleneck, soft resampling, transport-based
resampling, entropic OT resampling, and the bias-versus-differentiability
trade-off.  It does not yet absorb the learned/amortized OT layer except where
needed to state clearly that it is a further downstream approximation.

Target reader-facing file for this phase:
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`

The file will serve as the rewrite target for the resampling/OT chapter in
substance, even if later chapter renaming remains possible.

## Mathematical goals

The rewritten chapter must provide a self-contained treatment of:

1. standard resampling as a discontinuous map;
2. why pathwise differentiation fails for standard resampling;
3. soft resampling and what it preserves versus biases;
4. equal-weight resampling as a transport or projection problem;
5. entropic OT and the Sinkhorn scaling form;
6. barycentric transport as the differentiable resampling output;
7. the bias-versus-differentiability trade-off across the main resampling
   families.

## Required section map

1. **The resampling bottleneck**
   - define the weighted particle ensemble and the equal-weight objective.

2. **Standard resampling as a discontinuous map**
   - formalize categorical ancestor selection and why pathwise gradients vanish
     almost everywhere.

3. **Soft resampling**
   - define the relaxed map;
   - state what moments are preserved;
   - derive or carefully present the source of bias.

4. **Resampling as transport / projection**
   - express equal-weighting as a measure-transport problem.

5. **Entropic OT resampling**
   - present the OT primal;
   - present the entropic regularization;
   - define the Sinkhorn form and barycentric projection.

6. **Bias versus differentiability**
   - compare standard resampling, soft resampling, and OT resampling at the
     level of exactness, differentiability, and target perturbation.

7. **Chapter boundary**
   - state that learned/amortized OT is a further approximation layer to be
     treated next.

## Source discipline

Primary source families for this phase:
- CIP differentiable-resampling and OT chapter;
- supporting OT and resampling literature;
- student materials only as comparison/caveat checks, not as the chapter voice.

## Execution rule

The chapter must remain mathematically centered and must not drift into:
- architecture commentary about neural operators beyond what is needed to set up
  the next chapter;
- broad HMC conclusions better reserved for the later HMC-suitability chapter;
- implementation-governance prose.

## Tests

- compile the monograph after the rewrite;
- check for undefined citations/references;
- scan for unsupported uses of `unbiased`, `exact`, or OT language that would
  overclaim the target status;
- run `git diff --check`.

## Audit criterion

The phase succeeds only if the rewritten chapter makes the
bias-versus-differentiability trade-off mathematically explicit and distinguishes
clearly among exact resampling, relaxed resampling, and transport-based
relaxations.

## Primary criterion and veto diagnostics

Primary criterion:
- the rewritten chapter is mathematically self-contained and makes the
  differentiability-versus-bias issue explicit enough that the later HMC and
  learned-OT chapters can be written without ambiguity.

Veto diagnostics:
- if the chapter treats OT resampling as if it were identical to exact
  multinomial resampling, stop and revise;
- if the source of bias in soft or OT resampling remains vague, stop and revise;
- if learned/amortized OT begins to dominate the chapter before the OT baseline
  is mathematically clear, do not proceed to the next phase.
