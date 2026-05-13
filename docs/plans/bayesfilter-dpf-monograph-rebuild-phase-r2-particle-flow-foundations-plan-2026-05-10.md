# Phase R2 plan: particle-flow foundations rewrite

## Date

2026-05-10

## Purpose

This is the second reader-facing rewrite phase of the DPF monograph rebuild.
Its purpose is to write the mathematically rigorous particle-flow foundations
chapter that follows the particle-filter baseline.

## Scope

This phase covers only the particle-flow foundations layer.  It does not yet
handle PF-PF proposal correction as a full chapter, and it does not yet rewrite
differentiable resampling, OT, or the HMC-assessment chapters.

Target reader-facing file for this phase:
- `docs/chapters/ch19b_dpf_literature_survey.tex`

The file is to be treated as a rewrite target, not as a draft to polish lightly.
It should become the particle-flow foundations chapter in substance, even if the
filename stays the same temporarily.

## Mathematical goals

The rewritten chapter must provide a self-contained treatment of:

1. the motivation for continuous transport versus discrete reweighting and
   resampling;
2. the homotopy path between predictive and filtering laws;
3. the continuity equation / flow PDE framing;
4. EDH under Gaussian prior / Gaussian likelihood closure;
5. the linear-Gaussian recovery statement as an exact special case;
6. LEDH and local linearization;
7. numerical stiffness and homotopy-discretization implications.

## Required section map

1. **Motivation and chapter position**
   - explain why particle-flow methods are introduced after the particle-filter
     baseline;
   - frame them as transport approximations before proposal correction.

2. **Homotopy density and pseudo-time path**
   - define the interpolating density from predictive law to filtering law.

3. **Continuity equation and flow PDE**
   - derive the conservation law carefully;
   - make the assumptions explicit.

4. **EDH under Gaussian closure**
   - derive or reconstruct the affine EDH ODE;
   - explain where Gaussian closure enters.

5. **Linear-Gaussian recovery**
   - state clearly the exact special-case recovery result.

6. **LEDH and local linearization**
   - define the local Jacobian / local precision form;
   - identify the approximation step.

7. **Stiffness and discretization**
   - explain why low observation noise or sharp likelihoods create stiffness;
   - connect this to later implementation stress tests.

## Source discipline

Primary source families for this phase:
- CIP DPF and LEDH/PF chapters;
- source literature underlying EDH, LEDH, and stiffness discussion;
- student materials only as background checks for coverage or caveats, not as
  the voice of the chapter.

## Execution rule

The chapter must remain mathematically centered.  It must not drift into:
- implementation-governance prose,
- reader-facing repo commentary,
- or PF-PF correction discussion beyond what is needed to explain why a later
  chapter is required.

## Tests

- compile the monograph after the rewrite;
- check for undefined citations/references;
- scan for unsupported exactness or correctness language;
- run `git diff --check`.

## Audit criterion

The phase succeeds only if the rewritten chapter:
- gives a mathematically coherent account of particle flow in its own right;
- states the exact special-case and approximate-status distinctions clearly;
- prepares the reader for PF-PF correction without preempting that later chapter.

## Primary criterion and veto diagnostics

Primary criterion:
- the rewritten chapter is mathematically self-contained, materially stronger
  than the current draft, and explicit about where approximation enters.

Veto diagnostics:
- if the chapter still reads as a broad survey rather than a theory chapter,
  stop and revise;
- if the EDH/LEDH distinction is not mathematically clear, stop and revise;
- if exact-special-case and approximate-flow statements are not separated
  cleanly, do not proceed to the PF-PF rewrite phase.
