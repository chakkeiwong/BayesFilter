# Phase R3 plan: PF-PF and proposal-correction rewrite

## Date

2026-05-10

## Purpose

This is the third reader-facing rewrite phase of the DPF monograph rebuild.  Its
purpose is to write the mathematically rigorous PF-PF / proposal-correction
chapter that follows the particle-flow foundations chapter.

## Scope

This phase covers only the proposal-corrected particle-flow layer.  It does not
yet rewrite differentiable resampling, OT, or the HMC-assessment chapters.

Target reader-facing file for this phase:
- `docs/chapters/ch19c_dpf_implementation_literature.tex`

The file is to be treated as a rewrite target, not as a draft to patch.

## Mathematical goals

The rewritten chapter must provide a self-contained treatment of:

1. flow transport as a proposal rather than as the final corrected target;
2. proposal-density transformation under a flow map;
3. PF-PF / importance-corrected weights after flow;
4. Jacobian determinant and log-determinant evolution;
5. what the correction restores mathematically;
6. what still remains approximate after correction.

## Required section map

1. **Why proposal correction is needed**
   - explain why raw transport is not enough outside exact special cases.

2. **Flow map as a change of variables**
   - define the map from pre-flow to post-flow particles.

3. **Proposal density under transport**
   - derive the transformed density via the Jacobian determinant.

4. **Corrected importance weights**
   - derive the PF-PF weight formula carefully;
   - distinguish the target density from the proposal density.

5. **Jacobian/log-determinant evolution**
   - derive the auxiliary ODE or equivalent identity for practical evaluation.

6. **Status of the corrected method**
   - explain what correction restores and what remains approximate due to
     discretization, finite particles, or local linearization.

## Source discipline

Primary source families for this phase:
- PF-PF / Li-Coates and related literature;
- CIP LEDH/PF-PF chapter material;
- particle-flow background from the R2 chapter;
- student materials only as coverage/caveat checks, not as the chapter voice.

## Execution rule

The chapter must remain mathematically centered and should not drift into:
- implementation-governance commentary,
- learned resampling / OT material,
- or broad HMC conclusions beyond the value-side interpretation of proposal
  correction.

## Tests

- compile the monograph after the rewrite;
- check for undefined citations/references;
- scan for unsupported exactness / corrected-target language;
- run `git diff --check`.

## Audit criterion

The phase succeeds only if the rewritten chapter:
- makes the change-of-variables and corrected-weight logic explicit;
- clearly distinguishes raw flow from proposal-corrected flow;
- states what is restored and what is still only approximate.

## Primary criterion and veto diagnostics

Primary criterion:
- the rewritten chapter is mathematically self-contained and makes the
  proposal-correction logic precise enough that later HMC-target analysis can be
  written without ambiguity.

Veto diagnostics:
- if the change-of-variables argument is not stated clearly, stop and revise;
- if the chapter still blurs raw flow with corrected PF-PF, stop and revise;
- if the status of the corrected method remains ambiguous (exact vs approximate
  vs finite-particle), do not proceed to the resampling chapter.
