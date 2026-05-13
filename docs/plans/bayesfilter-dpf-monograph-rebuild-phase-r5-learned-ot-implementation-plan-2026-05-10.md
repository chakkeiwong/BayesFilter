# Phase R5 plan: learned/amortized OT and implementation-mathematics rewrite

## Date

2026-05-10

## Purpose

This is the fifth reader-facing rewrite phase of the DPF monograph rebuild.  Its
purpose is to write the mathematically disciplined chapter on learned or
amortized OT operators and the implementation-facing mathematics that follows
from treating them as approximations to an already relaxed OT baseline.

## Scope

This phase covers:

- teacher-versus-learned map structure;
- learned or amortized OT as an approximation layer above OT itself;
- approximation residuals and their interpretation;
- training-distribution dependence and extrapolation risks;
- implementation-facing mathematical constraints relevant to later HMC and
  production interpretation.

Target reader-facing file for this phase:
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

This file is to be treated as a rewrite target in substance.  Even though the
filename currently reflects the old assessment draft, this phase will use it for
the learned/amortized OT and implementation-mathematics chapter, since the HMC
assessment chapter is better treated as a later standalone rewrite after the
approximation stack is fully explicit.

## Mathematical goals

The rewritten chapter must provide a self-contained treatment of:

1. a teacher-student view of learned OT;
2. the learned map as approximation to a previously defined OT barycentric map;
3. approximation residuals and how they differ from posterior error;
4. training-distribution restrictions and extrapolation risks;
5. implementation-facing mathematical consequences for dimensions, runtime,
   deterministic seeding, and compiled execution.

## Required section map

1. **Why learned OT is a new approximation layer**
   - state the hierarchy: exact resampling -> OT relaxation -> learned OT
     approximation.

2. **Teacher map and learned map**
   - define the OT teacher map and the learned approximation to it.

3. **Approximation residuals**
   - explain what residuals mean mathematically;
   - distinguish map-approximation residual from posterior-target change.

4. **Training-distribution dependence**
   - explain state-dimension restrictions, weight-geometry restrictions,
     regularization-range dependence, and out-of-distribution risk.

5. **Implementation-facing mathematics**
   - explain what the approximation means for runtime, deterministic evaluation,
     and later HMC-target interpretation.

6. **Chapter boundary**
   - make explicit that the chapter does not yet settle HMC validity;
   - reserve the final rung-by-rung target judgment for the later HMC chapter.

## Source discipline

Primary source families for this phase:
- advanced student repo architecture note and related notebooks as comparison
  inputs;
- OT baseline already established in the previous chapter;
- any primary literature needed to support learned-transport claims.

## Execution rule

The chapter must remain mathematically centered and must not drift into:
- implementation-governance prose,
- a repository walkthrough,
- or a premature HMC verdict that belongs to the later assessment chapter.

## Tests

- compile the monograph after the rewrite;
- check for undefined citations/references;
- scan for unsupported language that conflates learned OT with exact OT;
- run `git diff --check`.

## Audit criterion

The phase succeeds only if the rewritten chapter makes the hierarchy of
approximations explicit and prevents learned/amortized OT from being read as a
free acceleration with no target consequences.

## Primary criterion and veto diagnostics

Primary criterion:
- the rewritten chapter is mathematically self-contained and makes the
  teacher-versus-learned map distinction explicit enough that the later HMC
  chapter can refer to learned OT as a clearly identified surrogate layer.

Veto diagnostics:
- if the chapter treats learned OT as equivalent to the OT baseline, stop and
  revise;
- if approximation residuals are discussed only operationally and not
  mathematically, stop and revise;
- if the chapter starts making final HMC-suitability claims, defer those claims
  to the later assessment chapter.
