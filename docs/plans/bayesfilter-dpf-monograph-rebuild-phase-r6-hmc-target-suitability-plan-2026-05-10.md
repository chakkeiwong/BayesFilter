# Phase R6 plan: HMC-target correctness and structural-model suitability rewrite

## Date

2026-05-10

## Purpose

This is the final major reader-facing rewrite phase of the current DPF monograph
rebuild sequence.  Its purpose is to write the mathematically disciplined HMC-
target and structural-model suitability chapter that draws together the earlier
chapters into a rung-by-rung target analysis.

## Scope

This phase covers:

- what quantity each DPF rung evaluates;
- what quantity each rung differentiates;
- whether the value and gradient correspond to the same target;
- what that implies for HMC;
- why nonlinear DSGE and MacroFinance models make target drift especially
  important.

Target reader-facing file for this phase:
- `docs/chapters/ch21_hmc_for_state_space.tex`

This file will remain the BayesFilter HMC chapter, but this phase will expand it
so that the DPF-target question is explicitly integrated into the chapter's
reader-facing mathematical exposition.

## Mathematical goals

The rewritten material must provide a self-contained treatment of:

1. HMC validity as a target-and-gradient consistency question;
2. rung-by-rung DPF target analysis;
3. exact-target, unbiased-estimator, relaxed-target, and learned-surrogate
   interpretations;
4. nonlinear DSGE structural difficulties;
5. MacroFinance latent-factor and compiled-path difficulties;
6. the BayesFilter recommendation for the first justified HMC-relevant DPF rung.

## Required section map

1. **Target contract for DPF-based HMC**
   - restate that HMC validity depends on the same scalar and gradient pair.

2. **Rung-by-rung DPF target analysis**
   - EDH;
   - EDH/PF with correction;
   - soft resampling;
   - OT resampling;
   - learned/amortized OT.

3. **Why differentiability alone is insufficient**
   - explain the difference between pathwise smoothness and target correctness.

4. **Nonlinear DSGE structural-model stress points**
   - state partitions, determinacy, local-linearization risk, and sensitive
     regions.

5. **MacroFinance stress points**
   - high dimension, mixed frequency, sharp curvature, compiled reproducibility.

6. **BayesFilter recommendation**
   - identify the first justified HMC-relevant development rung.

## Source discipline

Primary source families for this phase:
- the earlier DPF rewrite chapters;
- BayesFilter's existing HMC doctrine;
- the M5 HMC-suitability plan/result;
- only enough student comparison context to sharpen the interpretation, not to
  drive the exposition.

## Execution rule

The chapter must remain mathematically centered and must not drift into:
- generic narrative about sampler usability divorced from the target question;
- implementation-governance commentary;
- broader non-DPF HMC exposition already adequately handled elsewhere.

## Tests

- compile the monograph after the rewrite;
- check for undefined citations/references;
- scan for unsupported HMC-correctness or exact-target wording;
- run `git diff --check`.

## Audit criterion

The phase succeeds only if the chapter makes the rung-by-rung target question
explicit enough that the DPF block has a mathematically coherent endpoint.

## Primary criterion and veto diagnostics

Primary criterion:
- the rewritten HMC material makes the value/gradient/target correspondence
  explicit for each DPF rung and explains why that matters more in nonlinear
  DSGE and MacroFinance models than in toy systems.

Veto diagnostics:
- if the chapter still speaks as if differentiability nearly implies HMC
  validity, stop and revise;
- if the rung-by-rung table is missing or vague, stop and revise;
- if DSGE and MacroFinance difficulties are treated only informally rather than
  structurally, do not declare the phase complete.
