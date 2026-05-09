# Phase R1 plan: reader-facing rewrite execution for particle-filter foundations

## Date

2026-05-09

## Purpose

This is the first actual reader-facing rewrite phase of the DPF monograph
rebuild.  Its purpose is to replace the current short gateway chapter on
particle filters with a mathematically serious particle-filter foundations
chapter that can serve as the baseline for all later DPF chapters.

## Scope

This phase covers only the particle-filter foundations layer.  It does not yet
rewrite the particle-flow, PF-PF, resampling, or HMC-assessment chapters.

Target reader-facing file:
- `docs/chapters/ch19_particle_filters.tex`

Related files that may need small supporting edits in this phase only if the
rewrite requires them:
- `docs/main.tex` (only if chapter title/ordering metadata needs adjustment)
- `docs/references.bib` (only if the rewritten chapter requires currently
  missing bibliography entries)
- `docs/source_map.yml` (to record the rewrite provenance if needed)
- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`
- one phase result/audit note under `docs/plans/`

## Mathematical goals

The rewritten chapter must provide a self-contained baseline treatment of:

1. nonlinear state-space model and filtering recursion;
2. empirical filtering measures;
3. sequential importance sampling and resampling;
4. bootstrap particle-filter likelihood estimator;
5. unbiasedness status of the bootstrap PF likelihood estimator;
6. weight degeneracy, effective sample size, and dimensionality stress.

## Required section map

1. **State-space and filtering setup**
   - define latent state, observations, predictive law, filtering law, and
     marginal-likelihood factorization.

2. **Empirical particle approximation**
   - define the weighted atomic approximation;
   - define approximation of posterior expectations.

3. **Sequential importance sampling**
   - derive the weight update carefully;
   - distinguish proposal density and target density.

4. **Sequential importance resampling and bootstrap PF**
   - define the bootstrap choice of proposal;
   - derive the corresponding likelihood estimator.

5. **Likelihood-estimator status**
   - state the estimator-status claim precisely and with scope.

6. **Degeneracy and dimensionality**
   - explain ESS and collapse rigorously enough to motivate later chapters.

## Source discipline

Primary source families to use in this phase:
- bootstrap PF / SMC sources already identified in Phase M1;
- CIP DPF chapter where it discusses bootstrap PF;
- student materials only as background comparison, not as the voice of the
  chapter.

## Execution rule

The rewritten chapter must read like a mathematical monograph chapter, not like
an internal memo and not like a commentary on student work.

## Tests

- compile the monograph after the chapter rewrite;
- check for undefined citations/references;
- scan for unsupported high-stakes words such as exact/unbiased where they are
  not source-qualified;
- run `git diff --check`.

## Audit criterion

The phase succeeds only if the new chapter stands on its own mathematically and
prepares the reader for the later flow/PF-PF/resampling chapters without
introducing HMC claims too early.

## Primary criterion and veto diagnostics

Primary criterion:
- the rewritten chapter is mathematically self-contained, uses correct source
  support for the bootstrap PF and likelihood-estimator status, and improves
  materially on the prior short gateway treatment.

Veto diagnostics:
- if the rewrite still reads mainly as program commentary rather than a theorem-
  and derivation-based chapter, stop and revise before moving on;
- if the estimator-status discussion cannot be supported clearly from sources,
  stop and record the blocker;
- if compilation fails in a way that suggests the rewrite introduced unresolved
  notation or citation gaps, stop and fix before declaring the phase complete.
