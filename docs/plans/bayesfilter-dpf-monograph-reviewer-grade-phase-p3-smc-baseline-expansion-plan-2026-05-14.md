# Phase P3 plan: classical filtering and SMC baseline expansion

## Date

2026-05-14

## Target chapter

- `docs/chapters/ch19_particle_filters.tex`

## Governing prerequisites and lane guard

- Governing setup results:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-result-2026-05-15.md`
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`
- P2 established bibliography-spine support only; do not claim
  ResearchAssistant-reviewed SMC source support unless a later artifact records
  it.
- Allowed write set: this target chapter and the P3 result artifact.  Touch
  `docs/references.bib` only if a citation check finds a missing required key
  and the result records the source identity check.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.
- P4/P5 provisional results dated `2026-05-15` may already be present.  Do not
  revert them.  At the end of P3, record whether the new baseline definitions
  require a P4/P5 repair before P6.

## Purpose

Make the classical filtering and bootstrap particle-filter baseline strong
enough that all later differentiable constructions can be judged against a
clear reference object.

## Required implementation instructions

1. Add a notation/object inventory covering:
   - latent state;
   - observation;
   - parameter;
   - transition density;
   - observation density;
   - predictive law;
   - filtering law;
   - marginal likelihood factor;
   - proposal law;
   - normalized and unnormalized weights;
   - empirical measure;
   - bootstrap likelihood estimator.
2. Expand the filtering recursion:
   - derive prediction from the Chapman-Kolmogorov identity;
   - derive update from Bayes' rule;
   - derive marginal likelihood factorization.
3. Expand SIS derivation:
   - trajectory target;
   - trajectory proposal;
   - unnormalized weight;
   - recursive incremental weight;
   - bootstrap proposal specialization.
4. Strengthen the likelihood-estimator proposition:
   - state assumptions;
   - define filtration or conditioning variables;
   - show conditional expectation step;
   - explain product/telescoping argument;
   - distinguish unbiased likelihood estimator from unbiased log likelihood.
5. Add a subsection on differentiability:
   - resampling pathwise discontinuity;
   - difference between differentiating a realized path and estimating a score;
   - why later DPF methods change the object or the estimator.
6. Add an audit table:
   - exact identity;
   - Monte Carlo estimator claim;
   - asymptotic or variance claim;
   - implementation diagnostic.

## Required source use

- SMC textbooks and bootstrap particle-filter sources.
- Pseudo-marginal references only for likelihood-estimator interpretation, not
  for DPF-HMC conclusions.
- High-dimensional degeneracy sources for ESS/dimensionality warnings.
- Because P2 found no local ResearchAssistant summaries, use these sources as
  bibliography-spine provenance and derive load-bearing estimator statements
  locally in BayesFilter notation.

## Mathematical audit rules

- Every probability density must state conditioning variables.
- Every estimator claim must specify whether it concerns likelihood, log
  likelihood, filtering expectation, or posterior.
- Do not claim pathwise differentiability from unbiased likelihood estimation.
- Do not imply finite-particle exactness.

## Required local tests/checks

- Confirm all new labels are unique.
- Confirm all new citations exist in `docs/references.bib`.
- Search for `unbiased` and verify every use has assumptions and target object.
- Search for `exact` and verify it refers only to model identities or explicitly
  named special cases.
- Create a P3 result artifact with an actual completion date in the filename.
- Record a derivation-obligation table for the filtering recursion, SIS weights,
  and likelihood-estimator status.  Mark each obligation as `MathDevMCP`,
  `manual`, or `blocked`, and explain any skipped MathDevMCP check.
- Run the established LaTeX build and record DPF-local warnings, or record the
  bounded reason the build could not be run.
- Add a P4/P5 impact note: unchanged, needs narrow repair, or blocks P6.

## Veto diagnostics

The phase fails if:

- the likelihood-estimator proof remains only a citation-backed sketch;
- conditioning is ambiguous in the SIS derivation;
- unbiased likelihood and log-likelihood are blurred;
- resampling differentiability is treated informally;
- later DPF chapters still rely on undefined baseline objects.
- the phase edits or stages student-baseline files;
- ResearchAssistant-empty source status is converted into source-reviewed
  support;
- MathDevMCP/manual derivation-obligation evidence is omitted;
- build status is omitted.
- the P4/P5 impact note is omitted while provisional P4/P5 results are present.

## Exit gate

The chapter is ready only if a mathematically mature non-specialist can
identify the exact filtering target, the particle approximation, the bootstrap
likelihood estimator, and the differentiability gap without reading another
paper.
