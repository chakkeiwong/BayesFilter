# BayesFilter literature seed list

## Date
2026-05-02

## Purpose
This note seeds the ResearchAssistant literature pass for the BayesFilter
monograph. It is not an approved bibliography and should not be treated as
claim support by itself.

The immediate goal is to turn candidate references from the source projects
into reviewed local summaries before drafting literature-heavy chapters.

## ResearchAssistant workspace

Temporary workspace used in this execution pass:

```bash
PYTHONPATH=/home/chakwong/research-assistant/src \
python -m research_assistant.cli --root /tmp/ra-bayesfilter-monograph init
```

```bash
PYTHONPATH=/home/chakwong/research-assistant/src \
python -m research_assistant.cli --root /tmp/ra-bayesfilter-monograph doctor
```

The workspace is in offline mode and has local PDF parsing support. Because it
is under `/tmp`, the durable state for this phase is this seed list plus the
reset memo. A production literature workspace should be created under a
persistent BayesFilter-adjacent location before long-form drafting.

## Source bibliographies to audit

- `/home/chakwong/python/docs/references.bib`
- `/home/chakwong/latex/CIP_monograph/references.bib`
- `/home/chakwong/MacroFinance/analytic_kalman_derivatives.bib`
- Current BayesFilter seed bibliography: `docs/references.bib`

Do not merge source bibliographies wholesale. Merge candidate entries only
after checking key conflicts, duplicate spellings, and whether the cited paper
actually supports the claim being made.

## Core seed topics

### Linear Gaussian state-space filtering

Required before drafting:
- prediction-error decomposition;
- diffuse initialization;
- numerically stable Kalman filtering;
- square-root and QR-based filtering;
- missing-data and mixed-frequency filtering;
- large-scale LGSSM complexity.

Candidate source keys observed locally:
- `kalman1960`
- Durbin and Koopman state-space text entries
- Anderson and Moore optimal filtering entries
- Koopman/Durbin fast filtering entries

### Analytic likelihood derivatives

Required before drafting:
- score recursion for Kalman likelihood;
- Hessian or observed-information recursion;
- structural derivatives from parameter maps into state-space matrices;
- solve-form, Cholesky, QR, and square-root derivative formulations;
- validation standards against finite differences and autodiff.

Candidate source keys observed locally:
- the MacroFinance analytic derivative bibliography includes papers on
  gradient and Hessian computation for state-space log likelihoods.

### Sigma-point and SVD filters

Required before drafting:
- unscented transform and square-root UKF;
- cubature Kalman filtering;
- central-difference or sigma-point filtering in DSGE settings;
- numerical stability of factorized covariance updates;
- differentiability/eigen-gap limitations of eigendecomposition and SVD;
- when a sigma-point likelihood is HMC-safe, approximate-only, or diagnostic.

Candidate source keys observed locally:
- `wan2000unscented`
- `julier2004unscented`
- `arasaratnam2009cubature`
- van der Merwe square-root UKF entries
- DSGE nonlinear-filter comparison entries

### Particle filters and pseudo-marginal methods

Required before drafting:
- unbiased likelihood estimators for particle MCMC;
- pseudo-marginal MCMC conditions;
- particle filter degeneracy in high dimension;
- differentiable resampling and entropy-regularized optimal transport;
- distinction between differentiating an approximate likelihood and sampling
  from a correct pseudo-marginal target.

Candidate source keys observed locally:
- `andrieu2010particle`
- pseudo-marginal entries by Andrieu and Roberts
- particle degeneracy/high-dimensional filtering entries
- `corenflos2021differentiable`
- filtering variational objectives entries

### HMC, NUTS, and geometry

Required before drafting:
- Hamiltonian dynamics and leapfrog error;
- NUTS and adaptation;
- divergences and geometry diagnostics;
- Riemannian manifold HMC;
- mass matrices and observed-information preconditioning;
- boundary handling for finite log targets and finite gradients.

Candidate source keys observed locally:
- `neal2011mcmc`
- `hoffman2014nuts`
- `betancourt2017conceptual`
- Girolami and Calderhead Riemann-manifold HMC entries
- delayed-rejection and continuously tempered HMC entries

### Transport maps, NeuTra, and surrogates

Required before drafting:
- transport map accelerated MCMC;
- neural transport/NeuTra;
- normalizing-flow or OT-flow surrogates;
- delayed-acceptance correction when using approximate Hamiltonian or neural
  surrogates;
- explicit bias budget for surrogate and filter approximation layers.

Candidate source keys observed locally:
- `hoffman2019neutra`
- Parno and Marzouk transport-map entries
- OT-flow entries
- Hamiltonian neural network entries

## Claim-support policy

Before a BayesFilter chapter makes one of the following claims, there must be
an approved ResearchAssistant note or manual source note:

- a filter is exact, unbiased, stable, or HMC-safe;
- a gradient is valid for the target sampled by HMC;
- a particle-filter estimator yields a valid pseudo-marginal target;
- an SVD/eigendecomposition path is differentiable enough for HMC in a
  specified dimension/model class;
- a runtime, memory, or scaling threshold is production-relevant;
- a literature method dominates another method for a stated class of models.

## Phase-2 gate outcome

Current status:
- Workspace setup: passed.
- Local reviewed summaries: not yet available.
- Citation-neighborhood notes: not yet available.
- Literature-heavy drafting: blocked.

Interpretation:
- The next safe work is either a dedicated literature ingestion/review phase or
  a scoped internal-only drafting phase limited to provenance, notation, and
  implementation contracts with no unsupported literature claims.
