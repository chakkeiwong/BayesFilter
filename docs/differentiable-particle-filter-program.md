# Differentiable Particle Filter Program

## Date

2026-05-08

## Purpose

This note governs the BayesFilter differentiable particle-filter program.  It
summarizes the agreed direction for developing a production-quality reusable
filtering and HMC likelihood component for generic state-space models, DSGE
models, and MacroFinance latent-factor models.

The target is not a one-off research script.  BayesFilter should become the
shared filtering layer that can be called from this repository and adapted by
the downstream projects under `~/python/dsge_hmc` and
`~/python/MacroFinance`.

## Program Goal

Build a TensorFlow / TensorFlow Probability differentiable particle filter
that:

- evaluates a likelihood or explicitly labeled likelihood surrogate for
  nonlinear state-space models;
- exposes gradients with respect to structural parameters for HMC;
- reuses BayesFilter's generic structural state-space model interface;
- supports linear and nonlinear SSM benchmarks before DSGE or MacroFinance
  promotion;
- records, for every algorithm variant, whether the output is exact,
  unbiased Monte Carlo, a biased differentiable surrogate, or a value-only
  engineering approximation.

The final research target is LEDH particle-flow filtering with importance
weights and neural optimal-transport resampling.  The production path must
reach that target by audited intermediate phases rather than implementing the
full stack at once.

The first production implementation rung is TensorFlow/TFP EDH
linear-Gaussian recovery against Kalman references.  It should establish the
flow equations, generic model boundary, value comparison, gradient checks, and
compiled-path feasibility.  It must not claim soft resampling, OT, neural OT,
HMC readiness, or production importance-weight correctness.

## Source Hierarchy

Use the following hierarchy when resolving conflicts:

1. BayesFilter code and documentation define the production API and release
   gates.
2. The CIP monograph provides the theory source, especially:
   - `/home/ubuntu/python/latex-papers/CIP_monograph/main.tex`;
   - `chapters/ch26_differentiable_pf.tex`;
   - `chapters/ch27_ledh_pfpf_neural_ot.tex`;
   - `chapters/ch32_diff_resampling_neural_ot.tex`;
   - nonlinear filtering and HMC chapters that those chapters depend on.
3. `~/python/dsge_hmc` and `~/python/MacroFinance` provide client workflow,
   adapter, model, and benchmark requirements.
4. The student repositories
   `https://github.com/ljw9510/2026MLCOE` and
   `https://github.com/younghwan-cho-dev/advanced_particle_filter` are
   exploratory source material.  They may suggest implementation tactics, but
   they are not correctness references or quality bars until audited.
5. Paper implementations and author code may be used only after their
   assumptions, licenses, and mathematical contracts are recorded.

## Experimental Tracks

The repository may contain quarantined experimental projects under
`experiments/`.  These projects support testing, comparison, and debugging.
They do not define BayesFilter production code.

### Student baseline comparison track

Student code may be copied or referenced only for internal testing and
comparison when permission is recorded.  Its allowed uses are:

- reproducing the students' reported results;
- comparing independent experimental implementations on common fixtures;
- generating baseline tables for filtering behavior, runtime, ESS, and failure
  modes;
- identifying useful algorithmic ideas that may later be reimplemented behind
  BayesFilter contracts.

Student code must not be used as:

- production implementation;
- BayesFilter public API;
- shared production utility;
- HMC target construction;
- correctness certification;
- a direct dependency of any `bayesfilter/` module.

The default status for student code is `comparison_only`.  Any useful idea
must be reimplemented or adapted through BayesFilter-owned code and tests.

### Controlled experimental baseline track

The repository may also contain a clean experimental DPF project that is
implemented by us, inside `experiments/`, for faster debugging and comparison
than the production path allows.  This track may simplify engineering choices
to produce a more consistent experimental baseline, but it remains quarantined.

Allowed uses are:

- implementing small, inspectable EDH/LEDH/PF variants for comparison;
- testing common fixtures and metrics before production APIs are finalized;
- debugging numerical issues, seeding, ESS behavior, and resampling variants;
- producing reproducible experimental reports.

This controlled experimental baseline must not become a production dependency.
Promotion into `bayesfilter/` requires a separate reimplementation or audit
showing that the code satisfies the production API, TensorFlow/TFP, metadata,
testing, and compiled-path gates.

### Quarantine rule

No production module under `bayesfilter/` may import from or depend on
`experiments/`.  Experimental reports may inform plans and tests, but they do
not justify correctness, HMC readiness, or production release claims.

## Canonical API Direction

BayesFilter owns the canonical generic state-space-model interface.  New DPF
code should fit the same interface family as the existing Kalman,
sigma-point, structural, and particle-filter backends rather than introducing
model-specific entry points.

The existing contract to preserve is centered on
`bayesfilter/structural.py`:

- `StatePartition` declares stochastic, deterministic, auxiliary, and external
  state roles;
- `StructuralStateSpaceModel` supplies `initial_mean`, `initial_cov`,
  `innovation_cov`, `observation_cov`, `transition`, and `observe`;
- `FilterRunMetadata` records backend provenance, integration space,
  deterministic-completion policy, approximation label, differentiability
  status, and compiled status.

Every future DPF backend should return both a likelihood-like scalar and
metadata/diagnostics that identify:

- algorithm rung;
- proposal and flow family;
- resampling family;
- differentiability status;
- likelihood status;
- random seed / common-random-number policy;
- TensorFlow eager/compiled status;
- approximation labels and known blockers.

DSGE and MacroFinance should connect through adapters.  The DPF backend should
not bake DSGE timing, MacroFinance factor semantics, or model-specific
economic restrictions into the filter core.

## Algorithm Ladder

The agreed development ladder is:

1. EDH particle flow baseline.
2. EDH particle-flow particle filter with correct importance weights.
3. EDH/PF with differentiable or soft resampling.
4. EDH/PF with optimal-transport resampling.
5. LEDH particle-flow particle filter with neural optimal transport.

This ladder deliberately separates:

- global EDH from local LEDH;
- particle flow mechanics from importance-weight correction;
- differentiability of the resampling map from correctness of the likelihood
  estimator;
- exact or unbiased Monte Carlo claims from biased differentiable surrogate
  claims.

Intermediate phases are allowed to be useful approximations, but they must be
labeled honestly.  No HMC target should be promoted merely because a filter
produces smooth gradients.

## Likelihood and HMC Contract

For each algorithm rung, the documentation and code must state whether the
reported scalar is one of:

- exact likelihood in a special case;
- unbiased likelihood estimator;
- biased differentiable likelihood surrogate;
- value-only diagnostic score;
- unvalidated experimental objective.

HMC integration is allowed only after the value, gradient, differentiability,
and static-shape gates for the relevant rung pass.  The HMC acceptance
criterion on synthetic data is posterior recovery with credible sampler
diagnostics:

- estimated parameters should be in a reasonable predeclared range of the true
  parameters or covered by the planned posterior intervals;
- multi-chain diagnostics should report usable ESS and R-hat close to 1;
- divergences, finite-value failures, and gradient failures must be recorded;
- convergence claims require strict convergence evidence, not smoke tests.

The repo's existing experiment-template language should be reused: smoke tests
show plumbing, medium recovery shows sampler usability, and strict convergence
requires the stronger diagnostic bar.

## Filtering Benchmark Ladder

Pure filtering must be evaluated before HMC.  The benchmark ladder is:

1. Linear-Gaussian SSM:
   - compare filtered means, filtered covariances, and log likelihood against
     Kalman filtering;
   - use this as the required EDH sanity check.
2. Small nonlinear SSM:
   - compare state RMSE, predictive likelihood, finite diagnostics, and
     qualitative posterior behavior against UKF/EKF and existing BayesFilter
     nonlinear filters.
3. Stress nonlinear SSMs:
   - test weight degeneracy, resampling behavior, gradient stability, and
     particle-flow stiffness.
4. DSGE and MacroFinance adapters:
   - promote only after the generic SSM contract passes and model-specific
     adapter semantics are audited.
5. HMC synthetic recovery:
   - run only after the relevant filter has passed value and gradient gates.

The pure-filtering standard is consistency with established nonlinear filters
where no exact reference exists.  It is not, by itself, evidence that HMC is
valid.

## TensorFlow and TFP Policy

The production implementation should use TensorFlow and TensorFlow
Probability.  NumPy backends may remain as reference value tests or simple
baselines, but they do not satisfy the differentiable-HMC target.

The TensorFlow implementation must be designed for:

- differentiability with respect to model parameters;
- common random numbers or other controlled randomness for gradient tests;
- `tf.function` compatibility where feasible;
- static-shape policies compatible with HMC;
- explicit dtype policy;
- stable linear algebra with finite diagnostics;
- no silent Python-side graph breaks in the HMC target path.

## Evidence Labels

Use these labels in plans, audits, metadata, and reset-memo entries:

| Label | Meaning |
| --- | --- |
| `exact_special_case` | Exact for a stated model class such as linear Gaussian. |
| `unbiased_likelihood_estimator` | The likelihood estimator is unbiased under recorded assumptions. |
| `biased_differentiable_surrogate` | Differentiability is obtained by a relaxation that changes the target. |
| `value_only_reference` | Useful for value comparison but not differentiable-HMC promotion. |
| `comparison_only` | Experimental or student code used only for testing and comparison. |
| `filter_consistency_evidence` | Filtering results agree with declared references without sampler claims. |
| `gradient_checked` | Gradients passed finite-difference or independent autodiff checks. |
| `compiled_checked` | The complete value-gradient path passed the planned compiled-shape gate. |
| `hmc_smoke_only` | HMC plumbing ran; no convergence claim is allowed. |
| `sampler_not_justified` | Value or gradient evidence exists, but HMC promotion remains blocked. |
| `sampler_usable` | Medium recovery diagnostics support further HMC experiments. |
| `strict_convergence_evidence` | Predeclared strict convergence diagnostics passed. |
| `source_missing` | Literature/code support has not been found or audited. |
| `human_review_required` | A derivation or claim is not tool-certified and needs written review. |

## Hard Stop Rules

Stop and request direction before a phase would:

- promote a biased differentiable surrogate as an exact likelihood;
- claim HMC convergence from a smoke run;
- use student code as a correctness reference without audit;
- import `experiments/` code from a production `bayesfilter/` module;
- treat student or experimental-baseline agreement as correctness
  certification without Kalman, UKF/EKF, or other declared references;
- change DSGE or MacroFinance economic semantics inside the BayesFilter core;
- bypass the generic SSM adapter boundary;
- add artificial noise to deterministic structural coordinates without an
  approximation label and test plan;
- ignore state-partition metadata when structural deterministic completion is
  required;
- claim TensorFlow/TFP production readiness without compiled value-gradient
  tests;
- proceed to neural OT before EDH/PF value, weighting, and resampling gates
  have passed.

## Immediate Next Phase

The next executable phase is a Phase 1 audit, not implementation.  It should
produce a written source/API/code audit that:

- inventories theory sources, external student repos, and existing BayesFilter
  code;
- maps EDH, importance weighting, soft resampling, OT, and neural OT to
  implementation obligations;
- states the generic DPF API contract;
- defines the first benchmark fixtures and tolerances;
- records blockers before TensorFlow/TFP code is added.
