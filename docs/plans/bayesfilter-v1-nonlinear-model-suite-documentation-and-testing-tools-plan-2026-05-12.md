# BayesFilter V1 Nonlinear Model Suite Documentation And Testing Tools Plan

## Date

2026-05-12

## Parent Plan

```text
docs/plans/bayesfilter-v1-nonlinear-filtering-master-testing-program-2026-05-11.md
```

## Motivation

The nonlinear filters need a benchmark suite that is small, cited, and useful
for debugging.  The goal is not to collect every standard state-space model.
The goal is to build reusable BayesFilter-local fixtures that expose exact
linear recovery, deterministic structural completion, nonlinear transition
error, nonlinear observation error, and reference-oracle limitations.  Angle
residuals are a later tracking-model issue, not a first-rung execution target.

This subplan covers the first program stream: document the accepted nonlinear
models in the monograph and implement them as general testing tools.  It does
not implement analytic gradients and it does not switch MacroFinance or DSGE
clients to BayesFilter.

## Lane

Allowed write lane:

```text
bayesfilter/testing/*
tests/test_nonlinear_*
tests/test_*sigma*
docs/chapters/ch28_nonlinear_ssm_validation.tex
docs/references.bib
docs/plans/bayesfilter-v1-*
docs/source_map.yml
```

Protected unless explicitly requested:

```text
docs/chapters/ch18b_structural_deterministic_dynamics.tex
docs/plans/bayesfilter-structural-*
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

## Scope

In scope:
- literature verification and model documentation;
- reusable TF model builders and deterministic data generators for tests;
- dense-reference oracles for small one-dimensional models;
- high-particle SMC reference policy design, if needed, as an optional
  diagnostic artifact;
- no production NumPy in BayesFilter implementation modules.

Out of scope:
- analytic sigma-point gradient implementation;
- HMC readiness claims;
- general SMC reference implementation;
- angle residual or non-additive observation-noise API design;
- GPU benchmarks except optional later gates from the parent plan;
- external-project coupling.

## Drift Check

This phase must stay on the first-rung model suite.  It should not implement
analytic gradients, tune HMC, benchmark GPU/XLA throughput, change production
filter APIs, or promote Models D-F.  The only acceptable production-package
edits are reusable testing fixtures under `bayesfilter/testing`; value filter
implementation changes belong to the remaining-master-program subplan.

## Accepted First-Rung Models

### Model A: Affine Gaussian Structural Oracle

Purpose:
- exact linear Gaussian recovery;
- deterministic structural completion;
- singular structural innovation support.

Implementation target:
- `bayesfilter/testing/nonlinear_models_tf.py`;
- `make_affine_gaussian_structural_oracle_tf`;
- exact Kalman reference through existing linear/structural test helpers.

Closure:
- SVD cubature, SVD-UKF, and SVD-CUT4 match the exact Kalman likelihood within
  tolerance;
- deterministic residual and support residual are recorded.

### Model B: BayesFilter Nonlinear Accumulation

Purpose:
- smooth nonlinear structural transition with deterministic completion:
  \[
    m_t=\rho m_{t-1}+\sigma\varepsilon_t,\qquad
    k_t=\alpha k_{t-1}+\beta\tanh(m_t),
  \]
  \[
    y_t=m_t+k_t+\eta_t.
  \]

Implementation target:
- `make_nonlinear_accumulation_model_tf`;
- short-horizon deterministic observation fixtures;
- dense one-step reference oracle.

Closure:
- value tests are finite for all three sigma-point backends;
- one-step mean/covariance moments match a dense reference at the documented
  tolerance;
- deterministic identity residual is explicit.

### Model C: Univariate Nonlinear Growth Model

Purpose:
- standard nonlinear transition and quadratic observation benchmark:
  \[
    x_t=\frac{x_{t-1}}{2}
      +\frac{25x_{t-1}}{1+x_{t-1}^2}
      +8\cos(1.2t)+\sigma_x\varepsilon_t,
  \]
  \[
    y_t=\frac{x_t^2}{20}+\sigma_y\eta_t.
  \]

Literature status:
- verify exact equation, indexing convention, and parameter settings from a
  primary or standard source before promoting the implementation;
- candidate sources from the parent plan include Kitagawa (1996) and the SMC
  tutorial literature.

Implementation target:
- `make_univariate_nonlinear_growth_model_tf`;
- autonomous testing embedding with a deterministic phase coordinate
  \(\tau_t=\tau_{t-1}+1\), because the current structural TF filter interface
  does not pass an explicit time index into `transition_fn`;
- dense quadrature oracle for short horizons.

Closure:
- LaTeX block states the equation, source, parameter defaults, and reference
  oracle;
- tests compare filter likelihood and filtered moments against dense quadrature
  on small horizons.

## Deferred Models

Model D, bearings-only tracking:
- document after angle residual and wrapping policy are fixed;
- implementation requires an explicit circular-residual test oracle.

Model E, radar range-bearing tracking:
- document after bearings-only residual policy is stable;
- implementation should test mixed-scale observation covariance and
  cross-covariance diagnostics.

Model F, stochastic volatility:
- defer until the nonlinear observation-noise interface supports non-additive
  observation laws or an explicit approximation contract.

## Phase Plan

### Phase M0: Lane Recovery And Source Inventory

Actions:
- run `git status --short --branch`;
- confirm active dirty files are inside the v1 lane before editing;
- use local research-assistant first for paper summaries;
- use web/DOI verification only if the local index cannot support the model
  equations.

Primary criterion:
- all planned writes remain inside the allowed lane.

Veto diagnostics:
- required source evidence is absent;
- a model requires external-project changes;
- a model requires production NumPy.

### Phase M1: Literature Verification Notes

Actions:
- create one short source note per accepted model in this plan or a result
  artifact;
- record exact equations, parameter defaults, citation key/DOI, and whether the
  equation was verified from a primary source;
- state which oracle is used: exact Kalman, dense quadrature, or SMC.

Primary criterion:
- Models A-C have enough documented support to be written in Chapter 28.

Veto diagnostics:
- the univariate growth equation cannot be verified;
- a cited paper does not contain the stated model;
- source notes cannot distinguish model law from approximation policy.

### Phase M2: Chapter 28 Documentation

Actions:
- add a compact nonlinear benchmark section to
  `docs/chapters/ch28_nonlinear_ssm_validation.tex`;
- add missing bibliography entries only for accepted benchmark citations;
- include law, noise distribution, parameter defaults, oracle, and stress
  property for each accepted model;
- add a small status table marking Models D-F as deferred and why.

Primary criterion:
- every implemented model has a LaTeX model block and oracle statement.

Veto diagnostics:
- documentation implies HMC readiness or production client switch-over;
- documentation omits the reference law for an implemented test fixture.

### Phase M3: General Testing Tools

Actions:
- implement model builders under `bayesfilter/testing/`;
- implement deterministic observation fixtures and small reference oracles;
- keep fixtures parameterized and reusable by value, score, branch, and
  benchmark tests.
- implement Model C as an autonomous phase-state fixture rather than changing
  the production structural filter signature.

Primary criterion:
- tests can construct Models A-C without copying model equations into each
  test file.

Veto diagnostics:
- fixture constructors import MacroFinance or DSGE code;
- implementation uses NumPy inside production `bayesfilter/nonlinear/*`;
- the dense oracle is mislabeled as the exact nonlinear likelihood when it is
  only a moment-matched Gaussian projection reference;
- random seeds or generated observations are not recorded.

### Phase M4: Value And Oracle Tests

Target tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  -p no:cacheprovider
```

Primary criterion:
- Models A-C pass constructor, value, and oracle checks on CPU.

Veto diagnostics:
- nonfinite likelihoods on documented parameter defaults;
- deterministic/support residuals not reported;
- eager/graph parity fails for model constructors or value paths.

### Phase M5: Provenance And Reset-Memo Update

Actions:
- register artifacts in `docs/source_map.yml`;
- update the v1 reset memo with results, interpretation, and next-phase
  justification.

Primary criterion:
- a future agent can recover the model suite status from the reset memo and
  source map.

## Hypotheses To Test

H-M1:
Models A-C are enough to catch the first correctness failures before we add
tracking or stochastic-volatility examples.

H-M2:
Dense quadrature can serve as a stable oracle for the nonlinear accumulation
and univariate growth models at short horizons.

H-M3:
The model builders can be shared by value tests, score tests, branch tests, and
benchmarks without duplicating equations.

H-M4:
Angle and non-additive observation-noise models should stay deferred until
their residual/noise contracts are explicit.

## Done Definition

This subplan is complete when:
- Models A-C are documented in Chapter 28 with equations, citations or source
  notes, and oracles;
- reusable model builders and reference tools exist under `bayesfilter/testing`;
- value/oracle tests pass on CPU;
- source-map and reset-memo entries record the status and next justified step.
