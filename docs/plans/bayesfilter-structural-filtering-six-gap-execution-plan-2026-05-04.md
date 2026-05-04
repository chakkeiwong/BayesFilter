# Plan: structural filtering six-gap execution pass

## Date

2026-05-04

## Purpose

This plan closes the six immediate documentation and contract gaps after the
worked structural UKF example:

1. state the general structural-filtering doctrine;
2. add a reusable algorithmic pattern;
3. connect the pattern explicitly to DSGE timing;
4. define the BayesFilter implementation requirements;
5. write executable tests that guard the doctrine;
6. keep the implementation path centered in BayesFilter before DSGE or
   MacroFinance adapters consume it.

The goal is not to claim NK, SGU, EZ, NAWM, or MacroFinance HMC convergence.
The goal is to prevent future agents from reintroducing the structural bug:
injecting independent stochastic variation into deterministic-completion
coordinates when a model declares those coordinates to be completed by a
transition map.

## Motivation

The monograph now contains a worked UKF example, but examples are fragile
unless they become a reusable contract.  Large DSGE and macro-finance state
vectors contain mixed roles: shock-driven coordinates, deterministic lag or
accounting coordinates, and model-specific auxiliary coordinates.  Exact
linear Kalman filtering can often consume a collapsed singular transition law,
but nonlinear filters need the structural transition map.  Sigma points and
particles must live on the declared pre-transition uncertainty variables, then
the deterministic-completion block must be computed point by point.

BayesFilter should own this common rule because the same issue appears in
DSGE, AR(p) lag stacks, mixed-frequency aggregation states, and MacroFinance
models.  Client projects should expose structural metadata and maps through
adapters instead of maintaining private filter forks.

## Current baseline

- `docs/chapters/ch18b_structural_deterministic_dynamics.tex` contains a
  structural UKF worked example and a DSGE warning chapter.
- `bayesfilter/structural.py` contains partition/config/metadata contracts.
- `bayesfilter/filters/sigma_points.py` contains a NumPy reference structural
  SVD/cubature sigma-point backend over `(previous_state, innovation)`.
- `bayesfilter/testing/structural_fixtures.py` contains AR(2) and nonlinear
  accumulation fixtures.
- `tests/test_structural_ar_p.py`, `tests/test_structural_partition.py`,
  `tests/test_filter_metadata.py`, and `tests/test_structural_sigma_points.py`
  already cover several structural gates.

The pass should build on that baseline rather than duplicate it.

## Non-goals

- Do not rewrite DSGE or MacroFinance internals in this pass.
- Do not copy client-project model logic into BayesFilter core.
- Do not promote full-state mixed-model sigma-point approximations without an
  explicit approximation label.
- Do not promote tape gradients through SVD/eigen decompositions to HMC.
- Do not add a particle filter beyond fail-closed placeholders.
- Do not commit generated PDFs or LaTeX byproducts.

## Global execution cycle

Each phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Stop and ask for direction if a phase can pass only by adding artificial
independent noise to deterministic-completion coordinates, by dropping
structural metadata, or by making an unsupported convergence or derivative
claim.

## Phase S0: reset memo setup and baseline validation

### Actions

1. Record current dirty state.
2. Record the plan and independent audit.
3. Run the baseline test suite before edits.

### Tests

- `git status --short`
- `pytest -q`

### Gate

Continue if tests pass and unrelated dirty files can be kept out of the scoped
commit.

## Phase S1: doctrine and reusable algorithm

### Actions

1. Add a compact general doctrine section to Chapter 18b:
   - filters integrate over declared uncertainty variables;
   - deterministic-completion coordinates are model outputs, not new
     stochastic coordinates;
   - full-state mixed-model integration is a declared approximation.
2. Add an algorithm box for the structural nonlinear filtering pattern:
   - form a representation of `p(x_{t-1}|y_{1:t-1})`;
   - augment with current innovations or stochastic-state variables;
   - propagate each point/particle through the structural map;
   - evaluate observations;
   - perform the selected Gaussian or simulation update;
   - record metadata.

### Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- targeted grep for the new algorithm label.

### Gate

Continue if the monograph builds and the new section does not duplicate the
worked example.

## Phase S2: DSGE connection and BayesFilter API contract

### Actions

1. Strengthen the DSGE section so it states the adapter contract:
   - structural timing partition `(m,k)`;
   - innovation dimension;
   - `T_m`, `T_k`, observation map, and measurement law;
   - backend integration-space declaration.
2. Strengthen Chapter 4 API text with implementation requirements:
   - fail-closed metadata validation;
   - approximation-label propagation;
   - deterministic identity diagnostics;
   - static-shape and derivative-policy hooks for later HMC.

### Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- targeted grep for `deterministic identity diagnostics` and
  `structural_filter_step`.

### Gate

Continue if the docs distinguish exact structural transitions from labeled
approximations without claiming production readiness.

## Phase S3: executable UKF worked-example regression

### Actions

1. Add a fixture matching the chapter's UKF example:
   - `m_t = rho m_{t-1} + sigma epsilon_t`;
   - `k_t = phi k_{t-1} + gamma m_t^2`;
   - `y_t = m_t + k_t + e_t`.
2. Add a test that reproduces the chapter's one-step structural cubature/UKF
   numbers:
   - predicted mean;
   - predicted covariance;
   - predicted observation moments;
   - gain;
   - posterior mean/covariance;
   - log likelihood;
   - off-manifold artificial-noise contrast.
3. Add a pointwise deterministic identity test for the propagated sigma points.

### Tests

- `pytest -q tests/test_structural_sigma_points.py`
- `pytest -q`

### Gate

Continue if the documentation example has an executable numerical oracle.

## Phase S4: implementation contract audit and tidy

### Actions

1. Audit `StatePartition`, `StructuralFilterConfig`, `FilterRunMetadata`, and
   `StructuralSVDSigmaPointFilter` against the six-gap contract.
2. Add only small code comments or diagnostics if a gap is concrete.
3. Avoid broad refactors.

### Tests

- `pytest -q`
- `git diff --check`

### Gate

Continue if the code remains dependency-light, fail-closed, and adapter-neutral.

## Phase S5: final validation, reset memo, commit

### Actions

1. Run final tests:
   - `pytest -q`;
   - `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml',
     encoding='utf-8'))"`;
   - `git diff --check`;
   - `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
2. Update the reset memo with results, interpretation, and next hypotheses.
3. Stage only files belonging to this pass.
4. Commit the scoped pass.

### Gate

Commit only if tests pass and unrelated dirty files remain unstaged.

## Independent hypotheses to test later

- H-S1: Structural sigma-point propagation over `(x_{t-1}, epsilon_t)` avoids
  the off-manifold likelihood distortion shown in the worked UKF example.
- H-S2: AR(p), DSGE, and MacroFinance lag/accounting states can share the same
  `StatePartition` and deterministic-completion contract.
- H-S3: Full-state mixed-model sigma-point approximations can be useful only
  when visibly labeled and validated against structural references.
- H-S4: SVD numerical robustness is orthogonal to structural correctness; SVD
  factors do not repair a wrong integration space.
- H-S5: HMC promotion should remain blocked until value, derivative,
  compiled/eager, and sampler diagnostics pass on this structural target.
