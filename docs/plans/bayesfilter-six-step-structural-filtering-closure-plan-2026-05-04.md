# Plan: six-step BayesFilter structural filtering closure

## Date

2026-05-04

## Purpose

This plan turns the current BayesFilter status into an executable closure
roadmap for the six remaining workstreams:

1. audit before implementation;
2. analytic Kalman derivatives;
3. structural filtering implementation;
4. SVD sigma-point structural filter;
5. validation ladder;
6. literature and monograph completion.

The plan is written for another agent to audit and execute later.  It is not a
license to skip derivation checks, source-code audits, or model-specific
adapter gates.

## Motivation

BayesFilter is now the intended common infrastructure project for Bayesian
estimation of structural state-space models.  The key technical mistake to
avoid is treating all latent coordinates as exchangeable noisy Gaussian states
when the model actually has stochastic shock-driven coordinates and
deterministic endogenous, lag, accounting, or mixed-frequency coordinates.

The documentation now has an exact linear Gaussian likelihood spine and a
structural state-partition contract.  The next work must turn those documents
into a tested package while keeping three principles intact:

- exact linear Gaussian filtering remains the value oracle;
- nonlinear structural filtering propagates uncertainty over innovation or
  stochastic-state space and completes deterministic coordinates pointwise;
- no derivative or HMC claim is promoted until value, derivative, compiled, and
  stress gates pass.

## Current baseline

As of this planning pass:

- Latest committed BayesFilter documentation commit:
  `466de70 Strengthen BayesFilter linear Gaussian likelihood spine`.
- A candidate BayesFilter package and tests are present in the working tree:
  - `bayesfilter/structural.py`;
  - `bayesfilter/filters/kalman.py`;
  - `bayesfilter/filters/sigma_points.py`;
  - `bayesfilter/filters/particles.py`;
  - `bayesfilter/testing/structural_fixtures.py`;
  - `tests/test_*.py`.
- The candidate package passes `pytest -q` with 15 tests.  The only warning in
  the sandboxed run was a pytest-cache write warning.
- A candidate source/code audit note exists at
  `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`.

The current candidate package is useful as a baseline.  It is not yet a final
industrial or HMC-ready implementation.

## Non-goals

- Do not edit `/home/chakwong/python`, `/home/chakwong/MacroFinance`, or
  `/home/chakwong/latex/CIP_monograph` except by explicit adapter work in a
  later phase.
- Do not claim NK, Rotemberg, SGU, EZ, or NAWM nonlinear convergence.
- Do not claim SVD-gradient safety for HMC.
- Do not port MacroFinance derivative code without a proof-obligation and code
  audit.
- Do not turn particle filtering on until proposal, correction, resampling,
  differentiability, and likelihood-estimator semantics are audited.
- Do not commit generated PDFs or LaTeX byproducts.

## Global execution cycle

Every phase must follow:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo -> commit when coherent
```

The reset memo is:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
```

## Global stop rules

Stop and ask for direction if any of these occur:

- a derivation source cannot establish whether a path is exact, approximate, or
  unresolved;
- a test can pass only by adding artificial noise to deterministic coordinates;
- a mixed nonlinear model lacks structural metadata or deterministic-completion
  maps;
- a full-state sigma-point path is required for a mixed model without an
  explicit approximation label;
- derivative parity fails and the failure cannot be localized;
- SVD/eigen derivative claims rely on unverified singular/eigen gap behavior;
- a client adapter requires changing model economics or finance logic in the
  client repo;
- a phase would commit generated PDFs, downloaded papers, or local tool state.

## Workstream A: audit before implementation

### Objective

Make source derivations, code provenance, exact/approximate classifications,
and reuse decisions explicit before expanding the package.

### Inputs

- `docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md`
- `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`
- `docs/source_map.yml`
- `/home/chakwong/python/docs/monograph.tex`
- `/home/chakwong/latex/CIP_monograph/main.tex`
- `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`
- Candidate code under `/home/chakwong/python/src/dsge_hmc`
- Candidate code under `/home/chakwong/MacroFinance`

### Implementation instructions

1. Convert the current source/code audit note into a ratified gate document:
   - list source labels and chapters actually inspected;
   - list code paths actually inspected;
   - preserve the backend classification table;
   - state exact, approximate, blocked, and reuse decisions.
2. Use ResearchAssistant where paper or monograph citation support is needed.
3. Use MathDevMCP for small labeled proof obligations and assumption surfacing.
4. Treat tool failures as audit limitations, not as silent success.
5. Add a "gate decision" at the end of every audit note.

### Tests

- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"`
- `git diff --check`
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- targeted search for unsupported claims:

  ```bash
  rg -n "guarantee|certified|production-ready|industrial-ready|converged|will always|must always" docs
  ```

### Pass gate

- The audit note is committed.
- Every backend path has a reuse decision.
- Implementation phases that depend on unresolved math are blocked or scoped to
  fail-closed placeholders.

## Workstream B: analytic Kalman derivatives

### Objective

Consolidate analytic score and Hessian recursions for exact linear Gaussian
state-space models without changing the Chapter 5 likelihood value contract.

### Implementation instructions

1. Write a derivation reconciliation note before code:
   - map MacroFinance notation into BayesFilter notation;
   - identify the scalar likelihood differentiated;
   - list all parameter-dependent objects;
   - state initialization derivative policy;
   - state singular-`Q` assumptions and innovation regularity assumptions.
2. Split derivations into small labeled obligations:
   - prediction mean/covariance differential;
   - innovation differential;
   - solve-form log determinant differential;
   - quadratic-form differential;
   - filtered mean/covariance differential;
   - second differential/Hessian recursions;
   - initialization and Lyapunov derivative cases.
3. Implement value/score/Hessian providers only after the derivation note and
   code-reuse decision pass.
4. Prefer solve-form derivatives over explicit inverses in production paths.
5. Keep covariance-form derivatives as small reference tests if useful.

### Tests

- finite-difference gradient parity on tiny dense LGSSM cases;
- finite-difference gradient parity on singular-`Q` AR(p) cases;
- Hessian symmetry checks;
- finite-difference-on-gradient Hessian parity on small cases;
- missing-data and all-missing-step derivative checks;
- eager/compiled parity when a compiled backend exists;
- nonfinite diagnostic localization by time step and matrix block.

### Pass gate

- Score and Hessian match numerical references within stated tolerances on
  small cases.
- Failure diagnostics identify where parity breaks.
- Derivatives are not used for HMC until the derivative validation gate passes.

## Workstream C: structural filtering implementation

### Objective

Turn the structural state-partition contract into a small BayesFilter package
with fail-closed metadata validation, exact LGSSM reference filtering, and toy
fixtures.

### Current candidate implementation

The current working tree already contains a candidate implementation:

- `StatePartition`, `StructuralFilterConfig`, `FilterRunMetadata`;
- `validate_filter_config`;
- covariance-form `kalman_log_likelihood`;
- AR(2) and nonlinear accumulation fixtures;
- particle-filter fail-closed placeholder;
- tests for partitions, singular-`Q` Kalman, all-missing steps, metadata, AR(2)
  linear recovery, nonlinear finite likelihood, and derivative smoke.

Before extending it, audit the candidate code against this plan and either
adopt it or replace it with explicit rationale.

### Implementation instructions

1. Add packaging metadata if the project is to be installed outside the repo
   root.
2. Keep dependency-light NumPy reference backends separate from future
   TensorFlow/JAX compiled backends.
3. Preserve failure-closed behavior when structural metadata is missing.
4. Preserve approximation labels for non-exact nonlinear Gaussian closures.
5. Keep model-specific DSGE and MacroFinance logic out of BayesFilter core.

### Tests

- `pytest -q`
- partition coverage, overlap, uniqueness, and dimension tests;
- mixed full-state integration opt-in and approximation-label tests;
- AR(p) deterministic lag-shift identity tests;
- exact Kalman singular-`Q` finite likelihood tests;
- all-missing prediction-only tests;
- particle-filter fail-closed tests.

### Pass gate

- Candidate package and tests are committed.
- The core API is stable enough for derivative and adapter planning.
- No client-specific model logic is present in BayesFilter core.

## Workstream D: SVD sigma-point structural filter

### Objective

Provide a structural sigma-point backend that integrates over innovation or
stochastic-state space and completes deterministic coordinates pointwise.

### Implementation instructions

1. Keep the backend explicitly approximate unless the model is a linear
   Gaussian special case.
2. Generate sigma/cubature points in innovation or stochastic-state space.
3. Call the model transition for each point so deterministic coordinates are
   completed by the model.
4. Reject mixed-model full-state integration unless explicitly opted in and
   labeled as approximation.
5. Return diagnostics:
   - integration space;
   - approximation label;
   - minimum spectral value;
   - eigen/singular gap telemetry when spectral derivatives are used;
   - finite-failure labels.
6. Do not promote tape gradients through SVD/eigen paths to HMC without a
   derivative audit and spectral-gap stress tests.

### Tests

- linear AR(2) recovery against exact Kalman likelihood;
- nonlinear toy comparison against dense quadrature on one-step cases;
- deterministic manifold identity preservation pointwise;
- missing-data behavior;
- spectral floor and negative-eigenvalue failure tests;
- finite-difference smoke tests only, clearly labeled as smoke.

### Pass gate

- The structural sigma-point backend preserves deterministic-completion
  identities pointwise.
- Linear Gaussian recovery passes.
- Approximation labels are always present for nonlinear Gaussian closures.
- Gradient/HMC promotion remains blocked until Workstream B and derivative
  stress gates pass.

## Workstream E: validation ladder

### Objective

Create a staged validation matrix that prevents smoke tests from being mistaken
for convergence or production readiness.

### Implementation instructions

1. Build the ladder in this order:
   - exact LGSSM value tests;
   - singular-`Q` and AR(p) structural tests;
   - nonlinear toy SSM value tests;
   - derivative parity tests;
   - compiled/eager parity tests;
   - short HMC smoke tests;
   - medium recovery tests;
   - strict convergence tests;
   - model-specific DSGE/MacroFinance adapter tests.
2. Store result summaries under `docs/plans` or `docs/results` with date,
   commit, command, environment, and interpretation.
3. Keep labels precise:
   - `smoke`;
   - `value_parity`;
   - `derivative_parity`;
   - `compiled_parity`;
   - `clean_stress`;
   - `medium_recovery`;
   - `strict_convergence`.
4. NK, Rotemberg, SGU, EZ, and NAWM must each earn their own gate.  Do not
   transfer claims automatically from SmallNK or toy models.

### Tests

- exact Kalman oracle tests;
- structural sigma-point linear recovery tests;
- nonlinear dense quadrature comparisons;
- derivative finite-difference and Hessian checks;
- XLA or compiled-path checks when a compiled backend exists;
- HMC diagnostics: divergences, effective sample size, R-hat, energy behavior,
  acceptance, step size, and target finite-failure counts.

### Pass gate

- Each model family has an explicit status label.
- Failed or blocked gates are documented rather than hidden.
- No convergence claim is made from smoke tests.

## Workstream F: literature and monograph completion

### Objective

Continue the BayesFilter monograph as a publication-quality consolidation of
Bayesian estimation for structural state-space models.

### Implementation instructions

1. Close current primary-source blockers:
   - UKF and square-root sigma-point filtering;
   - PMCMC and pseudo-marginal likelihood estimators;
   - square-root Kalman filtering;
   - differentiable particle filtering;
   - analytic Kalman derivatives and factor derivatives.
2. Use ResearchAssistant to fetch and review primary sources.
3. Do not merge bibliography entries until citation keys and claim support are
   audited.
4. Use MathDevMCP for labeled derivation checks where it is useful; record
   abstentions and tool limitations.
5. Keep original source projects read-only.
6. Update `docs/source_map.yml` whenever a source is promoted from candidate to
   drafted or audited status.

### Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
- `git diff --check`
- source-map YAML parse;
- placeholder search;
- risky-claim search;
- citation-support audit for any new literature-backed claim.

### Pass gate

- Every new claim has provenance.
- The monograph builds.
- Unsupported literature, derivative, or industrial-readiness claims remain
  explicitly blocked.

## Suggested execution order

1. Ratify and commit the source/code audit plus current candidate package if it
   passes final review.
2. Add packaging metadata and CI-style test commands.
3. Complete analytic Kalman derivative derivation reconciliation.
4. Implement derivative providers and tests.
5. Strengthen the structural SVD sigma-point backend diagnostics.
6. Build the validation ladder.
7. Start DSGE and MacroFinance adapter pilots only after the generic package and
   derivative gates are stable.
8. Continue monograph/literature work in parallel, but do not let prose claims
   outrun tests.

## Independent audit of this plan

Audit findings:

1. The plan is sensible because it preserves the exact LGSSM likelihood as the
   oracle and blocks derivative/HMC promotion until parity tests pass.
2. The plan now explicitly accounts for the candidate package and tests already
   present in the working tree, so a later agent will not unknowingly duplicate
   that work.
3. The plan separates common BayesFilter infrastructure from DSGE and
   MacroFinance client logic.
4. The plan keeps particle filtering fail-closed until a separate audit exists.
5. The plan does not yet give a full package distribution story; Workstream C
   therefore includes a packaging metadata task before external use.
6. The plan correctly treats ResearchAssistant and MathDevMCP as audit tools,
   not automatic proof engines.

Audit modifications applied:

- Added explicit current-baseline section.
- Added package-metadata task.
- Added model-family-specific validation labels.
- Added global stop rules for adding noise to deterministic coordinates,
  unlabeled full-state integration, and derivative parity failures.

Audit result:

Proceed with this as the master closure plan.  The next execution agent should
start with Workstream A and ratify the current candidate package before
expanding implementation or derivative support.

## Reset handoff

At the next session:

```bash
cd /home/chakwong/BayesFilter
git status --short
pytest -q
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8')); print('source_map yaml ok')"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
sed -n '1,260p' docs/plans/bayesfilter-six-step-structural-filtering-closure-plan-2026-05-04.md
sed -n '1,260p' docs/plans/bayesfilter-structural-source-code-audit-2026-05-04.md
```

Do not start DSGE, MacroFinance, particle-filter, or derivative-HMC promotion
until the corresponding gate in this plan is explicitly passed.
