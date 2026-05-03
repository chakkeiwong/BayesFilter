# Plan: structural state-space monograph consolidation pass

## Date

2026-05-04

## Scope

This is a bounded first consolidation pass for the BayesFilter monograph.  It
does not attempt to finish the several-hundred-page manuscript.  It reframes
the book around:

```text
Bayesian estimation for structural state-space models
```

and then hardens the first layer of the monograph so later linear-Gaussian,
derivative, nonlinear-filter, HMC, DSGE, and MacroFinance chapters use one
contract language.

## Motivation

The recent endogenous/exogenous state partition issue shows that the
BayesFilter project is not merely about SVD sigma-point filters or one DSGE
debugging problem.  It is about Bayesian estimation for structural
state-space models where:

- model states can have stochastic, deterministic, auxiliary, lag, and
  accounting roles;
- filtering likelihoods define the HMC target;
- analytic or custom derivatives are often necessary for industrial-scale
  models;
- nonlinear filters require explicit approximation labels and structural
  transition contracts;
- DSGE and MacroFinance should consume BayesFilter core filters through
  adapters, not duplicate filter implementations.

## Inputs

- `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`
- `docs/plans/bayesfilter-monograph-writing-continuation-plan-2026-05-03.md`
- `docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`
- `docs/plans/dsge-structural-filtering-refactor-plan-2026-05-03.md`
- `docs/chapters/ch01_introduction.tex`
- `docs/chapters/ch02_state_space_contracts.tex`
- `docs/chapters/ch03_hmc_target_requirements.tex`
- `docs/chapters/ch04_bayesfilter_api.tex`
- `docs/chapters/ch18b_structural_deterministic_dynamics.tex`
- `docs/source_map.yml`

Original source projects remain read-only:

- `/home/chakwong/python`
- `/home/chakwong/latex/CIP_monograph`
- `/home/chakwong/MacroFinance`

## Execution cycle

Every phase must follow:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

The reset memo is:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
```

## Phase S0: hygiene baseline

Objective:

Record the current repository state, parse the source map, and confirm the
starting LaTeX build behavior.

Actions:

1. Run `git status --short`.
2. Parse `docs/source_map.yml`.
3. Run `git diff --check`.
4. Run `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` from
   `docs`.

Pass gate:

- Dirty state is recorded.
- Source map parses.
- Whitespace check passes.
- Build either passes or any failure is documented with a concrete blocker.

## Phase S1: title and framing

Objective:

Reframe the monograph as Bayesian estimation for structural state-space
models.

Actions:

1. Update `docs/main.tex` title.
2. Update `docs/chapters/ch01_introduction.tex` so the first page says the
   book is about structural state-space Bayesian estimation, not merely
   HMC-safe filtering.
3. Keep HMC-safe filtering as the production discipline, not the whole scope.

Pass gate:

- Introduction clearly distinguishes BayesFilter core from DSGE/MacroFinance
  applications.
- The reader map still flows from contracts to likelihoods, derivatives,
  nonlinear filters, HMC, and case studies.

## Phase S2: structural state-space contracts in Part I

Objective:

Make the structural state partition contract explicit before later chapters
reuse it.

Actions:

1. Extend `ch02_state_space_contracts.tex` with:
   - stochastic, deterministic, auxiliary, and observed blocks;
   - AR(p) as a generic degenerate-transition example;
   - the rule that the partition is metadata, not inferred only from `Q`;
   - full-state nonlinear integration as a labeled approximation.
2. Extend `ch04_bayesfilter_api.tex` with:
   - `StatePartition`;
   - structural transition protocol;
   - filter run metadata;
   - adapter boundary for DSGE and MacroFinance.
3. Keep formulas conservative and source-backed by local plans; do not add new
   derivative formulas.

Pass gate:

- Part I gives another implementation agent enough contract language to start
  the BayesFilter core API.
- No unsupported production-readiness claim is introduced.

## Phase S3: provenance and reset updates

Objective:

Record the new framing and implementation/source-of-truth decision.

Actions:

1. Update `docs/source_map.yml` with this pass plan and the BayesFilter
   structural core plan.
2. Append phase results to
   `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`.
3. Ensure the reset memo says that BayesFilter owns structural filtering and
   model projects own structural maps.

Pass gate:

- A rebooted agent can identify the canonical monograph plan and the next
  phase.

## Phase S4: final build, audit, and commit

Objective:

Perform mechanical checks and commit the completed documentation pass.

Actions:

1. Run YAML parse.
2. Run `git diff --check`.
3. Build the monograph with `latexmk`.
4. Search for dangerous unsupported claims in touched Part I chapters.
5. Stage only relevant BayesFilter documentation files.
6. Commit with a message such as:

   ```text
   Reframe BayesFilter around structural state-space estimation
   ```

Pass gate:

- Build passes.
- No generated PDF or LaTeX byproducts are staged.
- Dirty pre-existing or unrelated files are not accidentally reverted.

## Stop rules

Stop and ask for direction if:

- LaTeX build fails because of a structural preamble/chapter conflict.
- YAML parse fails and the source-map intent is ambiguous.
- The plan would require editing original source projects.
- A phase would require final derivative or HMC convergence claims that are not
  supported by source documents or tests.

## Expected next phases after this pass

1. W2: write the exact linear Gaussian likelihood spine.
2. W3: consolidate MacroFinance analytic Kalman score/Hessian derivations with
   proof-carrying audits.
3. BayesFilter core implementation: structural partition interfaces and AR(p)
   tests.
4. DSGE adapter implementation after the BayesFilter core exists.

## Hypotheses to test later

1. A small `StatePartition` and `StructuralStateSpaceModel` protocol can cover
   AR(p), MacroFinance affine models, and DSGE perturbation adapters without
   model-specific filter forks.
2. Exact/square-root Kalman with singular `Q` should validate the structural
   partition contract before nonlinear sigma-point filters are attempted.
3. Analytic Kalman derivatives from MacroFinance can serve as the production
   gradient spine for large linear Gaussian models and as the validation
   oracle for custom-gradient wrappers.
4. Nonlinear SVD sigma-point filtering should remain value/diagnostic-first
   until spectral-gap derivative audits and structural manifold tests pass.
