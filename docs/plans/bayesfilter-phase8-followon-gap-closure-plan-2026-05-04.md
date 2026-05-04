# Plan: Phase 8 follow-on MacroFinance metadata gap closure

## Scope

This plan closes the gaps left by the first Phase 8 MacroFinance adapter pilot.

Already complete:

- one-country value adapter;
- delegated derivative bridge;
- finite HMC-target conformance check;
- large-scale/cross-currency/production provider deferral audit.

Remaining gaps to close now:

1. Large-scale LGSSM mask-policy and parameter-unit metadata.
2. Cross-currency derivative coverage and finite-difference oracle provenance.
3. Production provider blocker-table, identification-evidence, sparse-backend,
   and final-readiness metadata.
4. BayesFilter-context HMC target gates for value/score/Hessian and
   eager/compiled parity without sampler migration.

## Non-goals

- Do not copy MacroFinance large-scale, cross-currency, production, Kalman,
  derivative, TensorFlow, or HMC sampler implementations into BayesFilter.
- Do not claim HMC convergence.
- Do not promote large-scale or production cross-currency providers to
  production-ready simply because metadata can be wrapped.
- Do not change the pre-existing dirty BayesFilter chapter file in this pass.

## Required Cycle

Each phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

## Phase C0: reset memo setup and latest-state check

Actions:

1. Record repo status.
2. Confirm MacroFinance remote/local state and local path-fix commit.
3. Record this plan and the independent audit.

Tests:

- `git status --short --branch` in both repos.
- MacroFinance `git log -1 --oneline`.
- BayesFilter `git log -1 --oneline`.

Gate:

- Continue if only known/unrelated dirty state remains.

## Phase C1: generic metadata primitives

Actions:

1. Add BayesFilter dataclasses for:
   - parameter units;
   - observation mask policy;
   - derivative coverage;
   - finite-difference oracle provenance;
   - readiness blockers;
   - identification evidence;
   - sparse backend policy.
2. Keep these primitives dependency-free.

Tests:

- Pure unit tests for construction, tuple coercion, and fail-closed readiness.

Gate:

- Continue if the metadata can be represented without importing MacroFinance.

## Phase C2: large-scale LGSSM mask/unit metadata

Actions:

1. Add a wrapper that extracts parameter names, parameter units, and observation
   mask policy from MacroFinance-like large-scale providers.
2. Do not run large-scale likelihood or HMC here.

Tests:

- Pure fake-provider test.
- Optional MacroFinance integration test using `LargeScaleLGSSMDerivativeProvider`
  and helpers if available.

Gate:

- Continue if metadata extraction works and does not change value/derivative
  adapter behavior.

## Phase C3: cross-currency coverage and oracle provenance

Actions:

1. Add a wrapper that normalizes derivative coverage rows from providers with
   `derivative_coverage_matrix`.
2. Add finite-difference oracle provenance metadata when a provider exposes
   `finite_difference_oracle` or equivalent.

Tests:

- Pure fake-provider test.
- Optional MacroFinance integration test against
  `CrossCurrencyStructuralDerivativeProvider` if available.

Gate:

- Continue if coverage rows are represented without asserting production
  readiness.

## Phase C4: production readiness metadata

Actions:

1. Add a wrapper that normalizes:
   - blocker table;
   - blocker summary;
   - identification evidence status;
   - sparse derivative backend policy;
   - final readiness validation.
2. Fail closed when the provider reports blockers or final readiness validation
   raises.

Tests:

- Pure fake-provider test.
- Optional MacroFinance integration test against
  `ProductionCrossCurrencyDerivativeProvider`.

Gate:

- Continue if readiness metadata fails closed and does not erase blockers.

## Phase C5: BayesFilter HMC target gates

Actions:

1. Add explicit HMC gate result metadata for:
   - value finite;
   - score finite;
   - negative Hessian finite and symmetric;
   - optional eager/compiled value/score parity.
2. Keep this as target readiness only; no sampler.

Tests:

- Pure fake posterior test.
- Optional MacroFinance one-country target test using the existing adapter.

Gate:

- Continue to final validation if readiness remains `not_converged` /
  `not_claimed`.

## Final Validation

BayesFilter:

- `pytest -q tests`
- YAML parse for `docs/source_map.yml`
- `git diff --check`

MacroFinance:

- only rerun touched/companion checks if needed; no new MacroFinance code is
  expected in this pass.

## Hypotheses

- H-C1: Metadata primitives can close provider-readiness gaps without importing
  or copying MacroFinance implementation code.
- H-C2: Large-scale LGSSM providers are adapter-ready once mask policy and
  parameter units are first-class metadata.
- H-C3: Cross-currency structural providers are adapter-ready once derivative
  coverage and finite-difference oracle provenance are first-class metadata.
- H-C4: Production providers can be safely exposed only if BayesFilter carries
  blocker, identification, sparse-backend, and final-readiness metadata and
  fails closed.
- H-C5: HMC target readiness can be represented as finite operation and parity
  gates without running a sampler or claiming convergence.
