# Plan: BayesFilter structural SVD 12-phase implementation

## Date

2026-05-05

## Purpose

This plan gives another coding agent explicit motivation, implementation
instructions, tests, audits, and stop rules for closing the structural SVD
filtering gap.  This session is documentation-only: it writes the plan and
audit, updates a reset memo, and stops before backend implementation.

## Final Goal

BayesFilter should become the shared filtering layer for structural
state-space models used by generic nonlinear SSMs, DSGE, MacroFinance, and
future NAWM-scale systems.  The final implementation must preserve three
distinct paths:

1. exact collapsed linear-Gaussian/Kalman filtering;
2. nonlinear structural filtering over innovation or stochastic coordinates
   with pointwise deterministic completion;
3. explicitly labeled approximation paths, including mixed full-state
   nonlinear sigma-point approximations.

No HMC target should be promoted until value, residual, derivative, and JIT
gates pass.

## Global Execution Protocol

Every phase must follow:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

After each phase, update the reset memo with:

- what was planned;
- what was executed;
- exact commands and results;
- interpretation;
- whether the next phase remains justified.

Stop and ask for direction if a phase would:

- change DSGE or MacroFinance economic semantics;
- add artificial process noise to deterministic coordinates;
- promote Rotemberg, SGU, EZ, or NAWM-style filtering without model-specific
  residual evidence;
- certify SVD/eigen derivatives by prose rather than proof/test evidence;
- claim HMC convergence from smoke tests;
- stage unrelated dirty files, generated PDFs, or another agent's changes.

## Phase 1: Mathematical Source Audit

### Motivation

The structural filtering problem is mathematical before it is computational.
The code must know which equations are exact, which are structural nonlinear
approximations, and which are labeled approximations only.  Without this audit,
BayesFilter risks recreating the original bug in a cleaner package.

### Implementation Instructions

1. Create a BayesFilter audit note under `docs/plans`.
2. Reconcile notation from:
   - `/home/chakwong/python/docs/monograph.tex`;
   - `/home/chakwong/latex/CIP_monograph/main.tex`;
   - `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex`;
   - BayesFilter Chapter 18b and related chapters.
3. Classify every path in a table:

| Path | Exact LGSSM | Structural nonlinear | Labeled approximation | Derivative status | Evidence |
| --- | --- | --- | --- | --- | --- |

4. Use ResearchAssistant for source/equation provenance where useful.
5. Use MathDevMCP for small derivation obligations where useful.
6. Record unresolved claims as blockers, not as assumptions hidden in prose.

### Tests

Documentation tests:

```bash
rg -n "exact|structural nonlinear|labeled approximation|deterministic completion" docs/plans
```

### Audit

- Confirm SmallNK is not the only DSGE evidence.
- Confirm zero rows of `eta` are consistency evidence only, not the source of
  truth for state roles.

### Exit Gate

No backend implementation begins until this written audit exists.

## Phase 2: Code Reuse and Migration Audit

### Motivation

The codebase already contains Kalman, SVD sigma-point, particle, derivative,
and adapter machinery.  A rewrite without audit risks losing tested behavior;
blind reuse risks preserving the structural bug.

### Implementation Instructions

1. Audit candidate code in:
   - `bayesfilter/`;
   - `/home/chakwong/python/src/dsge_hmc`;
   - `/home/chakwong/MacroFinance`.
2. Classify each candidate path:
   - reuse as-is;
   - reuse with localized fixes;
   - keep as labeled approximation;
   - reject and reimplement.
3. Identify required regression fixtures before migration.
4. Record client-owned logic that must stay outside BayesFilter.

### Tests

```bash
rg -n "Kalman|SVD|sigma|particle|gradient|Hessian|transition_points" \
  bayesfilter /home/chakwong/python/src/dsge_hmc /home/chakwong/MacroFinance
```

### Audit

- Prefer audited wrapping/extraction over greenfield rewrites.
- Reject code that violates state partition semantics.

### Exit Gate

Every planned implementation file has a reuse decision and test list.

## Phase 3: BayesFilter Structural Sigma-Point Core

### Motivation

BayesFilter currently has an eager structural sigma-point reference backend.
The production core must preserve structural semantics and expose honest run
metadata before any client adapter or HMC work depends on it.

### Implementation Instructions

1. Harden `bayesfilter/filters/sigma_points.py`.
2. Ensure mixed structural models default to innovation or stochastic-state
   integration.
3. Require explicit opt-in and nonempty approximation labels for mixed
   full-state integration.
4. Complete deterministic coordinates pointwise.
5. Preserve diagnostics for deterministic residuals and covariance eigenvalues.
6. Return `FilterRunMetadata` with:
   - integration space;
   - completion policy;
   - approximation label;
   - differentiability status;
   - compiled status.

### Tests

```bash
pytest -q \
  tests/test_structural_partition.py \
  tests/test_structural_sigma_points.py \
  tests/test_structural_ar_p.py \
  tests/test_filter_metadata.py
```

### Audit

- If toy structural residuals fail, stop before DSGE adapters.
- Do not rewrite passing code merely for style.

### Exit Gate

Toy structural likelihood, residual, and metadata tests pass.

## Phase 4: Exact Kalman and Degenerate Linear Spine

### Motivation

Exact collapsed LGSSM/Kalman filtering is mathematically valid even when the
process covariance is singular.  It must stay separate from nonlinear
structural sigma-point filtering.

### Implementation Instructions

1. Preserve exact and square-root Kalman paths.
2. Accept rank-deficient `Q` where mathematically valid.
3. Keep result metadata distinct from nonlinear approximation metadata.
4. Add tests showing exact linear filtering does not require deterministic
   pointwise completion when collapsed moments are exact.

### Tests

```bash
pytest -q tests/test_degenerate_kalman.py tests/test_filter_metadata.py
```

### Audit

- Do not force all linear models through the nonlinear structural backend.
- Do not add nuggets to deterministic coordinates unless explicitly labeled.

### Exit Gate

Exact linear tests pass independently from nonlinear structural tests.

## Phase 5: Generic Structural Fixtures

### Motivation

The endogenous/exogenous issue is not DSGE-specific.  AR(p), lag stacks,
accumulators, affine models, and state augmentation all create deterministic
coordinates.

### Implementation Instructions

1. Add or harden fixtures for:
   - AR(2) lag shift;
   - deterministic accumulator;
   - nonlinear stochastic plus deterministic completion toy model;
   - missing-data observation fixture.
2. Assert pointwise structural residuals.
3. Compare one-step likelihoods to dense quadrature or analytic references.

### Tests

```bash
pytest -q tests/test_structural_ar_p.py tests/test_structural_sigma_points.py
```

### Audit

- Fixture failures mean the core contract is not ready for DSGE or
  MacroFinance.

### Exit Gate

Generic fixtures show BayesFilter is solving a common structural SSM problem.

## Phase 6: MacroFinance Adapter and Analytic Derivative Spine

### Motivation

MacroFinance contains analytic Kalman derivative work and state-space
structures that should inform BayesFilter.  The goal is shared filtering
machinery, not copied financial-model logic.

### Implementation Instructions

1. Audit MacroFinance state-space and derivative code.
2. Add adapter gates that classify:
   - exact collapsed LGSSM;
   - affine or lag-stack structures;
   - nonlinear structural cases;
   - derivative-provider availability.
3. Keep MacroFinance economics in MacroFinance.
4. Treat analytic derivatives as candidate providers only after audit.

### Tests

```bash
pytest -q tests/test_macrofinance_adapter.py tests/test_degenerate_kalman.py
```

### Audit

- Do not migrate derivative formulas without source and numerical checks.
- Do not force nonlinear structural filtering onto exact affine LGSSM models.

### Exit Gate

MacroFinance adapter readiness is classified without changing MacroFinance
model semantics.

## Phase 7: DSGE Adapter Integration

### Motivation

The DSGE repo now exposes BayesFilter metadata and fail-closed guards.  The
next step is adapter-level BayesFilter integration without importing DSGE
economics into BayesFilter.

### Implementation Instructions

1. Consume DSGE metadata:
   - `bayesfilter_state_names`;
   - `bayesfilter_stochastic_indices`;
   - `bayesfilter_deterministic_indices`;
   - `bayesfilter_innovation_dim`;
   - `bayesfilter_deterministic_completion`.
2. Keep SmallNK all-stochastic.
3. Keep Rotemberg and SGU mixed structural but not structurally promoted until
   residual tests pass.
4. Keep EZ fail-closed until timing audit.
5. Preserve legacy full-state SVD labels for mixed DSGE smoke probes.

### Tests

BayesFilter:

```bash
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

DSGE client:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_structural_dsge_partition.py
```

### Audit

- Passing adapter metadata is not a nonlinear filtering correctness claim.

### Exit Gate

SmallNK, Rotemberg, and SGU metadata gates pass; EZ remains explicitly blocked.

## Phase 8: Model-Specific DSGE Completion Evidence

### Motivation

Rotemberg and SGU need model-specific residual evidence before nonlinear
structural filtering is safe.  EZ needs timing classification.

### Implementation Instructions

1. Rotemberg:
   - test `dy_next = y_next - y_current` pointwise;
   - add second-order/pruned residual tests before promotion.
2. SGU:
   - derive residuals for `d,k,r,riskprem`;
   - test completion over sigma-point grids and near-boundary parameters.
3. EZ:
   - audit timing and state partition;
   - expose metadata only after classification.

### Tests

Add focused tests in the owning DSGE repo once derivations are written.

### Audit

- Do not infer state roles solely from `eta`.
- Do not promote first-order bridges to second-order correctness.

### Exit Gate

Each model is either residual-tested or explicitly blocked.

## Phase 9: Derivative and Hessian Safety Gate

### Motivation

HMC needs gradients.  SVD/eigen derivatives can be unstable near repeated
eigenvalues, and symbolic-looking derivations can easily hide missing
assumptions.

### Implementation Instructions

1. Split derivative obligations:
   - transition;
   - deterministic completion;
   - observation;
   - covariance prediction;
   - factorization;
   - likelihood update;
   - parameter transform.
2. Use MathDevMCP for proof-carrying derivation checks where feasible.
3. Use finite-difference, JVP/VJP, and Hessian symmetry tests.
4. Add spectral-gap stress tests.
5. Label derivative status in result metadata.

### Tests

```bash
pytest -q tests/test_derivative_validation_smoke.py
```

### Audit

- If eigen/SVD gradients are unsafe, do not promote to HMC.  Consider custom
  derivatives or another factor backend.

### Exit Gate

Gradient and Hessian tests pass for the specific backend/model class being
promoted.

## Phase 10: JIT and Static-Shape Production Gate

### Motivation

HMC can be unusably slow if the target is not compiled or if hidden Python
callbacks remain.  JIT readiness is a separate gate from mathematical
correctness.

### Implementation Instructions

1. Identify static and dynamic dimensions.
2. Add compiled value and gradient parity tests.
3. Assert no Python callbacks in production HMC target paths.
4. Record compile time and steady-state runtime.
5. Keep eager-only backends labeled as such.

### Tests

Add backend-specific tests once the production backend exists.  At minimum:

```bash
pytest -q tests/test_backend_readiness.py
```

### Audit

- Do not reintroduce TFP NUTS as a default fix hypothesis.
- Do not call eager reference code HMC-ready.

### Exit Gate

Compiled target value and gradient match eager/reference results.

## Phase 11: HMC Validation Ladder

### Motivation

Only after value, residual, derivative, and JIT gates pass should HMC be used
as evidence.  The ladder must proceed from simple to complex.

### Implementation Instructions

Run in order:

1. LGSSM recovery;
2. nonlinear toy SSM;
3. generic structural AR/lag model;
4. SmallNK;
5. Rotemberg after second-order completion evidence;
6. SGU after equilibrium residual evidence;
7. EZ after timing audit;
8. NAWM-scale models only after smaller DSGE gates.

### Tests

Use dedicated HMC tests only after earlier gates pass.  Record:

- finite target and gradient;
- acceptance rate;
- divergences;
- R-hat;
- ESS;
- posterior recovery;
- compile and runtime.

### Audit

- Label smoke tests as smoke tests.
- No convergence claim without multi-chain diagnostics.

### Exit Gate

Each model moves to the next ladder stage only after diagnostics pass.

## Phase 12: Documentation, Provenance, and Release Gate

### Motivation

The structural/full-state distinction has already consumed weeks of debugging.
The release must make the distinction hard to miss for future agents.

### Implementation Instructions

1. Update BayesFilter monograph chapters and source map.
2. Update reset memos after every phase.
3. Register plan, audit, and result artifacts.
4. Use exact labels:
   - exact LGSSM;
   - structural nonlinear;
   - labeled approximation;
   - finite smoke;
   - converged posterior.
5. Do not commit generated PDFs unless explicitly requested.

### Tests

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
cd ..
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
```

### Audit

- Search for unsupported claims:

```bash
rg -n "converged|production-ready|structurally fixed|HMC-ready" docs bayesfilter tests
```

### Exit Gate

Docs, source map, tests, and reset memo agree with the evidence.

## Execution Status for This Session

This session executes the planning/audit/reset-memo phases only.  It does not
execute backend code phases because Phase 1 requires a mathematical source
audit before implementation, and the user reserved this session for
documentation.  The next coding agent should start at Phase 1 in this plan.
