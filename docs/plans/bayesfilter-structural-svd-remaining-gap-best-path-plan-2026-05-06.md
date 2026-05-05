# Plan: best path to close remaining structural SVD filtering gaps

## Date

2026-05-06

## Purpose

This plan turns the remaining structural SVD filtering gaps into an execution
path for another agent.  It is intentionally ordered by scientific dependency,
not by convenience:

```text
structural semantics
  -> model-specific residual evidence
  -> derivative and Hessian certification
  -> compiled target parity
  -> HMC validation
  -> release/provenance cleanup
```

The main lesson from the previous debugging cycle is that sampler symptoms are
not a reliable way to discover target-definition bugs.  Therefore this plan
puts DSGE structural completion evidence before gradients, compilation, and
HMC.

## Final goal

BayesFilter should be the shared filtering layer for:

- generic nonlinear state-space models;
- DSGE models in `/home/chakwong/python`;
- MacroFinance models in `/home/chakwong/MacroFinance`;
- future NAWM-scale structural systems.

The final system must preserve three separate paths:

1. exact collapsed LGSSM/Kalman filtering;
2. nonlinear structural filtering over declared stochastic or innovation
   variables with deterministic completion pointwise;
3. explicitly labeled approximations, including any mixed full-state nonlinear
   sigma-point approximation.

No HMC target may be promoted until value, residual, derivative, and compiled
target gates all pass.

## Current evidence baseline

The execution audit in
`docs/plans/bayesfilter-structural-svd-12-phase-execution-audit-2026-05-06.md`
found:

- BayesFilter structural contracts pass local tests.
- The eager structural SVD/cubature sigma-point reference passes toy tests.
- Exact covariance-form Kalman filtering passes degenerate linear tests.
- Generic AR/lag and nonlinear deterministic-completion fixtures pass.
- MacroFinance value adapter gates pass.
- DSGE metadata adapter gates pass.
- Derivative/backend readiness guardrails pass as guardrails.
- The full BayesFilter suite passed with `63 passed`.

The remaining gaps are promotion gates:

1. model-specific DSGE deterministic-completion residual evidence;
2. SVD sigma-point analytic gradient/Hessian certification;
3. compiled static-shape production backend;
4. HMC validation ladder;
5. DSGE client-side routing/integration;
6. release/provenance cleanup.

## Ownership boundaries

BayesFilter owns:

- generic structural metadata contracts;
- exact and approximate filter backends;
- filter result metadata;
- generic fixtures and backend readiness gates;
- documentation of exact, structural nonlinear, and approximation-only paths.

`/home/chakwong/python` owns:

- DSGE economic semantics;
- DSGE state timing and structural partitions;
- Rotemberg, SGU, EZ, and future NAWM completion maps;
- client-side residual tests and model-specific HMC experiments.

`/home/chakwong/MacroFinance` owns:

- financial-model construction;
- analytic derivative providers;
- model-specific parameter transforms and economic restrictions.

BayesFilter may adapt client-owned objects, but it must not import or rewrite
client economics.

## Best path summary

The best execution path is:

1. **Preflight and evidence freeze.**
   Record dirty files, commit boundaries, and existing gate evidence.
2. **Close DSGE structural residual evidence first.**
   Build client-side residual harnesses for SmallNK, Rotemberg, SGU, and EZ
   timing classification.  This is the highest-priority scientific blocker.
3. **Patch BayesFilter contracts only where residual work proves a missing
   generic abstraction.**
   Do not expand the core spec from imagination.
4. **Certify derivative/Hessian behavior after the structural target is fixed.**
   Use proof-carrying derivation, finite differences, JVP/VJP, Hessian symmetry,
   and spectral-gap stress tests.
5. **Build compiled backend parity after derivative policy is chosen.**
   The compiled backend must match eager value and gradient results before HMC.
6. **Run the HMC ladder from simple to complex.**
   LGSSM and toy structural fixtures precede DSGE models.
7. **Clean release/provenance last.**
   Only after evidence exists should docs claim stronger readiness labels.

## Execution protocol

Every phase must follow:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

After each phase, update a reset memo with:

- exact files reviewed or changed;
- exact commands run;
- test results;
- interpretation;
- whether the next phase remains justified;
- blockers and hypotheses.

Stop and ask for direction if a phase would:

- change DSGE or MacroFinance economic semantics;
- infer state roles solely from zero rows of a shock-impact matrix;
- add artificial process noise to deterministic coordinates;
- promote Rotemberg, SGU, EZ, or NAWM without model-specific residual tests;
- certify SVD/eigen derivatives by prose alone;
- claim HMC convergence from smoke tests;
- stage unrelated dirty files, generated PDFs, or another agent's changes.

## Phase 0: workspace hygiene and evidence freeze

### Motivation

There are pre-existing dirty and untracked files in both BayesFilter and
`/home/chakwong/python`.  The next implementation agent must not mix unrelated
Neutra, monograph, reference, or PDF work into the structural filter commit.

### Implementation instructions

1. In `/home/chakwong/BayesFilter`, run:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

2. In `/home/chakwong/python`, run:

```bash
git status --short --branch
git log -5 --oneline --decorate
```

3. Record both statuses in the reset memo.
4. If tracked code files are dirty from another agent, do not edit them until
   ownership is clear.
5. Stage explicit paths only.  Do not stage generated PDFs, templates,
   `Zone.Identifier` files, `.codex/`, `.serena/`, or unrelated Neutra work.

### Tests

No code tests are required.  The gate is clean ownership.

### Exit gate

Proceed only when the write set for the next phase is explicit.

## Phase 1: DSGE structural inventory and residual-test specification

### Motivation

The central scientific blocker is not BayesFilter's toy structural core.  It is
whether each DSGE model exposes the correct endogenous/exogenous timing and
completion identities.  SmallNK being all-stochastic does not prove Rotemberg,
SGU, EZ, or NAWM readiness.

### Implementation instructions

In `/home/chakwong/python`:

1. Inventory each target model:
   - SmallNK;
   - Rotemberg NK;
   - SGU;
   - EZ;
   - any NAWM-like prototype if present.
2. For each model, write a metadata table:

| Model | State names | Stochastic indices | Deterministic indices | Innovation dim | Completion map | Status |
| --- | --- | --- | --- | --- | --- | --- |

3. For each mixed model, write the intended structural equations:

```text
m_t = T_m(m_{t-1}, k_{t-1}, eps_t; theta)
k_t = T_k(m_t, m_{t-1}, k_{t-1}; theta)
y_t = G(m_t, k_t; theta) + e_t
```

4. State which variables are exogenous/shock-driven and which are
   endogenous/predetermined/deterministic conditional on the shock path.
5. Identify whether first-order, second-order, or pruned state components need
   separate timing treatment.
6. For EZ, do not guess.  Record the timing blocker explicitly if the structural
   timing is not established.

### Required artifact

Create a client-side plan/result note in `/home/chakwong/python/docs/plans`,
for example:

```text
docs/plans/dsge-structural-completion-residual-harness-plan-YYYY-MM-DD.md
```

### Tests

Documentation checks:

```bash
rg -n "bayesfilter_state_names|bayesfilter_stochastic_indices|bayesfilter_deterministic_indices|bayesfilter_deterministic_completion" src tests docs
rg -n "Rotemberg|SGU|EZ|deterministic completion|residual" docs/plans src tests
```

### Audit

Confirm that no model is promoted merely because the adapter metadata exists.

### Exit gate

Proceed to Phase 2 only when every model is either:

- classified with a residual-test specification; or
- explicitly blocked with the missing timing/completion information named.

## Phase 2: DSGE deterministic-completion residual harness

### Motivation

The filter must preserve the structural support pointwise.  For mixed DSGE
models, every propagated sigma point or particle should satisfy the
deterministic identities implied by the model.  This is the core bug-prevention
gate.

### Implementation instructions

In `/home/chakwong/python`:

1. Add tests under a focused contract path, for example:

```text
tests/contracts/test_dsge_structural_completion_residuals.py
```

2. Build a reusable residual harness that accepts:
   - model object;
   - parameter point;
   - previous state grid;
   - innovation grid;
   - deterministic completion map;
   - residual function.
3. Test each model in increasing difficulty:
   - SmallNK: all-stochastic metadata should pass without deterministic
     completion.
   - Rotemberg: first-order bridge residuals first; second-order/pruned
     residuals before promotion.
   - SGU: residuals for all declared deterministic coordinates.
   - EZ: remain blocked until timing is classified.
4. Generate sigma-point grids from deterministic toy covariances, not random
   Monte Carlo draws, so failures are reproducible.
5. Include near-boundary parameter points where stability and determinacy
   constraints are still valid.
6. Assert:
   - finite transition outputs;
   - residual norm below tolerance;
   - deterministic coordinates are computed by the completion map, not noised;
   - failure is explicit when completion maps are absent.

### Tests

Expected command shape:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_structural_completion_residuals.py
```

Keep existing metadata tests green:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_structural_dsge_partition.py
```

### Audit

Rotemberg and SGU may move forward only if residuals pass for the exact model
order being promoted.  First-order residuals do not certify second-order/pruned
filters.

### Exit gate

Proceed only for models with passing residual harnesses.  Block the rest.

## Phase 3: BayesFilter contract patching from proven client needs

### Motivation

BayesFilter should remain generic.  Its API should grow only when the DSGE
residual work proves that a reusable abstraction is missing.

### Implementation instructions

In `/home/chakwong/BayesFilter`:

1. Review failures or missing abstractions from Phase 2.
2. Patch only generic contracts, such as:
   - richer `StatePartition` metadata;
   - completion-map protocol typing;
   - static dimension metadata;
   - residual diagnostic containers;
   - support for `stochastic_state` integration if genuinely required.
3. Do not add DSGE-specific economics to BayesFilter.
4. Add generic toy tests before adapter tests.
5. Keep mixed full-state integration fail-closed unless an
   `approximation_label` is supplied.

### Tests

At minimum:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_structural_partition.py tests/test_structural_sigma_points.py tests/test_structural_ar_p.py tests/test_filter_metadata.py
pytest -q tests/test_dsge_adapter_gate.py
pytest -q
```

If DSGE adapter changes are involved:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

### Audit

Check that every new BayesFilter field is used by at least one generic fixture
or fail-closed adapter test.

### Exit gate

BayesFilter remains generic, all tests pass, and client models still own their
economics.

## Phase 4: derivative and Hessian certification plan

### Motivation

HMC needs reliable gradients.  Spectral/SVD/eigen derivatives can be unstable
near repeated singular or eigen values.  Passing value tests does not certify
gradient or Hessian behavior.

### Implementation instructions

1. Split derivative obligations into named blocks:
   - parameter transform;
   - transition and completion map;
   - sigma-point construction;
   - covariance prediction;
   - observation prediction;
   - innovation covariance factorization;
   - log likelihood update;
   - filtering recursion.
2. For each block, choose one derivative policy:
   - autodiff allowed;
   - custom derivative required;
   - finite-difference validation only;
   - blocked.
3. Use `/home/chakwong/MathDevMCP/docs/proof-carrying-derivation-agent-guide.md`
   as the workflow guide.
4. Use MathDevMCP/SymPy/Sage/Lean/Z3 where useful to check local algebraic
   obligations, but mark measure-theoretic or matrix-factorization assumptions
   as human-review-required unless they are actually certified.
5. For spectral factorization, record:
   - singular/eigen values;
   - minimum gap;
   - gap tolerance;
   - finite-difference check;
   - JVP/VJP check;
   - Hessian symmetry check.
6. Decide whether SVD/eigen derivatives are allowed for HMC or whether a
   non-spectral custom-gradient or alternative factor backend is required.

### Required artifact

Create a derivative certification plan/result note in BayesFilter:

```text
docs/plans/bayesfilter-structural-svd-derivative-certification-plan-YYYY-MM-DD.md
```

### Tests

Start with existing guardrails:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
```

Then add model/backend-specific tests:

```bash
pytest -q tests/test_structural_svd_derivative_certification.py
```

The new test should include:

- finite-difference gradient agreement;
- JVP/VJP parity;
- Hessian symmetry;
- small spectral-gap stress cases;
- failure when evidence is missing.

### Audit

Do not accept "autodiff runs" as certification.  The audit must explain the
validity domain and the behavior near repeated singular/eigen values.

### Exit gate

Proceed to compiled backend work only when the derivative policy for the target
backend is explicit and tested.

## Phase 5: compiled static-shape backend parity

### Motivation

An HMC target that is not compiled can be too slow, and a compiled function
that silently differs from eager reference code is dangerous.  Compilation is a
separate gate from mathematical correctness.

### Implementation instructions

1. Write a backend decision note:
   - JAX backend if BayesFilter independence, static shape control, and custom
     JVP/VJP support dominate;
   - TensorFlow backend if DSGE client compatibility dominates;
   - no TFP NUTS dependency as a default fix hypothesis.
2. Define static metadata:
   - state dimension;
   - innovation dimension;
   - observation dimension;
   - number of time periods;
   - mask convention;
   - parameter dimension.
3. Implement compiled value path only after eager value tests pass.
4. Implement compiled gradient path only after Phase 4 derivative policy passes.
5. Compare eager and compiled:
   - log likelihood;
   - gradient;
   - selected Hessian or Hessian-vector products;
   - metadata labels;
   - compile time;
   - steady-state runtime.
6. Keep eager-only paths labeled `eager` or `eager_numpy`.

### Tests

Add or extend:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_backend_readiness.py
pytest -q tests/test_compiled_structural_svd_parity.py
```

For DSGE client parity:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_bayesfilter_compiled_parity.py
```

### Audit

Compiled parity must be model-specific.  Passing a toy compiled target does not
certify Rotemberg, SGU, EZ, or NAWM.

### Exit gate

Proceed to HMC only for a model/backend pair whose eager and compiled value and
gradient paths agree within tolerance.

## Phase 6: HMC validation ladder

### Motivation

HMC is evidence only after the target is structurally correct, differentiable,
and compiled.  It must be run as a ladder, not as a rescue hypothesis.

### Implementation instructions

Run in this order:

1. exact LGSSM recovery;
2. nonlinear toy SSM;
3. structural AR/lag fixture;
4. SmallNK;
5. Rotemberg only after residual and derivative/JIT gates pass;
6. SGU only after residual and derivative/JIT gates pass;
7. EZ only after timing, residual, derivative, and JIT gates pass;
8. NAWM-scale models only after smaller DSGE models pass.

For each target record:

- finite target and finite gradient;
- compile time;
- steady-state runtime;
- step size and trajectory settings;
- acceptance rate;
- divergences;
- split R-hat;
- ESS;
- posterior recovery against known or trusted reference;
- comparison with exact Kalman where available.

### Tests

Each ladder rung should have a dedicated result file under `docs/plans` and a
repeatable command.  Example command shape:

```bash
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter python \
  scripts/run_structural_svd_hmc_ladder.py --target structural-ar2 --seed 123
```

### Success criteria

Minimum promotion criteria:

- no nonfinite target or gradient evaluations in the tested region;
- no divergences in the reported run;
- acceptance rate in the predeclared acceptable band;
- split R-hat below the predeclared threshold, typically `1.01`;
- ESS above the predeclared threshold;
- posterior recovery within tolerance for controlled targets.

### Audit

Label results precisely:

- `finite_smoke`;
- `value_validated`;
- `gradient_validated`;
- `compiled_parity_passed`;
- `hmc_diagnostics_passed`;
- `posterior_recovery_passed`.

Do not use `converged` unless multi-chain diagnostics and posterior recovery
support that word.

### Exit gate

Each model advances only after the previous rung passes.

## Phase 7: release and provenance cleanup

### Motivation

The repository currently has useful but scattered planning, reset-memo,
reference, and generated/PDF artifacts.  Release should make the evidence
auditable without committing accidental files.

### Implementation instructions

1. Review dirty/untracked files in BayesFilter and `/home/chakwong/python`.
2. Classify each as:
   - commit now;
   - keep untracked intentionally;
   - move to research cache;
   - ignore;
   - delete only with explicit approval.
3. Update BayesFilter documentation only for claims that are supported by
   evidence from Phases 1--6.
4. Update reset memos with final status and next hypotheses.
5. Update source maps only for committed documentation.
6. Do not commit generated PDFs unless explicitly requested.

### Tests

In BayesFilter:

```bash
pytest -q
git diff --check
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
```

If documentation changed:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### Audit

Search for unsupported claim labels:

```bash
rg -n "converged|production-ready|HMC-ready|certified|exact" docs bayesfilter tests
```

Every strong claim must point to a test, derivation, source audit, or result
artifact.

### Exit gate

The release is ready only when code, docs, reset memos, source maps, and test
results agree.

## Recommended immediate next action

Start with Phase 0 and Phase 1 in `/home/chakwong/python`, not with BayesFilter
backend code.  The next concrete deliverable should be a DSGE structural
completion residual harness plan and then executable residual tests.  That is
the shortest path to closing the highest-risk scientific blocker.

## Suggested commit strategy

Use small commits by evidence gate:

1. DSGE structural residual harness plan.
2. DSGE metadata inventory and blocked-model table.
3. Rotemberg residual harness.
4. SGU residual harness.
5. BayesFilter generic contract patches, if needed.
6. Derivative certification plan and tests.
7. Compiled backend parity.
8. HMC ladder results.
9. Documentation/provenance cleanup.

Do not combine Neutra, monograph bibliography cleanup, generated PDF handling,
and structural filter implementation in the same commit.
