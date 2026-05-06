# Plan: structural SVD remaining-gap closure master plan

## Date

2026-05-06

## Purpose

This plan turns the current structural SVD blocker state into an executable
closure path.  It supersedes the older "best path" planning note for future
execution, while preserving the six-blocker result as the current evidence
baseline.

The plan is deliberately ordered by scientific dependency:

```text
model semantics
  -> model-specific residual evidence
  -> derivative and Hessian certification
  -> compiled static-shape parity
  -> HMC diagnostics
  -> documentation and release provenance
```

The goal is not to make the blocker labels disappear by renaming them.  The
goal is to replace each blocker with executable evidence and an allowed claim
label.

## Current baseline

The following gates are already validated for BayesFilter-local scope:

- exact covariance-form Kalman value filtering;
- degenerate linear-Gaussian tests;
- structural sigma-point value filtering over declared innovation coordinates;
- deterministic completion for generic toy fixtures;
- AR/lag and nonlinear accumulation structural fixtures;
- MacroFinance value-adapter classification;
- DSGE adapter metadata gates;
- backend readiness guardrails;
- source-map YAML parsing;
- full BayesFilter pytest suite.

Recent observed results:

```text
BayesFilter guardrails: 10 passed
BayesFilter full suite: 63 passed, 2 warnings
DSGE strong residual gates: 19 passed, 3 warnings
source_map YAML parse: passed
git diff --check: passed
```

The current six named blockers are:

| Area | Current label | Meaning |
| --- | --- | --- |
| Rotemberg | `blocked_pruned_second_order_dy_identity_residual` | The second-order/pruned `dy` path does not yet satisfy `dy_next = y_next - y_current` pointwise. |
| SGU | `blocked_nonlinear_equilibrium_manifold_residual` | The current linear completion bridge does not satisfy nonlinear canonical residuals for `d,k,r,riskprem`. |
| EZ | `blocked_pending_source_backed_timing_metadata` | EZ lacks source-backed timing, state-ordering, and measurement metadata. |
| SVD/eigen derivatives | `value_only_blocked` | No backend/model pair has certified analytic gradients or Hessians. |
| Compiled parity | `compiled_parity_not_started` | No static-shape compiled structural SVD target has matched the eager reference. |
| HMC | `hmc_not_justified` | HMC is downstream of residual, derivative, and compiled parity evidence. |

Execution note: after this plan was drafted, the DSGE client at
`/home/chakwong/python` advanced to `59c05f5 Close Rotemberg structural
completion gate`.  That commit preserves the raw Rotemberg blocker as a
diagnostic, but adds committed evidence for
`rotemberg_second_order_dy_completion_residual_passed` on the tested
default-calibration grids.  Future execution should validate that evidence
rather than reimplementing the Rotemberg helper.

## Ownership boundaries

BayesFilter owns:

- generic structural metadata contracts;
- exact and approximate filter backends;
- generic filter result metadata;
- toy and generic structural fixtures;
- backend readiness and claim-label gates;
- documentation of exact, structural nonlinear, and approximation-only paths.

`/home/chakwong/python` owns:

- DSGE economic semantics;
- DSGE state timing and ordering;
- Rotemberg, SGU, EZ, and future NAWM completion maps;
- client-side residual tests;
- model-specific HMC experiments after BayesFilter gates are ready.

`/home/chakwong/MacroFinance` owns:

- MacroFinance model construction;
- analytic Kalman derivative providers;
- model-specific parameter transforms and economic restrictions.

BayesFilter may adapt client-owned objects, but it must not encode DSGE or
MacroFinance economics in the generic filter core.

## Hard stop rules

Stop and record a blocker if a step would:

- infer structural roles solely from zero rows of a shock-impact matrix;
- add artificial process noise to deterministic coordinates;
- promote first-order bridge evidence to second-order/pruned correctness;
- promote adapter readiness to nonlinear filtering correctness;
- certify SVD/eigen derivatives without finite-difference, JVP/VJP, Hessian,
  and spectral-gap evidence;
- run HMC before the same model/backend pair has residual, derivative, and
  compiled parity evidence;
- claim convergence from finite smoke tests;
- stage unrelated PDFs, templates, generated files, or another agent's dirty
  work.

GPU/CUDA note: any HMC benchmark, framework GPU probe, XLA-GPU test, or GPU
diagnostic must run with escalated sandbox permissions.  Deliberate CPU-only
runs must set the project-standard CPU-hiding environment variable before
framework import and record the CPU-only choice in the result artifact.

## Evidence labels

Use these labels consistently in tests, result notes, and documentation:

| Label | Meaning |
| --- | --- |
| `adapter_ready_only` | Metadata can be adapted, but nonlinear residual evidence is absent or failing. |
| `model_residuals_passed` | The model-specific deterministic completion residuals pass on declared grids. |
| `model_residuals_blocked:<reason>` | Residual work ran and produced a named blocker. |
| `source_backed_timing_metadata` | Timing and partition metadata are supported by a reviewed source/provenance note. |
| `value_only` | Value tests pass, but derivative/JIT/HMC promotion remains blocked. |
| `derivative_certified` | Gradient/Hessian checks pass inside a recorded validity region. |
| `custom_derivative_required` | Autodiff is insufficient and a custom derivative rule is required. |
| `factor_backend_switch_required` | SVD/eigen factorization is unsuitable for the target certification region. |
| `compiled_value_gradient_parity_passed` | Compiled value and derivative results match eager reference within tolerance. |
| `finite_smoke` | The target runs and returns finite values only. |
| `hmc_diagnostics_passed` | Multi-chain diagnostics and posterior recovery pass for a named target. |
| `hmc_not_justified` | Earlier gates have not closed for the target model/backend pair. |

## Phase 0: preflight and evidence freeze

### Objective

Freeze the baseline before changing either BayesFilter or the DSGE client.

### Work location

- `/home/chakwong/BayesFilter`
- `/home/chakwong/python`

### Actions

1. Record BayesFilter status:

```bash
cd /home/chakwong/BayesFilter
git status --short --branch
git log -5 --oneline --decorate
```

2. Record DSGE client status:

```bash
cd /home/chakwong/python
git status --short --branch
git log -5 --oneline --decorate
```

3. Record current BayesFilter guardrails:

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_dsge_adapter_gate.py tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
```

4. Record current DSGE residual blockers:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_dsge_structural_completion_residuals.py \
  tests/contracts/test_structural_dsge_partition.py
```

### Exit gate

Continue only after the write set is explicit.  If `/home/chakwong/python` has
dirty reset-memo or agent files, do not overwrite them.

## Phase 1: Rotemberg second-order/pruned completion

### Gap

Rotemberg is blocked by:

```text
blocked_pruned_second_order_dy_identity_residual
```

The current second-order/pruned completion path does not preserve:

```text
dy_next = y_next - y_current
```

pointwise on the structural support.

### Hypothesis

Rotemberg needs a second-order/pruned deterministic completion map that treats
`dy` as a deterministic identity derived from current and lagged `y`, instead
of transporting `dy` as an independently propagated coordinate.

### Work location

`/home/chakwong/python`

### Planned changes

1. Inspect the current Rotemberg state convention, including:
   - state names;
   - pruned state blocks;
   - measurement equation construction;
   - current and lagged `y` timing;
   - existing `bayesfilter_deterministic_completion`.
2. Create or update a focused client test:

```text
tests/contracts/test_rotemberg_structural_completion_residuals.py
```

3. Build deterministic sigma-point grids over:
   - previous state;
   - innovation;
   - representative parameters;
   - boundary-adjacent valid parameters.
4. Implement the narrowest completion-map change needed in the DSGE client.
5. Assert:

```text
abs(dy_next - (y_next - y_current)) <= tolerance
```

for first-order and second-order/pruned paths separately.
6. Preserve a fail-closed assertion for missing or failing metadata.

### Tests

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_rotemberg_structural_completion_residuals.py \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_structural_dsge_partition.py
```

BayesFilter adapter guard:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

### Exit gate

Rotemberg advances only when the intended model order earns:

```text
model_residuals_passed
```

Otherwise record:

```text
model_residuals_blocked:<exact_failed_identity>
```

## Phase 2: SGU nonlinear equilibrium-manifold completion

### Gap

SGU is blocked by:

```text
blocked_nonlinear_equilibrium_manifold_residual
```

The current linear `h_x` completion bridge is not enough for nonlinear
structural filtering.

### Hypothesis

SGU needs a nonlinear equilibrium-manifold completion for deterministic
coordinates `d,k,r,riskprem`, with residuals evaluated directly against the
canonical equations, not only against a linearized transition bridge.

### Work location

`/home/chakwong/python`

### Planned changes

1. Create or update:

```text
tests/contracts/test_sgu_structural_completion_residuals.py
```

2. Write residual functions for:
   - `d`;
   - `k`;
   - `r`;
   - `riskprem`.
3. Generate deterministic grids over:
   - previous state;
   - current innovations;
   - representative parameters;
   - boundary-adjacent valid parameters.
4. Add finite-output checks before residual checks.
5. Implement the narrowest SGU completion-map change in the DSGE client.
6. Test residual norms against predeclared tolerances.
7. Keep separate labels for first-order, second-order, and pruned evidence.

### Tests

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_sgu_structural_completion_residuals.py \
  tests/contracts/test_dsge_strong_structural_residual_gates.py \
  tests/contracts/test_structural_dsge_partition.py
```

BayesFilter adapter guard:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

### Exit gate

SGU advances only when all declared deterministic-coordinate residuals earn:

```text
model_residuals_passed
```

Any failing coordinate becomes:

```text
model_residuals_blocked:<coordinate>:<residual_name>
```

## Phase 3: EZ source-backed timing and partition metadata

### Gap

EZ is blocked by:

```text
blocked_pending_source_backed_timing_metadata
```

A two-state/two-shock code probe is not sufficient structural metadata.

### Hypothesis

EZ can probably become adapter-ready, but only after a source-backed timing
note identifies state ordering, shock timing, predetermined coordinates, and
measurement timing.

### Work location

`/home/chakwong/python`

### Planned changes

1. Create a timing/partition audit note:

```text
docs/plans/dsge-ez-bayesfilter-timing-partition-audit-2026-05-06.md
```

2. Record:
   - model source/provenance;
   - state names and ordering;
   - innovation names and timing;
   - current and lagged state convention;
   - predetermined or auxiliary coordinates;
   - measurement timing;
   - whether any deterministic completion map exists.
3. If the timing audit is sufficient, expose EZ metadata in the DSGE client.
4. Add or update fail-closed contract tests.
5. If the audit is not sufficient, keep EZ blocked and name the missing source
   or timing equation.

### Tests

If metadata is exposed:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_structural_dsge_partition.py
```

BayesFilter adapter guard:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/python/src pytest -q tests/test_dsge_adapter_gate.py
```

### Exit gate

EZ may advance to residual tests only with:

```text
source_backed_timing_metadata
```

Otherwise preserve:

```text
blocked_pending_source_backed_timing_metadata
```

## Phase 4: BayesFilter contract patch, only if needed

### Gap

BayesFilter currently passes generic structural value tests.  It should not be
rewritten merely because client models have blockers.

### Hypothesis

Most residual closure work belongs in the DSGE client.  BayesFilter changes are
justified only if Rotemberg, SGU, or EZ exposes a missing generic abstraction in
the structural metadata contract.

### Work location

`/home/chakwong/BayesFilter`

### Planned changes

Patch only generic code paths, such as:

- metadata validation in `bayesfilter/structural.py`;
- adapter normalization in `bayesfilter/adapters/dsge.py`;
- result metadata labels;
- generic structural fixture coverage.

Do not add DSGE-specific variable names, equations, or economic timing rules to
BayesFilter.

### Tests

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_structural_partition.py tests/test_structural_sigma_points.py \
  tests/test_structural_ar_p.py tests/test_dsge_adapter_gate.py tests/test_filter_metadata.py
pytest -q
```

### Exit gate

Any BayesFilter patch must preserve exact/structural/approximation separation
and keep the full suite green.

## Phase 5: derivative and Hessian certification

### Gap

SVD/eigen derivatives remain:

```text
value_only_blocked
```

### Hypothesis

Spectral derivatives will be fragile near small gaps unless BayesFilter records
validity regions, implements custom derivative rules, or switches factorization
policy.

### Work location

Primarily `/home/chakwong/BayesFilter`; optional client tests only after a DSGE
model earns `model_residuals_passed`.

### Entry gate

Do not start model-specific derivative certification until at least one
model/backend pair has residual evidence:

```text
model_residuals_passed
```

Generic derivative policy tests may start earlier on toy fixtures if they do
not claim DSGE readiness.

### Planned changes

1. Create:

```text
docs/plans/bayesfilter-structural-svd-derivative-certification-plan-2026-05-06.md
tests/test_structural_svd_derivative_certification.py
```

2. Split obligations into:
   - parameter transform;
   - transition and deterministic completion;
   - sigma-point construction;
   - covariance propagation;
   - observation prediction;
   - innovation covariance factorization;
   - log-likelihood update.
3. Add tests for:
   - finite-difference gradient agreement;
   - JVP/VJP parity;
   - Hessian symmetry;
   - Hessian-vector product consistency;
   - small spectral-gap stress;
   - explicit refusal to promote missing derivative evidence.
4. Record minimum spectral-gap thresholds and tolerances.
5. Choose one policy:
   - certify SVD/eigen only within recorded validity regions;
   - add custom derivatives;
   - switch to a non-spectral factorization for HMC targets;
   - keep the backend value-only.

### Tests

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_derivative_validation_smoke.py tests/test_backend_readiness.py
pytest -q tests/test_structural_svd_derivative_certification.py
```

### Exit gate

Each backend/model pair must receive exactly one status:

```text
derivative_certified
custom_derivative_required
factor_backend_switch_required
value_only_blocked
```

## Phase 6: compiled static-shape parity

### Gap

Compiled parity remains:

```text
compiled_parity_not_started
```

### Hypothesis

A compiled target can probably match the eager value path on exact LGSSM and
generic structural fixtures, but it must be proven before any HMC benchmark.

### Work location

`/home/chakwong/BayesFilter`

### Entry gate

Compiled value parity may begin on exact LGSSM and generic fixtures.

Compiled DSGE parity must wait for the same model/backend pair to have:

```text
model_residuals_passed
derivative_certified
```

if gradient or HMC claims are planned.

### Planned changes

1. Create:

```text
docs/plans/bayesfilter-structural-svd-compiled-parity-plan-2026-05-06.md
tests/test_compiled_structural_svd_parity.py
```

2. Choose backend policy:
   - JAX for custom JVP/VJP and static-shape control;
   - TensorFlow for DSGE client compatibility;
   - eager-only if compiled work is premature.
3. Declare static dimensions:
   - state dimension;
   - innovation dimension;
   - observation dimension;
   - time dimension;
   - parameter dimension;
   - mask convention.
4. Compare eager and compiled:
   - log likelihood;
   - filtered mean and covariance;
   - gradient, when certified;
   - selected Hessian-vector products, when certified;
   - failure behavior;
   - metadata labels.

### Tests

```bash
cd /home/chakwong/BayesFilter
pytest -q tests/test_compiled_structural_svd_parity.py tests/test_backend_readiness.py
pytest -q
```

Optional DSGE client parity after model gates pass:

```bash
cd /home/chakwong/python
PYTHONPATH=/home/chakwong/python/src:/home/chakwong/BayesFilter pytest -q \
  tests/contracts/test_dsge_bayesfilter_compiled_parity.py
```

### Exit gate

HMC may use only a target with:

```text
compiled_value_gradient_parity_passed
```

unless the HMC result is explicitly labeled eager-only exploratory work and not
a production claim.

## Phase 7: HMC diagnostics ladder

### Gap

HMC remains:

```text
hmc_not_justified
```

### Hypothesis

HMC diagnostics become meaningful only after the same model/backend pair has
residual, derivative, and compiled parity evidence.

### Work location

BayesFilter for generic ladder rungs; `/home/chakwong/python` for DSGE-specific
experiments.

### Entry gate

For every HMC target, require:

```text
value tests passed
model_residuals_passed
derivative_certified
compiled_value_gradient_parity_passed
```

Exact LGSSM may skip model residuals because it is the exact full-state
linear-Gaussian reference.

### Ladder order

1. exact LGSSM recovery;
2. generic nonlinear structural fixture;
3. structural AR/lag fixture;
4. SmallNK only if the adapter and target-specific value path are green;
5. Rotemberg after Phases 1, 5, and 6 pass;
6. SGU after Phases 2, 5, and 6 pass;
7. EZ after Phases 3, residual tests, 5, and 6 pass;
8. NAWM-scale only after smaller DSGE targets pass.

### Required result fields

Every HMC result note must record:

- target;
- backend and compiled status;
- seed;
- chain count;
- warmup count;
- sample count;
- adaptation settings;
- gradient finite checks;
- acceptance rate;
- divergences;
- split R-hat;
- effective sample size;
- posterior recovery metric;
- runtime;
- CPU/GPU execution choice.

### Tests and command shape

Use a dedicated script or pytest target, for example:

```bash
cd /home/chakwong/BayesFilter
PYTHONPATH=/home/chakwong/BayesFilter python \
  scripts/run_structural_svd_hmc_ladder.py --target exact-lgssm --seed 123
```

If the run uses GPU/CUDA, run with escalated sandbox permissions according to
the local GPU policy.

### Exit gate

A target advances only with:

```text
hmc_diagnostics_passed
```

Finite-only runs must remain:

```text
finite_smoke
```

## Phase 8: documentation, provenance, and release cleanup

### Gap

Documentation must not claim more than the evidence supports.

### Hypothesis

The safest release posture is to update claim labels after each gate, and to
delay monograph/Chapter 18b promotion language until residual, derivative,
compiled, and HMC evidence exists.

### Work location

`/home/chakwong/BayesFilter`

### Planned changes

1. Update the active reset memo after every phase.
2. Add result notes for each closed gate.
3. Register new BayesFilter artifacts in `docs/source_map.yml`.
4. Update `docs/chapters/ch18b_structural_deterministic_dynamics.tex` only
   when the evidence supports stronger structural claims.
5. Keep exact LGSSM, structural nonlinear filtering, and labeled
   approximations separate.
6. Search for stale promotion claims:

```bash
cd /home/chakwong/BayesFilter
rg -n "converged|production-ready|HMC-ready|certified|structurally fixed" docs bayesfilter tests
```

### Tests

```bash
cd /home/chakwong/BayesFilter
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8'))"
git diff --check
pytest -q
```

If LaTeX changed:

```bash
cd /home/chakwong/BayesFilter/docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

### Exit gate

Docs, tests, result notes, source map, and reset memo agree on the allowed
claim labels.

## Suggested commit sequence

Use one commit per evidence boundary:

1. Rotemberg residual completion and result.
2. SGU residual completion and result.
3. EZ timing/partition audit and metadata, if supported.
4. Any generic BayesFilter contract patch forced by client residual work.
5. Derivative certification policy and tests.
6. Compiled static-shape parity.
7. HMC ladder rung results.
8. Final documentation/provenance cleanup.

Do not mix BayesFilter docs with unrelated client changes in the same commit
unless the commit is explicitly a cross-repo handoff note.

## Immediate next action

Start with Phase 1 in `/home/chakwong/python`.

The first concrete task is to make the Rotemberg blocker constructive: write a
focused residual test for `dy_next - (y_next - y_current)`, implement the
narrowest second-order/pruned deterministic completion change that makes the
identity pass, and preserve fail-closed behavior when the residual gate fails.

Do not start derivative, compiled parity, or HMC work until at least one
model-specific residual gate has actually passed.
