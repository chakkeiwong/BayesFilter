# Plan: Post Seven-phase Filtering Gap Closure

## Date

2026-05-10

## Purpose

The seven-phase BayesFilter implementation pass finished the generic TensorFlow
filtering spine:

- static MacroFinance linear compatibility tests;
- SVD/eigen linear value filters;
- structural TensorFlow protocols;
- SVD cubature and UKF value filters;
- CUT4-G and SVD-CUT value filters;
- smooth-branch SVD-CUT score/Hessian;
- CPU graph parity tests.

This plan closes the remaining gaps without promoting unsupported claims.  The
execution path is intentionally staged: synchronize first, then switch over the
nearest real client, then certify target-specific nonlinear and GPU/HMC claims.

## Current Baseline

Committed implementation:

```text
68e1792 Implement TF SVD and CUT filtering gates
```

Final verification from the implementation pass:

```text
CUDA_VISIBLE_DEVICES=-1 pytest -q
154 passed, 2 warnings in 161.27s
```

Known warnings:

```text
TensorFlow Probability distutils deprecation warnings
```

Current switch-over boundary:

```text
docs/plans/bayesfilter-client-switch-over-boundary-2026-05-09.md
```

## Global Rules

1. Production BayesFilter filtering code remains TensorFlow / TensorFlow
   Probability only.
2. NumPy remains allowed in tests and `bayesfilter.testing` reference helpers.
3. No client default changes without client-side parity tests and rollback.
4. GPU/CUDA/XLA-GPU claims require escalated device probes and benchmark
   artifacts.
5. HMC readiness requires exact model/backend/derivative-branch parity, not only
   generic filter tests.
6. SGU production filtering remains blocked until the DSGE client supplies a
   causal local filtering target.
7. Each execution phase should end with:
   `plan -> execute -> test -> audit -> tidy -> update reset memo`.

## Gap A: Local Commit Not Pushed

### Motivation

The implementation is committed locally but not on `origin/main`.  Further
client work should start from a synchronized remote baseline.

### Phase A1: Push BayesFilter Baseline

Actions:

- run `git status --short`;
- run `git rev-list --left-right --count HEAD...origin/main`;
- if only ahead by the implementation commit, push;
- if remote has moved, fetch/pull with conflict review before pushing;
- leave unrelated untracked files unstaged.

Primary criterion:

- `origin/main` contains commit `68e1792` or its non-rewritten descendant.

Veto diagnostics:

- stop if remote has conflicting changes in the same files;
- stop if unrelated untracked files would be staged;
- stop if tests must be rerun after conflict resolution and have not passed.

Artifacts:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
```

## Gap B: MacroFinance Production Switch-over

### Motivation

BayesFilter already matches the static MacroFinance linear QR fixture inside
BayesFilter tests.  The remaining risk is client integration: imports, wrappers,
missing-observation conventions, parameter tensor ordering, and rollback.

### Phase B1: MacroFinance Adapter Audit

Actions:

- inspect MacroFinance filtering entry points read-only;
- identify the first production path to switch:
  - dense QR value;
  - masked QR value;
  - dense QR score/Hessian;
  - masked QR score/Hessian;
- identify environment/import strategy:
  - editable BayesFilter dependency;
  - local path adapter;
  - feature-flag wrapper;
- record rollback boundary.

Primary criterion:

- a single smallest MacroFinance switch-over target is identified, with exact
  parity tests and rollback command.

Veto diagnostics:

- stop if MacroFinance needs time-varying derivative tensors for the first
  production path;
- stop if production import strategy would create circular imports;
- stop if BayesFilter and MacroFinance disagree on masks, jitter, or derivative
  tensor ordering.

Artifacts:

```text
docs/plans/macrofinance-bayesfilter-switch-over-audit-YYYY-MM-DD.md
```

### Phase B2: MacroFinance Linear QR Switch-over Pilot

Actions:

- implement a feature-flagged MacroFinance wrapper that calls BayesFilter for
  the chosen static QR path;
- add MacroFinance-side parity tests against existing MacroFinance backends;
- keep MacroFinance default behavior unchanged until tests pass;
- run the relevant MacroFinance test subset.

Primary criterion:

- MacroFinance-side tests prove BayesFilter parity for the selected path.

Veto diagnostics:

- stop if MacroFinance parity fails outside documented tolerances;
- stop if wrapper requires production NumPy inside BayesFilter;
- stop if BayesFilter dependency setup is brittle or non-reproducible.

Artifacts:

```text
MacroFinance switch-over PR or local branch
docs/plans/macrofinance-bayesfilter-switch-over-result-YYYY-MM-DD.md
```

### Phase B3: MacroFinance Linear SVD Value Decision

Actions:

- search MacroFinance fixtures for singular or near-singular linear covariance
  cases;
- run BayesFilter SVD/eigen value backend only where needed;
- compare diagnostics against existing MacroFinance SVD behavior;
- decide whether SVD/eigen value should be exposed as a client option.

Primary criterion:

- SVD/eigen value is used only for documented singular or near-singular value
  cases.

Veto diagnostics:

- stop if anyone requests SVD/eigen derivatives from this phase;
- stop if implemented-law diagnostics are not surfaced to MacroFinance users;
- stop if SVD value changes a regular QR/Cholesky result without explanation.

## Gap C: DSGE Structural Switch-over

### Motivation

BayesFilter now owns generic structural protocols, but DSGE economics should
remain in the DSGE repository.  The first DSGE task is an adapter, not a model
rewrite.

### Phase C1: DSGE Target Inventory

Actions:

- inventory DSGE filtering targets:
  - toy affine controls;
  - existing UKF/SVD examples;
  - CUTSRUKF experimental example;
  - SGU candidates;
- label each target as:
  - ready generic structural fixture;
  - needs adapter work;
  - blocked by model law;
  - diagnostic only;
- identify one non-SGU generic nonlinear target for first parity.

Primary criterion:

- a first DSGE structural target is selected that does not depend on unresolved
  SGU causal-locality issues.

Veto diagnostics:

- stop if the only candidate is SGU without a causal local filtering target;
- stop if target requires BayesFilter to import DSGE economics;
- stop if target lacks observation/parity data.

Artifacts:

```text
docs/plans/dsge-bayesfilter-structural-target-inventory-YYYY-MM-DD.md
```

### Phase C2: DSGE Adapter Prototype

Actions:

- implement the adapter in DSGE or a test-only bridge, not in BayesFilter
  production economics code;
- map DSGE state partition into `TFStructuralStateSpace`;
- run BayesFilter SVD cubature/UKF/CUT value filters on the selected target;
- compare against DSGE existing filter output.

Primary criterion:

- target-specific value parity is established for at least one DSGE nonlinear
  structural fixture.

Veto diagnostics:

- stop if deterministic completion residuals are nonzero;
- stop if collapsed full-state metadata is used without explicit label;
- stop if parity requires hidden regularization not reported in diagnostics.

### Phase C3: SGU Causal Target Gate

Actions:

- keep SGU outside production filtering until DSGE supplies a local causal
  transition/observation target;
- treat residual-closing nonlocal two-slice objects as diagnostics or smoother
  candidates only;
- add a minimal SGU gate test in DSGE when a candidate target exists.

Primary criterion:

- `sgu_causal_filtering_target_passed` exists and proves locality.

Veto diagnostics:

- stop if SGU target uses future marginal utility or nonlocal residual repair
  inside a one-step predictive filter;
- stop if target only passes after smoothing information is injected.

## Gap D: GPU/XLA-GPU Evidence

### Motivation

CUT point counts can become large.  GPU/XLA may help throughput, but the point
count remains mathematical.  Performance claims need device and shape evidence.

### Phase D1: CPU Benchmark Harness

Actions:

- add benchmark scripts for:
  - linear SVD value;
  - SVD cubature/UKF value;
  - SVD-CUT value;
  - smooth-branch SVD-CUT derivatives;
- record:
  - dtype;
  - time length;
  - state dimension;
  - innovation rank;
  - observation dimension;
  - point count;
  - compile time;
  - steady-state time;
  - memory if available.

Primary criterion:

- CPU benchmark artifacts exist for representative small/medium dimensions.

Veto diagnostics:

- stop if benchmark hides compile time;
- stop if dimension/rank/point-count metadata is missing;
- stop if results mix eager and compiled paths without labels.

### Phase D2: Escalated GPU/XLA-GPU Benchmarks

Actions:

- run escalated `nvidia-smi`;
- run escalated TensorFlow GPU device probe;
- run benchmark subset with XLA-GPU where supported;
- record device, driver, TensorFlow, TFP, CUDA visibility, and shapes.

Primary criterion:

- GPU benchmark artifact compares CPU and GPU for exact same backend/model
  shapes.

Veto diagnostics:

- stop if GPU access is non-escalated;
- stop if device probe fails and is not rerun escalated;
- stop if benchmark uses different shapes or dtypes across CPU/GPU.

Artifacts:

```text
docs/benchmarks/bayesfilter-filter-backend-benchmark-YYYY-MM-DD.{json,md}
```

## Gap E: HMC Readiness

### Motivation

HMC requires stable values and gradients on the exact target posterior.  Generic
SVD-CUT derivative tests do not imply HMC readiness for DSGE or MacroFinance
targets.

### Phase E1: Target-specific Derivative Branch Audit

Actions:

- choose one exact target model/backend pair;
- sample parameter points around the intended posterior region;
- record:
  - active floor frequency;
  - weak spectral-gap frequency;
  - support residuals;
  - deterministic residuals;
  - nonfinite likelihood/gradient frequency.

Primary criterion:

- smooth derivative branch is active on the target region often enough to
  justify HMC experiments.

Veto diagnostics:

- stop if branch blocks occur in typical posterior regions;
- stop if Hessian/gradient parity fails at representative points;
- stop if diagnostics cannot distinguish model law from regularization.

### Phase E2: Small HMC Smoke

Actions:

- run a short TFP HMC/NUTS smoke on a small target with fixed seed;
- compare chains under the old and BayesFilter backends where possible;
- record acceptance, divergences, step size, ESS/Rhat if available, and
  nonfinite events.

Primary criterion:

- smoke run has finite log probability and gradients, no systematic branch
  failures, and acceptable sampler diagnostics for the small target.

Veto diagnostics:

- stop if gradients fail due to active floors or repeated spectra;
- stop if HMC stability requires undocumented jitter/floors;
- stop if sampler diagnostics are poor and not explained by model scale.

## Gap F: Linear SVD/eigen Derivatives

### Motivation

The current SVD/eigen linear backend is value-only.  That is intentional.
Linear SVD/eigen derivative claims should be developed only if a real client
needs them and if branch assumptions can be certified.

### Phase F1: Need Assessment

Actions:

- check MacroFinance and DSGE singular linear use cases;
- decide whether QR derivatives suffice for optimization/HMC;
- identify cases where SVD/eigen derivatives would be genuinely required.

Primary criterion:

- a concrete client need exists before derivative implementation begins.

Veto diagnostics:

- stop if request is only theoretical and QR derivatives cover production;
- stop if active floors are common in the target region.

### Phase F2: Smooth-branch Linear SVD Derivative Prototype

Actions, only if F1 passes:

- implement a testing/reference prototype first;
- require separated spectra and inactive floors;
- compare against autodiff and finite differences;
- keep production export blocked until diagnostics are mature.

Primary criterion:

- finite-difference/autodiff parity on smooth low-dimensional cases.

Veto diagnostics:

- stop if repeated spectra or active floors cannot be detected reliably;
- stop if Hessian symmetry fails.

## Recommended Execution Order

1. Phase A1: push/sync BayesFilter baseline.
2. Phase B1: MacroFinance adapter audit.
3. Phase B2: MacroFinance static linear QR switch-over pilot.
4. Phase B3: MacroFinance SVD value decision.
5. Phase C1: DSGE target inventory.
6. Phase C2: DSGE non-SGU structural adapter prototype.
7. Phase D1: CPU benchmark harness.
8. Phase D2: escalated GPU/XLA-GPU benchmarks.
9. Phase E1: target-specific derivative branch audit.
10. Phase E2: small HMC smoke.
11. Phase F1/F2 only if a real client needs linear SVD/eigen derivatives.

This order starts with the closest real client and prevents GPU/HMC work from
outrunning target-specific value and derivative evidence.

## Immediate Next Action

Run Phase A1 and then Phase B1.

Phase A1 should be quick:

```text
git status --short
git rev-list --left-right --count HEAD...origin/main
git push
```

Phase B1 should produce:

```text
docs/plans/macrofinance-bayesfilter-switch-over-audit-YYYY-MM-DD.md
```

That audit should decide the first MacroFinance switch-over target and whether
the pilot can proceed without time-varying derivative support.
