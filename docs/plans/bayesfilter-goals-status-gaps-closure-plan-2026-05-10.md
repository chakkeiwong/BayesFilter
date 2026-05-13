# Plan: BayesFilter Goals, Status, Gaps, and Gap-closure Path

## Date

2026-05-10

## Purpose

This note consolidates the current filtering implementation status after the
seven-phase TensorFlow filtering pass and the post-pass MacroFinance
switch-over audit.  It states the goals, what is already true, what remains
blocked, and the execution path for closing the gaps without leaving the
BayesFilter lane.

## Current Repository State

Baseline pushed to `origin/main`:

```text
f11938c docs: record fetch conflict check
```

The branch was synchronized before this plan:

```text
git fetch origin
git rev-list --left-right --count HEAD...origin/main
0 0
```

Unrelated untracked files remain outside this plan:

```text
docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md
docs/plans/templates/*:Zone.Identifier
singularity_test.png
```

## Goals

1. Make BayesFilter the shared TensorFlow/TensorFlow Probability filtering
   library for linear, nonlinear, structural, and derivative-aware filters.
2. Replace duplicated filtering logic in MacroFinance and the DSGE client only
   through client-side adapters with parity tests and rollback.
3. Keep BayesFilter production code free of NumPy and client-economics imports.
   NumPy remains allowed in tests and `bayesfilter.testing`.
4. Keep linear QR derivatives as the production derivative spine unless a real
   singular-client use case proves SVD/eigen derivatives are needed.
5. Use SVD/eigen factors for value-side singular or rank-deficient covariance
   handling, while reporting implemented-law diagnostics.
6. Use structural nonlinear filters only after the client supplies a causal
   local filtering target and parity observations.
7. Make GPU/XLA and HMC claims only after target-specific benchmark, branch,
   derivative, and sampler diagnostics pass.

## Status

### Completed

- Static MacroFinance compatibility tests exist in BayesFilter for dense and
  masked QR value, score, and Hessian fixtures.
- TensorFlow QR/square-root linear value filters exist for dense and masked
  observation paths.
- TensorFlow QR/square-root score/Hessian filters exist for dense and masked
  static derivative fixtures.
- TensorFlow SVD/eigen linear value filters exist for dense and masked paths.
- Generic TensorFlow structural protocols exist and distinguish stochastic,
  endogenous, deterministic-completion, and collapsed-state roles.
- SVD cubature and SVD-UKF value filters exist for structural TensorFlow
  targets.
- CUT4-G and SVD-CUT value filters exist.
- Smooth-branch SVD-CUT score/Hessian exists for the implemented regularized
  value law and fails closed on active floors or weak spectral gaps.
- CPU graph-compiled parity tests exist for promoted value and derivative
  paths.
- The BayesFilter implementation baseline has been pushed.
- The MacroFinance switch-over audit has selected the smallest first client
  target:

```text
/home/chakwong/MacroFinance/inference/hmc.py
tf_lgssm_log_likelihood_backend(
    backend="tf_direct_qr",
    observation_mask=None,
    ...
)
```

### Not Yet Authorized

- MacroFinance production defaults are not authorized to change.
- DSGE production defaults are not authorized to change.
- SGU production filtering remains blocked until the DSGE client supplies a
  causal local filtering target.
- GPU/XLA-GPU speed claims are not authorized.
- HMC readiness claims are not authorized for SVD-CUT or DSGE targets.
- Linear SVD/eigen derivatives are not authorized for production.

## Gaps

### Gap 1: MacroFinance Dense QR Switch-over

BayesFilter matches MacroFinance static QR fixtures inside BayesFilter tests,
but MacroFinance itself still calls its local filtering code.  The first missing
piece is a MacroFinance-side wrapper that optionally calls BayesFilter for
dense static `tf_direct_qr` value dispatch.

### Gap 2: MacroFinance Masked QR And QR Derivative Switch-over

Masked QR value and QR score/Hessian parity exist in BayesFilter tests, but
client-side MacroFinance dispatch has not yet been switched.  The masked path
must preserve the static dummy-row convention.  The derivative path must
preserve parameter-major derivative tensor ordering and posterior-adapter
behavior.

### Gap 3: MacroFinance SVD/eigen Value Decision

BayesFilter has SVD/eigen value filters, but MacroFinance has no currently
documented production-shaped fixture that requires them.  The current hard QR
surrogate does not trigger SVD fallback thresholds.

### Gap 4: DSGE Structural Target Inventory

BayesFilter has generic structural protocols and nonlinear filters, but the
DSGE client must choose a model-owned causal local target.  SGU remains in a
diagnostic lane until locality is proven.

### Gap 5: Nonlinear Client Adapter Parity

No real DSGE nonlinear value target has been switched to BayesFilter.  The
first adapter should be non-SGU and should prove deterministic-completion
residuals, support rank, and value parity.

### Gap 6: Benchmark Evidence

Current evidence is CPU correctness and graph parity.  There is no benchmark
artifact for representative dimensions, point counts, compile time,
steady-state time, or memory.

### Gap 7: Escalated GPU/XLA-GPU Evidence

GPU access must be checked with escalated permissions on this machine.  No
GPU/XLA-GPU filtering claim is valid until device probes and exact-shape
benchmark artifacts exist.

### Gap 8: HMC Readiness

Generic derivative tests do not establish HMC readiness.  HMC requires a
specific target model/backend pair, branch-frequency evidence, finite
gradient/Hessian behavior, compiled parity, and sampler diagnostics.

### Gap 9: Linear SVD/eigen Derivative Need

SVD/eigen derivatives are mathematically branch-sensitive and should not be
implemented merely because value-side SVD exists.  A real client need must be
demonstrated first.

## Execution Plan

Each phase should follow:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Continue automatically only when the primary criterion passes and no veto
diagnostic fires.

### Phase 1: MacroFinance Dense QR Value Pilot

Location:

```text
/home/chakwong/MacroFinance
```

Actions:

- add a MacroFinance-side optional backend, for example
  `bayesfilter_tf_direct_qr`;
- construct a BayesFilter `TFLinearGaussianStateSpace` from MacroFinance tensor
  arguments;
- call `tf_qr_linear_gaussian_log_likelihood(..., backend="tf_qr")`;
- keep existing MacroFinance defaults unchanged;
- add MacroFinance-side parity assertions against the existing `tf_direct_qr`
  path.

Primary tests:

```text
/home/chakwong/MacroFinance/tests/test_tf_kalman.py::test_tf_lgssm_backend_selector_direct_qr_matches_default
/home/chakwong/MacroFinance/tests/test_production_lgssm_backend_fixture.py::test_production_shaped_surrogate_exercises_default_and_direct_qr_backends
```

Primary criterion:

- MacroFinance dense QR value through BayesFilter matches existing MacroFinance
  dense QR value within current tolerances and does not change defaults.

Veto diagnostics:

- circular import between BayesFilter and MacroFinance;
- production NumPy added to BayesFilter;
- graph retracing or dtype drift;
- default backend behavior changes before parity is proven.

### Phase 2: MacroFinance Masked QR Value Pilot

Actions:

- add a BayesFilter-backed masked QR option only after Phase 1 passes;
- preserve MacroFinance's static dummy-row mask convention;
- surface the mask convention in diagnostics.

Primary tests:

```text
/home/chakwong/MacroFinance/tests/test_tf_masked_kalman.py
tests/test_macrofinance_linear_compat_tf.py
```

Primary criterion:

- masked BayesFilter-backed QR value matches MacroFinance masked QR and masked
  covariance references on all-true, sparse, and all-missing rows.

Veto diagnostics:

- mismatch in mask shape or dummy-row convention;
- all-missing row contributes a nonzero measurement likelihood;
- hidden mask-dependent retracing.

### Phase 3: MacroFinance QR Score/Hessian Pilot

Actions:

- add BayesFilter-backed dense QR score/Hessian option;
- keep QR derivatives as the derivative production spine;
- prove parameter-major derivative tensor ordering;
- check one-country analytical posterior behavior before any HMC use.

Primary tests:

```text
/home/chakwong/MacroFinance/tests/test_tf_qr_sqrt_differentiated_kalman.py
/home/chakwong/MacroFinance/tests/test_one_country_analytic_backend_parity.py
tests/test_macrofinance_linear_compat_tf.py
```

Primary criterion:

- value, score, and Hessian match existing MacroFinance QR derivative outputs
  on static fixtures and one-country analytical backend fixtures.

Veto diagnostics:

- derivative tensor order mismatch;
- Hessian asymmetry beyond current tolerance;
- posterior adapter calls Hessian during gradient-only HMC/MAP steps;
- client HMC behavior changes before explicit HMC gate.

### Phase 4: MacroFinance SVD/eigen Value Decision

Actions:

- search MacroFinance fixtures for singular or near-singular covariance cases;
- run BayesFilter SVD/eigen value only where QR/Cholesky diagnostics justify it;
- compare implemented-law diagnostics against existing MacroFinance SVD value
  behavior.

Primary criterion:

- either document no current SVD/eigen client need, or add an optional
  BayesFilter-backed SVD/eigen value backend for a specific singular fixture.

Veto diagnostics:

- SVD value changes regular QR/Cholesky results without explanation;
- implemented covariance diagnostics are not surfaced;
- derivative requests are mixed into this value-only decision.

### Phase 5: DSGE Target Inventory

Location:

```text
/home/chakwong/python
```

Actions:

- inventory toy affine controls, existing UKF/SVD examples, CUTSRUKF examples,
  and SGU candidates;
- classify each as ready, adapter-needed, blocked-by-law, or diagnostic-only;
- pick one non-SGU target for first structural BayesFilter parity.

Primary criterion:

- a first DSGE structural target is selected that does not depend on unresolved
  SGU causal-locality issues.

Veto diagnostics:

- only candidate is SGU without a causal local target;
- target requires BayesFilter production imports from DSGE;
- parity observations are unavailable.

### Phase 6: DSGE Non-SGU Structural Adapter

Actions:

- implement adapter in DSGE or a test-only bridge, not BayesFilter economics;
- map the target into `TFStructuralStateSpace`;
- run SVD cubature, SVD-UKF, or SVD-CUT value filter as appropriate;
- compare against the existing DSGE filter output.

Primary criterion:

- target-specific nonlinear value parity passes and deterministic-completion
  residuals are zero.

Veto diagnostics:

- collapsed full-state metadata used without label;
- hidden regularization needed for finite likelihood;
- support rank or deterministic residual diagnostics fail.

### Phase 7: CPU Benchmark Harness

Actions:

- benchmark linear QR, linear SVD/eigen value, SVD cubature/UKF, SVD-CUT value,
  and SVD-CUT smooth-branch derivatives;
- record dtype, time length, state dimension, innovation rank, observation
  dimension, point count, compile time, steady-state time, and memory where
  available.

Primary criterion:

- CPU benchmark artifacts exist for representative small and medium dimensions.

Veto diagnostics:

- compile time hidden;
- point count omitted;
- eager and compiled paths mixed without labels.

### Phase 8: Escalated GPU/XLA-GPU Benchmarks

Actions:

- run escalated `nvidia-smi`;
- run escalated TensorFlow GPU device probe;
- run exact-shape CPU/GPU benchmark comparisons where XLA-GPU is supported.

Primary criterion:

- benchmark artifact compares CPU and GPU on the same backend/model shapes and
  records device, driver, TensorFlow, TFP, CUDA visibility, dtype, and shape
  metadata.

Veto diagnostics:

- GPU probe run without escalation;
- failed non-escalated GPU probe treated as driver evidence;
- CPU/GPU shapes differ.

### Phase 9: Target-specific HMC Readiness Gate

Actions:

- choose one exact model/backend pair after value and derivative parity pass;
- sample representative posterior-region parameter points;
- record active floor frequency, weak spectral-gap frequency, support
  residuals, deterministic residuals, nonfinite likelihoods, and nonfinite
  gradients;
- only then run a small fixed-seed HMC/NUTS smoke if branch diagnostics pass.

Primary criterion:

- finite value/gradient behavior, acceptable branch frequency, and small HMC
  sampler diagnostics on the exact target.

Veto diagnostics:

- active floors or weak spectral gaps occur in typical posterior regions;
- gradients fail or disagree with finite differences/autodiff where expected;
- sampler stability requires undocumented jitter or floors.

### Phase 10: Linear SVD/eigen Derivative Need Assessment

Actions:

- check whether any MacroFinance or DSGE use case truly requires linear
  SVD/eigen derivatives;
- if QR derivatives suffice, document that no implementation is justified;
- if a real need exists, design a testing-only smooth-branch prototype with
  separated-spectrum and inactive-floor requirements.

Primary criterion:

- a concrete client need exists before any derivative implementation begins.

Veto diagnostics:

- request is theoretical only;
- active floors are common in the target region;
- repeated spectra cannot be detected reliably.

## Recommended Order

1. Phase 1: MacroFinance dense QR value pilot.
2. Phase 2: MacroFinance masked QR value pilot.
3. Phase 3: MacroFinance QR score/Hessian pilot.
4. Phase 4: MacroFinance SVD/eigen value decision.
5. Phase 5: DSGE target inventory.
6. Phase 6: DSGE non-SGU structural adapter.
7. Phase 7: CPU benchmark harness.
8. Phase 8: escalated GPU/XLA-GPU benchmarks.
9. Phase 9: target-specific HMC readiness gate.
10. Phase 10: linear SVD/eigen derivative need assessment.

This order closes the nearest client gap first and prevents GPU/HMC or
SVD-derivative claims from outrunning client-side value parity.

## Immediate Next Action

Request permission to edit `/home/chakwong/MacroFinance`, then execute Phase 1
as a MacroFinance-side optional backend with default behavior unchanged.

If MacroFinance edits are deferred, the best BayesFilter-only next action is
Phase 7: create CPU benchmark harness artifacts, because that does not require
client repository writes and does not alter production defaults.
