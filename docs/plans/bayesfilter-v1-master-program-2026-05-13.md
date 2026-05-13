# BayesFilter V1 Master Program

## Date

2026-05-13

## Purpose

This is the single coherent master program for BayesFilter V1.  Earlier V1
work produced useful but dispersed plans for API freeze, external
compatibility, benchmarks, HMC gates, nonlinear model suites, SVD sigma-point
scores, and derivative-provider tests.  This document is the controlling
roadmap that keeps those artifacts aligned and prevents future drift.

The V1 objective is not to switch MacroFinance or DSGE over immediately.  The
objective is to make BayesFilter independently testable, mathematically honest,
and ready for later client integration.

## Governing Principle

BayesFilter V1 should be a TensorFlow/TensorFlow Probability filtering package
with explicit structural state-space contracts, tested local correctness, and
clear evidence boundaries.

Every claim must identify one of four statuses:

- `certified`: covered by local tests or benchmark artifacts at the stated
  scope;
- `diagnostic`: useful evidence but not a public readiness claim;
- `deferred`: intentionally left for a later V1 phase;
- `blocked`: not allowed to proceed until a named gate is passed.

## Lane Boundary

This master program owns only the BayesFilter V1 lane:

```text
bayesfilter/*
tests/test_v1_public_api.py
tests/test_linear_*
tests/test_nonlinear_*
tests/test_svd_cut*
tests/test_structural_svd_sigma_point_tf.py
docs/benchmarks/benchmark_bayesfilter_v1_*.py
docs/benchmarks/bayesfilter-v1-*
docs/chapters/ch18_svd_sigma_point.tex
docs/chapters/ch28_nonlinear_ssm_validation.tex
docs/plans/bayesfilter-v1-*.md
docs/source_map.yml entries whose keys begin with bayesfilter_v1_
pytest.ini
```

Protected unless the user explicitly opens another lane:

```text
docs/chapters/ch18b_structural_deterministic_dynamics.tex
docs/plans/bayesfilter-structural-*
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

External projects are compatibility targets, not V1 production dependencies.

## Supporting Artifacts

This master supersedes the following as control documents while preserving them
as evidence:

- `docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md`;
- `docs/plans/bayesfilter-v1-api-freeze-criteria-2026-05-10.md`;
- `docs/plans/bayesfilter-v1-ci-runtime-tier-policy-2026-05-11.md`;
- `docs/plans/bayesfilter-v1-nonlinear-filtering-master-testing-program-2026-05-11.md`;
- `docs/plans/bayesfilter-v1-nonlinear-filtering-remaining-master-program-plan-2026-05-12.md`;
- all `bayesfilter-v1-*result*.md` files as phase evidence.

Future subplans must cite this master and explain which goal/gap they close.

## V1 Goals

### G1. Stable Public API

BayesFilter V1 must have an import-stable public API for:

- linear Gaussian value filters;
- linear QR square-root value filters;
- linear QR score/Hessian filters;
- linear SVD/eigen value filters;
- structural TensorFlow state-space contracts;
- nonlinear SVD sigma-point and SVD-CUT4 value filters;
- smooth-branch SVD sigma-point scores with explicit derivative providers.

Non-goals:

- no MacroFinance production switch-over in V1 stabilization;
- no DSGE production switch-over in V1 stabilization;
- no SGU production filtering claim;
- no public SVD/eigen Hessian claim without a named consumer and gate.

### G2. Linear Filtering Correctness

Linear filtering must retain:

- dense Kalman value correctness;
- masked-observation semantics;
- QR square-root value parity;
- QR score/Hessian parity on local fixtures;
- compiled parity where already tested;
- SVD/eigen value behavior with floor and PSD-projection diagnostics.

Current status:

- mostly `certified` for local tests;
- linear QR HMC evidence remains `diagnostic`, not convergence certification;
- linear SVD/eigen derivatives remain `deferred`.

### G3. Structural State-space Contract

The structural API must keep stochastic, deterministic, innovation, and
observation blocks explicit.  Deterministic completion should be model metadata
and residual diagnostics, not hidden inside a large singular covariance when
that obscures timing or model-law claims.

Current status:

- structural block metadata and affine structural conversion are implemented;
- deterministic-residual diagnostics are used in nonlinear tests;
- DSGE-specific economics stay out of BayesFilter production.

### G4. Nonlinear Value Filters

SVD cubature, SVD-UKF, and SVD-CUT4 value filters must have:

- point-count and polynomial-degree metadata;
- support residuals;
- deterministic-completion residuals;
- floor and PSD-projection diagnostics;
- exact affine Gaussian collapse tests;
- nonlinear approximation diagnostics on Models B-C.

Current status:

- Model A exact Kalman parity is `certified`;
- Models B-C value filters are `certified` as implemented filters;
- dense one-step projection errors for Models B-C are `diagnostic`, not exact
  full nonlinear likelihood certification.

### G5. Nonlinear Analytic Scores

Smooth-branch first-order scores for SVD cubature, SVD-UKF, and SVD-CUT4 must
match centered finite differences of the same implemented value filters.

Current status:

- affine smooth fixture is `certified`;
- Model B score parity is `certified` on the selected smooth branch;
- Model C score parity is `certified` only on a nondegenerate phase-state
  testing variant;
- default Model C with zero phase variance is `blocked` by the active-floor
  gate in the current collapsed SVD score path.  The next remedy is not a
  separate numerical fixed-null theory; it is the Chapter 18b structural
  sigma-point score path, where sigma points are placed on the pre-transition
  uncertainty and deterministic coordinates are completed by the structural
  map.

### G6. Hessian Policy

Hessian work is not a default V1 requirement.  It can proceed only when one of
the following consumers is named:

- Newton or trust-region optimization;
- Laplace approximation;
- Riemannian or curvature-aware HMC;
- observed-information diagnostics.

Current status:

- linear QR Hessian is `certified` on local fixtures;
- nonlinear SVD Hessian is `deferred`;
- testing-only raw autodiff oracles remain outside production.

### G7. GPU/XLA Policy

GPU/XLA evidence must be opt-in and escalated.  Non-escalated GPU failures are
sandbox evidence only.

Current status:

- small GPU-visible and tiny XLA-visible artifacts are `diagnostic`;
- no broad GPU speedup claim is certified;
- nonlinear CUT4 GPU/XLA scaling is `deferred` until score branch boxes are
  stable.

### G8. HMC Readiness

HMC readiness is target-specific.  Filter existence or score existence is not
enough.

Minimum gate for an HMC target:

- finite value and score on the target parameter box;
- branch-frequency diagnostics;
- compiled parity if the target will run compiled;
- sampler smoke with explicit chain diagnostics;
- convergence diagnostics only when explicitly claimed.

Current status:

- first linear QR HMC target has `diagnostic` smoke evidence;
- nonlinear HMC is `blocked` until wider score branch diagnostics and default
  Model C structural sigma-point score decisions are made.

### G9. External Compatibility

MacroFinance and DSGE are external compatibility targets.  V1 should not make
either project a production dependency.

Current status:

- MacroFinance live checks are optional;
- DSGE inventory is read-only;
- SGU remains blocked as a production filtering target;
- actual client switch-over is `deferred` until V1 local gates pass.

## Current Remaining Gaps

R1. Chapter 18b structural sigma-point score implementation.

Hypothesis:
The correct way to score default Model C is to implement the Chapter 18b
structural sigma-point recursion: place sigma points on the pre-transition
uncertainty \((x_{t-1},\varepsilon_t)\), or an equivalent support-reduced
parameterization, then complete deterministic coordinates pointwise through the
structural map \(F_\theta\).  This should avoid treating the deterministic
phase coordinate as an independently perturbed full-state SVD direction.

Test:
derive the structural score equations from Chapter 18b, write the concrete
document block, implement the derivative path, audit code-document consistency,
and compare scores against finite differences of the implemented structural
law on default Model C.

Controlling subplan:

```text
docs/plans/bayesfilter-v1-structural-sigma-point-score-model-c-plan-2026-05-13.md
```

Immediate subtargets:

1. Use Chapter 18b's predictive-pushforward proposition and UKF moment
   equations as the mathematical source of truth.
2. Write the structural score equations for the same implemented likelihood:
   sigma points on \(A_t=(x_{t-1},\varepsilon_t)\), pointwise completion
   \(X_t^{(j)}=F_\theta(A_t^{(j)})\), observation points
   \(Z_t^{(j)}=h_\theta(X_t^{(j)})\), moment closure, innovation likelihood,
   and first-order score recursion.  This is now documented in
   `sec:bf-svd-sp-structural-fixed-support-score` of Chapter 18.
3. Implement the structural fixed-support score path: preserve the declared
   sigma-rule dimension, keep structurally null factor columns at zero with
   zero derivatives, and block moving-null or floor-regularized branches.
4. Test default Model C without a phase nugget against finite differences of
   the same structural sigma-point likelihood.

R2. Wider nonlinear score branch diagnostics.

Hypothesis:
Model B and smooth-phase Model C have practical parameter boxes where all
three nonlinear score backends remain finite with no active floors or weak
spectral gaps.

Test:
run CPU branch grids over selected boxes and report ok fraction, active-floor
count, weak-gap count, deterministic residuals, and score finiteness.

R3. Nonlinear HMC target selection.

Hypothesis:
Model B is the first viable nonlinear HMC candidate because it is smooth,
score-certified, and already uses the structural pre-transition uncertainty
path without the default Model C deterministic-phase obstruction.

Test:
write a target-specific readiness plan for Model B only after R2 passes.

R4. Nonlinear Hessian need assessment.

Hypothesis:
V1 can proceed with score-first nonlinear workflows and defer nonlinear
Hessians unless optimization or Laplace approximation becomes a concrete
consumer.

Test:
record whether any V1 integration path actually needs nonlinear Hessians.

R5. GPU/XLA point-axis scaling.

Hypothesis:
CUT4's larger point count becomes less costly under batched point-axis
vectorization on GPU/XLA for moderate shapes.

Test:
run escalated GPU-visible and XLA-visible benchmarks only after R2 defines
stable nonlinear score/value boxes.

R6. Exact nonlinear reference gap.

Hypothesis:
For short horizons, dense quadrature or high-particle seeded SMC can provide
stronger reference evidence for Models B-C without becoming production
dependencies.

Test:
add optional reference artifacts that clearly distinguish exact, dense
projection, and Monte Carlo evidence.

R7. External client integration.

Hypothesis:
MacroFinance/DSGE switch-over should wait until V1 has stable local API,
benchmark, branch, and optional live compatibility evidence.

Test:
write a separate future integration plan; do not modify external source in the
V1 stabilization lane.

## Execution Order

Future work should follow this order unless the user explicitly changes the
priority:

1. Chapter 18b structural sigma-point score derivation and implementation for
   deterministic-completion models, with default Model C as the first target.
2. Wider nonlinear score branch diagnostics for Model B and smooth/default
   Model C.
3. Nonlinear benchmark refresh with score-branch metadata.
4. Model B nonlinear HMC readiness plan and tiny smoke, CPU-only.
5. Optional nonlinear GPU/XLA scaling diagnostics.
6. Nonlinear Hessian need assessment.
7. Optional exact-reference strengthening for Models B-C.
8. Future MacroFinance/DSGE integration plan.

## Anti-drift Rules

1. Do not start GPU/XLA, HMC, Hessian, or external-client work before its
   predecessor gate passes.
2. Do not promote a diagnostic artifact to a certified claim.
3. Do not edit MacroFinance or DSGE from this lane.
4. Do not stage the shared monograph reset memo from this lane.
5. Do not add production NumPy dependencies.
6. Do not hide structural deterministic coordinates inside numerical
   regularization when the issue is a model-law or derivative-branch question.
7. Every future subplan must name the master-program gap it closes.

## Default Validation Commands

Fast local:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
```

Focused V1 regression:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  tests/test_nonlinear_benchmark_models_tf.py \
  tests/test_nonlinear_reference_oracles.py \
  tests/test_nonlinear_sigma_point_values_tf.py \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  tests/test_compiled_filter_parity_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Full default CPU:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q -p no:cacheprovider
```

Extended CPU branch diagnostics:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

## Master Program Decision Rule

BayesFilter V1 is ready for a client-integration planning phase when:

- the public API import gate passes;
- focused and full CPU suites pass;
- nonlinear value and score claims are documented at their true scope;
- Chapter 18b structural sigma-point score handling for deterministic
  completion is either implemented or explicitly deferred away from the first
  client target;
- GPU/XLA and HMC claims are either backed by target artifacts or labeled
  deferred;
- source-map and reset-memo provenance are current;
- no out-of-lane files are staged.
