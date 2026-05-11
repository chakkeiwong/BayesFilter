# BayesFilter V1 SVD Filter Analytic Gradient Audit And Implementation Plan

## Date

2026-05-12

## Parent Plan

```text
docs/plans/bayesfilter-v1-nonlinear-filtering-master-testing-program-2026-05-11.md
```

## Motivation

Chapter 18 derives analytic SVD sigma-point and SVD-CUT score/Hessian
equations, but the current TensorFlow implementation still exposes a
`GradientTape` smooth-branch SVD-CUT4 score/Hessian path:

```text
bayesfilter/nonlinear/svd_cut_derivatives_tf.py
bayesfilter/nonlinear/__init__.py
```

That path is useful as a testing oracle, but it should not be the public
production derivative implementation for SVD-CUT4.  This subplan makes the
gradient path precise: audit the Chapter 18 equations, map them to tensor
operations, implement score-first analytic gradients for SVD cubature,
SVD-UKF, and SVD-CUT4, and test them against finite differences and the
autodiff oracle on smooth separated-spectrum branches.

## Key Decision

Implement analytic scores first.  Do not implement the analytic Hessian until
the score path passes model-suite, factor-reconstruction, finite-difference,
and branch diagnostics and a concrete Hessian consumer is named.

For HMC, score is required.  Hessian is optional for Newton/Laplace workflows,
curvature diagnostics, Riemannian samplers, or release certification.

## Lane

Allowed write lane:

```text
bayesfilter/nonlinear/*
bayesfilter/testing/*
tests/test_*svd*
tests/test_*sigma*
tests/test_nonlinear_*score*
docs/chapters/ch18_svd_sigma_point.tex
docs/plans/bayesfilter-v1-*
docs/source_map.yml
```

Protected unless explicitly requested:

```text
docs/chapters/ch18b_structural_deterministic_dynamics.tex
docs/plans/bayesfilter-structural-*
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

## Derivative Contract

The analytic gradient implementation must state and test the following target:
- the score is the derivative of the implemented sigma-point likelihood;
- SVD/eigen factors reconstruct the implemented covariance after any explicit
  flooring or PSD projection, not necessarily the pre-regularized covariance;
- smooth-branch tests require fixed rank, no active hard floors, separated
  eigenvalues, and fixed ordering/sign convention;
- likelihood solves require positive-definite innovation covariance after the
  documented regularization policy.

The production score path must return diagnostics that separate:
- model covariance;
- implemented covariance;
- active floor counts;
- spectral gaps;
- derivative branch status;
- residuals for factor reconstruction and deterministic identities.

## Phase Plan

### Phase G0: Lane Recovery And Baseline Tests

Actions:
- run `git status --short --branch`;
- verify no required edits touch another agent's structural or monograph files;
- run the current nonlinear derivative and value baseline on CPU.

Baseline command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Primary criterion:
- the existing value and oracle tests pass or failures are recorded before
  implementation starts.

Veto diagnostics:
- another agent's changes are required to proceed;
- current tests fail for reasons unrelated to this lane and cannot be
  isolated.

### Phase G1: Derivation-To-Code Audit

Actions:
- create a traceability table from Chapter 18 labels to tensor operations:
  `eq:bf-svd-sp-point-first`,
  `eq:bf-svd-sp-point-second`,
  `eq:bf-svd-sp-map-first`,
  `eq:bf-svd-sp-cov-first`,
  `eq:bf-svd-sp-score`,
  `eq:bf-svd-factor-reconstruction-first`,
  `eq:bf-svd-cut-point-first`,
  `eq:bf-svd-cut-moment-first`,
  `eq:bf-svd-cut-score`;
- use MathDevMCP as a source lookup/audit helper, but treat current automated
  status as non-certifying unless assumptions are explicitly discharged;
- split score-only tensors from Hessian tensors;
- state shape conventions for parameter dimension, state dimension,
  stochastic integration dimension, point count, and observation dimension.

Primary criterion:
- every score tensor has a source equation, shape, code target, and oracle.

Veto diagnostics:
- a required equation lacks clear dimensions;
- the implementation target cannot distinguish model law from regularized law.

### Phase G2: Move Raw Tape Derivatives To Testing Oracle

Actions:
- move or duplicate the current `tf_svd_cut4_score_hessian` raw tape logic into
  a testing-only module, for example
  `bayesfilter/testing/tf_svd_cut_autodiff_oracle.py`;
- remove public production export of raw tape SVD-CUT4 score/Hessian unless an
  analytic replacement is ready in the same pass;
- keep the existing branch-gated tests by pointing them at the testing oracle.

Primary criterion:
- public production nonlinear exports no longer advertise raw
  `GradientTape` SVD-CUT4 derivatives as analytic production derivatives.

Veto diagnostics:
- public API removal breaks unrelated value filters;
- the oracle loses branch diagnostics needed for finite-difference comparison.

### Phase G3: Shared Analytic Score Core

Actions:
- implement first-order derivative helpers for:
  covariance factorization on the smooth SVD/eigen branch;
  sigma-point placement;
  transition/observation propagation;
  weighted mean and covariance;
  innovation covariance, cross covariance, Kalman gain, and Gaussian
  likelihood contribution;
- write the core so SVD cubature, SVD-UKF, and SVD-CUT4 share the same score
  machinery with different fixed standardized rules.

Primary criterion:
- analytic score exists for at least one fixed-rule backend and its tensors
  match finite differences on Model A.

Veto diagnostics:
- active floor or weak spectral gap occurs at the default score-test point;
- score implementation duplicates backend-specific code in a way that prevents
  SVD-CUT4 reuse.

### Phase G4: Backend Integration

Actions:
- expose score result functions with explicit names, for example:
  `tf_svd_cubature_score`,
  `tf_svd_ukf_score`,
  `tf_svd_cut4_score`;
- return `TFFilterDerivativeResult` or a score-only derivative result with
  metadata stating `analytic_score_smooth_branch`;
- keep Hessian fields absent or explicitly `None` if the result type is
  separated.

Primary criterion:
- all three backends can evaluate value and analytic score on Models A-C.

Veto diagnostics:
- public API shape is ambiguous about Hessian availability;
- the SVD-CUT4 path still relies on raw tape for production score.

### Phase G5: Score Test Ladder

Target tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_nonlinear_sigma_point_scores_tf.py \
  tests/test_svd_cut_derivatives_tf.py \
  tests/test_nonlinear_sigma_point_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Checks:
- finite-difference score parity on Models A-C;
- testing-oracle parity on smooth separated-spectrum branches;
- eager/graph parity;
- branch diagnostics block active floors and weak spectral gaps;
- no production NumPy imports in `bayesfilter/nonlinear/*`.

Primary criterion:
- score residuals pass documented tolerances for SVD cubature, SVD-UKF, and
  SVD-CUT4 on the accepted smooth test cases.

Veto diagnostics:
- finite-difference residuals are unstable under reasonable step changes;
- branch diagnostics report frequent weak gaps in the default model suite;
- graph mode changes score values.

### Phase G6: Hessian Gate

Actions:
- write a result artifact deciding whether to implement Hessian now or defer;
- if deferred, state the named evidence required to reopen it;
- if implemented, create a separate Hessian plan with memory, shape, and
  curvature tests before code changes.

Primary criterion:
- Hessian status is explicit and does not block HMC score work.

Veto diagnostics:
- a downstream plan assumes Hessian exists without a passing implementation;
- score tests are not stable enough to justify second derivatives.

### Phase G7: Provenance And Reset-Memo Update

Actions:
- update the v1 reset memo after each executed phase;
- register implementation and test artifacts in `docs/source_map.yml`;
- keep out-of-lane dirty files unstaged.

Primary criterion:
- the next agent can tell which derivatives are analytic, which are oracle
  autodiff, and which are deferred.

## Hypotheses To Test

H-G1:
The Chapter 18 score equations can be mapped to a shared fixed-rule derivative
core for SVD cubature, SVD-UKF, and SVD-CUT4.

H-G2:
Autodiff can remain as a testing oracle without being exported as the
production derivative backend.

H-G3:
Analytic score tests on Models A-C are enough to justify score use in later HMC
readiness diagnostics.

H-G4:
Hessian implementation is not justified until score residuals, spectral branch
stability, and memory cost are measured.

## Done Definition

This subplan is complete when:
- the Chapter 18 score equations have an implementation traceability table;
- raw tape SVD-CUT4 score/Hessian is testing-only or clearly labeled as oracle;
- analytic score paths exist for SVD cubature, SVD-UKF, and SVD-CUT4;
- finite-difference, oracle, branch, and graph-parity score tests pass;
- Hessian is either explicitly deferred or has its own gated plan.
