# BayesFilter V1 Nonlinear Filtering Remaining Master Program Plan

## Date

2026-05-12

## Parent Plan

```text
docs/plans/bayesfilter-v1-nonlinear-filtering-master-testing-program-2026-05-11.md
```

## Motivation

Two streams have dedicated subplans:
- nonlinear benchmark documentation and reusable testing tools;
- analytic gradient audit and score-first implementation.

This plan covers the remaining parent-program work after those streams are
underway: value-filter consolidation, branch diagnostics, approximation
benchmarks, CI/runtime tiers, optional GPU/XLA gates, HMC readiness, and
provenance cleanup.  It keeps BayesFilter V1 decoupled from MacroFinance and
DSGE until the local package has enough evidence to merge safely.

## Lane

Allowed write lane:

```text
bayesfilter/nonlinear/*
bayesfilter/testing/*
tests/test_nonlinear_*
tests/test_*svd*
tests/test_*sigma*
docs/benchmarks/*
docs/chapters/ch16_sigma_point_filters.tex
docs/chapters/ch17_square_root_sigma_point.tex
docs/chapters/ch18_svd_sigma_point.tex
docs/chapters/ch28_nonlinear_ssm_validation.tex
docs/plans/bayesfilter-v1-*
docs/source_map.yml
pytest.ini
```

Protected unless explicitly requested:

```text
docs/chapters/ch18b_structural_deterministic_dynamics.tex
docs/plans/bayesfilter-structural-*
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

## Current Blocking Dependencies

Dependency D1:
Models A-C documented and implemented as reusable fixtures.

Dependency D2:
Raw tape SVD-CUT4 derivative path moved out of production or clearly blocked
from analytic promotion.

Dependency D3:
Analytic score implementation status decided for SVD cubature, SVD-UKF, and
SVD-CUT4.

Dependency D4:
Hessian status explicitly deferred or assigned to a separate gated plan.

## Phase Plan

### Phase R0: Baseline And Lane Audit

Actions:
- run `git status --short --branch`;
- confirm the active branch is in the v1 lane;
- run the focused nonlinear value and branch baseline.

Baseline command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_structural_svd_sigma_point_tf.py \
  tests/test_svd_cut_filter_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  tests/test_sigma_points_tf.py \
  tests/test_cut_rule_tf.py \
  -p no:cacheprovider
```

Primary criterion:
- baseline tests pass or failures are documented with ownership.

Veto diagnostics:
- work requires modifying protected files;
- out-of-lane dirty files must be staged to proceed.

### Phase R1: Value Filter Consolidation

Actions:
- review SVD cubature, SVD-UKF, and SVD-CUT4 value APIs for consistent
  observations, masks, covariance regularization, diagnostics, and metadata;
- ensure all value backends accept the model-suite fixtures from the model
  subplan;
- add tests for deterministic support residuals and implemented covariance
  reporting.

Primary criterion:
- value APIs are consistent enough for shared score tests and benchmarks.

Veto diagnostics:
- backend-specific metadata prevents comparing the three filters;
- regularization diagnostics do not reveal the implemented covariance.

### Phase R2: Branch Diagnostics Expansion

Actions:
- extend branch diagnostics from SVD-CUT4 to the shared SVD sigma-point
  placement path;
- record active floors, minimum spectral gaps, rank, support residuals, and
  deterministic residuals;
- add parameter-box sweeps for Models A-C.

Primary criterion:
- branch diagnostic tests can classify smooth, active-floor, weak-gap, and
  nonfinite failures before derivative/HMC claims.

Veto diagnostics:
- default model-suite regions show frequent weak spectral gaps;
- diagnostics conflate model law with numerical regularization.

### Phase R3: Approximation Effectiveness Benchmarks

Actions:
- create or extend benchmark harnesses under `docs/benchmarks`;
- compare cubature, UKF, and CUT4 on Models A-C against exact Kalman or dense
  quadrature references;
- record log-likelihood error, filtered mean error, covariance calibration,
  runtime, point count, and branch-blocker rate.

CPU command pattern:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_nonlinear_filters.py
```

Primary criterion:
- benchmark artifacts identify where CUT4 buys accuracy and where point count
  or branch fragility costs dominate.

Veto diagnostics:
- benchmarks lack fixed seeds and model parameter records;
- artifacts imply general performance claims from tiny examples.

### Phase R4: CI Runtime Tiering

Actions:
- place constructor and rule-moment tests in the fast tier;
- keep finite-difference score tests small and CPU-only by default;
- mark branch sweeps, approximation grids, GPU/XLA, and HMC as opt-in;
- update `pytest.ini` only if marker policy is missing or stale.

Primary criterion:
- routine test runs remain small while extended evidence remains reproducible.

Veto diagnostics:
- default tests require GPU or long HMC runs;
- optional markers are not documented.

### Phase R5: Optional GPU/XLA Gate

Actions:
- run only after CPU value and score tests pass;
- use escalated permissions for GPU/CUDA detection and execution under the
  local AGENTS policy;
- test fixed-shape point-axis vectorization and CPU/GPU matching-shape parity.

Example command, requiring escalation:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Primary criterion:
- GPU/XLA artifacts state device visibility, shapes, parity, runtime, and
  whether XLA is actually used.

Veto diagnostics:
- non-escalated CUDA failure is treated as environment evidence;
- GPU artifacts make broad speedup claims from one tiny shape.

### Phase R6: HMC Readiness Gate

Actions:
- start only after analytic score or approved custom-gradient score exists for
  the target backend;
- choose one target model and parameter box;
- run finite value/score scans, branch diagnostics, and short multi-chain
  target-specific HMC smoke tests;
- do not claim convergence without explicit R-hat, ESS, divergence, and
  posterior-recovery thresholds.

Primary criterion:
- HMC status is either target-specific ready for deeper diagnostics or blocked
  with named evidence.

Veto diagnostics:
- score path uses testing-only raw tape without an approved custom-gradient
  contract;
- branch diagnostics are missing at proposal-region scale;
- HMC output is described as convergence from smoke evidence alone.

### Phase R7: Documentation And Provenance Cleanup

Actions:
- update Chapter 18 to link derivative-validation equations to the model-suite
  rows only after tests exist;
- update Chapter 28 with benchmark status and limitations;
- register benchmark, test, and result artifacts in `docs/source_map.yml`;
- update the v1 reset memo after each execution phase.

Primary criterion:
- the documentation distinguishes value correctness, analytic score
  correctness, approximation quality, GPU/XLA evidence, and HMC readiness.

Veto diagnostics:
- documentation implies MacroFinance/DSGE switch-over;
- source map lacks entries for new benchmark/result artifacts.

## Hypotheses To Test

H-R1:
The three nonlinear value backends can share one diagnostic vocabulary without
weakening backend-specific information.

H-R2:
Branch diagnostics predict where analytic scores and HMC proposal regions are
unsafe.

H-R3:
CUT4 improves nonlinear moment accuracy on selected models enough to justify
its larger point count in CPU and optional GPU/XLA benchmarks.

H-R4:
GPU/XLA is useful only for fixed-shape, point-axis-heavy workloads after CPU
parity has already passed.

H-R5:
HMC readiness should remain target-specific and blocked until score and branch
diagnostics pass on the target parameter box.

## Done Definition

The remaining master-program work is complete when:
- value APIs and diagnostics are consistent across SVD cubature, SVD-UKF, and
  SVD-CUT4;
- branch diagnostics cover the shared SVD placement path;
- CPU benchmark artifacts compare approximation quality and runtime on the
  documented model suite;
- CI tiers keep fast tests small and extended diagnostics opt-in;
- optional GPU/XLA evidence is recorded only with escalated device visibility;
- HMC readiness is either target-specific and evidence-backed or explicitly
  blocked;
- source-map and reset-memo entries are current.
