# Plan: BayesFilter v1 External-compatibility Gap Closure

## Date

2026-05-10

## Purpose

This plan implements the pivot away from immediate MacroFinance switch-over.
BayesFilter should continue toward a stable v1 filtering API while treating
MacroFinance and the DSGE codebase as external compatibility targets.  The goal
is to gather strong evidence without forcing either client project to depend on
BayesFilter before v1 stabilizes.

## Governing Reset Memo

Use this lane-specific memo:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Do not use the shared monograph reset memo for this lane.  It currently
contains other agents' uncommitted MacroFinance/DSGE notes.

## Pivot Decision

Old emphasis:

- switch MacroFinance over through an optional client backend;
- then extend masked and derivative switch-over.

New emphasis:

- do not switch MacroFinance over before BayesFilter v1;
- certify compatibility from BayesFilter using local and optional external
  tests;
- keep MacroFinance and DSGE source trees independent;
- prepare a future v1 integration plan only after API, diagnostics, fixtures,
  and benchmarks are stable.

## Goals

1. Freeze a small public v1 filtering API surface for linear, structural, and
   nonlinear TensorFlow filters.
2. Preserve BayesFilter production independence:
   - TensorFlow/TensorFlow Probability only in production filtering code;
   - no MacroFinance or DSGE production imports;
   - no client-specific economics in BayesFilter production modules.
3. Convert MacroFinance work from "switch-over" to "external compatibility
   certification".
4. Build stable BayesFilter-local fixtures that do not require MacroFinance at
   CI time.
5. Keep live MacroFinance checks optional, local, and clearly labeled.
6. Identify DSGE non-SGU compatibility targets read-only before any client
   adapter work.
7. Add benchmark evidence before performance claims.
8. Gate GPU/XLA and HMC claims behind exact target-model diagnostics.

## Current Status

Pushed BayesFilter baseline:

```text
f11938c docs: record fetch conflict check
```

Local unpushed BayesFilter planning commit:

```text
a86a51f Plan filtering gap closure boundary
```

Existing BayesFilter capabilities:

- QR/square-root linear value filters for dense and masked observations;
- QR/square-root linear score/Hessian filters for dense and masked static
  derivative fixtures;
- SVD/eigen linear value filters for dense and masked observations;
- structural TensorFlow protocols;
- SVD cubature and SVD-UKF value filters;
- CUT4-G and SVD-CUT value filters;
- smooth-branch SVD-CUT score/Hessian for the implemented regularized law;
- CPU graph parity tests for promoted value and derivative paths;
- optional live MacroFinance compatibility tests when `/home/chakwong/MacroFinance`
  is present.

Out-of-lane files to leave untouched:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
docs/plans/dsge-sgu-marginal-utility-timing-implementation-request-2026-05-09.md
docs/plans/templates/*:Zone.Identifier
singularity_test.png
```

## Remaining Gaps Under The Pivot

### Gap A: v1 Public API Freeze Criteria

The package exports many useful symbols, but v1 needs a documented stable
surface and an unstable/internal surface.  Without this, external clients will
couple to whatever happens to be exported.

### Gap B: MacroFinance External Compatibility Matrix

BayesFilter has optional live MacroFinance compatibility tests, but there is
not yet a v1 compatibility matrix that states:

- which backends are certified;
- which tests are local CI versus optional external checks;
- which tolerances apply;
- which semantics are intentionally not certified yet.

### Gap C: Stable BayesFilter-local Compatibility Fixtures

Live MacroFinance tests are useful, but CI and v1 API work need stable local
fixtures that do not import MacroFinance.  These should cover dense QR, masked
QR, QR score/Hessian, SVD/eigen value, implemented-law diagnostics, static
shape behavior, and no-production-NumPy policy.

### Gap D: Optional Live External Test Policy

Optional tests should be explicitly labeled so a missing MacroFinance checkout
means "external compatibility not run", not "BayesFilter failure".

### Gap E: DSGE Read-only Compatibility Inventory

DSGE work should begin with read-only target inventory.  SGU remains blocked as
a production filtering target until a causal local target is proven.

### Gap F: CPU Benchmark Harness

Correctness tests exist, but benchmark artifacts for representative dimensions,
point counts, compile time, steady-state time, and memory are missing.

### Gap G: Escalated GPU/XLA-GPU Evidence

GPU evidence requires escalated probes on this machine.  No GPU claim is valid
from non-escalated sandbox behavior.

### Gap H: HMC Readiness Gate

HMC readiness requires exact target-model branch diagnostics.  Generic
smooth-branch SVD-CUT tests are not enough.

### Gap I: Linear SVD/eigen Derivative Need

SVD/eigen derivatives should remain deferred until external compatibility
evidence proves QR derivatives are insufficient for a real client target.

### Gap J: Future v1 Integration Plan

After v1 API and compatibility artifacts are stable, a separate plan should
define MacroFinance and DSGE adapter integration without surprising those
projects.

## Execution Plan

Each phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Continue automatically only when the primary criterion passes and no veto
diagnostic fires.  Stop before any phase that edits MacroFinance or DSGE source.

### Phase 1: Lane Isolation And Pivot Record

Actions:

- create or update the lane-specific reset memo;
- record that MacroFinance switch-over is deferred until BayesFilter v1;
- record out-of-lane files that must not be staged;
- register this plan and audit in `docs/source_map.yml`.

Primary criterion:

- a lane-specific reset memo exists and the shared monograph reset memo is not
  edited by this lane.

Veto diagnostics:

- shared reset memo changes are staged;
- MacroFinance or DSGE files are edited;
- unrelated untracked files are staged.

### Phase 2: v1 API Freeze Criteria

Actions:

- create a v1 API freeze note;
- list stable public symbols by group:
  - linear value;
  - linear QR derivatives;
  - SVD/eigen value;
  - structural TensorFlow protocols;
  - nonlinear sigma-point and CUT value;
  - smooth-branch SVD-CUT derivatives;
- list internal/testing-only symbols;
- define compatibility, diagnostics, dtype, and no-production-NumPy rules.

Primary criterion:

- a v1 API freeze artifact exists and maps stable symbols to current tests.

Veto diagnostics:

- freeze note exposes client-specific economics;
- freeze note promotes SVD/eigen derivatives;
- freeze note claims GPU or HMC readiness.

### Phase 3: MacroFinance External Compatibility Matrix

Actions:

- create a compatibility matrix artifact;
- classify checks as:
  - BayesFilter-local CI;
  - optional live MacroFinance;
  - deferred v1 integration;
- record tolerance, fixture, mask, jitter, derivative-order, and diagnostic
  requirements.

Primary criterion:

- the matrix clearly replaces switch-over language with external compatibility
  certification.

Veto diagnostics:

- matrix implies MacroFinance default changes;
- matrix requires MacroFinance source edits;
- matrix hides optional-test skips.

### Phase 4: Stable Local Fixture Gap Audit

Actions:

- audit existing BayesFilter tests for local fixture coverage;
- identify any missing local fixtures for:
  - dense QR value;
  - masked QR value;
  - QR score/Hessian;
  - masked QR score/Hessian;
  - SVD/eigen value diagnostics;
  - compiled static-shape behavior;
  - no-production-NumPy policy;
- decide whether new local fixtures are needed now.

Primary criterion:

- a fixture-gap result says either "local coverage sufficient for v1 planning"
  or names exact missing tests to implement.

Veto diagnostics:

- a required v1 local fixture depends on live MacroFinance imports;
- fixture gap cannot be closed without changing MacroFinance.

### Phase 5: Optional Live External Test Policy

Actions:

- document how optional external tests are run;
- document skip semantics when MacroFinance is absent;
- document required environment variables and path setup;
- keep optional tests outside production dependency paths.

Primary criterion:

- live external checks are clearly optional and reproducible.

Veto diagnostics:

- optional check becomes required CI;
- optional check modifies external projects;
- optional check imports MacroFinance in BayesFilter production code.

### Phase 6: DSGE Read-only Target Inventory Plan

Actions:

- create a read-only DSGE inventory plan for non-SGU compatibility targets;
- keep SGU production filtering blocked until causal locality is proven;
- do not edit `/home/chakwong/python`.

Primary criterion:

- target inventory plan exists with read-only scope and stop rules.

Veto diagnostics:

- plan asks BayesFilter to own DSGE economics;
- plan treats SGU as production-ready without locality evidence.

### Phase 7: CPU Benchmark Harness Plan

Actions:

- define representative benchmark shapes for:
  - linear QR value and QR derivatives;
  - linear SVD/eigen value;
  - SVD cubature/UKF;
  - SVD-CUT value;
  - SVD-CUT smooth-branch derivatives;
- define benchmark output schema.

Primary criterion:

- CPU benchmark plan exists with shape metadata and compile-time rules.

Veto diagnostics:

- plan hides compile time;
- plan omits point count;
- plan labels generic benchmarks as client readiness.

### Phase 8: Escalated GPU/XLA-GPU Gate Plan

Actions:

- define escalated device-probe commands;
- require exact CPU/GPU shape matching;
- record that non-escalated GPU failures are sandbox evidence only.

Primary criterion:

- GPU gate plan matches the repository AGENTS.md policy.

Veto diagnostics:

- plan asks for non-escalated GPU claims;
- plan changes shapes between CPU and GPU.

### Phase 9: HMC Readiness Gate Plan

Actions:

- define branch-frequency and derivative diagnostics for exact target models;
- require value, gradient, Hessian, compiled parity, and sampler smoke evidence;
- keep SVD-CUT HMC claims blocked until a target passes.

Primary criterion:

- HMC gate plan is target-specific and does not promote generic derivative
  tests to sampler readiness.

Veto diagnostics:

- plan claims HMC readiness from generic tests;
- plan permits hidden jitter/floor changes.

### Phase 10: Future v1 Integration Decision Plan

Actions:

- create a later integration decision checklist for MacroFinance and DSGE;
- require stable BayesFilter v1 API, compatibility matrix, local fixtures,
  external test pass, benchmarks, and rollback plan;
- keep actual adapter changes out of this lane.

Primary criterion:

- future integration checklist exists and clearly separates certification from
  switch-over.

Veto diagnostics:

- checklist reintroduces immediate client coupling;
- checklist requires production dependency cycles.

## Recommended Immediate Execution

In this pass, execute Phases 1--5 because they are BayesFilter-local planning
and certification artifacts.  Continue to Phases 6--10 if they remain
documentation-only and do not require client source edits, GPU execution, or HMC
runs.

Stop if any phase requires external repository writes, escalated GPU execution,
or changes to the shared monograph reset memo.
