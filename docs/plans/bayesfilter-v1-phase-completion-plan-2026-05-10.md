# Plan: Complete The BayesFilter v1 External-compatibility Phase

## Date

2026-05-10

## Purpose

This plan summarizes the goals, remaining gaps, and hypotheses for the current
BayesFilter v1 external-compatibility phase.  It then defines an execution path
to complete the phase without switching MacroFinance or DSGE over to
BayesFilter before v1.

The phase is not trying to make BayesFilter the production backend for
MacroFinance or DSGE yet.  It is trying to make BayesFilter v1 independently
testable, externally comparable, and honest about which claims are proven,
deferred, or blocked.

## Governing Lane

Use the lane-specific reset memo:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Do not use or stage the shared monograph reset memo for this lane.

## Executive Summary

The current phase should end with a clean BayesFilter-local v1 evidence bundle:

- a small, explicit public API import gate;
- a focused local regression gate for QR, QR derivatives, SVD values, and
  compiled parity;
- a benchmark harness with CPU smoke evidence and a clear next step for medium
  shapes and memory metadata;
- a read-only DSGE inventory that keeps SGU blocked and keeps Rotemberg/EZ as
  optional future fixtures;
- an external compatibility policy that keeps MacroFinance checks optional and
  does not make MacroFinance a BayesFilter dependency;
- documented blockers for GPU/XLA-GPU, HMC, and linear SVD/eigen derivatives.

The critical distinction is evidence discipline: local BayesFilter tests can
graduate into CI gates; live MacroFinance/DSGE tests are compatibility evidence;
GPU and HMC claims require separate target-specific artifacts.

## Phase Goals

1. Stabilize the BayesFilter v1 public filtering API surface.
2. Keep BayesFilter production code independent from MacroFinance and DSGE.
3. Treat MacroFinance as an external compatibility target, not a dependency or
   switch-over client.
4. Treat DSGE targets as external and read-only until a separate v1 integration
   decision.
5. Keep SGU blocked as production filtering until causal-locality gates pass.
6. Provide local CI fixtures for v1 behavior so BayesFilter can develop without
   requiring external checkouts.
7. Provide optional live external checks that are useful but not required.
8. Add benchmark artifacts before performance claims.
9. Keep GPU/XLA-GPU and HMC claims behind explicit target-specific gates.
10. Commit only lane-owned artifacts when this phase is actually executed.

## Current Evidence

Already created or in progress in this lane:

```text
docs/plans/bayesfilter-v1-api-freeze-criteria-2026-05-10.md
docs/plans/bayesfilter-v1-external-compatibility-matrix-2026-05-10.md
docs/plans/bayesfilter-v1-local-fixture-gap-audit-2026-05-10.md
docs/plans/bayesfilter-v1-optional-external-test-policy-2026-05-10.md
docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-plan-2026-05-10.md
docs/plans/bayesfilter-v1-dsge-readonly-target-inventory-result-2026-05-10.md
docs/plans/bayesfilter-v1-benchmark-and-gpu-gates-2026-05-10.md
docs/plans/bayesfilter-v1-hmc-and-integration-gates-2026-05-10.md
tests/test_v1_public_api.py
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.json
docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.md
```

Observed BayesFilter-local test evidence:

```text
31 passed, 2 warnings in 83.38s
```

Observed DSGE read-only external evidence:

```text
16 passed, 3 warnings in 14.08s
```

Observed CPU benchmark smoke:

- all benchmark rows completed with `status = ok`;
- benchmark artifact is CPU-only and smoke-scale;
- no GPU, HMC, MacroFinance switch-over, or DSGE switch-over claim is made.

## Remaining Gaps

### Gap 1: Commit And Sync Current Lane Artifacts

The current lane has uncommitted artifacts.  They should be committed without
staging the shared monograph reset memo or unrelated files.

Closure target:

- lane-owned artifacts are either committed together or explicitly left
  uncommitted with a handoff note;
- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md` remains outside
  the staged set.

### Gap 2: v1 API Import Gate Is New But Not Yet In Mainline

`tests/test_v1_public_api.py` exists locally and passes.  It should be
committed with the lane artifacts.

Closure target:

- the API import gate becomes part of the BayesFilter-local test suite;
- the gate checks client independence, not only successful imports.

### Gap 3: Benchmark Harness Is Smoke-scale Only

The current benchmark is useful as a harness proof, but not as performance
evidence for client-scale or GPU claims.

Closure target:

- smoke-scale benchmark remains labeled as a harness proof;
- no client-scale, GPU, or HMC claim uses the smoke artifact as proof.

### Gap 4: CPU Benchmark Lacks Memory Metadata

The benchmark records timing and shapes but not process memory or peak memory.
This should be added before any performance interpretation.

Closure target:

- benchmark rows record available memory metadata, or the artifact states why
  memory metadata was unavailable on this platform.

### Gap 5: Medium-shape Benchmark Artifact Missing

The current benchmark uses very small shapes.  A medium-shape artifact is
needed before v1 performance statements.

Closure target:

- add one medium-shape CPU artifact after memory metadata is added, or record a
  runtime veto with the attempted shape and failure mode.

### Gap 6: Optional Live MacroFinance Check Not Run In This Phase Completion

This is acceptable under the pivot, but the result should be explicitly labeled
as "not run" rather than hidden.

Closure target:

- optional live MacroFinance status is one of `passed`, `skipped`, `failed`, or
  `not run by policy`;
- no optional live result blocks BayesFilter-local CI unless the user starts a
  separate client-integration lane.

### Gap 7: DSGE Optional Bridges Not Implemented

Rotemberg and EZ are future optional live compatibility candidates.  No
BayesFilter adapter implementation is justified in this phase.

Closure target:

- SGU remains blocked for production filtering;
- Rotemberg/EZ remain future optional live fixtures, not BayesFilter-owned
  economics.

### Gap 8: GPU/XLA-GPU Evidence Missing

No escalated GPU probe or GPU benchmark artifact exists.  GPU claims remain
blocked.

Closure target:

- GPU/XLA-GPU claims stay blocked until an escalated `nvidia-smi`,
  TensorFlow-device probe, and matching benchmark artifact exist.

### Gap 9: HMC Readiness Missing

No exact target-model branch-frequency, derivative, compiled parity, or sampler
artifact exists.  HMC claims remain blocked.

Closure target:

- HMC claims stay blocked until a named target model has derivative diagnostics,
  branch-frequency diagnostics, compiled parity, and sampler evidence.

### Gap 10: Linear SVD/eigen Derivative Need Not Proven

No real client need requires linear SVD/eigen derivatives.  They remain
deferred.

Closure target:

- keep SVD/eigen value filters as value backends only;
- do not promote SVD/eigen analytic derivatives until a client need and
  spectral-gap policy justify the work.

## Gap-Hypothesis Matrix

| Gap | Hypothesis | Test or evidence | Closing condition |
| --- | --- | --- | --- |
| G1 | Lane artifacts can be committed without staging another agent's files. | `git status --short`, `git diff --cached --name-only`, `git diff --cached --check`. | Only `docs/plans/bayesfilter-v1-*`, v1 benchmark artifacts, `tests/test_v1_public_api.py`, and `bayesfilter_v1_` source-map entries are staged. |
| G2 | The public API is import-stable and client-independent. | `tests/test_v1_public_api.py`. | Top-level public symbols import and `bayesfilter` does not import MacroFinance/DSGE modules. |
| G3 | Local v1 compatibility is covered without external checkouts. | Focused local pytest subset for QR, QR derivatives, SVD value, and compiled parity. | All focused tests pass under deliberate CPU-only settings. |
| G4 | Benchmark rows can carry enough metadata for interpretation. | Benchmark harness row schema inspection plus smoke run. | Rows include device scope, shapes, point counts, first-call time, steady time, and memory metadata or an explicit memory caveat. |
| G5 | Medium CPU shapes remain feasible for v1 benchmark candidates. | Medium-shape CPU benchmark. | Rows complete or fail explicitly with backend, shape, and failure reason. |
| G6 | Live MacroFinance checks can remain optional without hiding evidence. | Optional live policy plus optional test result if run. | Status is explicit and does not become a required BayesFilter CI dependency. |
| G7 | DSGE candidates can remain read-only external targets. | Read-only inventory result. | SGU remains blocked; Rotemberg/EZ remain future optional fixtures. |
| G8 | GPU/XLA-GPU claims require separate escalated evidence. | Escalated GPU probe and GPU benchmark, not part of this CPU pass. | Until those artifacts exist, GPU claims remain blocked. |
| G9 | HMC readiness requires target-model evidence, not filter existence. | Future branch-frequency, derivative, compiled parity, and sampler diagnostics. | Until those artifacts exist, HMC claims remain blocked. |
| G10 | Linear SVD/eigen analytic derivatives are not needed for v1. | Client-need review. | Defer unless a concrete client path cannot use QR derivatives or autodiff-safe alternatives. |

## Hypotheses To Test

H1. The declared v1 public API surface is import-stable from top-level
`bayesfilter` and does not import external client packages.

Test:

```text
tests/test_v1_public_api.py
```

Pass criterion:

- all declared v1 public symbols are top-level importable;
- importing `bayesfilter` does not import MacroFinance or DSGE modules.

H2. Existing BayesFilter-local linear QR, QR derivative, SVD value, and CPU
compiled parity fixtures are sufficient for local v1 compatibility.

Test:

```text
tests/test_linear_kalman_qr_tf.py
tests/test_linear_kalman_qr_derivatives_tf.py
tests/test_linear_kalman_svd_tf.py
tests/test_compiled_filter_parity_tf.py
```

Pass criterion:

- focused local subset passes under deliberate CPU-only settings.

H3. The benchmark harness can time all v1 candidate backend groups at fixed
small shapes while recording first-call timing, steady timing, dimensions, and
point counts.

Test:

```text
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
```

Pass criterion:

- all benchmark rows have `status = ok`;
- JSON and Markdown artifacts state CPU-only scope;
- no GPU or client-readiness claim is made.

H4. Medium-shape benchmarks remain feasible and reveal whether graph mode
continues to be practical for v1 candidates.

Test:

- run a medium-shape benchmark artifact after adding memory metadata.

Pass criterion:

- all planned rows complete or failures are reported explicitly;
- compile time, steady time, dimensions, point counts, and memory metadata are
  recorded.

H5. Optional live MacroFinance checks can remain separate from CI and still
provide useful compatibility evidence.

Test:

```text
tests/test_macrofinance_linear_compat_tf.py
```

Pass criterion:

- optional live check passes when MacroFinance is present, or skips clearly
  when absent;
- failure is reported as external compatibility evidence, not a BayesFilter CI
  blocker.

H6. DSGE Rotemberg and EZ can remain future optional live fixtures without
BayesFilter owning DSGE economics.

Test:

- no implementation test in this phase;
- read-only inventory result remains the artifact.

Pass criterion:

- SGU remains blocked;
- Rotemberg/EZ are labeled future optional live candidates only.

H7. The phase can complete without changing MacroFinance, DSGE, shared
monograph files, or structural-SVD chapter files.

Test:

```text
git status --short
```

Pass criterion:

- staged files stay within this lane's ownership boundary;
- out-of-lane modified or untracked files remain untouched.

## Completion Plan

Each implementation phase should follow:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

### Phase A: Lane Boundary And Artifact Audit

Actions:

- run `git status --short --branch`;
- list the files that are in lane and out of lane;
- update this plan and the lane reset memo if the boundary has changed;
- do not stage or edit the shared monograph reset memo.

Primary criterion:

- the working tree boundary is understood before any execution.

Veto diagnostics:

- shared monograph reset memo would need editing;
- MacroFinance or DSGE source would need editing;
- source-map entries for new v1 artifacts are missing.

### Phase B: Public API Import Gate

Actions:

- keep `tests/test_v1_public_api.py`;
- ensure symbol list matches the API freeze criteria;
- run the test.

Primary criterion:

- `tests/test_v1_public_api.py` passes.

Veto diagnostics:

- top-level import pulls MacroFinance or DSGE modules;
- symbol list includes unstable/testing-only helpers;
- symbol list promotes blocked SVD/eigen derivatives.

### Phase C: Local Compatibility Regression Gate

Actions:

- run the focused local v1 subset:
  - v1 API import;
  - QR value;
  - QR score/Hessian;
  - SVD/eigen value;
  - CPU graph parity.

Primary criterion:

- focused local subset passes under CPU-only settings.

Veto diagnostics:

- GPU is accidentally used;
- local subset depends on MacroFinance or DSGE checkouts.

### Phase D: Benchmark Harness Hardening

Actions:

- add memory metadata to the benchmark harness if practical;
- rerun the existing small-shape artifact;
- add one medium-shape artifact if runtime remains reasonable.

Primary criterion:

- benchmark artifacts include timing, shape, point-count, and memory metadata,
  or explicitly document why memory metadata is unavailable.

Veto diagnostics:

- benchmark hides compile time;
- benchmark changes CPU/GPU or shapes across rows without labels;
- benchmark result is described as client-scale evidence.

### Phase E: Optional External Check Decision

Actions:

- decide whether to run optional live MacroFinance tests now or defer;
- if run, record MacroFinance path and result separately;
- if deferred, record that no live MacroFinance certification was run in this
  completion pass.

Primary criterion:

- optional live external status is explicit.

Veto diagnostics:

- optional external check is treated as required CI;
- external project must be edited for the check to pass.

### Phase F: DSGE Candidate Containment

Actions:

- keep the DSGE read-only inventory result as the current evidence;
- do not implement Rotemberg/EZ bridges yet;
- keep SGU blocked.

Primary criterion:

- compatibility matrix and reset memo keep DSGE candidates in optional
  live/read-only status.

Veto diagnostics:

- SGU promoted to production filtering;
- BayesFilter starts owning DSGE economics.

### Phase G: GPU/HMC/SVD-derivative Blocker Review

Actions:

- confirm GPU/XLA-GPU claims remain blocked pending escalated probes;
- confirm HMC readiness remains blocked pending target-specific evidence;
- confirm linear SVD/eigen derivatives remain deferred pending client need.

Primary criterion:

- all blocked claims remain blocked with explicit next evidence required.

Veto diagnostics:

- performance, GPU, HMC, or SVD-derivative claims exceed evidence.

### Phase H: Commit And Handoff

Actions:

- stage only lane-owned files:
  - `docs/plans/bayesfilter-v1-*.md`;
  - `docs/benchmarks/benchmark_bayesfilter_v1_filters.py`;
  - `docs/benchmarks/bayesfilter-v1-filter-benchmark-2026-05-10.*`;
  - `tests/test_v1_public_api.py`;
  - `docs/source_map.yml` entries whose keys begin with `bayesfilter_v1_`;
- run `git diff --cached --check`;
- commit with a v1 external-compatibility message;
- leave out-of-lane files unstaged.

Primary criterion:

- the executed phase has a clean commit and a reset-memo handoff.

Veto diagnostics:

- shared monograph reset memo is staged;
- `Zone.Identifier` sidecars, images, DSGE request notes, or client-repo files
  are staged;
- tests or benchmark artifacts are missing from the handoff.

## Recommended Immediate Path

1. Finish Phase A by confirming the current lane boundary and registering this
   phase-completion plan in `docs/source_map.yml`.
2. Run Phases B and C as focused CPU-only tests.
3. Run Phase D after adding memory metadata; run the medium benchmark only if
   the smoke harness remains stable.
4. Record Phase E as deferred unless the user explicitly asks to run optional
   live MacroFinance checks.
5. Keep Phases F and G as documentation/audit containment.
6. Commit in Phase H only after the phase is executed, and stage only lane-owned
   files.

## Phase Completion Definition

This phase is complete when:

- v1 lane artifacts are committed;
- public API import gate passes;
- local compatibility regression subset passes;
- benchmark harness has at least smoke-scale artifact and an explicit next step
  for medium/memory metadata;
- optional external check status is explicit;
- DSGE candidates remain contained;
- GPU/HMC/SVD-derivative blockers remain explicit;
- lane-owned artifacts are committed or deliberately handed off as uncommitted
  work with exact file lists.
