# Plan: Complete BayesFilter v1 External-compatibility Phase

## Date

2026-05-11

## Governing Lane

This is the active phase-completion plan for this agent's lane:

```text
BayesFilter v1 external-compatibility lane
```

Use this reset memo:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

This plan supersedes the May 10 draft:

```text
docs/plans/bayesfilter-v1-phase-completion-plan-2026-05-10.md
```

It must not use or stage the shared monograph reset memo, structural SVD reset
memo, SGU plan files, Chapter 18/18b files, MacroFinance source files, or DSGE
source files unless the user explicitly opens a separate lane.

## Phase Goal

Complete the BayesFilter-local v1 external-compatibility phase.  "Complete"
means BayesFilter has a coherent evidence bundle for the v1 API, local
filtering regressions, benchmark harness, external-compatibility policy, DSGE
target containment, and blocked future claims.

This phase does not switch MacroFinance or DSGE over to BayesFilter.  It makes
BayesFilter ready to be compared against those clients without coupling them
to BayesFilter before v1.

## Goals To Achieve

1. Keep this lane isolated from the other active agent's structural SVD/SGU
   work.
2. Stabilize the declared v1 top-level filtering API and test that importing
   `bayesfilter` does not import MacroFinance or DSGE packages.
3. Preserve BayesFilter-local regression evidence for QR value, QR
   score/Hessian, SVD/eigen value, and CPU graph parity.
4. Harden the CPU benchmark harness so timing artifacts include dimensions,
   point counts, first-call timing, steady-call timing, device scope, and memory
   metadata where available.
5. Add at least one medium-shape CPU benchmark artifact, or record a clear
   runtime veto with the exact attempted shape and backend.
6. Make optional live MacroFinance status explicit: passed, skipped, failed, or
   not run by policy.
7. Keep DSGE targets read-only: SGU remains blocked, Rotemberg and EZ remain
   future optional live fixtures, and BayesFilter does not own DSGE economics.
8. Keep GPU/XLA-GPU, HMC readiness, and linear SVD/eigen derivative claims
   blocked until separate evidence exists.
9. Commit only lane-owned artifacts or leave an exact uncommitted handoff list.

## Current Evidence

Completed or present in this lane:

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

Observed evidence already recorded in the reset memo:

```text
v1 API import gate: 2 passed, 2 warnings
BayesFilter-local compatibility subset: 31 passed, 2 warnings
DSGE read-only inventory subset: 16 passed, 3 warnings
CPU benchmark smoke: all rows status = ok
```

## Remaining Gaps

### Gap 1: Uncommitted Lane Artifacts

The v1 lane artifacts are still local.  The shared monograph reset memo and
unrelated sidecars are also dirty or untracked, so staging must be selective.

Closure target:

- only lane-owned files are staged;
- shared monograph reset memo, structural SVD/SGU files, sidecars, images, and
  client-repo files remain unstaged.

### Gap 2: Current-date Active Handoff

The May 10 plan exists, but the active user request is on 2026-05-11.

Closure target:

- this May 11 plan is registered in `docs/source_map.yml`;
- the lane reset memo names this May 11 plan as the active plan;
- the May 11 audit reviews this May 11 plan.

### Gap 3: Memory Metadata Missing From Benchmark Rows

The benchmark artifact records timing, dimensions, and point counts, but not
memory metadata.

Closure target:

- benchmark rows include available memory metadata, preferably process
  `ru_maxrss` on Linux;
- if memory metadata cannot be collected, the JSON and Markdown artifacts state
  the caveat explicitly.

### Gap 4: Medium-shape CPU Benchmark Missing

The current benchmark is a small smoke artifact.  It proves the harness works,
not client-scale performance.

Closure target:

- add one medium-shape CPU artifact, or record a veto with backend, shape,
  runtime, and failure reason.

### Gap 5: Optional Live MacroFinance Status Is Not Final

The pivot allows optional live MacroFinance checks to be deferred, but the
completion record should state the status explicitly.

Closure target:

- status is `not run by policy`, `skipped`, `passed`, or `failed`;
- no MacroFinance source file is edited;
- no optional live result becomes required BayesFilter CI.

### Gap 6: DSGE Optional Bridges Are Not Implemented

Rotemberg and EZ are candidates for future optional live fixtures.  SGU remains
blocked.

Closure target:

- no BayesFilter production DSGE adapter is added in this phase;
- SGU production filtering remains blocked;
- Rotemberg/EZ remain test-only future candidates.

### Gap 7: GPU/XLA-GPU Evidence Missing

No escalated GPU probe or matching GPU benchmark exists.

Closure target:

- GPU claims remain blocked;
- future GPU phase requires escalated `nvidia-smi`, escalated TensorFlow device
  probe, and CPU/GPU shape matching.

### Gap 8: HMC Readiness Evidence Missing

No target-model sampler evidence exists.

Closure target:

- HMC readiness remains blocked until value, score, Hessian, branch-frequency,
  compiled parity, and sampler smoke evidence exist for a named target.

### Gap 9: Linear SVD/eigen Derivative Need Not Proven

SVD/eigen value filters exist, but derivative promotion is not justified.

Closure target:

- linear SVD/eigen derivatives remain deferred until a real client need and
  spectral-gap policy are documented.

### Gap 10: Final Commit Or Handoff Not Done

The phase is not complete until the lane artifacts are committed or the
uncommitted file list is intentionally handed off.

Closure target:

- commit lane-owned artifacts after tests pass, or record an exact handoff list.

## Hypotheses For The Gaps

| ID | Hypothesis | Gap | Test or evidence | Closing decision |
| --- | --- | --- | --- | --- |
| H1 | Lane artifacts can be isolated from the other agent's files. | G1 | `git status --short`, staged-file audit | Proceed only if staged files are lane-owned. |
| H2 | The May 11 plan can be the active handoff without rewriting structural/SGU plans. | G2 | source-map entry and reset-memo update | Active plan path is recorded under `bayesfilter_v1_`. |
| H3 | Top-level v1 API imports are stable and client-independent. | G1, G2 | `tests/test_v1_public_api.py` | Pass means API surface imports and no MacroFinance/DSGE import. |
| H4 | Local v1 regression tests are sufficient for BayesFilter CI evidence. | G1 | focused local pytest subset | Pass means local compatibility remains green without external checkouts. |
| H5 | Memory metadata can be added without new dependencies. | G3 | benchmark JSON schema and smoke run | Rows record `ru_maxrss` or an explicit unavailable marker. |
| H6 | Medium-shape CPU benchmarks are feasible for the current harness. | G4 | medium benchmark command | Rows finish, or veto is explicit and reproducible. |
| H7 | Optional live MacroFinance checks can stay optional without hiding evidence. | G5 | optional policy plus status note | Status is explicit and no MacroFinance source is edited. |
| H8 | DSGE candidates can remain read-only external targets. | G6 | compatibility matrix and inventory result | SGU blocked, Rotemberg/EZ test-only future candidates. |
| H9 | GPU/HMC claims are unsupported until separate evidence exists. | G7, G8 | blocker review | Claims remain blocked with exact next evidence. |
| H10 | SVD/eigen derivatives are unnecessary for v1 unless a client need appears. | G9 | client-need review | Keep value-only status unless a real target proves QR insufficient. |

## Execution Plan

Every phase uses:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

### Phase A: Lane Boundary Audit

Actions:

- run `git status --short`;
- classify files as lane-owned, out-of-lane dirty, or unrelated untracked;
- confirm no structural SVD/SGU or chapter files are required.

Tests:

```bash
git status --short
git diff --check
```

Pass criterion:

- lane-owned files are clearly identified;
- no out-of-lane file is needed to complete this phase.

Stop if:

- the shared monograph reset memo or structural SVD/SGU files must be edited;
- client repositories must be edited.

### Phase B: Plan And Source-map Finalization

Actions:

- register this plan in `docs/source_map.yml`;
- update only the v1 lane reset memo with the plan path and interpretation;
- align the May 11 audit with this active plan.

Tests:

```bash
rg -n "bayesfilter_v1_phase_completion_plan_2026_05_11" docs/source_map.yml
rg -n "bayesfilter-v1-phase-completion-plan-2026-05-11" docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Pass criterion:

- source map, audit, and reset memo identify this as the active plan.

### Phase C: API And Local Regression Gate

Actions:

- run the v1 API import test;
- run the focused local v1 regression subset.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_tf.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_linear_kalman_svd_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

Pass criterion:

- focused subset passes under deliberate CPU-only settings.

Stop if:

- importing `bayesfilter` imports MacroFinance or DSGE modules;
- local tests require external checkouts.

### Phase D: Benchmark Metadata Hardening

Actions:

- add memory metadata to `docs/benchmarks/benchmark_bayesfilter_v1_filters.py`;
- rerun a small smoke benchmark;
- update the benchmark Markdown note with memory interpretation.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 1 \
  --timesteps 4 \
  --state-dim 2 \
  --observation-dim 2 \
  --parameter-dim 2
```

Pass criterion:

- all rows report `status = ok`;
- rows include memory metadata or an explicit memory caveat.

### Phase E: Medium-shape CPU Benchmark

Actions:

- run one medium-shape linear benchmark artifact;
- include nonlinear rows only if point count and runtime remain modest;
- record any runtime veto explicitly.

Candidate command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 2 \
  --timesteps 24 \
  --state-dim 10 \
  --observation-dim 5 \
  --parameter-dim 2 \
  --output docs/benchmarks/bayesfilter-v1-filter-benchmark-medium-2026-05-11.json
```

Pass criterion:

- all planned rows finish, or failures are captured in the artifact with exact
  backend and shape.

### Phase F: Optional External Status Decision

Actions:

- decide whether to run optional live MacroFinance compatibility now;
- if not run, record `not run by policy` in the reset memo;
- if run, run only tests and do not edit MacroFinance.

Optional command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_macrofinance_linear_compat_tf.py \
  -p no:cacheprovider
```

Pass criterion:

- status is explicit and does not become a required CI gate.

### Phase G: DSGE, GPU, HMC, And SVD-derivative Blocker Review

Actions:

- confirm the compatibility matrix still blocks SGU production filtering;
- confirm Rotemberg/EZ remain future optional live candidates only;
- confirm GPU/XLA-GPU, HMC, and SVD/eigen derivative claims remain blocked.

Pass criterion:

- blocked claims are explicit and have named future evidence requirements.

### Phase H: Commit Or Handoff

Actions:

- stage only lane-owned files;
- run staged diff checks;
- commit only if the user authorizes or the execution request includes commit;
- otherwise record exact uncommitted handoff state.

Allowed staged patterns:

```text
docs/plans/bayesfilter-v1-*.md
docs/benchmarks/benchmark_bayesfilter_v1_filters.py
docs/benchmarks/bayesfilter-v1-filter-benchmark*.json
docs/benchmarks/bayesfilter-v1-filter-benchmark*.md
tests/test_v1_public_api.py
docs/source_map.yml entries whose keys begin with bayesfilter_v1_
```

Forbidden staged files:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
docs/plans/bayesfilter-structural-svd-12-phase-execution-reset-memo-2026-05-06.md
docs/plans/bayesfilter-structural-sgu-goals-gaps-next-plan-2026-05-08.md
docs/chapters/ch18b_structural_deterministic_dynamics.tex
docs/plans/templates/*:Zone.Identifier
singularity_test.png
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

## Completion Criteria

This phase is complete when:

- this May 11 plan is registered and reset-memo recorded;
- v1 API import gate passes;
- focused local v1 regression subset passes;
- benchmark harness records timing, shape, point-count, device, and memory
  metadata or an explicit memory caveat;
- a medium-shape CPU benchmark artifact exists or a veto is recorded;
- optional live MacroFinance status is explicit;
- DSGE candidates remain contained;
- GPU/HMC/SVD-derivative blockers remain explicit;
- lane-owned artifacts are committed or handed off with an exact file list.

## Immediate Recommendation

Execute Phases A through D first.  Continue to Phase E only if the hardened
small benchmark remains stable.  Record Phase F as `not run by policy` unless
the user explicitly asks for optional live MacroFinance checks.  Keep Phase G
as a documentation audit.  Do not commit until the user requests execution with
commit or explicitly approves committing the lane-owned files.
