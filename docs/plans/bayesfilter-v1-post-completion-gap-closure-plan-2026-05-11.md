# Plan: BayesFilter v1 Post-completion Gap Closure

## Date

2026-05-11

## Governing Lane

This plan belongs only to the BayesFilter v1 external-compatibility lane.

Use the lane reset memo:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

Do not edit or stage:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
docs/plans/bayesfilter-structural-svd-12-phase-execution-reset-memo-2026-05-06.md
docs/plans/bayesfilter-structural-sgu-goals-gaps-next-plan-2026-05-08.md
docs/chapters/ch18b_structural_deterministic_dynamics.tex
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

MacroFinance and DSGE remain external compatibility targets until a later v1
integration lane is explicitly opened.

## Starting Point

The previous phase completed and was committed as:

```text
b83d4af Complete v1 external compatibility phase
```

Relevant result artifact:

```text
docs/plans/bayesfilter-v1-phase-completion-result-2026-05-11.md
```

Evidence already available:

- v1 API import gate: `2 passed, 2 warnings`;
- local v1 QR/SVD/compiled-parity regression subset:
  `31 passed, 2 warnings`;
- CPU smoke benchmark: all rows `status = ok`;
- medium CPU benchmark: all rows `status = ok`;
- optional live MacroFinance status: `not_run_by_policy`;
- DSGE status: read-only containment; SGU blocked; Rotemberg/EZ future
  optional fixtures;
- GPU/XLA-GPU, HMC, and linear SVD/eigen derivative claims remain blocked.

## Goals For This Phase

1. Keep the v1 lane isolated while several agents are active in the repository.
2. Turn the medium CPU benchmark finding into a concrete diagnosis of QR
   score/Hessian first-call tracing and memory cost.
3. Build a reproducible benchmark ladder that separates first-call, steady-call,
   graph, eager, value, score, and Hessian costs.
4. Decide whether GPU/XLA-GPU work is justified by running escalated probes and,
   if available, matching-shape GPU benchmark artifacts.
5. Run optional live MacroFinance compatibility as read-only evidence, or record
   a clean skip/failure without changing MacroFinance.
6. Design, but do not yet productionize, test-only DSGE compatibility fixtures
   for Rotemberg and EZ.
7. Pick the first HMC readiness target conservatively, favoring the strongest
   existing QR derivative path before SVD/CUT derivative branches.
8. Keep linear SVD/eigen derivatives deferred unless a real client gap proves
   QR derivatives are insufficient.
9. Produce clear result artifacts and a scoped commit for only v1-lane files.

## Gaps Remaining

### Gap 1: QR Score/Hessian First-call Cost Is Too Large To Ignore

The medium benchmark showed:

```text
eager QR score/Hessian first call: about 57.6 seconds
eager QR score/Hessian high-water RSS increase: about 3335.8 MB
graph QR score/Hessian first call: about 11.1 seconds
graph QR score/Hessian high-water RSS increase: about 486.3 MB
```

Closure target:

- identify whether the cost is dominated by tracing, nested derivative
  contractions, Hessian materialization, parameter dimension, time dimension, or
  graph caching;
- produce a diagnostic artifact before any optimization claim.

### Gap 2: Benchmark Ladder Is Still Too Sparse

The phase has one small and one medium CPU artifact.  That is enough to expose
a problem but not enough to characterize scaling.

Closure target:

- add a small shape ladder with fixed seeds and explicit row labels;
- record value-only, score-only if available, and score/Hessian costs
  separately when possible;
- preserve CPU-only and no-client-claim labels.

### Gap 3: GPU/XLA-GPU Evidence Is Missing

No escalated GPU probe or matching GPU artifact exists.

Closure target:

- run escalated GPU probes before any GPU benchmark;
- if TensorFlow sees a GPU, run a matching-shape GPU benchmark;
- if XLA-GPU fails, record unsupported operation/backend details rather than
  hiding the failure.

### Gap 4: Optional Live MacroFinance Compatibility Is Not Certified

The previous phase explicitly recorded `not_run_by_policy`.

Closure target:

- run `tests/test_macrofinance_linear_compat_tf.py` read-only when appropriate;
- record MacroFinance path and commit hash;
- do not edit MacroFinance and do not switch defaults.

### Gap 5: DSGE Optional Fixtures Are Only Classified, Not Bridged

Rotemberg and EZ are future optional live candidates; SGU remains blocked.

Closure target:

- design test-only bridge requirements for Rotemberg and EZ;
- keep DSGE economics in `/home/chakwong/python`;
- do not import DSGE modules from BayesFilter production code.

### Gap 6: HMC Readiness Has No First Target

HMC readiness is target-specific and cannot be inferred from generic derivative
tests.

Closure target:

- choose a first HMC smoke target with the strongest current evidence;
- require value, score, Hessian/curvature, compiled parity, nonfinite event,
  and sampler diagnostics.

### Gap 7: SVD-CUT Derivative Branch Readiness Is Not Yet Quantified

SVD-CUT value behavior exists, but derivative/HMC claims depend on floor and
spectral-gap branch diagnostics.

Closure target:

- record active-floor frequency, weak spectral-gap frequency, and branch
  stability for SVD-CUT derivative fixtures;
- keep SVD-CUT HMC blocked until smooth-branch dominance is proven.

### Gap 8: Linear SVD/eigen Derivatives Still Lack A Client Need

Value backends exist, but derivative promotion remains unjustified.

Closure target:

- keep SVD/eigen derivatives deferred unless MacroFinance/DSGE compatibility
  evidence proves QR derivatives are insufficient.

### Gap 9: CI And Runtime Policy Need A Practical Boundary

The focused local subset takes about 84.5 seconds.  The medium benchmark can
take materially longer.

Closure target:

- decide which tests are fast CI, extended CPU diagnostics, optional external
  checks, GPU checks, and HMC checks.

### Gap 10: Repository Coordination Remains Dirty Outside This Lane

The worktree contains out-of-lane changes from other agents.

Closure target:

- stage only v1-lane files;
- leave shared monograph reset memo, sidecars, local images, and client worktree
  changes untouched;
- check origin before pushing or integrating if the user requests sync.

## Hypotheses To Test

| ID | Hypothesis | Gap | Test or evidence | Closing rule |
| --- | --- | --- | --- | --- |
| H1 | QR score/Hessian first-call cost is mostly tracing/materialization rather than steady recurrence cost. | G1 | Benchmark value-only, graph warmup, score/Hessian rows across fixed shapes. | Close if first-call costs dominate and steady calls remain small with reproducible memory metadata. |
| H2 | Parameter dimension and time dimension drive QR derivative memory more than state dimension at current shapes. | G1, G2 | Shape ladder varying one axis at a time. | Close if scaling table identifies the dominant axis or rejects the hypothesis. |
| H3 | The benchmark harness can support a diagnostic ladder without external dependencies. | G2 | CPU-only ladder artifact under `docs/benchmarks`. | Close if all rows finish or failures are captured per backend/shape. |
| H4 | Escalated TensorFlow can see the GPU on this machine, and XLA-GPU may improve steady calls for some backends. | G3 | Escalated `nvidia-smi`, escalated TensorFlow probe, matching CPU/GPU benchmark. | Close if GPU artifact exists, or if probe failure is recorded as real escalated evidence. |
| H5 | Optional live MacroFinance compatibility passes read-only on the current external checkout. | G4 | `tests/test_macrofinance_linear_compat_tf.py` plus external commit hash. | Close as `passed`, `skipped`, or `failed` without editing MacroFinance. |
| H6 | Rotemberg can be a useful test-only DSGE fixture without BayesFilter owning DSGE economics. | G5 | Bridge design note plus future optional live test plan. | Close design phase if production imports remain forbidden and required residual diagnostics are named. |
| H7 | EZ can be a metadata-only optional fixture while determinacy/HMC labels remain blocked. | G5 | Bridge design note with explicit BK/QZ/HMC veto labels. | Close design phase if metadata checks cannot imply determinacy or posterior readiness. |
| H8 | The first HMC target should be a linear QR derivative target, not SVD/CUT. | G6 | Target-selection audit comparing evidence strength. | Close if a named target and diagnostics ladder are selected. |
| H9 | SVD-CUT derivative/HMC promotion is blocked by branch diagnostics, not by value-filter absence. | G7 | Branch frequency and spectral-gap diagnostic plan/test. | Close if blocker status is quantified and retained. |
| H10 | Linear SVD/eigen derivatives are unnecessary for v1 unless optional live compatibility proves otherwise. | G8 | Client-need review after MacroFinance/DSGE checks. | Close if no client path requires them, or open a separate derivative prototype plan. |
| H11 | v1 CI can be split into fast local, extended CPU, optional external, GPU, and HMC tiers. | G9 | CI/runtime policy artifact. | Close if each test has a tier and default execution policy. |
| H12 | The phase can be completed without touching other agents' files. | G10 | `git status`, staged-file audit, `git diff --cached --check`. | Close only if staged files are v1-lane owned. |

## Execution Plan

Each phase should use:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

### Phase A: Lane And Origin Audit

Actions:

- run `git status --short --branch`;
- classify files as v1-lane, out-of-lane dirty, or unrelated untracked;
- optionally run `git fetch` and compare against origin if a sync/push step is
  requested;
- do not stage out-of-lane files.

Primary criterion:

- execution can proceed without editing shared monograph, structural SVD/SGU,
  Chapter 18/18b, MacroFinance, or DSGE files.

Veto diagnostics:

- any required change belongs to another lane;
- origin conflict requires merge decisions outside this plan.

### Phase B: QR Derivative Cost Diagnostic Plan And Harness Extension

Actions:

- add or plan benchmark switches for:
  - value-only;
  - score/Hessian;
  - graph warmup;
  - linear-only runs;
  - shape ladder output naming;
- keep CPU-only default behavior.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 python \
  docs/benchmarks/benchmark_bayesfilter_v1_filters.py \
  --repeats 2 \
  --timesteps 4 \
  --state-dim 2 \
  --observation-dim 2 \
  --parameter-dim 2
```

Primary criterion:

- existing smoke behavior remains green and memory metadata remains present.

### Phase C: CPU Shape-ladder Artifact

Actions:

- run small CPU ladders varying one axis at a time;
- record JSON and Markdown artifacts under `docs/benchmarks`;
- extract a summary table of first-call and high-water RSS behavior.

Candidate one-axis ladders:

```text
v1_time_ladder:
  (timesteps=4,  state_dim=2, observation_dim=2, parameter_dim=2)
  (timesteps=8,  state_dim=2, observation_dim=2, parameter_dim=2)
  (timesteps=12, state_dim=2, observation_dim=2, parameter_dim=2)
  (timesteps=16, state_dim=2, observation_dim=2, parameter_dim=2)

v1_parameter_ladder:
  (timesteps=8, state_dim=2, observation_dim=2, parameter_dim=2)
  (timesteps=8, state_dim=2, observation_dim=2, parameter_dim=3)
  (timesteps=8, state_dim=2, observation_dim=2, parameter_dim=4)

v1_state_observation_ladder:
  (timesteps=8, state_dim=2, observation_dim=2, parameter_dim=2)
  (timesteps=8, state_dim=3, observation_dim=2, parameter_dim=2)
  (timesteps=8, state_dim=4, observation_dim=3, parameter_dim=2)
```

Primary criterion:

- ladder artifact completes or records per-row failures without weakening
  claims.

Veto diagnostics:

- the ladder changes multiple axes while claiming one-axis attribution;
- a failed row is silently dropped rather than recorded with backend and shape.

### Phase D: GPU/XLA-GPU Probe And Matching Benchmark

Actions:

- run GPU checks only with escalated sandbox permissions;
- first run `nvidia-smi`;
- then run an escalated TensorFlow device probe;
- if GPU is visible, modify or parameterize the benchmark harness so GPU runs do
  not force `CUDA_VISIBLE_DEVICES=-1`;
- run matching CPU/GPU shape artifacts.

Primary criterion:

- either a valid GPU benchmark artifact exists, or a valid escalated failure
  artifact explains why GPU evidence remains blocked.

Veto diagnostics:

- non-escalated GPU failure is treated as real device evidence;
- CPU and GPU shapes differ without being labeled.

### Phase E: Optional Live MacroFinance Read-only Check

Actions:

- run the optional live MacroFinance test if the checkout is present;
- record external path and commit hash;
- do not edit MacroFinance.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_macrofinance_linear_compat_tf.py \
  -p no:cacheprovider
```

Primary criterion:

- status is recorded as `passed`, `skipped`, or `failed` with interpretation.

### Phase F: DSGE Optional-fixture Design

Actions:

- write a design note for Rotemberg and EZ test-only bridges;
- name required metadata, deterministic residual, support/rank, and value
  likelihood checks;
- keep SGU blocked.

Primary criterion:

- the design enables future optional live tests without adding DSGE production
  imports to BayesFilter.

### Phase G: First HMC Target Selection

Actions:

- compare candidate targets:
  - linear QR derivative fixture;
  - SVD-CUT smooth-branch fixture;
  - Rotemberg optional live fixture;
- select the first target for an HMC readiness smoke plan.

Primary criterion:

- a named target has an evidence ladder covering value, score, Hessian,
  compiled parity, nonfinite diagnostics, and sampler smoke criteria.

### Phase H: SVD-CUT Branch Diagnostic Gate

Actions:

- define active-floor, weak spectral-gap, rank, and support diagnostics for
  SVD-CUT derivatives;
- keep derivative/HMC labels blocked unless smooth-branch diagnostics dominate.

Primary criterion:

- SVD-CUT derivative promotion has a measurable gate rather than a verbal
  caution.

### Phase I: CI Tier Policy

Actions:

- classify tests into:
  - fast local CI;
  - extended CPU diagnostics;
  - optional live external;
  - escalated GPU;
  - HMC readiness;
- record expected runtime and trigger policy.

Primary criterion:

- developers can run the right level of evidence without accidentally requiring
  external projects or GPU access.

### Phase J: Result, Reset Memo, And Commit Boundary

Actions:

- create a phase result artifact;
- update only the v1 lane reset memo;
- register new artifacts in `docs/source_map.yml`;
- stage only v1-lane files;
- run `git diff --cached --check`;
- commit only when the execution pass is complete or the user asks for a
  partial checkpoint.

Primary criterion:

- no out-of-lane files are staged and all claims match evidence.

## Completion Definition

This phase closes when:

- QR derivative first-call cost has a diagnostic artifact and next decision;
- CPU benchmark ladder exists or has a recorded veto;
- GPU/XLA-GPU is either evidenced or still blocked by escalated evidence;
- optional live MacroFinance status is recorded;
- Rotemberg/EZ bridge design exists and SGU remains blocked;
- first HMC target is selected with diagnostics;
- SVD-CUT branch gate is explicit;
- CI tier policy is documented;
- v1-lane artifacts are committed or handed off with exact file lists.

## Recommended First Move

Start with Phases A through C.  These phases are BayesFilter-local, CPU-only,
and provide the clearest information about whether the QR score/Hessian cost is
a tractable benchmark/harness issue or a deeper implementation issue.  Defer
GPU and optional live external checks until after the CPU ladder clarifies the
shape and backend pairs worth testing.
