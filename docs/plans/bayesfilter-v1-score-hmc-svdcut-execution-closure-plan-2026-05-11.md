# Plan: BayesFilter v1 Score, HMC, And SVD-CUT Execution Closure

## Date

2026-05-11

## Lane And Motivation

This is the execution-ready plan for the BayesFilter v1 lane.  It tightens the
broader follow-up plan in:

```text
docs/plans/bayesfilter-v1-score-api-hmc-svdcut-gap-closure-plan-2026-05-11.md
```

The motivation is to close the remaining v1 filtering gaps locally before any
MacroFinance or DSGE switch-over.  MacroFinance and DSGE remain external
read-only compatibility targets.  This plan must not edit their repositories,
shared structural plans, monograph reset memos, or Chapter 18/18b text.

Allowed write lane:

```text
bayesfilter/*
tests/*
docs/benchmarks/*
docs/plans/bayesfilter-v1-*
docs/source_map.yml
pytest.ini
```

Out of lane:

```text
docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md
docs/plans/bayesfilter-structural-*
docs/chapters/ch18b_structural_deterministic_dynamics.tex
/home/chakwong/MacroFinance/*
/home/chakwong/python/*
```

## Goals For This Phase

1. Decide whether QR first-order score-only helpers should stay private or
   become a public v1 API.
2. If public promotion is justified, define dense and masked score-only
   semantics before implementation.
3. Extend the first linear QR HMC evidence from a tiny finite smoke to a
   longer target-specific multi-chain diagnostic, still without a convergence
   claim unless the evidence genuinely warrants it.
4. Widen SVD-CUT branch diagnostics so SVD-CUT HMC remains blocked or receives
   a separate target-specific plan based on quantified branch evidence.
5. Decide whether one escalated GPU/XLA QR derivative gate is worth running on
   a CPU-stable medium shape.
6. Keep expensive tests opt-in and keep default CPU CI fast.
7. Leave a clean reset trail, source-map entry, benchmark artifacts, and scoped
   commit boundary for the v1 lane.

## Current Status

Closed:

- BayesFilter v1 has TF/TFP production linear QR value and QR score/Hessian
  paths.
- Dense private QR score-only helpers exist and match full score/Hessian plus
  autodiff on the tiny fixture.
- Masked QR value and masked QR score/Hessian already use the static dummy-row
  convention.
- A first target-specific linear QR HMC smoke completed with finite target,
  gradient, curvature, and sample diagnostics.
- SVD-CUT score/Hessian has smooth-branch finite-difference and autodiff tests,
  with branch blockers for active floors and weak spectral gaps.
- HMC and SVD-CUT branch diagnostics are opt-in.

Not closed:

- Public QR score-only API contract is undecided.
- Masked score-only semantics are not yet specified or tested.
- HMC evidence is still smoke-level.
- SVD-CUT branch evidence is too narrow for sampler promotion.
- QR derivative GPU/XLA has no matching-shape derivative artifact.
- External-client release certification is intentionally deferred.

## Remaining Gaps And Hypotheses

| Gap | Hypothesis | Test Or Artifact | Closure Rule |
| --- | --- | --- | --- |
| G1: Score-only API contract | Public score-only QR is safe only if dense and masked semantics are explicit before export. | API decision note and `tests/test_v1_public_api.py`. | Close by either promoting a minimal API with dense and masked semantics or recording a blocker that keeps helpers private. |
| G2: Masked score-only semantics | Masked score-only can reuse the static dummy-row convention without changing all-true, sparse, or all-missing behavior. | Masked score-only parity tests against full score/Hessian and autodiff references. | Close only if all mask cases match existing score/Hessian tolerances. |
| G3: HMC diagnostic depth | The first QR HMC target remains finite under longer multi-chain CPU diagnostics, but evidence may still be diagnostic rather than convergence. | Fixed-seed multi-chain HMC benchmark artifact with acceptance, finite counts, log-probability ranges, step-size policy, and sample summaries. | Close as "diagnostic-ready" unless chain length, adaptation, and posterior summaries justify stronger language. |
| G4: SVD-CUT branch coverage | Smooth SVD-CUT behavior is local until weak-gap, active-floor, and wider target-region sweeps say otherwise. | Wider branch-frequency benchmark artifact with branch labels and residuals. | Close by keeping SVD-CUT HMC blocked with quantified reasons or by writing a separate HMC plan. |
| G5: GPU/XLA derivative evidence | GPU/XLA derivative testing is useful only after choosing a CPU-stable medium shape. | CPU medium-shape artifact plus escalated GPU probe and optional XLA derivative row. | Close as supported for one shape or explicitly deferred with device/blocker evidence. |
| G6: External compatibility | BayesFilter can progress locally without coupling MacroFinance or DSGE before v1. | Lane audit and optional read-only compatibility notes. | Close if no external source edits, imports, or default switch-over enter this phase. |
| G7: CI containment | New diagnostics can remain opt-in without slowing default development. | Marker/env-gate audit and default CPU test command. | Close if default tests skip HMC, GPU, external, and wide SVD-CUT sweeps. |

## Execution Path

Every phase follows:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Continue automatically only when the primary criterion passes and no veto
diagnostic fires.

### Phase 0: Lane Recovery And Baseline Tests

Plan:
- classify current dirty files as v1-lane or out-of-lane;
- verify no other agent has changed v1 files since the last committed state;
- run the focused CPU gate before editing.

Execute:
- `git status --short --branch`;
- inspect v1 diffs only if any appear;
- do not stage or revert out-of-lane files.

Test:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_hmc_linear_qr_readiness_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Primary criterion:
- focused default CPU gate passes or skipped opt-in tests remain skipped for
  the documented reason.

Veto:
- any required repair touches structural, monograph, MacroFinance, or DSGE
  files.

### Phase 1: Score-only API Decision

Plan:
- define dense score-only input/output semantics;
- define masked score-only behavior for all-true, sparse, and all-missing
  rows;
- decide whether public promotion is safer now or should remain blocked until
  masked score-only parity exists.

Execute:
- write a v1 API decision note under `docs/plans/bayesfilter-v1-*`;
- if promotion is blocked, stop score-only implementation and move to HMC/SVD
  diagnostics.

Test:
- `tests/test_v1_public_api.py`;
- documentation audit against `bayesfilter/__init__.py` and
  `bayesfilter/linear/__init__.py`.

Primary criterion:
- one unambiguous decision: promote with complete semantics, or keep private
  with explicit blockers.

Veto:
- dense-only public API is added while masked semantics remain undefined.

### Phase 2: Score-only Implementation Or Blocker Result

Plan:
- if Phase 1 promotes, expose the smallest public wrapper and no more;
- if Phase 1 blocks, record a blocker result and leave code unchanged.

Execute if promoted:
- add public dense and masked QR score-only wrappers;
- preserve existing score/Hessian wrappers;
- avoid adding NumPy to production code.

Test if promoted:
- dense score-only parity versus full score/Hessian and autodiff;
- masked all-true, sparse, and all-missing parity;
- `tests/test_v1_public_api.py`;
- `git diff --check`.

Primary criterion:
- parity passes, or blocker status is documented without public API changes.

Veto:
- HMC path becomes dependent on Hessian materialization;
- mask metadata diverges from `static_dummy_row`.

### Phase 3: Longer Linear QR HMC Diagnostic

Plan:
- extend the existing HMC smoke into a longer, still bounded, CPU-only
  multi-chain diagnostic;
- record diagnostics without claiming convergence by default.

Execute:
- extend `bayesfilter.testing` and/or `docs/benchmarks` HMC helper options;
- generate JSON and Markdown artifacts under `docs/benchmarks`.

Test:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
```

Primary criterion:
- finite target, score, samples, and log-probability diagnostics are recorded
  for the longer run.

Veto:
- nonfinite states or gradients recur;
- artifact language claims convergence without adequate chain evidence.

### Phase 4: Wider SVD-CUT Branch Sweep

Plan:
- extend branch-frequency diagnostics from the tiny smooth box to smooth,
  weak-gap, active-floor, and target-region controls.

Execute:
- update branch diagnostic helpers or benchmark modes;
- write JSON and Markdown artifacts with smooth fraction, weak gaps, floor
  counts, support residuals, and deterministic residuals.

Test:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Primary criterion:
- SVD-CUT HMC remains blocked with quantified branch evidence, or a separate
  HMC-specific plan is written.

Veto:
- regularization hides active floors or weak spectral gaps;
- tiny-box smoothness is generalized to sampler readiness.

### Phase 5: Optional GPU/XLA QR Derivative Gate

Plan:
- choose one CPU-stable medium shape from existing QR derivative artifacts;
- only run GPU/XLA if the user or current execution pass authorizes escalated
  GPU/CUDA commands.

Execute:
- run escalated `nvidia-smi`;
- run an escalated TensorFlow GPU device probe;
- run one matching CPU and GPU/XLA derivative row if probes pass.

Test:
- compare CPU and GPU/XLA value and score summaries for the chosen shape;
- record whether GPU placement is confirmed or merely GPU-visible.

Primary criterion:
- one-shape GPU/XLA status is supported or explicitly deferred.

Veto:
- non-escalated GPU failure is treated as real device failure;
- GPU-visible process is mislabeled as confirmed GPU placement.

### Phase 6: CI, Reset Memo, Source Map, And Commit Boundary

Plan:
- update the v1 reset memo with per-phase results and next-step justification;
- update `docs/source_map.yml`;
- keep default tests fast and opt-in diagnostics gated.

Execute:
- run focused default and opt-in tests appropriate to changed files;
- run `git diff --check`;
- stage only v1-lane files;
- commit only when requested or when executing the full plan.

Primary criterion:
- every final claim is tied to a test, benchmark artifact, or explicit blocker.

Veto:
- out-of-lane files are staged;
- expensive diagnostics become default tests.

## Completion Definition

This phase is complete when:

- QR score-only API is either promoted with dense and masked semantics or kept
  private with a clear blocker;
- masked score-only behavior is tested if implemented;
- longer QR HMC diagnostics are recorded and honestly labeled;
- SVD-CUT branch evidence is widened and does not imply HMC readiness without a
  separate plan;
- GPU/XLA derivative status is tested with escalation or deferred explicitly;
- reset memo and source map point to the resulting artifacts;
- no MacroFinance, DSGE, structural chapter, or shared monograph files are
  touched.
