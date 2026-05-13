# Plan: BayesFilter v1 Score API, HMC, And SVD-CUT Gap Closure

## Date

2026-05-11

## Lane

This is a BayesFilter v1 lane plan.  It follows:

```text
7526b72 Close v1 HMC branch diagnostics
3261a42 Record v1 HMC readiness addendum
```

Stay inside BayesFilter-local files:

```text
bayesfilter/*
tests/*
docs/benchmarks/*
docs/plans/bayesfilter-v1-*
docs/source_map.yml
pytest.ini
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

MacroFinance and DSGE remain read-only external compatibility targets.  This
phase must not switch client defaults.

## Goals For This Phase

1. Decide whether the private first-order QR score path should become a public
   v1 score-only API, and under what dense and masked-observation semantics.
2. If the API decision passes, implement the smallest safe public score-only QR
   surface and keep full score/Hessian as the curvature diagnostic.
3. Extend the first QR HMC smoke from a single tiny finite check to a longer
   target-specific multi-chain diagnostic without making a convergence claim
   unless the evidence supports it.
4. Widen SVD-CUT branch-frequency diagnostics beyond the tiny smooth box and
   decide whether a separate SVD-CUT HMC plan is justified.
5. Decide whether QR derivative GPU/XLA is worth testing on one CPU-selected
   medium shape, and run it only with escalated GPU/CUDA visibility.
6. Keep CI tiers small by default and make expensive score/HMC/SVD-CUT/GPU
   checks opt-in and reproducible.
7. Produce auditable artifacts, update the v1 reset memo and source map, and
   commit only v1-lane files if execution is later requested.

## Current Status

Closed already:
- private dense first-order QR score helpers match full score/Hessian and
  autodiff on a tiny dense fixture;
- score-only diagnostics are materially cheaper than full score/Hessian at the
  first target shape;
- the first QR HMC smoke finishes with finite samples and diagnostics;
- SVD-CUT branch-frequency telemetry exists for a tiny smooth box;
- HMC and SVD-CUT diagnostics are opt-in.

Not yet closed:
- public score-only QR API contract;
- masked score-only semantics;
- longer multi-chain QR HMC diagnostics;
- wider SVD-CUT branch coverage;
- QR derivative GPU/XLA matching-shape evidence;
- clean external-client release certification.

## Gaps Remaining

### Gap 1: Public QR Score-only API Is Undecided

The private first-order score path is useful, but public promotion requires a
contract for dense observations, masked observations, metadata, diagnostics,
backend names, and failure modes.

Closure target:
- write a short API decision note;
- decide whether dense-only public exposure is acceptable, or whether dense and
  masked score-only must ship together;
- preserve public API freeze tests.

### Gap 2: Masked Score-only Semantics Are Undefined

Masked score/Hessian works through the static dummy-row convention, but
score-only could accidentally diverge from full score/Hessian if dummy rows,
all-missing periods, or metadata differ.

Closure target:
- define masked score-only semantics before implementation;
- prove all-true, sparse, and all-missing mask parity against full
  score/Hessian and autodiff references.

### Gap 3: HMC Evidence Is Only A Tiny Smoke

The first target has finite value/gradient/curvature and a short HMC smoke, but
no longer multi-chain stability, adaptation, or energy/log-probability
diagnostics.

Closure target:
- run a CPU-only multi-chain diagnostic with fixed seeds;
- record acceptance, step-size policy, finite counts, log-probability ranges,
  simple posterior summaries, and explicit no-convergence or convergence
  labeling.

### Gap 4: SVD-CUT Branch Evidence Is Too Narrow

The tiny smooth box is encouraging but does not justify SVD-CUT HMC.  We need
to know whether floors, weak spectral gaps, support residuals, or deterministic
residuals appear in wider relevant regions.

Closure target:
- widen the branch sweep;
- include smooth box, weak-gap control, active-floor control, and a moderate
  target-region grid;
- write a separate SVD-CUT HMC plan only if branch evidence is robust.

### Gap 5: QR Derivative GPU/XLA Remains Untested

Prior GPU-visible and XLA-visible value tests do not certify derivative GPU/XLA
readiness.  GPU/CUDA runs require escalated permissions on this machine.

Closure target:
- choose one medium CPU shape from existing score/Hessian evidence;
- run matching CPU and escalated GPU-visible/XLA-visible rows only if the CPU
  shape is stable;
- distinguish GPU-visible process from confirmed GPU placement.

### Gap 6: External Compatibility Is Still Read-only Evidence

MacroFinance and DSGE are useful compatibility targets, but switching them to
BayesFilter before v1 stabilizes would couple projects too early.

Closure target:
- keep optional external checks read-only;
- add BayesFilter-local fixtures or descriptors rather than production client
  imports;
- defer client switch-over to a separate integration plan.

## Hypotheses To Test

| ID | Hypothesis | Gap | Test | Close If |
| --- | --- | --- | --- | --- |
| H1 | The private first-order QR path can be promoted to a public score-only API without destabilizing v1. | G1 | API decision note plus public API tests. | Dense and masked semantics are explicit, or promotion is blocked with rationale. |
| H2 | Dense score-only parity holds across the existing tiny QR derivative fixtures and a small shape ladder. | G1 | Compare score-only against full score/Hessian and autodiff. | Value and score residuals stay inside existing tolerances. |
| H3 | Masked score-only can reuse the static dummy-row convention without changing all-true, sparse, or all-missing behavior. | G2 | Masked all-true/sparse/all-missing parity tests. | Score-only equals full score/Hessian/autodiff within tolerances. |
| H4 | Longer QR HMC diagnostics remain finite under multi-chain CPU runs. | G3 | Fixed-seed multi-chain HMC/NUTS diagnostic. | No nonfinite samples/gradients; acceptance and log-probability ranges are recorded and interpretable. |
| H5 | The first QR target is still only smoke-ready, not convergence-certified. | G3 | Audit posterior summaries, chain lengths, and adaptation evidence. | Result labels remain honest: smoke, diagnostic, or convergence-ready. |
| H6 | SVD-CUT smoothness is local and may fail under wider parameter sweeps. | G4 | Wider branch-frequency sweep with controls. | Smooth fraction, floor counts, weak gaps, and residuals quantify promotion or blocker status. |
| H7 | SVD-CUT HMC should remain blocked unless a wider sweep is smooth and inactive-floor dominated. | G4 | Branch audit plus target-specific plan gate. | Either write a new SVD-CUT HMC plan or record blocker labels. |
| H8 | QR derivative GPU/XLA is useful only for a CPU-selected medium shape. | G5 | Matching CPU and escalated GPU-visible/XLA-visible benchmark rows. | GPU/XLA rows complete with device evidence, or deferral is documented. |
| H9 | External compatibility can remain read-only while BayesFilter v1 score/HMC work progresses locally. | G6 | Optional external policy audit. | No client source edits, no production client dependency. |
| H10 | CI tiers can contain the new work without slowing default development. | G1-G5 | Marker/env-gate audit and focused default test command. | Fast/default checks skip HMC, external, GPU, and wide branch sweeps. |

## Execution Plan

Run any future execution pass phase by phase:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

Continue automatically only when the phase primary criterion passes and no veto
diagnostic fires.

### Phase 0: Lane And Worktree Audit

Actions:
- run `git status --short --branch`;
- confirm only v1-lane files are needed;
- ignore existing out-of-lane dirty files.

Primary criterion:
- no structural, monograph, MacroFinance, or DSGE edit is needed.

Veto diagnostics:
- any required change crosses into another agent's lane.

### Phase 1: Score-only API Decision

Actions:
- write the score-only API decision note;
- define dense and masked semantics, metadata, backend names, and failure
  modes;
- decide whether implementation proceeds now or is blocked.

Tests:
- documentation audit;
- public API freeze test remains unchanged unless promotion is approved.

Primary criterion:
- a clear promote/block decision exists.

Veto diagnostics:
- dense score-only is promoted while masked behavior remains ambiguous;
- API surface conflicts with existing v1 naming.

### Phase 2: Score-only Implementation Or Blocked Result

Actions if approved:
- expose the smallest score-only QR wrapper;
- implement masked score-only only if Phase 1 approved semantics;
- keep full score/Hessian wrappers unchanged.

Actions if blocked:
- write a blocked result with exact missing decisions.

Tests:
- dense parity against full score/Hessian and autodiff;
- masked all-true/sparse/all-missing parity if masked score-only is added;
- `tests/test_v1_public_api.py`.

Primary criterion:
- implementation passes parity, or blocked status is explicit.

Veto diagnostics:
- HMC path starts depending on Hessian materialization by default;
- score-only silently changes mask conventions.

### Phase 3: Longer QR HMC Diagnostic

Actions:
- add a multi-chain CPU diagnostic script or benchmark mode;
- record finite counts, acceptance, step-size policy, log-probability ranges,
  sample moments, and runtime;
- keep it opt-in.

Tests:
- `BAYESFILTER_RUN_HMC_READINESS=1` focused HMC diagnostics.

Primary criterion:
- longer target-specific HMC diagnostic artifact exists.

Veto diagnostics:
- nonfinite states or gradients recur;
- result is labeled as convergence when chain length/tuning evidence is only
  smoke-level.

### Phase 4: Wider SVD-CUT Branch Sweep

Actions:
- add a wider branch-frequency benchmark mode;
- include smooth, weak-gap, active-floor, and target-region grids;
- record branch labels and residuals.

Tests:
- `BAYESFILTER_RUN_EXTENDED_CPU=1` branch diagnostic tests;
- JSON/Markdown branch artifacts.

Primary criterion:
- SVD-CUT HMC is either still blocked with quantified evidence or promoted only
  to a new plan.

Veto diagnostics:
- regularization hides active floors or weak gaps;
- tiny-box evidence is generalized to a broad HMC claim.

### Phase 5: Optional QR Derivative GPU/XLA Gate

Actions:
- choose a CPU-stable medium shape;
- run escalated `nvidia-smi` and TensorFlow GPU probe before GPU tests;
- run matching CPU and GPU/XLA rows only if probes pass.

Tests:
- escalated GPU/CUDA device probe;
- matching CPU/GPU benchmark artifacts.

Primary criterion:
- GPU/XLA derivative status is either supported for the one shape or deferred
  with exact blocker evidence.

Veto diagnostics:
- non-escalated GPU failure is treated as real device failure;
- GPU-visible process is mislabeled as confirmed GPU placement.

### Phase 6: CI, Reset Memo, Source Map, Commit

Actions:
- update v1 reset memo with phase results;
- update `docs/source_map.yml`;
- run focused default and opt-in tests;
- stage only v1-lane files;
- commit only after validation if execution is requested.

Primary criterion:
- every claim is backed by a test, benchmark, or explicit blocked status.

Veto diagnostics:
- out-of-lane files are staged;
- expensive tests become default CI by accident.

## Suggested Commands

Fast focused gate:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_hmc_linear_qr_readiness_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

HMC opt-in:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
```

SVD-CUT opt-in:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

GPU/XLA commands are intentionally omitted from the default plan body because
they require escalated GPU/CUDA visibility under the local sandbox policy.

## Completion Rule

This phase is complete when:

- public score-only QR is either safely implemented or explicitly blocked;
- QR HMC has a longer target-specific diagnostic artifact;
- SVD-CUT has wider branch evidence and remains blocked or gets a separate HMC
  plan;
- GPU/XLA derivative status is tested or explicitly deferred;
- CI tiers remain opt-in for expensive diagnostics;
- reset memo and source map are updated;
- only v1-lane files are committed.
