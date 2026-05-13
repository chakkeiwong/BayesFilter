# Plan: BayesFilter v1 HMC And Branch-diagnostic Closure

## Date

2026-05-11

## Governing Lane

This plan belongs only to the BayesFilter v1 external-compatibility lane.

Use this reset memo:

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

MacroFinance and DSGE remain external compatibility targets.  This phase does
not switch client defaults and does not add production DSGE imports to
BayesFilter.

## Starting Point

The prior v1 lane commits are:

```text
b83d4af Complete v1 external compatibility phase
ec4f498 Close v1 post-completion diagnostics
```

The active result artifact is:

```text
docs/plans/bayesfilter-v1-post-completion-gap-closure-result-2026-05-11.md
```

Evidence already available:

- v1 local API/regression/SVD-CUT branch suite:
  `35 passed, 2 warnings in 92.35s`;
- CPU benchmark smoke and QR score/Hessian ladders with memory metadata;
- escalated GPU probe evidence and GPU-visible smoke artifacts;
- tiny XLA-visible linear value artifact;
- optional live MacroFinance QR compatibility:
  `4 passed, 2 warnings in 74.54s` on checkout `0e819889...`;
- DSGE remains read-only: Rotemberg future optional structural fixture, EZ
  metadata-only fixture, SGU blocked;
- first HMC target selected:
  `linear_qr_score_hessian_static_lgssm`;
- SVD-CUT derivative/HMC promotion blocked pending branch-frequency evidence.

## Goals For This Phase

1. Keep the lane isolated from the structural SVD/SGU and shared monograph
   work.
2. Turn the QR score/Hessian memory finding into a targeted implementation
   experiment, not just a benchmark observation.
3. Build the first target-specific HMC readiness artifact for
   `linear_qr_score_hessian_static_lgssm`.
4. Add SVD-CUT branch-frequency diagnostics over a small parameter box while
   keeping SVD-CUT HMC blocked unless branch evidence justifies promotion.
5. Decide whether QR derivative XLA/GPU work is worth a follow-up based on the
   small HMC and memory results.
6. Preserve optional MacroFinance and DSGE as external evidence only.
7. Produce a result artifact, reset-memo update, source-map entry, and scoped
   commit when the phase is complete.

## Remaining Gaps

### Gap 1: QR Score/Hessian Memory Driver Is Identified But Not Reduced

The parameter ladder showed graph warmup and max RSS grow sharply with
parameter dimension:

```text
parameter_dim=2: warmup about 20.4 s, max RSS delta about 1236.8 MB
parameter_dim=4: warmup about 64.3 s, max RSS delta about 3319.8 MB
```

Closure target:

- isolate score-only, Hessian-only, and full score/Hessian paths;
- identify whether Hessian materialization, second-order state propagation, or
  parameter-major tensor contractions dominate memory;
- produce a before/after diagnostic artifact if a safe implementation change is
  made.

### Gap 2: First HMC Target Has Been Selected But Not Executed

The chosen target is `linear_qr_score_hessian_static_lgssm`, but no sampler
artifact exists.

Closure target:

- define a fixed small LGSSM target with parameter transform and prior;
- validate value, score, Hessian, compiled parity, and nonfinite diagnostics;
- run a short fixed-seed TFP HMC/NUTS smoke;
- record acceptance, step size, finite log-prob/gradient status, and divergence
  or failure diagnostics.

### Gap 3: SVD-CUT Branch-frequency Evidence Is Missing

SVD-CUT smooth-branch derivative tests exist, but no parameter-box branch
frequency artifact exists.

Closure target:

- sweep a small parameter box;
- record active placement-floor count, active innovation-floor count, weak
  spectral-gap count, finite value/score/Hessian count, support residual, and
  deterministic residual;
- keep HMC blocked unless smooth separated-spectrum inactive-floor behavior
  dominates the target region.

### Gap 4: QR Derivative XLA/GPU Is Not Tested

Tiny XLA-visible linear value rows pass, but QR score/Hessian XLA is untested.

Closure target:

- defer QR derivative XLA unless CPU HMC and memory diagnostics show it is
  worth testing;
- if tested, use escalated GPU permissions, matching shapes, and separate
  XLA-visible artifacts.

### Gap 5: Optional Live External Evidence Is Not Clean Release Certification

MacroFinance optional live compatibility passed on a dirty external checkout.

Closure target:

- keep the pass as useful compatibility evidence;
- do not promote it to release certification or switch-over readiness;
- optionally rerun only after the external checkout is clean and explicitly in
  scope.

### Gap 6: DSGE Bridges Are Design-only

Rotemberg and EZ have design notes, not optional live bridge tests.

Closure target:

- keep DSGE out of this phase unless the user opens a separate DSGE optional
  fixture lane;
- keep SGU blocked.

### Gap 7: CI Tier Policy Is Written But Not Enforced In Tooling

The tier policy exists, but no command wrapper or CI marker split exists.

Closure target:

- decide whether this phase needs tooling enforcement or whether documentation
  is sufficient for v1;
- avoid accidental inclusion of extended CPU, GPU, optional external, or HMC
  tests in fast local CI.

## Hypotheses To Test

| ID | Hypothesis | Gap | Test or evidence | Closing rule |
| --- | --- | --- | --- | --- |
| H1 | Full QR score/Hessian memory is dominated by Hessian materialization and parameter-pair contractions. | G1 | score-only/full-Hessian diagnostic rows, tensor-shape audit, before/after benchmark. | Close if removing or delaying Hessian materialization materially reduces warmup/RSS without changing full-result correctness. |
| H2 | Score-only QR derivatives are cheap enough to use as the first HMC gradient path. | G1, G2 | score-only value/gradient parity and benchmark rows. | Close if score-only returns finite gradients and substantially lower warmup/RSS than full Hessian. |
| H3 | The selected static LGSSM HMC target can run a tiny fixed-seed smoke without nonfinite log-prob/gradient events. | G2 | HMC/NUTS smoke artifact. | Close if all chains finish with finite states and recorded acceptance/step diagnostics. |
| H4 | Hessian diagnostics are useful for curvature checks but should not be in the default HMC log-prob path. | G1, G2 | Compare HMC path with and without Hessian materialization. | Close if HMC smoke only requires value/score and Hessian remains diagnostic. |
| H5 | SVD-CUT branch blockers occur often enough that SVD-CUT HMC must remain blocked. | G3 | Parameter-box branch-frequency sweep. | Close if active floors or weak gaps are non-negligible, or if smooth branch dominance is insufficiently proven. |
| H6 | If the SVD-CUT sweep is entirely smooth on a tiny box, it still only justifies a later target-specific HMC plan, not immediate promotion. | G3 | Branch-frequency artifact plus policy audit. | Close if artifact labels remain diagnostic-only. |
| H7 | QR derivative XLA/GPU should be deferred until CPU target and memory evidence identify a worthwhile shape. | G4 | CPU HMC and memory result review. | Close if GPU/XLA remains optional follow-up, or if a narrowly justified escalated XLA test is defined. |
| H8 | Dirty optional MacroFinance evidence should not block BayesFilter-local HMC work. | G5 | External-status note. | Close if compatibility remains optional and no client source is edited. |
| H9 | DSGE optional bridges are not required for the first HMC target. | G6 | Lane audit. | Close if no DSGE source or production import is needed. |
| H10 | CI tier documentation is enough for this phase unless tests are accidentally slow/default. | G7 | Test selection audit. | Close if fast/local commands remain explicit and extended/GPU/HMC commands stay opt-in. |

## Execution Plan

Every phase should use:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

## Execution Tightening From Pre-run Audit

The pre-run audit tightens the implementation path without changing the phase
goals:

- Do not add a public QR score-only API in this pass.  The current public
  derivative surface returns value, score, and Hessian together, and a safe
  analytic score-only refactor would require separating first- and second-order
  state propagation throughout the QR recursion.
- Treat the first HMC path as QR value plus TensorFlow autodiff gradient on the
  fixed static LGSSM target.  The analytic QR score/Hessian remains the parity
  and curvature diagnostic, not the sampler log-prob implementation.
- Force diagnostic benchmark rows to materialize score and Hessian tensors when
  measuring full derivative paths.  Value-only rows must not be interpreted as
  full derivative memory rows.
- Keep SVD-CUT branch-frequency evidence diagnostic-only.  Even an all-smooth
  tiny-box sweep only justifies a later target-specific SVD-CUT HMC plan.
- Keep GPU/XLA optional and escalated.  CPU target and memory diagnostics must
  justify any matching-shape QR derivative GPU/XLA follow-up.

### Phase A: Lane And Worktree Audit

Actions:

- run `git status --short --branch`;
- classify v1-lane files and out-of-lane files;
- confirm no shared monograph, structural SVD/SGU, MacroFinance, or DSGE edit
  is required.

Primary criterion:

- execution can proceed with only BayesFilter v1-lane files.

Veto diagnostics:

- the next action requires editing external client code;
- out-of-lane dirty files must be staged to continue.

### Phase B: QR Derivative Memory-reduction Design

Actions:

- inspect `tf_qr_linear_gaussian_score_hessian` implementation;
- identify whether a score-only public/private helper already exists;
- design the smallest safe change:
  - expose score-only path only if already natural and clearly private; or
  - add diagnostic-only benchmark selectors without changing public API; or
  - document why no safe change is possible in this phase.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  -p no:cacheprovider
```

Primary criterion:

- derivative correctness tests remain green;
- no public API expansion unless justified by tests and docs.

### Phase C: QR Score/Hessian Diagnostic Artifact

Actions:

- run focused CPU benchmark rows for:
  - value-only;
  - QR value plus autodiff score for the HMC path;
  - QR value plus autodiff score/Hessian as a curvature diagnostic;
  - full analytic QR score/Hessian with score and Hessian materialized;
  - graph warmup with fixed shape;
- record JSON and Markdown artifacts under `docs/benchmarks`.

Primary criterion:

- artifact identifies the dominant cost path for the fixed first HMC target, or
  records why broader analytic score-only separation cannot be made safely in
  this pass.

### Phase D: First HMC Target Contract

Actions:

- create a small target fixture for `linear_qr_score_hessian_static_lgssm`;
- define parameter transform and priors;
- add value/score/Hessian finite diagnostics;
- add compiled/eager parity for the target function.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
```

Primary criterion:

- target value and gradient are finite on the fixed seed fixture;
- compiled/eager parity passes;
- Hessian symmetry and curvature diagnostics are recorded if Hessian is used.

### Phase E: First HMC Smoke

Actions:

- run a tiny fixed-seed TFP HMC or NUTS smoke on CPU;
- keep iteration count small;
- record acceptance, step size, finite state/log-prob/gradient status, and
  failure diagnostics.

Primary criterion:

- smoke finishes and records diagnostics, or fails with a reproducible blocker.

Non-goal:

- no convergence claim.
- no claim that analytic QR Hessian should be used in the sampler log-prob
  path.

### Phase F: SVD-CUT Branch-frequency Sweep

Actions:

- add a small parameter-box diagnostic for SVD-CUT smooth-branch derivatives;
- record branch labels and blocker counts;
- keep runtime in extended CPU tier.

Tests:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
```

Primary criterion:

- branch-frequency artifact exists;
- SVD-CUT HMC remains blocked unless smooth-branch dominance is proven and a
  later target-specific HMC plan is written.

### Phase G: GPU/XLA Decision Review

Actions:

- decide whether QR derivative XLA/GPU is justified now;
- if not, record deferral with CPU evidence;
- if yes, run only escalated matching-shape XLA/GPU tests and keep claims
  narrow.

Primary criterion:

- GPU/XLA claims match evidence exactly.

### Phase H: CI Tier And External-status Review

Actions:

- confirm new tests are marked/documented as fast local, focused local,
  extended CPU, optional external, escalated GPU, or HMC readiness;
- record MacroFinance status as optional evidence on dirty checkout;
- keep DSGE optional bridge work deferred.

Primary criterion:

- no extended/GPU/optional/HMC command becomes accidental fast CI.

### Phase I: Result, Reset Memo, Source Map, Commit

Actions:

- write a phase result artifact;
- update only the v1 lane reset memo;
- register new artifacts in `docs/source_map.yml`;
- run:

```bash
git diff --check
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_compiled_filter_parity_tf.py \
  -p no:cacheprovider
```

- stage only v1-lane files;
- run `git diff --cached --check`;
- commit when all primary criteria pass.

## Completion Definition

This phase is complete when:

- QR derivative memory cost has a separated diagnostic result and, if safe, a
  measured reduction;
- first HMC target contract exists and passes value/score/compiled diagnostics;
- first tiny HMC smoke exists or has a reproducible blocker;
- SVD-CUT branch-frequency artifact exists and HMC status is correctly
  retained or narrowed;
- GPU/XLA follow-up is justified or explicitly deferred;
- MacroFinance and DSGE remain external and read-only;
- CI tier impact is documented;
- result/reset/source-map artifacts are updated and committed as v1-lane files.

## Recommended First Move

Start with Phases A through C.  The QR derivative memory result is the most
concrete blocker and directly affects the first HMC target.  Do not start HMC
sampling until the target log-prob and gradient path is confirmed to avoid
unnecessary Hessian materialization in the sampler path.
