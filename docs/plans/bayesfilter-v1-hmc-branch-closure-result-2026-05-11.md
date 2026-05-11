# Result: BayesFilter v1 HMC And Branch-diagnostic Closure

## Date

2026-05-11

## Scope

This result closes the active v1 external-compatibility plan:

```text
docs/plans/bayesfilter-v1-hmc-branch-closure-plan-2026-05-11.md
```

Governing reset memo:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

## Executive Result

The phase completed without crossing the lane boundary.

Closed in this pass:
- QR first-order-vs-Hessian cost is now isolated on a fixed CPU diagnostic;
- the first target-specific linear QR HMC smoke runs and records finite
  diagnostics;
- SVD-CUT branch-frequency telemetry exists for a tiny smooth box and a
  weak-gap control;
- HMC and SVD-CUT diagnostic tests are opt-in rather than accidental fast CI;
- GPU/XLA QR derivative work is deferred with an explicit rationale.

Still not claimed:
- no HMC convergence claim;
- no production sampler claim;
- no SVD-CUT HMC promotion;
- no DSGE or MacroFinance switch-over;
- no QR derivative GPU/XLA claim.

## Phase Results

### Phase A: Plan And Lane Audit

Result:
- passed.

Actions:
- tightened the plan to avoid public score-only API expansion;
- wrote `docs/plans/bayesfilter-v1-hmc-branch-closure-plan-audit-2026-05-11.md`;
- confirmed out-of-lane dirty files must remain unstaged.

Interpretation:
- automatic execution was justified because no veto diagnostic was active.

### Phase B: QR Derivative Design

Result:
- passed.

Actions:
- added private first-order QR diagnostic helpers;
- kept the public v1 QR derivative API unchanged;
- kept analytic Hessian as diagnostic rather than sampler-path requirement.

Validation:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_v1_public_api.py \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  -p no:cacheprovider
12 passed, 2 warnings in 84.95s
```

Interpretation:
- the first-order helper is correct on the tiny fixture and remains private.

### Phase C: QR Cost Artifact

Result:
- passed.

Artifacts:
- `docs/benchmarks/bayesfilter-v1-qr-derivative-materialization-diagnostic-2026-05-11.json`;
- `docs/benchmarks/bayesfilter-v1-qr-derivative-materialization-diagnostic-2026-05-11.md`.

Observed CPU-only graph rows:

```text
linear_qr_score:
  warmup = 2.4352 s
  steady = 0.0025 s
  max RSS delta = 123.0 MB

linear_qr_score_hessian:
  warmup = 11.6858 s
  steady = 0.0048 s
  max RSS delta = 636.0 MB
```

Interpretation:
- H1 is supported: second-order/Hessian materialization dominates warmup and
  process-memory cost on the fixed first-target shape.
- H2 is supported for the diagnostic shape: first-order QR is materially
  cheaper than full score/Hessian.

### Phases D-E: First Linear QR HMC Target

Result:
- passed.

Artifacts:
- `bayesfilter/testing/tf_hmc_readiness.py`;
- `tests/test_hmc_linear_qr_readiness_tf.py`;
- `docs/benchmarks/bayesfilter-v1-linear-qr-hmc-readiness-smoke-2026-05-11.json`;
- `docs/benchmarks/bayesfilter-v1-linear-qr-hmc-readiness-smoke-2026-05-11.md`.

Validation:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
3 passed, 2 warnings in 39.00s
```

Observed:

```text
initial_target_log_prob = -1.3568111688046285
initial_gradient = [-0.40893740245060484, -1.0379645721254034]
value_residual = 0.0
score_residual = 4.44e-16
hessian_residual = 6.66e-16
hessian_symmetry_residual = 0.0
negative_hessian_eigenvalues = [2.3467420404247332, 2.998234138312041]
finite_sample_count = 12
nonfinite_sample_count = 0
acceptance_rate = 1.0
```

Interpretation:
- H3 is supported for the fixed target smoke.
- H4 is supported: Hessian remains useful for curvature diagnostics, but QR
  value plus autodiff score is enough for the tiny sampler path.
- This is not a convergence claim.

### Phase F: SVD-CUT Branch-frequency Diagnostic

Result:
- passed.

Artifacts:
- `bayesfilter/testing/tf_svd_cut_branch_diagnostics.py`;
- `tests/test_svd_cut_branch_diagnostics_tf.py`;
- `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-diagnostic-2026-05-11.json`;
- `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-diagnostic-2026-05-11.md`.

Validation:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
2 passed, 2 warnings in 11.31s
```

Observed:

```text
smooth box total = 3
smooth count = 3
active floor count = 0
weak spectral-gap count = 0
nonfinite count = 0
min placement eigen gap = 0.02189900512583248
max support residual = 0.0
max deterministic residual = 0.0

weak-gap control total = 2
weak-gap control weak-gap count = 2
```

Interpretation:
- H5 is not supported inside the deliberately tiny smooth box.
- H6 controls promotion: even all-smooth tiny-box evidence only justifies a
  future target-specific SVD-CUT HMC plan, not immediate HMC promotion.

### Phases G-H: GPU/XLA And CI Tiers

Result:
- passed by documented deferral and containment.

Actions:
- updated `docs/plans/bayesfilter-v1-ci-runtime-tier-policy-2026-05-11.md`;
- kept HMC and SVD-CUT diagnostic tests opt-in through environment variables;
- deferred QR derivative GPU/XLA because CPU diagnostics and CPU HMC smoke were
  sufficient for this phase.

Interpretation:
- H7 closes by deferral;
- H8 and H9 remain closed because MacroFinance and DSGE are external/read-only;
- H10 is strengthened by opt-in test gates and marker declarations.

## Remaining Gaps

1. Public score-only API decision:
   private first-order QR diagnostics are useful, but public promotion needs a
   separate API-freeze review, masked-observation design, docs, and broader
   tests.

2. Larger HMC target diagnostics:
   the first target has only a tiny fixed-seed smoke.  It needs longer chain
   diagnostics, adaptation policy, multi-chain summaries, and failure-mode
   reporting before any readiness claim beyond smoke.

3. SVD-CUT HMC:
   branch-frequency telemetry exists, but SVD-CUT HMC remains blocked pending a
   separate target-specific plan.

4. QR derivative GPU/XLA:
   not tested in this phase.  A follow-up should use escalated permissions and
   matching shapes only after choosing a worthwhile CPU shape.

5. Optional external certification:
   MacroFinance live compatibility remains useful evidence on an observed dirty
   checkout, not clean release certification.

6. DSGE bridges:
   Rotemberg and EZ remain optional/design-only; SGU remains blocked outside
   this v1 local HMC phase.

## Suggested Next Hypotheses

H11:
The private first-order QR path can become a public score-only API without
breaking v1 stability if masked-observation semantics, metadata, and parity
tests are added.

Test:
- add masked score-only design;
- compare dense/masked score-only against full score/Hessian and autodiff;
- rerun public API freeze review.

H12:
The first linear QR HMC target remains finite under longer multi-chain CPU
smoke diagnostics.

Test:
- add an extended artifact with multiple chains, more draws, acceptance
  summaries, energy/log-prob ranges, and explicit no-convergence or convergence
  labels.

H13:
QR derivative GPU/XLA is only worthwhile for larger fixed shapes than the tiny
HMC fixture.

Test:
- choose one medium CPU shape from the QR derivative ladder;
- run escalated matching-shape GPU-visible and XLA-visible diagnostics;
- compare against CPU-only rows with the same shape.

H14:
SVD-CUT branch behavior remains smooth only in narrow boxes and becomes blocked
under wider target regions.

Test:
- sweep a wider economically relevant parameter region;
- record active floors, weak gaps, support/deterministic residuals, and parity;
- write a separate SVD-CUT HMC plan only if smooth dominance is robust.

## Final Status

The phase is complete and ready for scoped validation and commit.
