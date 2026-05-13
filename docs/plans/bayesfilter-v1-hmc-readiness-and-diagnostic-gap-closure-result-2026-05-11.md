# Result: BayesFilter v1 HMC Readiness And Diagnostic Gap Closure

## Date

2026-05-11

## Scope

This result closes:

```text
docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-plan-2026-05-11.md
```

It updates the lane reset memo:

```text
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```

## Executive Result

The phase completed inside the BayesFilter v1 lane.

Closed:
- private dense QR score-only helpers were added and tested for diagnostics;
- score-only QR cost is now separated from full score/Hessian cost;
- a small state/observation score-envelope ladder completed;
- the first target-specific linear QR HMC smoke completed with finite
  diagnostics;
- SVD-CUT branch-frequency telemetry exists for a tiny smooth box;
- HMC and SVD-CUT diagnostic tests are opt-in;
- GPU/XLA, MacroFinance switch-over, DSGE switch-over, and SVD-CUT HMC remain
  deliberately unclaimed.

## Phase Results

### Phase 0: Lane And Worktree Audit

Result:
- passed.

Interpretation:
- all required edits stayed under `bayesfilter`, `tests`, `docs/benchmarks`,
  `docs/plans/bayesfilter-v1-*`, `pytest.ini`, and `docs/source_map.yml`;
- shared monograph, structural SVD/SGU, Chapter 18/18b, MacroFinance, DSGE,
  sidecar, and local image files remained out of lane.

### Phase 1: Plan Tightening And Audit

Result:
- passed.

Artifact:
- `docs/plans/bayesfilter-v1-hmc-readiness-and-diagnostic-gap-closure-plan-audit-2026-05-11.md`.

Interpretation:
- private dense QR score-only diagnostics are allowed for this phase;
- public score-only API promotion remains blocked;
- full QR score/Hessian remains a parity and curvature diagnostic;
- SVD-CUT branch evidence remains diagnostic-only.

### Phase 2: Private Dense QR Score-only Diagnostic Path

Result:
- passed.

Implementation:
- `_tf_qr_sqrt_kalman_score`;
- `_tf_qr_linear_gaussian_score`;
- first-order QR/Cholesky/covariance factor helpers.

Contract:
- dense observations only;
- diagnostic helper only;
- no public v1 score-only API was promoted.

Validation:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/test_linear_kalman_qr_derivatives_tf.py \
  tests/test_hmc_linear_qr_readiness_tf.py \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  tests/test_v1_public_api.py \
  -p no:cacheprovider
12 passed, 5 skipped, 2 warnings in 76.12s
```

Interpretation:
- score-only matches full score/Hessian and autodiff on the tiny dense fixture;
- public API expansion remains blocked until a separate API-freeze review.

### Phase 3: QR Derivative Cost Decomposition

Result:
- passed.

Artifact:
- `docs/benchmarks/bayesfilter-v1-qr-derivative-cost-decomposition-2026-05-11.*`.

Observed CPU-only graph rows:

```text
linear_qr_score:
  warmup about 2.44 s
  steady about 0.0024 s
  max RSS delta about 123.5 MB

linear_qr_score_hessian:
  warmup about 11.44 s
  steady about 0.0043 s
  max RSS delta about 634.2 MB
```

Interpretation:
- H3 is supported at the first target shape: Hessian materialization dominates
  graph warmup and process-memory cost;
- the score-only diagnostic path identifies why Hessian should stay out of the
  sampler path.

### Phase 4: State/Observation Score-envelope Ladder

Result:
- passed.

Artifact:
- `docs/benchmarks/bayesfilter-v1-qr-score-state-observation-ladder-2026-05-11.*`.

Observed:
- rows `(T,n,m,p) = (8,2,2,2), (8,3,2,2), (8,4,3,2)` all completed;
- warmup ranged from about `2.7s` to `4.1s`;
- max-RSS deltas stayed at or below about `221 MB`.

Interpretation:
- H4/H5 are supported for the small score-only envelope;
- this is a development envelope, not a release performance target.

### Phase 5: First QR HMC Smoke

Result:
- passed.

Artifact:
- `docs/benchmarks/bayesfilter-v1-qr-hmc-smoke-2026-05-11.*`.

Target:
- `linear_qr_score_hessian_static_lgssm`.

Gradient path:
- QR value plus TensorFlow autodiff score.

Observed:

```text
finite_sample_count = 12
nonfinite_sample_count = 0
acceptance_rate = 1.0
initial_gradient_finite = true
negative_hessian_eigenvalues about [2.3467, 2.9982]
```

Validation:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_HMC_READINESS=1 pytest -q \
  tests/test_hmc_linear_qr_readiness_tf.py \
  -p no:cacheprovider
3 passed, 2 warnings in 41.82s
```

Interpretation:
- H1/H2 are supported for this fixed target smoke;
- this is not a convergence claim, production sampler claim, MacroFinance/DSGE
  claim, GPU/XLA claim, or SVD-CUT HMC claim.

### Phase 6: SVD-CUT Branch-frequency Diagnostic

Result:
- passed.

Artifact:
- `docs/benchmarks/bayesfilter-v1-svd-cut-branch-frequency-2026-05-11.*`.

Observed:

```text
smooth box total = 3
smooth count = 3
active floor count = 0
weak spectral-gap count = 0
nonfinite count = 0
max support residual = 0.0
max deterministic residual = 0.0
status = diagnostic_smooth_box
```

Validation:

```text
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=-1 \
BAYESFILTER_RUN_EXTENDED_CPU=1 pytest -q \
  tests/test_svd_cut_branch_diagnostics_tf.py \
  -p no:cacheprovider
2 passed, 2 warnings in 11.53s
```

Interpretation:
- H6 is only partially supported: this tiny box is smooth;
- SVD-CUT HMC remains blocked because this is not broad target-region or
  sampler evidence.

### Phase 7: CI Tier Containment And GPU/XLA Deferral

Result:
- passed.

Actions:
- added pytest markers `extended`, `hmc`, `external`, and `gpu`;
- HMC tests require `BAYESFILTER_RUN_HMC_READINESS=1`;
- SVD-CUT branch tests require `BAYESFILTER_RUN_EXTENDED_CPU=1`;
- QR derivative GPU/XLA tests were deferred.

Interpretation:
- H7 closes by deferral: CPU evidence was sufficient for this phase, and any
  GPU/CUDA derivative work must use escalated device visibility;
- H8 remains closed because external projects stayed read-only.

## Remaining Gaps

1. Masked score-only QR:
   dense score-only remains private/diagnostic.  Public dense and masked
   score-only semantics and tests remain a separate API-freeze item.

2. Larger HMC diagnostics:
   the first target has a tiny smoke only.  Longer multi-chain CPU diagnostics,
   adaptation summaries, and failure-mode reporting are still needed.

3. SVD-CUT HMC:
   branch-frequency telemetry exists, but SVD-CUT HMC remains blocked pending
   wider branch coverage and target-specific sampler evidence.

4. QR derivative GPU/XLA:
   not tested in this phase.  A follow-up should use escalated GPU visibility
   and matching CPU/GPU shapes.

5. External certification:
   MacroFinance and DSGE remain read-only optional compatibility targets, not
   switch-over targets.

## Suggested Next Hypotheses

H11:
Public QR score-only can be added without destabilizing v1 if dense and masked
semantics, dummy-row behavior, and score parity are specified first.

H12:
The first linear QR HMC target remains finite under longer multi-chain CPU
smoke diagnostics with adaptation summaries.

H13:
QR derivative GPU/XLA is worthwhile only at medium shapes selected from CPU
evidence.

H14:
SVD-CUT smooth-branch behavior is fragile outside tiny smooth boxes and should
be tested on wider economically relevant parameter regions.

H15:
MacroFinance/DSGE integration should wait until BayesFilter v1 has local
score-only, HMC, and branch diagnostics stabilized.
