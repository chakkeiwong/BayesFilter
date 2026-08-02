# SVX-ZC UKF Initializer Default And Admission Rerun

Date: 2026-08-01

## Research Intent

Question: does the existing repository UKF initializer improve the numerical
fixed-branch SVX-ZC admission fit when it is the default warm start?

Candidate: the monograph-authority fixed adjacent-state squared-TT route
(`zhao_cui_fixed_adjacent_state_squared_tt_v1`) with the existing P76 UKF
square-root initializer.  The scalar-SV UKF moments come from the existing
augmented-noise Gaussian-closure UKF route and are geometry-only warm-start
information; they are not an exact transformed-SV filtering target.

Baseline: the valid `attempt03` route, which uses
`orthonormal_mode_diagonal_norm_balanced_v1` initial cores, with the same model,
data, horizon, degree, quadrature order, coordinate map, fitter, ranks, seeds,
and gates.

## Evidence Contract

Primary question: whether UKF-initialized cores lower the fixed-fit residual or
otherwise improve the declared admission diagnostics without changing the
finite target program.

Promotion criterion: all existing hard gates pass, including finite value and
score, coordinate/Jacobian consistency, positivity, retained marginal closure,
condition ceiling, same-scalar finite-difference branch identity, no retained
tensor-product route, and rank-saturation residual `<= 1e-8`.

Hard vetoes: nonfinite quantities, failed structural invariants, condition
number above `1e10`, failed same-scalar FD check, forbidden route use, or fit
residual above `1e-8`.  The residual threshold is not relaxed for this test.

Explanatory diagnostics: initial and adjacent fit residuals by time, dense
value gap per observation, condition number, positivity, closure, and UKF
center/covariance/initializer hashes.  These diagnose the effect but do not
establish exact filtering, posterior correctness, HMC readiness, or statistical
superiority.

Nonclaims: the UKF scout is not truth; the scalar UKF closure is not the exact
transformed-SV target; this run is not author-source reproduction, NeuTra
training evidence, HMC evidence, or a default-readiness claim.

## Implementation Scope

1. Audit and repair the existing P76 initializer so its UKF center and local
   linear map are consumed by the projected guide rather than only recorded.
2. Add a narrow moments gateway so one-axis and adjacent two-axis SVX-ZC seeds
   reuse `p76_build_ukf_initializer` machinery without duplicating it.
3. Adapt the existing scalar-SV UKF output into the repository UKF scout result
   contract, explicitly labeling its geometry-only target mismatch.
4. Make UKF cores the SVX-ZC default. Preserve norm-balanced cores only as an
   explicit comparator/debug option, not the default.
5. Add focused tests for frame consumption, default identity, core use, and
   finite branch behavior.

## Defaults And Assumptions Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| UKF initializer | Owner request and existing P76 implementation | Requested default policy and reusable code already exists | UKF closure may be a poor geometry guide | UKF moments finite; core hash differs; residual comparison | reviewed default for this route |
| Scalar UKF moments | Existing `actual_transformed_sv_independent_panel_augmented_noise_ukf_filter` | Only existing scalar UKF path; avoids duplicate implementation | It is augmented-noise closure, not exact transformed-SV | manifest target mismatch and unchanged exact target code | geometry-only warm-start hypothesis |
| Fixed coordinate map | Valid `attempt03` comparator | Isolates initializer effect | UKF frame may not match global map | record map unchanged and compare residuals | frozen baseline |
| Rank ladder `(1,2,4,6)` | `attempt03` | Same bounded admission ladder | No rank may pass | preserve rank-level residuals | frozen baseline |
| Residual veto `1e-8` | Existing admission contract | Prevents post-result threshold relaxation | all candidates may remain blocked | report absolute residuals | hard veto |

## Pre-Mortem And Stop Conditions

The run could appear improved while still being invalid if the UKF fields are
metadata-only, if the scalar UKF target is silently treated as exact, or if the
target/config changes alongside initialization. Focused tests and manifest
hashes distinguish those cases. It may fail because the warm start is poor,
because the fixed rank/basis is insufficient, or because the adapter exposes a
shape/conditioning bug; these are repair triggers, not evidence against the
monograph route.

Stop the ladder on a harness/target/artifact invalidity, missing diagnostics,
nonfinite output, or exhausted bounded attempt. A candidate failing a hard
admission gate does not stop the research direction; it remains blocked and
the residual is reported unchanged.

## Commands And Budget

Focused checks:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 python -m pytest -q \
  tests/highdim/test_p76_ukf_initializer.py \
  tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py \
  tests/highdim/test_svx_zc_ukf_initializer.py
```

Fresh CPU admission attempt (never overwrite prior attempts):

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 python \
  docs/benchmarks/run_neutra_svx_zc_monograph_admission_20260731.py \
  --output-root docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt04
```

The bounded ladder is ranks `(1,2,4,6)` at `T=10`, degree `8`, quadrature
order `17`, with one fresh output root. No GPU, package mutation, or budget
increase is authorized by this plan.

Attempt `04` is retained but invalid: the comparator used UKF cores while the
runner's structural recomputation used norm-balanced cores, and the first
implementation also propagated a nonbaseline density floor. It is not
evidence. Attempt `05` repaired both execution mismatches and is numerically
valid, but was emitted before this plan path was wired into the run manifest;
it remains diagnostic provenance. The terminal rerun is `attempt06` after the
manifest-only repair.

## Artifact And Decision Rules

The runner must record the initializer identity, UKF target/nonclaim labels,
UKF moment summaries/hashes, initial and adjacent core hashes, git commit,
command, CPU choice, seeds, data hash, plan path, wall time, and all rank-level
gates under `attempt04/`.

If a rank passes, only the numerical admission blocker is removed; no NeuTra
training or HMC admission follows automatically. If no rank passes, preserve
`SVX-ZC` as numerically blocked and use the residual comparison to choose the
smallest next fixed-branch repair. No ranking claim is made from descriptive
residual or value-gap differences alone.

## Skeptical Audit Record

Audit passed on 2026-08-01 after identifying and including the required frame
consumption repair and the one-axis/adjacent moments gateway. The plan keeps the
exact target, comparator, gates, nonclaims, and budget fixed, and treats the
scalar UKF closure as geometry-only rather than as same-target evidence.
