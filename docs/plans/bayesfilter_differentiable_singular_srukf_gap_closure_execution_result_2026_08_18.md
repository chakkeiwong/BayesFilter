# Differentiable Singular SR-UKF Gap-Closure Execution Result

Date: 2026-08-18  
Plan: `bayesfilter_differentiable_singular_srukf_gap_closure_plan_2026_08_18.md`  
Plan review: `bayesfilter_differentiable_singular_srukf_gap_closure_plan_review_2026_08_18.md`  
Canonical route: `direct_qr_block_conditional`  
Scope: direct-factor SR-UKF; SSL-LSTM remains excluded.

## Outcome

The reviewed bounded plan was executed. The repository now contains a
score-bearing fixed-rank, fixed-pivot rectangular QR route for singular
covariance supports, while rank discovery remains explicitly value-only. The
full-rank canonical direct-factor route retains direct block QR and does not
form a covariance matrix or decompose a covariance at each time step.

The result is suitable for final mathematical and release audit within the
declared fixed-rank/fixed-chart scope. It does not claim a differentiable score
through rank changes, support changes, pivot/chart changes, QR sign changes,
signed-weight PSD boundaries, HMC convergence, or exact nonlinear Bayesian
inference.

## Implemented closure

| Gap | Closure | Evidence |
|---|---|---|
| Positive pivot policy | Scale-aware `1e-12` default and explicit branch diagnostics | `bayesfilter/linear/block_qr_conditional_tf.py`, `bayesfilter/nonlinear/rectangular_srukf_tf.py` |
| Rectangular factor derivatives | Fixed-chart thin QR, factor and derivative propagation | `bayesfilter/linear/rectangular_factor_tf.py` |
| Singular support likelihood | QR support density and derivative, with on/off-support status | `batched_fixed_support_qr_likelihood` and focused tests |
| Singular conditional update | Fixed-support gain, mean increment, rectangular posterior factor, and derivatives | `batched_fixed_support_qr_update` and rank-two authority |
| Temporal score route | Static-shape/XLA fixed-rank recursion with generated branch identity | `bayesfilter/nonlinear/rectangular_srukf_tf.py` |
| Hidden spectral fallback | Route guard and source call-graph test | `tests/test_factor_srukf_route_guard.py` |
| Documentation gap | Canonical chapters and Fable mirror synchronized | Chapters 12, 14, 17, 18, and 23; both `main.tex` files include them |

## Mathematical defect found and repaired

Rank-one checks were insufficient to establish matrix orientation. The
non-diagonal rank-two authority found three real errors: an `R^{-T}` solve was
used where the gain requires `R^{-1}`, the retained-column projector contracted
different rank indices, and two support-coordinate derivatives used independent
observation indices rather than the required dot products. These were repaired.
The corrected rank-two gain, posterior covariance represented by the factor,
mean increment, likelihood, and derivatives agree with the dense covariance
authority and centered finite differences to the declared tolerances.

## Verification

Focused post-edit command:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true pytest -q \
  tests/test_rectangular_factor_tf.py \
  tests/test_block_qr_conditional_tf.py \
  tests/test_rectangular_srukf_tf.py \
  tests/test_factor_srukf_route_guard.py
```

Result: **25 passed, 1 warning**.

Terminal direct-factor/model regression campaign (the already-running
repository suite, including prior model/adaptor coverage and the new singular
rank-two tests): **143 passed, 3 warnings** in 420.77 seconds. The warnings
are third-party environment warnings from `h5py`/TensorFlow Probability and
are not test failures.

Coverage includes:

- rank-one and non-diagonal rank-two QR reconstruction and derivatives;
- dense covariance authority for gain, posterior factor, and mean update;
- renormalized `P_epsilon = G G' + epsilon I` limit;
- on-support and off-support likelihood behavior;
- rank-zero value-only discovery;
- anisotropic near-rank-change/chart rejection;
- malformed and duplicate permutations;
- nonfinite input rejection;
- eager/XLA CPU parity;
- temporal fixed-rank score finite differences;
- route guard against SVD/eigendecomposition/covariance-to-factor calls in the
  admitted score call graph; and
- existing direct-factor/model regression and historical comparison tests.

`git diff --check` passes. The canonical and Fable-mirror versions of all five
changed chapters are byte-identical, and `docs/main.tex` and
`docs/fable-rewrite/monograph/main.tex` both include those chapters.

## GPU and document-build status

The host exposes four RTX 4090 devices. The prescribed gate is GPU `3`, then
`2`, `1`, `0`, with memory growth configured before TensorFlow initialization.
The release-gate command was attempted with GPU `3`, but the required
escalated execution authorization failed at the approval service with HTTP
502. The gate therefore produced no GPU artifact and no GPU claim is made.
The reproducible gate remains at
`scripts/run_fixed_rank_srukf_gpu_gate_20260818.py`; it refuses to overwrite a
prior artifact root and records GPU selection, memory growth, XLA, allocator
telemetry, CPU parity, and checksums when authorized.

The changed chapters were traversed by `pdflatex`. Final PDF emission remains
blocked by the pre-existing TeX installation error
`pdfTeX error: Font tcrm1200 at 600 not found`; `algorithm.sty` was also absent
and was only worked around with temporary read-only local style links. No TeX
package or environment was installed or changed. This is a documentation
environment blocker, not a mathematical or code-test failure.

## Release nonclaims and follow-up gates

The implementation is not evidence for a global smooth singular-Gaussian
score. A caller must preflight and freeze rank, row permutation, QR signs,
support chart, tolerances, and observation branch. A branch event invalidates
the score and must be handled as a value-only or rejected result. The final
release audit should also inspect the GPU gate once trusted execution is
available and resolve the TeX font-cache issue before publishing a PDF.
