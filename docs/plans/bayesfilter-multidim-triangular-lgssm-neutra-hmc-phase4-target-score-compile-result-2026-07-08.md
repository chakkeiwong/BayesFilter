# Phase 4 Result: Lower-Triangular LGSSM Target Score And XLA Compile

Date: 2026-07-08

## Decision

`PASS_PHASE4_VALUE_SCORE_COMPILE_DIAGNOSTIC`

Phase 4 implemented a testing-lane exact Kalman log posterior and manual
first-order score path for the Phase 1/2 lower-triangular LGSSM target, then
ran a CPU-hidden `jit_compile=True` compile diagnostic.

No `jit_compile=false` runtime run, GPU/CUDA command, NeuTra training, HMC
sampling/tuning, posterior/reference sampling, product/default promotion, or
scientific promotion was performed.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the target adapter compute the declared lower-triangular LGSSM log posterior and score in an XLA-compatible path? |
| Baseline/comparator | Phase 1-3 contract/artifacts, finite-difference tests, exact Kalman likelihood. |
| Primary criterion | Finite value/score and compile diagnostic with `jit_compile=True`; score residuals within tolerance. |
| Veto diagnostics | Runtime `GradientTape`, `jacobian`, `batch_jacobian`, `jit_compile=false`, score mismatch, nonfinite values, target/signature mismatch. |
| Result | Pass for value/score compile diagnostic; no HMC/NeuTra/posterior convergence gate is passed. |

## Implementation

Updated:

- `bayesfilter/testing/multidim_triangular_lgssm_tf.py`
- `tests/test_multidim_triangular_lgssm_tf.py`

The testing helper now provides:

- manual first derivatives of the lower-triangular transition matrix;
- manual first derivatives of diagonal `Q/R`;
- manual discrete Lyapunov derivative solve for `dP_inf`;
- a `TFLinearGaussianStateSpaceFirstDerivatives` payload;
- exact QR Kalman likelihood/score evaluation through the existing
  `tf_qr_linear_gaussian_score` backend;
- independent Gaussian prior value/score on raw coordinates;
- log posterior value/score for the Phase 1/2 fixture.

No runtime `GradientTape`, `jacobian`, or `batch_jacobian` is used in the
admitted helper.

## Compile Diagnostic Artifact

`docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_phase4_value_score_compile_diagnostic_seed20260708.json`

Key fields:

| Field | Value |
| --- | --- |
| Decision | `PASS_PHASE4_VALUE_SCORE_COMPILE_DIAGNOSTIC` |
| `jit_compile` | `true` |
| `jit_compile_false_runtime_executed` | `false` |
| CPU hidden | `true`, `CUDA_VISIBLE_DEVICES=-1` |
| GPU executed | `false` |
| HMC executed | `false` |
| Training executed | `false` |
| Posterior sampling executed | `false` |
| Log posterior value at truth | `-89.01243443204292` |
| Max abs posterior score at truth | `17.97431931889728` |
| First call wall seconds | `6.477313329000026` |
| Second call wall seconds | `0.22245534602552652` |
| Compile-time proxy seconds | `6.2548579829744995` |
| Graph node count | `277` |
| GraphDef bytes | `784061` |
| Artifact payload hash | `sha256:cc39fe258aef6d93e3681d1b822aa9c5a2811f4516eee0467804dc418e4c5c22` |

The compile diagnostic emitted TensorFlow CUDA initialization warnings even
with `CUDA_VISIBLE_DEVICES=-1`. This is not GPU execution evidence; the
artifact records CPU-hidden execution and `gpu_executed=false`.

## Checks Run

```text
CUDA_VISIBLE_DEVICES=-1 python -m py_compile \
  bayesfilter/testing/multidim_triangular_lgssm_tf.py \
  tests/test_multidim_triangular_lgssm_tf.py
```

Result: pass.

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_multidim_triangular_lgssm_tf.py
```

Result: `7 passed, 2 warnings`.

```text
rg -n "GradientTape|jacobian|batch_jacobian|jit_compile=False|jit_compile=false" \
  bayesfilter/testing/multidim_triangular_lgssm_tf.py
```

Result: no matches.

```text
python -m json.tool \
  docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_phase4_value_score_compile_diagnostic_seed20260708.json
```

Result: pass.

```text
git diff --check -- \
  bayesfilter/testing/multidim_triangular_lgssm_tf.py \
  tests/test_multidim_triangular_lgssm_tf.py \
  docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase4-target-score-compile-subplan-2026-07-08.md \
  docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_phase4_value_score_compile_diagnostic_seed20260708.json
```

Result: pass.

## Diagnostic Summary

| Diagnostic | Status |
| --- | --- |
| Manual derivative blocks vs finite difference | Passed |
| Log posterior score vs finite difference | Passed |
| CPU-hidden `jit_compile=True` materialization | Passed |
| CPU-hidden `jit_compile=True` value/score | Passed |
| Runtime helper source scan | Passed |
| Compile diagnostic JSON | Passed |

The finite-difference diagnostics are tests of local score correctness; they
are not runtime score implementation, posterior convergence evidence, or HMC
readiness evidence.

## Review Notes

Same-foreground Codex substitute review of the refreshed Phase 4 subplan
returned `VERDICT: AGREE`.

## Decision Table

| Decision | Primary Criterion Status | Veto Diagnostic Status | Main Uncertainty | Next Justified Action | Not Concluded |
| --- | --- | --- | --- | --- | --- |
| Admit value/score target for Phase 5 reference posterior planning | Met | No Phase 4 veto triggered | Posterior recoverability and sampler behavior remain untested | Refresh/review Phase 5 reference-posterior subplan | No posterior correctness, HMC readiness, NeuTra usefulness, global identifiability, product/default readiness, or scientific validity |

## Plain-Language Gate

Claimed target: exact Kalman log posterior and first-order score for the named
lower-triangular LGSSM fixture.

Computed quantity: TensorFlow QR Kalman value/score with manual first
derivatives, finite-difference tests, and CPU-hidden `jit_compile=True`
compile diagnostic.

Verdict: `correct` for the tested value/score path on the named fixture;
`not checked` for posterior convergence, NeuTra training, HMC sampling, GPU
compile, product/default readiness, or scientific validity.

## Next Phase Handoff

Phase 5 may begin only after its subplan is refreshed and reviewed. Phase 5
must define a reference posterior route or a blocker. It must not treat Phase 4
compile success or finite-difference score checks as HMC convergence or
posterior recoverability evidence.
