# Phase 3 Result: Lower-Triangular LGSSM Stationary Materialization

Date: 2026-07-08

## Decision

`PASS_PHASE3_STATIONARY_MATERIALIZATION`

Phase 3 implemented focused testing-lane helpers for materializing the Phase
1/2 lower-triangular LGSSM model and stationary covariance. Derivatives and
full target score correctness are deferred to Phase 4.

No GPU/CUDA command, NeuTra training, HMC sampling/tuning,
posterior/reference sampling, or `jit_compile=false` runtime run was performed.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can BayesFilter materialize the Phase 1/2 lower-triangular LGSSM model and stationary covariance without hidden nonstationarity or runtime autodiff? |
| Baseline/comparator | Phase 1 contract, Phase 2 fixture, Lyapunov equation residual, and existing local stationary utilities. |
| Primary criterion | Transform/shape/stationary residual tests pass under CPU-hidden local checks. |
| Veto diagnostics | Runtime autodiff, non-XLA-compatible operations in admitted route, nonfinite covariance, contract mismatch, or derivative mismatch if derivatives are implemented. |
| Result | Pass for stationary materialization; no score/HMC/NeuTra gate is passed. |

## Implementation

Added:

- `bayesfilter/testing/multidim_triangular_lgssm_tf.py`
- `tests/test_multidim_triangular_lgssm_tf.py`

The helper provides:

- Phase 1 contract loading;
- Phase 2 synthetic data loading;
- raw-truth reconstruction from contract transforms;
- raw-to-lower-triangular transition construction;
- diagonal positive `Q/R` construction;
- stationary covariance solve for `P = A P A' + Q`;
- `TFLinearGaussianStateSpace` materialization with `H = I_4`;
- Lyapunov residual evaluation.

This is intentionally under `bayesfilter/testing`; it is not a promoted public
BayesFilter API.

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

Result: `4 passed, 2 warnings`.

```text
rg -n "GradientTape|jacobian|batch_jacobian|jit_compile=False|jit_compile=false" \
  bayesfilter/testing/multidim_triangular_lgssm_tf.py
```

Result: no matches.

```text
git diff --check -- \
  bayesfilter/testing/multidim_triangular_lgssm_tf.py \
  tests/test_multidim_triangular_lgssm_tf.py \
  docs/plans/bayesfilter-multidim-triangular-lgssm-neutra-hmc-phase3-stationary-implementation-subplan-2026-07-08.md
```

Result: pass.

## Diagnostic Summary

| Diagnostic | Status |
| --- | --- |
| Raw truth reconstructs Phase 1 truth | Passed |
| `A` lower triangular and matches Phase 1 truth | Passed |
| `H = I_4` | Passed |
| `Q/R` diagonal positive | Passed |
| Phase 2 `P_inf` reproduced | Passed |
| Lyapunov residual | `<= 1e-14` in test |
| Stationary covariance positive definite | Passed |
| CPU-hidden `jit_compile=True` materialization smoke | Passed |
| Runtime helper source scan for `GradientTape`/`jacobian`/`batch_jacobian` | Passed |

## Review Notes

Same-foreground Codex substitute review of the Phase 3 subplan returned
`VERDICT: AGREE`. The reviewer suggested an optional explicit
posterior-correctness nonclaim; this result includes that boundary.

## Decision Table

| Decision | Primary Criterion Status | Veto Diagnostic Status | Main Uncertainty | Next Justified Action | Not Concluded |
| --- | --- | --- | --- | --- | --- |
| Admit stationary materialization helper for Phase 4 target work | Met | No Phase 3 veto triggered | Full value/score correctness is not implemented or checked yet | Refresh/review Phase 4 score/XLA subplan | No posterior correctness, HMC readiness, NeuTra usefulness, score correctness, global identifiability, product/default readiness, or scientific validity |

## Plain-Language Gate

Claimed target: materialize the Phase 1/2 lower-triangular LGSSM and stationary
initial covariance.

Computed quantity: TensorFlow helper plus focused tests for transforms,
stationary covariance, fixture reproduction, and CPU-hidden `jit_compile=True`
materialization.

Verdict: `correct` for stationary materialization under the named synthetic
fixture; `not checked` for log posterior score correctness, NeuTra, HMC,
posterior recovery, or scientific validity.

## Next Phase Handoff

Phase 4 may begin only after its subplan is refreshed and reviewed. Phase 4
must build the exact Kalman value/score adapter, preserve the no-runtime-
autodiff policy, run only `jit_compile=True` compile evidence, and write a
compile/score diagnostic artifact before any NeuTra or HMC phase.
