# Phase 3 Result: Analytical Score Dtype Cleanup

Date: 2026-07-09

## Status

`PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

## Phase Objective

Make public QR analytical score, score/Hessian helper paths, derivative payload
containers, and derivative result envelopes preserve explicit `float32` and
`float64` dtype.

## Skeptical Execution Audit

| Risk | Finding |
| --- | --- |
| Wrong baseline | Baseline stayed on existing FP64 analytical-score behavior plus FP32 autodiff/value references on small fixtures. |
| Proxy metric misuse | Tests assert observed output dtype, diagnostics dtype, derivative container dtype, and CPU/XLA execution, not only fixture dtype. |
| Missing stop condition | The subplan stop conditions covered derivative container breakage, unclear FP32 parity failure, and hidden FP64 coercion. |
| Environment mismatch | Checks were CPU-hidden debug/reference checks and do not support GPU/default-readiness claims. |
| Artifact mismatch | This result records commands, repairs, review, and Phase 4 handoff before benchmark harness execution. |

Audit status: `PASSED_FOR_PHASE_3_CPU_HIDDEN_DTYPE_CHECKS`.

## Source Changes

- Migrated analytical QR score and score/Hessian helper conversions in
  `bayesfilter/linear/kalman_qr_derivatives_tf.py` to infer and preserve one
  supported floating dtype.
- Preserved dtype for derivative payload containers in
  `bayesfilter/linear/types_tf.py`, including zero second-derivative tensors
  produced from first-derivative payloads.
- Preserved one shared dtype across `log_likelihood`, `score`, and optional
  `hessian` in `TFFilterDerivativeResult` in `bayesfilter/results_tf.py`.
- Added FP32/FP64 derivative container, public wrapper, masked score/Hessian,
  and CPU/XLA dynamic-score smoke tests in
  `tests/test_linear_qr_dtype_contracts.py` and
  `tests/test_linear_kalman_qr_derivatives_tf.py`.

## Checks Run

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py
```

Result: `12 passed`.

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py tests/test_linear_kalman_qr_derivatives_tf.py
```

Result: `29 passed`.

```bash
git diff --check -- bayesfilter/linear bayesfilter/results_tf.py tests docs/plans docs/reviews
```

Result: passed.

```bash
python -m py_compile bayesfilter/linear/kalman_qr_derivatives_tf.py bayesfilter/linear/types_tf.py bayesfilter/results_tf.py
```

Result: passed.

## Repair Log

| Issue | Status |
| --- | --- |
| Phase 3 subplan referenced a nonexistent derivative test file path. | Repaired to `tests/test_linear_kalman_qr_derivatives_tf.py` before execution. |
| Analytical score helpers and wrappers still created FP64 tensors/constants internally. | Repaired by threading inferred dtype through observations, constants, identities, jitter, accumulators, masks, and diagnostics. |
| Derivative payload containers and result envelopes still had dtype-preservation gaps. | Repaired and covered with direct dtype contract tests. |
| One historical `_diagnostics(..., dtype=tf.float64)` fallback remains. | Non-blocking for current wrappers because they pass dtype explicitly; future call sites should continue doing so. |

## Review Record

Claude review remains unavailable for this run unless the user explicitly
approves the external disclosure risk.  A weaker bounded Codex substitute
review was used.

| Round | Reviewer | Verdict | Finding |
| --- | --- | --- | --- |
| 1 | Ramanujan | `AGREE` | No blocking findings. Residual risks are limited to CPU-hidden scope, indirect coverage of QR helper internals, and the historical diagnostics fallback default. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Analytical QR score preserves explicit FP32/FP64 dtype under the scoped tests. |
| Baseline/comparator | Existing FP64 analytical score behavior remains covered; FP32 analytical score/Hessian matches FP32 autodiff references on small fixtures. |
| Primary criterion | Passed: observed FP32 outputs and diagnostics for public, dense, masked, and CPU/XLA score paths. |
| Veto diagnostics | No active Phase 3 veto: scoped parity, dtype, finite-output, and CPU/XLA checks passed. |
| Explanatory diagnostics | TensorFlow emitted existing `gast` warnings during pytest; no check failed after repair. |
| Not concluded | Benchmark dtype controls, batch-native analytical score, runtime superiority, GPU/default readiness, HMC readiness, posterior correctness, or scientific validity. |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Advance to Phase 4 | Passed scoped CPU-hidden analytical-score dtype checks | No Phase 3 veto active | Benchmark harness still needs requested/observed dtype fail-closed behavior | Execute Phase 4 benchmark dtype controls | Benchmark readiness beyond smoke, GPU evidence, batching, and speed claims |

## Phase 4 Handoff

Phase 4 may start with the existing subplan:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase4-benchmark-dtype-controls-subplan-2026-07-09.md`

Known handoff items:

- Benchmark dtype control must be explicit in the CLI and artifact manifest.
- Requested dtype and observed value/score dtype must be recorded for both
  analytical and autodiff arms.
- CPU-hidden smoke artifacts are permitted in Phase 4; GPU benchmarking remains
  deferred unless separately approved under a reviewed phase.
