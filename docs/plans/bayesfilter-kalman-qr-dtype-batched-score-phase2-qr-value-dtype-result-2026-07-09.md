# Phase 2 Result: QR Value Dtype Cleanup

Date: 2026-07-09

## Status

`PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

## Phase Objective

Make scalar, while-loop, batched-static, filtered-reference, masked, and public
dispatcher QR value paths preserve explicit `float32`/`float64` dtype.

## Skeptical Execution Audit

| Risk | Finding |
| --- | --- |
| Wrong baseline | Baseline remained current FP64 QR value behavior and scalar/batched parity tests. |
| Proxy metric misuse | Tests assert observed output dtype, not just fixture dtype. |
| Missing stop condition | First test run exposed shared `TFLinearGaussianStateSpace` hard-coercion; repair stayed within dtype contract. |
| Environment mismatch | Checks were CPU-hidden debug/reference checks and do not support GPU/default-readiness claims. |
| Artifact mismatch | This result records commands, failures, repairs, review rounds, and Phase 3 handoff. |

Audit status: `PASSED_FOR_PHASE_2_CPU_HIDDEN_DTYPE_CHECKS`.

## Source Changes

- Added dtype inference/conversion through QR value helpers in
  `bayesfilter/linear/kalman_qr_tf.py`.
- Preserved dtype in QR factor helpers touched by value paths in
  `bayesfilter/linear/qr_factor_tf.py`.
- Made `TFLinearGaussianStateSpace` preserve one explicit shared floating dtype
  for model tensors in `bayesfilter/linear/types_tf.py`.
- Made `TFFilterValueResult` preserve value/filter output dtype and tightened
  `TFFilterDerivativeResult` to use one shared dtype for value, score, and
  optional Hessian in `bayesfilter/results_tf.py`.
- Added FP32/FP64 value-path and container tests in
  `tests/test_linear_qr_dtype_contracts.py` and
  `tests/test_linear_qr_compact_loglik_tf.py`.

## Checks Run

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_dtype_contracts.py tests/test_linear_qr_compact_loglik_tf.py
```

Result: `26 passed`.

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_factor_tf.py
```

Result: `4 passed`.

```bash
git diff --check -- bayesfilter/linear bayesfilter/results_tf.py tests docs/plans docs/reviews
```

Result: passed.

## Repair Log

| Issue | Status |
| --- | --- |
| QR value kernels still called `_to_tensor(...)` without dtype and emitted FP64 constants. | Repaired by inferring one dtype per kernel and threading it through tensors, identities, constants, casts, jitter, and masks. |
| FP32 model fixtures failed because `TFLinearGaussianStateSpace` coerced tensors to FP64. | Repaired by shared dtype inference in the model container. |
| Public dispatcher/value envelope failed on FP32 because `TFFilterValueResult` coerced `log_likelihood` to FP64. | Repaired by dtype-preserving result conversion. |
| Phase 3 subplan omitted derivative payload/result containers. | Repaired after review Round 1 by expanding Phase 3 scope. |

## Review Record

Claude review remains unavailable for this run unless the user explicitly
approves the external disclosure risk.  A weaker bounded Codex substitute
review was used.

| Round | Reviewer | Verdict | Finding |
| --- | --- | --- | --- |
| 1 | Nietzsche | `REVISE` | Phase 2 value evidence was acceptable, but Phase 3 handoff omitted derivative payload containers in `types_tf.py` and derivative result behavior in `results_tf.py`. |
| 2 | Nietzsche | `AGREE` | No remaining findings after Phase 3 subplan repair; residual risk is that Phase 3 still has to implement and test derivative payload cleanup. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | QR value kernels now preserve explicit FP32/FP64 dtype under the scoped tests. |
| Baseline/comparator | Existing FP64 value parity tests still pass; FP32 values compare against FP64 references within FP32 tolerance. |
| Primary criterion | Passed: observed FP32 outputs for compact, while-loop, batched-static, masked, filtered, and dispatcher value paths. |
| Veto diagnostics | No hidden FP64 value-path coercion detected by the scoped tests; CPU/XLA smoke tests passed. |
| Explanatory diagnostics | TensorFlow emitted existing `gast` deprecation warnings; no check failed after repair. |
| Not concluded | Analytical score dtype support, benchmark dtype controls, batch-native analytical score, runtime superiority, GPU/default readiness, HMC readiness, posterior correctness, or scientific validity. |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Advance to Phase 3 | Passed scoped CPU-hidden dtype/value tests | No Phase 2 veto active | Derivative containers and score kernels still have known FP64 sites | Execute Phase 3 analytical-score dtype cleanup under refreshed subplan | Score support, benchmark readiness, batching, and speed claims |

## Phase 3 Handoff

Phase 3 may start with the refreshed subplan:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-subplan-2026-07-09.md`

Known handoff items:

- `bayesfilter/linear/kalman_qr_derivatives_tf.py` still contains hard-coded
  `tf.float64` score-path sites.
- `bayesfilter/linear/types_tf.py` derivative payload containers still need
  dtype cleanup and tests.
- `bayesfilter/results_tf.py` derivative result envelope has been tightened,
  but Phase 3 should add direct tests for derivative-result dtype behavior.
