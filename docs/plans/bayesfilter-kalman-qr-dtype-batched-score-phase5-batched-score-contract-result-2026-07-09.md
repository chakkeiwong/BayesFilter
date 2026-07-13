# Phase 5 Result: Batched Analytical Score Contract

Date: 2026-07-09

## Status

`PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

## Phase Objective

Define the exact batch-native analytical QR score API, shape contract,
limitations, reference baselines, and Phase 6 implementation boundary before
source implementation.

## Skeptical Execution Audit

| Risk | Finding |
| --- | --- |
| Wrong baseline | Baselines are the current scalar analytical score and existing batched-static QR value/autodiff reference, not a new unvalidated kernel. |
| Proxy metric misuse | Contract completeness and source-contract checks are the Phase 5 criterion; no runtime or correctness claim is made. |
| Missing stop condition | Shape ambiguity, dtype ambiguity, scalar-wrapper-as-final-kernel, and need for new derivation remain vetoes. |
| Unfair comparison | No timing comparison is made in this phase. |
| Environment mismatch | No GPU or long benchmark run is authorized by this contract phase. |
| Artifact mismatch | Exact source, test, comparator, result, and handoff paths are named below. |

Audit status: `PASSED_FOR_PHASE_5_CONTRACT_ONLY`.

## Contract: Batch-Native Analytical QR Score

The Phase 6 target is a TensorFlow/XLA-compatible batch-native score kernel
for independent time-invariant linear Gaussian state-space model rows.  The
leading dimension `B` is an independent model/parameter-proposal batch; the
derivative dimension `P` is the number of parameters per model row.  These axes
must never be conflated.

Target function name for Phase 6:

- `tf_qr_sqrt_kalman_score_batched_static` in
  `bayesfilter/linear/kalman_qr_derivatives_tf.py`.

Required dense inputs:

| Name | Shape |
| --- | --- |
| `observations` | `[T, M]` shared observations |
| `transition_offset` | `[B, N]` |
| `transition_matrix` | `[B, N, N]` |
| `transition_covariance` | `[B, N, N]` |
| `observation_offset` | `[B, M]` |
| `observation_matrix` | `[B, M, N]` |
| `observation_covariance` | `[B, M, M]` |
| `initial_state_mean` | `[B, N]` |
| `initial_state_covariance` | `[B, N, N]` |

Required first-derivative inputs:

| Name | Shape |
| --- | --- |
| `d_initial_state_mean` | `[B, P, N]` |
| `d_initial_state_covariance` | `[B, P, N, N]` |
| `d_transition_offset` | `[B, P, N]` |
| `d_transition_matrix` | `[B, P, N, N]` |
| `d_transition_covariance` | `[B, P, N, N]` |
| `d_observation_offset` | `[B, P, M]` |
| `d_observation_matrix` | `[B, P, M, N]` |
| `d_observation_covariance` | `[B, P, M, M]` |

Required outputs:

| Name | Shape |
| --- | --- |
| `log_likelihood` | `[B]` |
| `score` | `[B, P]` |

Dtype contract:

- All floating inputs must share one supported dtype, currently `tf.float32` or
  `tf.float64`.
- Outputs must preserve that requested dtype.
- Mixed floating dtypes must fail closed through the shared dtype helpers.
- TF32 mode, when enabled on GPU, is device execution provenance and must not
  be treated as requested tensor dtype.

Scope limitations:

- Dense observations only.
- Shared observations `[T, M]` only.
- Time-invariant model tensors only.
- First-order score only; Hessian is out of scope.
- No public API export beyond the internal TensorFlow function unless Phase 6
  explicitly reviews that boundary.
- Masked observations and time-varying/batched-time tensors remain future work.

Implementation boundary:

- The final Phase 6 kernel must be batch-native over `B`.  It must not use
  `tf.vectorized_map`, `tf.map_fn`, a Python loop over batch rows, or a wrapper
  that calls scalar `tf_qr_sqrt_kalman_score` once per row as the final
  implementation.
- A scalar-row wrapper may be used only as a test/reference comparator.
- A parameter-axis loop may remain only if it is the existing derivative-helper
  pattern or a reviewed first implementation choice; the target should prefer
  tensor operations over `P` where practical.
- The implementation should mirror existing batched value primitives in
  `bayesfilter/linear/kalman_qr_tf.py` and add batched first-derivative helper
  code only where needed.

## Exact Source And Comparator Paths

Source paths for Phase 6:

- Main implementation: `bayesfilter/linear/kalman_qr_derivatives_tf.py`
- Shared helper candidates: `bayesfilter/linear/qr_factor_tf.py`
- Existing batch value primitives: `bayesfilter/linear/kalman_qr_tf.py`

Required test path:

- `tests/test_linear_qr_batched_analytical_score_tf.py`

Required comparators:

- Scalar analytical row comparator:
  `bayesfilter/linear/kalman_qr_derivatives_tf.py::tf_qr_sqrt_kalman_score`
- Batched-static value/autodiff comparator:
  `bayesfilter/linear/kalman_qr_tf.py::tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop`
- Existing batched value contract/source checks:
  `tests/test_linear_qr_compact_loglik_tf.py`
- Dtype helper/container contract checks:
  `tests/test_linear_qr_dtype_contracts.py`

## Required Phase 6 Tests

Phase 6 must add tests that cover:

- FP64 `B=2, P=2` dense batched score equals stacked scalar analytical scores.
- FP32 `B=2, P=2` dense batched score preserves FP32 and matches an autodiff
  reference within declared FP32 tolerance.
- CPU/XLA compiled batched score preserves dtype and does not retrace for an
  identical signature.
- Source-contract check that the final kernel source does not contain
  `tf.vectorized_map`, `tf.map_fn`, or scalar score calls as its final batch
  route.
- Fail-closed shape checks distinguishing `B` and `P` for at least one
  derivative tensor.

## Checks Run

```bash
git diff --check -- docs/plans tests bayesfilter/linear
```

Result: passed.

## Review Record

Claude review remains unavailable for this run unless the user explicitly
approves the external disclosure risk.  A weaker bounded Codex substitute
review was used.

| Round | Reviewer | Verdict | Finding |
| --- | --- | --- | --- |
| 1 | Zeno | `AGREE` | No blocking findings. Minor non-blocking note: Phase 5 result should make exact artifact paths concrete. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | The batch-native analytical score contract is precise and implementable without confusing `B` and `P`. |
| Baseline/comparator | Existing scalar analytical score and batched-static QR value/autodiff references are named concretely. |
| Primary criterion | Passed: contract states inputs `[B, ...]`, derivative tensors `[B, P, ...]`, outputs `[B]` and `[B, P]`, dtype behavior, time-invariant limitation, and scalar/autodiff references. |
| Veto diagnostics | No active Phase 5 veto: scalar-wrapper-as-final-kernel is forbidden, dtype is explicit, and parity baselines are named. |
| Explanatory diagnostics | Source-contract checks and shape examples are specified for Phase 6. |
| Not concluded | Implementation correctness, runtime performance, GPU behavior, statistical ranking, HMC readiness, posterior correctness, default-readiness, or scientific validity. |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Advance to Phase 6 | Passed contract and substitute review | No Phase 5 veto active | Batch-native derivative helper implementation still untested | Implement Phase 6 under refreshed subplan | Correctness, performance, GPU evidence, and speed claims |

## Phase 6 Handoff

Phase 6 may start with the refreshed subplan:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-subplan-2026-07-09.md`

Phase 6 must preserve the contract above and stop if implementation requires
new unreviewed mathematics, cannot preserve dtype, or cannot pass scalar and
autodiff parity on small fixtures.
