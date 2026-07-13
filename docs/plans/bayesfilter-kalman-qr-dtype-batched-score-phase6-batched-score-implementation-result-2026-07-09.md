# Phase 6 Result: Batch-Native Analytical Score Implementation

Date: 2026-07-09

## Status

`PASSED_WITH_WEAKER_CODEX_SUBSTITUTE_REVIEW`

## Phase Objective

Implement `tf_qr_sqrt_kalman_score_batched_static`, a batch-native analytical
QR score kernel for independent model rows, preserving dtype and XLA
compatibility.

## Skeptical Execution Audit

| Risk | Finding |
| --- | --- |
| Wrong baseline | Baselines were the scalar analytical score rows and batched-static value/autodiff reference from existing QR value code. |
| Proxy metric misuse | Passing source-contract tests does not prove performance; Phase 6 claims only correctness/dtype/shape under scoped fixtures. |
| Missing stop condition | Source-wrapper use, shape mismatch, dtype mismatch, parity failure, nonfinite output, and XLA failure remained vetoes. |
| Hidden assumption | Initial `B=P=N=M`-like tiny coverage was strengthened after review to include `B=3`, `P=2`, `N=2`, `M=2`. |
| Environment mismatch | Checks were CPU-hidden and XLA smoke only; they do not support GPU or performance claims. |
| Artifact mismatch | This result records source changes, tests, reviews, repair loop, and Phase 7 handoff. |

Audit status: `PASSED_FOR_PHASE_6_CPU_HIDDEN_IMPLEMENTATION_CHECKS`.

## Source Changes

- Added batched first-derivative QR/Cholesky/factor-covariance helpers in
  `bayesfilter/linear/kalman_qr_derivatives_tf.py`.
- Added `tf_qr_sqrt_kalman_score_batched_static` with output shapes `[B]` and
  `[B, P]`.
- Kept the final batch kernel batch-native over `B`; it does not call scalar
  `tf_qr_sqrt_kalman_score` per row and does not use `tf.vectorized_map` or
  `tf.map_fn` as the final implementation route.
- Repaired the no-jitter update branch so derivative update terms use the same
  observation update covariance factor as the primal update.
- Added `tests/test_linear_qr_batched_analytical_score_tf.py` with scalar-row
  parity, batched-autodiff parity, FP32 dtype preservation, CPU/XLA smoke,
  `B != P` multi-dimensional coverage, default-jitter dtype preservation, and
  source-contract checks.

## Checks Run

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py
```

Result: `7 passed`.

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_linear_qr_batched_analytical_score_tf.py tests/test_linear_qr_dtype_contracts.py
```

Result: `19 passed`.

```bash
python -m py_compile bayesfilter/linear/kalman_qr_derivatives_tf.py tests/test_linear_qr_batched_analytical_score_tf.py
```

Result: passed.

```bash
git diff --check -- bayesfilter/linear tests docs/plans
```

Result: passed.

## Repair Log

| Issue | Status |
| --- | --- |
| First test run failed due to an incorrect batched trace einsum tying batch and observation axes. | Repaired by using `tf.einsum("bij,bpji->bp", innovation_precision, dS)`. |
| Initial review found likely derivative mismatch when `jitter_updates_filtered_covariance=False`. | Repaired batch path and matching scalar dynamic score path to use `observation_update_covariance_factor` in derivative update terms. |
| Initial review found weak `B` vs `P` coverage. | Added `B=3`, `P=2`, `N=2`, `M=2` parity test exercising the no-jitter branch. |
| Initial review found source-contract test only inspected the top-level batch function. | Expanded source-contract inspection to key batched helper sources. |
| Focused re-review prompt over-forbade file reads and produced a prompt-shape `REVISE`. | Relaunched with read-only file inspection allowed for the exact two paths; focused re-review returned `AGREE`. |

## Review Record

Claude review remains unavailable for this run unless the user explicitly
approves the external disclosure risk.  A weaker bounded Codex substitute
review was used.

| Round | Reviewer | Verdict | Finding |
| --- | --- | --- | --- |
| 1 | Sagan | `REVISE` | No-jitter update derivative risk, weak `B`/`P` fixture coverage, and narrow source-contract inspection. |
| 2 | Bacon | `REVISE` | Prompt-shape failure: review prompt forbade commands needed for read-only file inspection. Not an implementation finding. |
| 3 | Beauvoir | `AGREE` | Prior Phase 6 findings fixed; no remaining blocker in focused slice. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Batch-native analytical score returns `[B]` values and `[B, P]` scores matching scalar analytical and autodiff references under scoped checks. |
| Baseline/comparator | Scalar `tf_qr_sqrt_kalman_score` rows and autodiff through `tf_qr_sqrt_kalman_log_likelihood_batched_static_while_loop`. |
| Primary criterion | Passed: FP32/FP64 outputs have requested dtype and match references within declared tolerances under CPU/XLA. |
| Veto diagnostics | No active Phase 6 veto: source-wrapper, shape, dtype, parity, finite-output, and XLA checks passed. |
| Explanatory diagnostics | TensorFlow emitted existing `gast` warnings; no check failed after repair. |
| Not concluded | Runtime superiority, full benchmark ladder, GPU evidence, statistical ranking, HMC readiness, posterior correctness, default-readiness, or scientific validity. |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Advance to Phase 7 | Passed scoped CPU-hidden correctness/dtype/source-contract checks | No Phase 6 veto active | Benchmark harness still needs batch-native analytical score arm and exact grids | Refresh and execute Phase 7 benchmark ladder | Performance ranking, GPU/default readiness, and scientific claims |

## Phase 7 Handoff

Phase 7 must refresh the benchmark harness before running the ladder:

- Add a batch-native analytical score arm using
  `tf_qr_sqrt_kalman_score_batched_static`.
- Keep scalar analytical row-loop and batched-autodiff comparators separate.
- Record requested/observed dtype, JIT, TF32, device provenance, batch size,
  parameter count, dimension, compile+first-call time, warm-start time, and
  repeated warm-call summaries.
- CPU-hidden artifacts remain debug/reference evidence only.
- GPU artifacts require trusted provenance and must not hide GPU devices.
