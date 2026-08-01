# Phase 2 Result: Batch-Native LGSSM Materialization

Date: 2026-07-14

## Outcome

**PASS_PHASE2_AND_CONTINUE.** A new TensorFlow-only materializer converts
`[B,18]` raw parameters into exact batch-leading LGSSM model tensors, prior
values/scores, stationary covariances, and first derivatives. It uses no
sample-axis loop or mapping primitive, NumPy, or host callback.

The shared `16 x 16` Lyapunov operator solves the stationary covariance and all
18 sensitivity right-hand sides without changing the scalar equations.

## Evidence

| Check | Result |
| --- | --- |
| New materialization suite | `11 passed` |
| Existing scalar LGSSM and SVD authority regression | `19 passed` |
| All emitted model/derivative tensors compared for three rows | pass at `rtol=atol=2e-13` |
| Prior value/score scalar parity | pass |
| Row permutation equivariance | pass |
| Primal Lyapunov residual | max `<=2e-15` |
| Differentiated Lyapunov residual | max `<=3e-15` |
| Representative finite differences | pass |
| Dynamic graph and fixed CPU-XLA | pass |
| Compiled graph operation audit | 27 operation types; no `While`, `Map`, or host callback |
| Python compile and `git diff --check` | pass |

The CUDA initialization warning seen during the graph-operation audit occurred
with `CUDA_VISIBLE_DEVICES=-1`; it is expected CPU-hidden sandbox output and is
not GPU evidence or a GPU defect.

One combined regression command initially named a non-existent test file and
stopped before testing. The corrected command used the discovered path
`tests/test_linear_kalman_svd_tf.py` and passed. This was a harness typo, not an
implementation failure.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit batch materialization | every row/tensor/derivative matches scalar authority | no numerical or policy veto | GPU performance not measured | implement batch SVD/eigh recursion | no likelihood or training admission |
| Retain shared multi-RHS solve | scalar parity and Lyapunov residuals pass | no orientation mismatch | solve ordering gives tiny non-bitwise rounding | keep tight tolerance tests | no speed claim |

## Handoff

Phase 3 may start. Its reviewed subplan is
`docs/plans/bayesfilter-neutra-batch-native-training-phase3-svd-graph-status-kernel-subplan-2026-07-14.md`.

