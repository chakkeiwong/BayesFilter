# Phase 3 Subplan: Batch-Native SVD/Eigh Graph-Status Kernel

Date: 2026-07-14

## Phase Objective

Lift the frozen scalar SVD/eigh Kalman likelihood, first-order score, and
graph-status program over a leading proposal batch while retaining exactly one
TensorFlow time loop and no proposal/sample loop.

## Entry Conditions Inherited From Phase 2

- `[B,18]` model and derivative materialization passes scalar parity and
  Lyapunov residual gates.
- Scalar `tf_svd_linear_gaussian_score_first_order_graph_status` remains the
  mathematical and status authority.
- No adapter binding or NeuTra optimizer update is allowed yet.

## Required Artifacts

- `bayesfilter/linear/batched_kalman_svd_derivatives_tf.py` with an XLA-default
  batch kernel and tensor result/status type.
- Focused tests for regular, active-floor, and invalid-row cases; scalar parity;
  row permutation; graph operations; and CPU-XLA.
- Phase 3 result and reviewed Phase 4 integration subplan.

## Mathematical Contract

For each row and time step, preserve the scalar equations for prediction,
innovation, SVD/eigh solve/logdet, score, gain, Joseph covariance update, and
their first derivatives. `tf.linalg.eigh` receives `[B,M,M]` and operates over
leading rows.

Invalid eigensolver input must be guarded per row. A finite benign identity is
substituted only for that row so graph execution completes; its status is code
`2`. Active floors are counted per row and yield code `1`. Valid pre-regularized
rows yield code `0`. Status precedence is invalid over active floor, matching
the scalar authority.

## Required Checks

1. Exact value, score, and all five adapter status-field parity against scalar
   rows near truth.
2. Active-floor parity with a deliberately high floor.
3. Mixed valid/invalid row test proving per-row isolation and status precedence.
4. Row permutation equivariance.
5. One `While` operation and no map/host-callback operations in the graph.
6. CPU-XLA finite regular-case execution.
7. Source audit: no NumPy, Python loop, `tf.map_fn`, `tf.vectorized_map`, or
   host callback.
8. Existing scalar SVD/LGSSM tests, Python compile, and `git diff --check`.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Does the batched kernel compute the same per-row SVD/eigh value, score, and status law as the scalar authority? |
| Pass criterion | Regular, floor, invalid, permutation, graph, and CPU-XLA gates pass. |
| Hard veto | Any value/score/status mismatch, cross-row contamination, missing one-loop topology, mapping/callback fallback, nonfinite valid row, or XLA failure. |
| Explanatory only | Compile time, operation count, and CPU runtime. |
| Artifact | Focused tests and Phase 3 result. |
| Nonclaims | Kernel parity does not establish adapter/trainer binding, GPU speed, transport quality, HMC readiness, or scientific validity. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Batched `tf.linalg.eigh` | TensorFlow leading-batch semantics | row ordering or solver output differs | scalar row and permutation parity | reviewed mechanism |
| Per-row benign identity guard | scalar `psd_eigh_graph_status` | batch-wide guard hides valid rows or invalid precedence | mixed valid/invalid test | reviewed requirement |
| One time `tf.while_loop` | scalar authority and Kalman dependence | XLA carries large `[B,P,N,N]` state inefficiently | graph audit now; performance Phase 6 | reviewed default |
| Parallel iterations one | scalar kernel deterministic ordering | reduced throughput | parity first; tune only in Phase 6 with same math | frozen initial choice |

## Skeptical Subplan Audit

- Wrong baseline: scalar SVD/eigh output on identical rows, not Cholesky output.
- Status proxy risk: all required status fields are compared, not only finite
  values.
- Batch poisoning risk: mixed invalid/valid rows are required.
- Hidden topology: graph operation inventory must show one loop and no map.
- Performance leakage: no timing target or GPU promotion occurs in Phase 3.

Audit verdict: **PASS**. Per-row invalid guarding and status precedence close
the main batch-specific correctness hazard.

## Forbidden Claims And Actions

- Do not replace eigensolves with Cholesky/QR or change Joseph updates.
- Do not reduce status across the batch before returning it.
- Do not add a scalar fallback or adapter binding.
- Do not run NeuTra training or use CPU timing as GPU evidence.

## Exact Next-Phase Handoff Conditions

Phase 4 starts only after kernel parity/status/XLA/topology gates pass and its
subplan binds the exact adapter without removing scalar HMC/parity methods.

## Stop Conditions

Stop only for an unrepairable mathematical/status mismatch, cross-row
contamination that cannot be isolated, missing XLA support for required math,
or exhausted focused repair budget. Shape and formula bugs trigger repair.

## Phase-End Procedure

1. Run all required local checks.
2. Write the Phase 3 result/close record.
3. Draft or refresh the Phase 4 integration subplan.
4. Review Phase 4 suitability and continue when no real blocker exists.

