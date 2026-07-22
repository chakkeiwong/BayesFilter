# Phase 2 Subplan: Batch-Native LGSSM Materialization

Date: 2026-07-14

## Phase Objective

Implement the target-local transformation from raw `[B,18]` parameters to
batch-native model tensors, analytical prior values/scores, stationary
covariances, and all first derivatives without a sample-axis Python/TensorFlow
map, NumPy, or host callback.

## Entry Conditions Inherited From Phase 1

- The exact scalar materializer and SVD/eigh graph-status target remain frozen
  parity authorities.
- Selected training topology is one GPU/XLA batch target; CPU worker sharding is
  an alternative repair topology only.
- Reverse-KL base noise is generated in-graph; there is no offline training
  dataset in the current contract.
- Cholesky, QR, sigma-point, and DSGE defaults cannot replace target-local math.

## Required Artifacts

- `bayesfilter/testing/multidim_triangular_lgssm_batched_tf.py` containing only
  TensorFlow batch materialization and prior operations.
- Focused test module covering shapes, row parity, row permutation, Lyapunov
  residuals, derivative parity, finite differences, graph, and CPU-XLA.
- Phase 2 result and reviewed Phase 3 SVD/eigh kernel subplan.

## Exact Tensor Contract

With `B` dynamic per call, `P=18`, `N=M=4`:

- model vectors: `[B,N]` or `[B,M]`;
- model matrices/covariances: `[B,N,N]`, `[B,M,N]`, `[B,M,M]`;
- derivative vectors: `[B,P,N]` or `[B,P,M]`;
- derivative matrices: `[B,P,N,N]`, `[B,P,M,N]`, `[B,P,M,M]`;
- prior value `[B]`, prior score `[B,P]`.

The result container must expose tensors directly rather than force batch axes
through the scalar `TFLinearGaussianStateSpace` dataclasses.

## Stationary Solve Contract

Construct `L_b = I - A_b kron A_b`. Solve `L_b vec(P_b)=vec(Q_b)`.
Then construct all derivative right-hand sides and solve them together as
`[B,16,18]`. This shared multi-right-hand-side solve is admitted only if every
row matches the scalar materializer and satisfies value/derivative Lyapunov
residual gates.

## Required Checks

1. Exact output-shape test at `B=1`, `B=3`, and a dynamic graph input.
2. Row-wise equality against `materialize_lower_triangular_lgssm_with_first_derivatives`.
3. Batch row-permutation equivariance.
4. Stationary Lyapunov and differentiated Lyapunov residual checks.
5. Central finite-difference checks for representative transition, process-noise,
   observation-noise, and stationary-covariance derivatives.
6. Eager, `tf.function`, and CPU-XLA execution with finite outputs.
7. Static source audit rejecting NumPy, Python loops, `tf.map_fn`,
   `tf.vectorized_map`, and host callbacks.
8. Existing scalar target materialization tests.
9. Python compile and `git diff --check`.

## Evidence Contract

| Item | Phase contract |
| --- | --- |
| Question | Does one TensorFlow tensor program materialize every LGSSM batch row and first derivative identically to the scalar authority? |
| Pass criterion | Shape, scalar parity, permutation, residual, finite-difference, graph, and CPU-XLA checks pass. |
| Hard veto | Any row/derivative mismatch, invalid residual, sample mapping/loop, NumPy, host callback, nonfinite output, or XLA failure. |
| Explanatory only | Compile time and materialization runtime. |
| Artifact | Test output and Phase 2 result. |
| Nonclaims | Materialization does not establish batch SVD likelihood parity, training speed, transport quality, HMC readiness, or scientific validity. |

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Scatter-free basis/einsum construction | Scalar fixed lower-triangular layout | parameter-to-entry misindexing | exact row parity and representative finite differences | implementation hypothesis |
| Shared multi-RHS stationary solve | Same Lyapunov operator for all sensitivities | vectorization convention or RHS orientation error | scalar parity plus differentiated residual | reviewed optimization hypothesis |
| Scalar dataclasses not reused for batch | Existing dataclasses require scalar leading shapes | new container could drift from downstream kernel needs | exact Phase 3 shape handoff | reviewed interface choice |
| Dynamic leading batch | Trainer may use fixed compiled batches, tests need row generality | XLA recompilation or unsupported dynamic shape | fixed CPU-XLA plus dynamic graph test; performance deferred | convenience choice |

## Skeptical Subplan Audit

- Wrong baseline: exact scalar model tensors and derivatives are compared per
  identical row; Cholesky batch output is not the parity authority.
- Hidden convention risk: Kronecker/vectorization orientation is checked by
  scalar parity and both primal/derivative residuals.
- Proxy promotion: XLA compilation does not replace numerical parity.
- Unfair tolerance: compare float64 results with tight absolute/relative
  tolerances, and report any solve-order residual rather than demanding bitwise
  equality.
- Phase leakage: no likelihood recursion, adapter binding, optimizer, or GPU
  timing is added in Phase 2.

Audit verdict: **PASS**. The shared-solve optimization is mathematically
identical if its explicit parity/residual gates pass, and the phase remains
small enough to repair locally.

## Forbidden Claims And Actions

- Do not add a sample-axis map or loop, even temporarily as an active fallback.
- Do not use NumPy for construction or derivative computation.
- Do not change scalar target functions, parameter order, transforms, priors,
  jitter, or status law.
- Do not bind the exact adapter or run NeuTra training in this phase.

## Exact Next-Phase Handoff Conditions

Phase 3 starts only after all materialization gates pass and the Phase 3 subplan
defines the batch SVD/eigh solve/logdet, per-row graph status, one time loop,
scalar parity cases, invalid/floor cases, and XLA checks.

## Stop Conditions

Stop only for an unrepairable scalar mismatch, a mathematical ambiguity in the
stationary derivative, missing scalar authority, or exhausted focused repair
budget. A coding, shape, or XLA defect triggers local repair and retry.

## Phase-End Procedure

1. Run all required local checks.
2. Write the Phase 2 result/close record.
3. Draft or refresh the Phase 3 subplan.
4. Review Phase 3 suitability and continue when no real blocker exists.

