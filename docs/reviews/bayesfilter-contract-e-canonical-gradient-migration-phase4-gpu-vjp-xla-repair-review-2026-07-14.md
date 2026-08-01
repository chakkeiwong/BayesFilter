# Phase 4 GPU VJP XLA Repair Review

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Reviewer: fresh bounded Codex substitute reviewer

Reviewed path:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase4-gpu-vjp-xla-fusion-repair-2026-07-14.md`

## Finding

The VJP-local broadcast multiplication and reduction computes the same
row-wise action as the failed matrix multiplication:

```text
sum_j X[b,n,j] A[b,i,j].
```

This identity does not require symmetry. The only numerical difference is
floating-point reduction order, which is covered by the completed exact,
autodiff, persisted-diagnostic, and CPU-XLA revalidation.

The repair preserves inputs, coordinates, dtype, JIT/TF32 policy, reset
semantics, and `O(B*N*d^2)` rather than `O(N^2)` storage. At the frozen
`B=1,N=10000,d=3` shape, it directly removes the fatal GEMM-fusion route and is
sufficient to justify the second and final trusted-GPU VJP attempt.

The repaired source hash and final-attempt result must be recorded separately.

`VERDICT: AGREE`
