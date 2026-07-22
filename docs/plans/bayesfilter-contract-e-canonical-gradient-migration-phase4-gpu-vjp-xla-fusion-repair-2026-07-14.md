# Phase 4 GPU VJP XLA Fusion Repair

Date: 2026-07-14

Status: `REPAIR_REVALIDATED_REVIEWED_SECOND_ATTEMPT_AUTHORIZED`

## Actual Production-Shape Failure

After the pre-output trace repair, the unchanged trusted-GPU analytic VJP
reached XLA compilation at `B=1,N=10000,d=3`, then the TensorFlow process
aborted with exit `134`:

```text
FAILED_PRECONDITION: Can not combine dim orders and requirements.
Failure occurred when compiling fusion gemm_fusion_dot.94
```

The fatal HLO was a `f32[10000,3]` by `f32[3,3]` dot inside the Contract E
pullback. It included the exact `2/N = 0.0002` multiplier from
`_uniform_covariance_vjp`, which anchors the failure to a large-`N`, tiny-`d`
row-wise linear action. XLA aborted the process during GEMM-fusion autotuning,
so Python could not write a fresh normal result artifact. This is a real
GPU/XLA derivative-feasibility failure and a repair trigger. It is not an OOM,
nonfinite derivative, mathematical-parity failure, or scientific result.

## Target-Preserving Repair

For centered points `X[b,n,j]` and the symmetrized covariance cotangent
`A[b,i,j]`, `_uniform_covariance_vjp` computed

```text
Y[b,n,i] = matmul(X, A, transpose_b=True)
         = sum_j X[b,n,j] A[b,i,j].
```

The repair implements only this VJP-local row action as a TensorFlow broadcast
multiply and reduction:

```text
reduce_sum(X[:,:,None,:] * A[:,None,:,:], axis=-1).
```

The shared `_apply_rows` helper remains the original matrix multiplication for
forward and JVP code. This narrowly removes the exact failing large-`N`, tiny-`d`
VJP GEMM-fusion route without changing the target, dtype, prepared inputs,
derivative coordinates, or asymptotic storage. Live state is `O(B*N*d^2)` for
this small-dimensional operation, not `O(N^2)`. No XLA environment flag, JIT
exception, TF32 exception, threshold, floor, clip, stop-gradient, or alternate
reset is introduced.

## Required Revalidation

- Phase 3 exact cloud certificates and dense/autodiff checks;
- Phase 4 eager quotient/composition checks;
- isolated Phase 4 CPU-XLA wrappers;
- source/diff/hash checks; and
- bounded implementation review before the second and final production-shape
  VJP attempt.

The successful forward artifact remains evidence for the source hash that
actually produced it. The repaired-source VJP attempt must receive its own hash
in the final manifest. All adequacy, admission, and scientific claims remain
blocked.

## Revalidation And Review

- Phase 0-3 compatibility and exact certificates: `134 passed, 2 warnings`;
- Phase 4 eager/reference checks: `7 passed, 1 deselected, 2 warnings`;
- isolated Phase 4 CPU-XLA wrappers: `1 passed, 7 deselected, 2 warnings`;
- Phase 3 and Phase 4 persisted diagnostic hashes remained exactly unchanged;
- Python compilation and scoped diff checks passed; and
- bounded repair review returned no material finding and `VERDICT: AGREE`.

The second and final production-shape VJP attempt is therefore authorized under
the unchanged Phase 4 evidence contract.
