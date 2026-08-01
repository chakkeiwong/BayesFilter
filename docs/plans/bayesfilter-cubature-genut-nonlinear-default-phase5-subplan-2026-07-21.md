# Phase 5 Subplan: Candidate GPU/XLA/TF32 Smoke

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_GPU_XLA_MECHANICS_ONLY`

> **Correction, 2026-07-22:** This is an arbitrary-input device smoke, not an
> SV experiment.  It is retained only for GPU/XLA engineering provenance.

## Question

Does the generic candidate nonlinear core compile and execute on the trusted
GPU with float32/TF32 and XLA for a tiny exact-SV fixture without changing the
finite value/score result materially?

## Scope

- exact transformed-SV adapter;
- `T=2`, `N=12`, one fixed seed;
- float32, TF32 enabled, memory growth enabled;
- `tf.function(jit_compile=True)`;
- CPU/eager result retained as the reference arm.

## Pass/Stop

Pass requires visible trusted GPU, memory-growth verification, finite GPU/XLA
value and score, no device fallback, and small CPU/GPU differences recorded in
the artifact. Stop on compile failure, allocator-policy failure, nonfinite
output, or a material CPU/GPU discrepancy. This is a smoke gate only, not
high-dimensional or full-horizon readiness.
