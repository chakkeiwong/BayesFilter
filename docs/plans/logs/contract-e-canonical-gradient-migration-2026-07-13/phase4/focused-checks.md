# Phase 4 Focused Check Log

Date: 2026-07-14

## CPU-Hidden Reference And XLA Checks

`CUDA_VISIBLE_DEVICES=-1` was set before TensorFlow import. These commands are
reference/debug evidence, not GPU evidence.

```text
python -m pytest -q <Phase 0-3 compatibility paths>
134 passed, 2 warnings in 9.84s
exit: 0

python -m pytest -q tests/highdim/test_ledh_contract_e_streaming_phase4.py \
  -k 'not owned_boundaries_and_public_cpu_xla_wrappers'
7 passed, 1 deselected, 2 warnings in 15.96s
exit: 0

python -m pytest -q tests/highdim/test_ledh_contract_e_streaming_phase4.py \
  -k owned_boundaries_and_public_cpu_xla_wrappers
1 passed, 7 deselected, 2 warnings in 26.11s
exit: 0
```

The warnings are the existing TensorFlow Probability `distutils.version`
deprecations.

Python compilation, JSON parsing, source prohibition checks, scoped
`git diff --check`, and the local HLO audit passed. The persisted Phase 3 and
Phase 4 descriptive diagnostics recomputed exactly after the narrowed XLA
repair.

## Trusted GPU Checks

Device probe:

```text
NVIDIA GeForce RTX 4080 SUPER, driver 591.86, 16376 MiB total
TensorFlow 2.19.1, CUDA build 12.4, compute capability 8.9
logical GPU visible, TF32 enabled
```

Forward preflight:

```text
timeout 600s ... benchmark_contract_e_streaming_phase4_gpu_preflight.py \
  --output .../gpu-forward-preflight.json
exit: 0
status: GPU_FORWARD_EXECUTED_VALID_CHART_FEASIBILITY_DESCRIPTIVE_ONLY
```

The initial shell launch before that run failed at local-package import and is
documented as a pre-output attempt 0.

Analytic VJP attempt 1 reached XLA and aborted in GEMM-fusion compilation with
exit `134`. The HLO-anchored VJP-local repair was reviewed and revalidated.

Final analytic VJP:

```text
timeout 600s ... benchmark_contract_e_streaming_phase4_gpu_preflight.py \
  --mode vjp --output .../gpu-vjp-preflight-final.json
exit: 0
status: GPU_ANALYTIC_VJP_EXECUTED_VALID_CHART_FINITE_COTANGENTS_FEASIBILITY_DESCRIPTIVE_ONLY
```

## Interpretation

Exact quotient identities, local direct-plus-transport duality, generic payload
VJP, CPU-XLA wrappers, and production-shape forward/VJP feasibility pass. General
dense/autodiff/chunk differences remain descriptive. Numerical and scientific
promotion remains blocked.
