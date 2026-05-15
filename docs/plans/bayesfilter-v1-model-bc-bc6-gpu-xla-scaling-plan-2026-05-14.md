# BayesFilter V1 Model B/C BC6 GPU/XLA Scaling Ladder Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase BC6 in:

```text
docs/plans/bayesfilter-v1-model-bc-thorough-testing-master-program-2026-05-14.md
```

## Purpose

Test whether point-axis, horizon, or batch scaling makes GPU/XLA useful for
Model B/C nonlinear sigma-point filters.

## Entry Gate

BC6 may start only after BC1 stable boxes and BC3 horizon/noise envelopes exist
for the shapes tested.  BC6 does not require BC5 HMC classification.  Every
GPU/CUDA/NVIDIA command requires escalated/trusted permissions.

## Evidence Contract

Question:
- For which tested shapes, if any, do CPU graph, CPU XLA, GPU graph, or GPU XLA
  materially improve runtime without changing correctness diagnostics?

Baseline:
- P6 tiny GPU/XLA diagnostic and BC1 stable boxes.

Primary criterion:
- Each row makes a shape-specific performance statement only.

Veto diagnostics:
- non-escalated GPU failure is treated as CUDA failure;
- one tiny shape becomes a broad speedup claim;
- benchmark changes production behavior.

What will not be concluded:
- Broad GPU speedup, production deployment policy, or HMC performance.

Artifact:
- BC6 benchmark artifacts and result file.

## Required Comparisons

Where feasible:
- CPU graph;
- CPU XLA;
- GPU graph;
- GPU XLA.

Shapes should vary:
- horizon;
- sigma-point count/filter;
- batched parameter points;
- batched independent panels if practical.

## Execution Steps

1. Run escalated device visibility check before TensorFlow import.
2. Record CPU/GPU device visibility, TensorFlow version, and XLA mode.
3. Compare first-call and steady-call timings under a documented warmup policy.
4. Record shape, horizon, batch axes, point count, branch status, and memory
   notes.
5. Verify value/score diagnostics for the benchmark rows.
6. Write the BC6 result artifact and update the V1 reset memo.

## Primary Gate

BC6 passes if every benchmark row is interpreted at its exact shape and device
scope.

## Veto Diagnostics

Stop and ask for direction if:
- escalation is unavailable for GPU commands;
- benchmark code changes production semantics;
- branch instability is ignored in timing rows;
- GPU failure is diagnosed from a non-escalated run.

## Expected Artifacts

Use the execution date in result filenames.  The plan date remains
2026-05-14, but future result artifacts should use `YYYY-MM-DD`.

```text
docs/plans/bayesfilter-v1-model-bc-bc6-gpu-xla-scaling-result-YYYY-MM-DD.md
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
docs/benchmarks/bayesfilter-v1-model-bc-gpu-xla-*.json
```

## Continuation Rule

Continue to BC7 after recording shape-specific GPU/XLA status.  BC6 cannot
unlock Hessian implementation or HMC convergence claims by itself.
