# BayesFilter V1 P6 GPU/XLA Scaling Plan

## Date

2026-05-14

## Governing Master Program

This plan executes Phase P6 / R7 in:

```text
docs/plans/bayesfilter-v1-master-program-2026-05-13.md
```

## Entry Gate

P6 may start only after P2 identifies stable nonlinear score/value boxes and
P3 provides benchmark commands.  GPU commands require escalated sandbox
permissions under `AGENTS.md`.

## Motivation

CUT4 has many more sigma points than cubature or UKF.  Modern GPU/XLA
architectures may reduce the cost when computations are vectorized over the
point axis, but this is an empirical diagnostic claim, not a mathematical
correctness claim.

## Required Comparisons

Where feasible, compare:
- CPU eager;
- CPU graph;
- GPU eager;
- XLA-visible execution.

Record:
- device visibility;
- shape and horizon;
- backend and point count;
- warmup policy;
- first-call and steady-call timing;
- memory notes if available;
- branch status on the benchmark box.

## Primary Gate

P6 passes if the result artifact states whether GPU/XLA helps for the tested
shapes and limits the claim to those shapes.

## Veto Diagnostics

Stop and ask for direction if:
- a non-escalated GPU failure is treated as a real CUDA failure;
- broad GPU speedup claims are made from a tiny artifact;
- benchmark code changes production behavior;
- branch instability is ignored in timing rows.

## Expected Artifacts

```text
docs/plans/bayesfilter-v1-p6-gpu-xla-scaling-result-2026-05-14.md
docs/benchmarks/bayesfilter-v1-*gpu*xla*
docs/plans/bayesfilter-v1-external-compat-reset-memo-2026-05-10.md
```
