# Cubature/GenUT Nonlinear Default Program: Phase 5 Scaling Result

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_GPU_XLA_RESOURCE_SCALING_ONLY`

> **Correction, 2026-07-22:** Direct-Normal transformed observations are not SV
> data.  Timing, compile, memory, placement, and finiteness evidence remain;
> every SV accuracy, score, or method implication is ineligible.

## Outcome

The checkpointed GPU/XLA ladder completed for the exact transformed-SV scalar
adapter at `N={12,24,48,96}` and `T={2,10,50}`. The accepted artifacts are
the manifest-bearing complete ladder and the isolated particle-count
confirmations:

- `docs/benchmarks/artifacts/cubature_genut_gpu_xla_scaling_20260721/attempt06_manifest/result.json`
- `docs/benchmarks/artifacts/cubature_genut_gpu_xla_scaling_20260721/attempt01/result.json`
- `docs/benchmarks/artifacts/cubature_genut_gpu_xla_scaling_20260721/attempt02_n12/result.json`
- `docs/benchmarks/artifacts/cubature_genut_gpu_xla_scaling_20260721/attempt03_n24/result.json`
- `docs/benchmarks/artifacts/cubature_genut_gpu_xla_scaling_20260721/attempt04_n48/result.json`
- `docs/benchmarks/artifacts/cubature_genut_gpu_xla_scaling_20260721/attempt05_n96/result.json`

All 12 cells are finite, GPU-resident, XLA-compiled, and hard-valid under the
diagnostic harness. The maximum recorded reset residuals across the ladder
were approximately `1.3e-6` for row/column marginal checks and `3.0e-7` for
the mean check.

## Timing Summary

| N | T=2 compile / warm (s) | T=10 compile / warm (s) | T=50 compile / warm (s) |
|---:|---:|---:|---:|
| 12 | 3.99 / 0.0017 | 14.68 / 0.0056 | 73.30 / 0.0853 |
| 24 | 5.30 / 0.0020 | 13.57 / 0.0060 | 68.38 / 0.0922 |
| 48 | 4.02 / 0.0012 | 11.94 / 0.0085 | 68.66 / 0.1010 |
| 96 | 4.55 / 0.0015 | 12.61 / 0.0162 | 70.13 / 0.0864 |

The dominant cost in this implementation is shape-specific XLA compilation as
horizon increases, not warmed execution or allocator growth for this scalar
ladder. Peak TensorFlow allocator readings were about `134 MB` for most
shorter cells; the `T=50` cells reported lower per-cell peaks after allocator
reset, and should not be interpreted as a hard memory cap.

## Failure And Repair Record

The first all-cell launch (`attempt01`) completed all 12 cells and wrote a
valid result artifact, but XLA GEMM autotuning emitted NaN/reference mismatch
diagnostics at larger shapes. The harness was then repaired to checkpoint
every cell and serialize Python exceptions; isolated reruns completed all
particle-count blocks as confirmations. The compiler warnings remain visible
in the logs and are recorded as a numerical/compiler risk; they were not
hidden or converted into a correctness claim.

The accepted manifest-bearing run took `347.7 s` wall time and records commit,
exact command, Python/TensorFlow versions, CUDA visibility, and wall time in
its `run_manifest`. Its twelve rows are all hard-valid.

## Decision Table

| Decision | Status |
|---|---|
| Scalar GPU/XLA feasibility through `N=96,T=50` | Passed diagnostic |
| Finite candidate value/recursive score | Passed for all 12 fixed fixtures |
| High-dimensional feasibility | Not established; state dimension remained `d=1` |
| FP64/reference accuracy | Not tested |
| Compile scalability for target production route | Unacceptable to promote without staged/loop-native graph work |
| Full-horizon nonlinear accuracy | Not established |
| Default/leaderboard readiness | False; policy unchanged |
| Next justified action | Repair XLA graph structure and begin target-bound model/comparator pilots |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Passed for all scalar cells; attempt01 compiler diagnostics remain an explanatory risk |
| Statistically supported ranking | None |
| Descriptive-only differences | Compile/runtime and allocator values only |
| Default readiness | Not eligible |
| Evidence needed next | Dimension ladder, FP64/reference arm, model-scope tuning, full-horizon claims, and same-target Contract E comparison |

## Nonclaims

This result does not establish high-dimensional accuracy, exact nonlinear
filtering, unbiasedness, score precision, method superiority, HMC readiness,
leaderboard admission, default promotion, or a NAWM result.
