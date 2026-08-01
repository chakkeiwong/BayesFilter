# Phase 4 Trusted-GPU VJP Attempt 1 Fatal Log

Date: 2026-07-14

Command:

```text
timeout 600s /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/benchmark_contract_e_streaming_phase4_gpu_preflight.py \
  --mode vjp \
  --output docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase4/gpu-vjp-preflight.json
```

Exit: `134` (`timeout: the monitored command dumped core`).

The run initialized the RTX 4080 SUPER, entered XLA compilation, and aborted in
GPU GEMM-fusion autotuning before execution:

```text
FAILED_PRECONDITION: Can not combine dim orders and requirements.
Failure occurred when compiling fusion gemm_fusion_dot.94
```

The emitted HLO fused a `f32[10000,3]` by `f32[3,3]` dot with the exact
`2/N = 0.0002` multiplier from `_uniform_covariance_vjp`. It was not an OOM or
a nonfinite result. The process-level abort prevented Python from replacing the
pre-existing structured trace-failure JSON, so this Markdown record is the
authoritative attempt-1 terminal record. The trace-failure JSON was preserved
separately before this launch.

Repair:
`docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase4-gpu-vjp-xla-fusion-repair-2026-07-14.md`.

Interpretation: derivative GPU/XLA feasibility failed for the first compiled
form and triggered a target-preserving implementation repair. It did not
invalidate the exact small-chart mathematics or establish any scientific
claim.
