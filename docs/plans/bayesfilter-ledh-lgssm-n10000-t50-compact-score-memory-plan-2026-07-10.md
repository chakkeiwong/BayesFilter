# LGSSM LEDH N10000 T50 Compact Score Memory Plan

Date: 2026-07-10

## Question

For the repaired LGSSM LEDH compact score route, what are the GPU memory and
wall-time costs of the score computation at the full `N=10000,T=50` setting
with `row/col/particle = 2500/2500/2500`?

## Evidence Contract

Engineering question: measure memory and time consumption for the compact
LGSSM LEDH score computation at `N=10000,T=50`.

Baseline/comparator: the completed `N=10000,T=10` compact-score diagnostic
artifact:
`docs/plans/ledh-lgssm-n10000-t10-compact-score-memory-2500-2026-07-10.json`.

Primary diagnostic output:

- `score_gpu_memory_info_after.peak`;
- `elapsed_seconds`;
- `compile_and_first_call_seconds`;
- `warm_call_timing_summary_seconds`;
- score route/provenance metadata;
- GPU/TF32/dtype metadata.

Pass for this diagnostic means:

- terminal artifact is emitted;
- score stage completes;
- score route is
  `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`;
- `manual_score_diagnostic.uses_full_history_reverse_route` is false;
- score tensors are on GPU;
- precision records `dtype=float32` and TF32 enabled;
- score memory peak is present.

Veto conditions:

- CPU execution represented as trusted GPU evidence;
- `float64` execution;
- TF32 disabled;
- full-history/manual reverse route usage;
- score failure or missing score memory peak;
- nonfinite score;
- no terminal JSON artifact.

Explanatory diagnostics only:

- exact Kalman value comparison;
- row/column Sinkhorn residuals;
- transient process-level memory from external tools if observed.

Nonclaims:

- no same-scalar finite-difference score correctness claim;
- no exact Kalman score claim;
- no HMC/NUTS readiness claim;
- no nonlinear-model score claim;
- no full leaderboard admission from a score-only diagnostic.

## Command

Run with trusted GPU execution:

```bash
python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --device-scope visible --cuda-visible-devices 0 --device /GPU:0 \
  --expect-device-kind gpu \
  --batch-seeds 81120 --num-particles 10000 --time-steps 50 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 --sinkhorn-epsilon 0.5 \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --transport-ad-mode full \
  --row-chunk-size 2500 --col-chunk-size 2500 --particle-chunk-size 2500 \
  --score-mode compact-sensitivity --score-diagnostic-stage score-only \
  --dtype float32 --tf32-mode enabled --history-mode value-only \
  --warmups 0 --repeats 1 \
  --output docs/plans/ledh-lgssm-n10000-t50-compact-score-memory-2500-2026-07-10.json \
  --markdown-output docs/plans/bayesfilter-ledh-lgssm-n10000-t50-compact-score-memory-result-2026-07-10.md
```

## Skeptical Audit

Passed before launch. The command measures the requested full `T=50` score-only
diagnostic and preserves the repaired compact route as the tested mechanism.
The artifact fields answer memory and timing directly. The plan blocks the
known wrong substitutions: CPU-only execution, float64, disabled TF32,
historical full-history reverse routing, and treating score-only diagnostics as
score correctness or leaderboard admission.
