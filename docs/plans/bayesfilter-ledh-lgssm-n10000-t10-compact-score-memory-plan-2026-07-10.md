# LGSSM LEDH N10000 T10 Compact Score Memory Plan

Date: 2026-07-10

## Question

After the LGSSM score wiring repair, does the compact score route complete for
`N=10000,T=10` with `row/col/particle = 2500/2500/2500`, and what peak GPU
score memory is reported?

## Command Contract

Run the LGSSM runner with:

- trusted GPU execution;
- `--num-particles 10000`;
- `--time-steps 10`;
- one seed, `81120`;
- `--row-chunk-size 2500`;
- `--col-chunk-size 2500`;
- `--particle-chunk-size 2500`;
- `--score-mode compact-sensitivity`;
- `--score-diagnostic-stage score-only`;
- `--dtype float32`;
- `--tf32-mode enabled`;
- `--history-mode value-only`.

## Pass / Veto Conditions

Pass for this diagnostic means:

- terminal artifact completed;
- score stage completed;
- score route is
  `compact_forward_sensitivity_no_autodiff_same_scalar_lgssm_ledh_pfpf_ot`;
- `uses_full_history_reverse_route` is false;
- score tensors are on GPU;
- precision records `dtype=float32` and TF32 enabled;
- `score_gpu_memory_info_after.peak` is present.

Veto conditions:

- CPU execution represented as trusted GPU evidence;
- `float64` execution;
- TF32 disabled;
- full-history/manual reverse route usage;
- score failure or missing score memory peak;
- nonfinite score.

## Nonclaims

This is not N10000,T50 full-row admission, not same-scalar FD correctness, not
exact Kalman score evidence, and not HMC readiness.
