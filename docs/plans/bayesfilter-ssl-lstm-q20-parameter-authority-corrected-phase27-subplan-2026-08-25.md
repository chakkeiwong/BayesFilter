# Corrected q=20 Parameter-Authority Phase 27 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Date: 2026-08-25  
Status: `READY_TO_EXECUTE`  
Local cap: 1800 s (from the remaining campaign pool, not additive)

## Question

Does the active q=20 target and the first modular transport boundary operate
on `theta in R^4`, with a common theta measure, while keeping the 60D UKF state
and 20D innovation internal?

## Evidence contract

- Primary gate: static `[N,4]` target evaluation, finite/status-valid rows,
  target signature, and explicit internal dimensions.
- Primary gate: affine chart Jacobian cancellation in the target/proposal
  ratio, with theta and chart log densities stored separately.
- Role gate: ETPF accepts/returns `[N,4]` and its source-moment residuals are
  finite. This does not create a density or IID law.
- Vetoes: wrong rank/shape, non-finite/status-invalid rows, nonzero ratio
  identity beyond tolerance, overwritten artifact, or hidden GPU.
- Explanatory only: mode fraction, covariance residual, negative transform
  entries, and runtime.
- Nonclaims: no authority admission, mode theorem, posterior correctness,
  whitening theorem, HMC readiness, or LEDH claim.

## Commands

First run the two bounded MathDevMCP commands in the master plan and save raw
stdout/stderr. Then run:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase27_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase27-measure-contract
```

The runner must fail closed if the output root exists or if a GPU is visible.

## Repair and refresh

If the run fails, classify the failure as harness, target implementation,
measure/schema, numerical, or candidate. Repair only the smallest issue while
holding target, criteria, hardware class, and budget fixed. Re-run in a fresh
`phase27-attempt2-*` directory and record the prior failure. If all hard gates
pass, write the result and refresh Phase 28 with the measured batch cost,
target status counts, and any support warning; do not promote inherited
epsilon, schedule, or geometry values.

## Planned artifact

`result.json`, `result.md`, `mathdevmcp-jacobian.txt`,
`mathdevmcp-rank.txt`, and a run manifest with source hashes, environment,
seed, device policy, and wall time.

