# Phase 21 Actual q=20 GenUT Feasibility/Status Probe

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `GENUT_Q20_INFEASIBLE_SCOPE`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Input: Phase 8 metadata-bound N=300 bank and Phase 20 GenUT implementation  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase21`

## Objective

Evaluate the source-faithful GenUT feasibility equations on the actual q=20
weighted cloud, then evaluate transformed sigma points with the q=20 target
only if the equations produce nonnegative, finite sigma weights. This phase
answers whether GenUT can be a global one-step q=20 quadrature arm under the
current authority cloud; it does not treat sigma points as an authority bank.

## Evidence contract

Hard gates: metadata/protocol/target hash; finite weighted moments; positive
discriminants, offsets, and central weight; weight sum; and, conditional on
feasibility, finite/valid q=20 target status for all 41 sigma points. If the
central weight is negative or another feasibility condition fails, the result
is a valid `INFEASIBLE_SCOPE` diagnostic rather than a harness failure.

Explanatory diagnostics include standardized skew/kurtosis, sigma support,
mode-axis occupancy, and target values. No density, IID, global mode, posterior,
HMC, or default claim is allowed.

## Pre-mortem

- A single global GenUT rule may be infeasible for a multimodal/heavy-tailed
  cloud. That is a scope result; do not clip weights or silently split modes.
- A feasible rule may still collapse global modes into local axis points. Record
  support/mode diagnostics and retain quadrature-only status.
- A target-status success does not repair missing density identity. Preserve the
  role boundary explicitly.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_genut_q20_probe_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase21-attempt1
```

## Refresh

On a feasible/status-valid result, refresh Phase 22 toward a local proposal
utility check, not authority replacement. On infeasibility, refresh Phase 22
to a reviewed per-mode/local GenUT option or close GenUT as a global arm while
retaining the source-faithful fixture. A true blocker requires the master
program's explicit conditions.
