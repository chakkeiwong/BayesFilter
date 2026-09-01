# Phase 22 Per-Mode/Local GenUT Feasibility Probe

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `LOCAL_GENUT_INFEASIBLE_SCOPE`  
Budget cap: `3600 s` within the unchanged `64800 s` campaign cap  
Input: Phase 8 N=300 bank, bound mode axis `2`, and Phase 20 GenUT implementation  
Output root: `docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase22`

## Objective

Test whether the global GenUT infeasibility is caused by combining the two
sign-separated empirical regions. Split the same immutable bank by the bound
mode diagnostic `theta[:,2] < 0`/`>=0`, renormalize weights within each subset,
and evaluate GenUT feasibility and conditional target/status independently.

This is a local proposal/representation diagnostic. The sign split is not a
finite-run mode-discovery theorem and the two local sigma clouds cannot be
pooled as IID posterior rows.

## Evidence contract

Hard gates: input/protocol/target hash, finite subset tensors, explicit subset
indices and counts, GenUT feasibility receipts, and target/status validity for
every feasible local sigma cloud. A negative central weight is a local
candidate veto, not a code failure. No clipping, reweighting, or mode balancing
is introduced.

Explanatory diagnostics are local skew/kurtosis, central weights, support
excursions, mode fractions, and target values. They cannot promote GenUT to an
authority or establish density fidelity.

## Pre-mortem

- The sign split may be an arbitrary chart rather than a true mode partition;
  record it as a diagnostic and do not call it exhaustive mode coverage.
- One subset may remain too skewed for nonnegative GenUT weights. Preserve that
  negative result and move to LEDH rather than clipping.
- A locally feasible sigma rule may still omit the bridge/global geometry.
  Target/status success is therefore role-limited.

## Exact command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_local_genut_probe_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase22-attempt1
```

## Refresh

If local GenUT is feasible, refresh a proposal-utility screen with explicit
local weights and no IID claim. If it is infeasible in both regions, retain the
fixture as source evidence and shift Phase 23 to the invertible LEDH-PFPF
density contract. A candidate failure is not a real blocker.
