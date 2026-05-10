# Student DPF baseline EDH/PFPF adapter-spike result

## Date

2026-05-11

## Scope

This report covers the bounded EDH/PFPF adapter spike selected by MP4.
It is comparison-only evidence.  It does not promote student code,
modify vendored snapshots, or make production BayesFilter claims.

## Command

`python -m experiments.student_dpf_baselines.runners.run_edh_pfpf_adapter_spike`

Working directory: `/home/ubuntu/python/BayesFilter`

## Provenance

- `advanced_particle_filter`: `d2a797c330e11befacbb736b5c86b8d03eb4a389`
- `2026MLCOE`: `020cfd7f2f848afa68432e95e6c6e747d3d2402d`

## Panel

- base fixture: `range_bearing_gaussian_moderate`
- reduced horizon: `8`
- particles: `64`
- flow steps: `10`
- seed: `17`

## Implementation Summary

| Implementation | Method | Status | Runtime seconds | State RMSE | Position RMSE | Final-position error | Avg ESS | Min ESS | Resampling count |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | PFPF_EDH | `ok` | 3.90794 | 0.0824308 | 0.0710176 | 0.102514 | 24.1437 | 12.8388 | 6 |
| advanced_particle_filter | EDHParticleFilter | `ok` | 0.559431 | 0.118031 | 0.0778454 | 0.107727 | 8.44569 | 2.96348 | 8 |

## Cross Summary

- `status`: available_proxy_only
- `position_rmse_difference`: 0.00682785
- `state_rmse_difference`: 0.0356001
- `runtime_ratio_first_over_second`: 0.143152
- `interpretation`: Differences are proxy diagnostics only and are not correctness claims.

## Hypothesis Results

- `E1_advanced_edh_pfpf_runs`: `supported`
- `E2_mlcoe_pfpf_edh_runs`: `supported`
- `E3_proxy_comparison_interpretable`: `supported_proxy_only`
- `E4_next_phase_decision`: `adapter_spike_success_needs_replication`

## Decision

`adapter_spike_success_needs_replication`

## Interpretation

The spike result is proxy evidence only.  Latent-state and position
RMSE are evaluated against the shared simulated fixture.  ESS and
resampling semantics are implementation-specific diagnostics.  The
report does not use student agreement or likelihood values as
correctness evidence.

## Next Phase Recommendation

Create a replicated EDH/PFPF panel with both nonlinear fixtures, multiple seeds, and the same adapter-owned bridges.
