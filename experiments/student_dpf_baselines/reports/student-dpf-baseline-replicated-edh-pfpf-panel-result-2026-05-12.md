# Student DPF baseline replicated EDH/PFPF panel result

## Date

2026-05-12

## Scope

This report covers the replicated EDH/PFPF panel in the quarantined
student DPF experimental-baseline stream.  It is comparison-only
evidence and does not promote student code, modify vendored snapshots,
or make production BayesFilter claims.

## Command

`python -m experiments.student_dpf_baselines.runners.run_replicated_edh_pfpf_panel`

Working directory: `/home/ubuntu/python/BayesFilter`

## Provenance

- `advanced_particle_filter`: `d2a797c330e11befacbb736b5c86b8d03eb4a389`
- `2026MLCOE`: `020cfd7f2f848afa68432e95e6c6e747d3d2402d`

## Panel

- base fixtures: `range_bearing_gaussian_moderate, range_bearing_gaussian_low_noise`
- reduced horizon: `8`
- seeds: `17, 23, 31`
- particles: `64`
- flow steps: `10`
- runtime warning threshold seconds: `30`
- planned records: `12`

## Implementation Summary

| Implementation | Runs | OK | Failed | Median runtime seconds | Max runtime seconds | Median position RMSE | Median avg ESS | Runtime warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | 6 | 6 | 0 | 0.615342 | 3.67247 | 0.0648082 | 18.2589 | 0 |
| advanced_particle_filter | 6 | 6 | 0 | 0.0178602 | 0.565224 | 0.0648937 | 13.2151 | 0 |

## Fixture Summary

| Implementation / fixture | Runs | OK | Median state RMSE | Median position RMSE | Median final-position error | Median obs proxy RMSE | Median avg ESS | Minimum min ESS | Median resampling count | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE/range_bearing_gaussian_low_noise | 3 | 3 | 0.131712 | 0.0652021 | 0.019639 | 0.0235993 | 11.7002 | 3.39599 | 8 | 0.616938 |
| 2026MLCOE/range_bearing_gaussian_moderate | 3 | 3 | 0.0738508 | 0.0642094 | 0.0607893 | 0.0730309 | 24.1437 | 9.12639 | 6 | 0.586528 |
| advanced_particle_filter/range_bearing_gaussian_low_noise | 3 | 3 | 0.116553 | 0.0643605 | 0.0266764 | 0.0176801 | 17.3402 | 1.29583 | 6 | 0.0188052 |
| advanced_particle_filter/range_bearing_gaussian_moderate | 3 | 3 | 0.101325 | 0.0654269 | 0.10611 | 0.0733206 | 9.25542 | 2.96348 | 8 | 0.0169152 |

## Low-Noise Pressure

### advanced_particle_filter

- any pressure signal: `True`
- `median_average_ess_decreased`: `False`
- `minimum_min_ess_decreased`: `True`
- `median_resampling_count_increased`: `False`
- `median_runtime_seconds_increased`: `True`
- `median_position_rmse_increased`: `False`
- `median_observation_proxy_rmse_increased`: `False`

### 2026MLCOE

- any pressure signal: `True`
- `median_average_ess_decreased`: `True`
- `minimum_min_ess_decreased`: `True`
- `median_resampling_count_increased`: `True`
- `median_runtime_seconds_increased`: `True`
- `median_position_rmse_increased`: `True`
- `median_observation_proxy_rmse_increased`: `False`

## Hypothesis Results

- `R1_both_paths_remain_runnable`: `supported_all_planned_runs_ok`
- `R2_low_observation_noise_increases_pressure`: `supported_directional_pressure_signal_observed`
- `R3_proxy_comparison_remains_interpretable`: `supported_proxy_only`
- `R4_runtime_remains_bounded`: `supported_no_runtime_warnings`
- `R5_next_baseline_decision`: `replicated_panel_ready`

## Decision

`replicated_panel_ready`

## Interpretation

The panel result is proxy evidence only.  Latent-state and position
RMSE are evaluated against the shared simulated fixtures.  ESS and
resampling semantics are implementation-specific diagnostics.  The
report does not use student agreement or likelihood values as
correctness evidence.

## Next Phase Recommendation

Use the replicated EDH/PFPF panel as a quarantined experimental baseline artifact, then test whether the same adapter discipline can support a controlled full-horizon sensitivity panel.
