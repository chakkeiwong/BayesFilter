# Student DPF baseline full-horizon EDH/PFPF confirmation result

## Date

2026-05-12

## Scope

This report covers the full-horizon EDH/PFPF confirmation panel in the
quarantined student DPF experimental-baseline stream.  It is
comparison-only evidence and does not promote student code, modify
vendored snapshots, or make production BayesFilter claims.

## Command

`python -m experiments.student_dpf_baselines.runners.run_full_horizon_edh_pfpf_confirmation`

Working directory: `/home/ubuntu/python/BayesFilter`

## Provenance

- `advanced_particle_filter`: `d2a797c330e11befacbb736b5c86b8d03eb4a389`
- `2026MLCOE`: `020cfd7f2f848afa68432e95e6c6e747d3d2402d`

## Panel

- fixtures: `range_bearing_gaussian_moderate, range_bearing_gaussian_low_noise`
- horizon: full fixture horizon, 20 observations
- seeds: `31, 43, 59, 71, 83`
- particles: `128`
- low-noise flow steps: `20`
- moderate-noise flow steps: `10, 20`
- runtime warning threshold seconds: `45`
- planned records: `30`

## Implementation Summary

| Implementation | Runs | OK | Failed | Median runtime seconds | Max runtime seconds | Median position RMSE | Median avg ESS | Runtime warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | 15 | 15 | 0 | 1.00414 | 3.88405 | 0.0639813 | 59.3066 | 0 |
| advanced_particle_filter | 15 | 15 | 0 | 0.0779734 | 0.581302 | 0.0653164 | 22.6455 | 0 |

## Grid Summary

| Implementation / fixture / steps | Runs | OK | Median position RMSE | Median obs proxy RMSE | Median avg ESS | Minimum min ESS | Median resampling count | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE/range_bearing_gaussian_low_noise/N128/steps20 | 5 | 5 | 0.0468415 | 0.0156068 | 42.7296 | 14.9629 | 17 | 1.01099 |
| 2026MLCOE/range_bearing_gaussian_moderate/N128/steps10 | 5 | 5 | 0.0660938 | 0.0774868 | 59.3066 | 11.75 | 11 | 0.610349 |
| 2026MLCOE/range_bearing_gaussian_moderate/N128/steps20 | 5 | 5 | 0.0689353 | 0.0732131 | 64.3471 | 21.3094 | 11 | 1.00374 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/N128/steps20 | 5 | 5 | 0.0475423 | 0.0162379 | 27.6236 | 2.32936 | 18 | 0.0801186 |
| advanced_particle_filter/range_bearing_gaussian_moderate/N128/steps10 | 5 | 5 | 0.0687264 | 0.0822046 | 21.5713 | 1.2855 | 18 | 0.0504168 |
| advanced_particle_filter/range_bearing_gaussian_moderate/N128/steps20 | 5 | 5 | 0.0747989 | 0.0848042 | 20.4721 | 1.07129 | 19 | 0.0779734 |

## Low-Noise Confirmation

### advanced_particle_filter

- confirmed: `True`
- `median_average_ess_not_materially_worse`: `True`
- `median_position_rmse_not_materially_worse`: `True`
- `median_observation_proxy_rmse_not_materially_worse`: `True`
- `median_resampling_count_within_horizon`: `True`
- `finite_outputs_dominant`: `True`

### 2026MLCOE

- confirmed: `True`
- `median_average_ess_not_materially_worse`: `True`
- `median_position_rmse_not_materially_worse`: `True`
- `median_observation_proxy_rmse_not_materially_worse`: `True`
- `median_resampling_count_within_horizon`: `True`
- `finite_outputs_dominant`: `True`

## Moderate-Noise Flow-Step Policy

- recommendation: `moderate_keep_both_as_diagnostic`
- rationale: implementation-specific benefit differs

- `advanced_particle_filter` bounded benefit: `False`, runtime ratio 20/10: `1.54658`, position improved: `False`, obs proxy improved: `False`
- `2026MLCOE` bounded benefit: `True`, runtime ratio 20/10: `1.64453`, position improved: `False`, obs proxy improved: `True`

## Clean-Room Specification Inputs

- status: `ready_for_specification`
- fixtures: `range_bearing_gaussian_moderate, range_bearing_gaussian_low_noise`
- horizon: `full_fixture_horizon_20_observations`
- particle count: `128`
- flow-step policy: `{'low_noise': 20, 'moderate': 'moderate_keep_both_as_diagnostic'}`

## Hypothesis Results

- `C1_selected_full_horizon_setting_seed_stable`: `supported_seed_stable`
- `C2_low_noise_128_particle_pressure_reduction_persists`: `supported_low_noise_pressure_reduction_persists`
- `C3_moderate_noise_flow_step_policy_resolved`: `resolved_moderate_keep_both_as_diagnostic`
- `C4_clean_room_baseline_specification_ready`: `supported_clean_room_inputs_ready`
- `C5_next_baseline_decision`: `confirmation_ready_with_caveats`

## Decision

`confirmation_ready_with_caveats`

## Interpretation

The confirmation result is proxy evidence only.  Latent-state and
position RMSE are evaluated against the shared simulated fixtures.
ESS and resampling semantics are implementation-specific
diagnostics.  The report does not use student agreement or
likelihood values as correctness evidence.

## Next Phase Recommendation

Write a caveated clean-room specification only after explicitly carrying forward the low-noise or moderate-flow-step caveats.
