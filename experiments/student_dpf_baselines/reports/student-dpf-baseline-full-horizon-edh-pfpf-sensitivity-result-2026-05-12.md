# Student DPF baseline full-horizon EDH/PFPF sensitivity result

## Date

2026-05-12

## Scope

This report covers the full-horizon EDH/PFPF sensitivity panel in the
quarantined student DPF experimental-baseline stream.  It is
comparison-only evidence and does not promote student code, modify
vendored snapshots, or make production BayesFilter claims.

## Command

`python -m experiments.student_dpf_baselines.runners.run_full_horizon_edh_pfpf_sensitivity`

Working directory: `/home/ubuntu/python/BayesFilter`

## Provenance

- `advanced_particle_filter`: `d2a797c330e11befacbb736b5c86b8d03eb4a389`
- `2026MLCOE`: `020cfd7f2f848afa68432e95e6c6e747d3d2402d`

## Panel

- fixtures: `range_bearing_gaussian_moderate, range_bearing_gaussian_low_noise`
- horizon: full fixture horizon, 20 observations
- seeds: `17, 23`
- particles: `64, 128`
- flow steps: `10, 20`
- runtime warning threshold seconds: `45`
- planned records: `32`

## Implementation Summary

| Implementation | Runs | OK | Failed | Median runtime seconds | Max runtime seconds | Median position RMSE | Median avg ESS | Runtime warnings |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | 16 | 16 | 0 | 0.990633 | 3.66617 | 0.0537337 | 30.4403 | 0 |
| advanced_particle_filter | 16 | 16 | 0 | 0.0721145 | 0.576604 | 0.0519453 | 19.3242 | 0 |

## Grid Summary

| Implementation / fixture / particles / steps | Runs | OK | Median position RMSE | Median obs proxy RMSE | Median avg ESS | Minimum min ESS | Median resampling count | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE/range_bearing_gaussian_low_noise/N64/steps10 | 2 | 2 | 0.0469951 | 0.0191389 | 14.1657 | 3.39599 | 20 | 0.60574 |
| 2026MLCOE/range_bearing_gaussian_low_noise/N64/steps20 | 2 | 2 | 0.047299 | 0.017241 | 22.4101 | 5.73143 | 17 | 1.00319 |
| 2026MLCOE/range_bearing_gaussian_low_noise/N128/steps10 | 2 | 2 | 0.0484655 | 0.0189767 | 28.6217 | 4.5231 | 20 | 0.609014 |
| 2026MLCOE/range_bearing_gaussian_low_noise/N128/steps20 | 2 | 2 | 0.047955 | 0.016635 | 43.8026 | 13.5635 | 17 | 1.00716 |
| 2026MLCOE/range_bearing_gaussian_moderate/N64/steps10 | 2 | 2 | 0.0709751 | 0.0858296 | 28.8772 | 9.12639 | 12.5 | 2.13282 |
| 2026MLCOE/range_bearing_gaussian_moderate/N64/steps20 | 2 | 2 | 0.0660188 | 0.0747591 | 32.2274 | 10.6065 | 10.5 | 0.996224 |
| 2026MLCOE/range_bearing_gaussian_moderate/N128/steps10 | 2 | 2 | 0.0651872 | 0.0803488 | 58.771 | 25.5192 | 11.5 | 0.608323 |
| 2026MLCOE/range_bearing_gaussian_moderate/N128/steps20 | 2 | 2 | 0.0697974 | 0.0723378 | 63.4594 | 21.187 | 11 | 0.996266 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/N64/steps10 | 2 | 2 | 0.0463517 | 0.0164471 | 18.4415 | 1.29583 | 16.5 | 0.0492268 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/N64/steps20 | 2 | 2 | 0.0478549 | 0.0162262 | 16.3482 | 1.22143 | 18 | 0.0721145 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/N128/steps10 | 2 | 2 | 0.0472809 | 0.0154493 | 30.4044 | 2.20858 | 16.5 | 0.0505594 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/N128/steps20 | 2 | 2 | 0.0466126 | 0.0164528 | 29.5952 | 1.48094 | 18 | 0.0782058 |
| advanced_particle_filter/range_bearing_gaussian_moderate/N64/steps10 | 2 | 2 | 0.0721197 | 0.0821578 | 12.0654 | 1.91165 | 18.5 | 0.312773 |
| advanced_particle_filter/range_bearing_gaussian_moderate/N64/steps20 | 2 | 2 | 0.0696785 | 0.0805892 | 11.9712 | 1.5459 | 19.5 | 0.0755747 |
| advanced_particle_filter/range_bearing_gaussian_moderate/N128/steps10 | 2 | 2 | 0.056792 | 0.0789837 | 23.8582 | 1.33122 | 18.5 | 0.0504101 |
| advanced_particle_filter/range_bearing_gaussian_moderate/N128/steps20 | 2 | 2 | 0.0719022 | 0.0794808 | 21.1528 | 2.38177 | 19.5 | 0.0772175 |

## Particle Sensitivity

- `advanced_particle_filter/range_bearing_gaussian_low_noise/steps_10` pressure reduction signal: `True`
- `advanced_particle_filter/range_bearing_gaussian_low_noise/steps_20` pressure reduction signal: `True`
- `2026MLCOE/range_bearing_gaussian_low_noise/steps_10` pressure reduction signal: `True`
- `2026MLCOE/range_bearing_gaussian_low_noise/steps_20` pressure reduction signal: `True`

## Flow-Step Sensitivity

- `advanced_particle_filter/range_bearing_gaussian_moderate/particles_64` bounded benefit: `True`, runtime ratio 20/10: `0.241628`
- `advanced_particle_filter/range_bearing_gaussian_moderate/particles_128` bounded benefit: `False`, runtime ratio 20/10: `1.53179`
- `advanced_particle_filter/range_bearing_gaussian_low_noise/particles_64` bounded benefit: `True`, runtime ratio 20/10: `1.46494`
- `advanced_particle_filter/range_bearing_gaussian_low_noise/particles_128` bounded benefit: `True`, runtime ratio 20/10: `1.54681`
- `2026MLCOE/range_bearing_gaussian_moderate/particles_64` bounded benefit: `True`, runtime ratio 20/10: `0.467093`
- `2026MLCOE/range_bearing_gaussian_moderate/particles_128` bounded benefit: `True`, runtime ratio 20/10: `1.63773`
- `2026MLCOE/range_bearing_gaussian_low_noise/particles_64` bounded benefit: `True`, runtime ratio 20/10: `1.65615`
- `2026MLCOE/range_bearing_gaussian_low_noise/particles_128` bounded benefit: `True`, runtime ratio 20/10: `1.65376`

## Hypothesis Results

- `FH1_full_horizon_remains_runnable`: `supported_all_planned_runs_ok`
- `FH2_more_particles_reduce_low_noise_pressure`: `supported_pressure_reduction_signal_observed`
- `FH3_more_flow_steps_have_bounded_benefit`: `supported_bounded_benefit_signal_observed`
- `FH4_proxy_comparison_remains_interpretable`: `supported_proxy_only`
- `FH5_next_baseline_decision`: `full_horizon_sensitivity_ready`

## Decision

`full_horizon_sensitivity_ready`

## Interpretation

The panel result is proxy evidence only.  Latent-state and position
RMSE are evaluated against the shared simulated fixtures.  ESS and
resampling semantics are implementation-specific diagnostics.  The
report does not use student agreement or likelihood values as
correctness evidence.

## Next Phase Recommendation

Keep EDH/PFPF as a full-horizon quarantined experimental baseline. Next, write a small confirmation plan that fixes the most useful particle/flow setting and tests additional seeds before any controlled clean-room extraction.
