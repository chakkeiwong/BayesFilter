# Student DPF baseline MP2 nonlinear reference-panel result

## Date

2026-05-10

## Scope

This report covers the MP2 nonlinear reference/proxy-metric spine for
the quarantined student DPF experimental-baseline stream.  It is
comparison-only evidence and does not promote student code into
production.

## Implementation Summary

| Implementation | Runs | OK | Failed | Min position RMSE | Max position RMSE | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | 16 | 16 | 0 | 0.0469209 | 1.27813 | 0.13444 |
| advanced_particle_filter | 14 | 14 | 0 | 0.0455544 | 0.0692199 | 0.00496878 |

## Method Summary

| Implementation / fixture / method | Runs | OK | Median state RMSE | Median position RMSE | Median final-position error | Median avg ESS | Min avg ESS | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE/range_bearing_gaussian_low_noise/BPF | 5 | 5 | 0.11706 | 0.0485862 | 0.0660111 | 31.8502 | 29.2674 | 0.144304 |
| 2026MLCOE/range_bearing_gaussian_low_noise/EKF | 1 | 1 | 0.101547 | 0.0469209 | 0.0592777 | null | null | 0.0812553 |
| 2026MLCOE/range_bearing_gaussian_low_noise/EKF_origin_diagnostic | 1 | 1 | 0.720381 | 0.984584 | 1.7013 | null | null | 0.0754729 |
| 2026MLCOE/range_bearing_gaussian_low_noise/UKF | 1 | 1 | 0.102019 | 0.046965 | 0.0598255 | null | null | 0.130122 |
| 2026MLCOE/range_bearing_gaussian_moderate/BPF | 5 | 5 | 0.085892 | 0.0680171 | 0.107421 | 70.1926 | 55.716 | 0.130599 |
| 2026MLCOE/range_bearing_gaussian_moderate/EKF | 1 | 1 | 0.0890458 | 0.0661617 | 0.114942 | null | null | 0.255998 |
| 2026MLCOE/range_bearing_gaussian_moderate/EKF_origin_diagnostic | 1 | 1 | 0.927937 | 1.27813 | 2.16827 | null | null | 0.0746584 |
| 2026MLCOE/range_bearing_gaussian_moderate/UKF | 1 | 1 | 0.0891922 | 0.0665197 | 0.115354 | null | null | 0.270737 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/BPF | 5 | 5 | 0.120702 | 0.0467642 | 0.0630499 | 49.4518 | 47.0437 | 0.00517709 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/EKF | 1 | 1 | 0.101613 | 0.0469176 | 0.0597985 | null | null | 0.00239606 |
| advanced_particle_filter/range_bearing_gaussian_low_noise/UKF | 1 | 1 | 0.10254 | 0.0469524 | 0.0602821 | null | null | 0.00529338 |
| advanced_particle_filter/range_bearing_gaussian_moderate/BPF | 5 | 5 | 0.0898371 | 0.066008 | 0.101766 | 113.918 | 111.36 | 0.00469674 |
| advanced_particle_filter/range_bearing_gaussian_moderate/EKF | 1 | 1 | 0.0885614 | 0.0658618 | 0.112841 | null | null | 0.00247325 |
| advanced_particle_filter/range_bearing_gaussian_moderate/UKF | 1 | 1 | 0.0897583 | 0.0663859 | 0.1131 | null | null | 0.00518165 |

## Comparison Summary

- `2026MLCOE/range_bearing_gaussian_low_noise/bpf_position_rmse_seed_std`: 0.000705202
- `2026MLCOE/range_bearing_gaussian_low_noise/ekf_position_minus_ukf_position_rmse`: -4.41213e-05
- `2026MLCOE/range_bearing_gaussian_moderate/bpf_position_rmse_seed_std`: 0.00413574
- `2026MLCOE/range_bearing_gaussian_moderate/ekf_position_minus_ukf_position_rmse`: -0.000358002
- `advanced_particle_filter/range_bearing_gaussian_low_noise/bpf_position_rmse_seed_std`: 0.00146125
- `advanced_particle_filter/range_bearing_gaussian_low_noise/ekf_position_minus_ukf_position_rmse`: -3.47693e-05
- `advanced_particle_filter/range_bearing_gaussian_moderate/bpf_position_rmse_seed_std`: 0.00354181
- `advanced_particle_filter/range_bearing_gaussian_moderate/ekf_position_minus_ukf_position_rmse`: -0.00052407

## Hypothesis Results

### n1_shared_nonlinear_fixture

Supported.  Both snapshots produced at least one EKF/UKF result on the shared Gaussian range-bearing fixtures without vendored-code edits.

### n2_mlcoe_ekf_zero_behavior

Supported.  MLCOE EKF is usable away from the origin, while the origin diagnostic is materially worse, consistent with a range-bearing Jacobian initialization artifact.

### n3_nonlinear_pf_degeneracy

Supported for this proxy panel.  BPF summaries show lower ESS or larger RMSE pressure on the low-noise fixture for both snapshots.

### n4_comparison_only_reporting

Supported.  Records include target labels and rely on latent-state RMSE, EKF/UKF diagnostics, PF dispersion, ESS, and runtime rather than student agreement or direct likelihood comparison.

## Interpretation

All MP2 metrics are proxy/reference metrics against a shared Gaussian
range-bearing fixture.  They do not certify either student
implementation as production quality.  Likelihood values are not
used for cross-student conclusions.
