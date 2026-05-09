# Student DPF baseline linear-stress result

## Date

2026-05-10

## Reference Agreement

| Implementation | Runs | OK | Failed | Max Kalman log-likelihood error | Max mean RMSE | Median runtime seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026MLCOE | 45 | 45 | 0 | 2.84217e-14 | 2.13764e-15 | 0.0748205 |
| advanced_particle_filter | 45 | 45 | 0 | 7.10543e-15 | 2.20332e-15 | 0.0128747 |

## Advanced Bootstrap-PF Diagnostics

| Fixture / particles | Median PF log-likelihood error | Median avg ESS | Min avg ESS | Median resampling count |
| --- | ---: | ---: | ---: | ---: |
| lgssm_1d_long/N=128 | 0.471026 | 79.9997 | 78.7019 | 19 |
| lgssm_1d_long/N=512 | 0.205961 | 315.673 | 314.22 | 17 |
| lgssm_1d_long/N=64 | 0.518797 | 39.9127 | 39.6885 | 17 |
| lgssm_cv_2d_long/N=128 | 2.26901 | 43.5292 | 43.1761 | 39 |
| lgssm_cv_2d_long/N=512 | 1.51607 | 172.086 | 169.181 | 39 |
| lgssm_cv_2d_long/N=64 | 12.2277 | 22.2596 | 20.7457 | 39 |
| lgssm_cv_2d_low_noise/N=128 | 9.06772 | 21.102 | 17.0158 | 50 |
| lgssm_cv_2d_low_noise/N=512 | 0.766504 | 83.2624 | 80.0396 | 50 |
| lgssm_cv_2d_low_noise/N=64 | 24.148 | 10.6926 | 10.1708 | 50 |

## H1 Interpretation

H1 is supported for particle diagnostics: Kalman paths remain reference-consistent, while the advanced bootstrap-PF diagnostics show materially larger log-likelihood error under the low-noise stress fixture.  MLCOE particle diagnostics remain unsupported in this adapter cycle.
