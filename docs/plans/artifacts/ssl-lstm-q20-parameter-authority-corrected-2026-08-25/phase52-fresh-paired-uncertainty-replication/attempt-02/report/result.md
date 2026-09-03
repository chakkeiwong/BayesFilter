# v3.4 Six-Bank Paired Proposal Report

Status: `PASS_V3_4_FRESH_PAIRED_REPORT`
Branch: `fresh_geometry_uncertainty_incompatible`

| Replicate | Support mean0 | Geometry mean0 | Support negative mass | Geometry negative mass | Support ESS | Geometry ESS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.030323 | 0.792305 | 0.491473 | 0.505658 | 0.623410 | 0.855114 |
| 2 | 0.222369 | 0.427997 | 0.518741 | 0.521846 | 0.634125 | 0.852869 |
| 3 | -0.630075 | 0.044803 | 0.549981 | 0.477894 | 0.577043 | 0.822552 |
| 4 | -0.603512 | -0.164142 | 0.516688 | 0.517703 | 0.619004 | 0.833827 |
| 5 | 0.648913 | 0.047312 | 0.459094 | 0.525674 | 0.747127 | 0.818587 |
| 6 | 0.348910 | 0.256842 | 0.476077 | 0.505027 | 0.797233 | 0.861300 |

| Metric | Support spread | Geometry spread | Geometry - support | 95% lower | 95% upper | Upper <= 0? |
|---|---:|---:|---:|---:|---:|---|
| `theta_mean_0` | 1.278988 | 0.956447 | -0.322541 | -0.895794 | 0.182382 | False |
| `covariance_offdiag_max_abs` | 2.222133 | 0.719168 | -1.502965 | -1.620721 | -0.555580 | True |
| `negative_mode_fraction` | 0.090887 | 0.047780 | -0.043107 | -0.046140 | -0.011081 | True |
| `root_count` | 11.000000 | 9.000000 | -2.000000 | -4.000000 | 6.000000 | False |
| `weighted_ess_fraction` | 0.220190 | 0.042712 | -0.177478 | -0.181442 | -0.013805 | True |

The intervals describe this six-bank replication only. They do not establish a population ranking, posterior correctness, IID whitening, HMC readiness, or default readiness.
