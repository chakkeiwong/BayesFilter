# A10 fresh confirmation comparison

All continuous aggregates below are descriptive. MSE is normalized filtering-mean error against the independent reference. Failed/missing cases remain visible.

| d | Method | Completed / eligible | MSE (completed) | ESS mean / minimum | Guide + build s | Particle s |
|---|---|---|---:|---:|---:|---:|
| 1 | baseline | 12/12 / 12/12 | 0.001747021 | 376.71 / 122.68 | 2.819 | 0.396 |
| 1 | guide | 12/12 / 12/12 | 0.001645962 | 376.40 / 117.12 | 2.906 | 0.350 |
| 1 | stable | 12/12 / 12/12 | 0.001718318 | 372.27 / 88.38 | 2.936 | 0.354 |
| 1 | full | 12/12 / 12/12 | 0.00181066 | 366.24 / 42.53 | 2.936 | 0.467 |
| 1 | transition | 12/12 / 12/12 | 0.002100885 | 320.37 / 64.46 | 0.000 | 0.192 |
| 1 | stationary_prior | 12/12 / 12/12 | 0.007275155 | 253.86 / 1.69 | 0.000 | 0.225 |
| 1 | sgqf_gaussian | 12/12 / 12/12 | 0.008617023 | 316.15 / 4.97 | 0.810 | 0.203 |
| 1 | sgqf_joint | 12/12 / 12/12 | 0.002382256 | 370.94 / 14.63 | 0.994 | 0.200 |
| 4 | baseline | 12/12 / 12/12 | 0.002289973 | 323.55 / 26.82 | 17.807 | 1.286 |
| 4 | guide | 12/12 / 12/12 | 0.002342476 | 324.82 / 29.56 | 17.981 | 1.108 |
| 4 | stable | 11/12 / 11/12 | 0.002612686 | 300.19 / 38.21 | 20.239 | 1.111 |
| 4 | full | 11/12 / 11/12 | 0.003277141 | 299.50 / 2.13 | 20.239 | 1.237 |
| 4 | transition | 12/12 / 12/12 | 0.006834566 | 178.29 / 5.62 | 0.000 | 0.191 |
| 4 | stationary_prior | 12/12 / 11/12 | 0.02548585 | 77.06 / 1.27 | 0.000 | 0.229 |
| 4 | sgqf_gaussian | 12/12 / 11/12 | 0.00647018 | 202.11 / 11.21 | 1.891 | 0.205 |
| 4 | sgqf_joint | 12/12 / 11/12 | 0.002444711 | 313.90 / 38.74 | 2.096 | 0.207 |

Particle time is per N512, T20 repetition. Build time is per fitted path and includes the appropriate guide. The full mixture reuses the stable TT fit; its standalone cost includes that fit. Compilation and fixed execution order are included, so this is not a randomized runtime benchmark.

Machine-readable coverage, numerical checks, unmodified inference, controls, guide routes, reference failures and source-integrity checks: comparison.json.
