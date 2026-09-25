# A10 fresh confirmation comparison

All continuous aggregates below are descriptive. MSE is normalized filtering-mean error against the independent reference. Failed/missing cases remain visible.

| d | Method | Completed / eligible | MSE (completed) | ESS mean / minimum | Guide + build s | Particle s |
|---|---|---|---:|---:|---:|---:|
| 1 | baseline | 12/12 / 12/12 | 0.001879755 | 383.29 / 62.46 | 3.059 | 0.417 |
| 1 | guide | 12/12 / 12/12 | 0.001755443 | 383.61 / 143.32 | 3.071 | 0.368 |
| 1 | stable | 12/12 / 12/12 | 0.001653346 | 375.14 / 164.18 | 3.161 | 0.365 |
| 1 | full | 12/12 / 12/12 | 0.001523218 | 374.61 / 114.69 | 3.161 | 0.488 |
| 1 | transition | 12/12 / 12/12 | 0.002004215 | 322.02 / 20.20 | 0.000 | 0.204 |
| 1 | stationary_prior | 12/12 / 12/12 | 0.005627447 | 255.31 / 4.86 | 0.000 | 0.242 |
| 1 | sgqf_gaussian | 12/12 / 12/12 | 0.00591393 | 313.75 / 2.71 | 0.904 | 0.212 |
| 1 | sgqf_joint | 12/12 / 12/12 | 0.002061025 | 379.80 / 18.01 | 1.101 | 0.213 |
| 4 | baseline | 12/12 / 11/12 | 0.002396514 | 325.01 / 46.18 | 18.967 | 1.315 |
| 4 | guide | 12/12 / 11/12 | 0.002376697 | 323.52 / 45.74 | 19.481 | 1.099 |
| 4 | stable | 12/12 / 11/12 | 0.003099855 | 304.27 / 2.62 | 21.339 | 1.100 |
| 4 | full | 12/12 / 11/12 | 0.002721518 | 303.89 / 14.06 | 21.339 | 1.244 |
| 4 | transition | 12/12 / 11/12 | 0.009263219 | 181.43 / 1.16 | 0.000 | 0.218 |
| 4 | stationary_prior | 12/12 / 10/12 | 0.02251168 | 76.36 / 1.38 | 0.000 | 0.262 |
| 4 | sgqf_gaussian | 12/12 / 11/12 | 0.006459392 | 197.77 / 4.93 | 2.378 | 0.238 |
| 4 | sgqf_joint | 12/12 / 11/12 | 0.002617996 | 313.45 / 20.55 | 2.620 | 0.237 |

Particle time is per N512, T20 repetition. Build time is per fitted path and includes the appropriate guide. The full mixture reuses the stable TT fit; its standalone cost includes that fit. Compilation and fixed execution order are included, so this is not a randomized runtime benchmark.

Machine-readable coverage, numerical checks, unmodified inference, controls, guide routes, reference failures and source-integrity checks: comparison.json.
