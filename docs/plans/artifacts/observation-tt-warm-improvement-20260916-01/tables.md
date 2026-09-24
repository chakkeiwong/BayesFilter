# A09 fresh confirmation results

Descriptive sequence averages on matched, reference-valid cases. MSE is normalized by stationary variance. ESS is out of 512.

| d | Method | Sequences | MSE | Mean/min ESS | Max weight | Build s | Particle s / repetition |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 12 | 0.0018203 | 381.14 / 55.63 | 0.1182 | 1.923 | 0.392 |
| 1 | capacity | 12 | 0.0016838 | 385.10 / 162.59 | 0.0535 | 1.923 | 0.355 |
| 1 | preservation | 12 | 0.0016727 | 384.39 / 157.15 | 0.0555 | 1.965 | 0.319 |
| 1 | tt_pair_block | 12 | 0.0018654 | 381.59 / 110.65 | 0.0717 | 2.273 | 0.346 |
| 1 | transition | 12 | 0.0023192 | 316.42 / 29.21 | 0.0926 | 0.000 | 0.169 |
| 1 | stationary_prior | 12 | 0.0048183 | 250.75 / 13.82 | 0.2308 | 0.000 | 0.200 |
| 1 | sgqf_gaussian | 12 | 0.0039619 | 311.83 / 19.48 | 0.2141 | 0.000 | 0.186 |
| 1 | sgqf_joint | 12 | 0.0021032 | 375.16 / 93.05 | 0.0722 | 0.166 | 0.183 |
| 4 | baseline | 11 | 0.0043825 | 322.20 / 3.41 | 0.5182 | 9.385 | 0.882 |
| 4 | capacity | 11 | 0.0042766 | 325.17 / 3.66 | 0.5094 | 15.792 | 1.235 |
| 4 | preservation | 11 | 0.0043281 | 322.39 / 3.36 | 0.5360 | 15.736 | 1.070 |
| 4 | tt_pair_block | 11 | 0.0046212 | 315.04 / 1.29 | 0.8800 | 28.513 | 0.748 |
| 4 | transition | 11 | 0.0068766 | 181.83 / 1.70 | 0.7599 | 0.000 | 0.187 |
| 4 | stationary_prior | 11 | 0.0220395 | 78.75 / 1.15 | 0.9301 | 0.000 | 0.218 |
| 4 | sgqf_gaussian | 11 | 0.0089454 | 201.75 / 1.90 | 0.7256 | 0.000 | 0.199 |
| 4 | sgqf_joint | 11 | 0.0045756 | 316.85 / 4.67 | 0.3356 | 0.196 | 0.198 |

## Primary filtering contrasts

Candidate minus standalone warm baseline; simultaneous 95% exploratory sequence-bootstrap intervals.

| d | Candidate | Mean difference | Lower | Upper | Improvement supported |
|---|---|---:|---:|---:|---|
| 1 | capacity | -0.00013650 | -0.00024803 | -0.00002497 | True |
| 1 | preservation | -0.00014760 | -0.00025589 | -0.00003931 | True |
| 4 | capacity | -0.00010589 | N/A | N/A | False |
| 4 | preservation | -0.00005433 | N/A | N/A | False |

## Fixed-target regression

Nine audited target/time cases per dimension and method; descriptive calibration evidence only.

| d | Method | Initial H2 | Fitted H2 | Amplitude RMS | Max KKT | Max Gram condition |
|---|---|---:|---:|---:|---:|---:|
| 1 | baseline | 0.003286 | 0.001078 | 0.04744 | 5.7274e-05 | 115.9 |
| 1 | capacity | 0.003210 | 0.000303 | 0.02534 | 1.4666e-05 | 360.7 |
| 1 | preservation | 0.003210 | 0.000292 | 0.02525 | 3.2714e-06 | 360.7 |
| 4 | baseline | 0.012490 | 0.003260 | 0.08641 | 0.0048748 | 3.63e+08 |
| 4 | capacity | 0.012230 | 0.001532 | 0.06117 | 0.013454 | 5.235e+10 |
| 4 | preservation | 0.012230 | 0.002087 | 0.07104 | 0.013667 | 2.857e+10 |

## Observed conditional heuristic losses

Descriptive differences only. These trigger the predeclared default-promotion screen, not rejection of TT repair.

| d | TT arm | Heuristic | Observation regime | Sequences | Mean difference |
|---|---|---|---|---:|---:|
| 4 | baseline | transition | large | 11 | 0.03832167 |
| 4 | baseline | stationary_prior | large | 11 | 0.02762804 |
| 4 | baseline | sgqf_joint | large | 11 | 0.00010982 |
| 4 | capacity | transition | large | 11 | 0.03778982 |
| 4 | capacity | stationary_prior | large | 11 | 0.02709619 |
| 4 | preservation | transition | large | 11 | 0.03829392 |
| 4 | preservation | stationary_prior | large | 11 | 0.02760029 |
| 4 | preservation | sgqf_joint | large | 11 | 0.00008207 |

All conditional contrasts, log-evidence screens, resampling rates, guide time and total/amortized timings are retained in report.json.
