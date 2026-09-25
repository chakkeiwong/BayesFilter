# Bounded R iAPF replication: numerical tables

Original-author data and unspecified solver/floor settings were not recovered. Published table entries are descriptive comparisons on different simulated observations. Interrupted batches cannot support a success-only ranking; their uncertainty intervals are suppressed because bootstrap cannot correct outcome-dependent missing runs. A false heuristic-veto flag in an incomplete group is not a passed screen.

Worker time: 1291.20/1550 seconds; 9/10 launches.

| Attempt | d | Requested | Completed iAPF | Status | Seconds |
|---|---:|---:|---:|---|---:|
| attempt01-sensitivity | 5 | 1 | 0 | failed | 3.37 |
| attempt02-sensitivity | 5 | 1 | 0 | failed | 46.29 |
| attempt03-sensitivity | 5 | 1 | 0 | complete | 110.30 |
| attempt04-replication-d5 | 5 | 32 | 31 | timeout | 400.02 |
| attempt05-replication-d10-part1 | 10 | 16 | 16 | complete | 342.51 |
| attempt06-replication-d5-resume | 5 | 2 | 2 | complete | 17.06 |
| attempt07-replication-d10-part2 | 10 | 16 | 16 | complete | 339.67 |
| attempt08-pilot-d20 | 20 | 1 | 0 | failed | 10.42 |
| attempt09-pilot-d20-nlminb | 20 | 1 | 0 | failed | 21.57 |

## d=5, data seed 65000005

Fitter: relative_l2; equation (15): False; floor power: 2; doubling: first_full_window.

Complete paired repeats: 32/32. All requested repeats complete: True.

| Method | Mean Zhat/Z | Mean 95% CI | SD | Paper SD | Resamples | Mean N | Seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| iapf | 1.001 | [0.9897, 1.013] | 0.03289 | 0.09 | 2 | 2000 | 5.943 |
| bpf | 1.085 | [0.8105, 1.41] | 0.8768 | 0.51 | 99 | 1e+04 | 0.3549 |
| fully_adapted | 0.9714 | [0.9479, 0.9949] | 0.06724 | 0.1 | 30.16 | 5000 | 0.1486 |
| sis | 1.705e-131 | [2.901e-136, 4.985e-131] | 8.919e-131 | N/A | 0 | 1e+04 | 0.2833 |

| iAPF variance / comparator variance | 95% interval | Ranking status |
|---|---|---|
| bpf | [0.0005238, 0.006569] | ineligible |
| fully_adapted | [0.1212, 0.4587] | iapf_lower |
| sis | [3.498e+256, 1.311e+267] | ineligible |

| Innovation situation | iAPF prefix log MSE | BPF | Fully adapted | SIS | Supported heuristic veto |
|---|---:|---:|---:|---:|---|
| ordinary | 0.0008923 | 0.2901 | 0.003203 | 3.115e+04 | False |
| large_innovation | 0.0009294 | 0.286 | 0.003164 | 2.645e+04 | False |

Fit diagnostics: {"count": 19200, "loss_underflows": 0, "residual_over_half": 0}.

## d=10, data seed 65000010

Fitter: relative_l2; equation (15): False; floor power: 2; doubling: first_full_window.

Complete paired repeats: 32/32. All requested repeats complete: True.

| Method | Mean Zhat/Z | Mean 95% CI | SD | Paper SD | Resamples | Mean N | Seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| iapf | 1.004 | [0.9886, 1.02] | 0.04733 | 0.14 | 5.719 | 1969 | 16.42 |
| bpf | 0.2528 | [0.1344, 0.4005] | 0.3855 | 6.4 | 99 | 1e+04 | 0.4544 |
| fully_adapted | 0.9892 | [0.9418, 1.044] | 0.1452 | 0.17 | 52.91 | 5000 | 0.2449 |
| sis | 4.447e-323 | [0, 1.383e-322] | 2.619e-322 | N/A | 0 | 1e+04 | 0.4577 |

| iAPF variance / comparator variance | 95% interval | Ranking status |
|---|---|---|
| bpf | [0.006716, 0.07608] | ineligible |
| fully_adapted | [0.05751, 0.1963] | iapf_lower |
| sis | N/A | ineligible |

| Innovation situation | iAPF prefix log MSE | BPF | Fully adapted | SIS | Supported heuristic veto |
|---|---:|---:|---:|---:|---|
| ordinary | 0.001989 | 3.493 | 0.008583 | 1.842e+05 | False |
| large_innovation | 0.001797 | 3.501 | 0.008292 | 1.576e+05 | False |

Fit diagnostics: {"count": 19200, "loss_underflows": 0, "residual_over_half": 0}.
