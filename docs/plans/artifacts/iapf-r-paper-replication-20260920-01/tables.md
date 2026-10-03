# Bounded R iAPF replication: numerical tables

Original-author data and unspecified solver/floor settings were not recovered. Published table entries are descriptive comparisons on different simulated observations. Interrupted batches cannot support a success-only ranking; their uncertainty intervals are suppressed because bootstrap cannot correct outcome-dependent missing runs. A false heuristic-veto flag in an incomplete group is not a passed screen.

Worker time: 248.38/1800 seconds; 7/8 launches.

| Attempt | d | Requested | Completed iAPF | Status | Seconds |
|---|---:|---:|---:|---|---:|
| attempt01-pilot-d5 | 5 | 1 | 1 | complete | 8.82 |
| attempt02-replication-d5 | 5 | 32 | 4 | failed | 37.47 |
| attempt03-pilot-d10 | 10 | 1 | 1 | complete | 50.08 |
| attempt04-repaired-pilot-d5 | 5 | 1 | 1 | complete | 6.17 |
| attempt05-replication-d5 | 5 | 32 | 1 | failed | 10.30 |
| attempt06-replication-d10 | 10 | 32 | 2 | failed | 74.99 |
| attempt07-paper-scale-oracles | 80 | 3 | 0 | complete | 60.55 |

Exact-twist control (not fitted-iAPF replication): {"checks": 15, "passed": 15, "dimensions": [5, 10, 20, 40, 80], "max_absolute_log_error": 1.81898940354586e-12, "max_terminal_weight_spread": 4.00177668780088e-11, "max_fully_adapted_difference": 0.0}.

## d=5, data seed 55000005

Complete paired repeats: 4/32. All requested repeats complete: False.

| Method | Mean Zhat/Z | Mean 95% CI | SD | Paper SD | Resamples | Mean N | Seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| iapf | 0.9789 | N/A | 0.01084 | 0.09 | 2 | 2000 | 6.021 |
| bpf | 0.8169 | N/A | 0.5124 | 0.51 | 99 | 1e+04 | 0.3307 |
| fully_adapted | 0.9709 | N/A | 0.06656 | 0.1 | 31.25 | 5000 | 0.139 |
| sis | 4.537e-147 | N/A | 9.073e-147 | N/A | 0 | 1e+04 | 0.7265 |

| iAPF variance / comparator variance | 95% interval | Ranking status |
|---|---|---|
| bpf | N/A | ineligible |
| fully_adapted | N/A | ineligible |
| sis | N/A | ineligible |

| Innovation situation | iAPF prefix log MSE | BPF | Fully adapted | SIS | Supported heuristic veto |
|---|---:|---:|---:|---:|---|
| ordinary | 0.0008128 | 0.1769 | 0.003232 | 2.967e+04 | False |
| large_innovation | 0.0007834 | 0.1859 | 0.003637 | 4.193e+04 | False |

Fit diagnostics: {"count": 2500, "loss_underflows": 0, "residual_over_half": 16}.

## d=5, data seed 57000005

Complete paired repeats: 1/32. All requested repeats complete: False.

| Method | Mean Zhat/Z | Mean 95% CI | SD | Paper SD | Resamples | Mean N | Seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| iapf | 1.043 | N/A | N/A | 0.09 | 2 | 2000 | 5.219 |
| bpf | 1.057 | N/A | N/A | 0.51 | 99 | 1e+04 | 0.307 |
| fully_adapted | 0.9271 | N/A | N/A | 0.1 | 31 | 5000 | 0.137 |
| sis | 1.252e-152 | N/A | N/A | N/A | 0 | 1e+04 | 0.244 |

| iAPF variance / comparator variance | 95% interval | Ranking status |
|---|---|---|

| Innovation situation | iAPF prefix log MSE | BPF | Fully adapted | SIS | Supported heuristic veto |
|---|---:|---:|---:|---:|---|
| ordinary | 0.001832 | 0.06593 | 0.002315 | 2.958e+04 | False |
| large_innovation | 0.004447 | 0.08351 | 0.003335 | 3.47e+04 | False |

Fit diagnostics: {"count": 600, "loss_underflows": 0, "residual_over_half": 7}.

## d=10, data seed 57000010

Complete paired repeats: 2/32. All requested repeats complete: False.

| Method | Mean Zhat/Z | Mean 95% CI | SD | Paper SD | Resamples | Mean N | Seconds |
|---|---:|---|---:|---:|---:|---:|---:|
| iapf | 0.5962 | N/A | 0.1485 | 0.14 | 48 | 4000 | 30.7 |
| bpf | 0.3341 | N/A | 0.3625 | 6.4 | 99 | 1e+04 | 0.495 |
| fully_adapted | 1.028 | N/A | 0.4024 | 0.17 | 53.5 | 5000 | 0.218 |
| sis | 0 | N/A | 4.941e-324 | N/A | 0 | 1e+04 | 0.45 |

| iAPF variance / comparator variance | 95% interval | Ranking status |
|---|---|---|
| bpf | N/A | ineligible |
| fully_adapted | N/A | ineligible |
| sis | N/A | ineligible |

| Innovation situation | iAPF prefix log MSE | BPF | Fully adapted | SIS | Supported heuristic veto |
|---|---:|---:|---:|---:|---|
| ordinary | 0.1314 | 2.14 | 0.02744 | 1.737e+05 | False |
| large_innovation | 0.1576 | 3.105 | 0.03725 | 2.319e+05 | False |

Fit diagnostics: {"count": 3300, "loss_underflows": 0, "residual_over_half": 413}.
