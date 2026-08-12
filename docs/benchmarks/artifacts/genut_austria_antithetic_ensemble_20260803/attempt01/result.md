# GenUT Austria SIR Antithetic-Ensemble Result

Status: `ANTITHETIC_PROMOTION_VETO_INVALID_OR_DERIVATIVE_FAILURE`

The comparison uses the same number of complete GenUT evaluations in both
arms. SGQF is an explanatory approximation, not truth.

## K=1

| Coordinate | Anti variance | Independent variance | Ratio | Log-ratio interval | Nominated |
|---|---:|---:|---:|---:|---|
| value | 0.482596 | 0.343745 | 1.40394 | [-1.052, 1.604] | explanatory |
| log_kappa_scale | 387878 | 17826.5 | 21.7586 | [-1.253, 4.491] | explanatory |
| log_nu_scale | 75389.9 | 4283.69 | 17.5993 | [-1.056, 5.161] | explanatory |
| log_observation_noise_scale | 42247.6 | 10300.3 | 4.10159 | [-0.584, 5.055] | explanatory |

## K=2

| Coordinate | Anti variance | Independent variance | Ratio | Log-ratio interval | Nominated |
|---|---:|---:|---:|---:|---|
| value | 0.532667 | 0.279778 | 1.90389 | [-0.09502, 2.161] | explanatory |
| log_kappa_scale | 201622 | 82207.1 | 2.45261 | [-0.8009, 4.2] | explanatory |
| log_nu_scale | 2.82695e+07 | 16335.7 | 1730.54 | [0.9137, 7.702] | explanatory |
| log_observation_noise_scale | 32266.3 | 1747.03 | 18.4692 | [1.414, 5.494] | explanatory |

## K=4

| Coordinate | Anti variance | Independent variance | Ratio | Log-ratio interval | Nominated |
|---|---:|---:|---:|---:|---|
| value | 0.45463 | 0.115228 | 3.94547 | [0, 3.598] | False |
| log_kappa_scale | 236627 | 149566 | 1.58209 | [-0.1959, 4.005] | False |
| log_nu_scale | 62786.4 | 38567.9 | 1.62794 | [-0.01685, 2.936] | False |
| log_observation_noise_scale | 18804.2 | 7539.67 | 2.49403 | [-0.8339, 2.349] | False |

## Decision

At least one constituent or the same-scalar derivative audit failed; no variance promotion is permitted, but all completed rows are retained.

No score-accuracy ranking is supported because Austria has no exact
`T=20` likelihood/score oracle in this experiment.
