# q20 budget after validation repair

Timing basis: `historical_GPU_measurements_new_work_counts`.

| Scope | Updates | Validation target rows | Raw hours | Hours with factor 2 |
| --- | ---: | ---: | ---: | ---: |
| calibration | 1,024 | 12,288 | 1.381 | 2.762 |
| floor_first_bank | 18,432 | 82,944 | 20.190 | 40.381 |
| floor_validation_cap | 18,432 | 1,327,104 | 56.884 | 113.767 |
| full_cap | 294,912 | 2,211,840 | 344.903 | 689.806 |

Old mixed floor/all-rungs reservation: 449.375 hours.

Available campaign allowance: 33.809 hours, including 11.316 diagnostic hours.

These are training scenarios, not a complete campaign quote. Unpriced work:

- HMC tuning across actual L and learned maps
- classical preparation
- posterior and confirmation chains
- replica exchange and chart mixtures
- reference
- cache storage/checkpoint and process overhead
- training initialization and beta-change preflight

Interpretation:

- update and validation counts are hypotheses, not demonstrated required training
- first-bank floor scenario assumes every assessment resolves without expanding
- cap scenario assumes every history reaches its maximum and every needed map expands fully
- factor two is an engineering reserve, not a statistical runtime bound
- no guarantee of posterior qualification at any priced cap
- fresh source timing was unavailable; changed cache and decision logic were tested separately
