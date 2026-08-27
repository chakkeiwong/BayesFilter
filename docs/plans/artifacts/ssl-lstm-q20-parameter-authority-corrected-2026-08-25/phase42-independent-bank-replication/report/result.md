# v2.4 Two-Bank Replication Report

Status: `PASS_V2_4_TWO_BANK_REPLICATION_REPORT`

Explanatory branch: `bank_to_bank_variability_repair_triggered` (not a statistical ranking)

| Source | rows | roots | ESS | neg-mode | theta mean[0] |
|---|---:|---:|---:|---:|---:|
| authority | 256 | 122 | 0.952283 | 0.530069 | 0.289568 |
| bank_a | 256 | 103 | 0.801812 | 0.756588 | 3.550030 |
| bank_b | 256 | 128 | 0.946687 | 0.517590 | 1.013180 |
| old_v2_2_train | 232 | 108 | 0.950551 | 0.533571 | 0.330821 |
| old_v2_2_validation | 12 | 8 | 0.983826 | 0.491953 | 1.181110 |
| old_v2_2_audit | 12 | 6 | 0.996452 | 0.491717 | -1.422781 |

| Arm | old validation mean/cov | bank A mean/cov | bank B mean/cov |
|---|---:|---:|---:|
| affine:compact | 0.424126 / 0.655261 | 1.138738 / 0.896074 | 0.199921 / 0.390038 |
| affine:wide_low_lr | 0.462624 / 0.637627 | 1.099791 / 0.661917 | 0.096337 / 0.208018 |
| identity:compact | 0.547410 / 0.893340 | 0.914067 / 0.679319 | 0.198587 / 0.344945 |
| identity:wide_low_lr | 0.583848 / 1.012230 | 1.045722 / 0.774507 | 0.247726 / 0.377084 |

Hard boundary gates passed. Bank A/B differences remain descriptive and trigger another support/particle diagnostic; no whitening or posterior claim is made.
