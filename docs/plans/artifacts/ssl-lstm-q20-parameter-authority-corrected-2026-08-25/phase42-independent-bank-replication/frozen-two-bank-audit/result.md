# v2.4 Two-Bank Frozen-Training Theta Audit

Status: `PASS_V2_4_TWO_BANK_BOUNDARY`

Each arm uses one trainer state trained on the old v2.2 train split, then evaluates banks A and B after the final update.

| Arm | Precondition | Status | A mean/cov | B mean/cov |
|---|---|---|---:|---:|
| identity:compact | identity | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 0.914067 / 0.679319 | 0.198587 / 0.344945 |
| identity:wide_low_lr | identity | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 1.045722 / 0.774507 | 0.247726 / 0.377084 |
| affine:compact | affine | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 1.138738 / 0.896074 | 0.199921 / 0.390038 |
| affine:wide_low_lr | affine | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 1.099791 / 0.661917 | 0.096337 / 0.208018 |

This is role-limited replication evidence. It does not establish IID Gaussian whitening, posterior correctness, or statistical superiority.
