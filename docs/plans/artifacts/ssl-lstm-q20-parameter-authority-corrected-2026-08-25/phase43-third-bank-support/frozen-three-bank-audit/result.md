# v2.5 Three-Bank Frozen-Training Theta Audit

Status: `PASS_V2_5_THREE_BANK_BOUNDARY`

Each arm uses one trainer state trained on the old v2.2 train split, then evaluates banks A, B, and C after the final update.

| Arm | Precondition | State hash | A mean/cov | B mean/cov | C mean/cov |
|---|---|---|---:|---:|---:|
| identity:compact | identity | `True` | 0.914067 / 0.679319 | 0.198587 / 0.344945 | 0.182851 / 0.399164 |
| identity:wide_low_lr | identity | `True` | 1.045722 / 0.774507 | 0.247726 / 0.377084 | 0.235210 / 0.476592 |
| affine:compact | affine | `True` | 1.138738 / 0.896074 | 0.199921 / 0.390038 | 0.217778 / 0.432846 |
| affine:wide_low_lr | affine | `True` | 1.099791 / 0.661917 | 0.096337 / 0.208018 | 0.298276 / 0.508351 |

This is role-limited support evidence. Exact v2.4 state-hash reconstruction is required; no IID Gaussian whitening, posterior correctness, or statistical superiority is established.
