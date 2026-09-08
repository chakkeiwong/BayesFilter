# v2.6 Four-Bank Frozen-Training Theta Audit

Status: `PASS_V2_6_LARGER_N_BOUNDARY`

Each arm uses one trainer state trained on the old v2.2 train split, then evaluates banks A, B, C, and N=512 after the final update.

| Arm | Precondition | State hash | A mean/cov | B mean/cov | C mean/cov | N=512 mean/cov |
|---|---|---|---:|---:|---:|---:|
| identity:compact | identity | `True` | 0.914067 / 0.679319 | 0.198587 / 0.344945 | 0.182851 / 0.399164 | 0.324063 / 0.469106 |
| identity:wide_low_lr | identity | `True` | 1.045722 / 0.774507 | 0.247726 / 0.377084 | 0.235210 / 0.476592 | 0.422946 / 0.411835 |
| affine:compact | affine | `True` | 1.138738 / 0.896074 | 0.199921 / 0.390038 | 0.217778 / 0.432846 | 0.381323 / 0.375972 |
| affine:wide_low_lr | affine | `True` | 1.099791 / 0.661917 | 0.096337 / 0.208018 | 0.298276 / 0.508351 | 0.389876 / 0.289003 |

This is role-limited support evidence. Exact v2.4 state-hash reconstruction is required; no IID Gaussian whitening, posterior correctness, or statistical superiority is established.
