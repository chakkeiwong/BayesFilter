# v2.7 Five-Bank Frozen-Training Theta Audit

Status: `PASS_V2_7_INDEPENDENT_N512_BOUNDARY`

| Arm | A | B | C | N512-a | N512-b |
|---|---:|---:|---:|---:|---:|
| identity:compact | 0.914067 / 0.679319 | 0.198587 / 0.344945 | 0.182851 / 0.399164 | 0.324063 / 0.469106 | 0.221470 / 0.214194 |
| identity:wide_low_lr | 1.045722 / 0.774507 | 0.247726 / 0.377084 | 0.235210 / 0.476592 | 0.422946 / 0.411835 | 0.228665 / 0.220628 |
| affine:compact | 1.138738 / 0.896074 | 0.199921 / 0.390038 | 0.217778 / 0.432846 | 0.381323 / 0.375972 | 0.368506 / 0.680965 |
| affine:wide_low_lr | 1.099791 / 0.661917 | 0.096337 / 0.208018 | 0.298276 / 0.508351 | 0.389876 / 0.289003 | 0.407516 / 0.538055 |

Role-limited support evidence; no IID Gaussian whitening, posterior correctness, or statistical ranking is established.
