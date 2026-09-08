# v2.3 Frozen-Training Independent Theta Audit

Status: `PASS_V2_3_INDEPENDENT_AUDIT_BOUNDARY`

The old v2.2 training rows are the only optimizer input. The fresh M0 bank is evaluated at the terminal step only.

| Arm | Precondition | Status | Fresh mean max | Fresh covariance offdiag max |
|---|---|---|---:|---:|
| identity:compact | identity | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 0.241250 | 0.285645 |
| identity:wide_low_lr | identity | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 0.178388 | 0.322732 |
| affine:compact | affine | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 0.400899 | 0.469012 |
| affine:wide_low_lr | affine | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | 0.420763 | 0.483285 |

This is role-limited independent-bank evidence; no IID, posterior, HMC, LEDH, or default claim is made.
