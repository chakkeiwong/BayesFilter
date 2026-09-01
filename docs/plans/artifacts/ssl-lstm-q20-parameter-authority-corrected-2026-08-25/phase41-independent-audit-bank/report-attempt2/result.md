# v2.3 Independent Theta Audit Report

Status: `PASS_V2_3_INDEPENDENT_AUDIT_REPORT`

Explanatory branch: `finite_holdout_mismatch_is_plausible` (not a statistical ranking)

| Source | rows | roots | ESS | neg-mode | theta mean[0] | affine mean max | affine covariance offdiag max |
|---|---:|---:|---:|---:|---:|---:|---:|
| old_v2_2_train | 232 | 108 | 0.950551 | 0.533571 | 0.330821 | 4.5102810375396984e-17 | 6.574601973952099e-16 |
| old_v2_2_validation | 12 | 8 | 0.983826 | 0.491953 | 1.181110 | 0.5377407073974609 | 0.7335909605026245 |
| old_v2_2_audit | 12 | 6 | 0.996452 | 0.491717 | -1.422781 | 0.39963823556900024 | 0.5799530744552612 |
| fresh_v2_3_M0 | 256 | 121 | 0.979248 | 0.543599 | 0.226798 | n/a | n/a |

| Arm | old validation mean/cov | fresh audit mean/cov |
|---|---:|---:|
| affine:compact | 0.376301 / 0.667096 | 0.400899 / 0.469012 |
| affine:wide_low_lr | 0.449028 / 0.825264 | 0.420763 / 0.483285 |
| identity:compact | 0.549102 / 0.882597 | 0.241250 / 0.285645 |
| identity:wide_low_lr | 0.584842 / 1.003937 | 0.178388 / 0.322732 |

The hard boundary passed, but residuals remain descriptive and no whitening or posterior claim is made.
