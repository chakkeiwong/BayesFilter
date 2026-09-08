# Corrected Theta Checkpoint Repair

Status: `PASS_CHECKPOINT_SELECTION_AUDIT_RECEIPT`

Selection rule: `min(validation_loss + validation_latent_mean_max_abs + validation_latent_covariance_max_abs_offdiag); ties choose smallest step`

| Precondition | Arm | selected step | selected validation score | selected audit mean | selected audit covariance | terminal audit mean | terminal audit covariance |
|---|---|---:|---:|---:|---:|---:|---:|
| identity | compact | 200 | 10.792858 | 0.222255 | 0.524386 | 0.222255 | 0.524386 |
| identity | wide_low_lr | 200 | 11.647487 | 0.340518 | 0.544977 | 0.340518 | 0.544977 |
| affine | compact | 200 | 7.310311 | 0.503940 | 0.675621 | 0.503940 | 0.675621 |
| affine | wide_low_lr | 150 | 8.178224 | 0.276358 | 0.397779 | 0.354079 | 0.601880 |

Selection used validation rows only. Audit rows were evaluated after selection and remain descriptive.

No IID Gaussian, posterior, HMC, canonical LEDH, or default-readiness claim is made.
