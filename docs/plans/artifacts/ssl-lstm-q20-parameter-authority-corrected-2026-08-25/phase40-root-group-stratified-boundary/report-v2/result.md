# v2.2 Root-Group-Stratified Checkpoint Repair

Status: `PASS_V2_2_CHECKPOINT_SELECTION_AUDIT_RECEIPT`

Selection rule: `min(validation_loss + validation_latent_mean_max_abs + validation_latent_covariance_max_abs_offdiag); ties choose smallest step`

| Precondition | Arm | selected step | selected validation score | selected audit mean | selected audit covariance | terminal audit mean | terminal audit covariance |
|---|---|---:|---:|---:|---:|---:|---:|
| identity | compact | 200 | 9.846842 | 0.520515 | 0.740395 | 0.520515 | 0.740395 |
| identity | wide_low_lr | 200 | 10.257230 | 0.576445 | 0.818326 | 0.576445 | 0.818326 |
| affine | compact | 200 | 6.759979 | 0.513618 | 1.133955 | 0.513618 | 1.133955 |
| affine | wide_low_lr | 150 | 7.105530 | 0.376715 | 0.631711 | 0.364732 | 0.649199 |

Selection used validation rows only; root groups are disjoint across train, validation, and audit.

No IID Gaussian, posterior, HMC, canonical LEDH, or default-readiness claim is made.
