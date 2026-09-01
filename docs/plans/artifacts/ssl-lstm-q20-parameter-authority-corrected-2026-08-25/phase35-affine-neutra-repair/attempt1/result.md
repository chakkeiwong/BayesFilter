# Corrected q=20 GPU/XLA NeuTra Boundary

Status: `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED`

The screen trains only on theta in R^4 rows with a batch-native weighted transport. It does not launch HMC.

| Arm | Status | Batch | Audit status |
|---|---|---:|---|
| compact | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | `40` | `True` |
| wide_low_lr | `PASS_NEUTRA_BOUNDARY_CANDIDATE` | `40` | `True` |

## Decision

This is GPU/XLA batch-native candidate evidence. Loss and latent covariance remain explanatory; no whitening or posterior claim is made.

## Nonclaims

- no SMC-U authority or posterior correctness claim
- no IID Gaussian whitening theorem
- no HMC or default promotion
