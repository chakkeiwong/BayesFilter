# Phase 13 Repair and Refresh Note

Status: `PASS_AFFINE_ORACLE_REFRESHED_PHASE14`

The affine oracle passed with finite positive-definite covariance and weighted
mean/covariance residuals below `1.2e-15`. This demonstrates that the finite
weighted cloud's first two moments are numerically well-conditioned and that
the learned NeuTra residuals are not forced by a singular covariance alone.
It does not establish a density, mode, posterior, or HMC claim.

Refresh Phase 14 toward an explicit affine-preconditioned representation or
measure diagnostic. Keep the affine map diagnostic-only unless a separate
source-faithful density contract is implemented and audited.
