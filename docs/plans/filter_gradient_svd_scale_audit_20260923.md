# Audit of SVD scale sensitivity after controller localization

03470 reproduces a28% XLA operator-norm error on identical small matrices.
03473/03474 verify the homogeneous-scaling/binary64 convergence repair against
independent and original authorities. The endpoint audit must determine whether
this compiler sensitivity also affects actual filtering values or diagnostics.

First inspect the existing checked Kalman spectral telemetry and value-only
rectangular SRUKF direct-stack/conditional endpoints. Execute their actual XLA
functions at identical matrix conditions with scales1e-12,1e-6,1,1e6. Compare
minimum/condition telemetry, singular values/rank, reconstructed covariance,
conditional gain/covariance and Gaussian likelihood to an independent NumPy
LAPACK reference. NumPy is diagnostic only. Evaluate residuals after removing
the known common scale; absolute residuals on tiny matrices are misleading.
This is explanatory attribution, not a promotion gate. Any mismatched rank,
relative covariance/gain error over1e-10 or value error beyond existing1e-10
is a repair trigger; preserve all records. Zero/tiny/repeated/rank-deficient
cases and derivatives become qualification gates for any subsequent repair.

Inspect other SVD callers and classify actual reachability, non-XLA-only or
existing normalization/tolerance controls; a text hit is not proof of a defect.
Preserve exact cutoffs, support rules, likelihood measures and analytical score
definitions. Proposed repairs must evaluate the same factorization accurately;
never hide unresolved ill-conditioning or tune scientific thresholds to match.
Default/reference comparisons and call-chain renewal follow any repaired route.

Reserve8 workers/2400 seconds under unchanged32CPU/52GPU campaign caps.
Initial attribution uses120seconds/backend, source frozen, numbered campaign
artifacts, explicit CPU or verified available non-desktop GPU/memory growth.
A source/shape/reference mismatch invalidates evidence and must be repaired;
exhausted budget stops launches. No posterior/HMC/default-readiness conclusion.

Review: a small Frobenius absolute residual could mask a completely wrong small
factor. Identical condition across scales isolates magnitude sensitivity.
Compare full factor covariance and conditional outputs rather than sign-ambiguous
singular vectors. Keep scientific gates fixed and current source hashes in the
runner manifests. This plan does not authorize touching frozen external runs.

03486 reproduces large relative errors on well-conditioned actual endpoints.
Execute the [bounded SVD repair](filter_gradient_svd_scale_repair_20260923.md)
under the same campaign caps. This attribution pass is not numerical acceptance.
