# Repair of scale-sensitive XLA singular-value decompositions

The03486 diagnostic reproduces inaccurate existing outputs with condition about6
at unchanged1e-12/1e-6 matrix scales: checked Kalman condition telemetry reports3
or4.21 instead of6.01; rectangular SRUKF stack covariance errors are about54%,
conditional gain errors51--61%, and log likelihood differs by up to.44. At unit
scale one gain also differs by2.0e-10. These are errors on well-conditioned
inputs, not an ill-conditioning exception. The original source/XLA outputs
remain preserved failure evidence. NumPy LAPACK and the existing non-XLA TF
route are independent comparison authorities for the same algebra.

Implement a shared real thin-SVD evaluation: divide each matrix by its maximum
absolute entry, call XLA SVD with the repository's existing100 sweeps/binary64
epsilon, and restore the singular-value scale. Preserve zero handling, all rank
cutoffs, support tolerances, order and likelihood formulas. Bind raw XLA output
shapes explicitly. Graph-reference execution retains TF's standard SVD. Apply
first to reproduced checked-Kalman telemetry and the two rectangular-factor
calls; do not blanket replace unresolved SVD sites. Inspect other callers under
the parent audit and record dispositions before terminal closure.

The helper preserves real thin-SVD differentiation using TF2.19's published
local implementation linalg_grad.py::_SvdGrad and _SafeReciprocal. Retain its
1e-20 reciprocal broadening convention; this is not authorization for a new
regularizer. Use invariant objectives so arbitrary vector signs cannot fail or
falsely pass a derivative comparison. Fixed-branch analytical QR filtering
scores retain their existing algorithms and require consumer regression.

Revise the audit allocation to12 workers/2400seconds, within unchanged total
campaign caps, to cover the reproduced repairs and both backends. Qualify: batched square/tall/wide/scalar
matrices; zero, deficient and repeated/near-tied singular values; scales1e-140
through1e140 with relative reconstruction residuals; original TF pullbacks and
independent directional finite differences; same-input HLO and one trace.
Then renew exact affected primitive/filter tests and a complete linear SRUKF
scale-equivariance fixture at multiple horizons. Compare complete likelihood,
filtered mean/covariance, rank and on-support diagnostics against graph mode
and a closed-form Kalman reference. No accurate output may be inferred from
finiteness alone. Record CPU/GPU and graph/XLA status; current tolerances stay.

New mismatches are repair triggers and block promotion. A rank/support/cutoff
change without independent mathematical support, unrecorded source mutation or
budget exhaustion stops the dependent run. Memory/time comparisons for this
helper and full route join the campaign's terminal matched-cost work. No
posterior, HMC or scientific promotion follows from the engineering repair.

Review: scaling alone leaves default convergence too loose near repeated modes;
convergence tolerance alone leaves tiny-input stopping problems. Both are needed.
A custom gradient must not silently remove the derivative accepted by the old
primitive. Preserve its exact real thin-SVD rule and test it separately. The
value-only adaptive-rank route must still not advertise an analytical score.

The source search found28 SVD calls (including explicit graph-reference branches
and diagnostic-only modules). Preserve the function/source-hash inventory as
svd-scale-source-inventory-03486.json. Only the three independently reproduced
Kalman/rectangular-factor calls are changed by this first repair. Remaining
callers require scope/scale dispositions in E6; listing them is not a defect
count or evidence that they are safe.
