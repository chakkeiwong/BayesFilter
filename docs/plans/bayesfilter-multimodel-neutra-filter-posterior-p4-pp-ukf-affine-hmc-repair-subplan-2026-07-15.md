# P4 Repair Subplan: PP-UKF Affine-Mass Plain HMC

Date: 2026-07-15

Status: `READY_FOR_EXECUTION`

## Objective And Entry Evidence

Repair the failed `PP-UKF` identity-mass comparator by freezing an affine HMC
coordinate system from its preserved tuning/warm-up evidence, then repeat
kernel nomination and convergence confirmation with fresh seeds.

The immutable source is
`phase-p4/PP-UKF/plain-hmc/attempt-02-20260715T133100Z`, result SHA-256
`d0f894257ec3c93a26e2900327637ea17e69d0a50c2ce87417e6c704c9a726f2`.
It completed with:

- exact typed target signature
  `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30`;
- ten healthy 1,000-draw warm-up chunks and no retained posterior draws;
- selected identity-mass step `0.032`, acceptance about `0.993-0.996`;
- final recent-window rank R-hat `1.1894` and folded R-hat `1.0604`;
- no energy, nonfinite, target-status, or movement veto; and
- within-chain source-coordinate covariance eigenvalues approximately
  `(0.0112,0.0255,0.0758,0.5108,1.0306,1.3170)`, condition number `117.8`.

The observed failure is sampler geometry/nonconvergence, not a target or
filter failure.

## Evidence Contract

| Field | Frozen repair contract |
| --- | --- |
| Question | Does a target-bound affine mass repair make the same PP-UKF posterior sampleable under the original modern diagnostics? |
| Source evidence | All 10,000 preserved identity-mass warm-up draws, tuning-only |
| Coordinate map | `theta = center + z @ factor.T`, with `center` the pooled warm-up mean and `factor=chol(regularized within-chain covariance)` |
| Covariance estimator | Center within each chain, pool residual outer products, denominator `draws*chains-chains` |
| Regularization | Symmetrize; eigenvalue floor `max(1e-8, 1e-6 * largest_eigenvalue)`; reconstruct and Cholesky |
| Kernel nomination | Health-valid maximum minimum rank-normalized bulk ESS in model coordinates; grid-order tie break |
| Promotion | Unchanged: warm-up recent-window modern R-hat `<=1.05`; retained modern R-hat `<=1.01`; minimum bulk ESS `>=1000`; minimum tail ESS `>=400`; health/status clear |
| Vetoes | Source/hash/identity drift, invalid covariance/factor, value/score chain-rule failure, nonfinite/energy/status failure, or 10,000-draw cap |
| Not concluded | NeuTra quality, UKF exactness, superiority, calibration, or readiness |

The affine target includes the constant log-absolute determinant. Fresh HMC
draws are transformed back to the admitted six-probit source coordinates for
all warm-up and retained diagnostics and posterior archives.

## Frozen Runtime Design

- Same four initial source-coordinate states as attempt 02, mapped into `z`.
- Eight leapfrog steps.
- Step-size grid `(0.05,0.10,0.20,0.30,0.40,0.50)`.
- `64` burn-in plus `128` tuning draws per probe.
- Probe root seed `(20260715,9000)`, fresh warm-up seed `(20260715,9101)`,
  fresh retained seed `(20260715,9201)`.
- Sequential chunk sizes, minima, R-hat/ESS thresholds, and 10,000 caps are
  unchanged from the parent comparator subplan.
- Source warm-up is used only to construct the mass artifact. It is never
  pooled with fresh warm-up or posterior inference.

## Required Checks, Handoff, And Stops

1. Verify the complete source recursive hash ledger and target identity.
2. Write and hash the affine mass artifact, including source archive hashes,
   estimator, regularization report, center, covariance, factor, and target
   signature.
3. Check affine forward/inverse round-trip and wrapper value/score chain rule.
4. Run fresh trusted GPU/XLA probes and sequential confirmation with separate
   archives and progress checkpoints.
5. Admit only on unchanged final gates. A cap hit remains
   `COMPARATOR_BLOCKED`; do not relax thresholds or reuse source warm-up as
   retained evidence.
6. On success, proceed to PP-SGQF comparator work. On failure, preserve the
   complete geometry result and assess whether the remaining comparator budget
   supports one reviewed full-mass/window repair; otherwise close the cell.

## Skeptical Audit

Decision: `PASS`.

The repair is directly triggered by a healthy but nonconvergent identity-mass
run and a measured condition number of `117.8`. It preserves the exact target
through an explicit bijection and constant Jacobian, uses failed-run draws only
for tuning, uses fresh confirmation seeds, keeps all original diagnostics and
caps, and cannot convert the failed attempt into posterior evidence.

