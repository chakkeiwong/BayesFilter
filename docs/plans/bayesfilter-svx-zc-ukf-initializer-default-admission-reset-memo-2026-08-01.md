# SVX-ZC UKF Initializer Reset Memo

Date: 2026-08-01

## Active State

The SVX-ZC actual-SV fixed-branch route now defaults to the existing repository
UKF initializer, `ukf_whitened_gaussian_sqrt_projection_v1`. Its UKF center and
local scale are operationally consumed by the projected guide. The scalar UKF
moments come from the existing augmented-noise Gaussian-closure route and are
geometry-only; they are not the exact transformed-SV target.

The terminal admission artifact is:

`docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt07/result.json`

All ranks `(1, 2, 4, 6)` remain blocked by the residual veto `1e-8`. The
residual plateau is approximately `0.05644` for ranks 4 and 6. Structural,
finite, positivity, closure, condition, FD, and route checks pass.

## What Changed

- Added an adapter for existing UKF mean/covariance paths to the repository
  `UKFScoutResult` contract.
- Reused `p76_build_ukf_initializer` for both the one-axis and adjacent seeds.
- Made the UKF center/local scale operational in P76 projection.
- Added explicit initializer identity to fixed-filter and adjacent-route branch
  manifests.
- Added focused SVX-ZC UKF wiring tests.

## Do Not Repeat

- Do not treat the augmented-noise UKF closure as exact transformed-SV truth.
- Do not relax the residual veto because the UKF run is finite.
- Do not compare `attempt04`; it is an invalid mixed-core harness artifact.
- Do not use `attempt05` or `attempt06` as the terminal claim artifact; use
  `attempt07`.
- Do not change the fixed coordinate map or density floor in a follow-up
  initializer comparison.

## Next Smallest Justified Action

Keep SVX-ZC blocked for admission and plan a bounded fixed-branch capacity or
representation repair under the same exact target. The next run must preserve
the UKF initializer as the default, reuse the untouched data contract, and
record whether the residual reduction comes from basis/rank/ALS capacity rather
than from a changed target or warm-start policy.
