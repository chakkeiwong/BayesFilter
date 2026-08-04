# SVX-ZC UKF Initializer Default Admission Result

Date: 2026-08-01
Plan: `docs/plans/bayesfilter-svx-zc-ukf-initializer-default-admission-plan-2026-08-01.md`
Terminal artifact: `docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt07/result.json`

## Decision

The existing UKF initializer is now wired into the SVX-ZC fixed-branch route
and is the default initializer for the actual-SV row. The final CPU admission
ladder still admits no rank. UKF initialization does not remove the numerical
fixed-branch blocker.

The exact transformed-SV target, fixed affine coordinate map, degree, order,
rank ladder, fitter, seeds, and residual veto were held fixed relative to the
valid `attempt03` baseline. The scalar UKF moments are from the existing
augmented-noise Gaussian-closure UKF path and are recorded as geometry-only
warm-start information, not exact transformed-SV evidence.

## Rank Results

| Rank | UKF hard result | Max residual | Attempt03 residual | Difference | Dense gap/obs | Max condition |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | blocked rank-saturation | 0.0663069567 | 0.0663071404 | -1.84e-7 | 0.1365298574 | 1.00 |
| 2 | blocked rank-saturation | 0.0566692874 | 0.0566692937 | -6.28e-9 | 0.1031100478 | 4.93 |
| 4 | blocked rank-saturation | 0.0564390909 | 0.0564390933 | -2.44e-9 | 0.1008784833 | 795 |
| 6 | blocked rank-saturation | 0.0564383308 | 0.0564383332 | -2.43e-9 | 0.1008536107 | 1.84e3 |

All four candidates failed only the declared rank-saturation residual gate
(`residual <= 1e-8`). Finite value/score, coordinate/Jacobian consistency,
positivity, retained marginal closure, condition ceiling, same-scalar FD
branch identity, and forbidden-route checks passed for every rank.

The residual changes are descriptive numerical differences from a changed
initializer. There is no statistical ranking evidence and no claim that UKF is
better. The residual plateau remains approximately `5.64e-2` at ranks 4 and 6,
far above the hard veto.

## Wiring Evidence

- Default initializer identity:
  `ukf_whitened_gaussian_sqrt_projection_v1`.
- The comparator manifest records UKF source
  `actual_transformed_sv_augmented_noise_gaussian_closure_ukf` and the explicit
  nonclaim `not exact transformed same-target admission`.
- One-axis and adjacent TT core hashes are recorded separately.
- The P76 projection now consumes the UKF center and local scale when
  evaluating its projected guide; those fields are no longer metadata-only.
- The fixed route branch manifest carries the actual initializer identity.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Failed rank-saturation residual for ranks 1, 2, 4, and 6 |
| Statistically supported ranking | None; this is a deterministic CPU ladder with no replicated stochastic ranking |
| Descriptive-only differences | Residual, dense gap, value, and condition differences |
| Default readiness | Not assessed; SVX-ZC remains out of NeuTra admission |
| Next evidence needed | A fixed-branch capacity/target-representation repair that lowers residual while preserving the exact target and all structural gates |

## Attempt Provenance

- `attempt04`: invalid harness evidence. Candidate score used UKF cores while
  structural recomputation used norm-balanced cores; the first implementation
  also propagated a nonbaseline density floor.
- `attempt05`: numerically valid after those repairs, but its manifest still
  pointed to the historical plan and is retained as diagnostic provenance.
- `attempt06`: valid manifest path, before the final route-level initializer
  identity repair.
- `attempt07`: terminal valid artifact after the route-level identity repair.

## Red-Team And Nonclaims

The strongest alternative explanation is that the fixed basis/rank and two-sweep
ALS representation, not the warm start, controls the residual plateau. The
result would be overturned as an initializer diagnosis if a same-target run
with the same UKF cores but a different fitter capacity materially changed the
residual. No exact filtering, author-source-faithfulness, posterior
correctness, NeuTra training quality, HMC convergence, superiority, or
production/default-readiness claim is made.
