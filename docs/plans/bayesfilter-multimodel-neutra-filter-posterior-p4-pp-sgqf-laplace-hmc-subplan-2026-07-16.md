# P4 Subplan: PP-SGQF Target-Specific Laplace HMC

Date: 2026-07-16

Status: `GEOMETRY_ADMITTED_READY_FOR_HMC`

## Objective And Entry Conditions

Build and test a same-target plain-HMC comparator for admitted `PP-SGQF`
level 2, typed signature
`8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad`.
The UKF comparator showed that identity-mass HMC is a poor default for this
six-probit model family. No UKF mass, samples, comparator, or target identity
may be reused. Its posterior mean may initialize a target-specific SGQF mode
search as a warm-start hypothesis only.

## Geometry Evidence Contract

| Field | Frozen contract |
| --- | --- |
| Question | Can a target-specific SGQF Laplace coordinate system support the unchanged plain-HMC gates? |
| Start | PP-UKF comparator posterior mean, warm start only |
| Mode route | Batched TensorFlow SGQF value/score, damped Newton, at most 8 iterations |
| Hessian | Centered finite difference of the analytic SGQF score, all 12 perturbations in one batch |
| FD steps | `1e-4` for iterations; terminal comparison with `5e-5` |
| Newton trust radius | rescale each proposed source-coordinate direction to infinity norm at most `1.0`; record raw and realized norms |
| Line search | fixed multipliers `(1,0.5,0.25,0.125,0.0625,0)`; choose highest finite status-valid SGQF value |
| Mode pass | infinity score norm `<=1e-4`, finite/status valid, or final step infinity norm `<=1e-5` with value improvement `<=1e-8` |
| Hessian stability | relative Frobenius gap between terminal FD steps `<=1e-3`; both finite and symmetric |
| Precision | terminal negative Hessian, symmetrized; eigenvalue floor `max(1e-8,1e-6*largest_abs_eigenvalue)` |
| Coordinate map | `theta=center+z@factor.T`, `factor=chol(inverse(regularized_precision))`, constant Jacobian included |
| Vetoes | identity/hash drift, invalid SGQF status, nonfinite mode/Hessian, no mode convergence, unstable Hessian, or invalid factor |
| Not concluded | SGQF exactness, superiority, NeuTra quality, calibration, robustness, or readiness |

No long SGQF HMC run begins from an invalid or unconverged Laplace diagnostic.

## Admitted Geometry And Frozen HMC Runtime

Geometry artifact:
`phase-p4/PP-SGQF/laplace-geometry/attempt-01-20260715T165000Z`, result
SHA-256 `b54343fdee59c3f86ffb8f8ac69ba0ea31b7a0c780a4f2eb290374df060cabc3`.
It passed after four mode evaluations with final score infinity norm
`5.80e-05`, Hessian step-size relative gap `4.00e-09`, no clipped precision
eigenvalues, regularized condition number `104.98`, and affine round-trip/value/
score gaps at or below `1.80e-16`.

The frozen HMC runtime is:

- four chains initialized at the Laplace center plus the parent subplan's fixed
  source-coordinate offsets, mapped into the SGQF affine coordinates;
- eight leapfrog steps;
- step-size grid `(0.05,0.10,0.20,0.30,0.40,0.50)`;
- `64` burn-in plus `128` tuning draws per probe;
- health-valid maximum minimum rank-normalized bulk ESS selection, grid-order
  tie break; acceptance remains explanatory only;
- probe root seed `(20260716,9300)`, warm-up seed `(20260716,9401)`, retained
  seed `(20260716,9501)`;
- warm-up chunks `1000`, minimum `2000`, recent window `1000`, cap `10000`,
  modern R-hat `<=1.05`;
- retained chunks `2000`, minimum `4000`, cap `10000`, modern R-hat `<=1.01`,
  minimum bulk ESS `>=1000`, minimum tail ESS `>=400`; and
- separate immutable warm-up and retained archives in SGQF source coordinates.

Skeptical runtime audit: `PASS`. The ladder brackets the successful UKF affine
scale but uses the distinct SGQF factor, target, identity, mode, and seeds. It
does not use UKF mass or draws. Short probes nominate only; fresh sequential
evidence controls admission.

## Defaults And Failure Interpretation

The UKF posterior mean is not assumed to be the SGQF mode. It is accepted only
as an inexpensive start because SGQF and UKF passed the same PF value screen;
the SGQF score, line search, and terminal mode gate must independently validate
it. Local Laplace curvature is a tuning hypothesis, not posterior evidence.
Failure blocks this geometry candidate only and triggers either a bounded SGQF
windowed-mass repair or comparator closure, subject to remaining budget.

## Required Artifacts And Handoff

1. Verify SGQF admission hashes and reconstruct the repository-issued identity.
2. Write every Newton iteration: point, value, score norm, Hessian spectrum,
   regularization, proposed step, line-search values/status, selected multiplier,
   and stopping decision.
3. Write terminal FD-stability, precision, covariance, factor, round-trip, and
   value/score chain-rule evidence with recursive hashes.
4. On geometry pass, refresh this subplan with the frozen HMC ladder and execute
   it. On geometry failure, write `COMPARATOR_BLOCKED_GEOMETRY` or a reviewed
   bounded repair; do not borrow UKF geometry.

## Skeptical Audit

Decision: `PASS_FOR_GEOMETRY_DIAGNOSTIC_ONLY`.

This avoids repeating a known bad identity-mass default without importing a
different filter's mass. The source target is evaluated directly for all
optimization and curvature evidence. The early geometry gate is cheaper and
more discriminating than another 10,000-draw failed warm-up, while the later
HMC promotion criteria remain unchanged.
