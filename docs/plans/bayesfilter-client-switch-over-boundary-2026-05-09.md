# Client Switch-over Boundary After Seven-phase Filtering Pass

## Date

2026-05-09

## Scope

This note records what the completed BayesFilter implementation pass does and
does not authorize for MacroFinance and DSGE switch-over.

## Promoted BayesFilter Capabilities

- Static MacroFinance linear compatibility tests pass for dense and masked QR
  value/score/Hessian fixtures.
- TensorFlow SVD/eigen linear value filtering exists for dense and masked
  observation paths, with implemented-law diagnostics.
- Generic TensorFlow structural protocols expose the innovation block,
  stochastic/endogenous block, deterministic-completion block, and collapsed
  full-state approximation label.
- Generic TensorFlow SVD sigma-point value filters exist for cubature and UKF
  rules.
- TensorFlow CUT4-G and SVD-CUT value filters exist with moment, affine, rank,
  support, point-count, and static-shape tests.
- Smooth-branch SVD-CUT score/Hessian exists for the implemented value law and
  fails closed on active floors or weak spectral gaps.
- CPU graph-compiled parity tests exist for promoted value and derivative
  paths.

## What Is Not Authorized Yet

- Do not change MacroFinance production defaults solely from this pass.  First
  add a MacroFinance-side switch-over PR or adapter layer with rollback tests.
- Do not change DSGE production defaults.  DSGE still needs a model-owned
  causal local filtering target for SGU and exact target-model parity fixtures
  for nonlinear filters.
- Do not claim GPU speedups from these tests.  GPU/CUDA/XLA-GPU claims require
  escalated device probes and benchmark artifacts.
- Do not claim HMC readiness for SVD-CUT on DSGE targets until the exact model,
  backend, derivative branch, compiled parity, and sampler diagnostics all pass.

## Suggested Switch-over Order

1. MacroFinance linear QR value and QR derivatives, guarded by the existing
   compatibility tests.
2. MacroFinance SVD/eigen linear value only for fixtures that require singular
   covariance handling, with diagnostics reviewed.
3. DSGE generic structural value fixtures after the DSGE repository supplies
   local causal filtering targets and parity observations.
4. DSGE SVD-CUT value after target-specific affine/rank/shape gates.
5. HMC only after target-specific smooth-branch derivative and compiled parity
   gates.

## Hypotheses To Test Next

- H1: MacroFinance can replace its static linear QR filtering path with
  BayesFilter without changing likelihood, score, Hessian, or missing-data
  convention.
- H2: Singular linear fixtures in MacroFinance, if any, are value-side SVD/eigen
  needs rather than derivative needs.
- H3: DSGE nonlinear targets can be expressed through the generic
  `TFStructuralStateSpace` protocol without importing DSGE economics into
  BayesFilter.
- H4: SVD-CUT point count is practical only when the declared stochastic
  integration dimension is small enough; GPU/XLA may help throughput but does
  not remove the `2q + 2^q` growth.
- H5: SVD-CUT derivatives are usable for HMC only on separated-spectrum,
  inactive-floor branches; branch frequency must be measured on target models
  before sampler promotion.
