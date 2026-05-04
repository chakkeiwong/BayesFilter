# Audit: factor backend and spectral derivative certification gates

## Date

2026-05-05

## Purpose

This note closes the BayesFilter-owned control-layer gates for factor backend
classification and SVD/eigen derivative promotion.  It does not certify any
client TensorFlow, QR, or SVD backend by itself.

## Backend Classification Policy

Every backend must be classified independently for:

- value exactness;
- derivative evidence;
- compiled/eager status;
- approximation label;
- HMC target eligibility.

BayesFilter now exposes `audit_factor_backend` and
`FactorBackendAuditResult`.  A backend can become `target_candidate` only when
it is value-exact, derivative-checked, compiled-supported, and has no blockers.
Value-only covariance Kalman remains exact for LGSSM likelihoods but blocked
for derivative/HMC promotion.

## Spectral Derivative Policy

SVD/eigen derivative claims require:

- finite spectral values;
- minimum gap above tolerance;
- finite-difference derivative evidence;
- JVP/VJP parity evidence.

BayesFilter now exposes `certify_spectral_derivative_region` and
`SpectralDerivativeCertificationResult`.  Small or repeated gaps produce the
warning label `spectral_gap_too_small` and block HMC eligibility.

## Validation

Focused tests verify:

- value-only covariance Kalman remains HMC-blocked;
- a value/derivative/compiled fake QR backend can become a target candidate;
- unlabeled approximations are blocked;
- spectral certification passes away from small gaps only when numerical checks
  are declared;
- near-repeated values block derivative certification and HMC eligibility.

## Remaining Blockers

- Client QR/SVD/TensorFlow backends need backend-specific parity and stress
  tests before promotion.
- MathDevMCP formula obligations should be attached to any future documented
  derivative formula, not to this metadata gate alone.
- HMC sampler promotion remains blocked until real target and chain diagnostics
  pass.
