# Audit: Phase 8 MacroFinance large-scale and cross-currency deferral

## Scope

This note records the Phase 8.5 decision for MacroFinance providers that are
not part of the first BayesFilter adapter implementation slice.

MacroFinance commit:

- `e23c31e Document one-country HMC remaining test gates`

## Large-Scale LGSSM Provider

Candidate:

- `large_scale_lgssm_derivative_provider.py`
- `LargeScaleLGSSMDerivativeProvider`
- `LargeScaleLGSSMPosteriorAdapter`

Useful properties:

- Generic LGSSM-shaped provider.
- Explicit parameter layout and parameter units.
- Active analytical derivatives for offsets, transition diagonal entries,
  process-noise scales, observation-loading shifts, and measurement-noise
  scales.
- Masked analytical backend dispatcher through covariance, square-root, and QR
  wrappers.

Why deferred:

- The provider introduces scaling and parameter-unit behavior that should be a
  separate adapter policy.
- Sparse/masked likelihood behavior needs its own BayesFilter missing-data
  contract audit.
- Backend selection and factor diagnostics are richer than the first
  one-country value/derivative bridge.

Second-stage hypothesis:

- H8-LGSSM: the Phase 8 adapter result types are sufficient for the large-scale
  provider if BayesFilter adds explicit mask policy metadata and parameter-unit
  metadata before running parity tests.

Candidate tests for a future pass:

- `tests/test_large_scale_lgssm_derivative_provider.py`
- `tests/test_large_scale_lgssm_missing_data_policy.py`
- `tests/test_masked_qr_sqrt_differentiated_kalman.py`

## Cross-Currency Structural Provider

Candidate:

- `cross_currency_structural_derivative_provider.py`
- `CrossCurrencyStructuralDerivativeProvider`

Useful properties:

- Rich structural parameter names and slices.
- Coverage of long-run means, mean-reversion scales, diffusion scales,
  spillovers, Nelson-Siegel decay parameters, and measurement-error blocks.
- Finite-difference oracle and coverage metadata.

Why deferred:

- It is no longer just a generic LGSSM adapter problem; it includes financial
  structural derivative coverage, block activation policy, measurement
  covariance modes, and finite-difference audit paths.
- The first BayesFilter adapter should not decide cross-currency identification
  or production-readiness semantics.

Second-stage hypothesis:

- H8-CCY: the derivative bridge can wrap cross-currency structural derivatives
  after BayesFilter adds coverage-matrix metadata and finite-difference oracle
  provenance to derivative results.

Candidate tests for a future pass:

- `tests/test_cross_currency_structural_backend_parity.py`
- `tests/test_cross_currency_structural_identification.py`
- `tests/test_provider_economics_contracts.py`

## Production Cross-Currency Provider

Candidate:

- `production_cross_currency_derivative_provider.py`
- `ProductionCrossCurrencyDerivativeProvider`
- `FinalCalibratedCrossCurrencyProvider`

Useful properties:

- Production contract rows.
- Blocker metadata.
- Identification evidence status.
- Sparse derivative backend policy.
- Explicit final ten-country readiness validation.

Why deferred:

- It intentionally includes blockers and readiness policy, not just numerical
  LGSSM conversion.
- BayesFilter should not erase those blockers by wrapping the provider too
  early.
- Final ten-country data/API readiness is a separate gate.

Second-stage hypothesis:

- H8-PROD: a future BayesFilter production adapter should treat blocker tables,
  derivative coverage, and identification evidence as first-class metadata and
  fail closed when final-provider readiness is false.

## Phase 8.5 Decision

The large-scale and cross-currency providers are not blockers for the
one-country Phase 8 pilot.  They should be handled in later passes after
BayesFilter has explicit metadata for masks, parameter units, derivative
coverage, finite-difference oracle provenance, and production readiness.
