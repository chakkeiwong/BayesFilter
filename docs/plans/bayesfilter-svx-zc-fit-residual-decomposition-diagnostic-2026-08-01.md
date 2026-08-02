# SVX-ZC Fit Residual Decomposition Diagnostic

Date: 2026-08-01
Source run: `docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt07/rank6.json`

## Finding

The `0.0564383308` residual at the final adjacent update is not an ALS
convergence failure. It is the irreducible projection error of the target onto
the declared degree-8 Legendre product space.

The fitter residual is the weighted RMS error of the fitted **square-root
density target** at the order-17 quadrature nodes:

\[
  r = \sqrt{\sum_i w_i(\hat h_i-h_i)^2/\sum_iw_i}.
\]

It is not a parameter-regression error and not the dense likelihood gap.

## Final-Step Decomposition

For the final `t=9` adjacent target, using the same nodes and weights as the
fitter:

| Quantity | Value |
| --- | ---: |
| Target weighted norm | `0.162172840338` |
| Degree-8 product-space projection norm | `0.152035341186` |
| Error outside degree-8 product space | `0.0564383307144` |
| Reported ALS residual | `0.0564383308167` |
| Difference from projection floor | about `1.0e-10` |
| Rank-6 tail within degree-8 coefficient space | `3.3982e-6` |

The residual therefore comes from basis truncation, not the rank-6 TT
factorization. The degree-8 space has nine basis functions per axis; the
transformed-SV observation and square-root operation are not degree-8
polynomials.

## Independent Checks

- Increasing ALS sweeps from `1` to `16` leaves the final residual at
  `0.056438330817`.
- UKF, norm-balanced, and three random initializations all converge to the
  same residual within numerical noise.
- Direct design-matrix evaluation and the fitted TT evaluation agree to below
  `3e-16` for each tested core update.
- The raw 17-by-17 sampled matrix has a much smaller rank-6 SVD tail, but that
  is not the relevant comparison: it permits arbitrary values at nodes and
  does not enforce the degree-8 polynomial basis.

## UKF Interpretation

The scalar augmented-noise UKF adapter reports mean zero and stationary
variance `1.5625` at every time. Its observation cross-covariance is zero for
the zero-mean multiplicative noise construction, so the UKF does not update
from observations. This explains why the UKF warm start does not improve the
fit, but it is not the cause of the residual plateau.

## Decision

The regression implementation is mechanically behaving as declared. The
active residual veto `<=1e-8` is incompatible with the current degree-8 basis
for this target. Do not relax the veto inside the existing admission result.
The next valid experiment is a reviewed degree/basis-capacity ladder (or an
explicit approximation-error gate) under the same target and coordinate map.
