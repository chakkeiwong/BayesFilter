# LGSSM Kalman Zero-Bias CI Criterion Amendment

Date: 2026-07-20
Status: `PROPOSED_FOR_FUTURE_CLAIMS_DIAGNOSTIC_APPLIED_TO_PRESERVED_ARTIFACTS`

## Decision

The historical score tolerance `[-5%,+5%]` is not derived from the LGSSM DGP,
the Kalman likelihood, a downstream loss, or a scientific decision threshold.
It is therefore retained only as historical engineering-policy evidence and is
not the primary scientific criterion for future claims.

The primary criterion going forward is:

> For every reported output, the two-sided simultaneous 95% confidence interval
> for the mean relative bias must contain zero.

The family is the six outputs: value and five HMC-coordinate scores. The
Bonferroni-Student critical value is `3.036283222821165` for 16 seeds. Ordinary
coordinate-wise 95% intervals are reported as explanatory diagnostics, not as
the familywise decision.

This is a zero-bias test, not an equivalence proof. Failure to reject zero does
not establish that the candidate and Kalman are practically interchangeable.
An equivalence claim would require a separate, scientifically justified margin
from the application or downstream decision.

## Reclassification Of Preserved Claims

The diagnostic artifact is:

`docs/benchmarks/artifacts/lgssm_kalman_zero_bias_ci_20260720/aggregate.json`

Under the revised zero-bias criterion:

| Scope | Value | `q_scale` | Overall simultaneous zero-bias status |
| --- | --- | --- | --- |
| `N=5000` | Rejected | Rejected | Rejected |
| `N=10000` | Rejected | Rejected | Rejected |

For `N=5000`, the simultaneous relative-bias intervals are:

- value: `[+0.1116%,+0.1848%]`, excluding zero;
- `q_scale`: `[-17.72%,-2.11%]`, excluding zero.

The ordinary coordinate-wise 95% `q_scale` interval is
`[-15.39%,-4.43%]`, also excluding zero. Thus the revised criterion does not
make the `N=5000` `q_scale` result pass; it removes the ungrounded `5%` margin
from the primary interpretation.

## Why `q_scale` Is The Concerning Coordinate

This is a diagnosis, not yet a proof of a code defect.

1. `q_scale` enters the DGP through both transition covariance
   `Q=q_scale^2 I` and stationary initial covariance
   `P0=q_scale^2/(1-phi^2)`. The candidate path also scales initial and
   transition particle noise by `q_scale`, so its total derivative carries all
   of these dependencies and the Contract E reset dependence.
2. At `T=50`, the exact Kalman `q_scale` score is cancellation-sensitive. Its
   time-local predictive HMC increments have RMS `1.0537` but sum to only
   `-0.6710`; there are 20 positive and 30 negative increments. Small persistent
   finite-particle/reset errors can therefore remain visible in the terminal
   sum.
3. The `N=5000` claim has a mean relative `q_scale` bias of `-9.91%`, standard
   error `2.57%`, and simultaneous interval `[-17.72%,-2.11%]`. Its mean is
   `3.86` standard errors below the Kalman target, despite being less than one
   single-seed standard deviation away. This is evidence of a mean shift, not
   merely large seed scatter.
4. The leading unresolved explanations are finite-particle covariance/moment
   error and Contract E reset derivative composition. The predeclared next
   diagnostic is a same-observation/same-stream time-local decomposition of
   stationary, transition/proposal, observation-weight/normalization,
   carried-weight, and reset contributions, each checked against its own
   partial scalar derivative.

## Decision Table

| Decision | Criterion status | Veto/status | Next action | Nonclaim |
| --- | --- | --- | --- | --- |
| Replace `5%` as primary scientific criterion | Adopted for future claims; old screen preserved | Requires fresh plan for a new claim | Use simultaneous zero-bias CI | No equivalence margin implied |
| `N=5000` `q_scale` | Zero bias rejected | Mean shift detected | Run time-local decomposition | Not proof of implementation bug |
| `N=10000` `q_scale` | Zero bias rejected more strongly | Mean shift detected | Run time-local decomposition | No monotone `1/N` claim |
| Nonlinear transfer | Blocked | LGSSM zero-bias criterion not met | Do not transfer yet | No nonlinear conclusion |

## Artifact And Reproducibility

The diagnostic was computed from preserved engineering-valid artifacts without
rerunning the algorithm. Original campaign artifacts remain immutable and retain
their historical `±5%` fields. The new report records both ordinary and
simultaneous intervals and the exact source hashes.
