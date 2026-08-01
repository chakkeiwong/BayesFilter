# GenUT Chapter 18b Structural Leaderboard Result

Date: 2026-07-22

Status: `CANDIDATE_LEADERBOARD_ROW_INCLUDED`

The row is included as a candidate GenUT extension under the existing
`STR-UKF-five-probit-T100-structural-innovation-v1` target. It is not automatic method admission or
default promotion.

## Value and Score

| Quantity | GenUT mean | GenUT 95% CI | Existing structural UKF |
|---|---:|---:|---:|
| value | -124.16018105 | [-124.72241867, -123.59794342] | -124.46262801 |
| source score `rho_source_probit` | 39.38078561 | [-64.40149318, 143.16306441] | -4.67638618 |
| source score `sigma_source_probit` | 42.21493949 | [-50.42155283, 134.85143181] | -8.28669043 |
| source score `phi_source_probit` | 13.94613566 | [-34.39442976, 62.28670107] | -6.63940613 |
| source score `gamma_source_probit` | 23.06897619 | [-34.88251270, 81.02046509] | -6.38616308 |
| source score `R_source_probit` | -1.79691128 | [-7.92525133, 4.33142877] | -2.05834495 |

## Structural and Numerical Gates

- Maximum pre-reset transition residual: `9.537e-07`.
- Maximum aggregate reset/marginal/score-sum residual: `3.901e-06`.
- Maximum reset/marginal residual: `3.901e-06`.
- Maximum relative score-increment accounting residual: `8.193e-07`.
- Process noise dimension: one scalar innovation; no independent `k` shock.
- Initial observation ordering: `y0` assimilated before the first transition.
- Runtime score: recursive forward sensitivity; no autodiff or finite difference.
- Frozen data hashes: state `fe77f0e0000db93281116e7e81ddd303e9706b9e402bfaf7141a1aa1005c0ca9`, observation `ab7885b135d8098c6e516e06733ef99399ea07f4a39292670b578da4a0efbae3`.

## Decision

| Decision | Status | Interpretation |
|---|---|---|
| Candidate leaderboard row | `included_candidate_not_admitted` | Existing STR-UKF target, GenUT method extension included with raw evidence |
| Exact likelihood/score | `not established` | Existing UKF is not an oracle and GenUT has no exact nonlinear oracle here |
| Default/HMC promotion | `not evaluated` | Requires the cross-model admission contract and stronger score evidence |

## Inference Status

| Evidence class | Status |
|---|---|
| Hard veto screen | Passed if result status is candidate-ready |
| Statistically supported ranking | None; UKF comparison is deterministic diagnostic evidence |
| Descriptive-only differences | GenUT-minus-UKF value and score differences |
| Default readiness | Not established |
| Next evidence needed | Cross-model candidate admission, independent score authority, and high-dimensional memory validation |
