# Cubature/GenUT Nonlinear Default Program: Phase 3 Result

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_FINITE_PROGRAM_MECHANICS_ONLY`

> **Correction, 2026-07-22:** The fixed observations were not produced by the
> SV DGP.  This result remains valid only for adapter equation wiring,
> finiteness, reset mechanics, replay, and same-finite-scalar FD parity.  It is
> not SV target, accuracy, score, variance, or model evidence.

## Outcome

Added a target-bound exact transformed-SV adapter in
`bayesfilter/highdim/cubature_genut_adapters.py` and validated it through the
generic candidate core. The adapter implements the declared chart and target:

- `gamma=Phi(theta_gamma)`, `beta=exp(theta_log_beta)`;
- stationary initial volatility;
- `h_t=gamma*h_{t-1}+sigma*eta_t`; and
- exact `log(y^2)` log-chi-square observation density.

The adapter is candidate-only and does not alter existing SV, Contract E, or
leaderboard routes.

## Checks

| Check | Result |
|---|---|
| Exact-SV pilot plus Phase 1/2/LGSSM tests | `21 passed` in `10.09 s` |
| Same-scalar FD parity | Passed both SV coordinates |
| Reset/marginal diagnostics | Passed on the pilot fixture |
| Python compilation | Pass |
| Runtime autodiff/FD/NumPy scan | Pass: none in candidate adapter/core |
| GPU/XLA | Not run; CPU hidden diagnostic only |

## Decision Table

| Decision | Status |
|---|---|
| Exact transformed-SV adapter mechanics | Passed pilot gate |
| Full SV target-horizon evidence | Not established |
| Predator-prey adapter | Not yet implemented in this phase |
| KSC-SV/generalized-SV/SIR | Not run; existing target/feature blockers remain |
| Default/leaderboard readiness | Not established and not changed |
| Next justified action | Write a bounded variance/precision subplan and then a predator-prey adapter pilot |

## Nonclaims

This pilot does not establish exact nonlinear filtering, target-horizon SV
validity, GPU/XLA or TF32 readiness, high-dimensional scaling, score precision,
method superiority, leaderboard admission, HMC readiness, or a NAWM result.
