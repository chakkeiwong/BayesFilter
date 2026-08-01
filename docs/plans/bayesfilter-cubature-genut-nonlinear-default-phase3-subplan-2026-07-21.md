# Phase 3 Subplan: Exact Transformed-SV Candidate Pilot

Date: 2026-07-21

Status: `HISTORICAL_NONDGP_FINITE_PROGRAM_MECHANICS_ONLY`

> **Correction, 2026-07-22:** The observation tensor was an arbitrary
> transformed-input fixture, not data from the SV DGP.  This subplan can support
> adapter wiring and same-finite-scalar derivative mechanics only.  It is
> scientifically irrelevant to SV performance.

## Question

Does the generic candidate core preserve the declared exact transformed-SV
target and compute a same-scalar recursive score on fixed randomness?

## Target Contract

- State: scalar latent volatility `h_t`.
- Parameter chart: `theta=(theta_gamma, theta_log_beta)`;
  `gamma=Phi(theta_gamma)`, `beta=exp(theta_log_beta)`.
- Initial law: stationary `N(0, sigma^2/(1-gamma^2))`, with `sigma=1`.
- Transition: `h_t=gamma*h_{t-1}+sigma*eta_t`.
- Observation: `z_t=log(y_t^2)` and `u_t=z_t-2 log(beta)-h_t`;
  `log p(z_t|h_t)=0.5*u_t-0.5*exp(u_t)-0.5*log(2*pi)`.
- Scope: diagnostic `T=4`, `N=12`, float32, fixed stateless innovations,
  Cubature residual, no XLA.

## Pass/Stop

Pass if adapter callbacks match the model equations, candidate value is finite
and replayable, reset/marginal diagnostics pass, and the recursive score agrees
with central FD of the same finite value for both coordinates.

Stop if target equations, parameter chart, or tangent callbacks disagree. This
phase does not claim target-horizon SV validity, GPU/XLA feasibility, default
readiness, or leaderboard admission.
