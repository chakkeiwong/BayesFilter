# Audit: structural particle-filter semantics

## Date

2026-05-05

## Purpose

This note closes the BayesFilter-owned semantics gate for the particle-filter
portion of the original structural state partition plan.  It does not promote
differentiable particle filters, PMCMC, or HMC through particle likelihoods.

## Structural Semantics

For a mixed structural model,

```text
x_t = pack(s_t, d_t)
s_t = T_s(x_{t-1}, epsilon_t; theta)
d_t = T_d(x_{t-1}, s_t; theta)
```

the audited bootstrap particle reference uses the proposal

```text
x_{t-1}^{(i)} ~ filtering approximation
epsilon_t^{(i)} ~ p(epsilon_t)
x_t^{(i)} = T(x_{t-1}^{(i)}, epsilon_t^{(i)}; theta)
```

and weights particles by the observation density
`p(y_t | x_t^{(i)}, theta)`.

The particle cloud lives in innovation space plus the previous-state cloud.
Deterministic-completion coordinates are transition outputs.  They are not
given independent artificial noise.

## Approximation Policy

Artificial deterministic-coordinate noise is allowed only as a declared
approximation with:

- nonempty `approximation_label`;
- `proposal_correction="declared_approximation"`;
- metadata that prevents promotion to exact structural filtering.

The current implementation is therefore labeled `monte_carlo_value_only`.

## Implementation Result

BayesFilter now exposes:

- `ParticleFilterConfig`;
- `ParticleFilterResult`;
- `particle_filter_log_likelihood`.

The implementation is NumPy/eager, bootstrap-only, and innovation-proposal-only.
It supports pointwise identity diagnostics supplied by tests or adapters.

## Validation

The focused tests verify:

- AR(2) particle likelihood is finite and close to the exact Kalman reference;
- deterministic lag identity is preserved pointwise;
- unlabeled deterministic-coordinate noise fails closed;
- labeled deterministic-coordinate noise remains an approximation-only config.

## Remaining Blockers

- Differentiable particle filters require a separate resampling/gradient audit.
- PMCMC/pseudo-marginal claims require source-backed unbiasedness and estimator
  variance gates.
- Client DSGE particle claims wait for explicit DSGE structural adapters.
