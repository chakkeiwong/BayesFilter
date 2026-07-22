# P5 Structural Mathematical Target Note

Date: 2026-07-16

The intended transition is the pushforward of the lagged filtering law and one
new scalar innovation:

```text
m_t = rho m_(t-1) + sigma epsilon_t,  epsilon_t ~ N(0,1)
k_t = phi k_(t-1) + gamma m_t^2
y_t = m_t + k_t + e_t,               e_t ~ N(0,R).
```

Conditional on `(m_(t-1), k_(t-1), epsilon_t)`, both next-state coordinates are
fixed and every intended propagated point satisfies
`k_t - phi*k_(t-1) - gamma*m_t^2 = 0`. The structural principal-square-root UKF
places its seven unscented points on `(m_(t-1),k_(t-1),epsilon_t)` and computes
`k_t` pointwise. It approximates the Gaussian filtering likelihood for this
declared structural pushforward; it is not an exact nonlinear filter.

The diagnostic negative control computes a different model:

```text
k_t = phi k_(t-1) + gamma m_t^2 + eta_k,
eta_k ~ N(0,0.04).
```

It therefore has a two-dimensional innovation, off-manifold residual equal to
`eta_k`, and an extra 0.04 contribution to the `k` predictive covariance. On
the Chapter 18b one-step fixture it changes the innovation variance from
`0.6121674304` to `0.6521674304` and the Gaussian log-likelihood contribution
from `-0.7029747609` to `-0.7328186210`. This is correct linear algebra for the
wrong transition law relative to the intended structural target. It is never a
fallback, posterior target, HMC route, or training route.

The candidate physical prior is independent Uniform over the five prospective
boxes in the target-design subplan. A five-probit map includes the complete
Jacobian, so physical prior plus Jacobian equals a standard-normal density in
source coordinates. This identity is a parameterization check, not evidence
that the likelihood identifies all five parameters. Identifiability screening
therefore excludes both the source prior and chart Jacobian and uses only
derivatives of the structural UKF predictive observation mean and innovation
variance.

This note defines the quantity the target-design code must compute. It does not
issue a typed posterior identity or establish posterior correctness, global
identifiability, HMC convergence, NeuTra quality, Zhao-Cui validity, calibration,
or readiness.
