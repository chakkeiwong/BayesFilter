# Audit: MacroFinance adapter contract for Phase 8

## Scope

This note records the Phase 8.1 contract audit before BayesFilter adapter code
is added.

MacroFinance commit:

- `e23c31e Document one-country HMC remaining test gates`

## Minimal Provider Target

The first integration target remains the one-country restricted AFNS provider:

- `OneCountryAFNSDerivativeProvider`
- `OneCountryTFDerivativeProvider`
- `OneCountryAnalyticThetaNoisePosteriorAdapter`

Parameter order:

```text
theta = [theta_0, theta_1, theta_2, log_measurement_error_std]
```

State-space object fields:

```text
initial_mean
initial_covariance
transition_offset
transition_matrix
transition_covariance
observation_offset
observation_matrix
observation_covariance
```

These fields match the BayesFilter exact LGSSM value contract.  MacroFinance
does not attach a BayesFilter `StatePartition`; Phase 8 must therefore accept
an optional partition supplied by the caller and must not infer mixed structural
timing from zero or low-rank process covariance.

## Jitter and Initialization

MacroFinance `RunConfig` defaults:

```text
dt = 1 / 12
jitter = 1e-9
kalman_backend = "tf_cholesky"
```

BayesFilter value tests must pass the same jitter when comparing likelihoods.

The TensorFlow one-country provider supports:

```text
initial_mean_policy in {"zero", "theta", "fixed"}
```

The NumPy one-country provider audited for the first value adapter currently
uses a zero initial mean and stationary initial covariance.  Any derivative
bridge metadata should record the initial-mean policy when present.

## Derivative Tensor Convention

MacroFinance derivative arrays use parameter-major shapes:

```text
d_initial_mean: (p, n)
d_initial_covariance: (p, n, n)
d_transition_offset: (p, n)
d_transition_matrix: (p, n, n)
d_transition_covariance: (p, n, n)
d_observation_offset: (p, m)
d_observation_matrix: (p, m, n)
d_observation_covariance: (p, m, m)

d2_*: (p, p, ...)
```

The one-country active blocks are:

- first-order `d_transition_offset` for `theta_0`, `theta_1`, `theta_2`;
- first- and second-order `observation_covariance` derivatives for
  `log_measurement_error_std`;
- optional first-order `d_initial_mean` in the TensorFlow provider when
  `initial_mean_policy == "theta"`.

## Workload Split

MacroFinance now separates first- and second-order analytic workloads:

- `tf_differentiated_kalman_loglik_grad` returns value plus score.
- `tf_differentiated_kalman_loglik_grad_hessian_graph` returns value, score,
  and Hessian.
- `OneCountryAnalyticThetaNoisePosteriorAdapter.log_prob_and_grad` uses
  first-order derivatives only.
- `OneCountryAnalyticThetaNoisePosteriorAdapter.log_prob_grad_hessian` uses
  second-order derivatives.
- `negative_log_prob_and_gradient` does not call Hessian work.
- `negative_log_prob_hessian` is the explicit Hessian path for local covariance
  construction.

BayesFilter Phase 8 must preserve this split in any derivative-facing API.

## Reuse Decision

Allowed in Phase 8:

- Convert MacroFinance-shaped LGSSM objects to BayesFilter exact value objects.
- Delegate value/score/Hessian computations to MacroFinance providers or
  posterior adapters in optional integration tests.
- Store result metadata and provenance in BayesFilter result objects.

Not allowed in Phase 8:

- Copy AFNS, pricing, Riccati, cross-currency, or production provider
  construction.
- Copy MacroFinance Kalman, differentiated Kalman, QR/square-root, SVD, masked,
  TensorFlow, or HMC sampler implementations.
- Promote HMC convergence claims.

## Gate Decision

The Phase 8.1 audit passes.

Step 8.2 remains justified: implement a value-only BayesFilter adapter around a
MacroFinance-shaped LGSSM object and prove value parity against the one-country
MacroFinance reference.
