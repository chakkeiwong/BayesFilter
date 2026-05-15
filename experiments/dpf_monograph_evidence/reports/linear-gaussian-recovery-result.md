# IE3 linear-Gaussian recovery

## Scope

This clean-room IE3 report compares a one-step analytic Kalman reference against replicated bootstrap PF summaries and a deterministic EDH linear-Gaussian special-case recovery.

## Fixture

- predictive mean: `0.315000000000`
- predictive variance: `1.291000000000`
- posterior mean: `-0.037630822312`
- posterior variance: `0.332586543991`
- one-step log likelihood: `-1.483609255109`

## Bootstrap PF descriptive diagnostics

| Particles | Mean abs. error | Mean MCSE | Variance abs. error | Variance MCSE | Log-likelihood abs. error | Log-likelihood MCSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 0.044946453465 | 0.010103964910 | 0.066146906800 | 0.018151820950 | 0.111824277715 | 0.039067141266 |
| 256 | 0.021578810033 | 0.008668251926 | 0.020995960579 | 0.005908684794 | 0.035728463261 | 0.006658639688 |

PF interpretation: descriptive engineering evidence only. The five-seed summaries show smaller errors at 256 particles than at 64 particles, but this is not treated as a statistically supported convergence claim.

## EDH deterministic special-case recovery

- posterior mean abs. error: `3.469e-17`
- posterior variance abs. error: `0.000e+00`
- interpretation: exactness is restricted to this linear-Gaussian special case and is not promoted to nonlinear settings.

## CPU-only manifest proof

- `CUDA_VISIBLE_DEVICES` before scientific imports: `-1`
- pre-import assertion: `True`
- cpu_only: `True`

## Exit label

`ie3_linear_gaussian_recovery_passed`
