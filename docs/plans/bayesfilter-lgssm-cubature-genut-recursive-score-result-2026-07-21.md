# Cubature/GenUT Recursive Score Result

Date: 2026-07-21

## Outcome

The Cubature/GenUT runtime score is now computed by an explicit compact forward
sensitivity with no TensorFlow autodiff. Central finite differences are no
longer the runtime score. They are restricted to fixed representative tuning
and audit points and differentiate the same value-only finite program.

The frozen route identifier is:

```text
compact_forward_sensitivity_no_autodiff_cubature_genut_v1
```

The recursion carries all five parameter tangents through the initial
stationary draw, transition, observation likelihood, log-weight normalization,
every finite Sinkhorn scaling iteration, barycentric transport, weighted and
uniform moments, residual injection, Cholesky factorizations, and restoration
triangular solve. Equal-weight reset makes the next weight tangent zero, while
the reset particle tangent carries all earlier dependence forward.

## Verification

| Check | Result |
|---|---|
| Focused tests | `9 passed` |
| Tiny `T=2, N=12` recursive vs FD | max absolute error about `1.82e-4` |
| GPU TF32 `T=2, N=1008` recursive vs FD | max absolute `0.00764`; max relative `2.13%` |
| GPU TF32 `T=50, N=1008`, representative point 0 | max absolute `0.02524`; max relative `1.63%` |
| GPU TF32 `T=50` runtime-only | finite, bitwise replayable, FD audit absent |
| Runtime-only wall time | about `8.78 s`, including trace and two replay evaluations |
| Runtime-only peak TensorFlow allocator | `400,739,328` bytes, about `382 MiB` |
| Actual one-row tuner smoke | `hard_valid=true`; max normalized FD residual `0.138` |

All GPU checks used float32 tensors, TF32 enabled, verified memory growth, no
XLA, and the managed-session trusted-GPU basis. XLA remains a disclosed debug
exception inherited from the prior reproducible Cholesky-gradient layout
failure; this result does not close that issue.

## Decision Table

| Decision | Status |
|---|---|
| Runtime score architecture | recursive compact forward sensitivity retained |
| Finite difference use | representative tuning/audit only |
| Autodiff use in candidate score | none |
| Engineering correctness | tiny and one full-horizon representative parity checks pass |
| Numerical validity | finite, replayable, reset and Sinkhorn diagnostics pass in smokes |
| Scientific bias conclusion | not evaluated by this implementation repair |
| Next justified action | run the full fixed representative-point v3 tuning and untouched claim under a separate bounded campaign |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | no veto in focused and one-row smokes |
| Statistically supported ranking | none |
| Descriptive-only differences | runtime and memory figures are single-process smokes |
| Default readiness | not established |
| Next evidence needed | all six representative points across calibration/validation seeds, then untouched 16-seed claim |

## Post-Run Red Team

The strongest remaining alternative explanation is that recursive/FD parity
could fail at another representative point, seed, or control tuple because
float32 subtraction and Sinkhorn conditioning vary across the parameter box.
The full representative audit is designed to expose that failure. Passing it
would establish consistency with the finite Cubature/GenUT value program at the
audited points, not equality to the exact model likelihood or validity for a
nonlinear application.

Forward sensitivity scales with parameter count. It is appropriate for this
five-parameter diagnostic and remains compact in time; a model with a very
large parameter vector may require a manual reverse scan instead. State
dimension alone does not invalidate this route.
