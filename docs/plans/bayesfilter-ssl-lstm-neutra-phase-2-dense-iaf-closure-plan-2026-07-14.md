# SSL-LSTM NeuTra Phase 2 Dense-IAF Closure Plan

Date: 2026-07-14

Status: `AUTHORIZED_TIER2_ENGINEERING_EXECUTION`

## Objective And Entry Conditions

Implement exact forward/inverse, score-pullback, and log-Jacobian-score
semantics for every component already accepted by the frozen dense-IAF schema.
Phase 0 is `PHASE_0_ACCEPTED_PHASE_1_AUTHORIZED`; Phase 1 is
`NO_ORACLE_DESIGN_VALID`. The immediate source and test files are clean relative
to the concurrent Kalman lane.

## Mathematical Contract

All batches use row vectors. A component maps row `x` to row `y`; a score row
`g_y` pulls back as `g_x = g_y J`, where the column-coordinate Jacobian is
`J = dy/dx`. Components compose in payload forward order and cotangents
propagate in reverse order.

For an autoregressive component,

```text
y_i = x_i exp(s_i(x_<i)) + t_i(x_<i)
s_i = s_max tanh(a_i / s_max)
log|det J| = sum_i s_i.
```

The direct pullback is `g_y * exp(s)`. The masked network receives output
cotangents `[g_y * x * exp(s) * ds/da, g_y]`. A manual reverse pass through
masked matrices and activation derivatives supplies the indirect term. For the
logdet score, the network output cotangent is `[ds/da, 0]`. No production path
uses `GradientTape`.

Autoregressive inversion is coordinate sequential: because `s_i,t_i` depend
only on `x_<i`, set
`x_i = (y_i - t_i(x_<i)) exp(-s_i(x_<i))`. Linear and affine inverses use exact
linear solves or elementwise division. A composition inverts children in
reverse order.

For a composed transport with component inputs `x_k`, total logdet score is
computed by the reverse recurrence

```text
q_K = 0
q_k = J_k(x_k)^T q_(k+1) + grad_xk log|det J_k(x_k)|.
```

In row-vector implementation this is the component pullback plus its local
logdet score.

## Evidence Contract And Skeptical Audit

| Field | Prospective contract |
| --- | --- |
| Question | Does the frozen schema implement the declared exact change of variables and score in CPU and trusted GPU/XLA execution? |
| Baseline | Hand-computable identity/affine/triangular fixtures and debug-only TensorFlow autodiff on tiny fixtures |
| Primary pass | All focused analytic, finite-difference, batch/permutation, roundtrip, serialization, transformed-target, and XLA checks pass |
| Vetoes | Wrong direction/order/transpose/sign, roundtrip failure, derivative mismatch, nonfinite value, malformed acceptance, serialization drift, or XLA failure |
| Explanatory only | Runtime, compiled graph size, and tail residuals below declared tolerances |
| Nonclaims | No transport quality, training, HMC, posterior, performance, or readiness conclusion |
| Result artifact | `docs/plans/bayesfilter-ssl-lstm-neutra-phase-2-dense-iaf-closure-result-2026-07-14.md` |

Audit findings: the baseline is not a sampler; debug autodiff is a test oracle
only; a loader test cannot promote HMC readiness; the trusted canary executes
the exact methods rather than checking only a JIT flag; and the run stops on
the first unresolved mathematical or XLA failure. Audit status:
`PASS_FOR_PHASE_2_ENGINEERING_EXECUTION`.

## Required Work And Checks

1. Extend the existing component classes with inverse, pullback, and local
   logdet-score operations without changing the persisted schema.
2. Add public scalar/batch closure methods to `FrozenDenseIAFTransport`.
3. Add fixtures for diagonal and non-symmetric dense affine maps, mixing,
   triangular autoregression, nested/two-stage composition, saturation/tails,
   malformed payloads, and transformed-target integration.
4. Run CPU-hidden focused tests, including debug-autodiff and directional
   finite-difference comparisons.
5. Run a small structured trusted GPU/XLA canary for forward, inverse,
   pullback, and logdet score. No HMC or training is authorized.
6. Perform one focused mathematical/source review, repair if needed, and rerun
   affected checks.

## Resource And Stop Boundary

CPU checks should finish within five minutes. The GPU/XLA canary is limited to
one tiny two-dimensional payload, center/shell/tail rows, and at most five
minutes. Stop for an unresolved inverse, transpose, composition, finiteness,
serialization, or compiled-execution failure. Preserve any failed output; do
not weaken tolerances or remove a fixture.
