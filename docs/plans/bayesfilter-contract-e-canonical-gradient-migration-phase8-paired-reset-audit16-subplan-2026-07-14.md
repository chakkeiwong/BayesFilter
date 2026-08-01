# Phase 8 Subplan: Paired 16-Seed Reset Audit At N=128

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Continuation ID: `contract-e-canonical-gradient-migration-continuation-20260714-115526`

Status: `LOCAL_AUDIT_PASSED_CLAUDE_UNAVAILABLE_EXECUTION_ACTIVE`

## Objective

Quantify the average Contract-E-versus-no-reset effect and its uncertainty at
the fixed diagnostic shape `T=2,N=128`, using the owner-selected audit count
`16`. Determine whether Contract E descriptively/statistically improves or
worsens per-seed absolute Kalman error for value and each HMC-gradient component
under the predeclared Student/Bonferroni model.

## Entry Conditions

- The `T=2` lower rung selected ridge `7.301568984985351e-09`, steps `20`,
  chunks `16/16`, epsilon `0.5`, scaling `0.9`.
- The one-seed `N=32,64,128` mechanism classifier was inconclusive; no setting
  is selected from that output. `N=128` is used because it is the largest
  predeclared diagnostic count, not because it had favorable observed error.
- `delta_grad=0.05` and audit count `16` are owner decisions.
- The production route factory remains empty; this is diagnostic only.

## Frozen Inputs

```text
T = 2
N = 128
dataset seed = 81100
fresh diagnostic estimator seeds = 81220..81235
arms = all_active_contract_e, no_reset_weighted
```

The fresh seed block is a pre-result convenience choice for this diagnostic and
is disjoint from lower-rung seed `80920`, calibration seeds `81020..81024`, and
the reserved primary audit pool `81120..81183`. It is not promoted to a
repository default.

## Evidence Contract

For seed `s`, define six signed normalized errors for arm `a`:

```text
z[a,s,value] = (L[a,s] - L_Kalman) / abs(L_Kalman)
z[a,s,k] = (g[a,s,k] - g_Kalman,k) / abs(g_Kalman,k)
```

and six paired absolute-loss differences:

```text
d[s,j] = abs(z[ContractE,s,j]) - abs(z[NoReset,s,j]).
```

Use two-sided Bonferroni Student intervals with familywise level `0.95` for the
six means `E[d_j]`. This model treats the 16 domain-separated Philox streams as
independent/exchangeable and assumes finite variances plus Student marginal
coverage. It has no power guarantee.

Classification per quantity:

- interval upper endpoint `<0`: Contract E has statistically supported lower
  mean absolute error under the model;
- interval lower endpoint `>0`: Contract E has statistically supported higher
  mean absolute error under the model;
- otherwise: inconclusive.

An overall mechanism label is emitted only when all six quantities have the
same non-inconclusive direction. Any mixed direction or any inconclusive member
produces `mixed_or_inconclusive`.

Separately report Contract E simultaneous mean-error intervals against the
center-scoped boundaries `value +/-0.001` and gradient components `+/-0.05`.
This remains a small-shape diagnostic and cannot establish primary-shape
equivalence.

## Required Checks And Artifacts

- exact 16-seed prepared-input identity and seed order;
- two batched CPU-hidden float64 XLA calls, one per arm, with identical data,
  noise keys, tuple, and source closure;
- per-seed value and physical/HMC gradients retained;
- all executed values, gradients, charts, branches, and required telemetry
  finite; inactive no-reset mass sentinel explicitly excluded only where not
  executed;
- exact Student/Bonferroni critical value, sample means, standard deviations,
  standard errors, and intervals serialized;
- exclusive JSON output, command, source hashes, wall time, and nonclaims;
- focused classifier/statistics tests, compilation, JSON parse, and diff checks;
- bounded read-only review or documented reviewer limitation; and
- a result note before any larger shape or factory-admission work.

## Budget And Stops

At most two executed batched nodes, each capped at 600 seconds, within the
existing two-hour continuation. Stop on source/prepared-input drift, invalid
chart, nonfinite executed output, timeout, missing per-seed quantities, zero
Kalman scale, or any need to tune settings/seeds/count from output.

## Forbidden Claims And Actions

- Do not call this primary-shape, HMC-ready, canonical-admitted, leaderboard,
  default, or release evidence.
- Do not tune `N`, tuple, `delta_grad`, seeds, or interval method after output.
- Do not claim adequate power, normality, causal mechanism, superiority beyond
  the exact interval classification, or distribution-free coverage.
- Do not consume the reserved primary audit pool.

## Handoff

If the result is mixed/inconclusive, write a blocker/repair result and stop for
a larger reviewed design rather than selecting a favorable component. If all
six paired loss directions agree, the result may motivate a separate repair or
larger-shape plan but still cannot promote or admit the route. Phase 9 remains
blocked until the primary scientific and factory gates pass.

## Review Record

Claude produced no output across three bounded one-path review windows and was
terminated. No verdict is inferred. Local audit passed: the analysis unit is the
paired estimator seed, the six-loss family is fixed before output, seed sets are
disjoint, all oracle scales are preflight nonzero requirements, and every mixed
or boundary-touching outcome remains inconclusive. Reviewer unavailability is
recorded under repository proportionality policy.
