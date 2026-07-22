# LGSSM q* Reparameterization Diagnostic

Date: 2026-07-20
Status: `COMPLETE_PRESERVED_ARTIFACT_CHAIN_RULE_DIAGNOSTIC`

## Proposed Map

The proposed coordinate is

```text
q*² = q_scale² A(phi)
A(phi) = [1/(1-phi1²) + 1/(1-phi2²) + (1-phi3²)] / 3.
```

At the DGP `phi=(0.72,0.55,0.35)` and `q_scale=0.35`:

```text
A = 1.4625345721
q* = 0.4232735346.
```

## Algebraic Verdict

The map is a valid local reparameterization for positive `q_scale`, but it
does not change the physical likelihood target. With `q=q*/sqrt(A(phi))`,

```text
g_phi_i = f_phi_i - f_q q A_phi_i/(2A)
g_q*    = f_q/sqrt(A).
```

The current certification uses log-scale HMC coordinates. Therefore

```text
q* g_q* = q f_q,
```

so the `q*` log-score is exactly the existing `log(q_scale)` score. The value
is exactly invariant at the same physical theta. A different value would mean
the implementation evaluated a different physical point or introduced
numerical operation-order effects. The physical `q*` score is only a positive
rescaling of the physical `q_scale` score, so its relative bias is unchanged.

The phi scores do change because holding `q*` fixed changes `q_scale` when
phi changes. This can redistribute apparent score bias between coordinates,
but it cannot remove a physical likelihood mismatch.

## Preserved-Artifact Test

The diagnostic transformed all 16 claim scores for the independently tuned
`N=5000` and `N=10000` artifacts using the exact Jacobian. No GPU rerun was
needed for this algebraic test.

| Coordinate | N=5000 mean relative bias | N=10000 mean relative bias |
| --- | ---: | ---: |
| Value | `+0.1482%` | `+0.1735%` |
| `phi1` in q* coordinates | `+0.5769%` | `+0.9655%` |
| `phi2` in q* coordinates | `+0.1449%` | `+0.8044%` |
| `phi3` in q* coordinates | `-15.4648%` | `-2.9697%` |
| `q*` log-score | `-9.9115%` | `-15.8989%` |
| `r_scale` | `-1.8144%` | `-3.6106%` |

The `q*` log-score is unchanged, as predicted. Value is unchanged. The
reparameterization changes the phi-coordinate scores but does not improve the
concerning q direction. This is a chain-rule diagnostic, not evidence that a
new kernel implementation was run.

## Decision

The proposed map makes sense as a conditioning/coordinate diagnostic and may
be useful for optimizer geometry. It is not a plausible standalone repair for
the observed q-score bias. Do not promote it as a new default or rerun the
long particle ladder solely for this map.

The next discriminating experiment remains a same-stream decomposition of the
q-score into initial covariance, transition/proposal, observation-weight and
normalization, carried-weight, and Contract E reset terms. If a downstream
optimizer needs q* coordinates, implement the full Jacobian and verify value
invariance plus score chain-rule equality before interpreting any change.

Artifact:
`docs/benchmarks/artifacts/lgssm_qstar_reparameterization_20260720/aggregate.json`.
