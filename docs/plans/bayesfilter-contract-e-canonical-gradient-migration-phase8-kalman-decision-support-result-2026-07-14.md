# Phase 8 Result: Kalman-Only Gradient-Margin Decision Support

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `KALMAN_ONLY_DECISION_SUPPORT_PASSED_OWNER_MARGIN_STILL_REQUIRED`

## Outcome

The frozen `d=3,T=50` Kalman observed-data likelihood was evaluated once at
the leaderboard center under an explicit CPU-hidden, non-JIT comparator-only
exception. No Contract E dependency was imported or loaded and no candidate
output was observed.

Every identity, finiteness, transform, direct-HMC-tape, chain-rule, dependency,
and manifest check passed. The deterministic HMC-coordinate oracle gradient has
no zero component at this center. Therefore a center-scoped componentwise
relative-gradient criterion can be used without an arbitrary near-zero floor.

## Oracle

Parameter order:
`(phi1, phi2, phi3, q_scale, r_scale)`.

| Quantity | phi1 | phi2 | phi3 | q_scale | r_scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| Physical gradient | 5.6554468804 | -3.8350564589 | 0.3023616838 | -1.9171802706 | 4.3542659190 |
| HMC gradient | 2.7236632176 | -2.6749518801 | 0.2653223776 | -0.6710130947 | 1.9594196636 |
| Benchmark-box radius | 2.7394258064 | 2.4501621366 | 2.1972245773 | 1.9459101491 | 2.1972245773 |
| Weighted oracle contribution | 7.4612733062 | 6.5540658138 | 0.5829728489 | 1.3057311912 | 4.3052850421 |
| Contribution share | 36.92% | 32.43% | 2.88% | 6.46% | 21.30% |

Kalman total log likelihood: `-136.0759748579247`.

The proposal's global scale is

```text
S_oracle = 4.041865640431955.
```

## Margin Interpretations

### Reviewed global-contribution metric

For the reviewed proposal,

```text
C_grad,k = r_k * abs(e_k) / S_oracle.
```

An owner choice `delta_grad=d` implies these absolute HMC-gradient error
budgets:

```text
abs(e_k) <= d *
  (1.4754426388, 1.6496319080, 1.8395323273,
   2.0771080527, 1.8395323273).
```

Expressed as ordinary relative-error allowances, one unit of `d` corresponds
to

```text
(0.5417125837, 0.6166959190, 6.9331970573,
 3.0954806532, 0.9388148754).
```

Thus this global metric is substantially more permissive for `phi3` and
`q_scale` because their oracle contributions are small relative to the other
coordinates. This is not a numerical bug; it is the consequence of the chosen
global loss. The owner should approve it only if contribution to a global
first-order scale, rather than ordinary componentwise relative accuracy, is the
scientific target.

### Center-componentwise relative metric

Because every deterministic HMC oracle component is nonzero here, the simpler
center-scoped gate is well-defined:

```text
R_component,k = abs(e_k) / abs(g_Kalman,k)
max_k R_component,k <= delta_grad.
```

No near-zero floor is needed for this specific center. This metric gives every
coordinate the same relative-error budget and directly rejects sign reversal
or order-one disagreement. It remains center-scoped and does not prove gradient
quality elsewhere in the benchmark box or along an HMC trajectory.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept oracle decision support | all identity/math/dependency checks | Passed | none for deterministic comparator | Use table for owner choice | Contract E accuracy |
| Use global-contribution metric | owner approves common global scale | Open human decision | weak-coordinate permissiveness | Approve metric and `delta_grad` | HMC readiness |
| Use componentwise-relative metric | all center oracle components nonzero | Open human decision | acceptable relative bias | Approve metric and `delta_grad` | Off-center validity |
| Execute Contract E lower rung | complete owner amendment plus exact reviewed harness | Blocked | metric/margin, scope, audit count, lower-rung approval | Obtain owner choices | Phase 8/9 completion |

## Evidence And Checks

- preflight: `5 passed, 2 warnings in 4.59s`;
- one attempt, no retry; shell wall time `7.4s`, recorded computation wall time
  `0.9417097419973288s`;
- artifact status: `KALMAN_ONLY_DECISION_SUPPORT_PASSED`;
- direct versus chain-rule HMC gradient absolute errors range from approximately
  `1.33e-15` to `2.51e-14`, all inside the serialized componentwise engineering
  allowances;
- result SHA-256:
  `608a93f3fcf7c38a451fcb507123aa870b0d8858f0c2e9e31a662a1ab1382f1d`;
- canonical core hashes remained unchanged.

Artifact:
`docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/kalman-decision-support-attempt1/result.json`.

## Nonclaims

This result does not select `delta_grad`, establish Contract E/Kalman
equivalence, justify an HMC trajectory-error tolerance, certify full-box
validity, authorize another runtime, or establish Phase 9, leaderboard, HMC,
release, or scientific readiness.
