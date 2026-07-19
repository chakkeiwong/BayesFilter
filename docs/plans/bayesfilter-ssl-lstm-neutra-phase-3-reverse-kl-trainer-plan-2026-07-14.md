# SSL-LSTM NeuTra Phase 3 Reverse-KL Trainer Plan

Date: 2026-07-14

Status: `AUTHORIZED_TIER2_ENGINEERING_EXECUTION`

## Objective And Entry Conditions

Implement a BayesFilter-owned TensorFlow dense-IAF reverse-KL trainer that can
run finite updates under GPU/XLA while treating the locked target's graph-native
score as the derivative authority. Phase 2 status is
`PHASE_2_DENSE_IAF_MATHEMATICAL_CLOSURE_PASSED`.

This phase is trainer correctness and a tiny throughput canary. It is not
material SSL-LSTM transport training and cannot select or promote a transport.

## Gradient Contract

For `theta=T_phi(z)` and target score `s(theta)=grad_theta log p(theta)`, the
reverse-KL objective, up to a constant, is

```text
L(phi) = E_z[-log p(T_phi(z)) - log|det J_phi(z)|].
```

Its parameter gradient is evaluated through the surrogate

```text
S(phi; stop(s)) = E_z[-stop(s(T_phi(z)))^T T_phi(z)
                      - log|det J_phi(z)|].
```

The implementation uses two transport evaluations. Target value/score is
computed outside `GradientTape` and stopped. Inside the tape, only the
transport and logdet are recomputed for `S`. Therefore the tape cannot traverse
the target/filter, while `grad_phi S` is the exact first-order reverse-KL
gradient at the current `phi`.

The reported loss is the actual `L`, never the surrogate value. Batch reduction
is a mean. Stateless base seeds follow the Phase 1 `2100` training and `2200`
validation namespaces.

## Evidence Contract And Skeptical Audit

| Field | Prospective contract |
| --- | --- |
| Question | Does the trainer implement the declared reverse-KL gradient, deterministic state, and compiled update path? |
| Baselines | Analytic diagonal/correlated Gaussian targets, a normalized curved-ridge control, and tiny debug autodiff through those separate controls |
| Primary pass | Objective/sign, gradient, reduction, permutation, finite update, clipping, resume, snapshot, and trusted GPU/XLA actual-target canary all pass |
| Vetoes | Wrong sign/reduction, target tape traversal, missing/nonfinite gradient, batch drift, resume mismatch, invalid frozen snapshot, CPU fallback, or XLA failure |
| Explanatory only | Loss trajectory, gradient norms, throughput, scale saturation, and validation loss |
| Nonclaims | No trained-transport quality, sampler, posterior, predictive, performance-ranking, or readiness claim |
| Result | `docs/plans/bayesfilter-ssl-lstm-neutra-phase-3-reverse-kl-trainer-result-2026-07-14.md` |

Audit status: `PASS_FOR_PHASE_3_ENGINEERING_EXECUTION`. The Gaussian optimum is
a code-control baseline, not a proxy promotion criterion. Loss reduction cannot
promote a candidate. A tiny SSL-LSTM update can prove compiled mechanics only.

## Required Implementation And Checks

1. Trainable diagonal-affine control and one-stage dense autoregressive IAF,
   with fixed mixing support reserved for later enhanced configurations.
2. Stateless initialization/base noise, bounded log scale, manual Adam state,
   global-norm clipping, finite assertions, and exact step counter.
3. Stable state payload and exact restore/replay; separate frozen transport
   payload generation with target/topology/tensor identity.
4. Analytic optimum, wrong-sign, debug-gradient, duplicated-row, batch-size,
   permutation, update, saturation, serialization, and resume tests.
5. Curved-ridge control sufficient to expose a nonlinear gradient path; its
   finite loss is not evidence for the SSL-LSTM posterior.
6. Trusted GPU/XLA canary using the locked four-coordinate target, at most two
   tiny updates, a fixed batch, role code `2101`, and a structured receipt.
7. Focused numerical/source review and repair/recheck if needed.

## Resource And Stop Boundary

CPU controls should finish within five minutes. The trusted canary is limited
to two updates, a small static batch, and five minutes. No HMC, external sample
generation, forecast, candidate ladder, seed search, or material training is
authorized. Stop on any gradient, resume, target-boundary, GPU placement, XLA,
or finiteness veto.
