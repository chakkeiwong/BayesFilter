# Phase 8 Subplan: Kalman-Only Gradient-Margin Decision Support

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status: `CLOSED_KALMAN_ONLY_DECISION_SUPPORT_PASSED`

## Objective

Compute the deterministic `T=50` Kalman oracle value and gradient at the frozen
LGSSM center, then evaluate only the normalization constants in the reviewed
owner-amendment proposal. This is decision support for `delta_grad`; it is not
a Contract E candidate run and cannot select the margin.

## Entry Conditions

- Contract E Phase 8 remains `BLOCKED_HUMAN_DECISION`.
- Dataset seed `81100`, center theta, benchmark box, coordinates, and value
  margin are already frozen.
- No Contract E target output may be observed in this subplan.
- The original campaign clock ends exactly at
  `2026-07-14T09:32:19+08:00` and is not reset.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What exact Kalman center score and proposal scale convert an owner-selected `delta_grad` into per-coordinate absolute gradient-error budgets? |
| Comparator | Float64 TensorFlow Kalman observed-data log likelihood for the frozen `d=3,T=50` dataset and stationary initial law |
| Primary checks | Frozen theta/domain/data identity; finite value and gradients; physical-to-HMC chain-rule engineering agreement against a direct HMC-coordinate tape; positive finite `S_oracle`; exact algebraic budget coefficients |
| Vetoes | Any import or loaded module in the frozen canonical dependency closure; identity mismatch; nonfinite result; zero/nonfinite normalization; failed chain-rule engineering predicate; overwrite |
| Explanatory only | Raw oracle component sizes and ordinary component contribution shares |
| Not concluded | A value of `delta_grad`; Contract E accuracy; numerical candidate adequacy; HMC readiness; Phase 9 or leaderboard readiness |
| Artifact | `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/kalman-decision-support-attempt1/result.json` |

## Required Artifacts And Checks

- one comparator-only emitter with no Contract E dependency;
- focused tests for transforms, radii, chain factors, and budget algebra;
- one exclusive-write JSON result and run manifest;
- Python compilation, JSON parse, source scan, scoped diff check, and SHA-256.

This is an explicit non-JIT CPU comparator/reference exception. Eager
`GradientTape` is used to provide two independently parameterized oracle
gradients. The artifact must record `jit_compile=false`, the reference-only
scope, and that it is not GPU/XLA/default/HMC/candidate evidence.

The chain predicate is the componentwise engineering allowance

```text
abs(error_k) <= 256*u64*max(1,abs(g_direct,k),abs(g_chain,k)),
u64 = 2^-53.
```

The fixed multiplier is a conservative consistency allowance for the two eager
autodiff traversals; it is not a formal kernel forward-error bound. The artifact
must serialize each tolerance and observed error.

The frozen forbidden dependency closure is:

```text
bayesfilter.highdim.ledh_contract_e_canonical_lgssm_tf
bayesfilter.highdim.ledh_contract_e_streaming_tf
bayesfilter.highdim.ledh_contract_e_reset_tf
bayesfilter.highdim.ledh_contract_e_lgssm_preparation_tf
bayesfilter.highdim.ledh_contract_e_identity
experiments.dpf_implementation.tf_tfp.resampling.annealed_transport_tf
```

Both emitter AST imports and runtime-loaded module names must be disjoint from
this set. The exact one-attempt, no-retry command is:

```bash
timeout 120s env CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 MPLCONFIGDIR=/tmp python docs/benchmarks/emit_contract_e_phase8_kalman_margin_decision_support.py --output docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase8/kalman-decision-support-attempt1/result.json
```

The exact deadline is `2026-07-14T09:32:19+08:00`. The result path must not
exist before launch. No retry is authorized.

## Skeptical Audit

Decision: `PASS_COMPARATOR_ONLY`.

- The command answers the decision-support question directly and cannot observe
  a Contract E result.
- It does not convert a confidence level or FD threshold into a scientific
  margin.
- It reports the exact budget coefficient as a function of the still-open owner
  value rather than nominating a convenient number.
- Kalman is the exact LGSSM comparator; no proxy criterion is promoted.
- The output is fresh, versioned, and bounded well inside the original clock.

## Handoff And Stop Conditions

On pass, add the exact oracle table to the owner-decision handoff and retain
`BLOCKED_HUMAN_DECISION`. Stop for any veto above or campaign-clock exhaustion.
Do not draft or execute a Contract E harness until the owner approves the
reviewed amendment choices.
