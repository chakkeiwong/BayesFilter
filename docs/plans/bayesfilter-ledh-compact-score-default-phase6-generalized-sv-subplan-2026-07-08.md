# Phase 6 Subplan: Generalized-SV Compact Score Port

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Build a compact forward-sensitivity score route for the generalized-SV source-route row, or write a blocker result if the score cannot be derived without changing the target scalar or parameter coordinate system.

The target is the admitted finite-`N` LEDH value scalar:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

for row:

```text
zhao_cui_generalized_sv_synthetic_from_estimated_values
```

The target observation policy is:

```text
source_route_prior_mean_generalized_sv
```

The active score coordinates are:

```text
gamma_unconstrained, log_tau, mu
```

under:

```text
source_route_active_transformed_prior_mean
```

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared contract blocks historical `manual_total_vjp*` routes from full score admission.
- Phases 2-5 established compact tiny routes for LGSSM, actual-SV, fixed-SIR, and predator-prey.
- Generalized-SV has an admitted value artifact:
  `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`.
- The generalized-SV value runner lives at:
  `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py`.
- There is no current generalized-SV score module in the compact-score runbook.

## Required Artifacts

- New or updated generalized-SV score implementation:
  `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_score.py`
- New generalized-SV score tests:
  `tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py`
- Updated score contract if a compact generalized-SV provenance constant is added:
  `bayesfilter/highdim/ledh_score_contract.py`
- Tiny compact score diagnostic artifact:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-tiny-compact-score-2026-07-08.json`
- Phase 6 result or blocker result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-result-2026-07-08.md`
- Phase 7 KSC-SV subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Precheck commands:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py
```

Focused CPU-only checks after implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_generalized_sv_score_phase6_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Tiny diagnostic command must be drafted after the compact generalized-SV entry point exists. It must use fixed randomness, small `N,T`, all three active coordinates, manual streaming finite transport value+JVP, and no autodiff.

Review:

- Claude read-only review of implementation, tests, Phase 6 result, and Phase 7 subplan when available.
- Use the probe ladder on timeout. If Claude is unavailable, write a Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can generalized-SV compute the same finite-`N` source-route prior-mean raw-y LEDH `log_likelihood` score in active transformed coordinates using compact forward sensitivity? |
| Baseline/comparator | Admitted generalized-SV value artifact, generalized-SV value runner, previous compact model ports, and tiny same-scalar finite differences. |
| Primary criterion | Generalized-SV default score route carries particles/log weights/tangents/log-likelihood tangents forward, emits compact provenance, matches the value scalar at the tiny gate, passes all-coordinate FD, and does not substitute actual-SV or KSC target semantics. |
| Veto diagnostics | Wrong target scalar, actual-SV/KSC substitution, KSC surrogate likelihood promoted as generalized-SV, wrong parameter coordinate/order, stopped partial derivative, tape/autodiff, reverse-record score default, nonfinite score, or tiny FD failure. |
| Explanatory diagnostics | Runtime, memory, per-coordinate FD error, chunk-size invariance, and route parity against value-only diagnostics. |
| Not concluded | Full `N=10000,T=1008` generalized-SV score admission, HMC readiness, posterior correctness, SP500 benchmark validity, author-default truth validity, runtime ranking, or scientific superiority. |
| Artifact | Tiny compact score JSON, Phase 6 result or blocker, tests. |

## Required Implementation Steps

1. Inspect the generalized-SV value route and admitted artifact:
   - `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py`
   - `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`
   - `tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py`
2. Freeze the exact target identity:
   - raw-y source-route prior-mean generalized-SV likelihood;
   - `target_observation_policy = source_route_prior_mean_generalized_sv`;
   - parameter order `gamma_unconstrained, log_tau, mu`;
   - theta values `[1.0824113944610982, -2.076793740349318, 0.0]`.
3. Derive compact forward sensitivities for:
   - `gamma = Phi(gamma_unconstrained)`;
   - `tau = exp(log_tau)`;
   - `mu`;
   - stationary variance;
   - transition mean `mu + gamma * (x - mu)`;
   - proposal/flow observation surface `tau * x`;
   - target observation density `y_t | x_t ~ Normal(0, exp(tau * x_t))`;
   - LEDH flow, correction weights, normalization, and streaming transport value+JVP.
4. Add compact generalized-SV provenance only after the route is implemented.
5. Add compact artifact normalization with compact provenance.
6. Add static tests against compact default symbols for:
   `records.append`, `reversed(records)`, reverse transport pullback, `GradientTape`, and `ForwardAccumulator`.
7. Add tiny same-scalar value match and all-coordinate FD tests.
8. Write Phase 6 result or blocker, draft Phase 7 KSC-SV subplan, and review the handoff.

## Forbidden Claims And Actions

- Do not substitute transformed actual-SV, KSC-SV, SP500, or native dense generalized-SV fixture targets.
- Do not treat the log-square Gaussianized LEDH observation surface as the target likelihood.
- Do not change row ID, target scalar, target observation policy, theta coordinate system, parameter order, observations, or value artifact.
- Do not run full `N=10000,T=1008` before tiny compact correctness passes and a reviewed amendment authorizes the full run.
- Do not use tape/autodiff, stopped partial derivatives, hidden autodiff, or reverse-record score as default.

## Exact Next-Phase Handoff Conditions

Phase 7 KSC-SV may start only if:

- generalized-SV default score route is compact or generalized-SV is explicitly blocked with a result artifact;
- tiny same-scalar FD either passes or the blocker result explains the smallest unresolved issue;
- Phase 7 subplan exists and is reviewed;
- Codex records review findings and confirms no unresolved boundary issue.

## Stop Conditions

Stop and ask for direction or write a blocker result if:

- generalized-SV compact JVP cannot be derived without changing the target scalar;
- target likelihood ambiguity cannot be resolved from the value artifact and value code;
- tiny same-scalar FD fails and cannot be localized in this phase;
- implementation would require unrelated dirty-worktree changes;
- memory/runtime behavior suggests even tiny compact generalized-SV is not viable;
- Claude and Codex review do not converge after five rounds on the same material blocker.

## Skeptical Audit Before Execution

Risks checked before launch:

- Wrong baseline: the baseline is the admitted generalized-SV LEDH value artifact, not actual-SV or KSC.
- Proxy metrics: tiny FD may pass but still cannot admit a full row without full-row memory evidence.
- Hidden assumption: the active coordinates are source-route transformed parameters, not generic SV coordinates.
- Environment mismatch: CPU-only tests must hide GPU before TensorFlow import; later GPU runs require trusted execution.
- Useless artifact risk: any result must state target observation policy, parameter order, and non-claims explicitly.

Audit status: ready for read-only review after Phase 5 result review.
