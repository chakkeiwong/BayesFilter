# Phase 4 Subplan: Predator-Prey Score

metadata_date: 2026-07-07
status: `DRAFT_AFTER_PHASE3_BLOCKER`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4

## Phase Objective

Admit, or explicitly block, the LEDH score for:

```text
zhao_cui_predator_prey_T20
```

The score target is the no-tape total derivative of the same realized
finite-`N` LEDH estimator admitted by the predator-prey value artifact:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

The parameter vector and order are:

```text
r, K, a, s, u, v
```

in coordinate system:

```text
physical
```

## Entry Conditions Inherited From Previous Phase

- Phase 1 score schema exists and passed review.
- Phase 2 LGSSM score is blocked, not admitted.
- Phase 3 fixed-SIR score is blocked, not admitted.
- Directional FD is diagnostic only and cannot by itself admit a score.
- Predator-prey value is admitted through:
  `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json`.
- The existing predator-prey value path has no admitted same-target no-tape
  total-score artifact.

## Required Artifacts

Source value artifact:

- `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json`

Current implementation/test artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py`
- `bayesfilter/highdim/models.py`
- `bayesfilter/highdim/ledh_forward_contract.py`
- `bayesfilter/highdim/ledh_score_contract.py`
- `tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py`
- `tests/highdim/test_ledh_score_contract_phase1.py`

Expected Phase 4 artifacts:

- Phase 4 score result or blocker:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-result-2026-07-07.md`
- If a score route is implemented, score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.json`
- If a score route is implemented, score summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.md`
- Phase 5 actual-SV subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-subplan-2026-07-07.md`
- Phase 4 review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase4-predator-prey-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

CPU-hidden preflight:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py \
  bayesfilter/highdim/models.py \
  bayesfilter/highdim/ledh_score_contract.py
```

Forward value replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

If a score adapter is added, add focused tests that:

- reject missing all-parameter correctness;
- reject directional-only correctness as full admission;
- reject wrong row id, wrong target observation policy, wrong theta coordinate,
  and wrong parameter order;
- reject `GradientTape`, `ForwardAccumulator`, stopped partials, hidden
  autodiff, and score routes that differentiate anything other than
  `observed_data_log_likelihood_estimator`;
- validate the score artifact only after all six parameters have same-scalar
  correctness evidence.

Trusted GPU `N=10000,T=20` execution may run only after the adapter passes tiny
and artifact-schema tests. The trusted run must write a replayable score
artifact and memory diagnostics.

Review:

- bounded read-only review of Phase 4 result and Phase 5 actual-SV subplan
  before Phase 5 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the predator-prey main row produce an admitted no-tape total derivative of the same finite-`N` LEDH `log_likelihood` scalar as the value artifact? |
| Baseline/comparator | Admitted predator-prey value artifact, predator-prey model manual local-density parameter scores, existing value runner, exact/manual local-density checks where available, and all-parameter same-scalar finite differences with fixed randomness otherwise. |
| Primary criterion | Score artifact validates with `require_admitted=True`; row id is `zhao_cui_predator_prey_T20`; parameter order is `[r,K,a,s,u,v]`; theta coordinate is `physical`; target observation policy is `additive_gaussian_predator_prey`; score route is `manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`; score is finite; no tape/stopped partial route is used; full-row T=20,N=10000 identity matches value artifact; memory gate passes; and all-parameter correctness is established. |
| Veto diagnostics | Missing score adapter; value-only artifact promoted as score; directional FD used as sole admission evidence; wrong target scalar; wrong observation policy; wrong theta coordinate; wrong parameter order; tape/ForwardAccumulator/stopped partial; hidden autodiff; nonfinite score; memory/device failure. |
| Explanatory diagnostics | Tiny route smoke, local-density parameter score checks, directional FD, runtime, memory, and device placement. |
| Not concluded | Native Zhao-Cui TT/SIRT predator-prey source-faithfulness, exact nonlinear likelihood correctness beyond the declared additive-Gaussian row, HMC readiness, posterior correctness, scientific superiority, runtime ranking, or all-algorithm comparison. |
| Artifact | Predator-prey score artifact/result or blocker, tests, review bundle, Phase 5 subplan. |

## Step-By-Step Plan

1. Re-read the predator-prey value artifact and contract.
2. Inventory existing predator-prey score-capable code:
   - value runner target scalar and transport fields;
   - `PredatorPreySSM.transition_mean_parameter_jacobian`;
   - `transition_log_density_parameter_score`;
   - `observation_log_density_parameter_score`;
   - any old leaderboard manual score rows.
3. Decide whether a bounded same-target no-tape total-score adapter can be
   implemented within this phase without changing the value target.
4. If no adapter exists or implementation would require a new derivation phase,
   write a blocker result instead of admitting a score.
5. If implementation is feasible, add a predator-prey score adapter that:
   - differentiates the same realized finite-`N` LEDH scalar;
   - uses the same transport algorithm as the value row;
   - propagates all six physical parameters through transition-density and
     transport total-derivative terms;
   - uses no tape, ForwardAccumulator, hidden autodiff, or stopped partials.
6. Add tests for artifact schema, target identity, parameter order, no-tape
   provenance, and rejection of directional-only admission.
7. Run tiny CPU-hidden checks.
8. Only if tiny checks pass, run the trusted GPU `N=10000,T=20` score/memory
   command with bounded logs.
9. Validate or block the score artifact.
10. Write Phase 4 result, draft Phase 5 actual-SV subplan, and run bounded
    read-only review.

## Forbidden Claims/Actions

- Do not use the predator-prey value artifact as score evidence.
- Do not claim a score before the no-tape total derivative route exists.
- Do not use directional FD alone as full score admission evidence.
- Do not use tape, ForwardAccumulator, hidden autodiff, or stopped partials.
- Do not change the row target, horizon, particle count, seed list, theta
  coordinate, parameter order, or observation policy.
- Do not claim native Zhao-Cui TT/SIRT source-faithfulness.
- Do not claim exact nonlinear likelihood correctness beyond the declared
  additive-Gaussian row, HMC readiness, posterior correctness, scientific
  superiority, runtime ranking, or all-algorithm comparison.

## Exact Next-Phase Handoff Conditions

Phase 5 actual-SV may start only if:

- Phase 4 writes an admitted predator-prey score result or an explicit blocker
  result;
- Phase 5 actual-SV subplan exists;
- review agrees the predator-prey decision is boundary-safe and does not
  promote value-only, directional-only, or local-density-only evidence as a
  full LEDH score.

## Stop Conditions

Stop and write a blocker result if:

- predator-prey score cannot be tied to the admitted value artifact;
- no same-target no-tape total-score route exists;
- all-parameter correctness cannot be established;
- no-tape provenance becomes ambiguous;
- a target/parameter/order/observation-policy mismatch appears;
- memory/device gates fail;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Start only from the admitted predator-prey value artifact and row contract. |
| Proxy metric promotion | Runtime, memory, finite output, local-density score, or directional FD cannot admit the score. |
| Missing stop condition | Stop on missing adapter, target mismatch, tape/autodiff, partial derivative, and missing all-parameter correctness. |
| Hidden assumption | Parameter coordinate is physical `[r,K,a,s,u,v]`, not transformed or log-scale. |
| Stale context | Inventory current value runner and model score helpers before implementing. |
| Environment mismatch | GPU/CUDA score-memory runs require trusted execution; CPU-only checks must hide GPU before TensorFlow import. |
| Useless artifact | Any score artifact must replay through `validate_ledh_score_artifact(..., require_admitted=True)` against the value artifact. |
