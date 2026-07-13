# Phase 4 Repair Subplan: Predator-Prey Manual Total-VJP Score Adapter

metadata_date: 2026-07-07
status: `DRAFT_REPAIR_LOOP`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 4-repair

## Phase Objective

Implement and test a predator-prey LEDH score adapter that computes the no-tape
total derivative of the same realized finite-`N` LEDH estimator:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

for row:

```text
zhao_cui_predator_prey_T20
```

The score vector must use physical parameter order:

```text
r, K, a, s, u, v
```

## Entry Conditions Inherited From Previous Phase

- Phase 4 value replay passed.
- Phase 4 route inventory found no admitted predator-prey total-score adapter.
- Predator-prey local-density manual scores exist but are not total LEDH score
  evidence.
- Directional FD is diagnostic only.
- LGSSM and fixed-SIR remain blocked/not admitted.

## Required Artifacts

Source value artifact:

- `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json`

Code artifacts to add or modify:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- possibly shared no-tape transport VJP helper code if duplicated benchmark
  helpers are factored.

Expected score artifacts if gates pass:

- tiny score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-tiny-score-artifact-2026-07-07.json`
- full score artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.json`
- score summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.md`
- Phase 4 repair result:
  `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-result-2026-07-07.md`
- Phase 5 actual-SV subplan:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-subplan-2026-07-07.md`
- repair review bundle:
  `docs/reviews/bayesfilter-ledh-score-per-model-phase4-predator-prey-repair-review-bundle-2026-07-07.md`

## Required Checks/Tests/Reviews

Pre-implementation checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Implementation checks to add:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py -q
```

Required test coverage:

- RK4 RHS VJP finite-difference unit test for state and all six parameters.
- RK4 transition-mean VJP finite-difference unit test for state and all six
  parameters.
- Tiny manual total-score route runs under runtime autodiff sentinel.
- Tiny all-parameter same-scalar finite-difference check.
- Score artifact rejects value-only and directional-only evidence.
- Score artifact validates only when all six parameters pass same-scalar
  correctness.
- Static guard verifies score route source contains no `GradientTape`,
  `ForwardAccumulator`, or `tf.stop_gradient`.

Trusted GPU/CUDA full-row run, only after tiny/schema checks pass:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  --device-scope visible \
  --expect-device-kind gpu \
  --device /GPU:0 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --num-particles 10000 \
  --time-steps 20 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 1.0 \
  --transport-plan-mode streaming \
  --transport-gradient-mode manual_streaming_finite \
  --transport-ad-mode full \
  --row-chunk-size 512 \
  --col-chunk-size 512 \
  --particle-chunk-size 512 \
  --dtype float32 \
  --tf32-mode enabled \
  --output docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.json \
  --markdown-output docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-score-artifact-2026-07-07.md
```

Review:

- bounded read-only review of the repair subplan before implementation;
- bounded read-only review of the repair result and Phase 5 actual-SV subplan
  before Phase 5 execution.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can we implement a predator-prey no-tape total derivative of the same finite-`N` LEDH `log_likelihood` scalar as the value artifact? |
| Baseline/comparator | Admitted predator-prey value artifact; value runner; model-local manual density scores; fixed-SIR manual reverse-scan structure; same-scalar finite differences with fixed randomness. |
| Primary criterion | A score artifact validates with `require_admitted=True` only after the manual total-VJP route passes tiny checks, all six parameters pass same-scalar correctness, full-row T=20,N=10000 identity matches the value artifact, memory passes, and no tape/stopped partial route is used. |
| Veto diagnostics | Missing state VJP through RK4; local-density-only score; value-only score; directional-only correctness; wrong scalar; wrong observation policy; wrong theta coordinate/order; tape/ForwardAccumulator/hidden autodiff; stopped partial derivative; nonfinite score; memory/device failure. |
| Explanatory diagnostics | RK4 VJP FD errors, tiny all-parameter FD errors, runtime, memory, device placement, component decomposition. |
| Not concluded | Native Zhao-Cui TT/SIRT source-faithfulness, exact nonlinear likelihood correctness beyond the declared additive-Gaussian row, HMC readiness, posterior correctness, scientific superiority, runtime ranking, or all-algorithm comparison. |
| Artifact | Score adapter, tests, tiny/full score artifacts if admitted, repair result, review bundle. |

## Step-By-Step Plan

1. Create `benchmark_ledh_same_target_predator_prey_score.py` as a score
   adapter sibling of the value runner.
2. Keep constants aligned with the value runner:
   - row id `zhao_cui_predator_prey_T20`;
   - dataset seed `81104`;
   - `T=20`, `N=10000`, batch seeds `81120..81124`;
   - physical theta `[0.6,114.0,25.0,0.3,0.5,0.5]`;
   - target observation policy `additive_gaussian_predator_prey`;
   - same transport policy and chunk defaults.
3. Implement predator-prey RHS VJP manually:
   - for interaction `I = prey * predator / (a + prey)`;
   - propagate cotangents to state `(prey,predator)` and parameters
     `(r,K,a,s,u,v)`;
   - use TensorFlow ops only.
4. Implement predator-prey RK4 transition-mean VJP manually:
   - store RK4 stage auxiliaries during forward replay;
   - reverse through the exact value-runner RK4 update, including the `k4`
     input `state + step * k3`;
   - return cotangent to previous particles and six-parameter score
     contribution.
5. Implement fixed-randomness tensor construction:
   - initial particles identical to the value runner;
   - process noise identical to the value runner's stateless seeds;
   - fixed resampling mask identical to the value runner.
6. Implement forward scan matching the value runner:
   - transition/prior mean;
   - pre-flow process-noise push;
   - identity observation surface;
   - LEDH linearized flow with aux;
   - target transition and observation log densities;
   - corrected log weights and normalization;
   - streaming manual transport forward.
7. Implement reverse scan:
   - transport total VJP;
   - log-weight normalization VJP;
   - correction-term VJPs;
   - transition Gaussian VJP;
   - observation Gaussian VJP;
   - LEDH linearized flow VJP;
   - pre-flow process-noise VJP;
   - RK4 transition VJP for all cotangents to prior means;
   - accumulation of per-seed six-parameter scores.
8. Add tiny all-parameter same-scalar FD:
   - central coordinate-wise finite differences for all six parameters;
   - fixed randomness and same scalar;
   - CPU-hidden or small GPU allowed only as diagnostic before full run.
9. Add score artifact normalization:
   - use Phase 1 score schema;
   - `score_derivative_provenance =
     manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot`;
   - require all-parameter correctness before admission.
10. Run focused local checks.
11. If tiny checks pass, run trusted full-row GPU score/memory.
12. If full all-parameter correctness or memory fails, write blocker result
    instead of admitting the score.
13. If full artifact validates, write admitted Phase 4 repair result.
14. Draft Phase 5 actual-SV subplan and run read-only review.

## Forbidden Claims/Actions

- Do not use local-density parameter scores as total LEDH score evidence.
- Do not omit state cotangent propagation through the RK4 dynamics.
- Do not use value-only artifacts as score evidence.
- Do not use directional FD alone as full score admission evidence.
- Do not use tape, ForwardAccumulator, hidden autodiff, or stopped partials in
  admitted score computation.
- Do not change row target, horizon, particle count, seed list, theta
  coordinate, parameter order, or observation policy.
- Do not claim native Zhao-Cui TT/SIRT source-faithfulness.
- Do not claim exact nonlinear likelihood correctness beyond the declared
  additive-Gaussian row, HMC readiness, posterior correctness, scientific
  superiority, runtime ranking, or all-algorithm comparison.

## Exact Next-Phase Handoff Conditions

Phase 5 actual-SV may start only if:

- Phase 4 repair writes an admitted predator-prey score result or an explicit
  blocker result;
- Phase 5 actual-SV subplan exists;
- review agrees the predator-prey decision is boundary-safe.

## Stop Conditions

Stop and write a blocker result if:

- manual RK4 state/parameter VJP cannot be validated;
- reverse scan cannot be tied exactly to the admitted value scalar;
- all-parameter correctness cannot be established;
- no-tape provenance becomes ambiguous;
- target/parameter/order/observation-policy mismatch appears;
- memory/device gates fail;
- review finds a material issue that does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Score adapter must replay the exact admitted predator-prey value target and constants. |
| Proxy metric promotion | Runtime, finite output, local-density score, and directional FD cannot admit score. |
| Missing stop condition | Stop on missing RK4 state VJP, missing all-parameter correctness, or no-tape ambiguity. |
| Hidden assumption | Physical theta `[r,K,a,s,u,v]`; no log transform or KSC/SV target reuse. |
| Stale context | Start by matching the current value runner source and artifact. |
| Environment mismatch | GPU/CUDA full-row run requires trusted execution; CPU-only checks hide GPU before TensorFlow import. |
| Useless artifact | Full score artifact must validate against the admitted value artifact with `require_admitted=True`. |
