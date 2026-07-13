# Phase 5 Subplan: Predator-Prey Compact Score Port

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Replace the predator-prey LEDH score default with compact forward sensitivity, or write a blocker result if the existing predator-prey score recurrence cannot be ported without changing the target scalar.

The current predator-prey score module exposes:

```text
manual_total_vjp_no_autodiff_same_scalar_predator_prey_ledh_pfpf_ot
```

That route is historical and wrong for leaderboard score admission under this master program. Phase 5 must not rename it or treat it as the admitted/default route.

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared contract blocks historical `manual_total_vjp*` routes from full score admission.
- Phase 2 froze LGSSM as compact reference.
- Phase 3 actual-SV has a compact tiny same-scalar route.
- Phase 4 fixed-SIR has a compact tiny same-scalar route.
- Predator-prey has an admitted value artifact:
  `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json`.
- Predator-prey score code currently lives in:
  `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`.
- Current predator-prey tests already include RHS/RK4 VJP finite-difference checks and tiny total-score checks, but they target the historical manual-total-VJP route.

## Required Artifacts

- Updated predator-prey score implementation:
  `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
- Updated predator-prey tests:
  `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
- Updated score contract if a compact predator-prey provenance constant is needed:
  `bayesfilter/highdim/ledh_score_contract.py`
- Tiny compact score diagnostic artifact:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-tiny-compact-score-2026-07-08.json`
- Phase 5 result or blocker result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-result-2026-07-08.md`
- Phase 6 generalized-SV subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase5-predator-prey-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Precheck commands:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py
```

Focused CPU-only checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Tiny diagnostic command must be drafted after a compact predator-prey entry point exists. It must use fixed randomness, a tiny `N,T` setting, all six physical parameters, manual streaming finite transport value+JVP, and no autodiff.

Review:

- Claude read-only review of implementation, tests, Phase 5 result, and Phase 6 subplan when available.
- Use the probe ladder on timeout. If Claude is unavailable, write a Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can predator-prey compute the same finite-`N` LEDH `log_likelihood` score in physical `(r,K,a,s,u,v)` coordinates using compact forward sensitivity instead of the historical reverse/manual-total-VJP route? |
| Baseline/comparator | Admitted predator-prey value artifact, existing predator-prey historical diagnostic route, fixed-SIR compact port pattern, LGSSM compact reference, and tiny same-scalar finite differences. |
| Primary criterion | Predator-prey default score route carries particles/log weights/tangents/log-likelihood tangents forward, emits compact provenance, matches the value scalar at the tiny gate, passes all-coordinate FD, and old manual-total-VJP cannot full-admit. |
| Veto diagnostics | Wrong target scalar, non-predator-prey row, old route used as default/admitted score, `manual_total_vjp*` full admission, reverse-record score default, tape/autodiff, stopped partial derivative, wrong physical parameter order, nonfinite score, tiny FD failure, or relabeling the historical route as compact. |
| Explanatory diagnostics | Runtime, memory, per-coordinate FD error, parity with historical diagnostic, chunk-size invariance, and component diagnostics. |
| Not concluded | Full `N=10000` predator-prey score admission, HMC readiness, posterior correctness, exact nonlinear likelihood correctness, Zhao-Cui source-faithfulness, runtime ranking, or scientific superiority. |
| Artifact | Tiny compact score JSON, Phase 5 result or blocker, tests. |

## Required Implementation Steps

1. Inspect the current predator-prey value route and score route:
   - `docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py`
   - `docs/benchmarks/benchmark_ledh_same_target_predator_prey_score.py`
   - `tests/highdim/test_ledh_predator_prey_score_phase4_contract.py`
2. Identify the exact value scalar, fixed RNG inputs, physical parameter order, transition covariance, observation covariance, correction formula, and resampling policy.
3. Inventory the existing reverse/manual VJP helpers and classify each as:
   - reusable value helper;
   - reusable local derivative formula;
   - historical reverse-only helper that must not be in the compact default.
4. Add a compact predator-prey route ID only after the compact recurrence is implemented.
5. Implement forward sensitivities for:
   - predator-prey RHS;
   - RK4 transition mean;
   - process-noise pushforward;
   - LEDH linearized flow;
   - transition and observation Gaussian log densities;
   - normalized log-weight increment;
   - streaming finite Sinkhorn transport value+JVP.
6. Add compact artifact normalization with compact provenance, separate from historical route normalization.
7. Keep the old manual-total-VJP route historical/diagnostic and rejected for full admission.
8. Add static tests against compact default symbols for:
   `records.append`, `reversed(records)`, reverse transport pullback, `GradientTape`, and `ForwardAccumulator`.
9. Add tiny same-scalar value match and all-coordinate FD tests.
10. Write Phase 5 result or blocker, draft Phase 6 generalized-SV subplan, and review the handoff.

## Forbidden Claims And Actions

- Do not rename the historical predator-prey manual-total-VJP route as compact.
- Do not use non-LEDH SGQF/UKF/dense predator-prey results as LEDH score admission.
- Do not change row ID, target scalar, target observation policy, physical parameter order, observations, or value artifact.
- Do not claim Zhao-Cui source-faithfulness unless the source-anchor gate is separately satisfied.
- Do not run full `N=10000` before tiny compact correctness passes and a reviewed amendment authorizes the full run.
- Do not use tape/autodiff, stopped partial derivatives, hidden autodiff, or reverse-record score as default.

## Exact Next-Phase Handoff Conditions

Phase 6 generalized-SV may start only if:

- predator-prey default score route is compact or predator-prey is explicitly blocked with a result artifact;
- old predator-prey manual-total route cannot full-admit;
- tiny same-scalar FD either passes or the blocker result explains the smallest unresolved issue;
- Phase 6 subplan exists and is reviewed;
- Codex records review findings and confirms no unresolved boundary issue.

## Stop Conditions

Stop and ask for direction or write a blocker result if:

- predator-prey compact JVP cannot be derived without changing the target scalar;
- existing value/score code does not expose enough forward value-path structure for compact recurrence without a larger refactor;
- tiny same-scalar FD fails and cannot be localized in this phase;
- implementation would require unrelated dirty-worktree changes;
- memory/runtime behavior suggests even tiny compact predator-prey is not viable;
- Claude and Codex review do not converge after five rounds on the same material blocker.

## Skeptical Audit Before Execution

Risks checked before launch:

- Wrong baseline: the baseline is the admitted LEDH predator-prey value artifact, not SGQF/UKF/dense predator-prey diagnostics.
- Proxy metrics: tiny FD may pass but still cannot admit a full row without memory evidence.
- Hidden assumption: the parameter order is physical `(r,K,a,s,u,v)` and must match the value artifact.
- Environment mismatch: CPU-only tests must hide GPU before TensorFlow import; later GPU runs require trusted execution.
- Useless artifact risk: any result must state score provenance, target scalar, and non-claims explicitly.

Audit status: ready for read-only review after Phase 4 result review.
