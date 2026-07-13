# Phase 4 Subplan: Fixed-SIR Compact Score Port

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Replace the fixed-SIR LEDH score default with compact forward sensitivity, or
write a blocker result if the p8p fixed-SIR recurrence cannot be ported without
changing the target scalar.

The current fixed-SIR adapter delegates to the parameterized SIR diagnostic
manual-total-VJP implementation and emits:

```text
manual_total_vjp_no_autodiff_same_scalar_fixed_sir_logscale_ledh_pfpf_ot
```

That route is historical and wrong for leaderboard score admission under this
master program. Phase 4 must not treat it as the admitted/default route.

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared contract blocks historical `manual_total_vjp*` routes from
  full score admission.
- Phase 2 froze LGSSM as the compact reference.
- Phase 3 actual-SV passed a tiny compact same-scalar gate and left full-row
  memory evidence unclaimed.
- Fixed-SIR has an admitted value artifact:
  `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json`.
- Fixed-SIR score code currently requires the historical route in
  `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`.

## Required Artifacts

- Updated fixed-SIR score implementation:
  `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py`
- Updated/added fixed-SIR tests:
  `tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py`
- Tiny compact score diagnostic artifact:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-tiny-compact-score-2026-07-08.json`
- Phase 4 result or blocker result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-result-2026-07-08.md`
- Phase 5 predator-prey subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Local CPU-only checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Additional targeted checks before implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python - <<'PY'
from docs.benchmarks import benchmark_ledh_same_target_fixed_sir_score as fixed_sir
from docs.benchmarks import benchmark_p8p_parameterized_sir_gradient as p8p
print(fixed_sir.FIXED_SIR_MANUAL_SCORE_ROUTE_ID)
print(hasattr(p8p, "_manual_value_and_score_from_components"))
PY
```

Tiny same-scalar diagnostic command must be drafted after the compact fixed-SIR
entry point exists. It must use the admitted fixed-SIR value artifact and a
small `N,T` setting.

Review:

- Claude read-only review of implementation, tests, Phase 4 result, and Phase
  5 subplan when available.
- Use the probe ladder on timeout. If Claude is unavailable, write a Codex
  substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can fixed-SIR compute the same finite-`N` LEDH `log_likelihood` score in `sir_log_scale_theta` coordinates using compact forward sensitivity instead of the p8p reverse/manual-total-VJP route? |
| Baseline/comparator | Admitted fixed-SIR value artifact, current fixed-SIR adapter, p8p historical diagnostic implementation, LGSSM compact style, and tiny same-scalar finite differences. |
| Primary criterion | Fixed-SIR default score route carries particles/log weights/tangents/log-likelihood tangents forward, emits compact provenance, matches the value scalar, passes tiny coordinate FD, and old manual-total-VJP cannot full-admit. |
| Veto diagnostics | Wrong target scalar, parameterized diagnostic row promotion, old p8p route used as default/admitted score, `manual_total_vjp*` full admission, `records.append` plus reverse scan in default score, reverse transport pullback as default, tape/autodiff, stopped partial derivative, wrong parameter coordinates, nonfinite score, tiny FD failure. |
| Explanatory diagnostics | Runtime, memory, per-coordinate FD error, parity with historical diagnostic, chunk-size invariance. |
| Not concluded | Full `N=10000` fixed-SIR score admission, HMC readiness, posterior correctness, source-faithful Zhao-Cui claim, runtime ranking, or scientific superiority. |
| Artifact | Tiny compact score JSON, Phase 4 result or blocker, tests. |

## Required Implementation Steps

1. Inspect `docs/benchmarks/benchmark_ledh_same_target_fixed_sir_score.py` and
   the p8p implementation it delegates to.
2. Identify the exact fixed-SIR value scalar, RNG/fixed-branch inputs, free
   parameter order, and current transition/observation correction formula.
3. Determine whether p8p exposes enough forward per-step structure to carry:
   particles, log weights, particle tangents, log-weight tangents, log
   likelihood, and log-likelihood tangents without all-time reverse records.
4. Add a compact fixed-SIR route ID only after the recurrence is actually
   implemented.
5. Implement the compact route or write a blocker result explaining the
   smallest missing p8p/value-path primitive.
6. Update the adapter so default score artifacts use compact provenance only
   after compact tests pass.
7. Keep old manual-total-VJP route historical/diagnostic and rejected for full
   admission.
8. Add static tests against the default fixed-SIR score route for:
   `records.append`, `reversed(records)`, reverse transport pullback,
   `GradientTape`, and `ForwardAccumulator`.
9. Add tiny same-scalar value match and coordinate FD tests.
10. Write Phase 4 result or blocker, draft Phase 5 predator-prey subplan, and
    review the handoff.

## Forbidden Claims And Actions

- Do not rename the p8p/manual-total-VJP route as compact.
- Do not use the parameterized SIR diagnostic row as the main fixed-SIR score
  artifact.
- Do not claim Zhao-Cui source-faithfulness unless the source-anchor gate is
  separately satisfied.
- Do not run full `N=10000` before tiny compact correctness passes and a
  reviewed amendment authorizes the full run.
- Do not change row ID, target scalar, parameter coordinate system, observation
  policy, or value artifact.
- Do not use tape/autodiff, stopped partial derivatives, hidden autodiff, or
  reverse-record score as default.

## Exact Next-Phase Handoff Conditions

Phase 5 predator-prey may start only if:

- fixed-SIR default score route is compact or fixed-SIR is explicitly blocked
  with a result artifact;
- old fixed-SIR manual-total route cannot full-admit;
- tiny same-scalar FD either passes or the blocker result explains the smallest
  unresolved issue;
- Phase 5 subplan exists and is reviewed;
- Codex records review findings and confirms no unresolved boundary issue.

## Stop Conditions

Stop and ask for direction or write a blocker result if:

- fixed-SIR compact JVP cannot be derived without changing the target scalar;
- p8p does not expose enough forward value-path structure for compact
  recurrence without a larger refactor;
- tiny same-scalar FD fails and cannot be localized in this phase;
- implementation would require unrelated dirty-worktree changes;
- memory/runtime behavior suggests even tiny compact fixed-SIR is not viable;
- Claude and Codex review do not converge after five rounds on the same
  material blocker.
