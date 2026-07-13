# Phase 3 Subplan: Actual-SV Compact Score Port

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Replace the actual-SV LEDH score route with the LGSSM-style compact forward
sensitivity mechanism.

The current actual-SV score route is historical and wrong for leaderboard score
admission because it stores all-time reverse records and uses reverse transport
pullbacks. The replacement must carry value state and parameter tangents
through the filtering loop and use streaming transport value+JVP.

## Entry Conditions Inherited From Previous Phase

- Phase 1 validator blocks `manual_total_vjp*` full admission.
- Phase 2 froze LGSSM as the compact reference style.
- Actual-SV has an admitted value artifact for the exact transformed
  `log_likelihood` scalar.
- Actual-SV current score code still contains `records.append(...)` and
  `reversed(records)`.

## Required Artifacts

- Updated actual-SV score implementation:
  `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- Updated/added tests:
  `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`
- Tiny score diagnostic artifact:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-tiny-compact-score-2026-07-08.json`
- Phase 3 result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-result-2026-07-08.md`
- Phase 4 fixed-SIR subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase3-actual-sv-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Local CPU-only checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Tiny no-tape same-scalar diagnostic:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  --source-value-artifact docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-tiny-compact-score-2026-07-08.json \
  --time-steps 2 --num-particles 9 --batch-seeds 81120 \
  --particle-chunk-size 4 --row-chunk-size 9 --col-chunk-size 9 \
  --sinkhorn-iterations 1 --transport-policy active-all \
  --dtype float64 --tf32-mode disabled
```

Trusted GPU checks may be planned only after tiny compact correctness passes.
Do not run full `N=10000,T=1000` in this phase unless a reviewed amendment
explicitly authorizes it.

Review:

- Claude read-only review of implementation, tests, Phase 3 result, and Phase
  4 subplan.
- Use probe ladder on timeout. If Claude is unavailable, write a Codex
  substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can actual-SV compute the same transformed finite-`N` LEDH `log_likelihood` score using compact forward sensitivity instead of reverse records? |
| Baseline/comparator | Admitted actual-SV value artifact, existing tiny same-scalar FD result, LGSSM compact score skeleton, and historical actual-SV reverse route as diagnostic only. |
| Primary criterion | Actual-SV compact route matches the value route scalar, uses no tape/autodiff, contains no all-time reverse-record score default, passes tiny all-coordinate same-scalar FD, and cannot full-admit until compact provenance and memory gates pass. |
| Veto diagnostics | Wrong target scalar, raw/KSC likelihood substitution, `records.append(...)` plus reverse scan in default score, reverse transport pullback as default, `manual_total_vjp*` full admission, tape/autodiff, stopped partial derivative, nonfinite score, tiny FD failure. |
| Explanatory diagnostics | Runtime, memory, per-coordinate FD error, parity with historical tiny route, chunk-size invariance. |
| Not concluded | Full `N=10000,T=1000` admission, HMC readiness, posterior correctness, runtime ranking, or scientific superiority. |
| Artifact | Tiny compact score JSON, Phase 3 result, tests. |

## Required Implementation Steps

1. Introduce an actual-SV compact route ID, distinct from the historical
   `manual_total_vjp*` route.
2. Add actual-SV parameter tangent helpers for:
   - the exact current `_gamma_beta` mapping, where `gamma` is the standard
     normal CDF transform of `gamma_unconstrained`;
   - the exact current `beta` mapping used by `_gamma_beta`;
   - stationary variance;
   - proposal mean/variance;
   - exact transformed target observation log density;
   - flow observation shift.
3. Implement compact LEDH flow value+JVP for actual-SV, mirroring the value
   route's streaming flow and padding behavior.
4. Implement compact transport by calling the same streaming transport
   value+JVP helper used by LGSSM.
5. Replace default score execution with a forward loop carrying:
   - particles;
   - log weights;
   - particle tangents;
   - log-weight tangents;
   - log likelihood;
   - log-likelihood tangents.
6. Keep the old reverse-record route only under a historical/diagnostic name or
   remove it from default dispatch.
7. Replace artifact-builder provenance and default score dispatch so compact
   actual-SV artifacts use the new compact route ID and the historical
   `ACTUAL_SV_MANUAL_SCORE_ROUTE_ID` cannot be emitted for any admitted/default
   score artifact.
8. Add static tests proving the admitted/default actual-SV score route does not
   use `records.append(...)`, `reversed(records)`, `GradientTape`, or
   `ForwardAccumulator`.
9. Add tiny coordinate FD tests against the same value scalar.
10. Write Phase 3 result and draft Phase 4 fixed-SIR subplan.

## Forbidden Claims And Actions

- Do not use the historical `manual_total_vjp*` route for admission.
- Do not run full `N=10000,T=1000` without a reviewed amendment after tiny
  compact correctness passes.
- Do not change actual-SV row ID, target scalar, observation policy, or
  parameter coordinate system.
- Do not claim KSC, generalized-SV, raw Gaussian observation, or
  augmented-noise Gaussian-closure evidence.
- Do not use tape/autodiff, stopped partial derivatives, or hidden
  autodiff.
- Do not describe tiny FD as broad mathematical correctness.

## Exact Next-Phase Handoff Conditions

Phase 4 fixed-SIR may start only if:

- actual-SV default score route is compact or actual-SV is explicitly blocked
  with a result artifact;
- tests prove old actual-SV manual-total route cannot full-admit;
- tiny compact same-scalar FD either passes or a blocker result explains the
  smallest unresolved issue;
- Phase 4 subplan exists and is reviewed;
- Codex records review findings and confirms no unresolved boundary issue.

## Stop Conditions

Stop and ask for direction if:

- actual-SV compact JVP cannot be derived without changing the target scalar;
- compact transport value+JVP is unusable for actual-SV after a focused
  diagnostic;
- tiny same-scalar FD fails and the error cannot be localized in this phase;
- memory/runtime behavior suggests even tiny compact route is not viable;
- implementation would require unrelated dirty-worktree changes;
- Claude and Codex review do not converge after five rounds on the same
  material blocker.
