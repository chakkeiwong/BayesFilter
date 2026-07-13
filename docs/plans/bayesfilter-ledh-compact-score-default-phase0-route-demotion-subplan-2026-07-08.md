# Phase 0 Subplan: Route Demotion And Policy Gate

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Demote the old memory-inefficient LEDH score routes to historical diagnostic
status and prepare the repository to reject them as leaderboard score
admission routes.

The old routes are wrong relative to the new leaderboard score target because
they keep all-time reverse records or use reverse transport pullbacks as the
full-row score mechanism. They may remain only as diagnostics during migration.

The only future admitted default style is compact forward sensitivity carrying
value state and parameter tangents through the filtering loop with streaming
transport value+JVP.

## Entry Conditions Inherited From Previous Phase

- User explicitly directed that the old memory-inefficient score path be
  demoted as historical and wrong.
- LGSSM compact route exists and is the reference memory style.
- Actual-SV wiring audit found `records.append(...)` and `reversed(records)`
  in the active score path.
- Current score validator still allows several
  `manual_total_vjp_no_autodiff_same_scalar_*` route labels, so policy and
  tests need to be tightened before model ports.

## Required Artifacts

- Phase 0 result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase0-route-demotion-result-2026-07-08.md`
- Updated execution ledger:
  `docs/plans/bayesfilter-ledh-compact-score-default-visible-execution-ledger-2026-07-08.md`
- Phase 1 subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase0-route-demotion-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Local checks:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_fixed_sir_score_phase3_contract.py \
  tests/highdim/test_ledh_predator_prey_score_phase4_contract.py -q
```

The phase suffixes in these test filenames come from prior runbooks and do not
match this new master program's phase numbering. Phase 0 must map tests to
models by file content and row ID, not by the old filename suffix.

Static inventory:

```bash
rg -n "records\\.append|reversed\\(records\\)|manual_total_vjp_no_autodiff|historical_diagnostic_manual_reverse|compact_forward_sensitivity" \
  bayesfilter/highdim docs/benchmarks tests/highdim
```

Review:

- Claude read-only review of the Phase 0 result and Phase 1 subplan if Claude
  is available.
- If Claude review hangs, run a tiny probe. If the probe succeeds, narrow the
  review prompt. If Claude is unavailable or policy-blocked, write a fresh
  Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the repository have a clear, enforceable boundary that old reverse/manual-total-VJP score routes are historical and wrong for leaderboard score admission? |
| Baseline/comparator | Current score validator allowlist, LGSSM compact route IDs, actual-SV reverse-record implementation, fixed-SIR/predator-prey manual-total route labels. |
| Primary criterion | Phase 0 identifies all current old-route admission surfaces, records the demotion decision, and drafts Phase 1 validator/static-test work to reject them. |
| Veto diagnostics | Any plan text says `manual_total_vjp*` remains admissible default; any old route can be collected as leaderboard score without compact migration; any full-admission command such as `--admit-full` is run before Phase 1 code guards land; target scalar ambiguity; missing stop conditions. |
| Explanatory diagnostics | Counts and file/line anchors for old route patterns; existing tests that still assume old route labels; expected breakages for Phase 1. |
| Not concluded | No model compact port is complete, no score artifact is newly admitted, no leaderboard result is rebuilt. |
| Artifact | Phase 0 result plus Phase 1 subplan. |

## Forbidden Claims And Actions

- Do not claim any new score admission.
- Do not run full `N=10000` score commands in Phase 0.
- Do not run any full-admission command or `--admit-full` command before
  Phase 1 validator/static guards reject `manual_total_vjp*` routes.
- Do not change row targets or parameter definitions.
- Do not remove historical diagnostic code in Phase 0 unless a focused local
  test proves it is unused and a reviewed plan explicitly authorizes removal.
- Do not call reverse/manual-total-VJP routes acceptable for future leaderboard
  score admission.
- Do not use `GradientTape`, `ForwardAccumulator`, stopped partial derivative,
  or hidden autodiff.

## Exact Next-Phase Handoff Conditions

Phase 1 may start only if:

- Phase 0 result classifies old route labels as
  `historical_wrong_for_leaderboard_score_admission`;
- Phase 0 result lists concrete validator and static-test edits needed;
- local checks either pass or any failures are documented as expected because
  Phase 1 will tighten the validator;
- Phase 1 subplan exists and names exact contract/test changes;
- read-only review returns `VERDICT: AGREE` or a substitute Codex review is
  recorded because Claude is unavailable.

## Stop Conditions

Stop and ask for direction if:

- current code cannot distinguish compact forward sensitivity from reverse
  record/manual-total-VJP route labels;
- the user wants old manual-total-VJP route labels to remain admissible;
- route demotion would require changing scientific row targets;
- local checks reveal unrelated broken tests that block safe validator work;
- Claude and Codex review do not converge after five rounds on the same
  material blocker.
