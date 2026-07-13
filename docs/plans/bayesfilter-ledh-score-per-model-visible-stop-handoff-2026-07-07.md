# LEDH Score Per-Model Visible Stop Handoff

Date: 2026-07-07

Status: `READY_FOR_USE_IF_BLOCKED`

Use this file if the visible runbook must stop before completion.

## Active Program

- Master program:
  `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
- Runbook:
  `docs/plans/bayesfilter-ledh-score-per-model-visible-gated-execution-runbook-2026-07-07.md`
- Ledger:
  `docs/plans/bayesfilter-ledh-score-per-model-visible-execution-ledger-2026-07-07.md`

## Non-Negotiable Boundaries

- Do not admit any score before the corresponding value row is admitted.
- Do not use `GradientTape`, `ForwardAccumulator`, hidden autodiff, or stopped
  partial derivatives for admitted LEDH score evidence.
- Do not promote
  `zhao_cui_spatial_sir_austria_j9_T20_parameterized_logscale` as the main
  fixed SIR row.
- Do not claim KSC exact native actual-SV likelihood.
- Do not claim HMC readiness, posterior correctness, scientific superiority,
  runtime ranking, or all-algorithm comparison.

## Required Stop Record Fields

When stopping, add:

- phase and state;
- exact blocker;
- command or review that exposed it;
- artifacts written;
- local checks passed/failed;
- whether the blocker is fixable inside the current phase;
- next safest action;
- approval needed, if any.

## Human-Approval Stop Conditions

Stop for human approval before:

- changing any row target definition;
- changing score pass/fail criteria after seeing results;
- using network/package/data fetches;
- destructive git or filesystem actions;
- public-release, funding, product-capability, model-file, or scientific-claim
  boundaries.

## Current Recoverable State

Last updated: 2026-07-07T21:29:00Z

- Current phase gate:
  `PHASE5_FULL_ROW_BLOCKED_STREAMING_FLOW_PARITY_REPAIR_PENDING_REVIEW`.
- Phase 2 LGSSM score:
  `CLOSED_BLOCKED_FULL_SCORE_NOT_ADMITTED`.
- Phase 3 fixed-SIR score:
  `CLOSED_BLOCKED_FULL_SCORE_NOT_ADMITTED`.
- Phase 4 predator-prey score:
  `CLOSED_BLOCKED_FULL_SCORE_NOT_ADMITTED`.
- Latest local check:
  Phase 5 actual-SV score combined replay/schema checks passed:
  `28 passed, 2 warnings`.
- Latest artifacts:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-result-2026-07-07.md`,
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-streaming-flow-parity-repair-subplan-2026-07-07.md`.

Next safest action:

- Review the Phase 5 streaming-flow parity repair subplan.
- Do not launch full `N=10000,T=1000` actual-SV score admission until the score
  route forward scalar matches the admitted streaming-flow value route and
  tiny no-tape all-coordinate FD passes again.
