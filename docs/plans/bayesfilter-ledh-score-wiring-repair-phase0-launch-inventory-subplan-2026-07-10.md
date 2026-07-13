# Phase 0 Subplan: Launch Inventory And Governance Freeze

Date: 2026-07-10

## Phase Objective

Freeze the current wiring inventory, verify the master program has the right
logical dependencies, and establish Phase 1 handoff conditions before any code
changes.

## Entry Conditions

- User requested a master program with Claude read-only review and visible
  gated execution.
- Owner directive: LEDH score production defaults to `float32` TensorFlow with
  TF32 enabled.
- Prior evidence: LGSSM `N=10000,T=50` compact score-only diagnostic completed
  with compact route and no full-history reverse route.

## Required Artifacts

- Master program:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-master-program-2026-07-10.md`
- Visible runbook:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-gated-execution-runbook-2026-07-10.md`
- Visible execution ledger:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-execution-ledger-2026-07-10.md`
- Phase 0 result:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase0-launch-inventory-result-2026-07-10.md`
- Phase 1 subplan draft:
  `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-subplan-2026-07-10.md`
- Claude review bundle:
  `docs/reviews/bayesfilter-ledh-score-wiring-repair-launch-review-bundle-2026-07-10.md`

## Required Checks, Tests, Reviews

- Local inventory commands:
  - `rg -n -- "manual_total_vjp|memory_style|COMPACT_SCORE_ROUTE_ID|score_derivative_provenance|--dtype|--tf32-mode" docs/benchmarks bayesfilter/highdim tests/highdim`
  - `python -m py_compile` on all six score runner modules and shared score
    contract files.
- Claude read-only review of launch bundle, with max five repair rounds.
- Skeptical plan audit before any code edits.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the master program correctly target the score wiring failures and prevent relabeling old routes as compact computation? |
| Baseline/comparator | Current code inventory and the model-by-model classification from 2026-07-10. |
| Primary criterion | Master program, runbook, Phase 0 result, and Phase 1 subplan exist and state compact/default route, precision, FD, review, and stop-condition gates. |
| Veto diagnostics | Missing model phase; hidden default float64 route; plan allowing historical route full admission; plan treating score-only memory as correctness; plan launching full GPU ladder before wiring tests. |
| Explanatory diagnostics | Existing tests, existing score-memory artifacts, route constants, and current dirty worktree status. |
| Not concluded | No code repair, no model score admission, no leaderboard completion, no scientific or HMC readiness claim. |

## Forbidden Claims And Actions

- Do not claim any model is fully repaired in Phase 0.
- Do not run full GPU score-memory ladders in Phase 0.
- Do not admit score artifacts from historical routes.
- Do not edit model score code in Phase 0 except documentation/plan artifacts.
- Do not change pass/fail criteria after seeing later results.

## Exact Next-Phase Handoff Conditions

Advance to Phase 1 only if:

- master program and runbook exist;
- Phase 0 result records the route/precision inventory;
- Phase 1 subplan exists and is reviewed;
- Claude review agrees, or Claude is documented unavailable and Codex substitute
  review agrees;
- local py-compile check passes or any failure is classified as a pre-existing
  blocker with a repair subplan.

## Stop Conditions

- Claude and Codex do not converge after five review rounds on the launch plan.
- Inventory reveals a boundary requiring human decision beyond the owner
  directive.
- Required files are missing and cannot be reconstructed from local context.
- Local checks fail in a way that invalidates the Phase 1 plan.

## Skeptical Plan Audit

Passed for launch. The phase is inventory-only and cannot accidentally promote
proxy metrics or historical artifacts. The primary artifact is a plan/result
pair, not a score claim. Full GPU score work is explicitly deferred until after
default wiring and tests are repaired.
