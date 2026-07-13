# Phase 8 Subplan: Gate-Approved Comparison Ladder

Date: 2026-07-11
Status: `DRAFT_NOT_EXECUTABLE_UNTIL_PHASE7_REFRESH`

## Phase Objective

Run only the CPU/GPU method/dtype/batch arms proven viable by Phases 6-7 and
produce descriptive or statistically supported comparisons according to a
prospectively refreshed replication/analysis contract.

## Entry Conditions

- Common engineering gates in Phases 0-5 pass; Phases 6-7 have classified CPU
  and GPU outcomes independently; at least one lane has a fair viable pair.
- This subplan is refreshed after Phase 7 with exact viable arms, commands,
  replication count, uncertainty method, time budget, and artifact paths.
- Claude/Codex review agrees the comparison is fair and bounded.

## Required Artifacts

- Source-fingerprinted per-run JSON/Markdown/logs under new 2026-07-11 names.
- Paired-run aggregation artifact with per-replication rows.
- Run manifest containing commit, dirty source hashes, environment, CPU/GPU status, seeds, wall time, commands, plan/result paths.
- Phase 8 result and refreshed Phase 9 subplan.

## Required Checks, Tests, And Reviews

- Exact viable grid and replication count must be filled before execution; placeholders are a stop condition.
- Verify the nested/common fixture contract across `P/B`: base-model and
  observation hashes, derivative-prefix consistency, proposal-cloud hash, and
  selected row identities. If it cannot hold, restrict conclusions to within-cell
  method comparisons and forbid causal/scaling attribution across cells.
- Verify the master program's timed callable boundary, fixture/source hashes,
  fresh-process/cache policy, synchronized warm timing, and parity before including a row.
- Freeze before data collection: sampling unit, pairing key, balanced/randomized
  method order, minimum usable replication count, estimand, uncertainty-bearing
  paired interval/test, and ranking rule. Otherwise state no supported ranking.
- Decision table, inference-status table, pre-mortem, and post-run red team.
- Material result review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Among viable equivalent true-batched methods, what runtime/compile behavior is supported on the declared devices? |
| Baseline | True-batched autodiff versus true-batched analytical, with scalar references excluded from large-batch timing ranking. |
| Primary criterion | All included rows pass hard vetoes and the declared paired analysis is complete. |
| Vetoes | Any invalid row, unfair semantics, stale artifact, missing replication, missing provenance, or post-hoc criterion change. |
| Explanatory only | Descriptive medians/tails/maxima, paired point estimates without uncertainty, and compile metrics unless the predeclared uncertainty procedure supports ranking. |
| Not concluded | Universal superiority, HMC/posterior/scientific/default/production readiness. |

## Forbidden Claims And Actions

- Do not execute this draft before Phase 7 refresh removes all placeholders.
- Do not rank from one run or descriptive tails.
- Do not silently drop failed replications or methods.
- Do not change criteria after observing results.

## Exact Next-Phase Handoff Conditions

- All launched artifacts complete or honest partial/failure result is written.
- Hard vetoes, viable arms, ranking status, descriptive-only differences, and next evidence are explicit.
- Phase 9 subplan is refreshed/reviewed.

## Stop Conditions

- Phase 7 leaves no fair pair of viable methods; write a negative/blocker result rather than forcing a comparison.
- Runtime exceeds reviewed budget.
- Artifact/parity/provenance veto fires.
- Five review rounds fail to converge.

## Mandatory Phase-End Sequence

1. Run required checks and aggregation validation.
2. Write Phase 8 result.
3. Refresh Phase 9 subplan.
4. Review and repair Phase 9 before advancing.
