# Phase 4 Subplan: Expansion Decision

Date: 2026-07-09
Status: `DRAFT_REFRESH_AFTER_PHASE_3`

## Phase Objective

Choose the next program direction based on validated evidence: dimensional
lift, Zhao-Cui source-anchor lane, covariance/tuning repair, or closeout
without expansion.

## Entry Conditions

- Phase 3 result exists, or Phase 3 is explicitly skipped by reviewed
  closeout-level rationale.

## Required Artifacts

- Phase 4 decision result.
- Draft next master program or explicit no-next-program handoff.
- Refreshed Phase 5 closeout subplan.

## Required Checks, Tests, And Reviews

- `git diff --check`.
- Review of decision boundaries and nonclaims.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What is the next justified work item after scalar HMC validation evidence? |
| Baseline/comparator | Completed Phases 0-3 and their veto diagnostics. |
| Primary criterion | Decision names evidence, residual gaps, and exact next boundary without overclaiming. |
| Veto diagnostics | Unsupported source-faithfulness, default-readiness, production-readiness, or broad scientific claim. |
| Explanatory diagnostics | Phase pass/fail table, residual uncertainty, and cost/risk of each next lane. |
| Not concluded | Any claim not supported by earlier gates. |

## Forbidden Claims And Actions

- Do not launch a new source-faithful Zhao-Cui lane without paper/source
  anchors.
- Do not change project defaults.
- Do not merge separate scientific claims into one promotion.

## Exact Next-Phase Handoff Conditions

Advance to Phase 5 after the decision result is reviewed and closeout scope is
clear.

## Stop Conditions

Stop if a human project-direction decision is needed or review does not
converge.
