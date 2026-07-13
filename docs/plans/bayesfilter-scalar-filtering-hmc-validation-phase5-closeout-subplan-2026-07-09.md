# Phase 5 Subplan: Final Closeout

Date: 2026-07-09
Status: `DRAFT_REFRESH_AFTER_PHASE_4`

## Phase Objective

Close the HMC validation master program with a decision table,
inference-status table, run manifest, artifact list, residual gaps, and
explicit nonclaims.

## Entry Conditions

- Phase 4 decision result exists and has been reviewed.

## Required Artifacts

- Phase 5 closeout result.
- Updated visible execution ledger.
- Updated stop handoff.
- Optional reset memo if important state changed.

## Required Checks, Tests, And Reviews

- `git diff --check`.
- Claim-boundary audit with Claude or Codex substitute review.
- `git status -sb` and `git ls-files --others --exclude-standard` before final
  handoff if committing is requested.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What did this program establish, what failed, and what remains unproved? |
| Baseline/comparator | Master program objective and phase results. |
| Primary criterion | Closeout accurately separates evidence, failures, repairs, residual gaps, and nonclaims. |
| Veto diagnostics | Missing result, unsupported claim, missing manifest, missing inference-status table, or unreviewed material boundary. |
| Explanatory diagnostics | Phase table, run manifests, review statuses, artifact list, residual uncertainties. |
| Not concluded | Anything not supported by explicit phase evidence. |

## Forbidden Claims And Actions

- Do not claim readiness/convergence/posterior correctness unless the exact
  preceding gates support that limited claim.
- Do not hide failed or inconclusive phases.
- Do not commit ignored generated logs unless they support a claim and are
  intentionally tracked.

## Exact Next-Phase Handoff Conditions

No automatic next phase.  Final handoff must identify any new reviewed plan
needed.

## Stop Conditions

Stop if closeout review finds unsupported claims or artifact mismatch that
cannot be fixed in place.
