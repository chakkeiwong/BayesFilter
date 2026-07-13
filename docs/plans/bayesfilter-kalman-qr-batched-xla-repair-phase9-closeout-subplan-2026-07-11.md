# Phase 9 Subplan: Closeout And Reset Memo

Date: 2026-07-11
Status: `DRAFT_AWAITING_PHASE8_HANDOFF`

## Phase Objective

Close the repair program with auditable engineering/scientific boundaries,
artifact maps, inference status, unresolved blockers, and a reboot-safe memo.

## Entry Conditions

- Phase 8 completed or wrote an honest blocker/negative result.
- All reached phase results and review trails exist.

## Required Artifacts

- Phase 9 result.
- Updated reset memo for the final reached state.
- Final artifact manifest/checksums and review trail.
- Updated ledger and stop handoff/final handoff.

## Required Checks, Tests, And Reviews

- Re-run focused non-long checks appropriate to touched files.
- Validate all JSON artifacts strictly and all referenced paths exist.
- Verify decision table and inference-status table contain hard veto screen,
  supported ranking, descriptive-only differences, default-readiness, and next evidence.
- Red-team unsupported claims and stale source hashes.
- Claude Opus max final read-only review or documented fresh Codex fallback only after confirmed transport failure.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Is the final record complete, reproducible, direct about failures, and bounded to evidence actually obtained? |
| Baseline | Master program and all reached phase contracts. |
| Primary criterion | Artifact references/checks pass and final claims match evidence classes. |
| Vetoes | Missing artifact, stale hash, unsupported ranking/readiness claim, hidden failed arm, or unresolved review finding. |
| Explanatory only | Future optimization ideas. |
| Not concluded | Anything outside explicit final evidence. |

## Forbidden Claims And Actions

- Do not delete failed artifacts.
- Do not upgrade compile viability or parity into runtime/scientific readiness.
- Do not commit/push unless separately requested.

## Exact Final Handoff Conditions

- Final status, reached phase, artifact paths, tests/runs, review trail,
  unresolved blockers, nonclaims, and safest next action are explicit.

## Stop Conditions

- Artifact integrity cannot be established.
- Final review fails to converge after five rounds.

## Mandatory Phase-End Sequence

1. Run final local checks.
2. Write Phase 9 result and reset memo.
3. Refresh final handoff/ledger.
4. Review the closeout and repair/recheck before declaring completion.
