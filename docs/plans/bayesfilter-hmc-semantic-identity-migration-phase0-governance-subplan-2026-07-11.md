# Phase 0 Subplan: Semantic Identity Migration Governance

Date: 2026-07-11

Status: `COMPLETED`

## Phase Objective

Create and review the master program, visible runbook, execution ledger, stop
handoff, and Phase 1 consumer-audit subplan without changing identity code or
running HMC.

## Entry Conditions

- The P7G exact legacy-hash gate is blocked and documented.
- The refreshed private replay exists and is preserved.
- The user authorized planning, scoped implementation, local checks, and
  governed review, while retaining explicit adoption/runtime boundaries.
- Concurrent LEDH/QR changes are user-owned and excluded.

## Required Artifacts

- `docs/plans/bayesfilter-hmc-semantic-identity-migration-master-program-2026-07-11.md`
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-runbook-2026-07-11.md`
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-ledger-2026-07-11.md`
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-visible-stop-handoff-2026-07-11.md`
- `docs/plans/bayesfilter-hmc-semantic-identity-migration-phase1-consumer-audit-subplan-2026-07-11.md`
- Phase 0 result and bounded review records.

## Required Checks And Reviews

- Verify all required paths exist and phase/result references are unique.
- Scan for contradictory authority, detached execution, silent repinning, and
  unsupported equality language.
- Run `git diff --check` on Phase 0 artifacts.
- Run the material master-program review with one path and one question.
- Run a fresh local findings-first consistency audit of the complete Phase 0
  artifact set.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the governance design solve the correct identity problem without weakening P7G or crossing adoption/runtime boundaries? |
| Baseline | Existing blocker, AGENTS policy, Claude review guide, visible-runbook template, and current source consumers. |
| Primary criterion | All artifacts exist, are consistent, and material review has no unresolved blocker. |
| Vetoes | Allowlist hashing, silent repin, detached launch, Claude execution authority, missing next-phase gate, or unsupported full equality. |
| Explanatory only | Review latency, number of review rounds, and document size. |
| Not concluded | No implementation correctness, semantic equality, replay readiness, convergence, or runtime claim. |

## Forbidden Claims And Actions

- Do not edit identity implementation.
- Do not regenerate or rewrite Phase 6 artifacts.
- Do not adopt a new baseline.
- Do not launch HMC, Phase 7 smoke/serious sampling, Phase 8, or NeuTra.
- Do not state that unavailable old private bytes are equal to the refresh.

## Exact Handoff

Phase 1 may start only when the Phase 0 result records successful local checks,
the material review status, a fresh Codex consistency verdict, and a reviewed
Phase 1 subplan. Phase 1 is read-only and must produce a source-anchored field
classification before any schema code is written.

## Stop Conditions

- A governance artifact permits silent repinning or runtime without approval.
- Required paths conflict with unrelated user work.
- Material review and Codex do not converge after five substantive rounds.
- Continuing requires an authority not granted for Phase 0.
