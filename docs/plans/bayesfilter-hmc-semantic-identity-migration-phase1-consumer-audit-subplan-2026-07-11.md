# Phase 1 Subplan: Runtime Consumer Audit And Field Classification

Date: 2026-07-11

Status: `COMPLETED`

## Phase Objective

Trace the refreshed private replay through reconstruction, transition runner
construction, and Phase 7 control. Classify every consumed field as transition,
execution, provenance, integrity, or excluded before defining any schema.

## Entry Conditions

- Phase 0 result is passed and records the review trail.
- The P7G blocker and refreshed replay remain unchanged.
- No identity implementation has started.

## Required Artifacts

- A source-anchored consumer graph from private replay to HMC transition.
- A complete field-classification table with reason, source consumer, identity
  role, and mismatch action.
- A list of derived objects whose validation must precede hashing.
- Phase 1 result and a drafted Phase 2 implementation subplan.

## Required Checks And Reviews

- Inspect `build_retained_frozen_kernel_hmc_adapter_from_tuning_payload`, mass
  reconstruction, adapter-signature logic, Phase 7 worker construction,
  `FixedSizeHMCChunkConfig`, and seed/chunk scheduling.
- Programmatically enumerate refreshed replay and Phase 7 config fields to
  expose unclassified keys.
- Require every execution-consumed value to have exactly one primary role;
  explicitly record legitimate cross-links rather than duplicating ownership.
- Review the Phase 1 result and Phase 2 subplan for omissions and hash/runtime
  drift risk.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Which exact values affect one transition, deterministic run reproduction, selection history, or artifact integrity? |
| Baseline | Current replay constructor, Phase 7 worker/controller, refreshed replay, and Phase 7 config. |
| Primary criterion | Every consumed or signed field is source-anchored and classified; no execution-affecting unknown remains. |
| Vetoes | Unclassified consumer, inferred field without source anchor, mechanics classified as provenance, or proposed allowlist-by-omission. |
| Explanatory only | Legacy stage hashes and unused payload fields. |
| Not concluded | No schema correctness, old/new semantic equality, Phase 7 readiness, or scientific claim. |

## Forbidden Claims And Actions

- Do not write schema or validator implementation.
- Do not update pins or migration decisions.
- Do not run HMC or mutate Phase 6 artifacts.
- Do not conclude equality from currently visible mechanics alone.

## Exact Handoff

Phase 2 may start only after the field ledger has no unknown execution consumer,
the Phase 1 result and Phase 2 subplan pass review, and the ledger/handoff are
updated.

## Stop Conditions

- An execution-affecting input cannot be derived or validated.
- Current source and refreshed replay disagree structurally.
- Scoped files change unexpectedly during the audit.
- Review does not converge after five substantive rounds.
