# Phase 5 Subplan Review, Iteration 2

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Reviewer: fresh bounded Codex substitute reviewer

## Remaining Material Findings

1. The finite scalar used `previous_log_weight` at `t=0` without freezing the
   initial weights.
2. The derivative-certificate section allowed Phase 6 cleanup after a blocked
   result while the single handoff gate required derivative certification for
   any Phase 6 start.

The frozen normalization, direction-axis mapping, aggregation, and
initialization dependency map were otherwise consistent. The `0 ULP` gate is
strict but fail-closed.

`VERDICT: REVISE`

## Visible Repair

- The scalar and route identity now set
  `logw_{-1,i}=-log(N)` explicitly, including its use at `t=0`.
- The handoff is split into certified progression and cleanup-only blocked
  progression. Cleanup-only Phase 6 carries no canonical-gradient authority and
  must stop before documentation/scientific/admission phases until Phase 5 is
  repaired and certified.
