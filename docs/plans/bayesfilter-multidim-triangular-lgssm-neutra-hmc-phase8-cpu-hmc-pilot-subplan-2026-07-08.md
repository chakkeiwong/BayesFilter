# Phase 8 Subplan: CPU HMC Pilot

Date: 2026-07-08

## Phase Objective

Run a small CPU-hidden multicore `jit_compile=True` HMC pilot to validate chain
machinery before serious estimation.

## Entry Conditions Inherited From Previous Phase

- Phase 7 frozen transport mechanics passed.
- Explicit approval for pilot HMC runtime has been granted.

## Required Artifacts

- Pilot HMC JSON/logs and Phase 8 result.

## Required Checks/Tests/Reviews

- CPU-hidden environment.
- `jit_compile=True` only.
- Worker return codes, finite samples, no GPU sample generation.
- No promotion language.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the HMC worker machinery run without immediate failures on this target? |
| Baseline/comparator | Phase 7 frozen transport and target adapter. |
| Primary criterion | Finite pilot samples and clean worker metadata. |
| Veto diagnostics | `jit_compile=false`, GPU sample generation, worker failure, nonfinite samples. |
| Explanatory diagnostics | Acceptance, timing, preliminary residuals. |
| Not concluded | Serious HMC validity or readiness. |
| Artifact | Pilot JSON/result. |

## Forbidden Claims/Actions

- Do not claim readiness.
- Do not tune based on audit/test data without a reviewed repair plan.

## Exact Next-Phase Handoff Conditions

Phase 9 may begin only if the pilot passes and serious HMC settings are
predeclared.

## Stop Conditions

Stop for policy violation or pilot instability requiring repair.
