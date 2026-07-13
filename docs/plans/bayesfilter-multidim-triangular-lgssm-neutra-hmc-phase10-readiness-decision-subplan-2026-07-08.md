# Phase 10 Subplan: Readiness Decision

Date: 2026-07-08

## Phase Objective

Classify the multidimensional triangular LGSSM NeuTra-HMC estimation evidence.

## Entry Conditions Inherited From Previous Phase

- Phase 9 wrote a serious HMC result or blocker.

## Required Artifacts

- Phase 10 decision result and optional JSON.
- Updated master/runbook/ledger.

## Required Checks/Tests/Reviews

- Parse Phase 0-9 artifacts.
- Verify all nonclaims and boundaries.
- Read-only review before final closeout.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What readiness classification is supported by Phase 0-9 evidence? |
| Baseline/comparator | All phase artifacts and the Phase 9 serious HMC result. |
| Primary criterion | Emit exactly one of `PASS_MULTIDIM_LGSSM_HMC_ESTIMATION`, `BLOCKED_FOR_STATIONARITY_OR_SCORE_REPAIR`, `BLOCKED_FOR_HMC_TUNING_REPAIR`, or `INSUFFICIENT_EVIDENCE_NO_PROMOTION`. |
| Veto diagnostics | Policy violation, malformed artifacts, missing diagnostics, unsupported claims. |
| Explanatory diagnostics | R-hat/ESS table, truth/reference diagnostics, uncertainty limits. |
| Not concluded | Product/default/scientific/broad LGSSM readiness unless separately proved. |
| Artifact | Phase 10 result/JSON. |

## Forbidden Claims/Actions

- Do not run new runtime.
- Do not promote descriptive differences to rankings.

## Exact Next-Phase Handoff Conditions

If passed, hand off to a new reviewed program for broader targets. If blocked,
write the smallest repair subplan. If insufficient, state missing evidence.

## Stop Conditions

Stop if the evidence does not support an unambiguous classification.
