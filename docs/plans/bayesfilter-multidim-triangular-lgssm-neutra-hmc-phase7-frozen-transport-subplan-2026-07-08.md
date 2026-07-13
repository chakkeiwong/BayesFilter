# Phase 7 Subplan: Frozen Transport Packaging

Date: 2026-07-08

## Phase Objective

Freeze the Phase 6 NeuTra transport into a BayesFilter payload and validate
loader/value-score mechanics.

## Entry Conditions Inherited From Previous Phase

- Phase 6 produced a finite GPU/JIT training artifact.

## Required Artifacts

- Frozen payload JSON, loader validation JSON, Phase 7 result.

## Required Checks/Tests/Reviews

- Hash/signature validation.
- Loader finalization.
- Fixed-transport finite value/score compile smoke with `jit_compile=True`.
- No HMC chains.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the trained transport be safely frozen and bound to the target? |
| Baseline/comparator | Phase 6 training state and Phase 4 target signature. |
| Primary criterion | Frozen payload loads and finite mechanics compile passes. |
| Veto diagnostics | Missing tensors, signature mismatch, nonfinite transport, hidden training/sampling. |
| Explanatory diagnostics | Payload size/hash, compile timing. |
| Not concluded | HMC convergence. |
| Artifact | Payload/result. |

## Forbidden Claims/Actions

- Do not run HMC chains.
- Do not repair payload by retraining without Phase 6 review.

## Exact Next-Phase Handoff Conditions

Phase 8 may begin only if frozen transport mechanics pass.

## Stop Conditions

Stop for malformed payload or mechanics failure.
