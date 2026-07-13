# Phase 3 Subplan: Trusted GPU/XLA Reproduction

Date: 2026-07-09
Status: `DRAFT_REFRESH_AFTER_PHASE_2`

## Phase Objective

Reproduce the accepted scalar HMC validation evidence under BayesFilter trusted
GPU/XLA provenance, without changing posterior or default-readiness claims.

## Entry Conditions

- Phase 2 reference agreement passes or a reviewed exception explains why GPU
  execution is needed before reference agreement.
- GPU/XLA runtime boundary is approved at execution time.

## Required Artifacts

- GPU provenance JSON.
- GPU/XLA HMC validation JSON/Markdown.
- Phase 3 result and refreshed Phase 4 subplan.

## Required Checks, Tests, And Reviews

- Trusted GPU visibility/provenance check.
- GPU/XLA benchmark with structured artifacts and quiet logs.
- Focused tests if GPU/XLA route code changes.
- Claude/Codex review of result and Phase 4 handoff.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the scalar validation route reproduce under trusted GPU/XLA provenance? |
| Baseline/comparator | Accepted CPU/reference Phase 1/2 artifacts. |
| Primary criterion | GPU/XLA artifact matches required validation fields and has no runtime/provenance/hard-veto failure. |
| Veto diagnostics | Hidden or unavailable GPU, missing trusted provenance, nonfinite output, runtime error, XLA route mismatch, positive native divergence if available, or unsupported production/default claim. |
| Explanatory diagnostics | Runtime, device, TF32/XLA settings, acceptance/log-accept summaries, reference metrics if repeated. |
| Not concluded | Production/default readiness, sampler superiority, broad HMC readiness, or source faithfulness. |

## Forbidden Claims And Actions

- Do not treat GPU/XLA execution alone as posterior correctness.
- Do not change default policy.
- Do not run untrusted GPU commands if trusted context is required.

## Exact Next-Phase Handoff Conditions

Advance to Phase 4 only after GPU/XLA result is reviewed and all nonclaims are
preserved.

## Stop Conditions

Stop for missing trusted GPU approval/provenance, runtime failure without a
repair plan, or review nonconvergence.
