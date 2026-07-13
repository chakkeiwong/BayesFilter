# Phase 0 Subplan: Governance, Runbook, Review Gate

Date: 2026-07-09

## Phase Objective

Create the master program, phase subplans, visible gated runbook, execution
ledger, stop handoff, and compact Claude review bundle for deterministic LGSSM
HMC tuning.

## Entry Conditions Inherited From Previous Phase

- User approved operationalizing the deterministic tuning plan.
- Current branch is `main`; dirty worktree must be preserved.
- No serious HMC, NeuTra training, or GPU/CUDA command is authorized by this
  phase.

## Required Artifacts

- Master program:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-master-program-2026-07-09.md`
- Runbook:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-gated-execution-runbook-2026-07-09.md`
- Ledger:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-execution-ledger-2026-07-09.md`
- Stop handoff:
  `docs/plans/bayesfilter-deterministic-lgssm-hmc-tuning-visible-stop-handoff-2026-07-09.md`
- Review bundle:
  `docs/reviews/bayesfilter-deterministic-lgssm-hmc-tuning-launch-review-bundle-2026-07-09.md`

## Required Checks, Tests, Reviews

- Local file-existence check for master, runbook, ledger, review bundle, and all
  phase subplans.
- Claude read-only review of the compact launch bundle, max five repair rounds.
- If Claude is unavailable, run a documented Codex substitute review and mark
  it weaker than Claude agreement.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the planned deterministic tuning program prevent agent-driven tuning and preserve runtime/scientific boundaries? |
| Baseline/comparator | Existing BayesFilter tuning APIs and project governance. |
| Primary pass criterion | Required artifacts exist and material launch review returns `VERDICT: AGREE` or documented fallback with no material blocker. |
| Veto diagnostics | Missing subplan field, missing artifact path, manual tuning allowed, non-XLA fallback allowed, runtime approval bypassed, unsupported final claims. |
| Explanatory diagnostics | Claude review status, dirty worktree snapshot, local path check. |
| Not concluded | No implementation correctness, HMC readiness, posterior recovery, convergence, or runtime feasibility claim. |

## Forbidden Claims / Actions

- Do not run HMC, NeuTra training, GPU probes, or package/network setup.
- Do not claim the LGSSM recovery test is ready.
- Do not change pass/fail criteria after seeing results.
- Do not edit unrelated dirty files.

## Exact Next-Phase Handoff Conditions

- Phase 1 subplan exists and is referenced from the runbook.
- Phase 0 result records local checks and review status.
- Any reviewer caveats are reflected in the runbook or result before Phase 1.

## Stop Conditions

- Claude and Codex review do not converge after five rounds for the same
  blocker.
- A required artifact cannot be written without overwriting unrelated work.
- Review identifies a material boundary violation that cannot be patched locally.
