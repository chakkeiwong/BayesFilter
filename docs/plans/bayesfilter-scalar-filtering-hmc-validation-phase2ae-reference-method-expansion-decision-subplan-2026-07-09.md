# Phase 2AE Subplan: Reference-Method Expansion Decision

Date: 2026-07-09
Status: `REVIEWED_READY_FOR_DECISION_RESULT_ONLY`

## Phase Objective

Decide whether to stop the current scalar sequential-reference branch as
blocked or open a materially different reference-method design after Phase 2AC
and Phase 2AD exposed a beta-completion versus ancestor-diversity tradeoff.

This phase is a decision and planning phase only.  It does not run HMC,
GPU/XLA, or another reference runtime.

## Entry Conditions

- Phase 2AB stalled at beta `0.3419540270406287`.
- Phase 2AC reached beta `1.0`, with terminal ESS ratio
  `0.9912539055044092` and max weight `0.010002188339361427`, but failed
  ancestor diversity at `0.21875 < 0.25`.
- Phase 2AD preserved ancestor diversity at `0.4140625` by skipping the
  final low-diversity resampling draw, but stalled at beta
  `0.9712250668187553`.
- No valid reference or HMC-reference agreement exists.
- Phase 3 GPU/XLA remains blocked.

## Required Artifacts

- Phase 2AE subplan review under `docs/reviews/`.
- Phase 2AE decision result:
  `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md`
- Refreshed master program, visible runbook, ledger, and stop handoff.

## Required Checks, Tests, And Reviews

- Review this subplan before writing the decision result.  Claude may be used
  as a read-only reviewer if safely available; otherwise record a local Codex
  substitute review as weaker than Claude.
- Before closeout, run `git diff --check`.
- No benchmark runtime is authorized by this subplan.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the current sequential-reference branch have enough reviewed evidence to justify another local repair, or should it be stopped in favor of a materially different reference-method design? |
| Baseline/comparator | Phase 2AB, Phase 2AC, and Phase 2AD artifacts. |
| Primary criterion | A decision result must separate current-branch blocker evidence from evidence against the target/HMC, name the next allowed branch if any, and preserve all nonclaims. |
| Veto diagnostics | Any claim of valid reference, posterior correctness, HMC readiness, GPU/XLA readiness, default readiness, source faithfulness, or sampler superiority; any plan to run another local policy tweak without a new reviewed mechanism. |
| Explanatory diagnostics | Phase 2AB beta stall, Phase 2AC beta-one/diversity failure, Phase 2AD diversity-preserving/beta stall, and runtime/check provenance. |
| What will not be concluded | No valid reference, no HMC-vs-reference agreement, no posterior correctness, no HMC readiness/convergence, no zero-divergence claim, no sampler superiority, no statistical ranking, no GPU/XLA readiness, no default readiness, and no Zhao-Cui source faithfulness. |
| Artifact preserving result | Phase 2AE result plus refreshed ledger/handoff. |

## Forbidden Claims And Actions

- Do not run HMC, GPU/XLA, or another benchmark runtime.
- Do not change defaults, public API behavior, package behavior, or model
  files.
- Do not claim a valid reference, posterior correctness, HMC readiness,
  convergence, zero divergences, sampler superiority, statistical ranking,
  GPU/XLA readiness, default readiness, or Zhao-Cui source faithfulness.
- Do not treat the current sequential-branch blocker as evidence against the
  scalar target or HMC mechanics.

## Exact Next-Phase Handoff Conditions

If the decision selects expansion, draft a new master-program branch or
subplan for a materially different reference method, such as stronger
move/rejuvenation design, deterministic transport, or a different
reference-family construction.  That new branch must have its own reviewed
evidence contract before runtime.

If the decision selects blocker closeout, update the runbook and stop with
Phase 3 GPU/XLA blocked.

## Stop Conditions

Stop for any unsupported scientific/default/GPU/HMC/source-faithfulness claim,
any missing Phase 2AB/2AC/2AD artifact, or any attempt to run another
benchmark without a reviewed runtime subplan.

## Skeptical Plan Audit

| Risk | Audit finding |
| --- | --- |
| Wrong baseline | Baseline is the sequential branch evidence across Phase 2AB/2AC/2AD, not HMC success. |
| Proxy metrics promoted | The decision can diagnose branch blocker mechanisms only; it cannot certify posterior correctness or HMC readiness. |
| Missing stop conditions | Runtime and claim boundaries are explicit. |
| Unfair comparison | No method ranking occurs. |
| Hidden assumptions | The phase assumes local fallback-resampling repairs are exhausted by reviewed Phase 2AD; a materially different design remains possible. |
| Stale context | Decision result must cite current Phase 2AB/2AC/2AD artifacts. |
| Environment mismatch | CPU-hidden artifacts cannot support GPU/XLA/default readiness. |
| Artifact mismatch | Decision/result and refreshed ledgers are predeclared. |

Audit status: `PASSED_FOCUSED_LOCAL_REVIEW_FOR_DECISION_RESULT_ONLY`.
