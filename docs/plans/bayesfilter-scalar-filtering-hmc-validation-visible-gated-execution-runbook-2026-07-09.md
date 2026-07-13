# Scalar Filtering HMC Validation Visible Gated Execution Runbook

Date: 2026-07-09

## Status

`PHASE_2AE_CURRENT_SEQUENTIAL_REFERENCE_BRANCH_BLOCKED_EXPANSION_REQUIRED`

## Role Contract

Codex in the current conversation is supervisor and executor.

Claude is a read-only reviewer only.  Claude cannot authorize human, runtime,
model-file, funding, product, release, public-benchmark, default-policy, or
scientific-claim boundaries.

This runbook is visible and recoverable in the current conversation.  The
template used for this runbook forbids detached launch, `codex exec`, detached
`tmux`, `nohup`, copied workspaces, and background phase runners.  A true
detached overnight run would require a separate detached-supervisor plan.

## Program

Master program:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-master-program-2026-07-09.md`

Reviewed plan artifacts:

- `docs/reviews/bayesfilter-scalar-filtering-hmc-validation-phase0-governance-review-bundle-2026-07-09.md`

Execution ledger:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-execution-ledger-2026-07-09.md`

Stop handoff:

- `docs/plans/bayesfilter-scalar-filtering-hmc-validation-visible-stop-handoff-2026-07-09.md`

## Quiet Visible Execution Pattern

Commands that may produce large output must write full stdout/stderr to a log
under `docs/benchmarks/logs/` or `.claude_reviews/`.  Chat updates should
summarize exit status, artifact paths, pass/fail fields, and only bounded log
tails on failure.

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Governance and telemetry policy audit | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase0-telemetry-policy-result-2026-07-09.md` |
| 1 | CPU-hidden short-chain validation screen | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase1-cpu-short-chain-result-2026-07-09.md` |
| 2 | Scalar reference agreement | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2-reference-agreement-result-2026-07-09.md` |
| 2S | Geometry centering repair | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2s-geometry-centering-repair-result-2026-07-09.md` |
| 2T | MAP-local reference handoff | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2t-map-local-reference-handoff-result-2026-07-09.md` |
| 2U | Retuned MAP-local HMC screen | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2u-retuned-map-local-hmc-screen-result-2026-07-09.md` |
| 2V | Longer selected MAP-local screen | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2v-longer-selected-map-local-screen-result-2026-07-09.md` |
| 2W | MAP-local importance reference agreement | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2w-importance-reference-agreement-result-2026-07-09.md` |
| 2X | Shifted-mixture reference repair | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2x-shifted-mixture-reference-repair-result-2026-07-09.md` |
| 2Y | Target geometry localization | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2y-target-geometry-localization-result-2026-07-09.md` |
| 2Z | Proposal strategy pilot | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2z-proposal-strategy-pilot-result-2026-07-09.md` |
| 2AA | Reference branch decision | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2aa-reference-branch-decision-result-2026-07-09.md` |
| 2AB | Transport or sequential reference pilot | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ab-transport-or-sequential-reference-result-2026-07-09.md` |
| 2AC | Sequential resampling repair | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ac-sequential-resampling-repair-result-2026-07-09.md` |
| 2AD | Diversity-preserving sequential repair | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ad-diversity-preserving-sequential-repair-result-2026-07-09.md` |
| 2AE | Reference-method expansion decision | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase2ae-reference-method-expansion-decision-result-2026-07-09.md` |
| 3 | Trusted GPU/XLA reproduction | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase3-gpu-xla-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase3-gpu-xla-result-2026-07-09.md` |
| 4 | Expansion decision | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase4-expansion-decision-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase4-expansion-decision-result-2026-07-09.md` |
| 5 | Closeout | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase5-closeout-subplan-2026-07-09.md` | `docs/plans/bayesfilter-scalar-filtering-hmc-validation-phase5-closeout-result-2026-07-09.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can scalar fixed-kernel HMC validation evidence be strengthened beyond mechanics smoke while preserving telemetry and claim boundaries? |
| Baseline/comparator | 2026-07-08 scalar filtering finite-telemetry closeout and artifacts. |
| Primary pass criterion | Every reached phase passes its predeclared gate, writes artifacts, and preserves nonclaims. |
| Veto diagnostics | Nonfinite runtime outputs, missing/invalid artifacts, unavailable divergence treated as zero, positive native divergence if available, changed criteria after results, or unsupported claim. |
| Explanatory diagnostics | Acceptance, log-accept summaries, target-log-prob summaries, sample summaries, R-hat/ESS when valid, runtime, and trace availability. |
| Not concluded | HMC readiness, convergence, broad posterior correctness, zero divergences without native telemetry, sampler superiority, default readiness, GPU/XLA production readiness, package readiness, or Zhao-Cui source faithfulness. |
| Artifacts | Master program, subplans/results, ledger, stop handoff, benchmark JSON/Markdown/logs, review bundles, ignored `.claude_reviews/` logs. |

## Visible State Machine

For each phase:

1. `PRECHECK`: read subplan, confirm prerequisites, restate evidence contract,
   append ledger entry.
2. `EXECUTE_MINIMAL`: run only visible commands needed for the phase.
3. `ASSESS_GATE`: compare outputs against primary criterion and vetoes.
4. `PASS_REVIEW`: use Claude read-only review for material plans/results when
   available; otherwise document fallback.
5. `REPAIR_LOOP`: patch fixable issues, rerun focused checks, and stop after
   five review rounds for the same blocker.
6. `ADVANCE_OR_STOP`: draft or refresh next subplan, review it, and continue
   only if handoff conditions pass.

## Skeptical Plan Audit

Before executing each phase, Codex must audit wrong baselines, proxy metrics,
missing stop conditions, unfair comparisons, hidden assumptions, stale context,
environment mismatch, and artifact mismatch.  If material flaws are found,
revise the plan or write a blocker before execution.

Initial audit status: `PASSED_FOR_PHASE_0_ONLY`.

## Human-Required Stop Conditions

Stop if continuing requires package installation, network fetch beyond Claude
review, credentials, model-file edits, destructive git actions, default-policy
changes, public/product claims, trusted GPU runtime without approval, or a
project-direction decision not already covered by this program.
