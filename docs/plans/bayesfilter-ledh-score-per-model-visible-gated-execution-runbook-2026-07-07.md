# LEDH Score Per-Model Visible Gated Execution Runbook

Date: 2026-07-07

## Status

`DRAFT_VISIBLE_EXECUTION_RUNBOOK`

## Role Contract

Codex in the current conversation is the supervisor and executor.

Claude is a read-only reviewer only.

This runbook must not launch a detached or nested agent. Do not use:

- `codex exec`;
- `overnight_gated_launch.sh`;
- `setsid`, `nohup`, or detached `tmux` supervisors;
- backgrounded phase runners;
- copied-workspace execution.

Execution is visible and staged. Launch Phase 0 only after the launch package
passes local checks and read-only review. Later phases may proceed only through
their handoff gates.

## Quiet Visible Execution Pattern

Full stdout/stderr for long TensorFlow, CUDA, benchmark, sampler, or Claude
commands is an artifact, not chat content.

Required pattern:

1. Predeclare log path and structured artifact path in the subplan or ledger.
2. Redirect full stdout/stderr to logs for long commands.
3. Prefer commands that write JSON/Markdown/result artifacts directly.
4. Print only bounded summaries: exit status, artifact paths, pass/fail fields,
   and at most the last 20-40 log lines on failure.
5. Treat excessive stdout/stderr as an execution-flow defect.

## Program

Master program:

- `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`

Upstream value result:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-result-2026-07-07.md`

Execution ledger:

- `docs/plans/bayesfilter-ledh-score-per-model-visible-execution-ledger-2026-07-07.md`

Stop handoff:

- `docs/plans/bayesfilter-ledh-score-per-model-visible-stop-handoff-2026-07-07.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Baseline and score governance | `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase0-baseline-governance-result-2026-07-07.md` |
| 1 | Score artifact schema and guards | `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase1-score-schema-result-2026-07-07.md` |
| 2 | LGSSM score | `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase2-lgssm-result-2026-07-07.md` |
| 3 | Fixed SIR score | `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase3-fixed-sir-result-2026-07-07.md` |
| 4 | Predator-prey score | `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase4-predator-prey-result-2026-07-07.md` |
| 5 | Actual-SV score | `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-result-2026-07-07.md` |
| 6 | Generalized-SV score | `docs/plans/bayesfilter-ledh-score-per-model-phase6-generalized-sv-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase6-generalized-sv-result-2026-07-07.md` |
| 7 | KSC-SV score | `docs/plans/bayesfilter-ledh-score-per-model-phase7-ksc-sv-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase7-ksc-sv-result-2026-07-07.md` |
| 8 | Value-score integration | `docs/plans/bayesfilter-ledh-score-per-model-phase8-integration-subplan-2026-07-07.md` | `docs/plans/bayesfilter-ledh-score-per-model-phase8-integration-result-2026-07-07.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can each admitted LEDH value row produce a no-tape total derivative of the same finite-`N` `log_likelihood` scalar? |
| Target scalar | `observed_data_log_likelihood_estimator`, reported as `log_likelihood`. |
| Baseline/comparator | Phase 8 value integration artifact, Phase 2-7 admitted value artifacts, exact derivatives where available, and same-scalar finite differences with fixed randomness otherwise. |
| Primary pass criterion | A score row is admitted only after same-target value admission, no-tape total derivative implementation, tiny correctness pass, `N=10000` correctness/memory pass, and replayable score artifact validation. |
| Veto diagnostics | Score before value; value/score row-set mismatch; diagnostic SIR promotion; tape/autodiff; stopped partial derivative; wrong scalar; wrong parameter vector; nonfinite score; FD/exact mismatch; memory failure; runtime-only promotion. |
| Explanatory diagnostics | Runtime, compile time, memory, FD error, exact-reference error, MCSE, per-seed dispersion, decomposition, and device placement. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, runtime ranking, and all-algorithm comparison. |
| Artifacts | Master program, runbook, ledger, stop handoff, subplans/results, review bundles, score artifacts, tests, logs, and final value-score artifact if gates pass. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Six-row score target set | Phase 8 value artifact | Score rows must match value rows | Diagnostic row or missing value row sneaks in | Phase 0 row-set freeze | baseline |
| No-tape total derivative only | User directive and score repair history | Tape gradients and partial derivatives caused prior score failures | Hidden autodiff or stopped terms called scores | Phase 0/1 static and runtime sentinels | baseline |
| One model per phase | Previous value runbook success | Model target/parameter dependencies differ | Broad phase hides target mismatch | Phase index separates rows | baseline |
| KSC distinct from exact actual-SV | Phase 7 value admission | KSC is a finite-mixture target row | KSC score overclaims exact SV | Phase 7 KSC flags and Phase 7 score subplan | baseline |
| Visible execution | User request and template | Recoverable current-conversation gating | Detached worker hides blockers | Runbook forbids detached agents | baseline |

## Skeptical Plan Audit

Before executing any phase, Codex must record a skeptical audit in chat and, for
material phases, in the execution ledger.

Check:

- wrong baselines;
- proxy metrics being treated as promotion criteria;
- missing stop conditions;
- unfair comparisons;
- hidden assumptions;
- stale context;
- environment mismatch;
- commands whose artifacts would not answer the phase question.

If the audit finds a material flaw, revise the plan or write a blocker note
before running the phase.

## Visible State Machine

For each phase:

1. `PRECHECK`
   - Read the phase subplan.
   - Confirm prerequisites.
   - Restate the phase evidence contract in chat.
   - Append a ledger entry.
2. `EXECUTE_MINIMAL`
   - Run only visible commands in the current conversation.
   - Prefer the smallest diagnostic or implementation needed to answer the
     phase question.
   - Preserve unrelated dirty worktree changes.
3. `ASSESS_GATE`
   - Compare outputs against the primary criterion and veto diagnostics.
   - Apply the plain-language gate before accepting the result artifact.
   - Write or update the required phase result artifact.
4. `PASS_REVIEW`
   - Send material phase results, repairs, implementation diffs, or final
     decisions to Claude as read-only review.
   - Continue only after `VERDICT: AGREE`, or revise and retry.
5. `REPAIR_LOOP`
   - For fixable blockers, write a blocker plan.
   - Get Claude review when material.
   - Apply the repair visibly.
   - Rerun focused checks.
   - Write a blocker result.
   - Stop after five Claude review rounds for the same blocker.
6. `ADVANCE_OR_STOP`
   - Advance only after the current phase gate passes.
   - Stop and write the handoff if a human-required blocker appears.

## Plain-Language Gate

Before accepting a phase result, blocker result, or final decision, Codex must
verify that the artifact:

- states the claimed target and computed quantity separately;
- uses direct classifications such as `correct`, `wrong relative to the stated
  target`, `unsupported`, `not checked`, or `heuristic only`;
- avoids unsupported soft language such as `surrogate`, `proxy`, `stabilized`,
  `contract`, `reasonable`, `practical`, or `approximately correct` unless the
  row target name itself contains the term and the target is explicitly
  declared;
- labels unsupported claims as `unsupported` or `not checked`;
- labels mismatches as `wrong relative to the stated target`;
- states what remains unproved or unevaluated.

## Claude Review Protocol

Use Claude only as read-only reviewer. Bounded review prompts must say:

```text
READ-ONLY REVIEW ONLY.

Do not edit files, run experiments, launch agents, or change state.

Review only the cited fixed paths or packet excerpts.

End with exactly:
VERDICT: AGREE
or
VERDICT: REVISE
```

If Claude does not respond, run a tiny probe. If the probe succeeds, narrow the
prompt. If Claude is unavailable or policy-blocked, replace the review with a
fresh Codex read-only review.

## Launch Rule

After the master program, runbook, ledger, stop handoff, Phase 0 subplan, and
launch review bundle are written and reviewed, launch Phase 0 only. Continue
to later phases only through the gated handoff conditions.
