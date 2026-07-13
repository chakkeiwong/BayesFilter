# LEDH Compact Score Default Visible Gated Execution Runbook

Date: 2026-07-08

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

- `docs/plans/bayesfilter-ledh-compact-score-default-master-program-2026-07-08.md`

Execution ledger:

- `docs/plans/bayesfilter-ledh-compact-score-default-visible-execution-ledger-2026-07-08.md`

Stop handoff:

- `docs/plans/bayesfilter-ledh-compact-score-default-visible-stop-handoff-2026-07-08.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Route demotion and policy gate | `docs/plans/bayesfilter-ledh-compact-score-default-phase0-route-demotion-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase0-route-demotion-result-2026-07-08.md` |
| 1 | Shared compact score contract | `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase1-contract-result-2026-07-08.md` |
| 2 | LGSSM reference freeze | `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase2-lgssm-reference-result-2026-07-08.md` |
| 3 | Actual-SV compact port | `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase3-actual-sv-result-2026-07-08.md` |
| 4 | Fixed-SIR compact port | `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase4-fixed-sir-result-2026-07-08.md` |
| 5 | Predator-prey compact port | `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase5-predator-prey-result-2026-07-08.md` |
| 6 | Generalized-SV compact port | `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase6-generalized-sv-result-2026-07-08.md` |
| 7 | KSC-SV compact port | `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-result-2026-07-08.md` |
| 8 | Leaderboard integration | `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-subplan-2026-07-08.md` | `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-result-2026-07-08.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can every LEDH leaderboard score row be made compact-forward-sensitivity by default, with reverse/manual-total-VJP routes demoted to historical and blocked from admission? |
| Target scalar | `observed_data_log_likelihood_estimator`, reported as `log_likelihood`. |
| Baseline/comparator | LGSSM compact score route, admitted value artifacts, historical reverse/manual-total-VJP routes as diagnostics only, same-scalar finite differences or exact references. |
| Primary pass criterion | Every model row is compact-admitted or explicitly blocked; no old route is default or admitted; validators and leaderboard integration reject old route labels. Compact admission requires predeclared same-scalar FD or exact-reference agreement at the tested point/coordinates, not a broad mathematical correctness claim. |
| Veto diagnostics | Wrong scalar; old route admitted; `manual_total_vjp_no_autodiff_same_scalar_*` admitted; any `--admit-full` run before Phase 1 code guards land; all-time reverse records used as score default; tape/autodiff; stopped partial derivative; nonfinite score; tested-coordinate FD/exact-reference failure; memory failure; diagnostic-row promotion; unsupported scientific claim. |
| Explanatory diagnostics | Runtime, memory, FD error, exact-reference error, per-seed dispersion, route parity against historical diagnostics. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, runtime ranking, public benchmark readiness, or all-algorithm readiness. |
| Artifacts | Master program, runbook, ledger, stop handoff, phase subplans/results, review bundles, compact score artifacts, tests, logs, and integration result. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Compact forward sensitivity is the only admissible default | User directive and LGSSM memory repair | It avoids all-time reverse record retention and uses streaming transport value+JVP | A model keeps old VJP route but relabels it compact | Phase 0 inventory and Phase 1 static guards | baseline |
| Reverse/manual-total-VJP routes are historical/wrong for leaderboard admission | User directive after actual-SV wiring audit | They caused scaling/memory failures and differ from LGSSM repaired style | Diagnostic route gets promoted because tiny FD passes | Validator route allowlist update | baseline |
| One model per phase | Prior runbook experience | Each row has distinct target and parameter tangent equations | Broad implementation hides scalar or theta mismatch | Per-model subplan and artifact gate | baseline |
| Claude is read-only reviewer | User approvals and review guide | External review improves boundary audit without transferring authority | Claude timeout or broad prompt blocks progress | Probe ladder and Codex fallback review | baseline |

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
- labels reverse/manual-total-VJP score admission, including
  `manual_total_vjp_no_autodiff_same_scalar_*`, as `wrong relative to the
  leaderboard score target`;
- labels unsupported claims as `unsupported` or `not checked`;
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
fresh Codex read-only review and record that limitation.

## Human-Required Stop Conditions

Stop if continuing would require:

- changing row target definitions;
- changing pass/fail criteria after seeing results;
- package installation, network fetches, or credentials;
- destructive git or filesystem action;
- modifying unrelated dirty user work;
- interpreting GPU results without trusted-context evidence;
- continuing after Claude/Codex review does not converge after five rounds.

## Launch Rule

After the master program, runbook, ledger, stop handoff, Phase 0 subplan, and
launch review bundle are written and reviewed, launch Phase 0 only. Continue to
later phases only through the gated handoff conditions.
