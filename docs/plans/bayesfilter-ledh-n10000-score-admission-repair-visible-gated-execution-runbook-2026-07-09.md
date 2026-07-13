# LEDH N10000 Score Admission Repair Visible Gated Execution Runbook

Date: 2026-07-09

Status: `DRAFT_VISIBLE_EXECUTION_RUNBOOK_RECREATED_AFTER_REVIEW_VISIBILITY_REPAIR`

## Role Contract

Codex in the current conversation is supervisor and executor. Claude, when
available, is a read-only reviewer only. This runbook is visible and must not
launch detached or nested agents.

## Quiet Execution Pattern

Long TensorFlow/CUDA/benchmark/review commands must write full output to logs
or structured artifacts; chat gets bounded summaries only.

## Program Artifacts

- Master program: `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-master-program-2026-07-09.md`
- Ledger: `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-visible-execution-ledger-2026-07-09.md`
- Stop handoff: `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-visible-stop-handoff-2026-07-09.md`

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can all main LEDH rows obtain validator-admitted compact `N=10000` score artifacts without changing the value target scalar? |
| Target scalar | `observed_data_log_likelihood_estimator`, reported as `log_likelihood`. |
| Baseline/comparator | July 8 Phase 8 blocked integration artifact, admitted value artifacts, shared score contract, tiny compact diagnostics, and historical routes as diagnostics only. |
| Primary pass criterion | Every main LEDH score row is admitted by `validate_ledh_score_artifact(..., require_admitted=True)` or has a precise blocker result with the next smallest fix. |
| Veto diagnostics | Wrong scalar; value/score artifact mismatch; historical route admitted; raw legacy JSON promoted; tape/autodiff; stopped partial derivative; nonfinite score; missing memory pass; correctness failure; diagnostic row promotion; KSC exact-native actual-SV overclaim. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, runtime ranking, public benchmark readiness, or non-LEDH all-algorithm completion. |

## Phase State Machine

For each phase: precheck, execute minimal commands, assess gate, write result,
review result and next subplan, repair if needed, then advance or stop.

## Human-Required Stop Conditions

Stop if continuing requires changing row targets, changing pass/fail criteria
after seeing results, package installation, network/data fetches, credentials,
destructive actions, unrelated dirty-worktree edits, or interpreting GPU
results without trusted-context evidence.
