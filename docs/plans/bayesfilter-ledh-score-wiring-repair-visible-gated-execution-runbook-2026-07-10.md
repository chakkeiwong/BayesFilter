# LEDH Score Wiring Repair Visible Gated Execution Runbook

Date: 2026-07-10

## Status

`ACTIVE_VISIBLE_EXECUTION_RUNBOOK`

## Role Contract

Codex in the current conversation is the supervisor and executor.

Claude is a read-only reviewer only.

This runbook must not launch a detached or nested agent. Do not use:

- `codex exec`;
- `overnight_gated_launch.sh`;
- `setsid`, `nohup`, or detached `tmux` supervisors;
- backgrounded phase runners;
- copied-workspace execution.

The program is overnight-capable by artifact structure, but execution remains
visible and recoverable inside the current conversation unless the user later
requests a separate detached-supervisor plan.

## Quiet Visible Execution Pattern

Long TensorFlow/GPU/XLA/Claude commands must write full stdout/stderr to logs
under `docs/plans/logs/` or `.claude_reviews/`. Chat updates include only
bounded summaries: exit status, artifact paths, pass/fail fields, and short log
tails on failure.

## Program

Master program:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-master-program-2026-07-10.md`

Reviewed plan artifacts:

- `docs/reviews/bayesfilter-ledh-score-wiring-repair-launch-review-bundle-2026-07-10.md`

Execution ledger:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-execution-ledger-2026-07-10.md`

Stop handoff:

- `docs/plans/bayesfilter-ledh-score-wiring-repair-visible-stop-handoff-2026-07-10.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Launch inventory and governance freeze | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase0-launch-inventory-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase0-launch-inventory-result-2026-07-10.md` |
| 1 | Shared score contract and precision gate | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase1-shared-contract-result-2026-07-10.md` |
| 2 | LGSSM compact default cleanup | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase2-lgssm-result-2026-07-10.md` |
| 3 | fixed-SIR compact default | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase3-fixed-sir-result-2026-07-10.md` |
| 4 | Predator-prey compact default | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase4-predator-prey-result-2026-07-10.md` |
| 5 | Actual-SV compact default | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase5-actual-sv-result-2026-07-10.md` |
| 6 | Generalized-SV compact precision gate | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase6-generalized-sv-result-2026-07-10.md` |
| 7 | KSC-SV compact precision gate | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase7-ksc-sv-result-2026-07-10.md` |
| 8 | Cross-model wiring and smoke tests | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase8-cross-model-tests-result-2026-07-10.md` |
| 9 | Trusted GPU score-memory ladder | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-result-2026-07-10.md` |
| 10 | Leaderboard rebuild and closeout | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase10-leaderboard-subplan-2026-07-10.md` | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase10-leaderboard-result-2026-07-10.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Are all LEDH same-target model score adapters wired so their default score computation uses the compact no-time-history score route, and can the leaderboard be rebuilt without admitted historical score routes? |
| Baseline/comparator | Current code inventory, the LGSSM compact `N=10000,T=50` score-only artifact, and existing per-model score contract tests. |
| Primary pass criterion | Every model default score path has a local test proving compact route use; historical routes are diagnostic-only and blocked from full admission; final leaderboard has no admitted historical score route. |
| Veto diagnostics | Default path calls memory-style reverse VJP; full admission uses historical route; production defaults to float64 or TF32 disabled; FD correctness calls score route; GPU claims from non-trusted runs. |
| Explanatory diagnostics | Runtime, score memory, compile time, exact/reference value comparisons, and smoke artifacts. |
| Not concluded | HMC readiness, posterior correctness, scientific superiority, exact nonlinear likelihood correctness, full actual-SV `N=10000,T=1000` score admission unless explicitly run and admitted. |
| Artifacts | Master program, subplans/results, ledger, review bundles, logs, JSON/Markdown benchmark artifacts, leaderboard output. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Compact score default | Owner directive and LGSSM repair evidence | Avoids full time-history reverse route and aligns with intended LEDH memory style | Relabeling old route as compact without changing calls | Source inspection plus monkeypatch/call-count tests | Baseline |
| `float32` plus TF32 enabled | Project AGENTS policy and owner directive | Production LEDH route is GPU TF32 | Existing CLIs default to float64/TF32 disabled | Precision default tests per runner | Baseline |
| Value-only FD comparator | Phase 2S/2W LGSSM repairs | Prevents FD correctness from recomputing score/JVP route | FD timing/memory hides score route failure | Monkeypatch score route forbidden during FD | Baseline |
| Claude read-only review | User instruction and local review guide | Adds skeptical review while Codex remains executor | Claude prompt too broad or unavailable | Review gate status and fallback record | Reviewed per phase |

## Skeptical Plan Audit

Before executing any phase, Codex must record a skeptical audit in chat and, for
material phases, in the execution ledger.

The standing audit checks:

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
   - Preserve unrelated dirty worktree changes.
3. `ASSESS_GATE`
   - Compare outputs against the primary criterion and veto diagnostics.
   - Write or update the required phase result artifact.
4. `PASS_REVIEW`
   - Send material phase results, repairs, implementation diffs, or final
     decisions to Claude as read-only review.
   - Continue only after `VERDICT: AGREE`, or revise and retry.
5. `REPAIR_LOOP`
   - For fixable blockers, patch the subplan or implementation visibly.
   - Rerun focused checks.
   - Stop after five Claude review rounds for the same blocker.
6. `ADVANCE_OR_STOP`
   - Advance only after current phase gate passes.
   - Stop and write handoff if a human-required blocker appears.

## Claude Read-Only Review Command

Use:

```bash
bash ~/python/claudecodex/scripts/claude_review_gate.sh \
  --cwd /home/chakwong/BayesFilter \
  --review-name <review-name> \
  --bundle /home/chakwong/BayesFilter/docs/reviews/<bundle>.md \
  --probe-timeout 90 \
  --timeout-seconds 120 \
  --max-retries 1 \
  --allow-bounded-fallback
```

Run with trusted/escalated permissions because Claude Code model/API calls may
need network, auth, process, and workspace access outside the normal sandbox.

## Human-Required Stop Conditions

Stop if continuing would require:

- a project-direction decision not already in the reviewed plan;
- package installation, network fetch, credentials, or environment setup;
- destructive git or filesystem action;
- changing pass/fail criteria after seeing results;
- changing default policy;
- modifying unrelated dirty user work;
- interpreting GPU results without trusted-context evidence;
- continuing after Claude and Codex do not converge after five review rounds.

## Final Visible Handoff

At completion or stop, write a result or stop handoff containing:

- completed phases;
- local checks;
- Claude review trail;
- artifacts;
- remaining blockers;
- explicit nonclaims.
