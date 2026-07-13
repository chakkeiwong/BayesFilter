# Kalman QR Dtype And Batched Score Visible Gated Overnight-Style Execution Runbook

Date: 2026-07-09

## Status

`PHASE_2_READY_VISIBLE_EXECUTION`

## Role Contract

Codex in the current conversation is the supervisor and executor.

Claude is a read-only reviewer only when separately approved for external
disclosure.  The Phase 0 Claude review-gate attempt was rejected by the
approval reviewer as external disclosure risk, so the active review path for
this run is a fresh read-only Codex substitute review with weaker review
status.  Claude must not edit files, run experiments, launch agents, approve
boundary crossings, or act as execution authority.

This is a visible, recoverable overnight-style runbook inside the current
conversation.  Based on the local template, this runbook does not launch
detached or nested agents and does not use `codex exec`,
`overnight_gated_launch.sh`, `setsid`, `nohup`, detached `tmux`, background
phase runners, or copied-workspace execution.  A true detached overnight run
would require a separate human-approved detached supervisor plan.

## Quiet Visible Execution Pattern

Commands that may produce large stdout/stderr, including TensorFlow, CUDA,
benchmarks, or Claude review commands, must predeclare log and artifact paths.
Full output goes to a log file.  Chat receives bounded summaries only: exit
status, artifact paths, pass/fail fields, and at most 20-40 log lines on
failure.

Recommended pattern:

```bash
mkdir -p docs/benchmarks/logs
timeout <seconds> <command> > docs/benchmarks/logs/<phase>.log 2>&1
```

Do not use quiet logging to hide failures.  Logs and structured artifacts must
be referenced from phase results.

## Program

Master program:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-master-program-2026-07-09.md`

Reviewed plan artifacts:

- `docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-governance-review-bundle-2026-07-09.md`
- `docs/reviews/bayesfilter-kalman-qr-dtype-batched-score-codex-substitute-review-2026-07-09.md`

Execution ledger:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-execution-ledger-2026-07-09.md`

Stop handoff:

- `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-visible-stop-handoff-2026-07-09.md`

## Phase Index

| Phase | Name | Subplan | Required result artifact |
| --- | --- | --- | --- |
| 0 | Contract and dtype inventory | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase0-contract-inventory-result-2026-07-09.md` |
| 1 | Dtype infrastructure | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase1-dtype-infrastructure-result-2026-07-09.md` |
| 2 | QR value dtype cleanup | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase2-qr-value-dtype-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase2-qr-value-dtype-result-2026-07-09.md` |
| 3 | Analytical score dtype cleanup | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase3-analytical-score-dtype-result-2026-07-09.md` |
| 4 | Benchmark dtype controls | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase4-benchmark-dtype-controls-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase4-benchmark-dtype-controls-result-2026-07-09.md` |
| 5 | Batched analytical score contract | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase5-batched-score-contract-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase5-batched-score-contract-result-2026-07-09.md` |
| 6 | Batch-native analytical score implementation | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase6-batched-score-implementation-result-2026-07-09.md` |
| 7 | Correctness and benchmark ladder | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase7-correctness-benchmark-result-2026-07-09.md` |
| 8 | Closeout | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase8-closeout-subplan-2026-07-09.md` | `docs/plans/bayesfilter-kalman-qr-dtype-batched-score-phase8-closeout-result-2026-07-09.md` |

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the QR Kalman lane honor requested dtype and then support true batch-native analytical score computation? |
| Baseline/comparator | Existing FP64-only QR value/score implementation, scalar analytical score, batched-static value path, and 2026-07-09 FP64 benchmark artifacts. |
| Primary pass criterion | Every reached phase passes its subplan gate, local checks, documented Codex substitute review, or separately approved Claude review gate, and result artifact requirements. |
| Veto diagnostics | Hidden FP64 coercion after cleanup, observed dtype mismatch, failed parity, nonfinite outputs, missing artifacts, unapproved GPU/runtime action, or unsupported claim. |
| Explanatory diagnostics | Timing, compile+first-call, warm-call summaries, TF32 flag, device placement, dtype inventory count, and tolerance deltas. |
| Not concluded | Statistical speed ranking, HMC readiness, posterior correctness, production/default readiness, source faithfulness, or broad scientific validity. |
| Artifacts | Master program, subplans/results, ledger, stop handoff, review bundles, Codex substitute review notes, `.claude_reviews/` logs only if separately approved and actually run, and benchmark JSON/Markdown/logs. |

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Start with Phase 0 inventory | User directive and hard-coded dtype discovery | Prevents source edits without current facts | Inventory misses key helper path | Bounded `rg` over QR files and benchmark harness | launch gate |
| CPU-hidden local checks first | Repo GPU policy permits CPU debug/reference | Avoids premature GPU boundary | CPU pass treated as GPU evidence | Result nonclaims and `CUDA_VISIBLE_DEVICES=-1` where runtime occurs | reviewed |
| Claude Opus max effort review | User directive, then approval-reviewer rejection | Material plan review would benefit from independent read-only critique, but external disclosure was rejected | Attempting to route around the rejection or pretending Claude reviewed the plan | Record rejection and use fresh Codex substitute review unless user separately re-approves disclosure risk | unavailable in current run |
| Fresh Codex substitute review | User fallback directive after Claude unavailability | Safer internal review path when Claude cannot be used | Substitute review overstated as equivalent to Claude review | Label as weaker review status in result artifacts | active fallback |
| Later GPU/XLA checks | BayesFilter default target is GPU/XLA | Needed for benchmark evidence | Untrusted or unrecorded GPU evidence | Phase 7 managed-session GPU trust manifest | future approval |

## Skeptical Plan Audit

Before each phase, Codex must audit:

- wrong baselines;
- proxy metrics being treated as promotion criteria;
- missing stop conditions;
- unfair comparisons;
- hidden assumptions;
- stale context;
- environment mismatch;
- commands whose artifacts would not answer the phase question.

If a material flaw is found, revise the subplan or write a blocker before
running that phase.

Initial audit status: `PASSED_FOR_PHASE_0_DOCUMENT_AND_INVENTORY_GATE_ONLY`.

## Visible State Machine

For each phase:

1. `PRECHECK`: read the subplan, confirm prerequisites, restate evidence
   contract, and append a ledger entry.
2. `EXECUTE_MINIMAL`: run only visible commands needed to answer the phase.
3. `ASSESS_GATE`: compare outputs against the primary criterion and vetoes.
4. `PASS_REVIEW`: perform fresh read-only Codex substitute review, or send
   material subplans/results to Claude read-only review only if separately
   approved for external disclosure.
5. `REPAIR_LOOP`: patch fixable issues, rerun focused checks, and stop after
   five Claude review rounds for the same blocker.
6. `ADVANCE_OR_STOP`: write phase result, refresh next subplan, review it, and
   advance only if handoff conditions pass.

## Review Protocol

### Active Codex Substitute Review

Because the Phase 0 Claude review-gate attempt was rejected as external
disclosure risk, the active review route is:

1. Spawn or request a fresh bounded Codex read-only review.
2. Limit the reviewer to exact plan/result paths.
3. Ask for findings first and a final `VERDICT: AGREE` or `VERDICT: REVISE`.
4. Patch fixable findings visibly.
5. Rerun focused local checks.
6. Record that the review is weaker than Claude review.
7. Stop after five review rounds for the same material blocker.

### Claude Review Only If Separately Approved

Use the review gate:

```bash
bash /home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh \
  --cwd /home/ubuntu/python/BayesFilter \
  --review-name <review-name> \
  --bundle /home/ubuntu/python/BayesFilter/docs/reviews/<bundle>.md \
  --model opus \
  --effort max \
  --probe-timeout 90 \
  --timeout-seconds 180 \
  --max-retries 1 \
  --allow-bounded-fallback
```

If probe succeeds but the material review times out or returns no verdict,
Claude is alive and the bundle/prompt must be reduced to a smaller exact path
before retry.  If the approval reviewer rejects the command for external
disclosure risk, do not retry or work around the rejection; use the active Codex
substitute review route.  If probe fails or transport is down after retry,
replace the review with a fresh Codex substitute review and record weaker
review status.

## Human-Required Stop Conditions

Stop if continuing requires package installation, network fetch or external
disclosure not separately approved, credentials, model-file edits, destructive
git actions, default-policy changes, public/product claims, unapproved GPU
runtime, or continuing after Codex/Codex-substitute or approved Claude review
does not converge after five review rounds for the same blocker.
