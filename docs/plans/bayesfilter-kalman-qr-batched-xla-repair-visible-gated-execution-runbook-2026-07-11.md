# Kalman QR Batched XLA Repair Visible Gated Overnight-Style Execution Runbook

Date: 2026-07-11
Status: `SUPERSEDED_2026_07_13_HISTORICAL_RUNBOOK`

Supersession note, 2026-07-13: this runbook is retained for provenance only.
Its per-phase gate, subplan, snapshot, and Claude/Codex convergence machinery is
not active. Current work follows the academic risk-tier workflow in `AGENTS.md`
and the single live plan
`docs/plans/bayesfilter-kalman-qr-batched-xla-lean-repair-plan-2026-07-13.md`.
Scientific evidence, resource stops, GPU trust, and claim boundaries remain
binding.

## Role Contract

Codex in the current conversation is supervisor and executor. Claude Opus at
max effort is read-only reviewer only. Claude cannot edit, execute phases,
launch agents, or authorize human/runtime/model-file/funding/product/default/
GPU-trust/scientific-claim boundaries.

This is visible, recoverable execution in the current conversation. It must not
use `codex exec`, detached Claude/Codex supervisors, `setsid`, `nohup`, detached
`tmux`, background phase runners, copied-workspace execution, or
`overnight_gated_launch.sh`.

## Quiet Visible Execution

- Predeclare every log and structured artifact in the active subplan/ledger.
- Redirect verbose TensorFlow/XLA/Claude output to logs; return bounded status.
- Poll bounded status rather than stream long output.
- Preserve exit code, failure-stage metadata, and at most 40 failure-tail lines.
- A timeout is a stage result, not permission to silently skip the case.

## Program

- Master: `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-master-program-2026-07-11.md`
- Ledger: `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-visible-execution-ledger-2026-07-11.md`
- Stop handoff: `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-visible-stop-handoff-2026-07-11.md`
- Reset baseline: `docs/plans/bayesfilter-kalman-qr-batched-xla-reset-memo-2026-07-10.md`

## Phase Index

The binding phase paths and required result paths are the Phase Index in the
master program. No phase executes without its dedicated subplan, current review,
and inherited entry conditions.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can equivalent true-batched analytical/autodiff score paths produce valid method-isolated CPU/GPU XLA comparison evidence? |
| Baseline | Current dirty source and historical failed grid, with scalar rows only as small-batch correctness references. |
| Primary criterion | Every reached phase passes local checks, hard vetoes, artifact requirements, and review gate. |
| Vetoes | Invalid/corrupt/stale artifact, compile failure beyond declared repair, non-finite/parity failure, unfair comparator, missing provenance, or boundary violation. |
| Explanatory only | Graph/HLO/compile/runtime/memory measurements until declared analysis supports ranking. |
| Not concluded | Universal superiority, HMC/posterior/scientific/default/production readiness. |

## Default And Assumption Audit

The master's table is binding. Each phase result must record any changed
prospective number and its provenance before use. Phase 8 placeholders are a
hard stop until refreshed after Phase 7.

## Skeptical Pre-Execution Audit

Before each phase, Codex records in the ledger:

- correct baseline/comparator;
- no proxy promotion;
- stop conditions and repair trigger remain active;
- comparison fairness;
- numeric/default provenance;
- current source/environment identity;
- command artifacts answer the phase question;
- expected failure is distinguished from harness invalidity.

## Visible State Machine

1. `PRECHECK`: read current subplan/result predecessor; verify entry conditions,
   source identity, authority, artifacts, and evidence contract.
2. `EXECUTE_MINIMAL`: perform only the smallest edits/checks/run needed.
3. `ASSESS_GATE`: classify hard vetoes, candidate failures, explanatory metrics,
   and what remains unproved.
4. `WRITE_RESULT`: write phase result before advancing.
5. `REFRESH_NEXT`: update the next subplan with actual inherited facts and exact
   commands/artifacts.
6. `REVIEW_NEXT`: review the next exact subplan for consistency, correctness,
   feasibility, artifact coverage, and boundary safety.
7. `REPAIR_LOOP`: patch same artifact visibly, rerun focused checks, and rereview.
8. `ADVANCE_OR_STOP`: proceed only after gate pass; otherwise write blocker/handoff.

## Claude Review Gate

Material reviews use:

```bash
bash /home/ubuntu/python/claudecodex/scripts/claude_review_gate.sh \
  --cwd /home/ubuntu/python/BayesFilter \
  --review-name <stable-review-name> \
  --bundle /home/ubuntu/python/BayesFilter/docs/reviews/<bounded-bundle>.md \
  --model opus \
  --effort max \
  --probe-effort low \
  --probe-timeout 90 \
  --timeout-seconds 180 \
  --max-retries 1
```

Do not enable bounded fallback as an automatic phase pass. If primary review
fails after a successful probe, redesign to the smallest exact path and retry.
If the probe establishes Claude transport is unavailable, use a fresh Codex
read-only reviewer and record weaker substitute status. Never represent silence,
timeout, or fallback agreement as full review.

Maximum five rounds for the same material blocker. Round count and review status
must be appended to the ledger.

Current review-boundary status: the user explicitly approved bounded repository
disclosure to Claude, but the trusted execution layer rejected the gate again
before its liveness probe. No content was sent. Claude review is policy-blocked,
not dead. Fresh bounded Codex substitute review is active as the only materially
safer route and is recorded as weaker than Claude review.

## Repair Loop Protocol

```text
PRECHECK
  -> smallest implementation/check
  -> local checks
  -> phase result
  -> refresh next subplan
  -> bounded read-only review
       AGREE  -> advance
       REVISE -> patch same artifact -> focused checks -> rereview (max 5)
       no response + probe OK -> shrink/redesign prompt -> rereview
       probe/transport down -> fresh Codex read-only substitute review
  -> stop only on declared continuation veto or human-required boundary
```

Candidate rejection does not end the program when a later declared phase repairs
that exact failure. The supervisor must state whether a result invalidated the
harness, implementation, target, environment, math, or only the current arm.

## Human-Required Stop Conditions

- Package/network/credential/model-file/default-policy/product/release/funding
  action outside existing authority.
- Destructive git/filesystem operation or overwrite of unrelated dirty work.
- Changing pass criteria after results.
- Untrusted GPU interpretation.
- Review non-convergence after five rounds.
- Phase 8 still contains placeholders.

## Final Handoff

Record final phase, status, artifacts, Claude/Codex review trail, tests/runs,
unresolved blockers, nonclaims, and safest next human decision.
