# Complete High-Dimensional Leaderboard Visible Gated Execution Runbook

Date: 2026-07-11

Status: `DRAFT_VISIBLE_EXECUTION_RUNBOOK`

## Role Contract

Codex in the current conversation is supervisor and executor. Claude is a
read-only reviewer only.

This visible runbook does not launch nested or detached agents. It does not use
`codex exec`, `overnight_gated_launch.sh`, `setsid`, `nohup`, detached `tmux`,
or background phase runners. Detached execution is governed only by
`docs/plans/bayesfilter-complete-highdim-leaderboard-detached-overnight-supervisor-plan-2026-07-11.md`.

## Program Artifacts

- Master:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-master-program-2026-07-11.md`
- Ledger:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-visible-execution-ledger-2026-07-11.md`
- Stop handoff:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-visible-stop-handoff-2026-07-11.md`
- Command manifest:
  `docs/plans/complete-highdim-leaderboard-exact-command-manifest-2026-07-11.json`
- Run-scoped risk amendment:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-run-risk-acceptance-amendment-2026-07-12.md`
- Mandatory post-run audit:
  `docs/plans/bayesfilter-complete-highdim-leaderboard-post-run-integrity-audit-plan-2026-07-12.md`

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the master program advance phase-by-phase while preserving targets, evidence roles, and authority boundaries? |
| Baseline | Phase 0 freeze artifact and the preceding phase result. |
| Primary criterion | Every phase passes its local criterion, veto screen, material review, result record, and next-subplan review. |
| Vetoes | Master-program binding boundaries and phase-specific vetoes. |
| Explanatory only | Runtime, memory below budget, prefix results, smoke tests, and review fallback status. |
| Nonclaims | Ranking, superiority, HMC/posterior correctness, and claims not explicitly admitted by a phase. |

## Quiet Visible Execution

For long tests, GPU work, Claude review, and benchmark commands:

1. predeclare log and structured artifact paths;
2. redirect full stdout/stderr to the log without `tee`;
3. use a bounded timeout;
4. inspect only exit status, structured fields, and at most 40 failure lines;
5. poll bounded progress artifacts rather than streaming output;
6. preserve every log referenced by a result.

## State Machine

For each phase:

1. `PRECHECK`: read the subplan, verify entry artifacts and hashes, record the
   skeptical audit and evidence contract in the ledger.
2. `EXECUTE_MINIMAL`: run the smallest discriminating implementation or test.
3. `ASSESS_GATE`: apply primary criteria and vetoes, then write result or
   blocker result.
4. `PASS_REVIEW`: obtain the narrowest material read-only review.
5. `REPAIR_LOOP`: patch actionable issues, rerun focused checks, and retry up
   to five rounds for the same blocker.
6. `ADVANCE_OR_STOP`: refresh and review the next subplan or write the handoff.

A candidate failure is not a continuation veto unless it invalidates the
harness, target, data, source identity, artifact, or a binding assumption.

## Claude Review Recovery

Use the trusted `claude_review_gate.sh` path for material reviews, model
`opus`, effort `max`, and low-effort probes. Start with one exact artifact path
and one exact question.

On failure:

1. health probe for exactly `CLAUDE_PROBE_OK`;
2. retry the health probe once in trusted context;
3. if healthy, check cwd/path/permissions and narrow to packet-read, one-file,
   symbol, or line range;
4. do not call Claude dead because a broad prompt timed out;
5. if trusted health remains unavailable, use a fresh read-only Codex
   substitute review and label it weaker evidence;
6. do not treat `bounded_fallback_agree` as full material review;
7. stop after five nonconvergent rounds for one material blocker.

The ordinary five-round rule remains unchanged. The owner-authorized sixth
round applies only to the exact launch-readiness waiver/audit/command package
for `complete-highdim-leaderboard-20260711-221500`; it cannot be used for a
phase, source-faithfulness, scientific result, future run, or release review.

Zhao-Cui source-faithfulness and final release require a usable primary Claude
verdict or a fresh explicit Codex substitute review, not bounded fallback alone.

## Plain-Language Gate

Every result must state the claimed target, actually computed quantity,
relationship between them, evidence anchor, and remaining unknowns. Use direct
classifications: `correct`, `wrong relative to the stated target`,
`unsupported`, `not checked`, or `heuristic only`.

## Human-Required Stops

- target, row-scope, threshold, default, public API, package, network, funding,
  credentials, release, or scientific-claim decision not already authorized;
- destructive operation or overlapping dirty-work edit;
- unavailable trusted GPU evidence for a GPU claim;
- source-faithfulness gap or `extension_or_invention` needed for Zhao-Cui;
- five-round review nonconvergence.

For the exact waived run, stop if schema-v7 fails to preserve all five accepted
limitations as unresolved, if the post-run audit is not hash-bound, or if the
final exact command has not received the owner's fresh explicit approval.
After a run, also stop on a missing/failing external post-lock receipt, any
current credential-value match in handoff/archive bytes, malformed or
state-changing Claude tool evidence, missing/failing semantic inspection,
nonzero structural-helper exit or missing `PASS_STRUCTURAL_POST_RUN_INTEGRITY`,
or a nonzero/partially failing Phase 8/9 completeness validator.

## Final Handoff

Record final phase, status, results, review trail, commands actually run,
unresolved blockers, nonclaims, and safest next human decision. This visible
runbook never auto-merges detached exports. For the waived run, no completion
or release claim is permitted until the separate post-run integrity audit
records `PASS_POST_RUN_INTEGRITY_AUDIT`; that pass does not replace scientific
release gates.
