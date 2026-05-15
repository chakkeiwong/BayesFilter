# Supervisor audit: reviewer-grade DPF master program and subplans

## Date

2026-05-15

## Scope

This audit covers the reviewer-grade DPF monograph planning lane only:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-phase-proposal-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p3-smc-baseline-expansion-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p4-edh-ledh-derivation-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p5-pfpf-jacobian-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p6-resampling-ot-sinkhorn-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p7-learned-ot-defensibility-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p8-hmc-banking-target-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p9-debugging-verification-contract-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p11-derivation-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p12-hostile-reader-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p13-final-readiness-plan-2026-05-14.md`

Out-of-lane files include student-baseline planning artifacts and are not part
of this audit.

## Claude worker review

Claude was launched as a bounded read-only worker through:

```text
bash /home/ubuntu/python/claudecodex/scripts/claude_worker.sh --cwd /home/ubuntu/python/BayesFilter --name dpf-plan-critical-review --model sonnet --effort high "<critical review prompt>"
```

Claude returned **REJECT**.

Blocking findings:

1. Master state was stale relative to P0's live branch/worktree state.
2. Result artifact names were date-inconsistent across plans and existing
   results.
3. P2 showed ResearchAssistant had no local DPF paper summaries, but later
   phases did not enforce the bibliography-spine fallback.
4. MathDevMCP and ResearchAssistant obligations were not hard prerequisites in
   P4-P11.
5. Dirty-worktree lane separation was documented but not operationalized in
   each editing phase.
6. Build/PDF gates were too late for table-heavy phases.
7. The master R0-R12 taxonomy conflicted with executable P0-P13 subplans.

## Codex audit

Codex independently inspected the master, phase proposal, P0-P13 plans, and
completed P0-P2 results.  Codex agreed with Claude's rejection.  Additional
Codex interpretation:

- P3 must be the next gating chapter phase because P1 identified baseline
  filtering targets and estimator-status definitions as prerequisites for later
  DPF target-status claims.
- The R0-R12 text can remain only as historical design context if the master
  explicitly states that P0-P13 governs execution.
- The result-date problem is best solved with an actual-completion-date naming
  rule rather than renaming existing artifacts.

## Repairs applied

The planning set was tightened as follows.

1. Master program:
   - added live-state delegation to the P0 result;
   - recorded P2's ResearchAssistant-empty source constraint;
   - added a required artifact naming rule;
   - made P0-P13 the governing taxonomy;
   - demoted R0-R12 to historical context;
   - added common lane-safety, source-grounding, tool-evidence, build/layout,
     and result-artifact contracts;
   - made P3 the next gating phase.
2. Phase proposal:
   - added an execution-safety amendment;
   - updated completed P0-P2 result names to the actual `2026-05-15` artifacts;
   - replaced fixed future result dates with `{YYYY-MM-DD}` patterns;
   - updated execution order to P0, P1, P2, then P3-P13.
3. P0-P2 plans:
   - updated expected result paths to the actual completed P0-P2 artifacts.
4. P3-P13 plans:
   - added prerequisites and lane guards;
   - added explicit ResearchAssistant-empty/bibliography-spine constraints;
   - added required write-set recording and student-baseline exclusion;
   - added derivation-obligation evidence requirements where applicable;
   - added build and table-readability gates where the phase risk requires
     them;
   - changed future result artifacts to actual-date naming patterns.

## Out-of-order in-lane artifacts

During supervisor repair, P4 and P5 in-lane results were found in the worktree:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p4-edh-ledh-derivation-result-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p5-pfpf-jacobian-audit-result-2026-05-15.md`

The corresponding chapter files also contain in-lane edits.  These artifacts
are not reverted.  Because the tightened master now makes P3 the next gate, P4
and P5 are treated as provisional until P3 completes and records whether its
baseline definitions affect those chapters.  P6 is blocked until that
reconciliation passes.

`docs/chapters/ch32_diff_resampling_neural_ot.tex` also contains an in-lane
partial P6 diff without a P6 result artifact.  P6 must begin by auditing that
diff and recording whether it is adopted, repaired, or superseded.  The partial
diff is not by itself evidence that P6 has passed.

## Remaining acceptance test

The repaired plan set should be sent to Claude for a second read-only critical
review.  Acceptance requires:

- Claude returns ACCEPT, or any REJECT items are non-blocking after Codex
  inspection;
- Codex verifies no active R-phase execution path remains;
- Codex verifies no fixed future result dates remain except historical plan
  creation dates and completed P0-P2 results;
- Codex verifies all P3-P13 execution plans contain lane, source, tool, and
  build/layout gates appropriate to their scope;
- Codex verifies the provisional P4/P5 state has a reconciliation rule before
  P6;
- Codex verifies the existing `ch32` partial diff has a P6 disposition rule.

## Second Claude worker review

Claude was launched again as a bounded read-only worker through:

```text
bash /home/ubuntu/python/claudecodex/scripts/claude_worker.sh --cwd /home/ubuntu/python/BayesFilter --name dpf-plan-critical-review-2 --model sonnet --effort high "<second critical review prompt>"
```

Claude returned **ACCEPT** for execution readiness.

Claude's rationale:

- P0-P13 is now the only governing taxonomy.
- R0-R12 is explicitly historical and non-executable.
- P0-P2 actual result dates are consistently recorded.
- P3 is the next gating phase.
- P4/P5 provisional results are preserved but blocked behind P3 impact
  reconciliation before P6.
- The existing `ch32` partial P6 diff has a required P6 disposition rule.
- ResearchAssistant-empty fallback, MathDevMCP/manual obligation evidence,
  per-phase lane guards, build checks, and table-readability gates are explicit.

Claude noted one non-blocking stale line in the P2 result that still pointed to
P4 as next phase.  That line was repaired to state the superseded recommendation
and the current P3/P4/P5/P6 gate.

## Current Codex verdict

Codex verdict: **ACCEPT**.

The set is now execution-safe enough for the next worker to begin P3.  P6 must
not begin until P3 passes, P4/P5 reconciliation passes, and the existing `ch32`
partial diff receives an explicit P6 disposition.
