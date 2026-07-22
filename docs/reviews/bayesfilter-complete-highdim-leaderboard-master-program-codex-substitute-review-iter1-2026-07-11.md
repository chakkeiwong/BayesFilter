# Complete Leaderboard Master Program Substitute Review, Iteration 1

Date: 2026-07-11

Review type: `fresh_codex_readonly_substitute`

Claude availability:
`docs/reviews/bayesfilter-complete-highdim-leaderboard-claude-availability-2026-07-11.md`

Reviewed exactly:
`docs/plans/bayesfilter-complete-highdim-leaderboard-master-program-2026-07-11.md`

## Verdict

`VERDICT: REVISE`

## Material Findings

1. Nine legacy non-LEDH cells lacked an explicit per-cell re-admission gate.
2. Canonical targets did not bind observation/data identity or all time and
   parameter conventions.
3. Five-seed value/score aggregation and the derivative relationship were not
   specified.
4. The FD-only rule did not say whether every seed and direction must pass or
   whether aggregation could hide failure.
5. The Zhao-Cui display id did not bind row-specific source routes and anchors.
6. New non-LEDH TensorFlow work had no explicit GPU/XLA/default-policy rule or
   reviewed exception.
7. July 7 LEDH artifacts were described too strongly.
8. `--require-complete` validation was underspecified.
9. The eight-hour plan lacked allocation, checkpoints, and timeout handoff.
10. The release phase alias was unclear.

## Repair

The master now defines candidate re-admission, canonical target signatures,
total-scalar paired aggregation, exact per-seed FD policy, row-specific
Zhao-Cui anchor ledgers, execution policy, strict final validation, and a
checkpointed eight-hour budget. During the repair Codex also found and fixed an
unreviewed scalar-label ambiguity: the admitted value is total log likelihood;
per-time average is display-only.

