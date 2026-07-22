# Complete Leaderboard Phase 1 Substitute Review, Iteration 2

Date: 2026-07-11

Reviewer type: `fresh_codex_readonly_substitute`

Reviewed path SHA-256:
`cd226be20e4855e545192102d987bec63ba08cef21669eff16f064bdbcde4f40`

## Findings

`VERDICT: REVISE`

- One exact-command hash was incorrectly required to be seed-invariant.
- The owner-corrected per-direction FD formula and FD-only
  `0.05 * sqrt(p)` threshold were not explicit.

The subplan was visibly repaired before iteration 3.

