# SQMC campaign reset memo — updated 2026-09-24

## Current task and scope

User authorization: finish the campaign documentation; commit all SQMC branch
work; merge it into main; fetch and merge origin/main; resolve conflicts; push
main to origin/main; merge main back into the SQMC branch.

Worktree: `/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909`.
Branch: `rqmc-sqmc-4route-comparison`.
The main checkout's `surrogate-hmc` work belongs to another task. Preserve it.
Use a separate integration worktree for main.

## Checked campaign status

Program: non-production float64 eager transfer diagnostics. Controls from the
3D T=20 tuning campaign are UNTUNED for changed scopes. Completion does not
establish score accuracy or production eligibility.

The executed successful attempts contain 112 finite values: Phase 0 has 16,
Phase 2 has 32, and Phase 3 has 64. The earlier failed Phase 2 attempt contains
32 serialization failures and remains preserved. Phase 1 replication was not
executed. Phase 2/3 call with_score=False, while Phase 0 scores are all [0.0].
Therefore the score-transfer question is unresolved, and the earlier
production/no-retuning conclusions are unsupported. The active per-scope
tuning rule remains unchanged.

Authoritative closeout:
[summary](../benchmarks/sqmc-campaign-final-summary-20260924.md) and
[inspected artifact metadata](../benchmarks/sqmc-campaign-closeout-audit-20260924.json).
These supersede the earlier completion conclusions in the Phase 0–4 reports.
Prior versions of this memo remain in Git history; the interrupted staged
version was saved under `/tmp/bayesfilter-sqmc-integration-20260924/initial-staged.log`.

## Integration checkpoint

Recovered Claude session: `4b6ba369-219e-4a25-bbc3-547c5214d1e9`.
Starting SQMC commit: `e6e5fc40f02b4225ae7c2f83f1e6da7ece72b538`.
Starting local main/cached origin/main: `89065bc6354801cb368d5e163ba49fd9c4372d10`.
Those refs differed by independent commits. The registered main worktree under
`/tmp/bayesfilter-main-sync-20260922` was missing when recovery inspected it.

The closeout corrects the documentation and preserves the unexecuted P44
replication draft with explicit diagnostic status. The horizon runner's pending
executable-bit change is intentional to preserve the recovered branch work.

Skeptical integration audit: proceed after correcting the overclaim; preserve
both histories without rebasing or force-pushing; validate conflicts with
focused checks. Detailed plan, original patches, command logs, and the live
integration checkpoint are under `/tmp/bayesfilter-sqmc-integration-20260924/`.

Pre-commit repair: the branch tracked unified-kernel oracle tests without their
modules. The missing unified reset/correction kernels and numerical-safety
dependency were copied unchanged from starting main before committing. The
first hook attempt failed collection; no numerical assertion failed in that
attempt. Full CPU-only oracle validation remains required.

Next action: review and commit this closeout, perform the authorized merges,
validate, push main, and synchronize this branch. No research compute is
allocated or consumed by this task. A future score-repair campaign needs its
own current evidence contract and bounded budget.
