# Active SQMC checkpoint — 2026-09-27

Active question: likelihood and every analytical score against Kalman for P44
d3 T=10,120, then full A/Q d3 and d10 at T=2,10,120; N>=1000.
Worktree /home/chakwong/BayesFilter-SQMC, branch sqmc-development.
HEAD f3995a06a467f16574f96bbc8a68ccbbc4e30dad; preserve pending repairs.
Plan: ../plans/sqmc-expanded-comparison-20260926.md.

Checked: full-model coordinate/callback/Kalman checks passed (4 CPU tests).
Full d10 T2 pilot attempt-04 completed in 317.42 seconds: 7/8 final cells
finite, one inverse-CDF cell invalid. This is a timing/validity pilot only.
Earlier attempts preserve two reporting exceptions and an invalid-cell abort.
Evidence: ../plans/artifacts/sqmc-expanded-20260926/attempt-04-pilot/.

Stage: runner repairs implemented; full ladder has NOT started.
Validation: 41 focused CPU-only tests passed in 43.96 seconds, including all
new full-model directional finite-program checks and reporting regressions.
Log: /tmp/sqmc-expanded-focused-checks.log. Preserve it with final artifacts.
The runner now uses all four data/filter combinations, measures heuristic
errors against Kalman, saves data/design hashes, runs each scope/route in a
separate process, and retains invalidity diagnostics and progress. Cached CPU
Kalman graphs replace repeated tracing. No scientific route/default changed.
Runner: docs/benchmarks/run_sqmc_expanded_comparison.py.
Next: finish skeptical runner review, GPU parity check, then run the ladder.
Before launch, reconcile the original 14-hour calendar window with the
interruption: first pilot started 2026-09-26T17:49:50Z; current clock was
2026-09-27T06:32:37Z. The original 12 GPU-hour resource budget remains;
the runner conservatively reserves 20 minutes for prior GPU work. Do not
silently extend resource/attempt limits or describe the campaign as complete.

Permission interruption: this Codex session is rooted at /home/chakwong/BayesFilter,
not this worktree. User was given verified --approve-for-me / --add-dir and
full-access CLI options; no live policy or global configuration was changed.
Use existing approved commands, with escalation for GPU and outside-root writes.

Prior N1008 result: ../benchmarks/sqmc-n1008-results-20260926.md.
P44 d3 T2: 72/72 finite; no supported permutation-versus-ablation ranking.
No HMC/default/production promotion.

# Historical SQMC campaign reset memo — updated 2026-09-24

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
attempt. The commit hook then passed all three CPU-only oracle tests (105.03 seconds).
Closeout commit: `c0e2c57227d0fe34b866a2915d0cce25efb4cd21`.

Integration resolution: main's TensorFlow time loop, shared batched correction,
moment-provider/schedule support, and fail-closed sentinel are retained. SQMC
ancestry and cumulative validity now run inside that loop, with ancestry and
validity included in diagnostic traces. The completed SQMC runner and its
compatibility entrypoints replace the earlier runner copies. See the
[integration validation note](../benchmarks/sqmc-main-integration-20260924.md).

The live checkpoint records the remaining Git operations and final ref checks:
`/tmp/bayesfilter-sqmc-integration-20260924/checkpoint.json`. No research compute is
allocated or consumed by this task. A future score-repair campaign needs its
own current evidence contract and bounded budget.
