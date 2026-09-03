# R2a repair and refresh reset memo

Date: 2026-09-02  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md`  
Result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-r2a-repair-result-2026-09-02.md`

## State transition

`R2 attempt-01 (invalid artifact-root/resource failure)` -> `R2a repair complete` ->
`corrected R2 attempt-02` -> `M3 continuation veto`.

The first canary was preserved and was not promoted to candidate evidence. The
launcher source-owned root defect was repaired. Its timing records are retained
as an explanatory performance diagnostic. The unchanged canary contract gets
one fresh retry; a repeated bounded resource failure is a real continuation
veto under the M3 subplan.

## Repairs and checks

- Launcher root corrected; no scientific identity or cap changed.
- Focused Phase 9A contract suite: `13 passed`.
- Shell syntax and whitespace checks passed.
- New source hashes and the exact retry command are recorded in the result.
- Failed attempt remains under the literal-brace directory as immutable evidence;
  it is not deleted or reused.

## Refreshed assumptions

The canary's `1800` second cap, eight-pair grid, selection replications, target,
GPU0, XLA, TF32, memory-growth, and seed namespace remain frozen. The timing
projection is a hypothesis about route cost, not a license to reduce the grid,
skip held-out checks, widen the cap, or reuse partial calls.

## Terminal entry and stop rule

Attempt-02 was run with the command in the result note and reached the same
bounded resource failure. The terminal M3 result/reset memo now closes the
campaign. Phase 9B and the six-scope replay remain blocked. No additional retry
or cap change is authorized by this memo.
