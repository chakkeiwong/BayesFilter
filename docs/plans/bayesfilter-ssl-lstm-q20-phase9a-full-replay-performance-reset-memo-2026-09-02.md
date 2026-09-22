# M3 terminal repair and refresh reset memo

Date: 2026-09-02  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md`  
Result: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-result-2026-09-02.md`

## State transition

`R2a repair complete` -> `corrected R2 attempt-02` ->
`M3 continuation veto / terminal closeout`.

Attempt-02 used the repaired source-owned root and a fresh output directory. It
reproduced the attempt-01 bounded resource failure: 21 calls completed in
`1645.967775` seconds, then the fixed `1800` second cap terminated the next
call. The two-runner trace/reuse pattern was unchanged. This is the second
bounded resource failure after R1, so the M3 continuation veto is active.

## Mandatory closeout actions completed

- Preserved both attempts and their manifests; no partial calls were reused.
- Verified the corrected root, target/profile/seed identities, and failure
  classification.
- Ran the smallest exact regression before the retry: `13 passed`, Python
  compilation, shell syntax, and `git diff --check` all passed.
- Wrote the terminal result with decision and inference-status tables, cost
  decomposition, uncertainty limits, and a post-run red team.
- Refreshed the master and subplan to block the six-scope replay and Phase 9B.

## Remaining repair boundary

There is no further local repair that preserves the current eight-pair schedule,
replicated selection, held-out verification, target, kernel semantics, GPU
class, and `1800` second cap while making the canary fit. Any next step must be
a new reviewed plan for a changed performance design or budget. It must not
silently reduce calls, widen the cap, or resume partial evidence.

## Next entry

The active program is terminal at M3. The next permissible work is a bounded
design note and skeptical audit for a new performance route; no GPU launch is
valid under the current plan. Phase 9B remains closed.
