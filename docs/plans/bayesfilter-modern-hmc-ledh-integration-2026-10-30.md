# Modern-HMC continuation with current remote LEDH changes

30 October 2026. The owner requests commit, remote merge, conflict resolution,
push, master refresh and continued execution, explicitly preserving the recent
LEDH algorithm changes.

## Integration plan and skeptical audit

The shared workspace has substantial unrelated dirty and untracked work.
Its HEAD is 8821e8c0c; fetched remote main is 9ee1b8d76, 349 commits ahead.
Use the isolated sparse worktree
`/tmp/BayesFilter-modern-hmc-integration-20261030` on
`integrate/modern-hmc-c3-20261030`. Preserve the shared workspace and original
campaign evidence. Do not stage all untracked data or roll back remote LEDH.

1. Merge the exact evaluated HMC/factor-native source commit f9291ee44.
2. Commit scoped C3 implementation, recovery tests, plans and compact evidence.
3. Merge current remote main. Resolve overlapping documentation semantically;
   preserve LEDH algorithm changes and test the merged HMC numerical paths.
4. Verify remote LEDH path identities and source ancestry. Run focused
   reference/recovery/HMC checks in the merged checkout and inspect applicable
   LEDH tests. Push normally to main after checks; never force-push.
5. Continue C4 from the merged source, with a prospective plan and unchanged
   completed C3 outcomes. Refresh the master, checkpoint and accounting.

Pre-execution audit passes for this integration: source preservation rather
than mass staging avoids overwriting concurrent work; unchanged LEDH blobs
and ancestry are the merge preservation criterion, not claims of scientific
correctness. Earlier C3 results retain their frozen source and are not silently
reclassified as results for the merged implementation. Any numerical conflict
or failed meaningful regression must be resolved before a new campaign.
No default thresholds, target definitions or campaign budget change here.

C3 has settled at 18,347.479901695042 seconds; remaining upper allocation is
160,729.47980995933 seconds with preparation still withheld. Separate SIR
funds remain 7,363.064379416 seconds. C4 diagnosis is next; new LEDH validation
requires its own current scope and must not reuse older LEDH conclusions.
