# Replication branch synchronization, 14 October 2026

The owner authorized completing the campaign documentation, committing this
branch, merging the work into main, fetching/merging remote main, pushing main,
and merging the synchronized main back into this branch.

## Scope and reviewed sequence

Use ledh-refactor-with-policy-fix in
/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922.
The shared /home/chakwong/BayesFilter checkout belongs to surrogate-hmc;
preserve its branch, index and all files. The earlier branch mismatch is an
operational error, not a change to the recorded numerical results.

1. Inventory both checkouts and preserve pre-migration copies of differing
   campaign files. Copy only the explicit campaign-owned implementation,
   harness, test, plan, and compact-result list to the LEDH worktree.
2. Refresh the completed replication checkpoint and final master summary.
   Commit documentation separately, then implementation/tests/evidence.
   Large raw outputs and repeated snapshots remain archived at their original
   recorded locations; scoped ignore rules exclude generated payloads only.
3. Fetch main early and integrate in a clean isolated worktree. Preserve all
   incoming LEDH changes. Resolve a conflict by inspecting both intentions;
   never replace a changed LEDH module with an older checkout wholesale.
4. Run CPU-only focused replication tests, the source-coverage self-tests,
   and the existing commit oracle checks. Verify copied source hashes.
   Recheck the closure against the current migrated code and archived evidence.
5. Refetch before pushing, merge further changes if any, and push main without
   force. Fast-forward the LEDH branch to the synchronized main; verify branch
   and tree identities and a clean LEDH worktree.

## Skeptical review and evidence contract

PASS for execution: the explicit selected replication paths do not overlap the
incoming main diff, after excluding the other agent's observation-model work.
The new remote main includes the user's LEDH changes. Neither clean-worktree
status nor a successful merge establishes scientific correctness. The relevant
integration evidence is source preservation, focused regression results, and
the unchanged checked replication consumers. The older branch and completed
campaign are not a baseline for overwriting the user's current LEDH algorithm.

Stop for unrelated staged/dirty main work, unexpected task ownership,
unresolved semantic conflicts, a failing regression, a changed numerical source
hash, or a concurrent remote update that has not been integrated. Local harness
and migration repairs may continue within the same scope. No force push,
destructive reset, stash of another agent's files, or GPU campaign is planned.

This is a repository integration task, not a new numerical comparison.
CPU-only checks deliberately hide GPUs before framework import. Allow at most
30 minutes of focused test wall time, with at most two localized repair
retries. A failed test triggers investigation, not relaxed acceptance.
No new scientific ranking, expanded replication scope, algorithm repair,
HMC readiness, or validation of all main-branch LEDH behavior is concluded.

Command logs and original inventories:
  /tmp/bayesfilter-replication-sync-20261014
Committed result and selected-file hash inventory:
  docs/plans/artifacts/replication-branch-sync-20261014/
The final result records exact commits, commands, tests and archive limitations.

## Recovered execution, 14 October 2026

The saved session ended after two remote compaction failures, not a scientific
failure. Its final fixture patch also failed because the sandbox mount helper
reported `mountinfo path is not absolute`. No campaign commit, merge or push
had occurred. The independent recovery inspected only selected tail events
and the on-disk checkpoint, inventories and test summaries.

The initial collection failure was repaired by restoring omitted Lorenz
helpers. The first retry exposed missing local reference inputs and reporting
fixtures without the now-required reference ledger. Restore archived inputs,
provide empty ledgers in the two isolated reporting fixtures, and explicitly
skip the native Octave reference test only when its optional local dependencies
are absent. Numerical implementation and admission criteria remain unchanged.
The final localized repair retry passed: 67 tests passed with seven skips.
After restoring the optional native source and clearing the inherited display
variables, all 68 runnable replication tests passed; six source-continuation
integration checks skipped because the local C0 snapshot is not in this checkout.
The recorded `-rs` recheck confirms this limitation. The current-code source audit passed
all 283 checks and 14 self-tests. The existing oracle contract passed three tests.

An additional untracked test in the original LEDH checkout targets the replaced
September 24 dispatcher API (`Campaign`, `detach_master`, `supervise_existing`).
It failed against the migrated September 28 dispatcher, but its 32 tests passed
against the byte-preserved pre-migration dispatcher. Preserve both files under
the synchronization evidence's `pre-migration` directory, outside active test
discovery. This is a documented retirement of stale migration residue, not an
algorithm repair, a relaxed scientific screen, or a discarded failing candidate.
`tests/replication/test_repair_master.py` checks the active dispatcher.
