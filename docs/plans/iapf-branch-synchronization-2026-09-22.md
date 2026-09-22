# Synchronize the iAPF research branch with main

The owner requested committing this branch's research work, merging it into
main, reconciling main with origin/main and pushing, then bringing
surrogate-hmc to exactly the same commit as main.

The source checkpoint includes the iAPF/score-study changes, their tests,
reference and diagnostic runners, plans, and compact research result notes.
Unrelated observation-aware TT work remains in the shared working tree.
Generated particle records, repeated source snapshots, live checkpoints, and
large run payloads remain local; the versioned plans and result notes identify
their locations. This synchronization does not change scientific status.

Execution review: use a separate worktree for main, preserving the live R
supervisor and its frozen source hashes. Commit only explicitly selected files.
Merge histories without a force push, resolve conflicts by preserving the
intended behavior on both sides, and run focused checks on affected code.
Verify main and surrogate-hmc have identical commit and tree identities at
completion, and verify the remote main commit after pushing. Other research
branches and unrelated working-tree edits are outside this synchronization.

The phase 16 R campaign continues under its existing budget. Its authoritative
state is in
`docs/plans/artifacts/iapf-adaptive-replication-ladder-20260922-01/checkpoint.json`.
No experiment-source change is permitted during its execution without a
recorded repair and renewed numerical verification. Git commit IDs may change;
each future launch records the actual commit and the frozen source hashes.

Current synchronization status: source commit `db72d33c` is complete and local
main has fast-forwarded to it in `/tmp/bayesfilter-main-sync-20260922`.
The initial SSH fetch stalled and was stopped. An all-branch HTTPS fetch
failed with an early EOF; the main-only HTTPS fetch is downloading a large
pack and still making progress. Remote main was observed at `8f992b20`.
The existing remote surrogate-hmc tip `8a5c23ab` is an ancestor of our branch.
No remote push has yet been made. Merge and final identity checks remain.

The cached remote history `d2124d42` merged cleanly in `c41d6477`; its ancestry
to the current remote tip was checked. The isolated checkout then passed 188
integration tests, with 30 external-reference tests skipped and one missing
fixture failure. The missing 1,729-byte underflow fixture is now versioned.
After copying the existing pinned R source and paper into the isolated local
resource directory, all 32 external-reference/fixture checks passed. The
availability check now requires the paper as well as R and source code, so a
fresh checkout reports missing references explicitly. No numerical algorithm
changed. The ignored custom-op binary was also copied from the original
checkout; its digest is recorded in the local synchronization logs.

GitHub SSH authentication over port 443 succeeded using the already trusted
github.com host key. A brief duplicate fetch on that connection was slower
and was stopped; the original HTTPS download remains active. The authenticated
port-443 connection is available for the requested pushes.

The first focused CPU/XLA run passed 133 tests and exposed one stale assertion
in the canonical reset rejection test. The shared executor has returned the
explicit rejection sentinel `(-inf, 0)` since commit `bdc0bdcb`; the test still
expected a non-finite score. The test now checks both exact sentinel values.
This is a test-contract repair and does not change the numerical executor or
any source frozen by the running R campaign. The focused CPU/XLA rerun passed
all 134 tests in 180 seconds, with GPU devices intentionally hidden.
The repository's pre-commit oracle contract also passed all three tests.
Archived CSV line endings and patch context are preserved verbatim; whitespace
checks pass with CRLF accepted for archived CSVs and archived patch files excluded.

Detailed local command logs and the explicit staging selection are in
`/tmp/bayesfilter-sync-20260922/`. Generated evidence required to resume older
diagnostics is preserved in the original checkout; a fresh clone alone does
not contain those raw experimental payloads.

Master refresh: stage 100 is complete (3,000 records, zero failed/capped
learners); stage 300 is active. The master now explicitly schedules completion
of 300 and 1,000 before terminal uncertainty/control review, removes its stale
no-worker statement, and points to the live budget. All nine frozen source
hashes matched before and after the source commit; all eight excluded tracked
files still matched their initial content hashes.
