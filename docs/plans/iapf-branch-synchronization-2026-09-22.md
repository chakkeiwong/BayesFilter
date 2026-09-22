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

Current synchronization status: preparing the source commit and focused CPU
checks; reconciling remote connectivity. The initial SSH fetch stalled and was
stopped. An all-branch HTTPS fetch failed with an early EOF; the next attempt
fetches only main. No remote push has yet been made.

The first focused CPU/XLA run passed 133 tests and exposed one stale assertion
in the canonical reset rejection test. The shared executor has returned the
explicit rejection sentinel `(-inf, 0)` since commit `bdc0bdcb`; the test still
expected a non-finite score. The test now checks both exact sentinel values.
This is a test-contract repair and does not change the numerical executor or
any source frozen by the running R campaign. The focused CPU/XLA rerun passed
all 134 tests in 180 seconds, with GPU devices intentionally hidden.
Archived CSV line endings and patch context are preserved verbatim; whitespace
checks pass with CRLF accepted for archived CSVs and archived patch files excluded.

Detailed local command logs and the explicit staging selection are in
`/tmp/bayesfilter-sync-20260922/`. Generated evidence required to resume older
diagnostics is preserved in the original checkout; a fresh clone alone does
not contain those raw experimental payloads.
