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

Source commit `db72d33c` is integrated into local main in
`/tmp/bayesfilter-main-sync-20260922`. The cached remote history `d2124d42`
merged in `c41d6477`; latest remote main `8f992b20` subsequently merged without
conflicts. Both histories and the remote HMC policy changes are preserved.
The existing remote surrogate-hmc tip `8a5c23ab` is an ancestor of our branch.
The final synchronization sequence is to commit the reviewed merge and master
refresh, push main, fast-forward surrogate-hmc to main, push that branch, and
verify both local and remote commit/tree identities. The actual terminal
verification is recorded in `/tmp/bayesfilter-sync-20260922/final-verification.json`.

The bulk fetch delay was caused by extensive committed remote experiment data.
An all-branch HTTPS fetch failed with early EOF; a later main-only bulk fetch
remained active but was superseded and stopped after a successful filtered
fetch obtained the complete remote commit/tree history. Origin now uses Git's
standard on-demand blob retrieval. Matching sparse checkout patterns defer
71,722 bulk payload paths in 31 newly introduced remote artifact directories;
they exclude no previously tracked local path. Code, plans, compact text
results, and all existing local experiments remain materialized. Deferred
payloads remain tracked at their original Git identities and can be fetched
when needed. This changes local materialization, not either branch's tree.
Pattern and preservation records are in the local synchronization log directory.

After the cached remote merge, the isolated checkout passed 188
integration tests, with 30 external-reference tests skipped and one missing
fixture failure. The missing 1,729-byte underflow fixture is now versioned.
After copying the existing pinned R source and paper into the isolated local
resource directory, all 32 external-reference/fixture checks passed. The
availability check now requires the paper as well as R and source code, so a
fresh checkout reports missing references explicitly. No numerical algorithm
changed. The ignored custom-op binary was also copied from the original
checkout; its digest is recorded in the local synchronization logs.
After the latest remote merge, the complete focused integration run passed
219 tests in 212.51 seconds, with GPU devices deliberately hidden. This covers
the iAPF reference/score consumers and affected Kalman/HMC integration contracts;
it is not a new scientific or GPU validation claim.
Final checkout checks also exposed two integration defects. A literal closing
brace in an upstream ignore pattern broke ripgrep parsing; `[}]` preserves the
intended malformed-directory match and restores successful searches. The NeuTra
route guard found nine records for deleted scripts, six unregistered historical
LEDH diagnostics, and one unregistered validation adapter. The ledger now
matches the merged source; all six route-policy tests pass. The historical
classification follows the terminated September 7 program, and the validation
adapter explicitly supplies diagnostic controls. Discovery rules and active
route classifications are unchanged. These are integration repairs, not new
sampling or numerical behavior.

GitHub SSH authentication over port 443 succeeded using the already trusted
github.com host key. The repository's SSH command now uses that verified
connection for lazy fetches and pushes; its previous configuration is backed
up in the local synchronization logs. No host-key verification was disabled.

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

Master refresh: stages 100 and 300 are complete; stage 1,000 is active.
Stage 300 retains all 9,000 records, including two failed/capped d80 score
learners and the resulting heuristic promotion veto. The master schedules
completion of 1,000 before terminal uncertainty/control review and points to
the live budget. All nine frozen source hashes matched in both checkouts after
the latest remote merge. Eight unrelated tracked edits are preserved. Two TT
documentation files changed concurrently during synchronization; preserve their
latest contents rather than restoring their initial hashes.
The incoming tracked NeuTra route ledger supersedes an older ignored local
copy. Preserve that old copy in the synchronization backup before updating the
working branch; use the reconciled ledger with its corresponding merged sources.
