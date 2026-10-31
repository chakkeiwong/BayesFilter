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

## Integration and verification result

The evaluated source was merged, scoped C3 work committed as f77bbc450, and
remote main 9ee1b8d76 merged as 45a2e13cf. The only merge conflict was the
active checkpoint document; retain the current C3 result and mark the older
October 14 status historical. There were no numerical merge conflicts.

The remote LEDH changes are present, including observation-weighted conditional
parents/SIR (a6ef2ed0f), KSC/range-bearing extension (12449d91f), and structural
scope guard (c11f1efa6). `bayesfilter/highdim`, `experiments/dpf_implementation`,
the observation-parent score diagnostic and LEDH paper sources match remote
main byte-for-byte. No unstaged shared-workspace edits were present in those
LEDH directories; unrelated dirty work remains untouched. This establishes
integration preservation, not new scientific LEDH admission.

Initial test collection exposed a missing native Sylvester library in the
new checkout. CMake built it from the merged source, using tfgpu Python,
before retry. The generated binary is ignored and is not committed. The
merged C3/reference/recovery/LEDH suite then passed 72 tests with one optional
consumer-module skip. Configuring the frozen real MacroFinance consumer
separately passed all ten of its tests. A further 139 numerical tests pass
for stationary derivatives, stack QR, typed candidate execution and the
optional batch-maximum estimator. These are CPU reference/engineering
checks, with XLA where specified; they do not qualify merged GPU performance.

The [C4 raw-draw audit](bayesfilter-modern-hmc-c4-raw-precision-result-2026-10-30.md)
completed all seven candidates/31 quantities after a localized reader repair.
Its eight focused tests pass. The prospective next execution is the
[fixed-count C4 replication](bayesfilter-modern-hmc-c4-fixed-count-replication-2026-10-30.md),
which keeps the original numerical source for a controlled comparison and
does not substitute it for the newly integrated LEDH implementation.

Withhold 3,600 seconds for this integration, audit and preparation inside the
existing allocation; the raw audit's 900-second allowance is included.
Reserve 98,215 seconds for the next all-member run, leaving an upper
58,914.47980995933 seconds unreserved. Unmeasured preparation is not silently
converted into additional credit. SIR's separate allocation is unchanged.

The reviewed continuation was committed as 6b8387efc and launched on GPU0
at 17:30:03 Asia/Shanghai, supervisor 1100045, worker 1100048. Its source
checks and original-evidence replay are progressing. The updated preflight
passes 38 tests; the final driver-only run passes 11. Compact launch receipts
and test XML are committed with the refreshed master.
