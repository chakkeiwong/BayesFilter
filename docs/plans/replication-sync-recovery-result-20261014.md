# Replication recovery and integration, 14 October 2026

The bounded controlled-SMC/KDM campaign was complete when the previous session
stalled. Its saved events show remote compaction stream failures at 02:55:44
and 02:58:12 UTC on 14 October 2026, and an earlier sandbox mount-helper failure.
No commit, merge or push had executed. These observations identify infrastructure
failures, not a failed replication or a proven context-window root cause.

The owner authorized documentation completion, scoped commits, integration into
main, remote synchronization and a non-force push, then updating the LEDH branch.
The shared surrogate-hmc checkout and its index remain outside this task.

## Checked integration evidence

| Check | Result | Interpretation |
|---|---|---|
| Current-code source closure | 283/283 checks; seven groups; 14 self-tests | Bounded source coverage, not a new numerical run |
| Replication regression | 68 passed; six missing-local-C0-snapshot skips; 11 subtests | Active harness and restored native reference pass; those six integration checks were not exercised |
| Existing oracle contract | Three passed | Existing commit gate only, not full LEDH scientific validation |
| Isolated-main full-score checks | 12 passed | Checkout-bound imports; focused integration regression only |
| Incoming-tree preservation | 117079 noncampaign entries; eight runtime hashes; 1848 scoped hashes | Incoming owner code and migrated numerical source preserved |
| Migrated file provenance | 1823 original records; four explained edits | No numerical runtime hash changed |
| Obsolete dispatcher baseline | 32 historical tests passed on predecessor | Preserve the historical pair outside active collection |

The four post-migration files changed are the campaign completion checkpoint and master
summary, the two reporting fixtures' required empty reference ledgers, and the
optional native-reference availability check. The original migration manifest
is preserved; a supplemental inventory records the final staged bytes.

The first migrated closure check failed one local trainer-source lookup; after
restoring that pinned file, the complete closure passed. The native Octave test
initially timed out with inherited display variables; the headless run passed.
Both failed attempts are retained. Neither repair changed filtering, training,
admission criteria or archived numerical evidence. The untracked predecessor
test was also checked against its original dispatcher before archival, rather
than silently dropped or made to pass against an incompatible API.

## Merges and environment repairs

The documentation and implementation commits are
`33fbbf2413683802e2b322bae03ac6b4a5e4518d` and
`9a4d50c759a50003db068c9db2a34173fba6f3e6`. Merging the campaign into
isolated main produced `9a95fbb45eeaeea187cea11a99c65193dfaf8754`.
Remote main `fceae6aac378bc89e4eeb687819e9d436ebfe356` then merged
cleanly, producing validated main `5f108dbfd3418d3cd9b9e3c30a226f125eb4998a`.
The campaign merge's sole conflict was `.gitignore`, resolved additively.

The first merged regression failed because sparse checkout hid a committed
compact specification. Oracle checks then exposed sparse-hidden experiments
imports. Materializing scoped committed files repaired these environment
failures; no algorithm changed. The investigation also found inherited
`PYTHONPATH` pointed at the shared checkout. Earlier hook passes are therefore
insufficient evidence for isolated-main imports. Final execution binds imports
to isolated main: 68 replication tests and 11 subtests, 12 full-score tests and
three oracle tests pass. Six C0-snapshot integration checks remain unexercised.
The source closure recheck and source-audit self-tests pass unchanged.

The executable preservation receipt compares incoming mode/blob identities,
not merely merge messages. All 117079 noncampaign incoming entries are retained,
all eight checked runtime hashes are unchanged, and both ignore rule sets are
present. Three final integration-document updates are separately hash-bound in
the receipt. The shared checkout remains on `surrogate-hmc` with an empty index;
the other agent's concurrent working-tree edits are outside this task and are
not asserted to be byte-for-byte unchanged over time.

## Evidence and limits

The synchronization evidence directory contains original and supplemental source
hashes, test receipts, the failed and accepted closure reports, and the archived
predecessor test/dispatcher. Full local logs are retained alongside the compact
receipts; the archive-binding records distinguish migrated code from the original
campaign output location. CPU-only checks set `CUDA_VISIBLE_DEVICES=-1` before
framework import. The environment is `tftwogpu`; no GPU training or new scientific
campaign ran during synchronization.

Raw campaign outputs, complete snapshots, papers, author repositories and native
reference dependencies remain in their recorded local archives. A fresh clone
contains the selected compact evidence and code, but is not self-contained for
numerical reruns. The accepted KDM scope remains the released 25-particle bearings
MDPF/A-MDPF runs. Exact Appendix B replication, whole-paper/population equivalence,
HMC readiness and production readiness are not established.

The integration plan limits focused tests to 1800 seconds and two localized
repair retries. The original collection failure and both repairs are recorded;
no further numerical or fixture repair is authorized by this synchronization.
Recorded test/commit-hook command wall time is 736.81 seconds before the final
receipt commit. The committed `integration-receipt.json` records pre-push
validation. The local terminal receipt
`/tmp/bayesfilter-replication-sync-20261014/terminal-git-receipt.json` records
actual final commit, push outcome, branch verification and final budget usage;
pre-push documentation alone does not establish delivery.
