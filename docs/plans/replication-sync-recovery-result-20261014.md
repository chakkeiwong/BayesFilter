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
Final merge identities, merged-tree checks, push outcome and branch verification
are recorded in the integration receipt and terminal checkpoint.
