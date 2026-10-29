# Completed LEDH/main integration validation

The merged source preserves the completed independent T=50 campaign and all
remote additions. The merge combined local main 285b43e111124b0f3337f81032905591b9a54324
with origin/main b1ccb8678c1ca68811eaa301bd4e1fd368e4bc23. Numerical LEDH sources
had no overlapping edits. The bibliography was the sole conflict: all 220 keys
remain, and the two differing shared records retain the metadata checked
against the publisher DOI responses. See bibliography-merge-review.json.

All 40 focused CPU integration checks passed. The initial check passed 38;
two failed because this worktree lacked the existing local author-source
cache. Copying that cache (474 files, 2202823 bytes) repaired the fixture;
both affected cases passed on the one allowed retry. These source files were already tracked but omitted from the sparse worktree;
all restored file contents match their indexed Git blobs. No numerical
implementation changed. Commands, hardware choice,
wall times and logs are recorded in focused-tests.json and fixture-retry.json.

The monograph builds to 746 pages with 217 overfull-box warnings. The standalone
paper builds to 25 pages with none. Both resolve all citations and references.
The main chapter list retains the covariance proposal and the remote chapters.
The merged empirical table on physical page 293 was visually inspected and
fits its margins. Build logs and PDFs remain local; compact receipts and the
reviewed page image are committed. The earlier development build is preserved.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Complete the authorized Git synchronization | 40 focused checks pass; both PDFs compile | No unresolved conflict, failed check or unresolved citation | Targeted checks do not cover every unrelated remote feature | Commit, push main, merge back and verify refs/cleanliness | New numerical, canonical-default or HMC evidence |

The campaign scientific findings remain in the master summary. No experiment
was rerun, retuned or promoted by this integration. The review budget allowed
one initial check and one localized retry; both were used. The final Git receipt
is /tmp/bayesfilter-ledh-final-sync-20261009.json, written after actual remote
and local refs have been compared. It is kept outside Git to avoid changing the
commit that it verifies.
