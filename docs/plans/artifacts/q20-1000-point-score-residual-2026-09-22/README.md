# Saved q20 Gaussian diagnostics

This directory preserves the 1,000-point score-residual diagnostic, its failed
setup attempt, and the source needed to interpret the saved results. The
[result note](../../bayesfilter-q20-1000-point-score-residual-result-2026-09-22.md)
reports the conclusions and limitations. The preceding
[Gaussian-step canary](../../bayesfilter-q20-gaussian-epsilon-canary-result-2026-09-22.md)
and its evidence are included in the same commit.

The successful point evaluations, manifest, control, and summaries are in
`../q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00006-gaussian-score-residual-1000/worker/`.
The exact frozen map, beta=1 qualification, and saved canary control bank are
preserved at their original repository-relative paths referenced by the runner.

`source-r2.tar.gz` contains the 491 source files named in the successful
worker's source manifest, including the platform-specific compiled operation.
Every archived file was checked against its recorded SHA-256. Its companion
`source-r2-manifest.json` records both the file hashes and archive hash. This is
the historical execution snapshot; the archive does not replace the current
library or promote its contents as a new default.

The original launch scripts retain their exact bytes and machine-local paths
so their recorded checksums remain verifiable. They expect the original
`/tmp/BayesFilter-q20-recovery-20260922-r2` snapshot, TensorFlow GPU environment,
and campaign directory. They also update that campaign's budget ledger.
Treat them as preserved execution records: rerunning in another checkout needs
a fresh output directory and allocation, with the archived numerical sources
and frozen inputs retained. This publication does not bundle every historical
training checkpoint or make the entire campaign independently resumable.

`publish-verification.json` records the pre-commit artifact verification; it
recomputes reported residual summaries from saved points without new target
evaluation or GPU work.
