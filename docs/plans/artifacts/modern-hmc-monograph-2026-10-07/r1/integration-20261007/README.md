# Integration with remote main

The owner requested commit, merge with remote, conflict resolution and push.
The shared checkout has unrelated changes, so the survey is integrated in an
isolated worktree based on remote main. Only the chapter, bibliography additions,
contents spacing fix, reports, PDF deliverables and selected audit evidence are
in scope. Third-party paper PDFs and code remain in the local source archive;
their source manifest is included.

Skeptical audit before integration: remote main.tex, references.bib and
preamble.tex match the saved baseline. The index is clean before transfer.
The copied final chapter must retain the hash used by MathDevMCP. Rebuild both
documents from the integrated tree and check citations, references and source
dependencies before committing. The original delivery PDFs and manifests remain
unchanged, with integration checks recorded separately here. This validates
document integration only, not sampler performance or formal proof.

Proceed with a normal merge of any newer origin/main and a non-force push to
main. Do not alter the shared checkout or include runtime-code changes.

Both integrated builds passed on 7 October 2026 with all citations and
cross-references resolved. The [standalone chapter](chapter-build/modern_hmc_survey.pdf)
has 39 pages, no overfull boxes, and exactly the same extracted text as the
original delivery. The [integrated monograph](book-build/main.pdf) has 702 pages;
Chapter 54 occupies printed pages 585–616. This is the PDF corresponding to
the source integrated with remote main. The earlier 691-page book is retained
as the original shared-worktree delivery.

Two existing chapters differ from the original delivery: entropic OT/Sinkhorn
and the HMC tuning interfaces. Their remote versions were preserved. All other
recorded source inputs match. Source hashes, the two differences, exact build
commands, timings and PDF hashes are recorded in source-checks.json,
validation.json and the per-build build-command.json files. The full book
retains 221 overfull-box warnings in existing material. Rendered review of
the contents, new chapter opening and chapter ending found no clipped content
or crowded chapter numbers. The original source-bound MathDevMCP evidence
remains unchanged; provenance-checks.json verifies all recorded audit hashes.

Only final label/symbolic audits, the initial document audit and report-assembly
error, source snapshots, build evidence and selected rendered pages are included
in this commit. Redundant earlier audit dumps and downloaded third-party papers
and code remain local. Their absence from Git is not absence from the review.
