# Explicit selection of proposal mixture probabilities

Question: how are the three probabilities beta chosen, and how do they enter
the proposal? The current shared chapter leaves them as unexplained frozen
configuration. This is a documentation and derivation repair, not a runtime
implementation or a filtering experiment.

Plan: preserve the current sources/PDFs; verify a primary source for convex
mixture-variance minimization; add an explicit independent-pilot optimization,
its defensive constraint and projected-gradient solver; connect it to the
whole-algorithm configuration; check the new algebra and rendered pages; build
both monograph and standalone document; update the active checkpoint.

Evidence contract: completion requires a reproducible selection algorithm,
correct density-mixture sampling, justified conditional variance identity,
an explicit role for the defensive floor, and consistent frozen-beta score
semantics. New labels/citations must resolve in both compiled documents.
Malformed densities, dependence of supposedly fixed maps on pilot evaluation
draws, omitted importance denominators, and a claim of score or full-recursive
optimality are vetoes. Symbolic checks and compilation verify bounded aspects,
not statistical improvement. No numerical beta, default, implementation,
cross-model non-regression or performance claim will be promoted.

Skeptical pre-execution audit: passed after identifying two essential limits.
Convexity applies only conditional on fixed incoming clouds/maps, not to the
entire recursive filter as beta changes. The second moment controls a local
likelihood estimator, not its analytical score. Independent held-out recursive
validation is therefore required before using a fitted beta in a claim-bearing
scope. Equal pilot probabilities are a coverage design, not a selected filter
default. The transition floor is a declared ratio-protection control, not a
universal numerical constant. Calibration/validation/claim data stay separate;
beta is frozen over all parameter evaluations in the admitted scope.

Budget: at most 45 minutes of local documentation work, at most four TeX
passes per document plus two repair passes if needed; bounded symbolic algebra
only. No frameworks, GPU jobs, model tests, campaigns, commits or pushes.
Artifacts: docs/plans/artifacts/ledh-proposal-beta-calibration-20261008-01/.
Stop scientific promotion if any mathematical condition remains unsupported;
record tool unavailability separately from a false mathematical conclusion.

Source comparison is limited to this selection mechanism. It is not a new
comprehensive literature survey or a claim that this is the newest/best tuner.

Completed 2026-10-08 within the 45-minute budget. Five TeX passes and one
BibTeX call per document; both PDFs verified and delivered. See
ledh-proposal-beta-calibration-results-20261008.md for outcomes and audit limits.
No scientific promotion or experiment was performed.
