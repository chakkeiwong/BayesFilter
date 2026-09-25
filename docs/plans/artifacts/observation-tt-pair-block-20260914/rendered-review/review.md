# Rendered manuscript review — 2026-09-14

The completed attempt05 manuscript has 48 pages. Final pdflatex passes 9 and 10
both exited zero, with no overfull boxes, undefined references or rerun request.
`latexmk` was unavailable; direct pdflatex supplied the completed build without
package changes. Exact PDF/source hashes, commands and review scope are in
[build-and-render-manifest.json](build-and-render-manifest.json); all build-pass
logs and the final TeX log are preserved here.

All 48 pages were inspected at overview scale. The abstract, pages 39–47
(the pair construction, propositions, recurrence, tests and results), and the
closing appendix page were also inspected at reading scale. The two overview
images and individual page images preserve that view. Tables 2 and 3 are legible
and carry the uncertainty and heuristic-veto qualifications. The positive-Gram
recurrence and counterexample sit directly after the conditional-density proof.
The layout fixes split the weighted objective into three aligned rows, allow
the result path to wrap, and keep the alternative-explanation paragraph intact
around the table float. No clipping or overlapping mathematics was observed.

The [preservation audit](../manuscript-preservation-audit.json) checks both
protected source hashes. All labels, citations, paths and inline mathematics
remain. It covers 215 current display-math blocks, compared with 213 before
sampler recovery and 199 before the pair addition; the only revised existing
display is the identical weighted-objective algebra in an aligned layout.
These counts include bracket displays and starred environments, so they are
broader than the first draft of the audit. Source diffs are preserved alongside
that audit. The removed promise of future checking is replaced by actual
results and explicit limits; no substantive finding was deleted for brevity.

Eight existing hyperref warnings concern mathematical tokens in PDF bookmarks
for unchanged older headings (current source lines 264, 351 and 3345). They
are retained as bookmark-only limitations. Existing underfull-box notices and
the older audit-oriented sections do not clip the printed content. The
abstract's untested full UKF extension is distinct from the executable SGQF
comparison; its historical qualification is retained. Numerical validity,
heuristic rejection, solver uncertainty, uncalibrated 5% defensive mass and
the limited algebraic checks remain explicit in the new results.

This is author-side layout and exposition inspection, not independent theorem
certification or a claim of human approval of the voice. The eight theorem
audits remain inconclusive; pair total gradients and HMC readiness remain
unchecked. Human manuscript assessment is pending.
