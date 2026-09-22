# Degenerate-transition correction to the Younis manuscript

Date: 2026-09-12. Status: COMPLETE.

The user identified that the proposed ancestor-averaged Fisher recursion is a
smoothing construction and is not applicable as written to the degenerate
DSGE target. The manuscript already contains a singular-transition qualification
and an AR(2) support example, but its abstract and proposed research sequence
recommend the method too broadly. The current conversational explanation made
the same error.

The correction will state the restriction where the algorithm is introduced,
derive why fixed particle ancestors with distinct deterministic successor
constraints cannot supply ordinary all-ancestor smoothing, and limit the
proposed density-based comparison to regular-transition reference models.
Degenerate DSGE requires a separately derived score construction. Its
implementation remains deferred. An innovation or chart change is not assumed
to restore the displayed smoothing recursion.

Skeptical audit: this is a mathematical applicability defect, not a problem
that a successful full-rank Kalman benchmark, more particles, or PaRIS's cost
reduction would resolve. Do not assert that every smoothing method fails for
every degenerate model: exact reduced coordinates or another proved
conditional construction can be possible. Do not add transition noise,
transfer model settings, change defaults, or launch a research campaign.

Preserve the existing equations, citations, historical results, and September
11 archive. Save a new baseline and final source/PDF under
`docs/plans/artifacts/younis-degenerate-transition-correction-20260912-01/`.
Update the short analysis and recovery checkpoints so they no longer nominate
this recursion as a general DSGE solution.

Verification: exact arithmetic for the existing AR(2) example (singular
one-step kernel, distinct ancestor supports, and nonsingular two-step marginal),
preservation of existing displayed equations/labels/citations, clean PDF build,
and visual inspection of affected pages. These are reference/document checks,
not score-quality evidence. No GPU use, stochastic experiment, new literature
claim, or campaign budget is involved.

## Result

The corrected manuscript identifies the score recursion as smoothing at its
introduction. New Section 7.1, page 20, derives the ancestor-support
obstruction and single-ancestry collapse. The abstract, historical PaRIS
discussion, support discussion, and proposed comparison now restrict the
ordinary recursion to models with valid transition densities and backward
conditionals. The analysis and both recovery checkpoints carry the same
restriction. No general solution for the degenerate DSGE score is established.

Exact rational reference checks pass: distinct AR(2) lag coordinates give an
identity compatibility matrix; duplicate coordinates allow more than one
compatible parent, supporting the stated qualification. The one-step
covariance has determinant zero; the two-step covariance has determinant
sigma^4 = 1/16. These results concern conditional support and exact marginal
covariance, not empirical particle-score quality.

All 75 previous displayed equation blocks are unchanged; the revised source
has 77. All 90 previous labels and all 10 cited keys survive; the bibliography
is unchanged. The 36-page PDF builds with no LaTeX warnings, undefined
references/citations, or overfull/underfull boxes after references settle.
Rendered pages 1, 18, 20, 21, 24, and 32 were inspected. The new derivation and
qualifications fit on page 20 without clipped equations or text. The earlier
34-page source/PDF is preserved in the baseline directory.

Evidence: `verification.json`, `verify_correction.py`,
`manuscript.diff`, `build-record.json`, build logs, rendered pages, and
`final-manifest.json` under the artifact root above. Exact commands, commit,
checksums, and document-build timings are recorded there. Final source,
bibliography, and PDF are in `final/` and match the live document.

Post-run audit: this correction excludes the ordinary density-based
all-ancestor recursion as a general method for degenerate DSGE. It does not
assert that every specialized smoother is impossible or that a singular
one-step transition forces a singular predictive marginal. Introducing
innovation coordinates, ambient smoothing, or additional process noise would
need its own explicitly defined target and derivation.

Next action: deliver the corrected manuscript. No DSGE experiment is
scheduled. Its score construction remains unresolved; any future reference
comparison must state and check its transition-support assumptions.
