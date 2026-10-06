# FAB failure and multimodal training: monograph revision

Owner request: document the calibration failure in the LaTeX monograph and
assess the literature for a way to cover multiple posterior modes. This task
edits documentation, inspects sources and builds the book; it launches no
training, posterior campaign or algorithm/default change.

Question: which conclusions follow from the completed frozen-map FAB study,
and which sampling/training design addresses missing modes while preserving
the canonical IAF and the existing NeuTra estimation objective?

Baseline: protect the current `ch26d_modern_importance_sampling.tex`,
`references.bib`, `main.tex` and preamble before editing. The shared checkout
contains unrelated changes, including to `main.tex` and the bibliography;
preserve them. Append citations only where needed and connect additions through
the existing importance-sampling chapter. Do not merge the execution checkout.

Evidence contract: report the exact finite-buffer replay comparison and the
five completed calibration screens; label the interrupted K43 measurement.
Present the tail theorem only for its proved real-valued UKF target, with the
executable covariance-margin/domain limitation. Source-faithfulness of the port
does not imply target integrability or successful optimization. Recommendations
must have inspected technical paper sections and explicit formulas, and must
not claim unknown-mode completeness or a statistically established method
ranking. Successful compilation and rendered-page inspection establish document
integrity, not empirical validation of the proposed remedy or human acceptance
of the prose.

Steps: (1) copy protected manuscript/evidence inputs to a versioned artifact
directory; (2) inspect the existing survey, historical result scope, source
implementations where available, and targeted modern SMC/tempering papers;
(3) add the failure derivation and comparison of remedies, with an explicit
posterior-sampling/weighted-forward-KL recommendation if the audit supports it;
(4) verify equations, source boundaries and citations; (5) build the complete
monograph in an isolated output directory and inspect the new rendered pages.
Keep a source-reading record and terminal build/review note here.

Skeptical review before execution: a prior proof about an ideal UKF extension
must not be promoted to a proof about a status-truncated numerical density.
Mode discovery, relative mass estimation and whitening answer different
questions. Better finite ESS cannot prove missed-region absence. Prior
defensive mixing may bound importance weights while remaining inefficient.
Ordinary posterior SMC and alpha-two FAB annealing have different terminal
targets; demonstrate the difference rather than simply recommending more
temperatures. Historical weighted-flow successes used retired architectures
and known mode support, so they are motivation only. The q20 target has four
free parameters despite its larger latent-state dimension. No mass-adaptation
prerequisite or all-method comparison is to be added. These checks permit the
bounded documentation work to proceed.

Numeric choices: retain measured calibration values and their uncertainty;
retain the derived prior variance 16, T=30 and interval proof coefficients.
New sampler budgets, ESS fractions, mixture weights and fitting tolerances
are to be determined by the eventual target-specific pilot, not invented here.
Document-build timeouts and protected copies are operational limits, not
scientific criteria. Store under
`docs/plans/artifacts/fab-monograph-coverage-2026-09-28/`.

## Recovered progress and source review

Protected inputs and executed evidence are saved; `protected-inputs.json`
records their hashes. The existing chapter remains the baseline, with two
insertions planned after the FAB tail discussion and before evaluation.
Three additional technical sources (Dai et al. 2022, Syed et al. 2022,
Gabrié et al. 2022) and their relevant author code have been inspected.
Detailed anchors and limitations are in
`.localresources/fab-coverage-followup-20260928/reading-notes.md`.

The source review supports ordinary posterior annealing followed by weighted
forward-KL fitting of the unchanged IAF as the next mechanism to study. It
does not establish superiority or unknown-mode completeness. Freeze kernel
parameters as well as the ladder for fixed-design evidence; do not import
the SMC reference runner's same-population mass adaptation. Compose invariant
local/global kernels using invariance, not an unjustified detailed-balance
claim. The FAB calibration uses K *interior* distributions, excluding its
two endpoints. Its actual fresh training loss retains the author's extra
1/B normalization; the chapter's stale statement to the contrary needs a
narrow correction. No new sampling or training is part of this revision.

## Completion

Both additions are integrated into Chapter 50. The complete 607-page monograph
and the 26-page chapter review compile with resolved citations/references;
rendered new pages were inspected. All 36 original equation blocks and all
42 protected-input hashes are retained. See
`bayesfilter-fab-monograph-coverage-result-2026-09-28.md` for the mathematical
review, build repairs, exact artifacts, limits and continuation state.
