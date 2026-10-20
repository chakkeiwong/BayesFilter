# FAB failure and multimodal coverage: manuscript result and continuation note

Completed the requested documentation and literature assessment. The full
monograph builds (607 pages); the separate chapter review builds with its
bibliography (26 pages). This work produced no new training, posterior sample,
GPU experiment, algorithm change or default change.

The recommendation is to investigate ordinary posterior annealing with SMC,
optionally aided by mode-guided defensive global proposals, then fit the
unchanged canonical IAF by weighted forward KL. A frozen NeuTra map can then
support plain NeuTra HMC or the existing tempered ensemble. Discovery,
relative mass estimation and whitening require separate evidence. The
recommendation is not a statistically supported ranking or an implemented
repair.

## Manuscript and sources

- Chapter 50, Section 50.8, printed pages 496–500: saved-buffer replay bias,
  five completed AIS screens, the deliberate K43 interruption, and the
  positive-volume tail derivation for the ordinary real-valued UKF target.
  The numerical covariance-margin and score-status domain remains distinct.
- Section 50.11, printed pages 503–509: posterior SMC weights and CESS,
  corrected global proposals, a defensive minorization, weighted forward-KL
  training, a finite-cross-entropy argument, region allocation correction,
  comparison of alternatives and the missing-basin limitation.
- Section 50.12: corrected the stale statement about fresh FAB loss scaling.
  The executed port retained the author's extra 1/B; the gradient diagnostic
  removes it for batch comparisons.

The two additions are `docs/chapters/ch26d_fab_calibration_failure.tex` and
`docs/chapters/ch26d_multimodal_recovery.tex`, included by the existing
importance-sampling chapter. The original 36 equation blocks, labels and
citations were retained. `main.tex` and the preamble are byte-identical to the
protected baseline; the bibliography preserves its baseline as an exact
prefix and appends three entries. Unrelated source changes were preserved.

New technical readings cover Dai et al. (2022) on SMC, Syed et al. (2022) on
non-reversible tempering, and Gabrié et al. (2022) on flow-assisted MCMC.
Relevant author code was inspected. The source notes preserve two cautions:
the inspected SMC runner's fixed ladder still adapts its mass from the current
population, and composition of reversible kernels establishes invariance but
need not establish reversibility. No foreign implementation was adopted.
PDFs, extracted text, source copies and hashes are in
`.localresources/fab-coverage-followup-20260928/`.

## Mathematical and rendered review

The audit checked the prior's squared-density coefficient (-1/8), interval
inverse slope, positive-volume tube, covariance contribution of innovation
pairs, and the ordinary-value/executable-target distinction. It also checked
the posterior annealing weights, independence-MH ratio, defensive
minorization, finite IAF inverse growth, importance coefficients for weighted
minibatches and region oversampling, swap ratio and defensive FAB gradient.
No global integrability claim is made for the uncharacterized numerical
status domain. The new proofs do not establish empirical sampler performance.

Both final builds have zero undefined citations/references. The revised
chapter has zero overfull boxes and one harmless underfull vertical box.
Rendered inspection covered full-book PDF pages 514–518 and 521–528, including
equations, both calibration tables, footnotes, the continued methods table
and the final training/estimation discussion. These correspond to printed
pages 496–500 and 503–510. Compilation and model self-review do not establish
human acceptance of the prose; that remains pending.

The first build reused an old root bibliography and was unsuitable for review.
The final build uses a unique job name and generated bibliography in its own
output directory. A PDF-tool extraction also emitted recursive-dictionary
warnings, so the supplied review PDF was instead independently compiled
from the same chapter source, with its own resolved bibliography.

Artifacts under `docs/plans/artifacts/fab-monograph-coverage-2026-09-28/`:

- `build-r2/bayesfilter-fab-coverage.pdf`: complete monograph.
- `modern-importance-sampling-review.pdf`: chapter plus bibliography.
- `documentation-manifest.json`: exact build commands, source/output hashes,
  commit, branch, environment, and verification.
- `baseline/`, `revision/`, `protected-inputs.json`: protected before/after
  sources; all 42 protected-input hashes checked unchanged.
- `evidence/`: completed calibration reports, interval records and source.
- `review-r1/`: rendered page images and page mapping.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Document and retire the current alpha-two recipe as qualified training | Scoped failure evidence preserved and derived | Intended ordinary-UKF auxiliary target is non-integrable at this initial map | Executable tail/domain semantics and onset of finite-time effects | Resolve target semantics before further calibration | Universal FAB failure or a complete cause of every earlier collapse |
| Recommend posterior SMC plus weighted forward KL as a study | Well-defined posterior path and finite objective under stated assumptions | No new sampler or map has been assessed | Unknown-mode discovery, finite-particle error, IAF capacity and cost | Target-specific pilot and evidence plan if execution is requested | Solved q20 coverage, successful canonical training or method superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | Preserve the ideal-target integrability failure and unresolved executable interpretation |
| Statistically supported ranking | None from this documentation study or the five AIS screens |
| Descriptive-only differences | AIS noise and finite-reference differences; replay intervals apply only to their exact frozen buffers |
| Default readiness | No new algorithm, training default or posterior estimate promoted |
| Next evidence needed | Evaluator validity, independent posterior region-mass evidence, heldout canonical IAF geometry and downstream estimation |

The strongest alternative explanation remains practical local tuning or
numerical-domain behavior rather than remote tails as the cause of the old
finite-time failures. The manuscript keeps that question open. SMC may also
miss the same basins repeatedly, and a finite forward KL need not be easy to
optimize. The next campaign needs to test those mechanisms rather than assume
the proposed sequence solves them.

The shared documentation checkout was on
`preserve/shared-main-before-fab-20260926`, commit
`de80aaff5812ebfbed551977476c0868551a2c88`, with pre-existing dirty work.
No commit, merge or push was requested or performed. The separate FAB
execution checkout remains separate; none of its workers was resumed.
