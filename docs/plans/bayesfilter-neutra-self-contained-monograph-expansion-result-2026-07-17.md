# NeuTra Self-Contained Monograph Expansion Result

Date: 2026-07-17

Decision: `COMPLETE_SELF_CONTAINED_NEUTRA_METHOD_AND_BOUNDED_EVIDENCE_CHAPTER`

## Outcome

`docs/chapters/ch26b_neutra_transport_hmc.tex`, included directly by
`docs/main.tex`, now gives a self-contained derivation of NeuTra HMC and a
complete, evidence-bounded account of the available BayesFilter experiments.
The final merged monograph is `docs/main.pdf` (415 pages).

The chapter derives rather than merely cites:

- the constrained-to-unconstrained posterior target;
- Hamiltonian dynamics, energy conservation, phase-volume preservation,
  leapfrog reversibility, Metropolis detailed balance, and acceptance;
- the exact Jacobian-corrected neural pullback target and score;
- the active three-stage dense autoregressive flow, triangular Jacobian,
  constructive inverse, stacked log determinant, reverse permutations, and
  fixed affine lift;
- reverse-KL training and its reparameterized gradient;
- source-space position-dependent geometry induced by latent Euclidean HMC;
- rank-normalized split, folded, and modern R-hat, conceptual ESS, and the
  posterior truth-tail diagnostic;
- UKF/SGQF prediction-error likelihood and score, tested point rules, SGQF
  normalization and covariance, and the bounded-uniform probit chart.

It documents five comparable truth-tail configurations and all 38 parameter
rows, followed by eight heterogeneous historical configurations.  The opening
boundary states explicitly that four inherited DSGE targets are evidence
summaries rather than re-derivations of their complete economic systems.

## Experimental Evidence

| Configuration | Parameters | Retained draws | Max modern R-hat | Min bulk/tail ESS | Min truth-tail | Classification |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LGSSM exact Kalman | 18 | 16,000 | 1.00215 | 4571.6 / 3977.0 | 0.06781 | central one-seed pass |
| Predator-prey UKF | 6 | 16,000 | 1.00081 | 27623.6 / 13394.1 | 0.21255 | noncentral one-seed pass |
| Predator-prey SGQF | 6 | 16,000 | 1.00033 | 26978.5 / 12974.7 | 0.21505 | noncentral one-seed pass |
| Parameterized SIR SGQF | 3 | 16,000 | 1.00007 | 16358.5 / 14568.5 | 0.37266 | central one-seed pass |
| Structural UKF | 5 | 16,000 | 1.00668 | 971.1 / 2354.5 | 0.28442 | qualified noncentral pass |

All 38 generating values are inside their empirical 95% intervals.  The
structural result is qualified because it missed the original bulk-ESS 1000
convenience gate by 2.9% and was accepted after the result under a disclosed
owner threshold of 900.  It is not represented as an automated pass of the
original rule.

Across the preserved evidence ledger, learned NeuTra reached transformed HMC
in ten model families and thirteen posterior-target configurations.  Nine are
clean/strong under their original screens and four are qualified/marginal.
Historical diagnostics are labeled by their preserved definitions; none is
silently promoted to modern rank/folded R-hat.

## Verification

| Check | Result |
| --- | --- |
| Skeptical plan/default audit | pass after narrowing the historical self-containedness boundary |
| Active implementation audit | pass for dense-IAF composition, training semantics, batching, XLA, and GPU policy |
| Structured archive/table comparison | pass: 38 rows, 190 numeric fields, 38 interval flags, five minima |
| Chapter duplicate-label scan | pass |
| Bare/malformed TeX command and tab scan | pass |
| `git diff --check` on task text artifacts | pass |
| Standalone chapter build | pass: 16 pages with bibliography |
| Full `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` | pass: 415 pages in an isolated merge of current `origin/main` and the NeuTra commit |
| NeuTra-only full-build warning slice | clean |
| Rendered-page inspection | pass for opening, derivations, model equations, both result tables, and historical section |

Claude read-only reviews:

1. Core mathematics, lines 1--435: `VERDICT: AGREE`.
2. Initial full review: `VERDICT: REVISE` for ambiguous structural tuple order.
3. Corrected experimental section: `VERDICT: AGREE`.
4. Added Hamiltonian, IAF inverse, and SGQF proofs: `VERDICT: AGREE`.

The review finding was repaired by fixing the reporting order explicitly to
$(\rho,\sigma,\phi,\gamma,R)$ in the priors, truth, prior mean, and table.

## Decision Table

| Field | Status |
| --- | --- |
| Decision | chapter complete and included in the built monograph |
| Primary criterion | method derivations, active implementation, and reported numbers are self-contained or explicitly bounded and traceable |
| Veto status | no unresolved mathematical, source, number, chapter-build, or chapter-layout veto |
| Main uncertainty | experiments remain one-fixture or heterogeneous historical evidence; four DSGE systems are summarized rather than re-derived |
| Next justified action | use the chapter as the monograph's NeuTra method and evidence reference |
| Not concluded | calibration, repeated-seed coverage, filter exactness, universal mode recovery, sampler superiority, production readiness, or universal default readiness |

## Inference Status

| Field | Status |
| --- | --- |
| Hard veto screen | no hard veto in the five common retained archives under their disclosed classifications |
| Statistically supported ranking | none; no method ranking was attempted or established |
| Descriptive-only differences | acceptance, runtime, loss, ESS, R-hat, posterior means, interval widths, and cross-model differences |
| Default-readiness | not established by this documentation result |
| Next evidence needed | additional seeds only for broader calibration/reliability claims, or direct baselines with uncertainty for superiority claims |

## Run Manifest

- Git commit at close: `d269f5bbd8531b878d4f25897a357fbc8f172488` with a dirty
  concurrent multi-lane worktree preserved.
- Command: `cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
- Environment: host TeX Live/pdfTeX with Latexmk 4.83; no conda environment.
- CPU/GPU: documentation build only; GPU not used and no scientific run launched.
- Data version: the two immutable structured archives named below.
- Seeds: reused preserved experiment artifacts; no new stochastic generation.
- Wall time: final incremental full build reported current/up-to-date status;
  earlier clean full rebuilds completed in seconds.
- Plan: `docs/plans/bayesfilter-neutra-self-contained-monograph-expansion-plan-2026-07-17.md`.
- Result: this file.
- Chapter SHA-256:
  `a9c069a60f03503e3f1bbc509e1a5f260c27219f4f5a43874a9bcbbf8a9c0c1c`.
- Full PDF SHA-256 at close:
  `8b2730d053517b848217ef5b64504cc65d4962216181e4c6482ce455618a6b20`.
- Retrospective JSON SHA-256:
  `3d7be7f1079b1947888b7e4cb0e26706df6c38a09ac42419c1d4423613dd9c6d`.
- Structural JSON SHA-256:
  `b05c9afd2feb70c3a5722e1c9b6719211fa049fb8620a5da30af08e66bb54cd7`.

The chapter and PDF hashes above were captured before this result note was
written and bind the reviewed content.  No result archive was regenerated or
overwritten.

## Post-Run Red Team

The strongest alternative explanation for the experimental success is fixture
favorability: LGSSM and SIR place truth at the prior center, predator-prey is
partly central, and every common result uses one frozen dataset.  Approximate
UKF/SGQF likelihoods also define filter posteriors that need not equal the exact
latent-model posterior.  A missed mode, repeated-seed truth-tail failures, or
invalid filter likelihood would overturn any stronger claim.  The weakest
common result is the post-result structural ESS adjudication, which remains
visible rather than being normalized away.

Terminal drift verdict: `NO_MATERIAL_DRIFT`.  No new experiment, baseline
change, diagnostic promotion, GPU run, package mutation, or claim beyond the
evidence contract occurred.
