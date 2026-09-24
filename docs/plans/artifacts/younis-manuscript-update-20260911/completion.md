# Younis manuscript revision, 11 September 2026

The requested revision is complete. The live manuscript and PDF are
`docs/papers/ledh_younis_kdm_score/ledh_younis_kdm_score.tex` and `.pdf`.
The PDF has 34 pages, compared with the protected 20-page source baseline.
The manuscript is the broader successor to the historical Section 3.6 note,
whose exact retained worktree path is given in the opening and source appendix.

## Coverage and scientific decisions

| Requested material | Revised treatment |
| --- | --- |
| Current implementation | Section 11 distinguishes research commit 804616e3 from main checkout 5cc59cfa, traces the existing endpoints, records adapter repairs and initialization/SIR limitations, and corrects the stale batch-reset statement. |
| Historical path and results | Section 5 connects the prior-weight sidecar, mixture algebra, observation integration, covariance-carry repair, raw versus normalized IWSG, and the two matrix-Gaussian comparisons. |
| Reasons for changing direction | Section 5.4 separates correct finite-program differentiation from model-score accuracy. The tested candidate losses do not reject all corrected mixture or ancestor-score constructions. The historical PaRIS cost exclusion is corrected. |
| Proposals and analysis | Sections 6–9 derive corrected marginal and retained-ancestor proposals, a fully adapted Gaussian example, Fisher recursion with initial-law terms, selective mixture gradients, valid controls and residuals, and other integration/inference options. Section 12 specifies the next discriminating comparisons. |

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the tested Phase 4B configurations | Paired intervals for excess score MSE are positive in both matrix scopes | No new artifact-invalidity finding; old power target not met | Untuned comparator controls, scope limits, and conditional bias not separately measured | Test a separately specified corrected-mixture/Fisher construction if execution is requested | All Younis mixture ideas fail |
| Retain corrected mixtures and ancestor averaging as the next hypothesis | Model factors and complete-data score supply a checked mathematical target | Initial-law, support, ratio, and implementation checks remain necessary | Finite-particle bias, equal-compute performance, and scalability | Begin with the proposed exact-reference checks and a bounded fresh study | The new combination is implemented or empirically superior |
| Retain selective gradients and centered controls | Exact mixture identities and Gaussian variance example pass reference checks | Invalid centering or wrong objective invalidates a claimed estimator | Covariance and downstream performance in a full filter | Compare estimators of the same declared expectation | Variance reduction removes existing baseline bias |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Current model initialization/support discrepancies still block conclusions that depend on them. No new filter run occurred. |
| Statistically supported ranking | Existing paired intervals support the losses of the tested raw-IWSG candidate in the two reported scopes. |
| Descriptive-only differences | Across-path bias/variance decomposition, value shifts, and numerical/computational diagnostics do not rank the new proposals. |
| Default readiness | No new algorithm, numerical default, HMC consumer, or runtime admission was changed. |
| Next evidence | Correct local densities and complete initialization, model-score oracles, nested replication, own-scope calibration, cheap conditional baselines, and equal-compute comparisons. |

## Preservation and review

The automated inventory retains all 53 original labels, all 51 original
displayed mathematical environments, and all five original citation keys.
The revision contains 75 displayed mathematical environments and ten cited
bibliography entries. Existing support/rotating-chart derivations, IWSG
identities, initial-law limitations, numerical failures, and negative results
remain in the main argument. The source appendix holds locations and provenance,
not a displaced technical derivation.

The mathematical proposition about positive bandwidth was rewritten: a centered
nonzero covariance changes the measure, but unequal scalar functions can still
have equal derivatives. The new proof states that limitation explicitly. The
old future comparator list was reconstructed around the new research question;
it now includes proper particle-score baselines, Gaussian approximations, and
the existing canonical comparator, with conditional evaluation. The old
statement that the current fused wrapper omits the reset is superseded by
inspection of its actual later body; its row-mapped scalar implementation still
does not meet the NeuTra training batching requirement.

Exact arithmetic verifies the positive-unbiased-normalizer/log-score
counterexample, Gaussian IWSG variance, finite-state Fisher recursion including
parameter-dependent initialization, scalar Gaussian density factorization,
and a two-component hybrid with changing weights, means, and integrand.
The static wiring assertions check actual imports, call arguments, row maps,
and shared correction dependencies. They do not establish runtime parity of
the later checkout; that question is explicitly recorded as NOT CHECKED.

The original Poyiadjis paper was recovered from the author's Oxford site and
inspected at its recursion, evaluation, theorem assumptions, and relevant
appendix definitions. PaRIS's method and expected-work assumptions were
inspected. Bibliographic author names were checked against the primary files
and Lai's official proceedings page. The two old invalid Poyiadjis HTML caches
remain excluded. The earlier analysis report and literature ledger were updated.

The PDF was built using pdflatex, bibtex, and repeated pdflatex passes;
latexmk is unavailable here. The final build has no undefined references,
undefined citations, or overfull/underfull box warnings. All pages were rendered
and inspected in contact sheets, with enlarged checks of the changed figure,
derivations, implementation list, and references. Two broken paragraphs were
kept together and the bibliography starts on a fresh page. Human feedback on
voice remains pending; this is a completed draft, not a claim of human acceptance.

The strongest alternative explanation for a future gain is improved proposal
adaptation or greater compute rather than the gradient identity. The proposed
ablations address that question. The weakest current scientific evidence is
the absence of an implemented corrected-mixture/Fisher comparison. Losing to
a tuned conventional score at equal compute would overturn its practical
priority without invalidating its algebra.

The final source/PDF archive and SHA-256 manifest accompany this note. Because
the live document directory is ignored on the main branch, the source is also
preserved under `final/` and in `final-manuscript.tar` here. No unrelated dirty
files, algorithms, session database, or provider settings were modified.

## Continuation

Read this note and `docs/plans/younis-score-analysis-recovery-2026-09-11.md`
first. The manuscript request is fulfilled; no research campaign is pending or
implicitly launched. Do not replay the 410 MB session or print nested tool
outputs. Use narrow source excerpts and saved result summaries. If execution
is requested, prepare the bounded study described in Section 12 with fresh
calibration/validation partitions; do not reuse consumed historical holdouts.
