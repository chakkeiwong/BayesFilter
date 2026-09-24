# Forward-snowball ledger

Forward search was performed through OpenAlex DOI records and the saved
Poyiadjis citing list. The public search endpoint returned transient 502/503
errors, so this is a bounded forward pass rather than an exhaustive census.
Citation counts are metadata only.

| Forward direction | Observed source or family | Decision relevance | Status |
|---|---|---|---|
| Poyiadjis → Kantas review | *On Particle Methods for Parameter Estimation…* (2015; OpenAlex cited-by count 351) | Consolidates later score, smoothing, iterated-filtering, and PMCMC branches and points to the omitted primary papers | Inspected locally as a review; primary branches checked selectively |
| Poyiadjis → PaRIS | Olsson & Westerborn, *The PaRIS algorithm* (2017) | Direct online smoothing alternative with a linear-cost and variance-growth theorem | Inspected locally; add explicit experiment arm |
| Poyiadjis → SQMC | Gerber & Chopin, *Sequential quasi-Monte Carlo* (2015) | Integration variance reduction and randomized coupling; no automatic derivative or support correction | Existing matrix row; source/conditions need explicit score-specific tests |
| Poyiadjis → nested/multilevel particle methods | Forward list includes nested and multilevel particle-filter work | Distinguishes estimator variance from discretization bias and allows telescoping only with a valid coupling | Add only after score-level coupling identity is derived |
| Differentiable PF/OT → later differentiable resampling and variational methods | Corenflos, Younis, Lai, and related differentiable-PF citations | Shows that continuous/OT methods often change the finite particle measure or optimize a variational objective | Current target registry covers the distinction; direct source-by-source audit remains needed |
| Iterated filtering → plug-and-play likelihood methods | Later POMP and simulation-based inference literature | Relevant when DSGE transition densities or derivatives are unavailable | Add as a separate derivative-free comparator, not as an exact-score route |
| Twisted PF → Guarniero--Johansen--Lee iAPF | *The Iterated Auxiliary Particle Filter* (JASA 2017), DOI 10.1080/01621459.2016.1222291 | Iterative future-data twisting is a proposal/normalizer-variance mechanism; it does not add a score identity | Add to Phase 2 only as a UKF/LEDH proposal-quality arm, subject to support and correction checks |
| Twisted PF → controlled/auxiliary SMC | Later controlled-SMC and proposal-learning literature | Proposal design can reduce normalizer variance while preserving a correction | Candidate Phase 2/4 proposal arm; score effect must be measured against fixed estimator |
| PaRIS → particle smoothing implementations | Later online smoothing, backward-sampling, and ancestor-sampling variants | Provides practical alternatives to full path storage and generic backward pairs | Add only variants whose transition support and target are explicit |

## Forward-search limitations

No forward result was used to claim that the program is bibliographically
complete. In particular, recent papers on coupled particle filters, weak/measure-
valued derivatives, constrained or manifold particle smoothing, and learned
controlled SMC remain an inspection queue. Their absence is a coverage risk,
not evidence that they solve the singular DSGE score problem.
