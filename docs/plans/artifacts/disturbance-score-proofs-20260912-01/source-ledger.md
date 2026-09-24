# Source ledger

Date: 2026-09-12

| Source | Local copy and inspected anchors | Used for | Boundary |
|---|---|---|---|
| Murray, Jones, Parslow (2013), *On Disturbance State-Space Models and the Particle Marginal Metropolis-Hastings Sampler* | `.localresources/papers/murray-jones-parslow-2013-disturbance-state-space.txt`, author manuscript dated 25 October 2018; Sections 1--2, transition-density motivation lines 42--61, disturbance target and change-of-variables discussion lines 93--127, APF construction lines 189--249 | Supports expressing a model through shocks and a deterministic state map, and using `p(u)/q(u)` in disturbance particle weights | Does not prove a low-variance score estimator or solve DSGE regime nonsmoothness. DOI metadata saved in `murray-doi-metadata.json` confirms publication in 2013, volume 1(1), pages 494--521; manuscript date is not publication date |
| Scibior and Wood (2021), *Differentiable Particle Filtering without Modifying the Forward Pass* | `.localresources/papers/scibior-wood-dpf-forward-pass.txt`; Section 3.1/equation (13) and Section 4.1; downloaded source page `scibior-arxiv.html` | Supports correcting derivatives for parameter-dependent discrete resampling and reparameterized proposals with a stop-gradient weight ratio | The correction does not make a finite-particle normalized log-score exact |
| Foerster et al. (2018), *DiCE: The Infinitely Differentiable Monte Carlo Estimator* | `.localresources/papers/foerster-2018-dice.txt`; Theorem 1 and equation (4.7); score-function/pathwise discussion | Supports separating score-function and pathwise terms and using a baseline only when its expectation-preserving condition is known | A generic baseline theorem does not establish a zero-mean baseline for this DSGE target; the manuscript instead derives the exact required integral |
| Poyiadjis, Doucet, Singh (2011), *Particle Approximations of the Score and Observed Information Matrix for State Space Models* | `.localresources/papers/poyiadjis-doucet-singh-2011-author-recovered-20260911.txt` and the companion PDFs | Provides the regular-transition Fisher/backward-recursion context that motivated the earlier route | Its ambient transition-density recursion is not a valid replacement for a lower-dimensional deterministic transition |
| BayesFilter canonical score implementation, inherited decision context | `bayesfilter/highdim/ledh_canonical_score_tf.py`, callable around lines 715--770 | Context for distinguishing a finite-program derivative from the model score | No consumer call-chain audit or implementation admission is performed in this document task; the new propositions make no implementation-correctness claim |

## Locally derived results

The nine new propositions and corollary are explicit project derivations,
not asserted novel results imported from these papers. They establish the
disturbance Fisher identity under dominated differentiation, importance
correction, an induction for the unnormalized particle measure, a ratio-bias
counterexample, exact control-variate centering and its variance formula,
a reference-model control, conditional integration, and a moving-line
Gaussian example. The reference model must use the same disturbance
coordinates and carry the derivative-density ratio; an approximate Gaussian
score alone is not an exact control. Exact algebra is checked separately in
`symbolic-verification.json` and `verification.json`.

## Coverage note

The source files above are local copies of primary or author-available texts.
The web search/open service returned HTTP 503/502 during this pass, so this
ledger does not claim an exhaustive literature survey. A direct DOI content
negotiation request succeeded and supports the Murray bibliographic fields.
Technical claims rely on the inspected local primary texts and displayed
derivations. No official-code implementation is reproduced or claimed faithful
here; no new algorithm implementation or empirical method comparison ran.
