# Source ledger

Date: 2026-09-12

| Source | Local copy and inspected anchors | Used for | Boundary |
|---|---|---|---|
| Murray, Jones, Parslow (2013), *On Disturbance State-Space Models and the Particle Marginal Metropolis-Hastings Sampler* | `.localresources/papers/murray-jones-parslow-2013-disturbance-state-space.txt`; transition-density motivation lines 42--61; disturbance target and change-of-variables discussion lines 93--127; APF construction lines 189--249 | Supports expressing a model through shocks and a deterministic state map, and using `p(u)/q(u)` in disturbance particle weights | Does not prove a low-variance score estimator or solve DSGE regime nonsmoothness |
| Scibior and Wood (2021), *Differentiable Particle Filtering without Modifying the Forward Pass* | `.localresources/papers/scibior-wood-dpf-forward-pass.txt`; Section 3.1/equation (13) and Section 4.1; downloaded source page `scibior-arxiv.html` | Supports correcting derivatives for parameter-dependent discrete resampling and reparameterized proposals with a stop-gradient weight ratio | The correction does not make a finite-particle normalized log-score exact |
| Foerster et al. (2018), *DiCE: The Infinitely Differentiable Monte Carlo Estimator* | `.localresources/papers/foerster-2018-dice.txt`; Theorem 1; score-function/pathwise discussion; baseline discussion | Supports separating score-function and pathwise terms and using a baseline only when its expectation-preserving condition is known | A generic baseline theorem does not establish a zero-mean baseline for this DSGE target |
| Poyiadjis, Doucet, Singh (2011), *Particle Approximations of the Score and Observed Information Matrix for State Space Models* | `.localresources/papers/poyiadjis-doucet-singh-2011-author-recovered-20260911.txt` and the companion PDFs | Provides the regular-transition Fisher/backward-recursion context that motivated the earlier route | Its ambient transition-density recursion is not a valid replacement for a lower-dimensional deterministic transition |
| BayesFilter canonical score implementation | `bayesfilter/highdim/ledh_canonical_score_tf.py`, callable around lines 715--770 | Establishes that the current analytical score differentiates the executed finite LEDH value program and initializes state sensitivities from the supplied cloud | It does not establish the exact DSGE observed-data score, stationary initial-law derivative, or disturbance-coordinate support identity |

## Coverage note

The source files above are local copies of primary or author-available texts.
The web search service returned HTTP 503/502 during this pass, so this ledger
does not claim an exhaustive literature survey. No external search result is
used as evidence for the decision. The exact finite checks are in
`verify_score_identities.py` and `verification.json`.
