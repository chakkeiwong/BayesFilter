# General covariance preservation: bounded literature review

Date: 2026-10-07. Checkout: sqmc-development, base 209223fdd.
Question: can a general covariance-preserving ensemble transform improve our
filter, with SIR serving as a test model rather than defining the algorithm?

## Decision and scope

Acevedo/de Wiljes/Reich supplies a general resampling requirement: preserve the
mean and covariance represented by normalized importance weights. It supplies
neither an independent posterior covariance oracle nor a universal choice of
covariance for an ancestor-specific LEDH flow. Analytic SIR mixture moments
remain an independent diagnostic, not the proposed general reset target.
No implementation, experiment, default change, or new campaign was executed.

Skeptical audit: moment equality is a resampling criterion, not a likelihood or
score promotion criterion. Importance-weight coverage, finite-ensemble support,
regularization and downstream higher-moment corrections remain separate risks.
A UKF-versus-Q comparison cannot be decided by this paper. Numerical parameters
from its experiments must not be transferred as justified defaults.

## Source-support ledger

Primary source: Walter Acevedo, Jana de Wiljes, Sebastian Reich,
Second-order accurate ensemble transform particle filters, local supplied PDF:
.localresources/papers/acevedo-dewiljes-reich-second-order-1608.08179.pdf.
Technical sections inspected: 2-6 (transform and correction), 7 (experiments,
including the negative support example), 8 (conclusion); no relevant appendix.

- Definition 3.1, (11)-(13): empirical mean/covariance match importance-weighted
  moments, using denominator N, not N-1.
- (16)-(25), (28)-(31), (42)-(44): transform constraints and Riccati correction.
- Section 6: Sinkhorn transform followed by covariance correction.
- (58): hybrid filters split the likelihood between their two updates.
- Sections 7.1-7.2: Lorenz-63/Lorenz-96 filtering demonstrations; endpoints
  include RMSE and CRPS, with rejuvenation/localization/hybrid choices.
- Section 7.3, Figures 6-7: the second-order corrected ETPF produces some samples
  outside prior support in the bounded SceneWalk parameter example.
- No likelihood-score accuracy guarantee was located in the inspected paper.
  Second-order here means moments, not second-order dynamics accuracy.

Own derivation, with particles as columns of X:
  mu = X w; P = X (W - w w^T) X^T; W = diag(w).
  D 1 = N w; D^T 1 = 1; Y = X D.
  Cov(Y) = X (D - w 1^T)(D - w 1^T)^T X^T / N.
A sufficient corrected-transform condition is
  (Dhat - w 1^T)(Dhat - w 1^T)^T = N (W - w w^T).
Writing B = D - w 1^T and Dhat = D + Delta, a symmetric zero-sum correction
satisfies
  B Delta + Delta B^T + Delta^2 = N (W - w w^T) - B B^T.
This is the paper's (31)/(42)/(43) in N notation. A converged solution preserves
weighted mean/covariance; a numerical implementation must measure residuals.
Negative transform entries can move particles outside the original convex hull.

For a nonnegative transport coupling pi_ij = D_ij/N, the weighted source is
X_I and the barycentric output is E[X_I | J]. Total covariance gives
  P_weighted = Cov(E[X_I | J]) + E[Cov(X_I | J)].
Thus barycentric transport loses a positive-semidefinite covariance term.
This algebra explains the reset effect, not the accuracy of the input weights.

## Citation-metadata ledger

Local version: arXiv v3, 10 April 2017. Official arXiv version history checked.
Published: SIAM Journal on Scientific Computing 39(5), A1834-A1850 (2017),
DOI 10.1137/16M1095184.
Primary records:
https://arxiv.org/abs/1608.08179
https://epubs.siam.org/doi/abs/10.1137/16M1095184
https://publications.imp.fu-berlin.de/2073/
Focused official/source search found no correction or retraction; this is not an
exhaustive negative finding. Local docs/references.bib entry near line 1306 has
incorrect author/journal metadata; it was not used as authority or edited.
No citation-count or venue-ranking argument is being made.

## Backward-snowball ledger

References checked in this paper for their role, not independently fully audited:
[21] Reich (2013), original ETPF; [25] Toedter/Ahrens (2015), NETF;
[16] Lei/Bickel (2011), moment matching; [6] Chustagulprom/Reich/Reinhardt
(2016), hybrid ensemble transform filter; [11] Frei/Kuensch (2013), EnKF/PF
bridging; [7] Cuturi (2013), Sinkhorn computation. These identify relevant
families; implementation or promotion requires inspecting the selected source.
No claim of a comprehensive survey.

## Forward-snowball and code ledger

Focused exact-title, DOI, follow-up, erratum and author-code searches located
the original paper and official records. No verified original-author code
repository or substantively audited follow-up was located in this bounded
search. This does not establish their absence. ResearchAssistant was not
available among callable tools. No author-code faithfulness claim is made.

## Claim and implementation ledger

Static source inspection:
ledh_canonical_score_tf.py -> ledh_canonical_reset_score_tf.py ->
batched_sinkhorn_contract_e_reset_triple_with_tangent in ledh_unified_reset_tf.py.
The shared reset constructs target_mean and target_cov from weighted children
(lines 180-196); it uses residual injection, ridges and Cholesky recolouring.
This targets the same moments but is not the paper's Riccati implementation.
The final empirical covariance after regularization and subsequent higher-moment
repair has not been verified by an executable parity test in this review.
No call-chain conformance certification or final exactness claim follows.

| Decision | Primary criterion | Veto / uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Use paper as resampling benchmark | Weighted moment identities derived | Numerical residuals untested | Specify residual tests for current reset/candidates | Current code implements this paper exactly |
| Keep SIR exact moments as diagnostics | Exact local conditional check | Not generally available | Include among independent model checks | General method requires a SIR oracle |
| Separate proposal and reset design | Covariance-loss algebra identifies reset effect | Missing modes/poor weights remain | Define general proposal policy in later repair plan | Moment correction repairs likelihood/score |
| Treat hybrids as further design | Paper (58) splits likelihood | Gaussian update requires score derivation | Review separately if selected | Full likelihood can be applied twice |

## Omission risks and next action

A full survey would need independent technical/code audits of NETF,
moment-matching filters, regularized/resample-move methods and hybrid EnKF/PF
filters, plus later support-preserving variants. This review answers the named
paper question only. Implementation needs a separate plan, bounded budget and
evidence contract covering proposal coverage, final moment residuals, support,
likelihood and all analytical scores across model scopes. No new experiment
budget has been assigned here.
