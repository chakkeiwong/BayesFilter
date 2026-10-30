# Primary-source support for the M13 marginal correction proposal

Access and inspection: 2026-10-04. This is a bounded method audit, not a survey
or novelty claim. ResearchAssistant was not available among the session tools.
Primary papers were downloaded locally and their technical sections inspected.

## source_support

| Source and role | Local full text and status | Inspected support | Allowed and forbidden uses |
|---|---|---|---|
| Li & Coates (2017), direct baseline method | .localresources/papers/m13-repair-20261004/li-coates-2017.pdf and .txt; arXiv v5 author version of IEEE TSP 65(16), 4102–4116 | Algorithm 1; equations (19)–(20); Lemma IV.2, Theorem IV.3, Lemma IV.4; particle-specific UKF covariance and auxiliary flow | Supports the baseline conditional importance correction and conditional affine invertibility argument. Does not describe the proposed marginal correction or prove its finite-design score accuracy. |
| Klaas, de Freitas & Doucet (2005), direct proposed mechanism | .localresources/papers/m13-repair-20261004/klaas-defreitas-doucet-2005.pdf and .txt; UAI 2005, 308–315, author-hosted copy | Section 3, Figure 3, equations (3)/(5); Section 3.1, equation (8), Proposition 1 | Supports marginalizing the predictive and proposal mixtures and the random-label total-variance argument. Does not justify transferring that comparison to arbitrary deterministic or stratified allocations. |
| Owen & Zhou (2000), optional safeguard | .localresources/papers/m13-repair-20261004/owen-zhou-2000.pdf; JASA 95(449), 135–143; author-hosted copy | Importance-sampling law/variance discussion; Section 3 equation (10), Section 3.1; distinction from control-variate equations (13)–(15) | Supports the basic defensive-mixture inequality, including the squared-mean term. Stronger control-variate guarantees are not claimed for this proposal. |

Source URLs:
- https://arxiv.org/abs/1607.08799
- https://www.cs.ubc.ca/~arnaud/klass_defreitas_doucet_marginalparticlefilterUAI2005.pdf
- https://artowen.su.domains/reports/seis.pdf

The Owen–Zhou PDF's text extraction was garbled. Pages 1–11 were rendered;
pages 4–11 have local OCR text, and the relevant equations were inspected.
OCR is an aid, not an independent authoritative transcription. The proposal
derives its defensive inequality explicitly.

The Li–Coates author publication page was stored at
.localresources/software/m13-repair-20261004/coates-publications.html.
It pointed to networks.ece.mcgill.ca/software. The HTTPS retrieval failed
with connection refused (curl exit 7). Search found third-party replications,
not a verified replacement author implementation. Author-code parity is
therefore NOT CHECKED. The current repository call chain was inspected instead.
No claim of unchanged author-code faithfulness is made for the extension.
No specific correction/retraction was found in the inspected primary records
and search; this is not exhaustive clearance.

## citation_venue_metadata

| Paper | Venue information | Citation count / venue ranking | Source and caveat |
|---|---|---|---|
| Li–Coates | IEEE Transactions on Signal Processing 65(16), 2017 | Not queried | Primary manuscript/publication page; venue is bibliographic information, not correctness evidence. |
| Klaas et al. | UAI 2005 proceedings | Not queried | Author PDF; later arXiv upload date is not the publication date. |
| Owen–Zhou | JASA 95(449), 2000 | Not queried | Author PDF; no citation-based selection or superiority claim. |

Counts/rankings are unnecessary for this bounded derivation and were not
invented. A comprehensive literature review would need a separate coverage
budget and appropriate metadata source.

## backward_snowball

| Lead from inspected seeds | Role | Action |
|---|---|---|
| Gordon et al. (1993), bootstrap filtering | Foundational/practitioner comparator | Constructed the actual same-design bootstrap comparator in the prior M13 diagnostic; do not claim a new paper audit here. |
| Pitt & Shephard (1999), auxiliary filtering | Related ancestor/proposal selection | Identified as context; no auxiliary-filter performance theorem imported. |
| Hesterberg (1995), defensive mixtures | Foundational safeguard | Owen–Zhou technical discussion inspected; basic inequality rederived locally. Original paper not separately audited. |
| Veach & Guibas (1995), multiple importance sampling | Related weighting strategies | Context for possible alternative allocation/weight designs; no theorem imported. |
| Daum–Huang flow references from Li–Coates | Baseline flow origins | Current proposal retains the existing specified flow; no new flow derivation or source-faithfulness claim. |

## forward_snowball

Primary-source search on 2026-10-04 found Lai et al. (2022), “Variational
Marginal Particle Filters,” PMLR 151. Its metadata establishes a relevant
later direction; its technical sections were not inspected, so it supports
no proposition in this proposal. Search also identified Elvira et al.,
“Generalized Multiple Importance Sampling,” Statistical Science 34(1), 2019
(arXiv:1511.03095), as a relevant weighting/allocation follow-up.
No citation ranking was obtained and no claim to have covered all recent
or highly cited work is made.

## claim_support

| Claim | Authority checked | Limit |
|---|---|---|
| M13's current score follows its finite likelihood while that likelihood can miss the reference badly | October 4 localization and preceding FD/score reports; current consumer → fused route → analytical kernel inspected | Eight development points; no global derivative guarantee. |
| The conditional M13 map is affine in the current innovation | Auxiliary zero-noise initialization and shared Euler updates; Li–Coates method anchors | Depends on preserving that implementation structure; new route wiring not yet implemented. |
| Marginal weights integrate the finite predictive mixture | Proposal Proposition 1, finite sum/integral proof; MPF section 3 | Conditional sampling law, not frozen-design or full-recursion unbiasedness. |
| Random-label variance cannot increase | Proposal Proposition 2; MPF Proposition 1; checked conditional expectation and nonnegative two-component expression | Does not apply automatically to current fixed/stratified allocation. |
| Universal stratified non-deterioration is false | Explicit two-state counterexample, exact rational arithmetic and MathDevMCP equalities | Refutes a universal claim, not the usefulness of the M13 candidate. |
| The score formulas are the total derivative of the proposed finite likelihood | Displayed chain rules, inverse-map and Gaussian derivations; symbolic identities and 2×2 diagnostic | Conditional on all upstream and reset/correction tangents; no implemented endpoint verification yet. |
| Defensive variance bound includes a mean term | Direct inequality in proposal; Owen–Zhou section 3 | No control-variate theorem or general non-harm guarantee. |
| Existing LGSSM/KSC computations can be preserved exactly | Identity by unchanged execution | An implementation obligation. No new consumer regression was executed in this documentation stage. |

## omitted_paper_risks

The strongest omission risk is that another multiple-importance weighting or
allocation scheme offers a better guarantee for stratified particles than
the proposed MPF weighting. Elvira et al. and the Veach–Guibas lineage
should be inspected if this candidate fails or if comparative optimality
is proposed. The present proposal claims neither novelty nor optimality,
and explicitly supplies a counterexample to the overly broad guarantee.

Variational MPF is a plausible later competitor, but adding learned proposals
would change this campaign's method and tuning burden; it is not silently
included. Hesterberg's original treatment is worth adding before a dedicated
defensive-mixture campaign. Original-author flow code remains a source gap.
None of these omissions affects the displayed elementary identities; they
limit source-faithfulness, completeness and comparative-method claims.
