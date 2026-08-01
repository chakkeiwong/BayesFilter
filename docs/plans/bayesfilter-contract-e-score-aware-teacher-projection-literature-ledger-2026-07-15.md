# Score-Aware Teacher Projection Literature Ledger

Date: 2026-07-15

Decision: `BOUNDED_SUPPORT_ADEQUATE_FOR_DERIVATION_SECTION`

## Source support

| Source | Class | Local full text and status | Inspected technical anchors | Supported claim | Forbidden claim |
|---|---|---|---|---|---|
| Del Moral, Doucet, and Singh, *Uniform Stability of a Particle Approximation of the Optimal Filter Derivative* | `DIRECT_METHOD` | `.localresources/papers/delmoral-doucet-singh-filter-derivative.pdf`; valid 27-page arXiv/technical-report full text; identical to arXiv `1106.2525` fetched 2026-07-15 | Section 2, especially Eq. (2.5), the backward representation and Algorithm 1; references inspected for the Poyiadjis/Cerou lineage | The filter derivative is the derivative of filtering expectations; the conditional path-score representation; the backward particle recursion uses pairwise parent-child sums and avoids the path-space degeneracy mechanism analyzed there | Does not prove BayesFilter's teacher coreset, feature projection, positivity chart, Contract E integration, or HMC readiness |
| Poyiadjis, Doucet, and Singh, *Particle Approximations of the Score and Observed Information Matrix...* | `FOUNDATIONAL`, source-blocked locally | Both `.localresources/papers/poyiadjis2011-score-observed-information*.pdf` files are HTML block/error pages, not PDFs | Not inspected directly in this task | Bibliographic lineage/context only through the inspected Del Moral paper | No theorem, equation, algorithm, or implementation claim is directly attributed to an inspected Poyiadjis full text |
| BayesFilter score-aware teacher-projection section | `PROJECT_DERIVATION` | `docs/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex` | Positive finite coreset proposition; fixed active-set smooth-chart proposition; retained-feature tangent proposition; one-step value/score proposition | Exact statements relative to a finite teacher, fixed feature span, and fixed positive chart | No universal nonlinear correctness, NAWM feasibility, global HMC chart, or canonical/default claim |
| 2D LGSSM TensorFlow reference witness | `IMPLEMENTATION_EVIDENCE` | Script and JSON artifact under `docs/benchmarks`; float64 CPU reference | 6561-point teacher, frozen seven-point positive chart, autodiff/FD, Kalman comparator | Numerical feasibility and residuals for the declared two-observation fixture | Does not exercise LEDH migration/PF-PF correction and is not GPU/XLA or production evidence |

Publication metadata for Del Moral et al. was checked through Crossref on
2026-07-15: SIAM Journal on Control and Optimization 53(3), 1278--1304,
DOI `10.1137/140993703`. Crossref reported 14 citing records. That count is a
dated coverage signal, not validity evidence. The publisher full-text endpoint
returned a block page; technical anchors therefore refer explicitly to the
inspected arXiv/technical-report version.

## Snowball and omission audit

Backward snowballing of the inspected Del Moral paper identified the direct
lineage through Cerou--Le Gland--Newton (linear tangent filtering), Poyiadjis et
al. (2005, 2009, 2011), and Doucet--Tadic (particle parameter estimation). The
chapter needs only the definition/recursion anchor, so these were classified as
lineage rather than additional theorem support.

Forward snowballing was not expanded for this bounded derivation section. The
new contribution is not a claim that the score literature lacks compression
methods, and no novelty or literature-completeness claim is made. A publication
or full literature-review version would need a dedicated search for particle
coresets, positive cubature reduction, optimal quantization, and differentiable
moment-constrained compression.

## Claim support and gaps

| Claim | Support class | Status |
|---|---|---|
| Tangent filtering measure is evaluated weakly through derivatives of feature expectations | `PRIMARY_TECHNICAL_SUPPORT` | Supported by inspected Del Moral et al. Section 2 and Eq. (2.5) |
| Backward particle tangent recursion has pairwise parent-child work | `PRIMARY_TECHNICAL_SUPPORT` | Supported by inspected Algorithm 1 |
| A finite positive teacher admits a coreset with at most the number of retained features | `PROJECT_DERIVATION` | Proved in the chapter by support-reduction linear dependence |
| A frozen positive nonsingular active set gives a local differentiable chart | `PROJECT_DERIVATION` | Proved by openness of nonsingularity/positivity and differentiated linear solve |
| Retaining the exact next finite predictive contribution preserves its log value and score | `PROJECT_DERIVATION` | Proved as an identity on the fixed chart |
| The displayed 2D LGSSM construction works numerically | `IMPLEMENTATION_EVIDENCE` | Passed with structured JSON and autodiff/FD evidence |

Top omission risk: related positive cubature/coreset literature may provide a
more standard name, stronger global existence conditions, or better algorithms.
That omission does not invalidate the elementary finite-support proof, but it
blocks a novelty claim and should be closed before publication.

What is not concluded: scholarly completeness, novelty, exact nonlinear score
preservation, structural-NAWM feasibility, HMC readiness, or Contract E--TP
canonical/default status.

