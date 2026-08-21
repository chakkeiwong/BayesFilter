# Forward-Snowball Ledger

Access date: 2026-08-18. The bounded public search used OpenAlex `cites` queries
for the two cross-field seeds most likely to reveal a missed method family:
Nishimura--Dunson--Lu (W2795604964) and OccBin (W2120422722). Results were
ranked by OpenAlex citation count and screened for direct method relevance.
Counts and ranking are discovery aids only.

## Discontinuous HMC seed

Query: `filter=cites:W2795604964`, 50 results, ordered by `cited_by_count:desc`.
OpenAlex reported 60 citing records at query time; the seed record itself
reported 57 citations, showing normal live-index inconsistency.

| Citing work | Year | Class | Decision |
|---|---:|---|---|
| Zhou, *Mixed Hamiltonian Monte Carlo for Mixed Discrete and Continuous Variables* | 2020 | DIRECT_METHOD | Included and technically inspected. It removes ordinal embedding but requires genuine mixed support. |
| Torgander et al., *HMC with Categorical Parameters Using the Concrete Distribution* | 2024 | COMPETITOR | Included and technically inspected as a relaxation arm. |
| Grathwohl et al., *Oops I Took a Gradient: Scalable Sampling for Discrete Distributions* | 2021 | COMPETITOR | Omitted from main derivation; gradient-informed discrete MCMC, not an event-aware nonlinear state-space HMC method. |
| Nishimura et al. follow-on applications in phylogenetics and exact-zero priors | 2022--2023 | EMPIRICAL_EXAMPLE | Omitted; they show transfer to other discontinuous targets but not ZLB filtering. |
| *Zig-Zag Sampling for Discrete Structures and Nonreversible Phylogenetic MCMC* | 2022 | COMPETITOR | Omitted; specialized nonreversible topology sampling and no direct ZLB likelihood treatment. |
| *The Hastings Algorithm at Fifty* | 2019/2020 | SURVEY_OR_TUTORIAL | Omitted as technical support; historical context only. |

No forward-citing record in the inspected result set claimed a solved general
nonlinear ZLB state-space posterior. This is absence in a bounded index query,
not proof that no such paper exists.

## OccBin seed

Query: `filter=cites:W2120422722`, 50 results, ordered by
`cited_by_count:desc`. The result set is dominated by economic applications, so
method papers were prioritized over highly cited applications.

| Citing work | Year | Class | Decision |
|---|---:|---|---|
| Giovannini, Pfeiffer, Ratto, *Efficient and Robust Inference...* | 2021 | DIRECT_METHOD | Included and technically inspected; defines the PKF benchmark. |
| Aruoba et al., *Piecewise-Linear Approximations and Filtering...* | 2020/2021 | DIRECT_METHOD | Included and technically inspected; defines PLC/COPF bridge. |
| Holden, *Computation of Solutions to Dynamic Models with Occasionally Binding Constraints* | 2016 | COMPETITOR | Omitted from main derivation but retained as solver-comparison risk. |
| Holden, *Existence and Uniqueness of Solutions to Dynamic Models with Occasionally Binding Constraints* | 2022 | DIRECT_METHOD | Retained as a major target-definition risk; a later project phase must inspect it before making uniqueness claims. |
| Boehl, *Efficient Solution and Computation of Models with Occasionally Binding Constraints* | 2022 | COMPETITOR | Retained for later nonlinear/economic solver comparison; does not itself supply the HMC target. |
| Bayer et al., *Shocks, Frictions, and Inequality in US Business Cycles* | 2024 | EMPIRICAL_EXAMPLE | Omitted; recent high-visibility application, not an inference-kernel source. |

## Queries not run

The experiment plan capped public discovery. Separate forward-citation queries
were not run for every particle-filter paper, NeurIPS paper, or working paper.
Their forward coverage is therefore `not available`. Backward references,
direct title searches, local corpus search, and the two cross-field forward
queries provide useful but non-exhaustive coverage.

## Truncated/constrained UKF seeds (2026-08-19)

No systematic OpenAlex `cites` queries were run for the nine Section 4.3
sources; their forward coverage is `not available`. A bounded web search around
the truncated-UKF seed surfaced two citing works, recorded as discovery aids:

| Citing work | Year | Class | Decision |
|---|---:|---|---|
| Li, Wang et al., *Auxiliary Truncated Unscented Kalman Filtering for Bearings-Only Maneuvering Target Tracking* (Sensors) | 2017 | DIRECT_METHOD | Omitted; specialized tracking application of the truncated update with no state-space economics or ZLB content. |
| *A Novel Improved Truncated Unscented Kalman Filtering Algorithm* | 2014 | DIRECT_METHOD | Omitted; incremental TUKF variant located only through a non-archival aggregator page; not inspected. |

Neither record was inspected in full text and neither supports any claim in the
manuscript.

## Section 13/14 sources (2026-08-19)

No OpenAlex `cites` queries were run for the twenty shadow-rate,
censored-filtering, and OBC sources added with Sections 13 and 14; their
forward coverage is `not available`. The largest known forward risks are
recorded in `omitted_papers.md`: post-2016 shadow-rate estimation refinements,
post-2021 Tobit-filter extensions, and post-2023 OBC estimation work citing
Boehl--Strobel.
