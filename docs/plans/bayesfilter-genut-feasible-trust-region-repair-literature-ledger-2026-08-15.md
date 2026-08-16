# GenUT Feasible Trust-Region Repair Literature Ledger

Date: 2026-08-15

Scope: finite positive equal-weight particle-cloud higher-moment correction,
differentiable value/JVP execution, feasibility diagnostics, and stable local
least-squares correction.

## 1. Source-Support Ledger

| Source | Class | Local source | Technical anchors inspected | Claims allowed |
|---|---|---|---|---|
| Ebeigbe et al. (2025), Generalized unscented transformation for forecasting non-Gaussian processes | DIRECT_METHOD | docs/Generalized unscented transformation for forecasting non-Gaussian processes Ebeigbe(25).pdf | Secs. III--V; Eqs. (19)--(32); Algorithms 1--2 | Classical GenUT matches selected diagonal moments under its conditions; constrained points can lose exact kurtosis while retaining lower-order accuracy |
| Easley and Berry (2021), A Higher Order Unscented Transform | DIRECT_METHOD / COMPETITOR | .localresources/genut_feasible_trust_region_repair_20260815/easley_berry_2021_hosput.pdf and .txt | Sec. 4; Definition 4.1; Theorem 4.2; Corollary 4.3; Remark 4.4; Algorithm 4.1 | HOUT uses approximate rank-one tensor decompositions and variable signed-weight quadrature with controlled moment error; conditioning can worsen as tolerance shrinks |
| Marquardt (1963), An Algorithm for Least-Squares Estimation of Nonlinear Parameters | FOUNDATIONAL | DOI metadata only; published primary text not locally retrieved | Citation and algorithm identity checked; technical full text not locally available | Historical attribution for LM nonlinear least squares only |
| Osborne (1976), Nonlinear Least Squares--The Levenberg Algorithm Revisited | FOUNDATIONAL / IMPLEMENTATION | .localresources/genut_feasible_trust_region_repair_20260815/osborne1976_levenberg_revisited.pdf and .txt | Sec. 3; implementation steps (iii)--(viii); Notes (ii)--(iv) | Column scaling and comparable damping are established implementation guidance |
| Johnson and Lowe (1979), Bounds on the Sample Skewness and Kurtosis | FOUNDATIONAL | DOI/OpenAlex metadata; publisher PDF blocked | Bibliographic record only; technical proof not inspected | Historical context for sample-moment bounds; not used as proof support |
| Pearson (1916), mathematical moment inequality | BACKGROUND | Cited by Ebeigbe; full local source not retrieved | Ebeigbe discussion around Eq. (32) | Context for the necessary kurtosis/skewness inequality; BayesFilter derives the used inequality directly |

No source is known to provide a drop-in positive equal-weight differentiable
reset for the existing Contract E cloud. The selected route is therefore
explicitly a new finite composition, not a claimed rediscovery of a published
algorithm.

## 2. Citation/Venue Metadata Ledger

Metadata lookup date: 2026-08-15.

| Source | DOI/arXiv | Metadata source | Citation count | Caveat |
|---|---|---|---:|---|
| Ebeigbe et al. | 10.1103/PhysRevE.111.054135 | local PDF and OpenAlex | OpenAlex count not used in claims | primary local PDF inspected |
| Easley--Berry | arXiv:2006.13429; 10.1137/20M135546X | OpenAlex and local arXiv PDF | OpenAlex returned 0 for DOI record | green open-access copy inspected |
| Marquardt | 10.1137/0111030 | OpenAlex/Crossref | 30,554 OpenAlex result | no local full text |
| Osborne | 10.1017/S033427000000120X | OpenAlex and Cambridge PDF | metadata not used for truth | open technical PDF inspected |
| Johnson--Lowe | 10.1080/00401706.1979.10489785 | OpenAlex/Crossref | 31 OpenAlex; 21 Crossref | closed primary text |

Counts and venue metadata are coverage signals only and were not used to
promote the method.

## 3. Backward-Snowball Ledger

| Seed | Relevant references considered | Action |
|---|---|---|
| Ebeigbe et al. | Julier--Uhlmann UT/GenUT predecessors; Pearson; Ponomareva et al. HOSPUT; Straka et al.; Rezaie--Eidsvik; Easley--Berry | UT predecessors retained as historical context; HOSPUT/HOUT inspected or classified; no direct positive equal-weight route found |
| Easley--Berry | Julier--Uhlmann; Kolda tensor decompositions; De Lathauwer HOPM; Kofidis--Regalia rank-one approximation; HOUT source algorithm | HOUT technical source inspected; tensor-decomposition papers classified as supporting machinery, not direct particle-reset alternatives |
| Osborne | Levenberg, Marquardt, Morrison, Beale | Osborne technical implementation inspected; Marquardt retained as foundational citation |

## 4. Forward-Snowball Ledger

OpenAlex bounded searches on 2026-08-15 nominated higher-order unscented
transforms, moment-matching sigma points, and nonlinear least-squares papers.
No forward paper was technically inspected and promoted beyond the sources
above. The ResearchAssistant provider campaign was blocked by a
Semantic-Scholar response-envelope error; this is recorded as a metadata
coverage limitation, not as evidence that no later method exists.

## 5. Claim-Support Ledger

| Chapter/result claim | Support class | Anchor |
|---|---|---|
| Classical GenUT matches selected diagonal moments only under stated conditions | PRIMARY_TECHNICAL_SUPPORT | Ebeigbe Secs. III--V, Algorithms 1--2 |
| Constrained GenUT can lose exact kurtosis | PRIMARY_TECHNICAL_SUPPORT | Ebeigbe Sec. V, Algorithm 2 and discussion |
| HOUT uses variable signed-weight rank-one construction and has tolerance/conditioning tradeoff | PRIMARY_TECHNICAL_SUPPORT | Easley--Berry Sec. 4, Theorem 4.2, Remark 4.4, Algorithm 4.1 |
| Pearson lower moment inequality | PROJECT_DERIVATION plus Ebeigbe context | Chapter Eq. bf-eot-hm-pearson-bound |
| Equal-weight necessary upper bound k <= N-1 | PROJECT_DERIVATION | Chapter Eq. bf-eot-hm-finite-n-kurtosis-bound |
| Column-scaled LM is established numerical practice | PRIMARY_TECHNICAL_SUPPORT | Osborne Sec. 3 and Marquardt attribution |
| New finite composition computes the same value/JVP program | PROJECT_DERIVATION and IMPLEMENTATION_EVIDENCE | Chapter score proposition; 36 focused tests; GPU replay |

## 6. Omitted-Paper Risk Register

| Candidate/area | Omission reason | Reviewer risk | Next action |
|---|---|---|---|
| Liu--West regularized particle filtering | changes the particle proposal/teacher rather than repairing the local reset | Moderate | evaluate only if weight/proposal regularization becomes the next hypothesis |
| Ensemble transform particle filters and second-order ETPF | already covered elsewhere in ch32c; lower-order transport baseline, not fourth-moment differentiable reset | Low | retain as comparator, not substitute |
| HOSPUT and related higher-order sigma-point rules | considered through Ebeigbe references and HOUT | Moderate | no positive equal-weight drop-in identified |
| Rank-constrained moment problems / semidefinite moment optimization | no checked local primary source in this bounded campaign | High for a publication-level completeness claim | run a dedicated literature campaign before publication claims |
| PaRIS/particle score estimators | address smoothing/score estimation rather than deterministic equal-weight moment reset | Low for this repair | outside current numerical failure scope |

## Literature Boundary

This ledger is sufficient to ground the current engineering repair and its
nonclaims. It is not a proof of literature completeness and does not support a
claim that the local composition is globally novel or superior.
