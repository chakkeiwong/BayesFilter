# Source and omission record

Access date: 2026-09-11. Purpose: support the recovered model-score analysis.
This is a bounded technical synthesis, not an exhaustive citation census.
No citation count or venue indicator determines the recommendation.

## Source support

| Source / role | Inspected technical anchors | Local full text and status | Permitted use and boundary |
|---|---|---|---|
| Younis–Sudderth, Differentiable and Stable Long-Range Tracking of Multiple Posterior Modes / DIRECT_METHOD | Section 4, (14)–(17); Section 5 objective; Appendix B.1; method-comparison paragraphs and references | `.localresources/papers/younis-sudderth-2023-long-range-tracking.pdf` and `.txt`; valid PDF. NeurIPS 2023 paper; local arXiv v1 dated 2024-04-12 matches the currently retrieved arXiv version history. No withdrawal notice observed on that page. | Fixed-proposal mixture-weight derivative identity; discriminative objective; stated all-pairs gradient cost. Does not establish unbiased finite-N generative model scores or correctness of OT/GenUT adaptations. |
| Younis–Sudderth, Learning to be Smooth / DIRECT_METHOD | Sections 2.3 and 4, (6)–(7), (17)–(23); training objective and computational requirements; smoothing references | `.localresources/papers/younis-sudderth-2024-learning-to-be-smooth.pdf` and `.txt`; valid PDF. Published NeurIPS 37 (2024), DOI `10.52202/079017-0228`, confirmed through Crossref. No correction relationship returned there; comprehensive errata check unavailable. | Forward/backward mixture proposal, explicit generative ratio versus learned numerator, and discriminative state loss. No proof that its learned smoother is the BayesFilter generative posterior or supplies complete transition scores. |
| Lai–Domke–Sheldon, Variational Marginal Particle Filters / DIRECT_METHOD | Section 4, Algorithm 2, Theorem 2 and proof sketch, (9)–(13); biased/unbiased variational-gradient discussion | `.localresources/papers/lai-domke-sheldon-2022-variational-marginal-particle-filters.pdf` and newly extracted `.txt`; PMLR 151:875–895 official page retrieved. No correction notice observed on page. | Full-mixture weights and unbiased normalizer for the specified MPF; local Rao–Blackwell comparison only. Does not guarantee a global variance ranking, unbiased log-score, or validity after an added deterministic reset. Supplement is not completely re-audited. |
| Ścibior–Wood, Differentiable particle filtering without modifying the forward pass / COMPETITOR | Section 2.1, (1); Section 4.2, (24)–(25); discussion of fixed-forward/custom gradients and consistency | `.localresources/papers/scibior-wood-dpf-forward-pass.pdf` and `.txt`; arXiv `2106.10314v2`, 2021-10-19. Current publication/correction status not independently refreshed. | Distinguish Fisher-score estimators from fixed-program derivatives; support the ancestor-averaged recursion. No claim that it makes normalized particle scores unbiased at finite N. Author implementation not inspected in this recovery; no implementation recommendation copied from it. |
| Del Moral–Doucet–Singh, filter derivative stability / FOUNDATIONAL | Section 2, (2.1)–(2.5); Section 3, Assumption A and Theorems 3.1–3.2 | `.localresources/papers/delmoral-doucet-singh-filter-derivative.pdf` and `.txt`; valid PDF. Current bibliographic status not refreshed. | Context for conditional-expectation derivatives and restrictive stability assumptions. Do not transfer time-uniform variance theorems to unrestricted Gaussian, degenerate, or nonlinear repository models; appendix proofs were not re-audited. |
| Poyiadjis–Doucet–Singh (2011), score/information particle approximation / FOUNDATIONAL, recovered during manuscript revision | Section 2.1 Fisher identity; Section 2.2 Algorithm 2, equations (12)–(22); Section 2.3 Gaussian evaluation; Theorem 1 assumptions and Appendix forgetting bounds | Valid author PDF now stored at `.localresources/papers/poyiadjis-doucet-singh-2011-author-recovered-20260911.pdf` and `.txt`, from https://www.stats.ox.ac.uk/~doucet/poyiadjis_doucet_singh_particlescoreparameterestimation.pdf . Earlier two invalid HTML caches remain excluded. | Direct primary support for the marginal score recursion. The linear-in-time variance discussion is a stability conjecture plus numerical evidence; Theorem 1 concerns a genealogical lower bound under specified assumptions. No universal finite-N unbiasedness or variance ranking is imported. |

The Poyiadjis problem is a cache-integrity failure, not evidence against the
paper. No source has been identified as retracted or scientifically
quarantined. Absence of an observed notice is not a complete retraction search.

## Author code

| Source | Verified anchor | What was checked |
|---|---|---|
| Younis official repository, local commit `b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18` | `src/models/kde_particle_filter/kde_particle_filter.py:738–748,750–790,927–934` | Fixed sampled positions, full mixture log-density gradient injection, raw multiplication before normalization; separate implicit/importance hybrid branch. This was source inspection, not execution or an audit of every caller. |
| VMPF official repository, commit `7c5970d11ffb63c5231965d9ceb8f0df70292003` | `algorithm/variational_marginal_particle_filter.py:53–61`; cached as `sources/vmpf-author.py` | Main filtering loop evaluates transition mixture and subtracts actual proposal density. Official PMLR page links this repository. No claim of BayesFilter backend, derivative, or API conformance. |

The componentwise hybrid formula in the report is a PROJECT_DERIVATION. The
author's hybrid branch uses implicit mixture reparameterization and is not
the same construction. Neither has been locally benchmarked by this recovery.

## Citation and venue metadata

| Source | Metadata and date | Limits |
|---|---|---|
| Younis 2023 | Current arXiv page retrieved 2026-09-11; NeurIPS 2023 identity also appears in Younis 2024 reference 14 | OpenAlex search returned HTTP 429; citation count unavailable. No venue ranking fetched. |
| Younis 2024 | Crossref work retrieved 2026-09-11: NeurIPS 37, 2024; `is-referenced-by-count=0` | This is Crossref's coverage-specific value, not evidence of zero actual citations. No venue ranking fetched. |
| Lai 2022 | Official PMLR page retrieved 2026-09-11 confirms authors, pages, year, PDF, supplementary ZIP, and author code | Citation count and venue ranking unavailable. |
| Other sources | Existing local bibliography/full texts | Current citation and ranking metadata not available and not needed for the algebraic conclusions. |

The cached metadata pages and hashes are listed in `source-inventory.json`.
Network calls downloaded bounded files and printed selected fields. No raw
metadata result was used as a mathematical authority.

## Backward snowball

| Seed references | Classification | Action and reason |
|---|---|---|
| Younis 2023 [1–4], 2024 [17–20]: classical particle filtering and surveys | FOUNDATIONAL / SURVEY_OR_TUTORIAL | Retain ordinary SMC as a constructed comparator; no need to re-survey textbook mechanics for this recommendation. |
| Younis 2023 [6], 2024 [16]: Ścibior–Wood | COMPETITOR | Inspect technical score recursion and separate it from fixed-program differentiation. |
| Younis 2023 [8], 2024 [12]: Corenflos et al. entropy-regularized OT | COMPETITOR | Already central to repository; local full text available, not freshly re-audited. No new convergence/consistency theorem asserted. |
| Younis 2023 [17–21], [34]: regularized PFs, kernel estimation, bandwidth selection | FOUNDATIONAL / BACKGROUND | Recognize smoothing-bias and bandwidth alternatives; defer rate claims until target-specific assumptions are checked. |
| Younis 2023 [22], 2024 [28]: Kantas et al. parameter-estimation survey | SURVEY_OR_TUTORIAL | Important coverage follow-up; not used as theorem support. |
| Younis 2023 [30–32]: DiCE, Gumbel-softmax, Concrete | COMPETITOR | Distinguish score-function/custom-gradient estimators from continuous relaxations. Replacing one relaxation with another is not itself a model-score bias repair. |
| Younis 2023 [35–39]: pathwise/implicit mixture gradients | COMPETITOR | Include hybrid alternative; derive the componentwise identity locally rather than transfer variance claims. |
| Younis 2023 [41–42]: Klaas MPF, Lai VMPF | DIRECT_METHOD | Inspect Lai technical proof/formula and author code. Original Klaas source not separately inspected; no original theorem claim. |
| Younis 2024 [22–27]: Bresler, Kitagawa, Doucet–Johansen, Klaas, Briers, Rauch–Tung–Striebel | FOUNDATIONAL / COMPETITOR | Include forward/backward proposals, model-corrected smoothing, and analytic Gaussian reference; detailed smoother selection deferred. |
| Younis 2024 [32–34]: resampling comparison, resampling survey, Liu–Chen | FOUNDATIONAL / BACKGROUND | Include stratification/conditional integration as efficiency options; no universal variance ordering asserted. |
| Computer-vision architectures, localization datasets, tracking applications in both seeds | EMPIRICAL_EXAMPLE / PERIPHERAL | Useful evidence for the papers' own tracking tasks, not evidence for this model-score target. Omitted from method selection. |

## Forward snowball

The Younis 2024 paper explicitly cites the 2023 paper and is technically
inspected as a direct follow-up. The fresh OpenAlex query for the 2023 work
returned HTTP 429, so a current citing-works traversal could not be completed.
Crossref's record for the 2024 paper does not provide a usable citing-works
census. No claim is made that recent competitors or corrections are exhausted.
Older saved-session citation counts were not presented as fresh metadata.

## Claim support

| Report claim | Support class | Anchor |
|---|---|---|
| Positive unbiased likelihood does not imply unbiased log-score | PROJECT_DERIVATION | Exact two-point counterexample in report; rational-arithmetic check in `verification.json` |
| IWSG differentiates a mixture expectation with a fixed proposal | PRIMARY_TECHNICAL_SUPPORT plus explicit derivation | Younis 2023 (14)–(15); report integral identity |
| KDM can be a proposal without adding its kernel to the target numerator | PROJECT_DERIVATION | Conditional importance integral in report; support condition explicit |
| Specified MPF normalizer is unbiased; local variance statement has limited scope | PRIMARY_TECHNICAL_SUPPORT | Lai Theorem 2, (9)–(12), adjacent qualification; author weight formula |
| Fisher-score recursion includes the initial law and all ancestors | PROJECT_DERIVATION; PRIMARY_TECHNICAL_SUPPORT for related recursion | Report joint-integral and backward-conditional derivation; Ścibior (24)–(25) |
| All-IWSG can have variance proportional to inverse squared bandwidth | PROJECT_DERIVATION | Single-Gaussian affine-function example; exact moment check in `verification.json`; no universal full-filter rate claimed |
| Centered control variates retain baseline expectation | PROJECT_DERIVATION | Linearity of expectation, fixed/independent beta and valid centering |
| Componentwise hybrid expectation identity | PROJECT_DERIVATION | Differentiate sum of component expectations; categorical term retained |
| Previously tested Phase 4B candidates lose in stated scopes | IMPLEMENTATION_EVIDENCE from recovered result | Saved branch result and its paired intervals; raw run artifacts not re-audited or rerun here |
| Initial-law tangent limitation exists at recovered commit | IMPLEMENTATION_EVIDENCE | `git show 804616e3:bayesfilter/highdim/ledh_canonical_score_tf.py`, lines 199–202; reported fixed-initial-cloud test scope |
| This combined new proposal is the best empirical method | SOURCE_GAP_BLOCKER | Not claimed; no new experiments exist |

## Omitted-paper risks and hostile review

| Potential omission | Why it matters | Current disposition / next action |
|---|---|---|
| Original Poyiadjis paper | Direct precedent for the proposed score recursion | Source recovered and technical method inspected during manuscript revision; the earlier access gap is closed. Local implementation and equal-compute performance remain unevaluated. |
| PaRIS/forward smoothing and Nemeth score approximations | Possible alternatives to exact quadratic ancestor sums | PaRIS arXiv 1412.7550v1 was inspected during manuscript revision: Section 3.1 Algorithm 2, Section 3.2.4 Theorem 10, Assumption 2. Expected linear work needs mixing/acceptance conditions. Nemeth and broader alternatives remain deferred. |
| Coupled particle smoothers and unbiased MCMC; Rhee–Glynn/randomized multilevel methods | Relevant if literal finite-sample unbiasedness is essential | Report only the conditional telescope identity; coupling, rate, and expected-work requirements are unproved for this target. |
| FIVO/VSMC/IWAE gradient literature | Distinguishes variational-objective gradients from generative scores | Lai's objective (13) supplies the needed present distinction; original papers needed for a broader optimization survey. |
| Stein control functionals and learned baselines | Could give useful exact-mean controls | No implementation selected; boundary terms and centering cannot be assumed. |
| Recent citing papers or corrections | May change novelty or practical priority | Forward metadata access limited. Do not claim novelty or exhaustive literature coverage. |

The skeptical review finds the main recommendation defensible as an untested
research design. It rejects stronger claims of unbiased finite-N scores,
universal Rao–Blackwell improvement, a drop-in derivative substitution, an
established best method, or production readiness. The remaining author-code and forward citation-coverage gaps must remain visible in a future implementation
plan. Reviewer unavailability is not being used as an execution gate.

## Manuscript revision update, 11 September 2026

The original Poyiadjis source gap has been closed with a valid author-hosted PDF. PaRIS was also inspected at the method and complexity-theorem level; the old Section 3.6 assertion that it is inherently quadratic is wrong. The two-author attribution for Ścibior and Wood was corrected against the retained arXiv version. Lai’s given name is Jinlin, verified against the paper and official proceedings bibliography. New source hashes and the completed manuscript are recorded under `../younis-manuscript-update-20260911/`. The earlier failed download and forward-search observations remain historical records, not current source-access verdicts.
