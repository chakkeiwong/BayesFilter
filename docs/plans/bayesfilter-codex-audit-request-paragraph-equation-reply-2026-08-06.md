# Reply: paragraph and equation audit of the rewritten monograph

- **Date:** 2026-08-06
- **Requested artifact:** `docs/fable-rewrite/monograph/main.tex` and its active input closure
- **Reply status:** **REVISE before final release**
- **Audit boundary:** bounded release audit with reading-order coverage, active-root inventory, high-risk paragraph/equation review, clean rebuild, label/citation checks, policy scan, and MathDevMCP diagnostic attempts

## Executive verdict

The rewrite is a substantially better and more honest development branch than the previous canonical source. It builds from its current tracked inputs, resolves active citations and references, states several approximation and HMC boundaries correctly, and repairs real defects in the particle-filter, LEDH, GenUT, ICNN, finite-fallback, and squared-TT passages.

It is **not ready for final publication or immediate canonical promotion**. The remaining release blockers are:

1. the SR-UKF signed update/downdate derivative is explicitly not derived in-book;
2. the latest squared-TT retained-first contraction rewrite has not received a successful scalar/vector derivation certificate;
3. theorem-level source support remains open for the HAC and Li-Coates lanes;
4. the ICNN trainer remains schematic rather than fully specified;
5. the active bibliography needs one final source-strength and metadata curation pass;
6. the PDF still has substantial typesetting warnings and one old-TeX `amsmath` warning.

The appropriate status is a **buildable repair branch with qualified claims**, not a final monograph. The rewrite should be preserved and finished through a narrow blocker pass. Do not reject the research direction because these are localized release defects.

## Corpus and coverage

The active closure contains:

| Item | Count | Evidence |
|---|---:|---|
| Active chapter/appendix TeX inputs | 66 | `main.tex` closure and filesystem inventory |
| TeX source lines | 34,568 | `wc -l` over active chapters and appendices |
| LaTeX labels | 1,231 | direct label inventory |
| Display-math environments | 1,186 | direct environment inventory |
| Theorem/proposition/lemma/corollary/algorithm blocks | 113 | direct environment inventory |
| MathDevMCP structural blocks | 2,009 | `.mathdevmcp/latex_index.json` |
| MathDevMCP equation rows | 1,667 | `.mathdevmcp/latex_index.json` |

The standalone directory also contains four inactive predecessor chapters not included by `main.tex`: `ch34_highdim_gaussian_and_sparse_quadrature.tex`, `ch35_highdim_particle_transport_tensor_filters.tex`, `ch36_nonlinear_ssm_hmc_research_program.tex`, and `ch37_highdim_filtering_candidate_synthesis.tex`. Their duplicate labels are snapshot-maintenance debt, not active-root build failures. They should not be silently promoted or deleted during migration.

### Exhaustiveness limitation

I did not produce a durable row for every one of the 1,667 equation rows and every paragraph in this reply. Claiming that I had fully certified all 34,568 lines from the preserved artifacts would be unsupported. The classifications below cover the material load-bearing equations and paragraphs identified by the request and prior release reviews. Every unlisted paragraph or equation remains **not checked**, not implicitly approved.

## Reading-order findings

The following findings are ordered by monograph reading order. Each row gives the claim, evidence classification, why it matters, and the required repair.

| Location | Claim or paragraph | Classification | Finding and repair |
|---|---|---|---|
| `chapters/ch01_introduction.tex:62-75` | Every substantive claim should be traceable to a source map, literature artifact, code/test, or project decision. | correct | This is an appropriate claim discipline. The book should additionally state that the cited source map and supporting ledgers are not themselves proof of the mathematical claim. |
| `chapters/ch01_introduction.tex:80-95` | The Reader Map describes Parts I-XI in reading order. | correct | The map now matches the 11 active parts. It is a navigation statement, not evidence that the parts are pedagogically balanced. |
| `chapters/ch02_state_space_contracts.tex:4-163` | Structural coordinates must preserve stochastic, deterministic, lag, and measurement roles. | not checked | The contract is coherent at prose level, but the full state-transition and mask equations were not independently derived against every downstream implementation. Preserve explicit nonclaims for adapter fidelity. |
| `chapters/ch03_hmc_target_requirements.tex:9-23` | The transformed posterior adds prior, filtering likelihood, and `\log|\det DT(u)|`. | correct | Correct under a differentiable, square, locally one-to-one transform with the stated support convention. Add those transform assumptions to the definition; they are currently implicit. |
| `chapters/ch03_hmc_target_requirements.tex:25-30` | A deterministic nearby force can preserve the target under exact endpoint Metropolis correction. | correct | Correct only for the declared deterministic reversible, volume-preserving correction setting. Keep the distinction from wrong-value, stochastic, asymmetric, or unadjusted dynamics explicit. |
| `chapters/ch03_hmc_target_requirements.tex:35-45` | A finite invalid-region fallback defines a modified target and may be improper on an unbounded region. | correct | This is a real improvement. Do not describe the finite fallback as equivalent to exact support rejection. |
| `chapters/ch09_kalman_score.tex:23-43` | Prediction derivatives include all parameter-dependent transition, covariance, and initial-condition terms. | correct | The displayed first-order recursion is algebraically correct under differentiability and the declared fixed/stationary initial-condition branch. The stationary Lyapunov derivative is deferred rather than fully derived here. |
| `chapters/ch09_kalman_score.tex:48-92` | Innovation and solve-form score equations. | correct | The score equations are correct on an SPD innovation branch. Production trace estimation and factor solves remain implementation evidence, not proof of all backend paths. |
| `chapters/ch09_kalman_score.tex:97-145` | Gain, filtered mean, and covariance derivatives propagate the same covariance object. | correct | The covariance-form identities are correct for the stated update. Joseph and square-root alternatives require their own reconstruction checks; the text says this appropriately. |
| `chapters/ch10_kalman_hessian.tex:18-35` | The Hessian is summarized by abstract `T_{ij}` terms. | not checked | The solve derivative is plausible and source-shaped, but the abstract Hessian decomposition is not a complete equation-by-equation derivation in this chapter. Keep it as a contract/summary, not a finished proof. |
| `chapters/ch11_structural_derivatives.tex:13-202` | Structural maps provide first and second derivative providers. | not checked | The provider interface is internally readable, but no full code/source closure was checked for every provider and mask branch. Do not call every adapter source-faithful without anchors. |
| `chapters/ch12_factor_derivatives.tex:29-288` | QR and ordinary Cholesky reconstruction identities supply factor derivative calculus. | correct | Correct as a bounded smooth-branch contract. It does not supply the ordered signed rank-one update/downdate derivative required by all negative-weight SR-UKF branches. |
| `chapters/ch13_custom_gradient_wrappers.tex:4-15` | Same-scalar custom gradients and exact endpoint correction are distinct from changed-value or unadjusted dynamics. | correct | This repairs an earlier overbroad HMC claim. Keep exact endpoint value and force provenance separate in every later chapter. |
| `chapters/ch14_derivative_validation.tex:7-177` | Finite differences, autodiff, analytic scores, and production readiness form a validation ladder. | correct | The role separation is policy-compliant. Passing a finite-difference smoke does not establish posterior correctness or production readiness. |
| `chapters/ch17_square_root_sigma_point.tex:185-195` | Chapter 12 does not derive all signed update/downdate derivatives; negative-weight score admission is conditional. | correct | This is honest and should remain. It blocks an unqualified final-release claim until the route is either fully derived or excluded from the admitted score contract. |
| `chapters/ch17_square_root_sigma_point.tex:209-228` | Signed state covariance and innovation factor reconstruction. | not checked | The covariance decomposition is definitional and the factor reconstruction target is clear, but `\operatorname{SR}_{\mathcal B_t}` and its gauge/feasibility assumptions are not fully defined. |
| `chapters/ch17_square_root_sigma_point.tex:294-338` | The filtered factor must include positive stack, negative stack, and `K S K^T` downdate. | not checked | This is the correct target repair relative to the displayed covariance, but the signed factor operation itself and its derivative are not independently certified. |
| `chapters/ch19_particle_filters.tex:370-406` | No-resampling propagation carries previous normalized weights through the bootstrap increment. | correct | The rewrite fixes the earlier branch mismatch by defining post-decision weights and the `N w^*_{t-1} g_t` increment. A full measure-theoretic proof remains a human derivation obligation, not a MathDevMCP certificate. |
| `chapters/ch19_particle_filters.tex:410-569` | The bootstrap likelihood estimator is unbiased while its log is downward biased. | not checked | The claim is standard under the stated Feynman-Kac assumptions, but the full proof and carried-weight branch were not machine certified in this audit. Keep the assumptions and do not infer unbiased scores. |
| `chapters/ch19_particle_filters.tex:599-616` | ESS diagnoses weight concentration. | correct | This is a diagnostic identity. ESS does not certify target correctness, convergence, or superiority. |
| `chapters/ch19b_dpf_literature_survey.tex:547-564` | The LEDH affine offset uses centered information `eta - Lambda m`. | not checked | The rewrite is algebraically better aligned with the local centered Gaussian notation, but the MathDevMCP label audit abstained and the durable Li-Coates source copy remains an explicit blocker. Preserve as a reviewed hypothesis until the source/linear-case reduction is recorded. |
| `chapters/ch19b_dpf_literature_survey.tex:572-633` | Li-Coates local and actual PF-PF routes are distinct. | unsupported | The source-faithfulness boundary is correctly stated in prose, but the exact primary source and durable local copy are not closed in the rewrite release package. Do not promote algorithm-level attribution until that source closure exists. |
| `chapters/ch20_filter_choice.tex:12-79` | Filter routes are selected by target, derivative, numerical, and evidence status. | correct | The register is appropriately conditional. Avoid any later shorthand that turns a diagnostic or historical route into a production recommendation. |
| `chapters/ch28a_neural_network_state_space_model_applications.tex:457-484` | The growing Bartlett/HAC estimator is a practical direction, not a closed consistency theorem. | correct | The nonclaim is explicit. The displayed estimator is a valid definition, but theorem-level consistency remains unsupported until exact source assumptions are inspected. |
| `chapters/ch28a_neural_network_state_space_model_applications.tex:815-869` | The controlled predictive-validation run failed power/geometry gates without invalidating the research direction. | correct | This separates candidate failure from direction failure. Reported rates are descriptive unless the declared uncertainty procedure supports stronger claims. |
| `chapters/ch32c_entropic_ot_sinkhorn.tex:480-530` | Barycentric projection preserves means but generally contracts covariance under a chosen coupling. | not checked | The law-of-total-covariance identity is valid for the declared coupling; the local finite-coupling implementation and all covariance inequalities require assumptions and code-facing checks. |
| `chapters/ch32c_entropic_ot_sinkhorn.tex:1663-1734` | The GenUT axis construction matches whitened diagonal moments and maps back to physical mean/covariance. | correct | The local algebra is correct under `CC^T=P`, `k_a>s_a^2`, finite weights, and the declared whitened-coordinate rule. It is not a blanket claim that this is Ebeigbe's full multivariate construction. |
| `chapters/ch32c_entropic_ot_sinkhorn.tex:1887-1935` | Stopping mean/scale normalization yields a partial derivative, not the total derivative of the executed scalar. | correct | This is an important policy repair. State the held-constant lifted map explicitly whenever this is used operationally. |
| `chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex:2294-2310` | Contract E, exact divisor chunking, and per-scope tuning are canonical policy bindings. | correct | The policy identifiers and no-override semantics are now explicit. One final pass should ensure later benchmark prose does not weaken this statement. |
| `chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:287-341` | The ICNN objective is schematic; fixed `varphi` makes the target expectation constant in `theta`. | correct | The mathematical correction is right. The trainer remains schematic and should not be presented as a fully specified canonical implementation. |
| `chapters/ch33_highdim_nonlinear_filtering_foundations.tex:1130-1237` | Fixed-branch value/derivative paths and high-dimensional contracts are separated. | not checked | The ledger discipline is useful, but the full tensor/coordinate derivations and source-faithfulness claims were not checked equation by equation. |
| `chapters/ch34_highdim_gaussian_projection_and_point_rule_foundations.tex:1-332` | Gaussian projection, point rules, and sparse-grid foundations define the high-dimensional lane. | not checked | The chapter is an active foundation source, but this reply did not rederive all quadrature weights, point counts, and error claims. Treat those as not checked beyond existing bounded artifacts. |
| `chapters/ch35_highdim_sparse_grid_quadrature_and_fixed_cloud_scalar.tex:1-427` | Fixed sparse-grid scalar and validation interfaces are explicit. | not checked | The prose distinguishes diagnostics from promotion, but the full quadrature and fixed-cloud equations require a dedicated ledger. |
| `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:171-290` | With `r_t=(x_t,x_{t-1})`, right contractions leave the current block explicit and carry the `e^{-c_t}` defensive scale. | not checked | The latest source now uses the intended right-side orientation (`M_{>j}` and `C_1`), correcting the earlier reviewed mismatch. A scalar/vector contraction derivation and numerical identity check are still required before calling it release-closed. |
| `chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:348-390` | The saved retained evaluator consumes reference coordinates and owns no physical/reference Jacobian. | not checked | The interface is now explicit, but the code-facing coordinate conversion and density-measure convention were not independently audited. |
| `chapters/ch38_highdim_validation_defect_calculus_and_promotion.tex:253-321` | Veto indicators must pass before runtime/ESS/rank comparisons are interpreted. | correct | This is a governance/diagnostic contract, not a theorem about candidate quality. The thresholds remain project defaults that require provenance and scope. |
| `appendices/app_b_matrix_calculus.tex:8-30` | Inverse, log-determinant, and quadratic-form derivative identities. | correct | Correct under invertibility and symmetry/SPD assumptions. The appendix should state the symmetry condition more prominently before the quadratic-form identity. |
| `appendices/app_c_factor_derivative_proofs.tex:4-14` | The factor-proof appendix is a bounded placeholder. | correct | This accurately records a release gap. It prevents a claim that every factor derivative is proven in-book. |
| `appendices/app_d_mathdevmcp_workflows.tex:4-21` | MathDevMCP is an audit assistant, not an oracle. | correct | This complies with the policy and matches the observed abstentions. |
| `appendices/app_e_researchassistant_workflows.tex:41-55` | Literature-heavy prose needs accepted source support or explicit nonclaims. | correct | The policy is sound, but the appendix itself identifies unresolved HAC and Li-Coates support blockers. |

## Equation-by-equation audit of important formulas

This is the bounded equation ledger for the load-bearing formulas inspected in this pass. It is not a certificate for every displayed row in the 1,667-row MathDev index.

| Label/location | Computed object and claim | Classification | Evidence and required action |
|---|---|---|---|
| `ch03:15-21`, `def:bf-filtering-target` | Transformed posterior target | correct | Standard change-of-variables formula under differentiable locally one-to-one transform. Add explicit transform/support assumptions. |
| `ch09:25-37`, `eq:bf-score-predicted-mean-first` / `eq:bf-score-predicted-cov-first` | First-order prediction recursion | correct | Product rule is correct; stationary initial covariance branch remains source/provider-dependent. |
| `ch09:50-75`, `eq:bf-score-innovation-first` / `eq:bf-kalman-score-contribution` | Innovation and Gaussian score | correct | Correct on differentiable SPD innovation branch. |
| `ch09:80-88`, `eq:bf-solve-score` | Solve-form score | correct | Algebraically equivalent to the covariance inverse expression under `S_t` invertible. |
| `ch10:18-35`, `eq:bf-solve-hessian-w-dot` / `eq:bf-solve-hessian-summary` | Hessian solve and abstract decomposition | not checked | The solve derivative is source-shaped; `T_{ij}` terms are not fully expanded or independently certified in the chapter. |
| `ch12:41-49`, `eq:bf-factor-reconstruction-first` / `eq:bf-factor-reconstruction-second` | Factor-to-covariance reconstruction | correct | Correct identity; branch/gauge assumptions must be supplied by each factor implementation. |
| `ch17:310-317`, `eq:bf-srukf-filtered-factor` | Signed filtered covariance factor | not checked | Target equation is explicit, but `SR_{\mathcal B_t}` and signed downdate feasibility/derivative are not certified. |
| `ch17:329-335`, `eq:bf-srukf-filtered-factor-first` | Factor derivative reconstruction | not checked | Correct reconstruction form if the declared factor derivative exists; the derivative primitive is intentionally missing. |
| `ch19:418-428`, `eq:bf-pf-average-weight` / `eq:bf-pf-bootstrap-likelihood-estimator` | Bootstrap likelihood estimator | not checked | The estimator form is standard under the stated assumptions; full carried-weight proof remains a human derivation obligation. |
| `ch19b:547-564`, `eq:bf-pff-ledh-A` / `eq:bf-pff-ledh-b` | LEDH local affine flow | not checked | Centered offset is the intended local-Gaussian correction, but source-level Li-Coates closure and a preserved linear reduction are required. |
| `ch28a:459-484`, `eq:bf-ssl-lstm-growing-hac` / `eq:bf-ssl-lstm-hac-bandwidth` | HAC estimator and `N^{1/3}` bandwidth | correct | The formulas define a candidate estimator and bandwidth. Any asymptotic sufficiency claim remains unsupported. |
| `ch32c:501-518`, `prop:bf-eot-barycentric-covariance-contraction` | Covariance decomposition for a coupling | correct | Law-of-total-covariance structure is valid for the declared joint coupling; local OT-specific inequality/implementation claims need assumptions. |
| `ch32c:1677-1701`, `prop:bf-eot-genut-axis` | GenUT axis weights/locations | correct | Algebra follows under `k_a>s_a^2`; positivity of the central weight is not implied. |
| `ch32c:1703-1723`, `prop:bf-eot-genut-moments` | Whitened moment and physical covariance identities | correct | Correct for the explicitly defined local whitened-axis rule, not automatically the external paper's general multivariate object. |
| `ch32c:1887-1908`, `prop:bf-eot-stopped-normalization-partial` | Stopped normalization derivative | correct | Correct as a partial derivative under the artificial held-constant map; wrong relative to a total-gradient claim unless the target is changed explicitly. |
| `ch32c2:2294-2306` | Canonical route/policy identifiers | correct | Policy contract, not a mathematical identity. Must be issued/enforced by repository code, not caller text. |
| `ch32e:312-341`, `eq:bf-neural-ot-direct-icnn-objective` | ICNN direct-map objective | heuristic only | Explicitly schematic; fixed `varphi` makes the target-sample expectation constant in `theta`. Specify a trainer or keep this status. |
| `ch36b:184-215`, `eq:bf-hd-squared-tt-right-mass-recursion` / `eq:bf-hd-squared-tt-retained-numerator-contraction` | Retained-first squared-TT contraction | not checked | Latest right-contraction rewrite is directionally consistent with `(x_t,x_{t-1})`, but no scalar/vector derivation certificate exists. |
| `ch36b:221-277`, `eq:bf-hd-squared-tt-dot-right-mass-recursion` / `eq:bf-hd-squared-tt-dot-retained-numerator` | Same-branch derivative contraction | not checked | Product-rule form is plausible; fixed-branch coordinate, basis, and defensive-term dependencies need a dedicated derivation record. |
| `ch37:348-390`, `eq:bf-hd-ttkr-retained-P` / `eq:bf-hd-ttkr-retained-query-rule` | Normalized retained evaluator and reference query | correct | Quotient and quadratic-form evaluations are algebraically correct under nonzero `\widehat Z_t` and the declared reference measure; code handoff remains not checked. |
| `ch38:259-318`, `eq:bf-hd-veto-finite` through `eq:bf-hd-veto-defaults` | Veto indicators and thresholds | correct | These are explicit diagnostic/policy definitions. They do not prove that a candidate passes or that thresholds are scientifically optimal. |
| `app_b:8-25` | Matrix inverse/logdet/quadratic-form derivatives | correct | Correct with invertible/SPD and symmetry conditions. |

## Policy compliance summary

### Compliant or materially improved

- Exact target versus modified finite fallback is distinguished in `ch03`.
- Same-scalar HMC and exact endpoint correction are distinguished in `ch03` and `ch13`.
- Proxy diagnostics are generally not promoted to proof or superiority claims.
- The rewrite explicitly labels source-faithful, fixed-HMC-adaptation, and `extension_or_invention` boundaries in the high-dimensional/OT material.
- Contract E, total-gradient composition, exact-divisor chunk policy, and per-scope tuning are stated in `ch32c2`.
- NeuTra sections distinguish convergence vetoes from acceptance and descriptive diagnostics.
- Appendix D correctly says MathDevMCP is diagnostic-only.
- Appendix E correctly says raw bibliography entries and search results are not source support.

### Noncompliance or release gaps

- The SR-UKF negative-weight analytical derivative is discussed as an admitted possibility while its primitive remains unproved. The conditional boundary must be repeated in all summary/recommendation passages.
- `ch28a` still carries a theorem-adjacent HAC estimator discussion whose exact source assumptions are explicitly open; this is acceptable only because the text now labels it an audit direction.
- The monograph embeds live repository paths such as `/home/chakwong/python/src/dsge_hmc/...` and `/home/chakwong/BayesFilter/.research/...`. These are useful provenance anchors but are not portable publication references.
- Some sections use active project language such as `p50 lane`, `Phase 2B`, promotion, admission, and policy identifiers. This is not mathematically wrong, but it is reader-facing release debt and should be moved to a project-status appendix or glossary in a publication edition.
- The active bibliography is mechanically resolvable but still needs source-strength curation; metadata alone cannot support theorem claims.

## Language and readability findings

### Strengths

- The rewrite generally uses direct claim boundaries: “not a theorem,” “diagnostic only,” “not posterior correctness,” and “target changed.”
- It separates textbook formulas from implementation and promotion consequences more clearly than the old source.
- The revised ICNN, PF, HMC, GenUT, and HAC passages explain what is and is not being claimed.

### Required language repairs

1. Define `p50 lane`, `Phase 2B`, and similar internal identifiers or remove them from reader-facing prose.
2. Replace absolute local paths with stable source-map references or publication-neutral descriptions.
3. Split very dense paragraphs in `ch18b`, `ch28a`, `ch32c`, `ch32c2`, `ch33`, and `ch34`; line-level correctness is not enough if a reader cannot identify the target and assumptions.
4. Keep `schematic`, `heuristic`, `candidate`, and `extension_or_invention` labels adjacent to the display they qualify.
5. Do not use “source-faithful” without both the cited paper technical anchor and local author-source anchor required by `AGENTS.md`.
6. Remove or relocate live governance/reporting prose from the main textbook flow in a later editorial pass.

## Build and mechanical checks

Fresh isolated build:

```bash
cd docs/fable-rewrite/monograph
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/bayesfilter-rewrite-audit-request main.tex
```

Result:

- exit code: `0`
- PDF: 493 pages
- undefined references/citations: none detected in the final log
- active duplicate-label warnings: none detected
- overfull boxes: 193
- underfull boxes: 769
- remaining `amsmath` foreign-command warning: one `\\atopwithdelims` warning

This proves buildability only. It does not certify mathematical correctness.

## MathDevMCP checks run

MathDevMCP was run from `/home/chakwong/anaconda3/envs/tf-gpu/bin/mathdevmcp` against the current rewrite root and cached index for these labels:

- `eq:bf-srukf-filtered-factor`
- `eq:bf-hd-squared-tt-retained-numerator-contraction`
- `eq:bf-custom-primal-target`
- `eq:bf-ssl-lstm-growing-hac`
- `eq:bf-pf-bootstrap-likelihood-estimator`
- `eq:bf-pff-ledh-b`

All six returned `inconclusive` rather than `verified` or `mismatch`. The common reason was parser/provenance abstention and the absence of a safe, fully formalized obligation. The tool identified missing dimension/constraint formalization for some rows. These results are diagnostic evidence only; they do not establish that the equations are wrong, and they do not certify them as correct.

The existing `.mathdevmcp/latex_index.json` is useful for navigation, but its parser inventory also contains expected labels from inactive/legacy source context. It must not be treated as a global proof ledger.

## Source-support and literature status

The rewrite follows the correct local policy in stating that source metadata, raw `.bib` entries, and downloaded PDFs do not by themselves support theorem-level claims. The active source-support gaps remain:

- exact primary-source technical support for the HAC consistency boundary in `ch28a`;
- durable local technical source and exact bibliographic closure for Li-Coates in `ch19b`;
- full source-faithfulness anchors for any Zhao-Cui or external implementation claim that is stronger than the local derivation;
- a complete omitted-paper, backward-snowball, forward-snowball, retraction, erratum, and version-conflict audit for the whole bibliography.

No source was classified as retracted or quarantined in this bounded audit. That is not a claim that a comprehensive retraction audit has been completed.

## Rewrite instructions before final release

### Release blockers

1. **SR-UKF:** either derive the ordered signed update/downdate derivative, including branch feasibility and reconstruction, or state that the negative-weight analytical-score route is outside the admitted release contract.
2. **Squared TT:** preserve the latest right-contraction rewrite, then add a scalar and vector derivation/identity check proving that `(x_t,x_{t-1})`, `M_{>j}`, `C_1`, the retained evaluator, and the next-step query all use the same coordinate and measure convention.
3. **HAC:** keep the current nonclaim until the exact theorem assumptions are inspected and recorded in a source-support ledger.
4. **Li-Coates:** store a durable local source copy and reconcile the local `ch19b` equations with the inspected author algorithm and metadata.
5. **ICNN:** either specify the target-side potential update and complete trainer or label the whole section as schematic/heuristic only.
6. **Bibliography:** restore the strongest verified publication record for NeuTra or record an explicit policy reason for choosing the arXiv record; remove any unused or incomplete entries that are not needed by active prose.
7. **Typesetting:** fix the `\\atopwithdelims` warning, triage the largest overfull boxes, and inspect all figure pages for clipping and label legibility.

### Recommended audit artifacts

Create a persistent ledger under `docs/plans/` with one row per active structural block or equation row containing:

- file and line anchor;
- target object and assumptions;
- evidence class using the five allowed labels;
- source anchor or project derivation anchor;
- MathDevMCP status and exact command/artifact;
- repair action and reviewer status;
- explicit nonclaim.

The ledger may be paged by chapter, but it must preserve stable IDs and checksums. A future agent should be able to determine which 1,667 equation rows were actually reviewed without relying on prose such as “paragraph-by-paragraph audit completed.”

## Explicit nonclaims

This reply does **not** certify:

- every paragraph in the 34,568-line active closure;
- every one of the 1,186 display environments or 1,667 MathDev equation rows;
- every theorem, proposition, algorithm, or proof in the unchanged chapters;
- global mathematical correctness of the monograph;
- source-faithfulness of every literature or author-code claim;
- completeness of the bibliography or omitted-paper coverage;
- absence of retractions, errata, withdrawals, or version conflicts across all sources;
- HMC posterior correctness, convergence, production readiness, or scientific validity of any candidate;
- publication-quality typography after the current build.

## Final decision

**REVISE before final release.** The rewrite is a credible and materially improved repair branch. It should be kept, repaired narrowly, and reviewed again using a persistent paragraph/equation ledger. It should not yet be called a fully audited final monograph or promoted to canonical status on the strength of build success and label-scoped diagnostic checks alone.
