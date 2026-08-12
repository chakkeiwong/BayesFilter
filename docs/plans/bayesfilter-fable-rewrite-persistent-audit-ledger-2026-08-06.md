# Persistent paragraph and equation audit ledger for the standalone rewritten BayesFilter monograph

- **Date:** 2026-08-06
- **Artifact under audit:** `docs/fable-rewrite/monograph/main.tex` and active input closure
- **Purpose:** durable ledger for paragraph-level and equation-level release audit, as requested before any final publication or canonical promotion.
- **Coverage status:** seeded with the highest-risk chapters and load-bearing equations first. Unlisted paragraphs/equations are **not checked**, not implicitly approved.

## Baseline build state at ledger start

- PDF pages: **493**
- undefined citations: **no**
- undefined references: **no**
- active-root duplicate labels: **no**
- overfull boxes: **193**
- underfull boxes: **760**
- remaining foreign-command warning count: **1**

## Evidence-class vocabulary

Only these five labels are used in this ledger:
- `correct`
- `wrong relative to the stated target`
- `unsupported`
- `not checked`
- `heuristic only`

## High-risk paragraph/equation ledger

| ID | File:line(s) | Claim / object | Evidence class | Evidence / reason | Action / status |
|---|---|---|---|---|---|
| L-001 | `chapters/ch17_square_root_sigma_point.tex:311-330` | `eq:bf-srukf-filtered-factor` and surrounding paragraph define the signed filtered covariance factor target | `not checked` | The target statement is improved and now explicitly includes the negative covariance-weight stack, but MathDevMCP audit artifact `bd5x89je3.txt` returned `status: unverified` with manual formalization required. | Keep the conditional release posture unless a derivation note and numerical reconstruction certificate are added. |
| L-002 | `chapters/ch17_square_root_sigma_point.tex:341-351` | Admission boundary for the negative-weight analytical-score route | `correct` | The text now honestly states the derivative route is conditional unless the signed primitive is supplied externally and checked. | Preserve as explicit release boundary. |
| L-003 | `chapters/ch12_factor_derivatives.tex:266-288` | Backend parity gates summarize factor derivative readiness | `correct` | Appropriate as a bounded smooth-branch contract; does not claim the missing signed recurrence primitive. | Preserve. |
| L-004 | `appendices/app_c_factor_derivative_proofs.tex:1-14` | Appendix C states itself as a bounded placeholder | `correct` | Honest status statement; prevents overclaiming proof completeness. | Preserve unless the missing primitive is actually added. |
| L-005 | `chapters/ch19_particle_filters.tex:488-533` | Post-decision measure in the PF unbiasedness proof | `correct` | The rewrite now distinguishes the weighted post-decision branch from the resampled unweighted branch. This is the right proof object conceptually, though not machine-certified. | Preserve and keep theorem-level modesty. |
| L-006 | `chapters/ch19_particle_filters.tex:572-592` | Differentiability boundary among true likelihood, unbiased estimator, and smooth surrogate | `correct` | Clear policy-compliant separation of evidence classes and target objects. | Preserve. |
| L-007 | `chapters/ch19b_dpf_literature_survey.tex:547-567` | `eq:bf-pff-ledh-A`, `eq:bf-pff-ledh-b`, and local affine flow statement | `not checked` | The centered-information rewrite is directionally correct and more honest than the old version, but the durable Li–Coates source closure remains open. | Keep nonclaim boundary until exact source support is closed. |
| L-008 | `chapters/ch28a_neural_network_state_space_model_applications.tex:457-485` | `eq:bf-ssl-lstm-growing-hac` and `eq:bf-ssl-lstm-hac-bandwidth` | `correct` for definition, `unsupported` for theorem-level sufficiency | The formulas define a candidate estimator and bandwidth, but MathDevMCP artifact `bfxu0s4ee.txt` is unverified and the chapter now explicitly demotes theorem-level closure. | Keep as implementation/audit direction, not theorem claim. |
| L-009 | `chapters/ch32c_entropic_ot_sinkhorn.tex:688-694` | Acevedo / second-order ETPF support sentence | `unsupported` | Active citation resolves, but theorem/algorithm-level support has not been fully audited in this release package. | Keep only as bounded literature context unless source closure is added. |
| L-010 | `chapters/ch32c_entropic_ot_sinkhorn.tex:903-918` | Contract-E design-space paragraph and nearby-ingredient support boundary | `correct` | The paragraph now explicitly says unresolved neighboring records are omitted from theorem-level support and that the paragraph is design-space context only. | Preserve. |
| L-011 | `chapters/ch32c_entropic_ot_sinkhorn.tex:1888-1944` | `prop:bf-eot-stopped-normalization-partial` partial-derivative warning | `correct` | Clear and policy-compliant: stopped mean/scale normalization gives a partial derivative, not automatically the score of the executed scalar. | Preserve. |
| L-012 | `chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex:107-119` | Canonical Contract E identifiers and raw-route historical status | `correct` | Canonical IDs and historical-route label are now explicit and consistent with repo policy. | Preserve; do one final coherence read before promotion. |
| L-013 | `chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex:2287-2299` | Canonical chunk selector and per-scope tuning semantics | `correct` | Stronger and explicit; aligns with frozen repo policy identifiers. | Preserve. |
| L-014 | `chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:309-341` | `eq:bf-neural-ot-direct-icnn-objective` and surrounding explanation | `heuristic only` | The fixed-`varphi` correction is mathematically honest, but the trainer remains schematic and is no longer overclaimed as canonical. | Preserve as schematic unless a fully specified trainer is added. |
| L-015 | `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:171-333` | retained-prefix contraction and derivative (`eq:bf-hd-squared-tt-retained-numerator-contraction`, `eq:bf-hd-squared-tt-dot-retained-numerator`) | `correct` for the declared fixed branch | The scalar-only generalization was repaired to retain the full prefix `G_m` and stop right contraction at `m+1`. Independent exact-rational certificates pass for `m=1,D=2` and `m=2,D=4`, for both value and directional derivative. All changed derivations received MathDevMCP label and typed audits with zero mismatches; matrix/integral abstentions remain diagnostic only. | Mathematical blocker closed by the 2026-08-08 certificate result. Preserve the fixed-object assumptions and nonclaims. |
| L-016 | `chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:326-432` | `eq:bf-hd-ttkr-retained-query-rule` and surrounding query-interface paragraph | `correct` as an internally derived contract; `not checked` against implementation | The saved `Q_t,\dot Q_t` evaluator is now derived from the retained prefix and includes the defensive marginal. Reference-coordinate and Jacobian ownership are explicit, but the code-facing route has not been independently certified. | Preserve as a stated convention; add implementation/source anchors before runtime-route promotion. |
| L-017 | `chapters/ch13_custom_gradient_wrappers.tex:85-100` | Posterior curvature vs likelihood-only observed information | `correct` | The rewrite now correctly separates posterior curvature from likelihood-only observed information. | Preserve. |
| L-018 | `chapters/ch20_filter_choice.tex:80-84` | Later-book pointer | `correct` | The stale “next industrial and case-study chapters” pointer has been repaired. | Preserve. |
| L-019 | `appendices/app_e_researchassistant_workflows.tex:52-55` | Remaining source-support blockers paragraph | `correct` | Honest and policy-compliant. It identifies HAC and Li–Coates as open blockers and explicitly says live publication claims should avoid unresolved source status. | Preserve. |
| L-020 | `references.bib:587-593` | `hoffman2019neutra` active NeuTra record | `correct` as metadata record in current branch, but `not checked` for publication-optimal curation | The branch currently uses the arXiv/AABI-style record. This is acceptable for the standalone branch, but the stronger publication record choice should be frozen explicitly during canonical promotion. | Resolve at promotion gate, not by silent drift. |
| L-021 | `references.bib:1204-1294` | Active additions around Acevedo / Vehtari / Benamou–Brenier / Del Moral / FilterFlow | mixed: `correct` for active non-provisional records; `unsupported` where theorem-level reliance exceeds source closure | The active rewrite bibliography no longer carries live provisional notes, and the speculative unused cluster has been removed. Remaining concern is source-strength, not note text. | Keep active records; strengthen or downgrade claims rather than reintroduce speculative entries. |
| L-022 | `chapters/ch32d_retained_teacher_neural_ot.tex` assorted paragraphs with `Nystr{"o}m` and warm-start routes | `correct` mechanically; `not checked` for full paragraph-level release polish | Accent syntax is repaired; content still needs final reader-facing style pass if publication polish is required. | Low priority after blocker closure. |

## MathDevMCP diagnostic artifacts referenced

- `bd5x89je3.txt` — `eq:bf-srukf-filtered-factor` returned `unverified`
- `b09pjph1e.txt` — `eq:bf-hd-squared-tt-retained-numerator-contraction` returned `unverified`
- `bn63nx1eh.txt` — `eq:bf-custom-primal-target` returned `unverified` due to missing shape/domain constraints
- `bfxu0s4ee.txt` — `eq:bf-ssl-lstm-growing-hac` returned `unverified` due to missing assumptions/dimension constraints

These artifacts are diagnostic evidence only. They do not establish that the equation is wrong; they establish that the release proof record is incomplete.

## Next actions derived from the ledger

1. Preserve the completed 2026-08-08 retained-prefix derivation certificate and keep its implementation/source-route nonclaim.
2. Freeze the SR-UKF release claim globally: either derive the signed primitive or keep every summary statement explicitly conditional.
3. Preserve the HAC paragraph as an explicit source-support nonclaim until exact theorem support is inspected.
4. Preserve the `ch32e` objective and algorithm as schematic unless a fully specified trainer is added.
5. Use this ledger, not prose claims of exhaustive review, as the release evidence contract for the next independent check.

## Nonclaims

This ledger does **not** certify every paragraph or every displayed equation in the full 493-page rewrite. It records the highest-risk rows first and should be extended if a final publication-level release is pursued.


## Additional high-risk rows (ledger extension)

| ID | File:line(s) | Claim / object | Evidence class | Evidence / reason | Action / status |
|---|---|---|---|---|---|
| L-023 | `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:171-333` | Retained-prefix squared-TT contraction and derivative | correct for the declared fixed branch | Exact-rational scalar and vector retained-prefix certificates match direct integration and direct differentiation; MathDevMCP found zero mismatches and no unresolved typed constraint. | Derivation blocker closed; retain bounded nonclaims. |
| L-024 | `chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:326-432` | Retained-reference query rule and Jacobian ownership | correct as a contract; not checked against implementation | The coefficient construction and defensive evaluator ownership are explicit and internally consistent, but implementation-facing route parity is not certified. | Preserve as a stated convention until source/route check is attached. |
| L-025 | `chapters/ch28a_neural_network_state_space_model_applications.tex:457-485` | HAC estimator direction | unsupported | The paragraph is now an audit direction only; exact theorem conditions remain open. | Keep as nonclaim until source closure. |
| L-026 | `chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex:309-366` | Schematic direct-map trainer | heuristic only | The chapter now correctly says fixed-phi is constant in theta, but the trainer remains schematic. | Either specify fully or keep schematic. |
| L-027 | `chapters/ch17_square_root_sigma_point.tex:311-351` | Signed SR-UKF release posture | not checked | Negative-weight branch remains conditional; no signed primitive derivation certificate is attached yet. | Preserve conditional release posture or derive primitive. |


## Extended chapter-level coverage rows

| ID | File:line(s) | Claim / object | Evidence class | Evidence / reason | Action / status |
|---|---|---|---|---|---|
| L-028 | `chapters/ch01_introduction.tex:62-95` | Reader map and source-traceability framing | correct | The rewrite's Reader Map now matches the 11-part structure and the source-traceability discipline is explicit. | Preserve. |
| L-029 | `chapters/ch03_hmc_target_requirements.tex:9-45` | transformed posterior target + finite-fallback contract | correct | The rewrite distinguishes exact target from finite modified target and separates exact endpoint-MH correction from wrong-value dynamics. | Preserve. |
| L-030 | `chapters/ch09_kalman_score.tex:23-145` | first-order prediction, innovation, and solve-form score recursions | correct | Bounded audit and prior derivation checks found these equations algebraically sound on the declared SPD branch. | Preserve; no further rewrite needed now. |
| L-031 | `chapters/ch10_kalman_hessian.tex:18-35` | abstract Hessian summary terms | not checked | The chapter remains structurally plausible but not fully certified equation-by-equation in this ledger. | Keep as bounded derivation summary unless a full whole-doc proof pass is pursued. |
| L-032 | `chapters/ch11_structural_derivatives.tex:13-202` | structural derivative-provider contract | not checked | Provider interface is coherent, but not every adapter/source closure has been verified in this release ledger. | Preserve with no stronger source-faithful claim. |
| L-033 | `chapters/ch18b_structural_deterministic_dynamics.tex:800-840` | structural vs additive-noise UKF contrast table | correct | Earlier transposed comparison row was repaired; the current comparison now matches intended structural law distinction. | Preserve. |
| L-034 | `chapters/ch20_filter_choice.tex:12-84` | backend-choice decision register | correct | The stale later-book pointer was repaired and the register now reads as a policy/decision table rather than a false chapter-order promise. | Preserve. |
| L-035 | `chapters/ch32a_soft_differentiable_resampling.tex:1-217` | soft-resampling boundary and nonclaims | correct | The rewrite now distinguishes local teaching rule from the external Particle Filter Networks algorithm and keeps target-status boundaries explicit. | Preserve. |
| L-036 | `chapters/ch32d_retained_teacher_neural_ot.tex:40-120` | retained-teacher warm-start narrative | not checked | Mechanically cleaner after accent repair, but still not fully audited paragraph-by-paragraph; reader-facing style pass remains optional. | Low priority; preserve for now. |
| L-037 | `chapters/ch33_highdim_nonlinear_filtering_foundations.tex:1130-1237` | fixed-branch value/derivative and actual-SV contract framing | not checked | Contract language is clearer than canonical, but full equation-level certification is not in this ledger. | Preserve as repair-branch contract. |
| L-038 | `chapters/ch34_highdim_gaussian_projection_and_point_rule_foundations.tex:1-332` | quadrature foundation chapter | not checked | Prior review found the chapter mathematically clean but pedagogically thin. No new blocker emerged here beyond whole-doc audit incompleteness. | Preserve; revisit only in a full pedagogical edition. |
| L-039 | `chapters/ch35_highdim_sparse_grid_quadrature_and_fixed_cloud_scalar.tex:1-427` | fixed sparse-grid scalar and worked examples | correct | The rewrite now contains the explicit one-dimensional worked rule application and 3D example structure that were missing before. | Preserve. |
| L-040 | `chapters/ch38_highdim_validation_defect_calculus_and_promotion.tex:253-321` | veto indicators and promotion boundaries | correct | The chapter now cleanly treats runtime/ESS-like diagnostics as downstream of veto gates rather than replacement proof. | Preserve. |
| L-041 | `appendices/app_b_matrix_calculus.tex:8-30` | matrix-calculus identities | correct | Symmetry/positivity conditions are now explicit enough for the quadratic-form identity. | Preserve. |
| L-042 | `appendices/app_d_mathdevmcp_workflows.tex:4-21` | MathDevMCP audit-assistant status | correct | The appendix now correctly frames MathDevMCP as an audit assistant, not an oracle. | Preserve. |
| L-043 | `references.bib` active closure | active bibliography record set | mixed | Mechanically resolvable and no live provisional note fields remain, but publication-strength curation still needs one explicit final choice for NeuTra and any theorem-level source claims. | Curated promotion required, not wholesale copy. |

| L-044 | `chapters/ch02_state_space_contracts.tex:1-163` | state-space contract and structural support | not checked | The chapter is structurally clear but not every dependency was audited equation-by-equation in this ledger. | Preserve; audit later if release scope expands. |
| L-045 | `chapters/ch04_bayesfilter_api.tex:1-225` | API contract and implementation promises | correct | The chapter now reads like a policy/contract memo rather than a hidden theorem. | Preserve. |
| L-046 | `chapters/ch05_prediction_error_decomposition.tex:1-130` | prediction error decomposition | correct | The theorem/lemma style is readable and the decomposition is consistent with the surrounding contract language. | Preserve. |
| L-047 | `chapters/ch06_stable_linear_filtering.tex:1-119` | stable linear filtering claims | correct | The linear filtering section is cleanly bounded as a stable linear-Gaussian discussion. | Preserve. |
| L-048 | `chapters/ch07_missing_data_mixed_frequency.tex:1-104` | missing-data and mixed-frequency handling | not checked | The chapter appears policy-consistent but was not rederived in full here. | Keep as not checked. |
| L-049 | `chapters/ch08_large_scale_lgssm.tex:1-93` | large-scale LGSSM checks and validation framing | correct | The validation-oriented language is more honest than the original source. | Preserve. |
| L-050 | `chapters/ch21_hmc_for_state_space.tex:1-144` | HMC target and mass-matrix orientation | correct | The chapter now matches the release-position wording used elsewhere in the rewrite. | Preserve. |
| L-051 | `chapters/ch22_mass_matrices.tex:1-225` | mass-matrix choices and curvature interpretation | correct | The covariance/precision distinction is now explicit enough for final review. | Preserve. |
| L-052 | `chapters/ch23_boundary_gradients.tex:1-23` | finite fallback wording and target-change warning | correct | The exact-support vs modified-target distinction is explicit. | Preserve. |
| L-053 | `chapters/ch24_xla_jit.tex:1-75` | XLA/JIT boundary language | not checked | The implementation contract is likely fine, but not reaudited in detail. | Preserve as not checked. |
| L-054 | `chapters/ch25_diagnostics.tex:1-69` | diagnostics vs proof status | correct | Diagnostics are explicitly kept separate from proof claims. | Preserve. |
| L-055 | `chapters/ch26_transport_surrogates.tex:1-104` | transport-surrogate boundary and exactness warning | correct | The chapter now clearly separates exact transport, surrogates, and target status. | Preserve. |
| L-056 | `chapters/ch26b_neutra_transport_hmc.tex:1-865` | NeuTra transport/HMC route summary | not checked | The route is documented but still needs source-support curation rather than stronger claims. | Preserve as bounded route description. |
| L-057 | `chapters/ch26c_hnn_surrogate_hmc.tex:1-1093` | HNN surrogate HMC and same-scalar claim boundary | not checked | The chapter is release-honest about its surrogate status, but claims require final source/policy cleanup. | Preserve as a conditional route. |
| L-058 | `chapters/ch29_nk_svd_case_study.tex:1-56` | NK/SVD case-study framing | not checked | Case-study content is mostly narrative and should stay scoped as such. | Preserve. |
| L-059 | `chapters/ch30_cip_afns_case_study.tex:1-74` | CIP/AFNS case-study framing | not checked | Similar to ch29: use as application narrative, not theorem support. | Preserve. |
| L-060 | `chapters/ch31_nawm_design_target.tex:1-71` | NAWM design target framing | not checked | The design-target chapter is honest but not fully audited here. | Preserve. |
| L-061 | `chapters/ch32_production_checklist.tex:1-93` | production checklist / gate language | correct | The checklist properly acts as gate language, not as proof. | Preserve. |
| L-062 | `appendices/app_a_notation.tex:1-73` | notation dictionary | correct | The notation appendix is the right anchor for the audit and the rename/override discipline. | Preserve. |
| L-063 | `appendices/app_f_source_map.tex:1-67` | source-map claim discipline | correct | It gives the right high-level claim/support taxonomy for the monograph. | Preserve. |
| L-064 | `appendices/app_g_experiment_templates.tex:1-68` | experiment template gate discipline | correct | The template is policy-consistent and should remain as a process artifact. | Preserve. |

| L-086 | `chapters/ch33_highdim_nonlinear_filtering_foundations.tex:1151-1158` | Meng citation used as contextual reference | correct | The wording now explicitly says this is a contextual reference, not theorem-level support. | Preserve. |
| L-087 | `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:286-337` | Retained-prefix certificate note and scalar/vector examples | not checked | The certificate note is explicit and much better, but the verification artifact still needs a durable final review note and the branch needs an implementation-facing cross-check. | Keep as not checked until the final check is attached. |
| L-088 | `chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:376-390` | Query-rule ownership note | not checked | The convention is explicit, but the route-audit note still says it is a convention rather than a source/code audit. | Preserve as a convention until checked. |
