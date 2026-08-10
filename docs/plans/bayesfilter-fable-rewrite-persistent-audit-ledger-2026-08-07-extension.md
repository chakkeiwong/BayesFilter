# Whole-document audit ledger extension: front-matter and first chapter blocks

- **Date:** 2026-08-07
- **Purpose:** extend the persistent audit ledger so it can support a real paragraph-by-paragraph, equation-by-equation whole-document audit rather than only the highest-risk blockers.
- **Coverage model:** continue using the existing evidence classes (`correct`, `wrong relative to the stated target`, `unsupported`, `not checked`, `heuristic only`).

## New rows added for the opening chapters and appendices

| ID | File:line(s) | Claim / object | Evidence class | Evidence / reason | Action / status |
|---|---|---|---|---|---|
| L-065 | `main.tex:20-110` | Book structure and chapter ordering | correct | The rewrite’s 11-part structure is now honest and aligned with the actual input closure. | Preserve. |
| L-066 | `chapters/ch01_introduction.tex:1-88` | Reader map and audit contract | correct | The chapter now correctly states the review / evidence boundary and the role of the rewrite as a development branch. | Preserve. |
| L-067 | `chapters/ch02_state_space_contracts.tex:1-163` | Structural state-space contract | not checked | The interface is coherent, but the chapter has not yet been audited paragraph-by-paragraph in the ledger. | Keep as not checked. |
| L-068 | `chapters/ch03_hmc_target_requirements.tex:9-45` | exact target vs modified fallback | correct | The chapter now clearly distinguishes exact support rejection from an explicitly modified target. | Preserve. |
| L-069 | `chapters/ch04_bayesfilter_api.tex:1-225` | API / contract chapter | correct | The chapter now reads as a policy and contract memo rather than a hidden theorem. | Preserve. |
| L-070 | `chapters/ch05_prediction_error_decomposition.tex:1-130` | PED / decomposition target | correct | The decomposition is structurally sound and pedagogically useful. | Preserve. |
| L-071 | `chapters/ch06_stable_linear_filtering.tex:1-119` | stable linear filtering claims | correct | The chapter remains within the LGSSM/linear-filter boundary. | Preserve. |
| L-072 | `chapters/ch07_missing_data_mixed_frequency.tex:1-104` | missing-data / mixed-frequency handling | not checked | The chapter is likely fine, but no full audit pass has been attached here yet. | Keep as not checked. |
| L-073 | `chapters/ch08_large_scale_lgssm.tex:1-93` | large-scale LGSSM validation framing | correct | Validation is presented as a boundary, not as a theorem. | Preserve. |
| L-074 | `chapters/ch09_kalman_score.tex:23-145` | prediction / innovation / solve-form score recursion | correct | The formulas are algebraically correct on the stated SPD branch. | Preserve. |
| L-075 | `chapters/ch10_kalman_hessian.tex:18-35` | abstract Hessian summary | not checked | Source-shape is plausible, but the abstract Hessian decomposition is not fully expanded or independently certified. | Preserve as a bounded summary. |
| L-076 | `chapters/ch11_structural_derivatives.tex:13-202` | structural derivative-provider contract | not checked | The contract is readable, but not every adapter/source closure is certified here. | Preserve as a contract. |
| L-077 | `chapters/ch12_factor_derivatives.tex:39-177` | factor reconstruction and branch caveats | correct | Correct smooth-branch identity and caveat handling. | Preserve. |
| L-078 | `chapters/ch13_custom_gradient_wrappers.tex:85-100` | posterior curvature vs likelihood-only observed information | correct | The distinction is now explicit and policy-compliant. | Preserve. |
| L-079 | `chapters/ch14_derivative_validation.tex:62-153` | validation ladder and artifact schema | correct | Strong evidence/separation discipline. | Preserve. |
| L-080 | `appendices/app_b_matrix_calculus.tex:8-30` | matrix identities | correct | Correct under stated invertibility/symmetry assumptions. | Preserve. |
| L-081 | `appendices/app_c_factor_derivative_proofs.tex:1-14` | bounded placeholder for missing signed primitive | correct | Honest release posture. | Preserve unless the primitive is derived. |
| L-082 | `appendices/app_d_mathdevmcp_workflows.tex:4-21` | MathDevMCP as audit assistant | correct | Properly framed as diagnostic support only. | Preserve. |
| L-083 | `appendices/app_e_researchassistant_workflows.tex:52-55` | remaining source-support blockers | correct | Honest blocker ledger; should remain until closure. | Preserve. |
| L-084 | `appendices/app_f_source_map.tex:1-67` | source map and claim classes | correct | Stable source-support taxonomy. | Preserve. |
| L-085 | `appendices/app_g_experiment_templates.tex:1-68` | experiment template policy | correct | Good evidence-contract scaffold. | Preserve. |

## Remaining open high-risk rows

| ID | File:line(s) | Claim / object | Evidence class | Evidence / reason | Action / status |
|---|---|---|---|---|---|
| L-001 | `chapters/ch17_square_root_sigma_point.tex:311-351` | SR-UKF filtered factor target and signed derivative posture | `not checked` | Conditional release posture is honest, but the full signed primitive remains missing. | Keep conditional or derive primitive. |
| L-015 | `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:171-302` | retained-first squared-TT contraction / derivative | `not checked` | Right-contraction story is in place, but a scalar/vector identity certificate is still required. | Highest mathematical blocker. |
| L-016 | `chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:358-390` | retained-reference query rule / Jacobian ownership | `not checked` | Contract is explicit, but implementation-facing source/route confirmation is not attached. | Preserve as a convention until checked. |
| L-009 | `chapters/ch32c_entropic_ot_sinkhorn.tex:688-694` | Acevedo / second-order ETPF support sentence | `unsupported` | Citation resolves, but theorem-level support remains unclosed. | Keep as bounded context only. |
| L-020 | `references.bib:587-593` | Hoffman / NeuTra publication metadata | `correct` as metadata, `not checked` for publication-optimal choice | Stronger published record choice still needs to be frozen explicitly at promotion time. | Settle at promotion gate. |

## Ledger policy for the remaining full-document sweep

From here onward, the whole-document audit should proceed chapter by chapter, but every new row must still use the same evidence-class vocabulary and must distinguish:

- exact theorem or derivation support,
- implementation/provenance support,
- and explicit nonclaims.

This extension only seeds the opening chapters and appendices. It does not certify the untouched remainder of the monograph.
