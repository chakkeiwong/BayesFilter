# Whole-document audit ledger: opening chapters and appendices (phase 1 seed)

- **Date:** 2026-08-07
- **Purpose:** seed the full paragraph-by-paragraph / equation-by-equation audit with the opening chapters and appendices.
- **Status:** phase-1 seed only; this does not yet cover the full 493-page closure.

## Coverage policy
This ledger row set is intentionally conservative. A paragraph or equation is only marked `correct` when the review has a concrete reason to do so; otherwise it stays `not checked` or `unsupported`.

## Seeded rows

| ID | File:line(s) | Claim / object | Evidence class | Evidence / reason | Action / status |
|---|---|---|---|---|---|
| L-065 | `main.tex:20-110` | Book structure and chapter ordering | correct | 11-part structure now matches the active input closure. | Preserve. |
| L-066 | `chapters/ch01_introduction.tex:1-88` | Reader map and audit contract | correct | Honest about the rewrite branch and evidence boundary. | Preserve. |
| L-067 | `chapters/ch02_state_space_contracts.tex:1-163` | Structural state-space contract | not checked | Coherent, but not yet audited paragraph-by-paragraph. | Keep as not checked. |
| L-068 | `chapters/ch03_hmc_target_requirements.tex:9-45` | Exact target vs modified fallback | correct | Explicit support-rejection vs modified-target distinction. | Preserve. |
| L-069 | `chapters/ch04_bayesfilter_api.tex:1-225` | API / contract chapter | correct | Reads as policy/contract material rather than hidden theorem. | Preserve. |
| L-070 | `chapters/ch05_prediction_error_decomposition.tex:1-130` | PED / decomposition target | correct | Standard decomposition under declared conditions. | Preserve. |
| L-071 | `chapters/ch06_stable_linear_filtering.tex:1-119` | Stable linear filtering claims | correct | Stays within LGSSM / linear-filter boundary. | Preserve. |
| L-072 | `chapters/ch07_missing_data_mixed_frequency.tex:1-104` | Missing-data / mixed-frequency handling | not checked | Likely fine, but not fully audited here. | Keep as not checked. |
| L-073 | `chapters/ch08_large_scale_lgssm.tex:1-93` | Large-scale LGSSM validation framing | correct | Validation presented as a boundary, not as a theorem. | Preserve. |
| L-074 | `chapters/ch09_kalman_score.tex:23-145` | Prediction / innovation / solve-form score recursion | correct | Algebraically correct on the stated SPD branch. | Preserve. |
| L-075 | `chapters/ch10_kalman_hessian.tex:18-35` | Abstract Hessian summary | not checked | Source-shape plausible, not fully expanded/certified. | Preserve as bounded summary. |
| L-076 | `chapters/ch11_structural_derivatives.tex:13-202` | Structural derivative-provider contract | not checked | Good interface, but not every adapter/source closure certified. | Preserve as contract. |
| L-077 | `chapters/ch12_factor_derivatives.tex:39-177` | Factor reconstruction and branch caveats | correct | Correct smooth-branch identity and caveat handling. | Preserve. |
| L-078 | `chapters/ch13_custom_gradient_wrappers.tex:85-100` | Posterior curvature vs likelihood-only observed information | correct | Policy-compliant distinction. | Preserve. |
| L-079 | `chapters/ch14_derivative_validation.tex:62-153` | Validation ladder and artifact schema | correct | Strong evidence / proof separation. | Preserve. |
| L-080 | `appendices/app_b_matrix_calculus.tex:8-30` | Matrix identities | correct | Correct under stated invertibility / symmetry assumptions. | Preserve. |
| L-081 | `appendices/app_c_factor_derivative_proofs.tex:1-14` | Bounded placeholder for missing signed primitive | correct | Honest release posture. | Preserve unless primitive is derived. |
| L-082 | `appendices/app_d_mathdevmcp_workflows.tex:4-21` | MathDevMCP as audit assistant | correct | Diagnostic only, not oracle. | Preserve. |
| L-083 | `appendices/app_e_researchassistant_workflows.tex:52-55` | Remaining source-support blockers | correct | Honest blocker ledger. | Preserve. |
| L-084 | `appendices/app_f_source_map.tex:1-67` | Source map and claim classes | correct | Stable support taxonomy. | Preserve. |
| L-085 | `appendices/app_g_experiment_templates.tex:1-68` | Experiment template policy | correct | Good evidence-contract scaffold. | Preserve. |
| L-086 | `chapters/ch17_square_root_sigma_point.tex:311-351` | SR-UKF filtered factor target + signed derivative posture | not checked | Honest conditional posture; full signed primitive still missing. | Keep conditional or derive primitive. |
| L-087 | `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:171-302` | Retained-first squared-TT contraction / derivative | not checked | Right-contraction story is in place, but scalar/vector identity certificate still required. | Highest mathematical blocker. |
| L-088 | `chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:358-390` | Retained-reference query rule / Jacobian ownership | not checked | Contract explicit, but implementation-facing confirmation not attached. | Preserve as a convention until checked. |
| L-089 | `chapters/ch32c_entropic_ot_sinkhorn.tex:688-694` | Acevedo / second-order ETPF support sentence | unsupported | Citation resolves, but theorem-level support remains unclosed. | Keep as bounded context only. |
| L-090 | `references.bib:587-593` | Hoffman / NeuTra publication metadata | correct metadata, not checked for publication-optimal choice | Stronger published record choice still needs explicit freeze at promotion time. | Settle at promotion gate. |

## Ledger extension rule
Continue chapter-by-chapter in reading order. When a paragraph or equation is obviously release-safe, mark it `correct`. When it is contract-like but not fully proven in the ledger, mark it `not checked`. When it relies on source support not yet closed, mark it `unsupported`.

## Nonclaim
This phase-1 seed does not certify the entire monograph; it only establishes a stable row format and a seed coverage set for the whole-document audit.
