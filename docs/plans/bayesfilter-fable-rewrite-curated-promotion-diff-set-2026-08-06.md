# Curated promotion diff set after the replacement-verdict repair pass

- **Date:** 2026-08-06
- **Source:** `docs/fable-rewrite/monograph/`
- **Target:** canonical `docs/`
- **Purpose:** define the specific source files, bibliography changes, and required assets that should be promoted into canonical `docs/` after the latest rewrite pass.

## Decision
Promote the rewrite by **curated source diff**, not by replacing `docs/` with the entire standalone subtree.

## Promote these source files

### Chapters
Promote the reviewed diffs from these active rewrite chapters into canonical `docs/chapters/`:

- `ch01_introduction.tex`
- `ch03_hmc_target_requirements.tex`
- `ch13_custom_gradient_wrappers.tex`
- `ch17_square_root_sigma_point.tex`
- `ch18b_structural_deterministic_dynamics.tex`
- `ch19_particle_filters.tex`
- `ch19b_dpf_literature_survey.tex`
- `ch19c_dpf_implementation_literature.tex`
- `ch19e_dpf_hmc_target_suitability.tex`
- `ch19f_dpf_debugging_crosswalk.tex`
- `ch20_filter_choice.tex`
- `ch23_boundary_gradients.tex`
- `ch28a_neural_network_state_space_model_applications.tex`
- `ch32a_soft_differentiable_resampling.tex`
- `ch32c_entropic_ot_sinkhorn.tex`
- `ch32c2_ledh_pfpf_ot_custom_gradient.tex`
- `ch32e_icnn_brenier_monge_gap_map_learning.tex`
- `ch34_highdim_gaussian_projection_and_point_rule_foundations.tex`
- `ch35_highdim_sparse_grid_quadrature_and_fixed_cloud_scalar.tex`
- `ch35b_highdim_fixed_cloud_filtering_and_sgqf_validation.tex`
- `ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
- `ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`

### Appendices
Promote the reviewed diffs from these appendices into canonical `docs/appendices/`:

- `app_b_matrix_calculus.tex`
- `app_c_factor_derivative_proofs.tex`
- `app_d_mathdevmcp_workflows.tex`
- `app_e_researchassistant_workflows.tex`

### Bibliography
Promote:
- `docs/fable-rewrite/monograph/references.bib`

with one caution:
- keep the curated active entries and corrected names/records,
- do not re-introduce unresolved speculative unused entries,
- preserve the current honest nonclaims where theorem-level source support remains open.

## Promote required figure assets only
Promote only the actual build dependencies needed to close the canonical build defect:

- the five SSL-LSTM PNG figure assets used in place of the missing PDF files.

They should be copied into a stable canonical asset path under `docs/` and the canonical `ch28a` include paths should be updated accordingly.

Do **not** promote the entire copied artifact subtree from the standalone branch.

## Do not promote these
Do **not** copy any of the following into canonical `docs/`:

- generated LaTeX byproducts:
  - `.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.toc`
- the standalone built PDF as source
- `.mathdevmcp/latex_index.json`
- copied experiment-artifact trees not required by the build
- standalone review packaging files that belong under `docs/fable-rewrite/` or `docs/plans/`, not canonical monograph source

## Promotion gate before merge
Before applying this curated diff into canonical `docs/`, require:

1. one final bounded independent review focused on:
   - SR-UKF release posture,
   - squared-TT retained-coordinate consistency,
   - source-support honesty,
   - canonical LEDH subsection normalization,
   - and release-quality typography acceptability;
2. canonical clean build after the promoted figure assets and source diffs land;
3. no undefined citations, undefined references, or active duplicate labels in canonical `docs/`;
4. a short migration result note recording the exact promoted files, asset paths, and remaining nonclaims.

## Final promotion instruction
Treat the standalone rewrite as the **authoritative repair branch** and canonical `docs/` as the destination for a selective merge. The migration should be chapter/appendix/bibliography/asset promotion, not subtree replacement.
