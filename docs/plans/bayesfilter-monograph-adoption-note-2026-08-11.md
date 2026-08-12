# Monograph archive and official adoption note

- **Date:** 2026-08-11
- **Decision:** the standalone rewritten monograph has now been adopted as the official canonical monograph source in `docs/`.

## What was archived

The previous canonical monograph state was preserved under:

- `docs/archive/monograph-pre-fable-rewrite-2026-08-09/`

Archived items include:
- `main.tex`
- `preamble.tex`
- `references.bib`
- `main.pdf`
- full `chapters/` snapshot
- full `appendices/` snapshot

This archive is historical evidence and should not be edited.

## What was promoted

The official `docs/` tree now contains the curated rewrite-derived source for:

### Chapters
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
- `app_b_matrix_calculus.tex`
- `app_c_factor_derivative_proofs.tex`
- `app_d_mathdevmcp_workflows.tex`
- `app_e_researchassistant_workflows.tex`

### Bibliography
- `docs/references.bib`

### Required assets
The five SSL-LSTM PNG figure assets and their local result JSON were copied into:
- `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/direct-visual-validation/`

## What was intentionally not promoted wholesale
The adoption did **not** copy the entire standalone subtree into canonical `docs/`. In particular, it excluded:
- generated LaTeX byproducts (`.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.toc`)
- `.mathdevmcp/` cache/index files
- standalone review packaging files
- copied experiment-artifact trees not needed by the canonical build
- standalone PDF build products as source files

## Canonical build result after adoption

Canonical rebuild command:

```bash
cd docs
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Final canonical build state:
- undefined citations: **none**
- undefined references: **none**
- duplicate active labels: **none**
- foreign-command / `amsmath` warnings: **none**
- output PDF: `docs/main.pdf`
- pages: **495**

## Remaining explicit nonclaims

Adopting the rewrite as the official canonical source does **not** mean:
- every paragraph and equation in the 495-page monograph has been fully certified line-by-line;
- theorem-level source support is fully closed for every surveyed family;
- the SR-UKF signed derivative primitive is fully derived in-book;
- the squared-TT lane is a proof of adaptive TT correctness rather than a fixed-branch contract plus certificate;
- the ICNN trainer is fully specified rather than explicitly schematic where stated;
- the document is publication-perfect in typography.

The official canonical source is now the rewrite-derived branch because it is materially better, builds cleanly from tracked inputs, and carries its remaining boundaries as explicit nonclaims rather than hidden contradictions.

## Final status

`docs/` is now the official monograph source going forward.

The standalone rewrite under `docs/fable-rewrite/monograph/` remains useful as an evidence and repair record, but canonical development should proceed from `docs/` from this point onward.
