# Request to Codex: full paragraph-by-paragraph and equation-by-equation audit of the rewritten monograph

- **Date:** 2026-08-06
- **Requested output:** a Markdown review file written back to this repo, with findings, evidence, and rewrite recommendations.
- **Target artifact:** `docs/fable-rewrite/monograph/main.tex` and its full input closure (all included chapters, appendices, bibliography, and required local assets).
- **Purpose:** audit the entire rewritten monograph for correctness, readability, policy compliance, and release readiness before any canonical promotion.

## What to do

Please audit the standalone rewritten monograph **paragraph by paragraph and equation by equation**.

For every paragraph and every displayed equation or theorem/proposition/algorithm block, check:

1. **Logical correctness**
   - Does the paragraph follow from earlier claims without hidden jumps?
   - Are assumptions, scope, targets, comparators, and nonclaims explicit?
   - Does the paragraph ever quietly change the target quantity, measure, or object?

2. **Mathematical correctness**
   - Is every equation supported by a checked derivation in the monograph’s notation or by a checked source anchor?
   - If an equation is claimed to be exact, is the exact computed quantity actually the same as the claimed target?
   - If a paragraph discusses approximation, does it say what is approximate and what is exact?

3. **Source support and literature audit**
   - Are claims supported by inspected primary technical sources, not just metadata or memory?
   - Are survey or design-space claims clearly downgraded when the supporting paper has not been inspected at theorem/algorithm level?
   - Are omitted-paper risks and provisional entries handled honestly?

4. **Readability and language quality**
   - Does the prose read naturally for a mathematically trained human reader?
   - Does it avoid internal-review language, project-lane jargon, or odd mechanistic phrasing?
   - Does it avoid evasive wording such as “reasonable,” “surrogate,” or “approximately correct” unless the modified target is explicitly defined?

5. **Policy compliance**
   - Check against the relevant policy files in this repo and user space:
     - `~/python/claudecodex/policies/global-scientific-coding-agent-policy.md`
     - `~/python/claudecodex/policies/scholarly-literature-audit-policy.md`
     - `AGENTS.md`
     - `CLAUDE.md`
   - Flag any paragraph that violates the policy on evidence discipline, source support, non-evasion, or publication readiness.

6. **Equation-by-equation proof audit**
   - For each important displayed equation, determine whether it is:
     - correct,
     - wrong relative to the stated target,
     - unsupported,
     - not checked,
     - or heuristic only.
   - If possible, use MathDevMCP-style derivation checks where relevant for the hardest equations.

## Required review style

- Review the monograph in **reading order**.
- Audit **every section**, not only the blocker chapters.
- Use a **paragraph ledger** with file/line anchors.
- For every material issue, state:
  - the exact claim,
  - the problem,
  - why it matters,
  - the evidence class,
  - and the repair recommendation.

## Files to inspect first

The rewrite tree root is:
- `docs/fable-rewrite/monograph/main.tex`

High-risk chapters and appendices include:
- `chapters/ch09_kalman_score.tex`
- `chapters/ch10_kalman_hessian.tex`
- `chapters/ch11_structural_derivatives.tex`
- `chapters/ch12_factor_derivatives.tex`
- `chapters/ch17_square_root_sigma_point.tex`
- `chapters/ch19_particle_filters.tex`
- `chapters/ch19b_dpf_literature_survey.tex`
- `chapters/ch20_filter_choice.tex`
- `chapters/ch28a_neural_network_state_space_model_applications.tex`
- `chapters/ch32c_entropic_ot_sinkhorn.tex`
- `chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`
- `chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex`
- `chapters/ch33_highdim_nonlinear_filtering_foundations.tex`
- `chapters/ch34_highdim_gaussian_projection_and_point_rule_foundations.tex`
- `chapters/ch35_highdim_sparse_grid_quadrature_and_fixed_cloud_scalar.tex`
- `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
- `chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`
- `chapters/ch38_highdim_validation_defect_calculus_and_promotion.tex`
- `appendices/app_b_matrix_calculus.tex`
- `appendices/app_c_factor_derivative_proofs.tex`
- `appendices/app_d_mathdevmcp_workflows.tex`
- `appendices/app_e_researchassistant_workflows.tex`

## Output requirements

Write back **one Markdown file** containing:

1. **Executive verdict** on whether the rewritten monograph is ready for release.
2. **Paragraph-by-paragraph findings** with file/line anchors.
3. **Equation-by-equation audit** for every important displayed formula.
4. **Policy compliance summary**.
5. **Language/readability findings**.
6. **MathDevMCP checks run** and what they concluded.
7. **Rewrite instructions**: what must be changed before final release.
8. **Explicit nonclaims**: what the review does not certify.

## Evidence classes to use
Use only these labels when classifying findings:
- `correct`
- `wrong relative to the stated target`
- `unsupported`
- `not checked`
- `heuristic only`

## Important constraints
- Do **not** assume build success means correctness.
- Do **not** promote diagnostics into proof.
- Do **not** let provisional bibliography resolution count as theorem-level support.
- If a paragraph is too dense to assess safely, split it into subclaims and audit them separately.

## Goal
Return a concise but thorough Markdown audit note that can be used to drive the final release rewrite.
