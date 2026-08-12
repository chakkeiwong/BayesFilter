# Plan to close the remaining release gaps in the rewritten monograph

- **Date:** 2026-08-09
- **Scope:** `docs/fable-rewrite/monograph/` only
- **Purpose:** close the remaining release gaps so the standalone rewrite can become a real replacement candidate for canonical `docs/`.

## Remaining gaps to close

1. **Squared-TT retained-coordinate derivation certificate**
2. **SR-UKF signed derivative release posture**
3. **Source-support and bibliography closure**
4. **Canonical LEDH policy normalization**
5. **ICNN trainer/status clarity**
6. **Final typesetting / release polish**

## Strategy

Do the smallest possible pass that closes or honestly narrows every remaining gap. Do not restart the rewrite, do not broaden scope, and do not copy standalone-snapshot packaging into canonical `docs/`.

---

## Phase 1 — close the squared-TT retained-coordinate gap completely

### Files
- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
- `docs/fable-rewrite/monograph/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`
- new supporting note under `docs/plans/`

### Actions
1. Keep the retained-first/right-contraction convention.
2. Add a fully explicit retained-prefix / suffix-mass certificate covering:
   - the scalar case `m=1, D=2`,
   - the first nontrivial vector-retained case `m=2, D=4`.
3. Ensure the retained numerator, derivative, and query rule all use the same retained/reference/Jacobian ownership story.
4. Preserve the route-audit note so this lane does not overclaim implementation certification.
5. Add a durable note in `docs/plans/` recording the explicit certificate and any small numerical identity check.

### Verification
- scalar and vector retained-prefix formulas are explicit in `ch36b`;
- `ch37` uses the same retained-reference evaluator story;
- direct rebuild succeeds;
- no retained-last/left-contraction mismatch remains.

---

## Phase 2 — freeze the SR-UKF release claim

### Files
- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex`
- `docs/fable-rewrite/monograph/chapters/ch12_factor_derivatives.tex`
- `docs/fable-rewrite/monograph/appendices/app_c_factor_derivative_proofs.tex`

### Actions
1. Keep the current value-side filtered-factor repair.
2. Decide one honest release posture:
   - either derive the signed primitive fully,
   - or keep the negative-weight analytical-score lane conditional everywhere.
3. Remove any lingering sentence fragments or wording inconsistencies.
4. Make sure no summary or recommendation paragraph sounds stronger than the actual derivation support.

### Verification
- no prose claims full signed derivative support unless the primitive is actually added;
- filtered-factor statements remain consistent with the declared covariance target;
- final wording matches the ledger.

---

## Phase 3 — complete source-support and bibliography closure

### Files
- `docs/fable-rewrite/monograph/references.bib`
- `docs/fable-rewrite/monograph/chapters/ch28a_neural_network_state_space_model_applications.tex`
- `docs/fable-rewrite/monograph/chapters/ch19b_dpf_literature_survey.tex`
- `docs/fable-rewrite/monograph/chapters/ch32c_entropic_ot_sinkhorn.tex`
- `docs/fable-rewrite/monograph/appendices/app_e_researchassistant_workflows.tex`

### Actions
1. Keep only bibliography entries needed by active release prose.
2. For each theorem-level/source-faithfulness claim, choose exactly one:
   - verified source anchor,
   - replaced with an already inspected source,
   - explicit nonclaim.
3. Keep the HAC theorem boundary as a nonclaim until exact source closure exists.
4. Keep the Li–Coates boundary explicit until the durable source support is closed.
5. Keep survey/design-space language honest and explicitly non-theorem where appropriate.

### Verification
- no active bibliography record is provisional or verify-before-publication;
- release-path theorem claims have source anchors or explicit nonclaims;
- bibliography is curated for the actual active text.

---

## Phase 4 — normalize canonical LEDH policy text

### Files
- `docs/fable-rewrite/monograph/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`
- optionally `docs/fable-rewrite/monograph/chapters/ch19b_dpf_literature_survey.tex` if continuity wording is needed

### Actions
1. Keep the current route identifiers and historical-route demotion.
2. Ensure the canonical statement appears once in its strongest form.
3. Prevent later benchmark or design-space prose from weakening that policy statement.
4. Keep the chunk/tuning semantics explicit and non-overridable.

### Verification
- identifiers are singular and consistent;
- historical routes remain historical/diagnostic-only;
- later prose does not blur policy and mathematical evidence.

---

## Phase 5 — settle the ICNN trainer status

### File
- `docs/fable-rewrite/monograph/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex`

### Actions
1. Choose one final status:
   - fully specified trainer, or
   - explicitly schematic direct-map pattern.
2. Remove any remaining canonical-trainer implication unless the trainer is complete.
3. Keep the fixed-`\varphi` correction intact.

### Verification
- no oscillation between canonical and schematic wording remains;
- trainer status is explicit and honest.

---

## Phase 6 — final release polish

### Files likely to touch
- `docs/fable-rewrite/monograph/references.bib`
- `docs/fable-rewrite/monograph/chapters/ch32c_entropic_ot_sinkhorn.tex`
- `docs/fable-rewrite/monograph/chapters/ch32d_retained_teacher_neural_ot.tex`
- worst-overfull pages identified by the build log
- repaired-label chapters such as `ch18b` and `ch33`

### Actions
1. Normalize accent syntax and old-TeX fraction forms.
2. Remove the remaining warning(s).
3. Triage the most severe overfull boxes by visual impact.
4. Remove the remaining process-register language only where it harms final readability.

### Verification
- clean rebuild;
- no undefined citations / references / duplicate labels;
- no foreign-command warning;
- severe overfulls reviewed.

---

## Phase 7 — final replacement-candidate review and promotion package

### Outputs
1. final replacement-candidate review note
2. refreshed persistent audit ledger rows for all changed paragraphs/equations
3. curated promotion package for canonical `docs/`

### Decision rule
Call the rewrite a replacement candidate only if:
- all six gaps have explicit outcomes,
- the branch rebuilds cleanly,
- the ledger and review note agree,
- and the remaining nonclaims are explicit rather than hidden.

## Critical files to modify
- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
- `docs/fable-rewrite/monograph/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`
- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex`
- `docs/fable-rewrite/monograph/chapters/ch12_factor_derivatives.tex`
- `docs/fable-rewrite/monograph/appendices/app_c_factor_derivative_proofs.tex`
- `docs/fable-rewrite/monograph/references.bib`
- `docs/fable-rewrite/monograph/chapters/ch28a_neural_network_state_space_model_applications.tex`
- `docs/fable-rewrite/monograph/chapters/ch19b_dpf_literature_survey.tex`
- `docs/fable-rewrite/monograph/chapters/ch32c_entropic_ot_sinkhorn.tex`
- `docs/fable-rewrite/monograph/appendices/app_e_researchassistant_workflows.tex`
- `docs/fable-rewrite/monograph/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`
- `docs/fable-rewrite/monograph/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex`
- `docs/fable-rewrite/monograph/chapters/ch32d_retained_teacher_neural_ot.tex`
- worst-overfull / release-polish pages identified by the build log

## Verification
- Keep the rewrite root buildable after every major cluster.
- Preserve derivation notes, source-support notes, audit-ledger updates, and final review notes under `docs/plans/`.
- Do not call the branch a replacement candidate until every remaining gap above has an explicit, documented resolution.
