# Reply to the replacement verdict on the standalone monograph rewrite

- **Date:** 2026-08-06
- **Reviewed artifact:** `docs/fable-rewrite-replacement-verdict-2026-08-06.md`
- **Decision on that verdict:** **Mostly agree. Use the rewrite as the migration base, but do not replace canonical `docs/` by a raw subtree swap.**

---

## 1. Bottom line

I agree with the verdict’s main conclusion.

The current canonical `docs/` tree is not the better baseline anymore. The rewrite branch is materially stronger and should become the basis of canonical monograph development. But the correct migration is:

1. preserve the rewrite branch,
2. repair the localized blockers,
3. rebuild and recheck,
4. promote a curated source diff back into canonical `docs/`,
5. archive the old canonical state as historical evidence.

That is the right operational decision.

---

## 2. What I agree with strongly

### 2.1 The rewrite is already the right migration base
I agree. The verdict is right that the old canonical source is not a defensible primary development baseline anymore. The rewrite fixes real defects, builds successfully, and states several mathematical and source boundaries more honestly.

### 2.2 The current canonical source should not remain primary
I agree. A source tree that fails clean build and still carries confirmed mathematical, target-definition, citation, and policy defects should not remain the live development version merely because it is canonical by history.

### 2.3 Whole-subtree replacement would be the wrong packaging move
I agree. The rewrite branch should not be copied into place wholesale because it includes standalone-snapshot packaging and generated state that are not canonical monograph source.

### 2.4 The remaining blockers are now localized
I agree. The verdict does not argue that the rewrite is broadly untrustworthy. It identifies a small number of concentrated issues:
- squared-TT retained-coordinate derivation,
- SR-UKF derivative completeness / release posture,
- source-support and bibliography curation,
- canonical LEDH subsection normalization,
- final typesetting debt.

That is exactly the right shape of remaining work.

---

## 3. What I would narrow or update after the latest pass

I agree with the verdict overall, but after the latest blocker-oriented rewrite pass I would update three points.

### 3.1 The squared-TT blocker remains the decisive one, but its status depends on the latest contraction rewrite
The verdict is right that the earlier retained-first rewrite was internally inconsistent. After that verdict, I rewrote the retained-numerator and derivative formulas again to align them with the retained-first convention. So the correct current status is:

- the verdict’s criticism of the earlier state was correct;
- the branch now needs a **fresh bounded check** on the updated `ch36b` formulas rather than a continued assumption that the old inconsistency is still present unchanged.

So I would rewrite that point as:

> The squared-TT retained-coordinate lane was correctly identified as inconsistent in the reviewed snapshot. It has since been rewritten again and now requires a fresh bounded derivation/consistency check before promotion.

### 3.2 The bibliography curation criticism is right, but one subpoint is now stale
The verdict is right that speculative unused additions should not be promoted blindly, and I agree with that.

However, after the latest curation pass:
- the unused speculative cluster has been removed from the active rewrite bibliography,
- active provisional note fields have been removed,
- the remaining concern is not “speculative records still appended everywhere,” but whether the surviving active records are publication-strong enough for the owner’s intended release standard.

So I would narrow that point to the active remaining bibliography and source-support boundary, rather than the broader stale description.

### 3.3 The NeuTra record point should be treated as a curation choice, not a settled correction
The verdict says the rewrite weakened the published NeuTra record. I agree that this is a curation concern, but I would treat it as a migration decision rather than a theorem-level blocker. In the latest pass I restored the weaker arXiv/AABI-style record because the independent audit had earlier objected to the stronger ICML/PMLR phrasing. So this point should be settled by one explicit bibliography policy choice before promotion, not left as a floating contradiction between review rounds.

---

## 4. What should be rewritten next

The next rewrite should stay narrow.

### 4.1 Recheck and, if needed, rewrite the squared-TT retained-coordinate lane one last time
Files:
- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
- `docs/fable-rewrite/monograph/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`

Task:
- run one fresh bounded derivation check on the latest retained-first contraction rewrite;
- if the new formulas are still not internally aligned, repair them before promotion;
- preserve a short scalar/vector derivation note and a numerical identity check artifact.

This remains the first rewrite target because it is the only issue the verdict classifies as a replacement-blocking mathematical inconsistency.

### 4.2 Freeze the SR-UKF release posture explicitly
Files:
- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex`
- `docs/fable-rewrite/monograph/chapters/ch12_factor_derivatives.tex`
- `docs/fable-rewrite/monograph/appendices/app_c_factor_derivative_proofs.tex`

Task:
- either derive the signed update/downdate primitive,
- or keep every summary statement explicitly conditional on its absence.

This is now a release-honesty problem more than a value-side math error.

### 4.3 Do one final bibliography/source-support curation pass
Files:
- `docs/fable-rewrite/monograph/references.bib`
- `docs/fable-rewrite/monograph/chapters/ch28a_neural_network_state_space_model_applications.tex`
- `docs/fable-rewrite/monograph/chapters/ch32c_entropic_ot_sinkhorn.tex`
- `docs/fable-rewrite/monograph/appendices/app_e_researchassistant_workflows.tex`

Task:
- keep only verified active additions in the monograph bibliography,
- keep theorem-level claims only where exact source support is inspected,
- preserve explicit nonclaims where support remains open,
- and record unresolved candidates in a ledger/note rather than in the active book if they are not needed.

### 4.4 Normalize the canonical LEDH subsection once more
File:
- `docs/fable-rewrite/monograph/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`

Task:
- keep the current strong policy binding,
- ensure those identifiers appear once in the strongest canonical statement,
- keep historical benchmark prose from weakening them.

### 4.5 Do only a bounded release-quality polish pass
Files likely:
- `docs/fable-rewrite/monograph/chapters/ch32c_entropic_ot_sinkhorn.tex`
- `docs/fable-rewrite/monograph/chapters/ch32d_retained_teacher_neural_ot.tex`
- `docs/fable-rewrite/monograph/references.bib`
- worst-overflow pages and figure-label pages

Task:
- fix the remaining warning and most severe overfulls,
- then stop. This should not become a broad editorial rewrite.

---

## 5. What I recommend operationally

I recommend this exact sequence:

1. one last bounded check of the updated squared-TT rewrite,
2. one explicit SR-UKF release-posture decision,
3. one final source-support/bibliography curation pass,
4. one LEDH subsection normalization pass,
5. one minimal final typesetting pass,
6. rebuild,
7. write the promotion result note,
8. then promote a curated diff into canonical `docs/`.

Do **not** do another broad monograph rewrite.

---

## 6. Promotion posture after that pass

After those localized rewrites, I would support:

- archiving the old canonical source/PDF state as history,
- adopting the rewrite-derived source as the new official canonical monograph basis,
- but doing so by **curated promotion**, not by replacing `docs/` with the whole standalone subtree.

That remains the right form of adoption.

---

## 7. Final answer

So my answer to the verdict is:

- **Yes, I mostly agree.**
- The rewrite should become the basis of canonical monograph development.
- We should **not** keep the old canonical source as primary.
- We should **not** replace by raw subtree copy.
- We should rewrite only the localized blocker set listed above, then promote a curated diff into `docs/`.

That is the cleanest and most honest migration path.
