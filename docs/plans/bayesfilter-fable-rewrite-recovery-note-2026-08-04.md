# Recovery Note: Fable rewrite monograph handoff after reboot

- **Date:** 2026-08-04
- **Purpose:** let the next agent resume the standalone monograph rewrite immediately after reboot.
- **Status at interruption:** the standalone rewrite tree exists and a large batch of high-confidence repairs has already been applied there, but the rewritten PDF has **not** yet been rebuilt successfully after those edits.

---

## 1. What the owner asked for

The owner clarified that they want **the actual rewritten monograph PDF**, not another planning note.

The rewrite is being done in a **standalone tree** so the original `docs/` monograph is not touched.

Primary target:
- `docs/fable-rewrite/monograph/main.tex`

Expected deliverable:
- `docs/fable-rewrite/monograph/main.pdf`

---

## 2. Files to read first after reboot

Read these in order before doing anything else:

1. `docs/fable-rewrite/execution-audit-2026-08-04.md`
2. `docs/fable-rewrite/bayesfilter-monograph-rewrite-freeze-and-ledger-2026-08-04.md`
3. `docs/fable-rewrite/bayesfilter-monograph-rewrite-master-document-2026-08-04.md`

These establish:
- chosen artifact = **current 55-chapter source tree**,
- bounded execution scope,
- confirmed-vs-under-specified findings,
- why the rewrite is being done in `docs/fable-rewrite/monograph/`.

---

## 3. Rewrite tree already created

The following standalone tree already exists:

- `docs/fable-rewrite/monograph/`

It already contains copied versions of:
- `main.tex`
- `preamble.tex`
- `references.bib`
- `chapters/`
- `appendices/`

Also copied into the rewrite tree:
- `docs/fable-rewrite/monograph/plans/artifacts/ssl-lstm-neutra-2026-07-14/...`

That copy was made so the rewritten document can use the tracked `.png` visual-validation figures locally.

---

## 4. Build state at interruption

An initial standalone build was run from:

- `docs/fable-rewrite/monograph/`

It failed first on the missing `ch28a` figures because the source referenced `.pdf` files that do not exist in git.

Those five `\includegraphics` lines in:
- `chapters/ch28a_neural_network_state_space_model_applications.tex`

were already changed from `.pdf` to `.png` in the rewrite tree.

So the **next build should be run again** to reveal the next blocker.

---

## 5. High-confidence edits that were already applied in the rewrite tree

These edits were already applied successfully in `docs/fable-rewrite/monograph/`:

### Core target / HMC wording
- `chapters/ch03_hmc_target_requirements.tex`
  - added explicit `+ constant` to the target definition
  - rewrote the surrogate-gradient wording to distinguish exact endpoint-MH correction from real target changes
  - rewrote the finite-fallback section to state exact-support-rejection vs declared modified-target contracts plainly
- `chapters/ch23_boundary_gradients.tex`
  - replaced the finite-fallback checklist wording with exact-support-rejection vs finite-modified-target wording

### Particle-filter and particle-flow repairs
- `chapters/ch19_particle_filters.tex`
  - repaired the skip-resampling branch by introducing post-decision weights in the algorithm/proof
  - added Jensen inequality wording for the log estimator
  - added log-domain/logsumexp guidance
- `chapters/ch19b_dpf_literature_survey.tex`
  - repaired the chapter-derived LEDH offset to use the centered information term
  - added reconciliation language tying that form to the later Li–Coates source form
  - removed one drafting-style sentence about how the chapter “should phrase” LEDH
- `chapters/ch19c_dpf_implementation_literature.tex`
  - cleaned one implementation-register phrase
  - replaced “OT survey companion note” wording with a direct boundary statement
- `chapters/ch19e_dpf_hmc_target_suitability.tex`
  - replaced “banking evidence” with “operational-claim evidence”
  - replaced “current repair lane” wording
- `chapters/ch19f_dpf_debugging_crosswalk.tex`
  - replaced “row-173 trace” with “one archived fixture trace from the comparison campaign”

### OT / learned transport repairs
- `chapters/ch32a_soft_differentiable_resampling.tex`
  - fixed the misleading “transport-based alternatives” grouping
  - added a sentence clarifying that this chapter’s “soft resampling” is a minimal local teaching rule, not the Particle Filter Networks algorithm
- `chapters/ch32c_entropic_ot_sinkhorn.tex`
  - removed the duplicated “Exact teacher versus exact engineering route” subsection
  - rewrote the GenUT section so it is explicitly a GenUT-motivated local whitened-moment variant rather than a false description of Ebeigbe’s multivariate target moments
  - softened the “assembled from familiar components” paragraph so provisional citations are design-space context rather than theorem-level support
- `chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`
  - changed chunk wording so chunk fields are recorded provenance, not caller-free knobs
  - added partial-derivative wording for stop-gradient and forward-vs-reverse caveat language
- `chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex`
  - fixed the fixed-`\varphi` explanation so the ν expectation is not falsely said to anchor the inner minimization
  - adjusted the training algorithm wording to mention the target-side potential update
  - removed one “scalable-OT survey” boundary sentence

### High-dimensional / quadrature / structural repairs
- `chapters/ch34_highdim_gaussian_projection_and_point_rule_foundations.tex`
  - softened the stale promise that the running quadratic cell “reappears here”
- `chapters/ch35_highdim_sparse_grid_quadrature_and_fixed_cloud_scalar.tex`
  - added the explicit 1D three-point-rule application for `h(X)=X^2`
  - removed the literal “the panel asked to see” register leak
  - added an explicit 3D test-function example
  - filled the previously empty “Exactness, Point Count, And The UKF Relation” section with actual content
  - converted the markdown-style exported-forward bullets into a LaTeX `itemize`
- `chapters/ch35b_highdim_fixed_cloud_filtering_and_sgqf_validation.tex`
  - tightened the Lane-B correction sentence so the cloud, not the Gaussian innovation scalar, is the subject of the evidence clause
  - replaced “source note” wording with “earlier derivation”
- `chapters/ch18b_structural_deterministic_dynamics.tex`
  - fixed the comparison-table row so the failure modes are assigned in the intended direction
  - renamed the TOC-visible “Reviewer Edge Case” subsection to a reader-facing title
- `chapters/ch01_introduction.tex`
  - rewrote the Reader Map to match the actual 11-part book structure

### Appendices / bibliography
- `appendices/app_b_matrix_calculus.tex`
  - added the missing symmetry hypothesis to the quadratic-form identity
- `appendices/app_d_mathdevmcp_workflows.tex`
  - updated `extract-latex-context` wording to note `latex-label-lookup` / compatibility alias status
- `appendices/app_e_researchassistant_workflows.tex`
  - rewrote the stale “Current Blockers” section to reflect the current review state
- `references.bib`
  - corrected `li2017particle` author to **Yunpeng** Li
  - corrected `hu2021particle` author to **Chih-Chi** Hu
  - replaced the bad NeuTra proceedings entry with an arXiv-preprint style record
  - appended provisional entries for several missing transport/ensemble keys and for Vehtari / Benamou–Brenier / Del Moral / FilterFlow where needed for the rewrite build

---

## 6. Important edits that did **not** land cleanly and still need attention

During the last large automated text-repair batch, two targeted replacements missed and still need manual editing:

1. `chapters/ch13_custom_gradient_wrappers.tex`
   - the “observed information” paragraph still needs the ambiguity fix so `-H(ψ)` is described as **posterior curvature**, with likelihood-only observed information stated separately if reported.

2. `chapters/ch20_filter_choice.tex`
   - the stale closing pointer still says “The next industrial and case-study chapters...” and should be changed to “The later high-dimensional, geometry, and case-study chapters...”

Also note:
- `appendices/app_c_factor_derivative_proofs.tex` and some other files were already changed in earlier partial batches, so do **not** assume they are still pristine original copies.

---

## 7. Additional defects still expected in the rewrite tree

Even after the landed edits above, expect at least these remaining issues to fix after the next build / audit pass:

1. **Mis-anchored labels in unnumbered displays**
   - `chapters/ch18b_structural_deterministic_dynamics.tex`
   - `chapters/ch33_highdim_nonlinear_filtering_foundations.tex`
   These were identified but not yet fully repaired. The attempted bulk conversion only partially succeeded.

2. **Duplicate/lingering labels and merge-artifact cleanup**
   - `ch32e` / `ch32f` duplicates were reduced, but re-check the rewrite tree mechanically.

3. **Bibliography may still need additional cleanup**
   - the provisional entries added for buildability are intentionally conservative and may need tightening if this rewrite is later promoted beyond internal reading.

4. **Internal-register leaks still present elsewhere**
   - examples seen in the grep inventory include:
     - `Phase 2B literature gate`
     - `row-173 trace`
     - `OT survey companion note`
     - `the panel asked to see`
     - `Reviewer Edge Case`
     - `banking evidence`
     - `current repair lane`
   Some were fixed, but not all.

5. **TOC / short-title / label namespace hygiene**
   - not fully rechecked yet in the rewrite tree.

---

## 8. First commands to run after reboot

From the repo root:

```bash
cd /home/chakwong/BayesFilter/docs/fable-rewrite/monograph
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Then inspect:

- `main.log`
- undefined citations / refs
- first fatal error if any

After that, do a targeted grep for still-live internal-register phrases:

```bash
grep -Rni "Phase 2B literature gate\|row-173 trace\|Lane B\|OT survey companion note\|the panel asked to see\|Reviewer Edge Case\|current comparator\|requested for this program\|the source note\|the scalable-OT survey\|the scalable-OT note\|banking evidence\|current repair lane" chapters appendices
```

---

## 9. Recommended next repair order after reboot

1. **Build immediately** to expose the next hard blocker.
2. **Fix the two missed edits** (`ch13`, `ch20`).
3. **Repair the label-in-unnumbered-display defects** in `ch18b` and `ch33` carefully.
4. **Re-run build**.
5. **Clear remaining duplicate labels / undefined citations / undefined refs**.
6. **Do one reader-facing register cleanup pass** on the worst remaining phrases.
7. **Build final PDF**.

---

## 10. Handoff sentence for the next agent

Use this as the restart brief:

> Continue the standalone monograph rewrite in `docs/fable-rewrite/monograph/`. The rewrite tree already exists and many high-confidence repairs have already landed. Read `docs/fable-rewrite/execution-audit-2026-08-04.md`, `docs/fable-rewrite/bayesfilter-monograph-rewrite-freeze-and-ledger-2026-08-04.md`, and `docs/fable-rewrite/bayesfilter-monograph-rewrite-master-document-2026-08-04.md` first. Then build `docs/fable-rewrite/monograph/main.tex` with `latexmk -pdf -interaction=nonstopmode -halt-on-error`. Two known missed edits still need manual repair: the observed-information wording in `chapters/ch13_custom_gradient_wrappers.tex` and the stale “next chapters” pointer in `chapters/ch20_filter_choice.tex`. After that, continue with label repairs, citation cleanup, and final PDF build.
