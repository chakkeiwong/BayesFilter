# Paragraph-by-paragraph and equation-by-equation publish-readiness review of the standalone BayesFilter rewrite

- **Date:** 2026-08-06
- **Artifact reviewed:** `docs/fable-rewrite/monograph/main.tex` and its active input closure
- **Artifact PDF reviewed:** `docs/fable-rewrite/monograph/main.pdf` (493 pages)
- **Review mode:** reading-order source review, paragraph-level logic pass, equation-level audit, MathDevMCP CLI diagnostics on load-bearing labels, active-root citation/label scan, and build-log inspection
- **Decision:** **REVISE before final release**

---

## 1. Executive verdict

The standalone rewrite is a materially improved repair branch and is now buildable. It fixes many real defects in the original monograph: target-definition wording, particle-filter boundary wording, LEDH centered-offset language, GenUT attribution honesty, ICNN fixed-`\varphi` explanation, and several structural / label / bibliography failures.

However, after a paragraph-by-paragraph and equation-by-equation pass over the active rewrite root, it is still **not publish-ready**. The remaining release blockers are concentrated and real:

1. the squared-TT retained-coordinate derivation still needs a fresh scalar/vector consistency certificate;
2. the SR-UKF negative-weight analytical-score lane remains conditional rather than fully derived in-book;
3. theorem-level source support remains incomplete in the HAC / Li-Coates / survey-support lanes;
4. the canonical LEDH policy subsection still needs one final coherence pass;
5. the ICNN trainer remains schematic rather than fully specified;
6. the release-quality typesetting layer still has substantial warning debt.

This review therefore recommends: **keep the rewrite branch, close the remaining blocker family, then perform one bounded final release review before any canonical promotion.**

---

## 2. Review method and scope

### What was reviewed
I reviewed the rewrite in reading order and paid special attention to paragraphs and displayed equations that carry:
- claim changes,
- proof obligations,
- target-definition changes,
- source-faithfulness claims,
- implementation-policy statements,
- or publication/release claims.

### What was explicitly checked with MathDevMCP
Targeted diagnostic checks were run on the highest-risk equations / labels, including:
- `eq:bf-srukf-filtered-factor`
- `eq:bf-hd-squared-tt-retained-numerator-contraction`
- `eq:bf-custom-primal-target`
- `eq:bf-ssl-lstm-growing-hac`

MathDevMCP returned these as **unverified** / diagnostically incomplete, which matches the human review result: the relevant paragraphs are not yet release-certified.

### What this review does not claim
This is a bounded publish-readiness review, not a claim that every one of the 493 pages has been fully certified equation-by-equation. The active ledger below covers the highest-risk paragraphs and formulas and is sufficient to drive the remaining blocker pass.

---

## 3. Paragraph-by-paragraph / equation-by-equation findings

## 3.1 `ch17_square_root_sigma_point.tex`

### Finding A: filtered-factor target is improved but still not machine-certified
- **Anchor:** `ch17_square_root_sigma_point.tex:311-335`
- **Equation:** `eq:bf-srukf-filtered-factor`, `eq:bf-srukf-filtered-factor-first`
- **Verdict:** `not checked`

The rewrite now states the signed-stack target more honestly than the canonical source. It explicitly includes the negative covariance-weight stack and says omitting it changes the covariance target. That is the right direction.

However, MathDevMCP could not extract the displayed row as a safe proof obligation, so the equation is still not release-certified. The text is also still a branch contract, not a complete derivation certificate.

**Required rewrite:** keep the conditional stance, or supply a derivation note and a deterministic numerical reconstruction fixture for a negative-weight case.

### Finding B: the derivative lane remains incomplete by design
- **Anchor:** `ch17_square_root_sigma_point.tex:341-351`
- **Paragraph:** admission boundary
- **Verdict:** `correct`

The rewrite now honestly says the signed update/downdate derivative is not fully derived in-book and that the analytical-score route is conditional unless an external primitive is supplied and checked. That is policy-compliant and should be preserved.

**Required rewrite:** none unless the primitive is actually derived.

### Policy / language note
This chapter is now more honest than the original, but it still needs one final pass to remove any lingering prose fragment immediately before the filtered-factor derivative equation.

---

## 3.2 `ch12_factor_derivatives.tex` and `appendices/app_c_factor_derivative_proofs.tex`

### Finding C: QR/Cholesky contracts are good, but they do not close the signed SR-UKF branch
- **Anchors:** `ch12_factor_derivatives.tex:263-288`, `app_c_factor_derivative_proofs.tex:1-14`
- **Verdict:** `correct` for the bounded smooth-branch contract; `not checked` for the missing signed primitive

Chapter 12 gives a mathematically sound factor-derivative contract for smooth Cholesky / QR branches. Appendix C is honestly labeled as a placeholder.

But that does not close the signed update/downdate primitive that Chapter 17 still needs for full negative-weight analytical-score release posture.

**Required rewrite:** either derive the ordered signed recurrence or keep the release claim conditional everywhere it is summarized.

---

## 3.3 `ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`

### Finding D: the retained-coordinate lane still needs a fresh derivation certificate
- **Anchor:** `ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:171-277`
- **Equation:** `eq:bf-hd-squared-tt-retained-numerator-contraction`, `eq:bf-hd-squared-tt-dot-retained-numerator`
- **Verdict:** `not checked`

The rewrite now uses a retained-first narrative with right-side contractions. That is better than the earlier inconsistent retained-last story.

But the paragraph is still not publication-grade because it needs a real scalar/vector derivation note that proves the displayed contraction orientation is the right one for the concrete branch. MathDevMCP still returns the corresponding obligation as unverified.

**Required rewrite:** add a short derivation note and a numerical identity check showing the retained-first contraction and its derivative match the intended target.

### Finding E: the defensive scale is now clearer
- **Anchor:** `ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex:197-220, 245-277`
- **Verdict:** `correct` in intent, but still release-sensitive

The rewrite now places the defensive term on the same `e^{-c_t}` scale as the declared represented density, which is the right correction.

However, because the retained-coordinate derivation is still not fully certified, the whole lane should stay in “repair branch” status until the new derivation note is added.

---

## 3.4 `ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`

### Finding F: the query-rule interface is now explicit, but still requires implementation-facing confirmation
- **Anchor:** `ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex:358-390`
- **Equation:** `eq:bf-hd-ttkr-retained-query-rule`
- **Verdict:** `correct` as a contract statement; `not checked` as an implementation-fidelity claim

The chapter now clearly says:
- the saved evaluator is a density under the retained-block reference measure,
- physical points are converted to reference coordinates before querying,
- the Jacobian belongs to the target-construction path, not the stored evaluator.

This is much better than the earlier under-specified version.

But the chapter still needs a final implementation-facing confirmation that the route actually follows that convention. Until then, it is a good contract statement, not a release-certified implementation theorem.

**Required rewrite:** preserve the explicit wording, then add a route-audit note if the actual code convention is to be promoted as final.

---

## 3.5 `ch19_particle_filters.tex`

### Finding G: the post-decision measure is now the right object, but the proof still needs one final tidy-up
- **Anchor:** `ch19_particle_filters.tex:488-555`
- **Equation:** `eq:bf-pf-bootstrap-likelihood-estimator`
- **Verdict:** `correct` in substance; `not checked` in full proof closure

The rewrite now correctly distinguishes the weighted post-decision measure from the resampled unweighted cloud. That is a real improvement and should be preserved.

The proof is mathematically much closer to the right object now, but the exact weight-carrying induction should still be tightened if the final release is meant to be publication-grade rather than merely buildable.

### Finding H: the differentiability boundary is now policy-correct
- **Anchor:** `ch19_particle_filters.tex:566-592`
- **Verdict:** `correct`

The chapter now clearly separates:
- true likelihood,
- unbiased randomized estimator,
- and a smooth relaxed scalar.

That is exactly the right release-facing distinction.

---

## 3.6 `ch19b_dpf_literature_survey.tex`

### Finding I: the LEDH offset is more honest, but source-support closure remains open
- **Anchor:** `ch19b_dpf_literature_survey.tex:547-567`
- **Equation:** `eq:bf-pff-ledh-A`, `eq:bf-pff-ledh-b`, `eq:bf-pff-ledh-ode`
- **Verdict:** `not checked`

The centered-information correction is the right direction. But the passage still depends on source-faithfulness claims that are not fully source-closed in the release package.

**Required rewrite:** keep the honest centered-information wording, but maintain explicit nonclaims until the Li-Coates source closure is finished.

### Finding J: the survey paragraph is now more careful, but not literature-complete
- **Anchor:** `ch19b:688-923`
- **Verdict:** `unsupported` for theorem-level completeness

The design-space paragraph now says clearly that the cited surrounding ingredients support context, not theorem-level composite-contract correctness. That is good.

However, the release still cannot claim full survey completeness or backward/forward snowball closure.

---

## 3.7 `ch28a_neural_network_state_space_model_applications.tex`

### Finding K: HAC passage is correctly downgraded, not certified
- **Anchor:** `ch28a:457-485`
- **Equation:** `eq:bf-ssl-lstm-growing-hac`, `eq:bf-ssl-lstm-hac-bandwidth`
- **Verdict:** `correct` as a bounded audit direction; `unsupported` as a theorem claim

This is now correctly framed as an implementation/audit direction rather than a closed asymptotic theorem.

That is release-honest and should stay that way until the exact source assumptions are inspected and recorded.

### Finding L: validation discussion is policy-compliant but still descriptive
- **Anchor:** `ch28a:815-869`
- **Verdict:** `correct`

The validation discussion correctly separates candidate failure from direction failure.

---

## 3.8 `ch32c_entropic_ot_sinkhorn.tex`

### Finding M: Contract E design-space wording is appropriately narrowed
- **Anchor:** `ch32c:898-918`
- **Verdict:** `correct`

The paragraph now says explicitly that unresolved neighboring records are omitted from theorem-level support and that the paragraph is design-space context only.

That is a good final-release move.

### Finding N: the GenUT local construction is internally consistent, but not a blanket claim about the external paper
- **Anchor:** `ch32c:1677-1723`
- **Equation:** `prop:bf-eot-genut-axis`, `prop:bf-eot-genut-moments`
- **Verdict:** `correct`

The local whitened-moment construction is mathematically coherent as written.

The release note should continue to say that this is a local variant unless the full external paper convention is explicitly re-audited.

### Finding O: the stopped-normalization proposition is a correct partial-derivative warning
- **Anchor:** `ch32c:1888-1944`
- **Equation:** `prop:bf-eot-stopped-normalization-partial`
- **Verdict:** `correct`

This is one of the best release-quality policy repairs in the rewrite. It should be preserved.

---

## 3.9 `ch32c2_ledh_pfpf_ot_custom_gradient.tex`

### Finding P: canonical Contract E policy identifiers are explicit and good
- **Anchor:** `ch32c2:107-119, 2287-2299`
- **Verdict:** `correct`

The canonical route identifiers, historical-route demotion, and chunk/tuning semantics are now explicit. This is the right policy binding.

### Finding Q: still one final coherence pass needed
- **Anchor:** `ch32c2:107-140`
- **Verdict:** `correct` but should be normalized once more

The policy language is strong, but final release should ensure the strongest canonical statement appears once and later prose does not weaken it.

---

## 3.10 `ch32e_icnn_brenier_monge_gap_map_learning.tex`

### Finding R: fixed-`
varphi` explanation is now mathematically honest
- **Anchor:** `ch32e:287-341`
- **Equation:** `eq:bf-neural-ot-direct-icnn-objective`
- **Verdict:** `correct`

This is a real repair. The chapter now correctly says the target-side expectation is constant in `\theta` for fixed `\varphi`.

### Finding S: the trainer is still schematic, not fully specified
- **Anchor:** `ch32e:348-366`
- **Verdict:** `heuristic only`

The algorithm remains a schematic direct-map training pattern rather than a fully canonical trainer. That is acceptable if explicitly labeled, but not if it is promoted as a final exact implementation.

---

## 3.11 `ch13_custom_gradient_wrappers.tex`

### Finding T: observed-information vs posterior-curvature distinction is correct
- **Anchor:** `ch13:85-100`
- **Verdict:** `correct`

This was a needed release fix and should remain exactly as written.

---

## 3.12 `appendices/app_e_researchassistant_workflows.tex`

### Finding U: source-support blocker statement is honest and policy-aligned
- **Anchor:** `app_e:52-55`
- **Verdict:** `correct`

The appendix now acts as a blocker ledger rather than pretending source closure is done. Good.

---

## 4. Policy compliance summary

### Compliant or improved
- Exact target vs modified fallback is clearly distinguished.
- Same-scalar HMC and exact endpoint correction are separated from wrong-value dynamics.
- Evidence/proof/source classes are more explicit than in the original source.
- Canonical LEDH identifiers are now visible.
- The PF and ICNN sections are more honest about target status.

### Remaining policy friction
- Some live review / lane / promotion language still appears in expository areas and should be minimized for a final release edition.
- Several source-faithful claims still need the final primary-source closure or an explicit nonclaim.
- The standalone branch is still a repair branch, not a publication-grade end state.

---

## 5. Rewrite instructions before final release

1. **Squared-TT:** add a short derivation note and numerical check for the retained-first contraction, or keep the section explicitly as a branch contract with a nonclaim.
2. **SR-UKF:** either derive the signed update/downdate primitive or narrow the release claim globally.
3. **Sources:** close or explicitly ledger the HAC and Li-Coates support boundary.
4. **LEDH:** keep the canonical policy section strong and singular.
5. **ICNN:** either specify the trainer fully or keep it explicitly schematic.
6. **Typesetting:** remove the remaining warning and the worst overfull boxes before any publication claim.

---

## 6. Nonclaims

This review does **not** certify every paragraph or equation in the 493-page rewrite. It is a bounded release-readiness audit focused on the highest-risk and load-bearing passages. Unlisted paragraphs and equations remain **not checked**, not implicitly approved.

---

## 7. Final verdict

**REVISE before final release.**

The rewrite is now a strong, buildable, and much more honest branch. But it is still not fully publication-ready because the squared-TT lane needs one last derivation certificate, the SR-UKF derivative lane remains conditional, and the source-support / typography debt is not yet fully closed.

The next pass should be narrow:
- finish the squared-TT derivation note,
- freeze the SR-UKF release posture,
- close the source-support blockers,
- do the final typesetting polish,
- and then request one last bounded independent review.
