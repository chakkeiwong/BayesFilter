# Publish-Readiness Review of the Standalone Rewritten BayesFilter Monograph

- **Date:** 2026-08-06
- **Artifact reviewed:** `docs/fable-rewrite/monograph/main.tex` and its full input closure
- **Artifact PDF reviewed:** `docs/fable-rewrite/monograph/main.pdf` (493 pages, file size 2.1 MB)
- **Review mode:** paragraph-by-paragraph source review, targeted mathematical audit, source-support audit, policy-compliance audit, and MathDevMCP CLI checks on high-risk chapters/equations
- **Decision:** **REVISE before final release. Do not publish this version as the final canonical monograph yet.**

---

## 1. Executive verdict

The standalone rewritten monograph is materially better than the original source tree and is now a genuine, readable, buildable repair branch. It closes the fatal missing-figure build failure, repairs several real target-definition and wording defects, removes active-root label/citation failures, and is substantially more honest about approximation status and source limitations.

However, after a paragraph-by-paragraph publish-readiness review, it is **still not ready as a final release**. The main reason is not build breakage or superficial prose quality; it is that a small number of mathematically load-bearing chapters still do not yet meet the combined standard of:

1. logically complete argument flow,
2. checked derivation or checked primary-source support,
3. explicit non-evasive language,
4. and policy-tight publication support.

The release blockers remain concentrated in five areas:

1. **SR-UKF branch correctness and derivative admission** (`ch17`, `ch12`, `app_c`)
2. **squared-TT branch consistency and interface specification** (`ch36b`, `ch37`)
3. **source-support / theorem-support boundary** (`references.bib`, `ch28a`, `ch32c`, `app_e`)
4. **canonical LEDH policy binding** (`ch32c2`)
5. **final release-quality typesetting / release discipline** (remaining severe overfulls and one remaining `amsmath`-style issue)

The correct next step is therefore:

- keep this artifact as the active rewrite branch,
- apply the targeted fixes listed below,
- rebuild,
- and only then seek a final bounded independent release review.

---

## 2. Baseline mechanical status

Baseline rebuild of `docs/fable-rewrite/monograph/main.tex`:

- **build succeeds:** yes
- **undefined citations:** none
- **undefined references:** none
- **active-root duplicate labels:** none
- **overfull boxes:** 193
- **underfull boxes:** 760
- **remaining foreign-command / amsmath-style warning count:** 1

These metrics matter, but they are **not** sufficient evidence of final-release readiness.

---

## 3. MathDevMCP checks run

I used MathDevMCP CLI in the rewrite root as a release-audit aid.

### Commands run
- `index-latex /home/chakwong/BayesFilter/docs/fable-rewrite/monograph`
- `plan-math-document-rigor-audit` on:
  - `chapters/ch17_square_root_sigma_point.tex`
  - `chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
  - `chapters/ch19_particle_filters.tex`
- `audit-derivation-v2-label` on:
  - `eq:bf-srukf-filtered-factor`
  - `eq:bf-hd-squared-tt-retained-numerator-contraction`
  - `eq:bf-custom-primal-target`
  - `eq:bf-ssl-lstm-growing-hac`

### What the MathDevMCP results mean
MathDevMCP did **not** certify those key labels as verified derivations. Instead it surfaced exactly the kinds of remaining release risks the independent verdict had identified:

- `eq:bf-srukf-filtered-factor` — **unverified**, with manual formalization still required.
- `eq:bf-hd-squared-tt-retained-numerator-contraction` — **unverified**, with manual formalization still required.
- `eq:bf-custom-primal-target` — **unverified**, missing shape/domain constraints for the Jacobian determinant.
- `eq:bf-ssl-lstm-growing-hac` — **unverified**, missing assumption/regularity conditions.

This is not a failure of the tool. It is strong evidence that these passages still need either:

- clearer constraints and scope statements, or
- narrower, more honest wording in the text.

Therefore the final release cannot claim these lanes are fully derivation-closed yet.

---

## 4. Findings by release risk

Findings are ordered by final-release impact.

### 4.1 Blocker: SR-UKF negative-weight branch and derivative admission are still not release-ready

**Files:**
- `docs/fable-rewrite/monograph/chapters/ch17_square_root_sigma_point.tex`
- `docs/fable-rewrite/monograph/chapters/ch12_factor_derivatives.tex`
- `docs/fable-rewrite/monograph/appendices/app_c_factor_derivative_proofs.tex`

**Current status:** improved, but still not fully closed.

#### What is now better
The rewrite no longer falsely implies that Chapter 12 already derives the full signed update/downdate primitive. It now says more honestly that the negative-weight signed update/downdate derivative route is conditional unless an external primitive is supplied and checked.

The filtered-factor text is also improved because it no longer pretends that the positive stack alone plus `KSK^T` downdate defines the same covariance target on negative-weight branches.

#### Why this is still a blocker
The chapter still does not provide a fully self-contained, audited, release-grade derivation of the admitted branch. The text is better, but the release standard is higher than "less false than before." For publication, one of two things must happen:

1. **narrow the claim further** so the book openly states that the negative-weight analytical-score route is not derived in-book and is outside the admitted release contract, or
2. **finish the derivation** with explicit branch/gauge/feasibility conditions and a reconstruction certificate.

MathDevMCP confirms this is still not machine-verifiable as written.

**Classification:** `not checked` to final-release standard; still a release blocker.

---

### 4.2 Blocker: squared-TT lane still needs one final consistency pass

**Files:**
- `docs/fable-rewrite/monograph/chapters/ch36b_highdim_squared_tt_recursion_and_fixed_branch_likelihoods.tex`
- `docs/fable-rewrite/monograph/chapters/ch37_highdim_fixed_branch_likelihoods_and_same_scalar_gradients.tex`

**Current status:** substantially improved, but still not fully publication-closed.

#### What is now better
The rewrite now places the defensive term on the same `e^{-c_t}` scale as the declared represented density and propagates that into the retained-numerator derivative. It also tightens the reference-coordinate query-rule wording and gives the handoff more explicit ownership.

#### Why this is still a blocker
Even after those repairs, this lane still reads more like a mathematically serious **branch contract** than a fully settled publication derivation. The concrete retained-coordinate convention and the route/interface statement are much better than before, but they still need one final internal pass to ensure:

- every sentence uses the same retained-coordinate convention,
- the concrete branch and the generic formulas use exactly matching variable names,
- the Jacobian/reference-measure ownership is stated once and never drifted later,
- and the paragraph flow does not require the reader to infer implementation conventions from surrounding context.

MathDevMCP again classifies the retained-numerator label as unverified and in need of manual formalization, which is exactly consistent with this diagnosis.

**Classification:** `not checked` to final-release standard; still a release blocker.

---

### 4.3 Blocker: theorem-level source support is still not closed in all live passages

**Files:**
- `docs/fable-rewrite/monograph/references.bib`
- `docs/fable-rewrite/monograph/chapters/ch28a_neural_network_state_space_model_applications.tex`
- `docs/fable-rewrite/monograph/chapters/ch32c_entropic_ot_sinkhorn.tex`
- `docs/fable-rewrite/monograph/appendices/app_e_researchassistant_workflows.tex`

**Current status:** much better than before, but still not fully publication-grade.

#### What is now better
- The rewrite no longer leans on live `references.bib` notes that say "provisional" or "verify before publication use" for the active Contract E ingredient paragraph.
- The `ch32c` design-space paragraph is narrower and more honest: it now cites only the already stable nearby sources and explicitly says unresolved neighbors are not used as theorem-level support there.
- The `ch28a` HAC paragraph is no longer written as if theorem-level consistency were already settled. It is now correctly demoted to an implementation/audit direction pending exact source support.
- `app_e` now describes the remaining blocker state much more honestly.

#### Why this is still a blocker
The release standard is not merely “no provisional notes in the active bib entries.” The final release still needs exact source closure for:

- the HAC theorem-level claim boundary,
- the Li–Coates durable local-copy and exact citation boundary,
- any remaining survey/design-space passages that still trade on uninspected neighboring work.

So this cluster has been **stabilized**, but not fully closed.

**Classification:** `unsupported` for final publication strength in a few remaining theorem-level lanes; still a release blocker.

---

### 4.4 Blocker: canonical LEDH policy text is now present, but needs one coherence pass

**File:**
- `docs/fable-rewrite/monograph/chapters/ch32c2_ledh_pfpf_ot_custom_gradient.tex`

**Current status:** major improvement.

#### What is now better
The rewrite now explicitly binds:
- `contract_e_chol_v1`
- `contract_e_chol_total_direct_moments_weights_plus_streaming_transport_v1`
- `dpf_transport_exact_divisor_cap3000_v1`
- `historical_raw_barycentric_diagnostic_only`

It also states the exact-divisor chunk rule and per-scope tuning semantics more explicitly than before.

#### Why this is still not fully passed
This is close, but because the chapter carries both mathematical exposition and policy-binding prose, it needs one final coherence pass to make sure:

- the identifiers appear exactly once in their strongest canonical statement,
- later benchmark prose never weakens or blurs them,
- and the chapter never sounds like policy evidence is mathematical evidence.

This is not the largest remaining blocker, but it is still part of the final release gate.

**Classification:** `correct but should be normalized once more before release`.

---

### 4.5 Major: the particle-filter proof is now much more honest, but still needs final polishing

**File:**
- `docs/fable-rewrite/monograph/chapters/ch19_particle_filters.tex`

The rewrite improved the proof object by distinguishing the post-decision measure and the weighted carried branch. That is a real repair.

However, for final release, the proof should still be tightened one more time so that the weighted post-decision branch is introduced with fully parallel notation and the reader does not have to infer the two-branch measure logic from prose.

This is no longer a top mathematical blocker, but it is still a release-quality refinement item.

**Classification:** `heuristic only` for some explanatory prose; not the main blocker anymore.

---

### 4.6 Major: the ICNN section is mathematically improved but still epistemically split

**File:**
- `docs/fable-rewrite/monograph/chapters/ch32e_icnn_brenier_monge_gap_map_learning.tex`

The rewrite correctly repaired the fixed-`\varphi` explanation, which was one of the real reviewed defects.

The remaining issue is not the same logical error. It is now an epistemic/readability problem:
- the section still oscillates between “concrete canonical objective” and “schematic practical reading,”
- the pseudocode still depends on a target-side update object whose exact role is not fully fixed in the local exposition.

For final release, choose one:
1. fully specify the trainer, or
2. clearly relabel the section as a schematic direct-map pattern.

**Classification:** `not checked` to publication-grade specification standard.

---

### 4.7 Major: the document is buildable, but still not typeset to final-release quality

The artifact now builds cleanly enough for review, which is a major improvement over the original source tree.

But the current PDF still carries:
- 193 overfull boxes,
- 760 underfull boxes,
- one remaining foreign-command / amsmath-style warning,
- and a likely concentration of wide displays/tables/paths that require manual triage.

This is not a correctness defect, but it is absolutely a **release-quality** defect.

**Classification:** `correctness-neutral but final-release blocking`.

---

## 5. What is already good enough and should be preserved

The next pass should preserve these rewrite gains rather than redo them:

1. buildability of the standalone tree,
2. figure-path repair via available assets,
3. corrected Reader Map,
4. corrected finite-fallback target wording,
5. corrected HMC same-scalar / exact-endpoint-MH distinction,
6. improved PF no-resampling algorithm,
7. repaired LEDH offset passage,
8. repaired GenUT attribution language,
9. repaired fixed-`\varphi` explanation,
10. corrected structural-UKF comparison row,
11. active-root label/citation cleanup,
12. quadrature worked-example improvements,
13. stop-gradient partial-derivative wording.

These are real strengths of the rewrite branch.

---

## 6. What should be changed before final release

In order:

1. **Finalize SR-UKF release posture**
   - either derive the admitted negative-weight branch fully,
   - or narrow the release claim so the analytical-score route is explicitly conditional / incomplete there.

2. **Finalize squared-TT release posture**
   - do one last consistency pass over `ch36b`/`ch37` so the branch contract is internally unified and the interface no longer reads as a design memo.

3. **Close the remaining theorem-support blockers**
   - HAC support,
   - Li–Coates durable support,
   - any remaining survey/design-space passages that still outrun inspected source support.

4. **Normalize the canonical LEDH policy subsection**
   - one clean authoritative statement, no drift.

5. **Choose one ICNN status**
   - fully specified, or explicitly schematic.

6. **Do a real release-quality typesetting pass**
   - large overfulls first,
   - final warning cleanup,
   - visual inspection of worst pages.

---

## 7. Section-by-section release-status summary

| Section cluster | Status |
|---|---|
| Parts I–III derivative spine | **Needs targeted math/policy review**, especially `ch12`, `ch17`, `app_c` |
| PF / DPF foundation cluster | **Substantially improved**, but `ch19` proof and `ch19b` source-support framing still need final tightening |
| OT / Contract E cluster | **Much better mechanically and conceptually**, but final release still depends on source-support closure and one canonical-policy normalization pass |
| Learned-map cluster | **Improved**, but ICNN trainer/status still needs one explicit choice |
| High-dimensional TT/KR cluster | **Still replacement-blocking** until the branch is fully unified |
| Case-study and application material | **Mechanically acceptable**, but `ch28a` HAC theorem boundary still not closed |
| Appendices | **More honest than before**, but `app_c` remains intentionally incomplete and must be treated that way in final release claims |

---

## 8. Nonclaims

This review does **not** conclude that:
- every unchanged paragraph in the 493-page rewrite is mathematically certified,
- bibliography completeness is fully established,
- every cited paper has undergone a full backward/forward snowball audit,
- or the rewrite is ready to replace `docs/` immediately.

It concludes only that:
- the rewrite is a strong, buildable repair branch,
- several major defects have been materially improved,
- and a narrow but still real release blocker set remains.

---

## 9. Final verdict

**REVISE before final release.**

The standalone rewritten monograph is close enough that it should be preserved and finished, not discarded. But the release standard the owner asked for — thoroughly checked logic, checked derivation support, checked source support, MathDevMCP-assisted auditability, and policy-compliant publish-ready language — is not fully met yet.

The next pass should therefore be a **narrow final-release pass**, not another broad rewrite:

1. close the SR-UKF release boundary,
2. close the squared-TT release boundary,
3. close the theorem/source-support blockers,
4. normalize canonical LEDH policy wording,
5. settle the ICNN trainer status,
6. and do the final typesetting pass.

After that, one last bounded independent review should be enough to decide whether to promote this branch into the canonical `docs/` tree.
