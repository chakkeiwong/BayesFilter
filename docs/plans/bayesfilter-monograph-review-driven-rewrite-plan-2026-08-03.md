# BayesFilter Monograph Review-Driven Rewrite Plan

- **Date:** 2026-08-03
- **Plan type:** documentation-repair campaign plan (monograph text, bibliography, and build), not a numerical experiment plan.
- **Inputs (adjudicated review record):**
  - `docs/plans/bayesfilter-monograph-main-readonly-review-revised-2026-08-03.md` (revised review; authoritative finding list and evidence classes)
  - `docs/plans/bayesfilter-monograph-main-readonly-review-revised-reply-2026-08-03.md` (independent reply; final adjudication corrections)
  - `docs/plans/bayesfilter-monograph-main-readonly-review-findings-2026-08-03.md` (independent audit; artifact-boundary corrections)
- **Execution status:** PLAN ONLY. No monograph source is changed by this document. A plain-language owner request to execute a phase is sufficient authorization for that phase.
- **Worker standard:** each work order below is written to be executable by a first-year-PhD-level worker with the stated file/line anchors, the stated fix, and the stated check. Where a fix requires a derivation, the target identity is given in the work order.

---

## 0. Question, goal, and evidence contract

**Question this plan answers:** what exact sequence of edits takes the monograph from its current state (real target/algorithm defects, unbuildable tree, incomplete source support, stale structure, internal register leakage) to a state that passes a clean rebuild, a re-run of the mechanical audits, and a bounded re-review?

**Primary success criterion (campaign level):**
1. `latexmk -pdf` from a clean tree completes with **zero missing files, zero undefined citations, zero undefined/multiply-defined labels**;
2. every finding classified **CONFIRMED** in the revised review §2 table is repaired and passes its per-item check;
3. every finding classified **UNDER-SPECIFIED** is resolved by an explicit specification (not silently deleted);
4. every **SOURCE-GAP BLOCKER** is either closed with an inspected source or explicitly downgraded to a stated non-claim in the text.

**Veto diagnostics (any one blocks phase completion):** a repaired equation that fails its stated recomputation check; a rebuild that reintroduces undefined citations/labels; a "fix" that deletes a derivation instead of repairing it (page-count guardrail below); a policy-text edit that contradicts repository `CLAUDE.md` rules.

**Explanatory-only diagnostics:** overfull-box counts, prose-style observations, page-count drift within the guardrail.

**What is not concluded by this plan:** that the unlisted remainder of the monograph is correct; that the bibliography will be complete after Phase 4 (it will be *consistent and resolvable*, not exhaustively snowballed); that any scientific claim in the book is promoted.

**Guardrails (owner-directed, from recorded feedback):**
- **Expansion, not compression:** readability repairs add derivations, worked examples, and definitions; they do not replace mathematics with prose. Duplicated *prose/scaffolding* may be consolidated by cross-reference; displayed derivations are never deleted, only corrected or moved. Track total page count per phase; a net loss of mathematical content is a defect.
- **Panel standard:** the review panel includes implementation engineers with veto power; every repaired construction must remain implementable from the text alone.
- **Read-only sources:** original source projects and papers are read-only inputs; fixes cite them, never edit them.

**Decision point D0 (owner) — artifact freeze.** Choose the repair target:
- **Option A (recommended):** repair the **current 55-chapter source tree** and regenerate `docs/main.pdf` from it at the end of Phase 1. The committed PDF is treated as a historical output, not a repair target. All ch26c-dependent reconciliations are then in scope.
- **Option B:** first reproduce a 54-chapter PDF-state baseline, then apply repairs on top. Higher effort, only worth it if the owner needs a byte-comparable artifact history.
This plan is written for Option A; Option B changes only Phase 0.

---

## 1. Phase 0 — Freeze, build restoration, and baseline (blocking; do first)

Nothing downstream can be verified until the tree builds.

| ID | Work order | Anchor | Fix | Check |
|---|---|---|---|---|
| P0.1 | Restore the five missing figures | `ch28a_neural_network_state_space_model_applications.tex:1002, 1012, 1031, 1077, 1098` | Either (a) regenerate the five `.pdf` figures from their producing scripts under `plans/artifacts/ssl-lstm-neutra-2026-07-14/...` and track them, or (b) change the five `\includegraphics` extensions to the tracked `.png` files. (b) is the low-risk default; record which option was taken. | isolated `latexmk -pdf -halt-on-error -outdir=/tmp/...` passes the previous failure point |
| P0.2 | Full clean rebuild + bibtex | `docs/main.tex` | `latexmk -pdf` twice with bibtex; regenerate `.bbl` | build log: `ebeigbe2025genut` and `flegal2010batchmeans` resolve; remaining undefined citations are exactly the eight known-missing ch32c keys (fixed in Phase 4) |
| P0.3 | Baseline audit snapshot | build log, `main.aux` | record: page count, chapter count (must be 55), undefined-citation list, multiply-defined-label list, overfull count | this snapshot is the Phase 11 comparison baseline |
| P0.4 | Quarantine orphan chapter sources | `chapters/ch34_highdim_gaussian_and_sparse_quadrature.tex`, `chapters/ch35_highdim_particle_transport_tensor_filters.tex`, `chapters/ch36_nonlinear_ssm_hmc_research_program.tex`, `chapters/ch37_highdim_filtering_candidate_synthesis.tex`, and `chapters_restart_staging/` | move to `chapters_archive/` (or add a `% ARCHIVED — not in main.tex` header if moving is undesirable); remove the duplicate chapter labels they carry (`ch:bf-highdim-hmc-research`, `ch:bf-highdim-candidate-synthesis`, `ch:bf-highdim-gaussian-quadrature`) from either the orphan or the active alias so each label is defined exactly once | grep: each `ch:bf-*` label defined exactly once across `chapters/`; build unaffected |

**Phase gate:** clean build; baseline recorded. Estimated size: S (half a day).

---

## 2. Phase 1 — Confirmed mathematical and target-definition repairs

These are the review's CONFIRMED errors. Each work order states the target identity so the fix is checkable.

### M1. SR-UKF filtered-factor identity (ch17)
- **Anchor:** `ch17_square_root_sigma_point.tex:286–299` (`eq:bf-srukf-filtered-factor`).
- **Defect:** `C_{t|t} = SR_{B_t}(C_{xx,+}, C_{KS})` reconstructs `P_{xx,*} + E₋E₋ᵀ − K S_* Kᵀ` on negative-weight branches.
- **Fix:** change the arguments to `C_{t|t} = SR_{B_t}(C_{xx,+}, [C_{xx,−}, C_{KS}])` (or reformulate via the completed predicted factor) so that `C_{t|t}C_{t|t}ᵀ = P_{xx,*} − K S_* Kᵀ` holds on **both** signed branches; add one sentence noting the branch on which the old form was valid.
- **Check:** symbolic/numeric recomputation of the reconstruction identity on a negative-weight instance (e.g. L=5, α=0.1, κ=0, β=2, where w₀^{(c)} ≈ −96) and on the all-positive CUT4-G branch. Preserve the check as a short script or worked note under `docs/plans/` per the reproducibility lesson of the review cycle.

### M2. Signed update/downdate derivative calculus (ch12 → ch17, app_c)
- **Anchor:** `ch17:185–189` (promise), `ch12_factor_derivatives.tex` (missing content), `appendices/app_c_factor_derivative_proofs.tex` (placeholder).
- **Defect:** ch17 attributes the update/downdate derivative primitive to ch12; ch12/app_c contain none.
- **Fix (two acceptable routes; pick one, state it):**
  1. **Add the calculus:** a new ch12 section deriving the ordered rank-one `cholupdate`/`choldowndate` derivative on a fixed full-rank, fixed-sign branch (differentiate `L₊L₊ᵀ = LLᵀ ± vvᵀ` via the ch12 Φ-operator applied to `L̇Lᵀ + LL̇ᵀ ± (v̇vᵀ + vv̇ᵀ)`), state the failure/branch conditions (downdate aborts when the target matrix is not PSD — tie to M1's negative-weight discussion), and extend or explicitly scope out the rectangular rank-deficient case that ch17:84–88 allows.
  2. **Narrow the promise:** rewrite ch17:185–189 to say the primitive is *not* supplied in this book, cite an external source for it, and mark the analytical-score chain as conditional on that primitive.
- **Fix (both routes):** app_c either receives the corresponding proofs or its placeholder text is updated to state exactly which proofs exist where; ch17's downdate-failure discussion gains the one-paragraph mathematical cause (indefiniteness under negative weights; cite Julier(96) p. 9).
- **Check:** route 1 — finite-difference check of the derived downdate derivative on a 3×3 instance; route 2 — grep confirms no remaining ch17 text promises ch12 content that ch12 lacks.

### M3. LEDH offset formula (ch19b) + downstream references
- **Anchor:** `ch19b_dpf_literature_survey.tex:550–554` (`eq:bf-pff-ledh-b`), source form at `ch19b:594–611`, downstream `ch19f:329, 537`.
- **Defect:** the bracket carries the uncentered information vector η⁽ⁱ⁾.
- **Fix:** replace the bracket term with the centered information `η_t^{(i)} − Λ_t^{(i)} m_{t|t−1}` so that `b^{(i)} = d m^{(i)}_λ/dλ − A^{(i)} m^{(i)}_λ` holds with the chapter's own local mean `m^{(i)}_{t,λ}`; add a short reconciliation paragraph showing the corrected form and the Li–Coates source form at 594–611 agree (they are algebraically identical once the fixed λ=0 anchor is substituted), so a reader sees one flow, not two.
- **Check:** (i) linear observation reduction: with `G^{(i)}=H_t` the corrected LEDH must reduce exactly to the chapter's EDH display; (ii) a preserved numerical endpoint test: integrate the corrected flow on a small linear-Gaussian case and verify the endpoint matches the Kalman mean to integration error. Preserve the script.

### M4. Bootstrap-PF estimator on the skip-resampling branch (ch19)
- **Anchor:** `ch19_particle_filters.tex:375–398` (algorithm), `415–423` (estimator), `462–534` (proof).
- **Defect:** Step 3 assigns `w̃_t^{(i)} = p_θ(y_t|x_t^{(i)})` (dropping carried weights); the proof identity `π̄_{t−1}=π̂_{t−1}` is false on the permitted carry branch.
- **Fix (choose one, state it):**
  1. **Carry weights correctly:** define post-decision weights `w*_{t−1}^{(i)} = 1/N` if resampled, `w_{t−1}^{(i)}` otherwise, set `w̃_t^{(i)} = N · w*_{t−1}^{(i)} · p_θ(y_t|x_t^{(i)})`, keep `Z_t^N = N^{-1}Σᵢ w̃_t^{(i)}`, and repair the proof branch to use the weighted empirical measure.
  2. **Restrict the algorithm:** require resampling at every step in the algorithm box and state the skip/adaptive variant separately as a modification whose estimator requires route 1's weights.
- **Additional fixes in the same pass:** add the initialization step (`x₀^{(i)} iid ∼ p_θ(x₀)`, `w₀^{(i)}=1/N`) and the induction base `E[π̂₀^N(φ)] = γ₀(φ)`; make eq `bf-pf-bootstrap-weight` and Step 3 consistent (the "∝" is not innocuous for `Z_t^N`); add the one-line Jensen statement `E[log p̂_θ^N] ≤ log p_θ(y_{1:T})` (strict unless the estimator is a.s. constant) with the −Var/2 heuristic; add a log-domain/logsumexp accumulation remark (float32 default makes `∏Z_t` underflow).
- **Check:** a small preserved Monte Carlo check (N=64, T=3, linear-Gaussian) comparing `E[∏Z_t^N]` against the Kalman likelihood under (a) resample-every-step, (b) skip-at-t=2 with the corrected weights — both must be unbiased within Monte Carlo error; the uncorrected skip branch may be shown as the counterexample.

### M5. Finite-fallback target definition (ch03, ch23) and reconciliation with ch26c
- **Anchor:** `ch03_hmc_target_requirements.tex:36–44`, `ch23_boundary_gradients.tex:13–23`, current-source `ch26c:855–861`.
- **Defect:** the "preferred" finite fallback silently defines a modified target (density e^c on the invalid region; improper if that region has infinite measure), and current source contradicts ch26c's `U_F=+∞` contract.
- **Fix:** in ch03, state the modified target explicitly: define `ℓ̃(u) = ℓ(u)` on the valid region and `= c` (or a stated fallback function) on the invalid region; state (i) the MH-invariant law is `∝ exp(ℓ̃)`, (ii) the bias is governed by the invalid-region mass under `exp(ℓ̃)`, (iii) the improper-target failure mode on unbounded invalid regions, (iv) the exact `−∞` alternative and its cost (non-finite branches under compilation), and (v) the practical near-equivalence when `c` is low enough that acceptance underflows — as an explicitly floating-point argument, not a measure-theoretic one. In ch23, replace "keeps sampler control flow alive long enough for HMC to reject bad proposals" with wording that does not assert guaranteed rejection, and cross-reference the ch03 definition. Add one sentence in ch03 or ch26c reconciling the two contracts (e.g. `U_F=+∞` is the exact contract; the finite fallback is the declared-modified-target engineering contract, and which one a backend implements must be recorded).
- **Check:** the three passages, read together, name the same two admissible contracts with consistent vocabulary; MathDevMCP `search-latex` for "fallback" shows no remaining passage claiming exactness for the finite variant.

### M6. Surrogate-gradient safety wording (ch03, ch13) aligned with ch26c's theorem
- **Anchor:** `ch13_custom_gradient_wrappers.tex:4–7`; wording family at `ch03:24–27, 41–43`; theorem in current-source `ch26c`.
- **Defect:** "safe for HMC only if it returns the derivative of the same scalar…" is wrong as a necessity claim for MH-corrected HMC; ch26c proves the counterexample theorem.
- **Fix:** rewrite the ch13 lede (and the ch03 sentences) to the correct decomposition: with exact target values in the MH acceptance and a deterministic position-only force, leapfrog remains a volume-preserving involution and the invariant law is exact — a wrong force costs **efficiency** (acceptance collapse), not correctness. Correctness genuinely fails when (i) the chain is unadjusted, (ii) the acceptance uses a surrogate *value*, or (iii) the force is stochastic/state-dependent in a way that breaks the involution. Keep the same-scalar rule as the stated *engineering default* (efficiency, diagnosability, uncorrected paths) — the rule survives; only its justification changes. Add a forward reference to the ch26c theorem.
- **Check:** grep for "bias" in ch03/ch13 confirms no remaining unconditional bias claim; the ch26c cross-reference resolves.

### M7. GenUT attribution (ch32c; current-source-only)
- **Anchor:** `ch32c_entropic_ot_sinkhorn.tex:1684–1704`, propagating to `2070–2074`, `2209–2218`.
- **Defect:** the section matches whitened moments `E[Z_a³], E[Z_a⁴]` (Z = C⁻¹(X−m)) and describes Ebeigbe (2025) in those terms; the paper matches *physical diagonal* skewness/kurtosis via `(√P^{∘3})⁻¹ Š`, `(√P^{∘4})⁻¹ Ǩ` (Eqs. (21)/(25)/(28)/(31), Alg. 1, Thm 1(3),(4)).
- **Fix (choose one, state it):** (1) change the constructed targets to the paper's physical-diagonal prescription, or (2) keep the whitened-moment variant but relabel it explicitly as a **local variant that is not the paper's construction**, state precisely what the paper's Theorem 1 matches, and correct the two downstream sentences that currently describe the paper in the variant's terms. Route 2 is smaller and honest; route 1 is more useful if the repo code follows the paper (check `bayesfilter/highdim/cubature_genut_*.py` first and match the text to the code's actual convention — record which convention the code implements).
- **Check:** the section's attribution sentences match the paper's theorem statement; a diagonal-√P example shows where variant and paper coincide; the code-vs-text convention is recorded.

### M8. Squared-TT lane: two confirmed inconsistencies + one under-specified handoff (ch36, ch36b, ch37)
- **M8a — missing `e^{−c_t}`.** Anchor `ch36b:190–206` (`eq:bf-hd-squared-tt-retained-numerator-contraction`). Fix: multiply the defensive term by `e^{−c_t}` (or restate the convention so that `a_t` is defined on the source scale and the evidence scale is applied once, consistently with `ch36:211–218` and `ch36b:44–49`). Check: `∫a_t dz_D = Ẑ_t = e^{−c_t}(R_t + τ_t)` holds by direct integration of the stated convention.
- **M8b — retained-coordinate order.** Anchors `ch36b:171–173` (retained-last derivation) vs `ch36b:363–372` (retained-first concrete branch). Fix: pick one ordering for the whole lane (recommend matching the repo implementation — inspect `bayesfilter/highdim/squared_tt.py` and the Zhao–Cui wrapper before choosing), rewrite the other passage, and add a one-line translation to the source paper's `(x_t, θ, x_{t−1})` convention. Check: the marginal-contraction formula and the concrete branch, composed literally, retain the correct block.
- **M8c — physical/reference handoff (UNDER-SPECIFIED; specify, do not "fix an error").** Anchor `ch37:366–381` (query rule), with `ch36:226–234` (the physical-filter equation that already states the conversion). Fix: state the interface explicitly — the saved evaluator consumes reference coordinates; the time-(t+1) fitting points are physical; therefore the query rule must state `z_t^{query} = Ψ_{t,ret}^{−1}(x_t^{(j)})` and whether the Jacobian factor `∏_k h_{t,k}^{−1}` is applied by the caller or absorbed in the stored object. If the repo code already fixes this convention, document the code's convention. Check: a worked one-step handoff (can reuse ch35b's oracle) shows the normalizer is invariant to the domain-map choice.
- **M8d — companion clarifications from the review:** `M_{<j}` naming (`ch36b:176–187`) — rename to `M_{≤j}` or redefine; representability condition for `a_t = bᵀQb` (`ch37:334–347`) — state the span condition and that it holds for the declared minimal branch; add the realized sweep-stop index `s*` to the branch-identity ledger and `𝓡_FD` (`ch36b:665–686`, `ch37:130–135`, `ch38:220–224`); resolve the init contradiction (`ch36b:522–531` vs `735–737`) by declaring one initialization as *the* branch definition.

### M9. ICNN objective (ch32e)
- **Anchor:** `ch32e_icnn_brenier_monge_gap_map_learning.tex:287–301` (explanation before the display), `306–328` (objective), `343–362` (algorithm).
- **Defect:** for fixed φ, `E_ν[φ(Y)]` is constant in θ; "anchors the target distribution" is wrong for the displayed min-only form; the algorithm has no φ-update.
- **Fix:** display the semi-dual maximin `sup_φ inf_θ { E_μ[½‖X−∇ψ_θ(X)‖² − φ(∇ψ_θ(X))] + E_ν[φ(Y)] }` (or the Makkuva-style alternating parameterization) as the canonical objective; rewrite the anchoring sentence to say the ν-term constrains the **outer φ problem**, and that for a *fixed* φ the inner problem ignores ν (state this plainly — it is exactly why the φ-update is required); add the φ-update step to the algorithm; move the explanation paragraph after the display. Cite the maximin-ICNN source (Phase 4 adds the entry if missing).
- **Check:** the fixed-φ statement and the maximin statement are both present and mutually consistent; the algorithm has an explicit φ-step.

### M10. Structural-UKF contrast table (ch18b)
- **Anchor:** `ch18b_structural_deterministic_dynamics.tex:818–820`.
- **Fix:** swap the two "Failure if misused" cells so the additive-noise UKF applied to the mixed structural model carries "model-law error plus moment approximation error" and the structural UKF carries "moment/quadrature approximation error"; rename the row header to name the misuse explicitly (e.g. "Failure when applied to the mixed structural model of this chapter").
- **Check:** row content agrees with `ch18b:1112–1201` and §19.9.

### M11. Small confirmed identities and hypotheses (batch)
| Anchor | Fix |
|---|---|
| `app_b:20–26` | add "symmetric (positive definite)" to identity 3's hypothesis |
| `ch18b:918–970` | import the integrability/growth hypothesis from the sibling proposition at 2109 |
| `ch18b` (scaled-UT weights, 377–381, 483–494, 573–581) | add the one-sentence PSD/invertibility caveat: all `w_j^{(c)} ≥ 0` plus `R_t ≻ 0` suffices; negative weights void the guarantee |
| `ch32c:2101–2113` | add one sentence: the τ-floored iteration targets a different fixed point (`a⊙(Kb) = u − τa`), declared as such |
| `ch32c:531–560` | restate the hypothesis: (X,J) jointly distributed *according to* the coupling |
| `ch32d:550–558` | state the Nyström identity's three hypotheses (entrywise-max norm, unit-diagonal kernel, PSD residual) and cite the checkable source |
| `ch36:240–253` | add the compatibility condition `(T_t)_♯ ρ̂_t ∝ η` (Zhao–Cui eq. (30)) |
| `ch35b:125–128` | rewrite the sentence so the **cloud** (not the Gaussian innovation scalar) is the subject of the "evidence only when…" clause, with the direct-reweighting scalar named as `Ẑ^dir` |
| `ch35b:559–562` | add one sentence stating the Ċ_xz simplification uses `Σ_r w_r = 1` |
| `ch33:260–272` | add the one-line hardening (log-χ²₁ mean ψ(½)+log 2 ≈ −1.27, skewed ⇒ generic value difference at t=1) or soften "changes" to the target-identity claim |
| `ch18b:1252–1260` | fix "collapses to a point" → the support is the line {(m,0)} when γ=0 |
| `ch28a:1209, 1249` | `y^star` → `y^\star` (twice) |
| `ch14:102–116` | add the dtype policy: FD plateau validation requires float64; state expected plateau scales in float32 vs float64 |
| `ch38:308–319` | give the float32 conditioning default alongside the double-precision one; define `ε_quad`; complete `V_grad^required` |
| `ch28a:791–795` | classify R-hat's role and threshold for that run (promotion criterion / veto / explanatory) |
| `ch28a:1132–1159` | clarify the roles of the 2-GB warning threshold vs the 600-second cap in the q=20 decision (the reply's adjudication: unclear roles, not a contradiction) |
| `ch19e:124–127` | weaken "volume preservation" from necessity to the standard sufficient condition; note the Jacobian-corrected alternative the chapter's own table allows |
| `ch19e` | add the missing taxonomy cell (fresh randomness + biased value inside MH) to the target-status trichotomy |

**Phase-1 gate:** every M-item's check passes; checks that involve computation are preserved as scripts or worked notes under `docs/plans/` (reproducibility lesson). Estimated size: L (the largest phase; M1–M4 and M8 are the heavy items).

---

## 3. Phase 2 — Policy-text reconciliation (repo governance vs monograph prose)

| ID | Anchor | Fix | Check |
|---|---|---|---|
| S1 | `ch32c2:2287–2289` (+ 682–684) | name the canonical selector `dpf_transport_exact_divisor_cap3000_v1`, the equal-chunk rule, the `K=N` rule *scoped to N≤3000* with the largest-divisor rule above, and the fail-closed/no-override validation; state that η_OT chunk fields **record** the repository-selected values for reproducibility and are not caller-free knobs | wording matches `bayesfilter/highdim/transport_chunk_policy.py` and repo `CLAUDE.md`; stale unconditional `K=N` phrasing removed |
| S2 | `ch32c2:49–58, 431–434, 548–551` | state the same-scalar/stop-gradient compatibility condition: the VJP equals `D_θ 𝓛_K` **iff** the stop set is empty, θ-independent, or derivative-neutral; otherwise the returned covector is the exact gradient of a *defined modified map* (name it); note the canonical route currently stops only a derivative-neutral LSE shift | proposition statement carries the condition; no remaining unconditional "equals the VJP of the executed finite program" |
| S3 | `ch32c2` (chapter-wide), `ch20`, wherever LEDH runs are claim-adjacent | add one paragraph stating the repo's **LEDH per-scope tuning rule** (offline tuning artifact per exact scope; transferred settings are warm starts only) and mark the K=20/L=8 witness settings as scope-bound, non-default | grep "per-scope"/"tuning artifact" now hits the monograph; witness settings carry the warm-start label |
| S4 | `ch20:38–45, 65` | rewrite step (iv) to name the promoted strict-SPD principal-square-root backend explicitly (`tf_principal_sqrt_ukf` route per ch18) and reserve "historical SVD/eigenderivative" for the demoted register entry; fix the closing pointer (next part is High-Dimensional Filtering, not the case studies); add `\ref`s to ch18's promoted backend and to the two gate tables (`tab:bf-dpf-hmc-banking-evidence` → renamed per R-phase, `tab:bf-dpf-debug-promotion-thresholds`) | register and procedure use disjoint vocabulary; all refs resolve |
| S5 | `ch32c2:589–591 vs 1779–1781, 2336–2338` | state the forward-JVP vs reverse-VJP regime split explicitly (small d_θ → forward; HMC-scale d_θ → reverse) and which is canonical for which route | one sentence at the contract; no remaining "canonical" ambiguity |
| S6 | `ch38:434–443` | add the Statistical-Evidence sentence at the comparison stage: ranking screen-passing candidates requires predeclared uncertainty evidence; descriptive metrics nominate only | matches policy text |

**Phase gate:** no monograph sentence contradicts a repo `CLAUDE.md` rule; MathDevMCP `search-latex` spot-checks. Size: S–M.

---

## 4. Phase 3 — Source-gap blockers (close or downgrade)

Use `research-assistant` (`.research/ra-bayesfilter-monograph` workspace) for each: fetch → verify metadata → inspect the technical section → write the claim-support note → only then edit text/bib.

| ID | Blocker | Action | Downgrade path if source unavailable |
|---|---|---|---|
| G1 | HAC consistency (`ch28a:459, 468–476`) | obtain Newey–West (1987) and Flegal–Jones (2010) (and Andrews (1991) for the `ℓ²/N→0` Bartlett result); restate the claim with the theorem-true conditions and bandwidth, cite section/theorem numbers, store local copies | restate as "consistency holds under conditions of [source]; the `2+η` assumption stated above does not by itself deliver it" — an explicit non-claim |
| G2 | Acevedo–de Wiljes–Reich Definition-3.1 anchor (`ch32c:725, 852–888, 936–944`) | obtain the paper; add the bib entry (also closes one of the eight missing keys); verify the Definition 3.1 / Sections 5–6 attribution; store a local copy | keep the citation but drop the definition-level anchor to a section-level "cf." until inspected |
| G3 | Li–Coates durable local copy | store the paper under `.localresources/papers/` (policy requires a local copy of the decision-driving source); re-verify the ch19b source-form block against it once stored | record the gap explicitly in the source map |

**Phase gate:** each G-item has either an inspected-source note or an explicit in-text non-claim. Size: S–M (mostly retrieval + focused reading).

---

## 5. Phase 4 — Bibliography and attribution repairs

All edits to `docs/references.bib` plus in-text `\citep` additions. Verify every new/changed entry's metadata via research-assistant before committing it (access-dated).

| ID | Item | Fix |
|---|---|---|
| B1 | Eight missing keys (`acevedodewiljesreich2017secondorder`, `bach2026learning`, `hosseini2025conditional`, `kirchgessner2017smoother`, `lei2011moment`, `todter2015secondorder`, `vanleeuwen2020consistent`, `zeng2026coupling`) | add verified entries (G2 covers one); rebuild; confirm printed p.≈169 region renders no `[?]` |
| B2 | `hoffman2019neutra` | correct to the arXiv record (arXiv:1903.03704, 2019; optional note on AABI presentation); remove the wrong ICML/PMLR volume/pages |
| B3 | `li2017particle` | author → **Yunpeng** Li (DOI 10.1109/TSP.2017.2703684) |
| B4 | `hu2021particle` | author → **Chih-Chi** Hu; verify issue/pages against DOI 10.1002/qj.4028 |
| B5 | `daumhuang2008` | resolve to the actual record(s): SPIE 2008 log-homotopy paper and/or SPIE 2010 exact-flow paper; add the **2010 exact-flow** entry and cite it where ch19b derives the EDH coefficients |
| B6 | Missing load-bearing entries + in-text citations | add and cite: Vehtari et al. 2021 (at `ch26b:320–352`); Amos et al. 2023 Meta OT (in ch32d's primary lane — entry exists, orphaned); Benamou–Brenier 2000 (+ optionally McCann 1997) in ch32f with the theorem stated; Karkus et al. 2018 (ch32a, to disambiguate "soft resampling"); FilterFlow software citation (ch19f); Del Moral 2004 (ch19 proof pointer); Makkuva et al. (M9); Christen–Fox 2005 and Andrieu–Roberts 2009 anchors in ch26c (derivations already in-book); fix "particle-MCMC literature" label at `ch19:401–406` |
| B7 | Citation-desert minimum pass (Parts I–III, ch15–18, ch34/ch35) | add the classical anchors actually used: Kalman 1960; a PED source (Schweppe or Harvey); Joseph/Bierman for square-root stability; Durbin–Koopman (already in bib, unused in these parts); Julier(96)+Merwe(03) in ch15–18 (PDFs already local); the sparse-grid source behind "the source theorem" and the "Jia, Xin, and Cheng" naming in ch34/ch35. This is a *resolvability* pass, not a completeness snowball — record that boundary |
| B8 | ch36 "Sources" paragraph (`ch36:344–354`) | rewrite to match reality: cite `zhao2024ttsequential` where used; either add and cite Oseledets–Tyrtyshnikov + Rosenblatt (entries exist for TT-cross) or delete the claim that the chapter uses them |
| B9 | Survey omission register (ch19b/ch19c; ch32a-c neighborhood) | add a short "omitted work" paragraph or table per survey chapter naming the known omissions (Daum–Huang 2010, Ding–Coates 2012, Bunch–Godsill 2016, stochastic flows, Crouse survey, Heng–Doucet–Pokern, van Leeuwen 2019; Karkus 2018, Ścibór–Wood, Gumbel-softmax, Sinkhorn-divergence debiasing) with one-line classifications — satisfies the audit policy without a full snowball |

**Phase gate:** rebuild has **zero** undefined citations; every in-text "the source/theorem/authors say" sentence has a resolvable citation or an explicit non-claim. Size: M.

---

## 6. Phase 5 — Labels and LaTeX mechanics

| ID | Anchor | Fix |
|---|---|---|
| L1 | `ch18b` ×9, `ch33` ×3 labels inside `\[…\]` | convert the referenced displays to `equation`/`align`; delete or move unreferenced labels; re-check `main.aux` shows no label resolving to a section/item counter |
| L2 | duplicate labels: `ch32c:126/198`, `ch32e:506/552`, `ch32f:166/504`, `ch32f:356/432` | delete the stray copy in each pair (per the review: ch32c keep the first; ch32e/f keep the second occurrences) — coordinate with N3 which deletes the duplicated content itself |
| L3 | `ch35` p-prefixed labels (`sec:p31-*`, `sec:p34-*`, `sec:p38-*`, `eq:p41-*`, `subsec:p32-*`) | rename into the `bf-hd-*` namespace; grep for references first |
| L4 | `ch35:391–401` markdown bullets | convert to `itemize` |
| L5 | end-of-line hyphen breaks (`ch26c:682–683, 902–903, 985–986`; `ch32d:16, 777, 794`; `ch32f:64, 366, 465, 586–587`; `ch19e:50–51`) | join lines or add `%` continuations |
| L6 | `Nystr"om` ×7 (`ch32d`) | → `Nystr\"om`, unify with the 4 plain "Nystrom" instances |
| L7 | misc: `ch09:49–63` align without `&`; prose inside displays (`ch32d:342`, `ch19c:366–370`); `\hbox` in math (ch11/12/14); empty `\widetilde` (ch32c:1360); one-label two-line align (ch32c:235–247); doubled parentheses + undefined τ_zero (ch35:341–347); broken display punctuation (ch35b:85–89, 242–245); dead `\providecommand`s (ch35b:3–5); `\(ch37\)` literal → `\ref` (ch36b:704); longtable-in-center (ch19f); `ass:` → `asm:` namespace (ch18b) | fix each |
| L8 | `ch35` stale short title `[Gaussian And High-Order Filters]` | replace with a short form of the real heading; check TOC/bookmarks after rebuild |

**Phase gate:** rebuild log: zero multiply-defined labels, zero undefined references; overfull count recorded (target: reduced, not necessarily zero). Size: S–M (mechanical).

---

## 7. Phase 6 — Structure and navigation

| ID | Anchor | Fix |
|---|---|---|
| N1 | `ch01:77–88` Reader Map | rewrite to the actual 11-part structure (one sentence per part; parts V–IX exist) |
| N2 | `ch33:48–49`, `ch38:111–127`, `ch38:326–328`, `ch34:29–31` running-cell/bootstrap claims | **Decision point D1 (owner):** either (a) thread the quadratic-observation cell through ch35/ch36/ch36b/ch37 as originally promised (larger, better teaching; consistent with the expansion guardrail), or (b) correct the promises/recaps to match where the cell actually appears. Fix ch38's bootstrap/SIR sentence to point at real content (or import a short bootstrap/SIR baseline discussion into ch36 from the archived file if (a)-style repair is preferred) |
| N3 | duplicated content | delete the second copy of the ch32c teacher-vs-engineering subsection (197–226); deduplicate the merge-artifact paragraphs in ch32e (570–575 vs 517–524) and ch32f (436–438/365–367; 464–466/438–439); consolidate ch38's duplicated benchmark paragraphs; ch36b/ch37: keep the dotted-derivative machinery in ONE chapter and cross-reference from the other (do not delete the derivations — move them); remove the duplicated thesis sentence (ch37:18–19/83–84) and the ch35 duplicated sentence (262–263/337–339); consolidate the ch18b restatements by reference (~25% of the chapter) keeping every derivation exactly once |
| N4 | `ch32c` post-boundary appendage (2062–3032) | move the higher-moment Contract E candidate and the Zhao–Cui moment-teacher sections to their own chapter (natural home: adjacent to ch36b, after squared-TT machinery exists) or to a clearly-marked research-note appendix; fix the dangling "tested below"/"campaign below" references to point at real artifacts; the chapter's closing "does and does not claim" section then describes the whole chapter again |
| N5 | `ch32c2` scope | move the executed certification records (SHA-256 blocks, telemetry, dated ladders) to a result-note appendix or docs/plans pointer; move the OPG score-comparison section to the diagnostics chapter (ch25's natural content); keep all derivations |
| N6 | `ch04:190–225` fixed-center section | either move to ch22 or compress here to a two-sentence pointer with a forward `\ref` (the section's ~15 terms are Part-X vocabulary) |
| N7 | stubs vs titles | `ch08`: retitle to its actual content (validation/test policy for large-scale LGSSM) **or** add the missing scale content (cost model, steady-state reuse) — owner choice, record it; Part XI title → "Validation Ladders, Case-Study Lessons, and Design Targets" or equivalent honest title; `ch35` burden-4 wording aligned with what the chapter actually exports (the cloud; the scalar lives in ch35b) |
| N8 | `ch19f:756–774` misplaced synthesis subsection | move under the section it summarizes |

**Phase gate:** all `\ref`s resolve to intended targets after moves (MathDevMCP label lookup pass); page-count delta recorded; no derivation deleted. Size: M–L (N2(a) and N4 are the big items).

---

## 8. Phase 7 — Linear teaching and worked examples

| ID | Anchor | Fix |
|---|---|---|
| T1 | UT definition placement | move/copy the scaled-UT definition (points, spreads, weights, γ/α/κ/λ) into ch16 where sigma-point rules are introduced; ch18b then references it; rename ch18b's other γ or add a disambiguating sentence |
| T2 | solve operator | define `𝒮_{S}(b)` at first use in ch09 (one line) and keep ch10's definition as a reminder |
| T3 | HMC formulation placement | add to ch21 the ~1-page core: `H = U + K`, leapfrog map, acceptance `min{1, e^{−ΔH}}`, and the covariance-vs-precision mass-matrix orientation (`p ∼ N(0,M)`, `K = ½pᵀM⁻¹p`, whitening-optimal `M ≈` target precision) — this also discharges the review's mass-matrix ambiguity finding against a concrete anchor; ch26b keeps the full treatment |
| T4 | KR maps | add the compact definition block to ch36 (triangular monotone map from 1-D conditional CDFs, existence conditions, triangular Jacobian/log-det with diagonal `p̂(x_k|x_{<k})` — the paper's Prop. 4 gives it in half a page); define `F_j(·|z_{<j})` before ch38's veto uses it; state where `ε_KR` enters the veto |
| T5 | Actual-SV definition order | add a 3-line model statement (equations for `y_t = β e^{h_t/2} ε_t`, the AR(1) for `h_t`, and the transformed law) at first use in ch32c2 with a forward ref to ch33's full derivation; fix ch33's own 30-lines-early usage; define "Lane B" once (or drop the name — see R2) |
| T6 | quadrature panel closure (requirements 2–6) | (i) apply the 3-point rule to h(X)=X² and display `(m−√3C)²/6 + m²·2/3 + (m+√3C)²/6 = m² + P` (closes the abandoned 1-D example); (ii) carry an explicit test function into the 3-D walkthrough (e.g. `F(x)=x₁² + x₁x₂ + x₃`) and evaluate the 6-point cloud on it against the exact Gaussian moments; (iii) fill the "Exactness, Point Count, And The UKF Relation" section with the actual content: exactness class (all total-degree-≤3 moments + pure quartics; **not** cross quartics — state it), point-count formula, and the exact identity *(b, L=2) merged cloud = UKF sigma set with κ = 3 − b*; (iv) plug the general combination-coefficient formula into the 3-D case once; (v) note the merged origin weight `1 − b/3 < 0` for `b ≥ 4` as the completion of the negative-coefficient warning |
| T7 | worked examples in dense lanes | derivative spine: add one scalar AR(1)+noise worked example computing `Ṡ_t`, the score, and an FD comparison side by side (float64, per M11's dtype policy); TT/KR lane: add the 2-D rank-1 squared-TT example with explicit `B_j`, Gram matrix, and marginal; ch32d: instantiate the promised three-particle warm-start cell and state the training loss; ch15: display the covariance update (and Joseph form) so the recursion iterates, and add the Julier y=x² failure example (paper is local) |
| T8 | orientation fixes | acronym expansions at first use (NK, EZ, SGU, CIP, AFNS, NAWM, SGQF, E-BFMI, dual averaging); ch19e/ch20 forward-pointer signposts ("Chapter N, later in Part X"); map the ch19e 8-rung opening to the 5-rung analysis; add the resampling trade-off sentence in ch19; per-chapter source-notation translation tables where imported notation is used (ch19b/c, ch36–38) |

**Phase gate:** a linear-read spot check of each repaired chapter (does every symbol used have a prior definition or explicit forward signpost?); panel-requirements table re-scored — target: 1–5 SATISFIED, 6 SATISFIED. Size: M–L.

---

## 9. Phase 8 — Register and language pass (reader-facing)

Principle: convert project history into content or citations; never delete the underlying claim without replacement.

| ID | Pattern | Fix |
|---|---|---|
| R1 | phantom internal referents | "the scalable-OT survey/note" (ch32b/c/d/e/f, ch19e/f) → either cite the real external sources that back each use (most content is separately supported by altschuler2019nystrom, scetbon2021lowrank, kolouri2019generalized) or cite the docs/plans artifact explicitly as a project note; "the source note" (ch09:48) → name the MacroFinance note via the source map; "row-173 trace"/"frozen manifest" (ch19f) → add pointers to the actual docs/plans artifacts and define "row-173" once; "OT survey companion note" (ch19c) → same treatment |
| R2 | revision-history vocabulary | "Lane B", "the previous framing", "current comparator", "p47/p50", "the panel asked to see", "Reviewer Edge Case" (also in TOC), "the reviewer's claim", "recent source-project audit", "the current repair lane", "banking evidence" → rewrite as content ("A natural objection is…", "the abandoned Gaussian-closure construction, which replaced the exact transformed law by…", "an earlier construction targeted the wrong scalar because…"); rename the banking-evidence table/label vocabulary to the ladder's own terms ("operational claim") |
| R3 | drafting voice | self-instructions ("The chapter should not phrase LEDH as…" → "LEDH should not be read as…"), "requested for this program", "why X belongs in the monograph" meta-commentary — rewrite in content voice |
| R4 | absolute paths and machine tags | `/home/chakwong/python/...` citations (ch16/17/26c) → move to the source map with repo-relative identity; `extension_or_invention`/`fixed_hmc_adaptation` tags → either define the claim-label taxonomy once (app_f is the natural home) or spell them out in prose |
| R5 | scaffolding density | one pass over ch04/ch06–08/ch21–25 converting the worst "should record:" lists into prose where a list adds nothing; trim template tics ("Plain-language takeaway" headers, "declared" repetition, duplicated aphorisms) — style-level, lowest priority, and bounded by the expansion guardrail (do not cut content, only scaffolding) |

**Phase gate:** grep inventory of the R1/R2 phrases returns zero unresolved internal referents; TOC contains no review-process language. Size: M.

---

## 10. Phase 9 — Notation reconciliation

Single dedicated pass, driven by MathDevMCP `reconcile-notation` plus the review's §10 collision table.

- **Priority collisions to resolve (rename or declare chapter-local overrides in app_a's escape-clause form):** `u` (sampler coordinate vs state noise vs preconditioned coordinate vs domain bound — rename state noise to `w_t` or `η_t^{state}` book-wide in Parts I–IV); `ε_t` (pick one role per chapter, declare it); `R_t` inside ch18 (rename the innovation square root); `F` in ch26c (rename the proposal map); `K` in ch32c2 (rename Sinkhorn budget and chunk extent); `θ` reserved for the model parameter — rename learner weights (ch32e/f) and slice direction; `γ_t`/`a` in ch19's index; `Φ` in ch35b vs ch33; `a_t` in ch18b; ch12 helper letters; `g_θ` vs `g_k` in ch19b/c (translation table per T8); `σ_x → σ_u` rename in ch28; transposes to `^⊤` everywhere; `\R` vs `\mathbb{R}`; `d_x/d_y` vs `n_x/n_y`.
- **app_a update:** add table rows for every noise symbol and for any deliberately retained chapter-local overrides.
- **Check:** re-run the collision grep set from the review; each remaining intentional override is declared in-chapter.

Size: M (mechanical but wide).

---

## 11. Phase 10 — Appendices and governance-doc refresh

| ID | Anchor | Fix |
|---|---|---|
| A1 | `app_c` | either fill with the M2 proofs (preferred if M2 route 1 is taken) or rewrite the placeholder to state exactly what exists where (MacroFinance note + tests) and reference it from ch12 so it is no longer orphaned |
| A2 | `app_d` | refresh to the current MathDevMCP surface (note `extract-latex-context` is a compatibility alias for `latex_label_lookup`; mention `audit-derivation-v2-label`); date the audit-status paragraph and point to its artifact |
| A3 | `app_e` | rewrite the "Current Blockers" list to the true state (sigma-point and PMCMC support now exist in ch18b/ch19 with local PDFs; name the *actual* open blockers from Phase 3: HAC sources, Acevedo copy, Li–Coates copy); reconcile the "papers stay out of Git" line with the tracked PDFs in `docs/` |
| A4 | `docs/source_map.yml` | normalize the four off-schema status values (`local_core_passed`, `partial_review`, `partially_unblocked`, `passed`) to the declared five-state schema |
| A5 | `app_g` | add the pointer to `docs/plans/templates/` |
| A6 | ch32 production checklist | resolve orphan strings (`diagnostic_center`/`map` — anchor to the ch04/ch22 API text), `\ref` the fallback order (ch22), define `B`, introduce E-BFMI in ch25 (or point to T3's ch21 content), unify the claim-status ladder vocabulary with ch29 (declare one ladder, one place) |

Size: S.

---

## 12. Phase 11 — Final rebuild, re-audit, and bounded re-review

1. **Full clean rebuild** (twice + bibtex). Gate: zero undefined citations/references, zero multiply-defined labels, chapter count 55, TOC free of register leakage, `[????]`/`[?]` grep on extracted PDF text returns nothing.
2. **Mechanical re-audit** (repeat of the review's coordinator checks): citation-key cross-check script, label/ref cross-check script, phrase-inventory grep (R1/R2 lists), panel-requirements re-score, page-count vs Phase 0 baseline (guardrail: mathematical content non-decreasing).
3. **Targeted re-verification of every Phase-1 fix** using the preserved check scripts/notes (M1 negative-weight identity, M3 Kalman-endpoint test, M4 Monte Carlo unbiasedness, M8a integral identity, T6 worked numbers). **Preserve all check artifacts under `docs/plans/`** with commands and outputs — this closes the reproducibility gap the audit cycle identified in the review process itself.
4. **Bounded re-review:** one review pass scoped to (a) the repaired passages, (b) each repair's blast radius (surrounding section), and (c) a fresh linear read of ch01–ch05 and one repaired dense chapter per part. Not a full re-review of all 431 pages.
5. **Result note:** a short campaign result under `docs/plans/` recording per-phase status, check outcomes, page-count ledger, the D0/D1/N7 decisions taken, and remaining non-claims (bibliography completeness, unreviewed remainder).

---

## 13. Ordering, dependencies, and effort summary

```
Phase 0 (build)  ──► Phase 1 (math) ──► Phase 11 rebuild gate needs everything
                 ├─► Phase 2 (policy text)      [independent of Phase 1 except S4↔ch18/ch20 wording]
                 ├─► Phase 3 (source gaps) ──► Phase 4 (bibliography)   [G1/G2 feed B-items]
                 ├─► Phase 5 (labels/LaTeX)     [after N3/N4 moves is cheapest — run L-pass after Phase 6]
                 ├─► Phase 6 (structure)        [N2 decision D1 first; N3/N4 before L2]
                 ├─► Phase 7 (teaching)         [T1/T3 before R-pass; T6 independent]
                 ├─► Phase 8 (register)         [after N/T so rewrites happen once]
                 ├─► Phase 9 (notation)         [last content pass before Phase 11]
                 └─► Phase 10 (appendices)      [A1 depends on M2 route choice]
```

Rough sizes: P0 S; P1 L; P2 S–M; P3 S–M; P4 M; P5 S–M; P6 M–L; P7 M–L; P8 M; P9 M; P10 S; P11 M. The heavy single items are M1–M4, M8, N2(a)/N4, T6–T7.

**Suggested execution batches** (each ends at a buildable state and can be committed as a checkpoint on owner approval):
1. Phase 0;
2. Phase 1 M1–M6 (the correctness core) + M10–M11;
3. Phase 1 M7–M8 + Phase 2 (the transport/TT/policy cluster);
4. Phases 3–4 (sources + bibliography);
5. Phase 6 + Phase 5 (structure then mechanics);
6. Phase 7 (teaching/examples);
7. Phases 8–10 (register, notation, appendices);
8. Phase 11 (gate + result note).

---

## 14. Stop conditions and change-of-plan triggers

- **Continuation veto:** a Phase-1 check fails in a way that indicates the *review finding itself* was wrong → stop that item, record the counter-evidence, update the review record, do not force the edit.
- **Scope veto:** a fix requires changing repository code semantics (not just monograph text) → out of scope for this plan; record as a separate repair candidate.
- **Guardrail veto:** any batch whose diff deletes displayed derivations without relocation → revert the batch.
- **Decision escalations to the owner:** D0 (artifact freeze — recommended Option A), D1 (thread the running cell vs correct the promises), N7 (ch08 retitle vs expand), M2 route (add calculus vs narrow promise), M7 route (match paper vs declare variant — resolve by checking the code's convention first).

## 15. Traceability

Revised-review headline findings → work orders: 1→M1; 2→M2; 3→M3; 4→M4; 5→M6; 6→M5; 7→M7; 8a→M8a; 8b→M8b; 8c→M8c; 9→B1–B6; 10→P0.1–P0.2; 11→L1–L2; 12→M10; 13→S4; 14→M9; 15→S1–S3; 16→N2/P0.4; 17→G1. Revised-review §3.3 support gaps → B6–B9, G2–G3. Reply's required corrections → §0 artifact decision, P0.4, B-phase metadata dating, Phase 11 step 3 (preserved evidence).
