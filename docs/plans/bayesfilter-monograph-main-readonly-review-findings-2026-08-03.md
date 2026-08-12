# Independent Audit of Fable's BayesFilter Monograph Review

- **Date:** 2026-08-03
- **Scope:** read-only audit of Fable's review against `docs/main.tex`, its included chapter and appendix sources, `docs/references.bib`, the committed `docs/main.pdf`, repository history, existing LaTeX artifacts, and selected primary-source metadata.
- **Repository state audited:** `6a11b689295bfb0e58de6e6d2f84918671b5a685` plus the pre-existing worktree. No monograph source was changed.
- **Verdict:** **Substantially agree, but not wholesale.** Most urgent mathematical, citation, build, label, structural, and register findings are real. Several findings need narrower classifications, and Fable's PDF/source provenance statement is materially wrong.

## 1. Bottom line

Fable found important defects that should be acted on. In particular, the SR-UKF negative-weight factor identity, the incomplete signed-update derivative exposition, the LEDH offset, the skip-resampling particle-filter recursion, finite fallback targets, GenUT attribution, two squared-TT inconsistencies, the ICNN explanation, missing citations, missing figures, labels, and stale cross-chapter claims survive independent scrutiny.

I do **not** accept three broader parts of the review:

1. It treats the committed PDF and current source tree as one artifact. They are not. This invalidates the printed-chapter mapping for `ch26c` and the claim that the PDF contains the theorem said to refute earlier chapters.
2. It calls some propositions or policies wrong where the evidence establishes only ambiguity or incomplete specification, especially the third squared-TT/Jacobian allegation, the covariance-contraction proposition, and part of the chunk-policy allegation.
3. It presents broad mathematical verification and exact numerical results as independently reproducible facts without preserving the scripts, inputs, matrices, seeds, or reviewer outputs needed to audit them.

The proper disposition is therefore **REVISE**, not reject: retain the confirmed defect list, correct the artifact boundary and overstated classifications, and downgrade unsupported verification claims.

## 2. Artifact and provenance correction

This is the most important correction to Fable's framing.

- `docs/main.pdf` is a 431-page A4 PDF with metadata creation time `2026-07-20 03:48:40 HKT`.
- The last commit touching that PDF is `41f2aa4f263d96e5575a6448d89bdd93bb262035`.
- The PDF contains 54 chapters. Current `docs/main.tex` inputs 55 chapters.
- `docs/chapters/ch26c_hnn_surrogate_hmc.tex` and its `main.tex` inclusion were added after the PDF, in commit `71f659aa2620adfaa9fdb34d66c0816543365c82` dated 2026-07-22.
- Between the PDF commit and the audited source tree, changes are not confined to `ch32c`. The changed monograph files are `ch19e`, new `ch26c`, substantially expanded `ch32c2`, substantially expanded `ch32c`, `main.tex`, and `references.bib`.

Consequences:

- Fable's assertion that only `ch32c` postdates the PDF is **wrong**.
- Its printed-chapter map assigning `ch26c` to printed Chapter 48 is **wrong for the committed PDF**. That chapter is absent from the PDF.
- The surrogate-force and fallback contradictions are **current-source contradictions**. The PDF cannot be said to contain the later theorem that refutes or sharpens its own earlier text.
- Findings in post-PDF additions must not be represented as rendered defects in `docs/main.pdf`.

All adjudications below distinguish current-source evidence from PDF evidence.

The build check was isolated from repository outputs:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/bayesfilter-main-audit-build main.tex
```

It stopped at the first missing figure, `ssl-lstm-launch-traces-z.pdf`, referenced
at `ch28a:1002`.

## 3. Headline finding adjudication

| # | Fable finding | Independent verdict | Audit result |
|---:|---|---|---|
| 1 | SR-UKF filtered factor omits negative state-deviation downdates | **Agree** | At `ch17:286`, `C_{t\mid t}` is formed from `C_{xx,+}` and the `KS` downdate only. Earlier, `ch17:120` defines the state covariance as positive contributions minus `E_-E_-^T`. On a negative-weight branch, the displayed filtered factor therefore does not reconstruct the displayed `P_{xx,*}-KSK^T`. This is wrong relative to the chapter's signed-branch contract. |
| 2 | SR-UKF signed update/downdate derivative chain is missing | **Agree, with qualification** | `ch17:185` says Chapter 12 supplies the QR and update/downdate primitive calculus. Chapter 12 supplies QR and ordinary Cholesky factor derivatives, but no ordered rank-one cholupdate/choldowndate derivative. Appendix C explicitly remains a placeholder. The promised derivation chain is incomplete. “Not implementable” is too absolute because an implementer could obtain the missing calculus elsewhere. |
| 3 | Chapter-derived LEDH offset is wrong | **Agree algebraically** | With local likelihood `exp(-x^T Lambda x/2 + eta^T x)` and prior mean `m`, the affine flow's constant term must account for the centered information `eta-Lambda m`. The formula at `ch19b:550` uses `eta` without that centering and does not reduce consistently to the chapter's global linear case. A different Li-Coates source-form expression appears at `ch19b:592`. Fable's quoted `~1.07` and `3e-5` numerical values are **not independently auditable** because its test artifact was not preserved. |
| 4 | Bootstrap-PF likelihood estimator is wrong when resampling is skipped | **Agree** | `ch19:375` first states the carried-weight recursion, but Step 3 at `ch19:390` drops the previous normalized weight while Step 4 permits the weighted cloud to be carried. The proof later equates an unweighted propagation empirical measure with the weighted empirical measure on the no-resampling branch. A consistent auxiliary/bootstrap formulation needs post-decision carried weights, e.g. an incremental factor proportional to `N w^*_{t-1} g_t`, or must require resampling before every propagation. |
| 5 | Same-scalar surrogate-gradient rule overstates HMC bias | **Agree, with artifact and wording qualifications** | The blanket safety claim at `ch13:4` is false for exact endpoint-MH correction using a fixed deterministic position force whose symmetric leapfrog map is reversible and volume-preserving. In that case the exact target value in the acceptance step preserves the target; a poor force harms efficiency. `ch03:24` already contains the qualifier “unless explicitly corrected.” The later proof is in current `ch26c`, absent from the PDF. The text should distinguish exact endpoint correction from unadjusted, stochastic, non-involutive, non-volume-preserving, or wrong-value cases. |
| 6 | Finite fallback changes the target | **Agree** | A finite constant log density over an invalid region assigns positive density there; it is not equivalent to support rejection via `-infinity`. On an unbounded invalid region it can make the target improper. `ch03:36` and `ch23` do not define this modified target, whereas current `ch26c:855` uses infinite potential for out-of-support endpoints. This is a real current-source contradiction and a target-definition defect. |
| 7 | GenUT is misattributed to whitened marginal moments | **Agree** | The local Ebeigbe et al. paper's equations (21), (25), (28), (30), and (31), plus Algorithm 1, construct physical-coordinate diagonal skewness and kurtosis using powers of a chosen covariance square root. They do not generally match the marginal third and fourth moments of `Z=C^{-1}(X-m)`. Those objects differ for correlated coordinates. The per-axis algebra may be usable for a separately defined whitened-moment rule, but it is wrong relative to the stated Ebeigbe attribution. |
| 8 | Three squared-TT/KR errors | **Partly agree** | The missing `e^{-c_t}` multiplier on the defensive term at `ch36b:190` is real under the source/evidence convention at `ch36:211`. The retained-last derivation at `ch36b:171` conflicts with the retained-first concrete ordering at `ch36b:362`. The claimed missing Jacobian at `ch37:358` is **not established as an algebraic error**: that display explicitly evaluates a reference-coordinate density at reference-coordinate queries. The physical-to-reference interface for next-step points is under-specified, but Fable did not prove its claimed normalizer shift. |
| 9 | Broken/corrupted bibliography | **Broadly agree; revise loaded wording** | Eight cited `ch32c` keys are absent. The NeuTra ICML/PMLR venue, volume, and pages are wrong; call this incorrect metadata, not “fabricated.” Official metadata confirms **Yunpeng Li**, not “Yanghai,” and **Chih-Chi Hu**, not “Chao”; the Hu issue/pages also need correction. The Daum-Huang entry is incomplete and may conflate records, but the exact bibliographic identity was not resolved sufficiently to call it definitively “unfindable.” |
| 10 | Current tree cannot rebuild the PDF | **Agree for current source; reject the asserted history** | An isolated `latexmk` build fails at `ch28a:1002` because `ssl-lstm-launch-traces-z.pdf` is missing. Five `.pdf` paths are referenced; only same-stem `.png` files are tracked, and no corresponding PDFs appear in Git object history. Current source is not reproducibly buildable. Fable's claim that the old PDF “embedded untracked local files” is a possible explanation, not proven by this evidence. |
| 11 | Mis-anchored and duplicate labels | **Agree** | The source contains nine labels inside unnumbered displays in `ch18b`, three in `ch33`, and four active duplicate labels. These can inherit unrelated counters or generate contradictory references. The four duplicates are the `sec:bf-eot-teacher-versus-engineering`, `sec:bf-neural-ot-direct-scalable-boundary`, `sec:bf-neural-ot-dynamic-same-scalar`, and `sec:bf-neural-ot-dynamic-sliced-localized` labels. |
| 12 | Structural-UKF comparison table is transposed | **Likely agree; heading is ambiguous** | Under the chapter's intended comparison, misapplying an additive-noise UKF to the structural model causes model-law error plus moment approximation error, while the structural route on its intended model retains moment approximation error. The row currently assigns the converse. However, “Failure if misused” could be read as misuse of each method on an unspecified foreign target. The row should name the target/misuse explicitly; under the surrounding chapter argument its entries are swapped. |
| 13 | Filter-choice register recommends a demoted route | **Agree, with qualification** | `ch20:38` demotes historical eigenderivative SVD routes, then `ch20:65` says to attempt “DSGE-scale SVD sigma-point HMC.” That phrase could mean the promoted strict-SPD/principal-square-root implementation, but the chapter does not say so. The “next ... chapters” pointer is false because two parts intervene. This is an ambiguous recommendation and stale navigation, not proof that every SVD sigma-point HMC route is forbidden. |
| 14 | ICNN objective's target anchoring explanation is wrong | **Agree** | For fixed `varphi`, `E_nu[varphi(Y)]` is constant in `theta`; it cannot transmit target-sample information into the displayed minimization. Target anchoring requires a target-informed potential learned beforehand or an objective that also optimizes/updates the target-side potential. The prose mentions these possibilities but the displayed objective and procedure do not define one. |
| 15 | `ch32c2` contradicts chunk and derivative policies | **Partly agree** | Fable overreads storing row/column chunk sizes in `eta_OT` as declaring them caller-free; a repository-selected value may legitimately be stored for reproducibility. Real defects remain: the chapter omits the canonical selector and policy ID, contains stale `K=N` wording that fails for `N>3000`, and does not say supplied settings must equal the repository selection. Its stop-gradient language does not cleanly preserve the claimed total derivative when stopped quantities depend on parameters. The repository's per-scope LEDH tuning policy is absent. |
| 16 | Cross-chapter running-cell and bootstrap claims are stale | **Agree** | Active `ch33` promises the cell through a range of chapters where it is mostly absent. `ch38` attributes roles to active chapters that do not contain them and points to a chapter for bootstrap/SIR material it lacks. Orphan predecessor files explain part of the drift, but do not make the active claims true. |
| 17 | HAC consistency claim is overbroad | **Not independently established; source-gap blocker** | The chapter declares mixing and `2+eta` moment assumptions and a Bartlett bandwidth of order `N^{1/3}`. Whether those exact assumptions suffice for its stated matrix HAC consistency requires the precise primary theorem conditions. The cited primary theorem sections were not available locally and were not fully audited. Fable correctly identifies a claim-support gap, but its theorem verdict remains unconfirmed. |

The eight confirmed missing keys in finding 9 are
`acevedodewiljesreich2017secondorder`, `bach2026learning`,
`hosseini2025conditional`, `kirchgessner2017smoother`, `lei2011moment`,
`todter2015secondorder`, `vanleeuwen2020consistent`, and `zeng2026coupling`.

## 4. Additional findings that survive audit

The following secondary findings are sufficiently supported by the repository evidence and should remain in a revised review:

- **Appendix B hypothesis gap:** the quadratic-form derivative at `app_b:20` requires the covariance/precision setting's symmetry conditions. As stated only with invertible `S`, it is not a general matrix identity.
- **Local Taylor assumptions:** the structural-UKF accuracy proposition needs a domination/integrability or suitable growth condition; local `C^3` regularity alone does not control a Gaussian expectation over unbounded support.
- **Floored Sinkhorn semantics:** inserting a positive floor inside the scaling denominator changes the finite fixed-point equations; it is not merely a numerically exact implementation of the preceding balancing equations.
- **Nyström norm assumptions:** the bare error identity in `ch32d` needs the relevant norm and kernel/residual assumptions stated.
- **Zhao-Cui preconditioning:** the load-bearing pushforward compatibility condition should accompany the imported preconditioning formulas.
- **ICNN and transport source gaps:** the Meta OT bibliography entry is orphaned while the method is used, and Benamou-Brenier is invoked without a bibliography entry or a definition of the geodesic construction.
- **MCMC diagnostic attribution:** `ch26b` gives rank-normalized/folded R-hat and bulk/tail ESS constructions without citing Vehtari et al.
- **Citation deserts:** Parts I-III and the core quadrature chapters contain no citations despite relying on classical results and source-language claims. This does not make every displayed derivation wrong, but it makes broad scholarly coverage and attribution inadequate.
- **Reader map and ordering:** the reader map describes an obsolete five-part progression while current source has eleven parts. Important objects are introduced after earlier use, including the full scaled-UT specification, solve notation, and basic HMC machinery.
- **Uneven structure:** the 82-line `ch15`, 84-line `ch20`, 1,468-line `ch28a`, and 2,757-line `ch18b`, together with a placeholder proof appendix, substantiate the imbalance concern.
- **Internal register leakage:** phrases such as “Phase 2B literature gate,” “row-173 trace,” “Lane B,” “p50 lane,” “the panel asked to see,” personal absolute paths, and internal review/governance tags are unresolved for a standalone reader. This finding is stronger and more objective than phrase-frequency claims about whether the prose “sounds human.”
- **Stale aliases and orphan sources:** active aliases and orphan predecessors create real navigation and maintenance hazards, including claims that match old chapter contents rather than the active inputs.
- **Build hygiene:** the pre-existing log contains many overfull boxes and unresolved references/citations. Those diagnostics should be regenerated only after the fatal missing-figure error is fixed; first-pass counts from an incomplete build are not final book-wide counts.

## 5. Fable claims that need revision or withdrawal

### 5.1 Incorrect claims

- **“Only `ch32c` changed after the PDF build.”** Wrong; six monograph files changed, including a new chapter.
- **“`ch26c` is printed Chapter 48.”** Wrong for `docs/main.pdf`; it is absent from that PDF.
- **“The PDF proves the refuting HMC theorem itself.”** Wrong. Current source contains the theorem; the PDF does not.
- **Covariance-contraction proposition is mathematically underdetermined.** Fable's secondary criticism goes too far. For any joint law/coupling used to define `E[X|J]`, the law-of-total-covariance identity holds. The prose could tie the conditional law more explicitly to the specific OT coupling, but nonuniqueness of couplings with the same marginals does not invalidate the proposition.
- **`ch36` has no Zhao-Cui citation.** Too broad. It names Zhao and Cui and cites `zhao2024ttsequential` later. Its source paragraph and coverage are incomplete, but the citation does exist.

### 5.2 Overstated claims

- **Third squared-TT/Jacobian error:** classify as an under-specified coordinate handoff unless a derivation shows that the actual next-step evaluator consumes physical coordinates without the required conversion.
- **Chunk sizes are “free per-call settings”:** not entailed by the settings record alone. The missing selector identity, stale `K=N`, and fail-closed validation language are the supportable findings.
- **The q=20 resource decision is internally contradictory:** not proven. The 2 GB level is called a warning threshold, and q=20 also violates the 600-second cap. The roles of the thresholds are unclear, but a logical contradiction does not follow.
- **NeuTra venue is “fabricated”:** incorrect metadata is established; intent or fabrication is not.
- **Daum-Huang is “unfindable”:** incomplete/conflated metadata is plausible, but the exact record needs a complete primary-source resolution before that stronger label.
- **Human-voice scan “passes”:** phrase counts cannot establish human authorship or prose quality. The reader-facing register defects can be assessed directly without an authorship inference.

### 5.3 Unsupported positive verification

Fable says essentially every major identity was re-derived and gives values such as `~1e-10`, “digit-exact,” “machine precision,” “all propositions,” and “every printed digit.” The review file preserves no calculation scripts, exact matrices, source excerpts, random seeds, input artifacts, logs, or per-claim derivation ledger. Those statements are therefore **not independently auditable from the artifact**.

This does not mean the formulas are wrong. It means the proper status is “reported by Fable, not reproduced in this audit,” not `CONFIRMED`. The same applies to the claim that 15 parallel reviewers read every included file fully and independently verified headline findings: no reviewer outputs or coverage ledger are preserved.

The executive claims that the mathematics is “remarkably sound” and that “nothing found requires reopening principal scientific positions” are broader than the retained evidence supports. A defect review can establish located failures; it cannot certify the unlisted remainder of a 431-page monograph without auditable coverage evidence.

## 6. Scholarship audit boundaries

### Sources directly checked in this audit

- Current repository LaTeX, bibliography, build artifacts, and Git history.
- Local Ebeigbe et al. GenUT paper, including the method equations and algorithm relevant to finding 7.
- Official metadata for DOI `10.1109/TSP.2017.2703684`, confirming Yunpeng Li.
- Official metadata for DOI `10.1002/qj.4028`, confirming Chih-Chi Hu and correcting the local bibliographic details.
- The official arXiv record for `1903.03704`, establishing that the local NeuTra venue/pages do not match that work.

### Limits

- No complete backward/forward citation snowball or omission-risk search was performed for the whole monograph.
- No comprehensive retraction, correction, or erratum search was completed.
- The exact Newey-West/Flegal-Jones theorem assumptions needed for finding 17 were not checked from primary theorem text.
- The Daum-Huang record was not resolved to a final canonical citation.
- The eight missing `ch32c` citations were confirmed missing from the bibliography, but their individual technical claims were not all checked against primary sources.
- Fable's reported numerical re-derivations were not reproduced because the necessary artifacts were not supplied.

No source is quarantined on the present evidence. Literature completeness, absence of retractions, and correctness of every citation-led claim are **not concluded**.

## 7. Recommended repair priority

1. **Separate and freeze the artifact being revised:** decide whether the target is the committed 54-chapter PDF state or current 55-chapter source, then rebuild from that exact source state.
2. **Repair target/correctness defects:** findings 1, 3, 4, 5, 6, 7, 8(a-b), and 14.
3. **Complete the derivative exposition:** finding 2 and Appendix C, with explicit signed rank-one derivative and failure-branch calculus.
4. **Restore reproducible compilation:** replace or regenerate the five figure references, then rerun LaTeX sufficiently to settle citations, labels, and references.
5. **Repair bibliography and attribution:** missing keys, NeuTra, Li, Hu, Daum-Huang, Vehtari, Meta OT, Benamou-Brenier, and other load-bearing source gaps.
6. **Reconcile active policy text:** canonical DPF chunk selector, Contract E total-gradient wording, and per-scope LEDH tuning.
7. **Repair navigation and reader contract:** stale aliases, running-cell claims, reader map, chapter order, and the ambiguous filter-choice/table summaries.
8. **Perform a reader-facing edit:** remove internal project-register and governance residue, consolidate duplicates, and add worked examples where the derivation spine is currently example-free.

## 8. Final verdict

**Substantially agree, but not wholesale.** Fable's review is valuable as a defect-finding pass and correctly identifies many high-priority repairs. It is not reliable as written as a final audit record because it conflates two different monograph artifacts, overclassifies several ambiguous or under-specified points as confirmed mathematical errors, uses unjustifiably loaded bibliographic language, and asserts exhaustive/numerical verification without preserving reproducible evidence.

The actionable core should be retained after those corrections. The current monograph source is not reproducibly buildable, contains several genuine target or algorithm errors, has incomplete source support in load-bearing sections, and needs structural and reader-register repair. Conversely, this audit does **not** establish that all mathematics not named here is correct or incorrect, that the principal scientific positions need no reconsideration, or that the bibliography is complete.
