# BayesFilter Monograph Rewrite Master Document

- **Date:** 2026-08-04
- **Purpose:** provide one readable, consolidated rewrite document for the monograph repair campaign, incorporating the review findings, the independent audit corrections, the revised-review reply, and the rewrite-plan critique.
- **Status:** planning/documentation artifact only. This is not a monograph text rewrite and does not itself modify `docs/main.tex` or any chapter source.
- **Primary audience:** the owner and future workers who need one coherent document to read before deciding whether to execute the repair campaign.

---

## 1. What this document is

This document replaces the scattered planning state with one readable master note. It is the integrated version of:

- the original read-only review,
- the independent audit of that review,
- the revised review,
- the reply to the revised review,
- the rewrite-plan critique,
- and the freeze-and-ledger execution boundary.

It answers four questions in one place:

1. **What is actually wrong with the monograph?**
2. **Which findings are confirmed, which are narrower, and which are still source-gap blockers?**
3. **What artifact are we repairing?**
4. **What exact sequence of repairs would bring the monograph to a bounded review-ready state?**

---

## 2. Executive summary

The monograph is **not ready as-is**, but the situation is also **not** “the whole thing is broken.” The review cycle converged on a stable core:

### Confirmed high-priority defects
These remain actionable and should be treated as real:

- SR-UKF filtered-factor identity on negative-weight branches (`ch17`)
- missing signed update/downdate derivative exposition (`ch17` → `ch12`/`app_c`)
- chapter-derived LEDH offset formula wrong (`ch19b`)
- bootstrap-PF likelihood recursion wrong on the optional skip-resampling branch (`ch19`)
- finite fallback defines a different target and can be improper (`ch03`, `ch23`)
- GenUT attribution wrong relative to Ebeigbe (`ch32c`)
- two squared-TT inconsistencies:
  - missing `e^{-c_t}` multiplier,
  - retained-last vs retained-first contradiction
- ICNN objective explanation wrong for the displayed fixed-φ minimization (`ch32e`)
- broken bibliography / missing citations / missing figure assets
- stale running-cell and bootstrap/SIR cross-chapter claims
- mis-anchored labels in `ch18b` and `ch33`

### Narrowed findings
These should still be repaired, but their old wording was too strong:

- the third squared-TT / missing-Jacobian claim is **under-specified**, not a proved algebraic error;
- the covariance-contraction proposition criticism is really a **statement/specification issue**, not a demonstrated theorem failure;
- the chunk-policy issue in `ch32c2` is about **missing selector identity, stale `K=N` wording, and no-override validation**, not a proved caller-free knob declaration;
- the NeuTra citation has **incorrect metadata**, not proven fabrication;
- the Daum–Huang citation is **bibliographically unresolved / likely conflated**, not definitively unfindable from the evidence preserved so far;
- the HAC consistency claim is a **source-gap blocker**, not yet a theorem-level confirmed error.

### Structural reality
The monograph's strongest chapters are mathematically serious and often well written. The weakest areas are:

- target-definition and cross-chapter consistency,
- bibliography and source support,
- build reproducibility,
- stale navigation/reader-map promises,
- internal project-register leakage,
- and missing worked examples in the densest formal lanes.

### Current execution verdict
A rewrite campaign is justified, but the **old rewrite plan should not be executed**. It needed a better execution boundary. That boundary is now frozen in the companion file:

- [bayesfilter-monograph-rewrite-freeze-and-ledger-2026-08-04.md](bayesfilter-monograph-rewrite-freeze-and-ledger-2026-08-04.md)

This master document gives the readable version of the same plan.

---

## 3. Artifact boundary: what exactly is being repaired

This was the biggest correction from the audit cycle.

### Committed PDF vs current source
They are **not the same artifact**.

- `docs/main.pdf` is a committed 431-page PDF built on 2026-07-20.
- The current `docs/main.tex` inputs 55 chapters.
- The committed PDF corresponds to a 54-chapter state.
- `ch26c_hnn_surrogate_hmc.tex` is absent from the committed PDF and exists only in the later source tree.
- Several later source files also changed materially after the PDF build, including `ch19e`, `ch32c`, `ch32c2`, `references.bib`, and `main.tex`.

### Repair choice
This campaign is frozen to:

> **current 55-chapter source tree**

not to the historical 54-chapter PDF-era source.

### Consequences
This means:

- contradictions involving `ch26c` are **current-source contradictions**;
- post-PDF additions must not be described as defects already rendered in the committed PDF;
- the current campaign is a source-repair campaign that will regenerate a new PDF later.

---

## 4. Evidence boundary

This document is intentionally careful about what it claims.

### Directly checked during the review process
The review process directly checked, among other things:

- the local Ebeigbe GenUT paper for the multivariate moment claim,
- the Li and Hu DOI metadata,
- the NeuTra arXiv record,
- the local Julier, van der Merwe, Corenflos, Del Moral–Doucet–Singh, Dhulipala, and Zhao–Cui materials,
- the relevant repo code for many named implementation claims,
- the build log, aux files, TOC, PDF text extraction, and label/citation inventories.

### Not established by this document
This document does **not** establish:

- that the bibliography is complete in a full scholarly-snowball sense,
- that every uncited or unmentioned formula is correct,
- that no cited source is retracted or superseded,
- or that the unnamed remainder of the monograph is globally certified.

The campaign should be read as a **repair program**, not a global proof of correctness.

---

## 5. The retained defect list, grouped by theme

## 5.1 Mathematical and target-definition defects

### A. SR-UKF branch correctness
- `ch17` filtered-factor identity fails on negative-weight branches.
- `ch17` promises a signed update/downdate derivative chain that the book does not supply.

### B. Particle-flow / particle-filter target defects
- `ch19b` chapter-derived LEDH offset is wrong relative to its own linear-Gaussian closure.
- `ch19` optional skip-resampling branch breaks the likelihood estimator as written.
- `ch03`/`ch23` finite fallback silently changes the target.

### C. Transport and learned-map defects
- `ch32c` misstates GenUT relative to the cited paper.
- `ch32e` misexplains the displayed ICNN objective.
- `ch32c2` does not yet state the chunk selector and total-derivative policy tightly enough.

### D. High-dimensional fixed-branch / TT defects
- `ch36b` missing `e^{-c_t}` factor.
- `ch36b` retained-coordinate ordering contradiction.
- `ch37` physical/reference handoff still under-specified.

## 5.2 Citation and source-support defects

- 8 missing `ch32c` citation keys.
- incorrect NeuTra metadata.
- incorrect Li and Hu author names.
- unresolved Daum–Huang identity.
- no Vehtari citation for modern R-hat/ESS.
- Meta OT orphaned while used.
- Benamou–Brenier invoked without source entry.
- load-bearing Li–Coates source lacks a durable local copy.
- Parts I–III and the core quadrature chapters remain citation deserts.

## 5.3 Build and label defects

- current tracked source cannot rebuild because of missing figure inputs,
- mis-anchored labels in `ch18b` and `ch33`,
- duplicate active labels in `ch32c`, `ch32e`, `ch32f`,
- stale short title and label hygiene issues,
- several LaTeX rendering defects (`Nystr"om`, markdown bullets, prose-in-display, etc.).

## 5.4 Structure and teaching defects

- obsolete five-part Reader Map for an eleven-part book,
- stale running-cell promises and false chapter recaps,
- weak payoff chapters and strong imbalance across parts,
- no worked examples in the derivative spine and TT/KR lane,
- quadrature chapters partially satisfy but do not complete the panel's example requirements,
- internal review/process language visible to the reader.

---

## 6. The governing execution decision

This campaign is now governed by a frozen execution boundary:

- artifact choice: **current 55-chapter source tree**,
- policy pair: **`AGENTS.md` + `CLAUDE.md`**, with `AGENTS.md` controlling where they differ,
- tracked roots in scope:
  - `docs/main.tex`
  - `docs/main_highdim_restart_staging.tex`

### Important consequence
`chapters_restart_staging/` is **not orphaned**. Any structural cleanup that touches it must be done root-scoped and deliberately, not as casual Phase-0 cleanup.

---

## 7. The rewrite campaign, rewritten as a clean execution package

The repair campaign should proceed through these workstreams.

## W0. Freeze-confirmed baseline and mechanical inventory

Already partially completed by the freeze-and-ledger note. Remaining execution work in W0 is:

1. choose whether to use tracked PNGs or regenerated PDFs for the five missing `ch28a` figures,
2. run a baseline build that completes past the fatal missing-figure point,
3. record page count, undefined citations, undefined refs, multiply-defined labels, and extracted-PDF `[?]` inventory,
4. **do not** move or archive alternate-root source trees here.

This is a **baseline build**, not yet a clean build.

## W1. Confirmed mathematical and target-definition repairs

This is the core technical workstream.

It includes:

- **W1.M1** SR-UKF filtered-factor branch repair
- **W1.M2a** narrow the false derivative-exposition promise in ch17
- **W1.M2b** optional full signed update/downdate derivative exposition, if pursued
- **W1.M3** LEDH offset repair and reconciliation with the correct Li–Coates source form
- **W1.M4** bootstrap-PF skip-resampling recursion repair, with algebra as primary evidence and Monte Carlo only as corroboration
- **W1.M5** finite-fallback target-definition repair
- **W1.M6** HMC exactness wording repair
- **W1.M7** GenUT attribution repair
- **W1.M8a/b/c/d** squared-TT lane repairs, with the handoff item handled as route audit first
- **W1.M9** ICNN objective/explanation repair without overprescribing a source form unless re-audited
- **W1.M10** structural-UKF comparison-row repair
- **W1.M11** small hypothesis/notation/math patches

Every one of these must have its own preserved evidence artifact.

## W2. Policy-text and contract wording repairs

This workstream aligns the text with the frozen policy boundary without changing code semantics.

It includes:

- chunk selector and no-override wording,
- stop-gradient partial-derivative wording,
- LEDH per-scope tuning rule insertion,
- filter-choice wording cleanup,
- forward-vs-reverse mode clarification,
- comparison-stage statistical-evidence sentence.

## W3. Literature, bibliography, and source-support repairs

This must satisfy a bounded version of the repository's own literature-audit standard.

Required ledgers:

- source-support ledger,
- claim-to-source ledger,
- metadata correction ledger,
- omitted-paper register,
- retrieval blocker ledger.

This workstream closes or downgrades:

- HAC theorem claim,
- Acevedo–de Wiljes–Reich anchor,
- Li–Coates local-copy gap,
- missing `ch32c` keys,
- NeuTra/Li/Hu/Daum–Huang fixes,
- Vehtari / Meta OT / Benamou–Brenier / FilterFlow additions,
- citation-desert minimum pass,
- survey omission register.

## W4. Structure, navigation, and scope repairs

This workstream is where the old plan was too broad.

It must be split into exact move-map subplans for:

- Reader Map rewrite,
- running-cell/bootstrap-claim resolution,
- duplicated-content clusters,
- `ch32c` appended research-note material,
- `ch32c2` certification / OPG scope split,
- fixed-center section placement,
- title honesty and chapter-scope alignment,
- misplaced synthesis subsection.

No move is executed without a destination, retained text map, label map, and rebuild gate.

## W5. Teaching, examples, and notation repairs

This closes the audience/readability defects.

It includes:

- moving the UT definition to first use,
- defining the solve operator at first use,
- adding core HMC formulation and mass-matrix orientation to `ch21`,
- defining KR maps with exact directional orientation,
- introducing Actual-SV where first needed,
- explicitly closing the quadrature panel requirements with worked 1D and 3D test-function calculations,
- adding worked examples to the derivative spine and TT/KR lane,
- cleaning acronym and forward-reference discipline,
- splitting notation reconciliation into cluster-specific passes rather than one giant sweep.

## W6. Final gate and bounded re-review

The final gate must be a **full matrix**, not a verbal promise.

For every work order, the matrix must record:

- target anchor,
- acceptance criterion,
- evidence class,
- artifact path,
- failure action.

Success requires:

- zero undefined citations,
- zero undefined references,
- zero multiply-defined labels within the selected root,
- every matrix row passed or explicitly blocked,
- result note written.

---

## 8. What changed from the old rewrite plan

This is the readable summary of the corrections.

### Removed or corrected from the old plan
- no archival move of `chapters_restart_staging/` in Phase 0,
- no overprescribed ICNN maximin repair without source re-audit,
- no generic physical→reference prescription for the squared-TT handoff,
- no implication that stop-gradient always gives the exact gradient of another scalar,
- no hand-wavy literature phase that treats unsupported citations as acceptable theorem support,
- no incomplete final verification gate,
- no `Nystr\"om` LaTeX mistake,
- no reliance on page count alone as the content-preservation metric.

### Added to the old plan
- a real Freeze Gate,
- policy freeze against `AGENTS.md` and `CLAUDE.md`,
- root-scoped execution rules,
- evidence ledgers for literature work,
- exact move-map requirements for structural edits,
- a complete verification-matrix requirement.

---

## 9. What I recommend doing next

If you want this campaign to proceed, the clean next sequence is:

1. read the freeze-and-ledger note,
2. accept the current-source 55-chapter artifact choice,
3. generate the five execution subplans under `docs/fable-rewrite/`:
   - math core,
   - policy/structure,
   - literature/bibliography,
   - teaching/notation,
   - final gate matrix,
4. only then execute repairs.

That is the first point where the rewrite campaign becomes operationally safe.

---

## 10. Final verdict

This is the consolidated readable version of the rewrite campaign.

It keeps the correct repair direction from the earlier plan, but it no longer pretends the old monolithic plan was execution-ready. The campaign should proceed only through the frozen, root-scoped, evidence-preserving workstreams summarized here.
