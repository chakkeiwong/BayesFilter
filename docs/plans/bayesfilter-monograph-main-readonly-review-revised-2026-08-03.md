# Revised Read-Only Review of the BayesFilter Monograph

- **Date:** 2026-08-03
- **Supersedes/revises:** Fable's original in-session read-only review of the monograph. **Provenance note:** the path `docs/plans/bayesfilter-monograph-main-readonly-review-findings-2026-08-03.md` no longer contains that original review; it now contains the later independent audit, and the original review artifact is not preserved at a stable repository path. This note therefore revises the original review **by content and session provenance**, not by an intact file-to-file lineage.
- **Reason for revision:** a subsequent independent audit correctly identified that the prior review mixed the committed PDF and the later source tree, overstated a few classifications, and stated broad verification claims more strongly than the preserved evidence supports.
- **Scope of this revised note:** read-only review of `docs/main.pdf`, `docs/main.tex`, the included chapter/appendix sources, `docs/references.bib`, repository history relevant to the PDF/current-source boundary, existing LaTeX artifacts, and selected primary-source metadata already checked during the review process. No monograph source was changed.
- **Bottom line:** **retain most of the defect list, but narrow several claims.** The monograph still has real mathematical, target-definition, citation, build, label, structure, and register defects that should be repaired. The main changes in this revised review are:
  1. separate **committed PDF** findings from **current-source-only** findings;
  2. downgrade several items from "confirmed error" to **under-specified**, **ambiguous**, or **source-gap blocker** where the evidence does not fully settle the stronger claim;
  3. remove unsupported global claims of exhaustive mathematical verification as independently auditable facts.

---

## 1. Explicit changes from the previous review

This section states exactly what is changed relative to the superseded review.

### 1.1 Artifact-boundary corrections

The previous review treated the committed PDF and the current source tree as if they were one artifact. This revision corrects that.

- `docs/main.pdf` is a **431-page** A4 PDF built on **2026-07-20 03:48:40 HKT**.
- The current `docs/main.tex` inputs **55 chapters**; the committed PDF contains **54 chapters**.
- `docs/chapters/ch26c_hnn_surrogate_hmc.tex` and its inclusion in `main.tex` were added **after** the committed PDF.
- Between the PDF build and the current source tree, the changed monograph files are not confined to `ch32c`; they include at least:
  - `ch19e`,
  - new `ch26c`,
  - expanded `ch32c2`,
  - expanded `ch32c`,
  - `main.tex`,
  - `references.bib`.

**What changes in the review because of this:**

- Any contradiction involving `ch26c` is a **current-source contradiction**, not a statement about what the committed PDF already contains.
- The printed-chapter mapping that treated `ch26c` as printed Chapter 48 in `docs/main.pdf` was wrong; that chapter is absent from the committed PDF.
- Findings in post-PDF additions must be reported as **current-source defects**, not as defects already rendered in `docs/main.pdf`.

### 1.2 Findings narrowed or reclassified

The previous review overstated several findings. This revision changes them as follows.

#### Keep as confirmed defects

These remain real and should stay in the defect list:

- SR-UKF negative-weight filtered-factor defect (`ch17`)
- missing SR-UKF signed update/downdate derivative exposition (`ch17`/`ch12`/`app_c`)
- chapter-derived LEDH offset defect (`ch19b`)
- skip-resampling particle-filter likelihood recursion defect (`ch19`)
- finite fallback changes the target (`ch03`, `ch23`)
- GenUT misattribution relative to Ebeigbe (`ch32c`)
- two confirmed squared-TT inconsistencies:
  - missing `e^{-c_t}` multiplier,
  - retained-last vs retained-first contradiction
- ICNN target-anchoring explanation wrong for the displayed objective (`ch32e`)
- missing/broken citations and missing figures
- mis-anchored labels in `ch18b` and `ch33`
- stale cross-chapter running-cell / bootstrap-SIR claims

#### Downgrade to under-specified / ambiguous / not fully established

These should no longer be presented as fully confirmed mathematical errors:

- **third squared-TT / missing-Jacobian allegation:** revise from "confirmed error" to **under-specified coordinate handoff** unless a derivation proves that the next-step evaluator consumes physical coordinates without the required conversion.
- **covariance-contraction proposition criticism:** revise from "wrong" to **needs clearer statement tying the conditional law to the specific coupling**.
- **chunk-policy allegation in `ch32c2`:** revise from "declares chunk sizes free per-call" to **fails to state the canonical selector, stale `K=N` wording for `N>3000`, and fail-closed/no-override rule**.
- **q=20 resource-veto contradiction:** revise from "internally contradictory" to **threshold roles are unclear; not a proved contradiction**.
- **Daum–Huang citation verdict:** revise from "unfindable" to **bibliographically unresolved / likely conflated metadata**.
- **NeuTra venue wording:** revise from **fabricated** to **incorrect metadata**, unless stronger intent evidence is produced.

### 1.3 Claims removed or softened

The previous review stated broad positive verification claims too strongly for the evidence preserved in the review artifact. This revision removes or softens claims such as:

- "all propositions verified"
- "every printed digit"
- "machine precision"
- "the mathematics is remarkably sound"
- "nothing found requires reopening principal scientific positions"

These may still be true as internal review-process observations, but the review note does not preserve enough scripts, matrices, seeds, logs, or reviewer outputs to make them independently auditable. The correct status is:

- **reported by the review process**, not
- **reproducibly certified by the review artifact itself**.

### 1.4 Language corrections

This revision avoids intent-implying or stronger-than-supported labels. In particular:

- "fabricated venue" → **incorrect venue metadata**
- "unfindable citation" → **bibliographically unresolved / incomplete / conflated metadata** where appropriate
- "book contradicts itself" → **current source tree contains a contradiction** when one side postdates the committed PDF

---

## 2. Revised headline adjudication

This is the short list of the most important issues after reclassification.

| # | Finding | Revised status | Artifact scope |
|---:|---|---|---|
| 1 | SR-UKF filtered factor omits negative-weight downdates | **CONFIRMED error** | current source and committed PDF |
| 2 | SR-UKF signed update/downdate derivative chain missing | **CONFIRMED incomplete exposition** | current source and committed PDF |
| 3 | Chapter-derived LEDH offset is wrong | **CONFIRMED algebraic defect** | current source and committed PDF |
| 4 | Skip-resampling PF likelihood estimator is wrong as stated | **CONFIRMED algorithm/proof defect** | current source and committed PDF |
| 5 | Same-scalar surrogate-gradient rule overstates HMC bias | **CONFIRMED wording/logic defect**, but `ch26c` proof is current-source-only | split: early chapters in PDF, refuting theorem current source |
| 6 | Finite fallback changes the target | **CONFIRMED target-definition defect** | early chapters in PDF; direct contradiction with `ch26c` is current-source-only |
| 7 | GenUT is misattributed to whitened moments | **CONFIRMED source-faithfulness defect** | **current source only** |
| 8a | Missing `e^{-c_t}` in squared-TT contraction | **CONFIRMED error** | current source and committed PDF |
| 8b | Retained-last vs retained-first contradiction | **CONFIRMED error** | current source and committed PDF |
| 8c | Missing Jacobian / normalizer-shift claim | **UNDER-SPECIFIED** | current source and committed PDF |
| 9 | Broken/corrupted bibliography | **BROADLY CONFIRMED**; wording narrowed | split |
| 10 | Tracked source corresponding to the committed PDF, and the current source, cannot be rebuilt reproducibly from tracked figure inputs | **CONFIRMED build/reproducibility defect** | **both PDF-era tracked source and current source** |
| 11 | Mis-anchored and duplicate labels | **CONFIRMED** | current source and committed PDF |
| 12 | Structural-UKF comparison row is transposed | **LIKELY defect; heading ambiguous** | current source and committed PDF |
| 13 | Filter-choice register recommends a demoted route | **CONFIRMED ambiguous/stale guidance** | current source and committed PDF |
| 14 | ICNN target anchoring explanation wrong | **CONFIRMED** | current source and committed PDF |
| 15 | `ch32c2` chunk/derivative policy contradiction | **PARTLY CONFIRMED** | current source only |
| 16 | Running-cell and bootstrap/SIR cross-chapter claims stale | **CONFIRMED** | current source and committed PDF |
| 17 | HAC consistency theorem claim overbroad | **SOURCE-GAP BLOCKER** | current source and committed PDF |

---

## 3. Revised core findings to retain

These remain the actionable core and should stay in the review.

### 3.1 Confirmed mathematical / target defects

1. **SR-UKF filtered-factor identity breaks on negative-weight branches** (`ch17:286`).
2. **SR-UKF downdate derivative chain is not supplied** despite being promised by cross-reference (`ch17:185` to `ch12` / `app_c`).
3. **LEDH offset formula in `ch19b:550` is wrong relative to the chapter's own linear-Gaussian closure**, while the correct Li–Coates source-form expression appears later in the same chapter.
4. **Particle-filter likelihood estimator in `ch19` is wrong on the optional no-resampling branch** because the carried weights are dropped while the algorithm and proof still rely on weighted propagation.
5. **Finite fallback values in invalid regions change the sampled target** and can make it improper on unbounded invalid regions (`ch03`, `ch23`).
6. **GenUT section in `ch32c` misstates what Ebeigbe et al. match**.
7. **The ICNN objective explanation in `ch32e` is wrong for the displayed fixed-`varphi` minimization.**
8. **Squared-TT lane contains at least two confirmed internal inconsistencies** (missing `e^{-c_t}`, retained-coordinate ordering contradiction).

### 3.2 Confirmed citation / build / structural defects

1. **Eight `ch32c` citation keys are absent from `references.bib`.**
2. **NeuTra citation metadata is wrong** (venue / volume / pages).
3. **Li and Hu first names are wrong** in load-bearing particle-flow entries.
4. **Current source tree, and the tracked source corresponding to the committed PDF, are not reproducibly buildable from tracked figure inputs** because the required `.pdf` figure files are absent from Git.
5. **Mis-anchored labels in `ch18b` and `ch33` are real and can print contradictory references.**
6. **Reader map and chapter-ordering/navigation claims are stale.**
7. **Internal register leakage is extensive and should be removed in a reader-facing pass.**

### 3.3 Confirmed support gaps

1. **Parts I–III and the core quadrature chapters are citation deserts** despite classical-source dependence.
2. **Meta OT is used but uncited; Benamou–Brenier is invoked without a bibliography entry.**
3. **No omission register exists for the survey-heavy chapters.**
4. **The most load-bearing particle-flow source lacks a durable local copy.**

---

## 4. Findings that should be narrowed in the current review

These are the items I would explicitly change in the current review text.

### 4.1 The third squared-TT/Jacobian allegation

**Old framing:** confirmed algebraic error with a normalizer shift.

**Revised framing:**

> The physical/reference-coordinate handoff at the next-step query rule is under-specified. The text does not state clearly whether the saved evaluator consumes physical coordinates or already-converted reference coordinates. That is a real specification defect. However, the claimed missing Jacobian and resulting normalizer shift are not fully established from the preserved evidence and should be downgraded from confirmed error to under-specified handoff.

### 4.2 Covariance-contraction proposition criticism

**Old framing:** proposition mathematically wrong / underdetermined.

**Revised framing:**

> The proposition should tie the conditional expectation explicitly to the specific coupling being used. The current statement is looser than ideal. But law-of-total-covariance remains valid for the chosen joint law, so this is better classified as a specification/statement-quality issue than as a proved mathematical error.

### 4.3 Chunk-policy allegation

**Old framing:** chapter declares row/column chunk sizes as free user knobs.

**Revised framing:**

> Recording chunk sizes inside `eta_OT` does not by itself prove they are caller-free. The supportable defect is that the chapter omits the canonical selector identity and fail-closed validation rule, contains stale `K=N` wording that fails for `N>3000`, and does not state that any provided chunk settings must equal the repository-selected values.

### 4.4 NeuTra venue wording

**Old framing:** fabricated venue.

**Revised framing:**

> incorrect publication metadata. The current bibliographic record does not match the arXiv paper and known venue history.

### 4.5 Daum–Huang wording

**Old framing:** unfindable citation.

**Revised framing:**

> unresolved / likely conflated bibliographic record. The review can state that the entry is not specific enough to support the chapter's historical claims and that the exact canonical citation still needs to be resolved.

### 4.6 q=20 resource-veto contradiction

**Old framing:** internal contradiction.

**Revised framing:**

> threshold roles are unclear. Since the text labels 2 GB as a warning threshold and q=20 also fails a runtime cap, contradiction is too strong. The supportable finding is that the roles of the thresholds are not clearly classified.

---

## 5. Claims from the previous review that should be withdrawn or softened

I would explicitly withdraw or soften these classes of statements:

### 5.1 Broad mathematical certification claims

Withdraw or soften any wording equivalent to:

- "all major identities were re-derived"
- "all propositions checked"
- "every printed digit"
- "machine precision throughout"
- "the mathematics is remarkably sound"

unless the revised note appends:

- scripts,
- matrices,
- seeds,
- page/equation logs,
- reviewer outputs,
- or a per-claim derivation ledger.

Without that, the proper wording is:

> reviewers reported these verifications, and several headline items were independently corroborated, but the note itself does not preserve enough evidence to make those global positive claims independently auditable.

### 5.2 Broad coverage certification claims

Soften claims like:

- "15 parallel reviews covered every chapter fully"
- "headline findings independently verified"

to:

> the review process used multiple chapter-batch reviewers and some key findings were independently rechecked, but the present note does not preserve a full reviewer-output ledger.

### 5.3 Broad scientific-position conclusion

Replace:

> nothing found requires reopening principal scientific positions

with:

> this review is a defect-finding pass. It establishes multiple concrete defects and support gaps, but it does not certify the unnamed remainder of the 431-page monograph or settle whether every principal scientific position is secure.

---

## 6. Compact evidence boundary and limitations

This revised note intentionally narrows claims to what the preserved evidence supports.

### Directly checked technical and metadata sources

- **GenUT source-faithfulness finding:** checked against the local Ebeigbe et al. paper in `docs/Generalized unscented transformation for forecasting non-Gaussian processes Ebeigbe(25).pdf`, specifically the multivariate moment-construction equations, algorithm, and theorem statements used in the GenUT finding.
- **Metadata corrections:**
  - NeuTra: arXiv `1903.03704`, accessed 2026-08-03.
  - Li author correction: DOI `10.1109/TSP.2017.2703684`, accessed 2026-08-03.
  - Hu author correction: DOI `10.1002/qj.4028`, accessed 2026-08-03.
- **Repository-history boundary:** Git-history inspection of the PDF commit and later monograph-file changes.

### What was not completed book-wide

- backward snowball audit: **not completed book-wide**;
- forward snowball audit: **not completed book-wide**;
- comprehensive retraction / erratum / version-conflict audit: **not completed book-wide**;
- bibliography completeness audit: **not established**;
- correctness of every citation-led claim in the monograph: **not established**.

### Quarantine status

- quarantined sources: **none on present evidence**.

### Non-claims

This revised note does **not** conclude:

- that the bibliography is complete,
- that no cited source is retracted or superseded,
- that every uncited or unmentioned formula is correct,
- or that the unnamed remainder of the monograph is mathematically certified.

---

## 7. What I would still say strongly in the revised review

Even after the above narrowing, the revised review should still say plainly:

- the **current source tree and the tracked source corresponding to the committed PDF are not reproducibly buildable from tracked figure inputs**;
- the monograph contains **several real target/algorithm errors**;
- the bibliography and source-support state is **not review-ready** in several load-bearing sections;
- the book contains **stale structural claims** and **reader-facing internal register leakage**;
- the derivative exposition and some target-definition chapters need **substantive repair**, not cosmetic edits.

This is still a **revise substantially** verdict, not a mild polish verdict.

---

## 8. Recommended rewrite structure for the current review

I recommend the revised review use this structure:

1. **Artifact boundary and scope**
   - committed PDF vs current source tree
   - what is concluded for each
2. **Explicit changes from the prior review**
   - what was corrected, downgraded, or withdrawn
3. **Headline findings table with evidence class**
   - confirmed error / under-specified / source-gap blocker / not independently auditable from this note
4. **Retained confirmed defects**
5. **Narrowed findings and open blockers**
6. **Unsupported positive claims removed from the prior review**
7. **Repair priorities**
8. **What is not concluded**

---

## 9. Revised repair priority (unchanged in substance, clearer in scope)

1. **Freeze the artifact under repair:** choose committed 54-chapter PDF state or current 55-chapter source state.
2. **Repair target/correctness defects:** SR-UKF branch identity, signed-update derivative exposition, LEDH offset, skip-resampling PF recursion, finite fallback target definition, GenUT attribution, squared-TT `e^{-c_t}` and coordinate-order issues, ICNN objective explanation.
3. **Restore reproducible compilation:** replace/regenerate the five figures, rerun LaTeX, then re-evaluate citations/labels against a successful build.
4. **Repair bibliography and source support:** missing keys, NeuTra, Li, Hu, Daum–Huang, Vehtari, Meta OT, Benamou–Brenier, Li–Coates local copy, HAC theorem sources.
5. **Reconcile policy text tied to already-adjudicated findings:** DPF chunk selector / no-override validation / stale `K=N` wording (`finding 15`), Contract E derivative wording (`finding 15`), and HMC exactness wording (`findings 5–6`).
6. **Repair navigation/reader contract:** Reader Map, running-cell promises, stale aliases, ambiguous filter-choice row and route recommendation.
7. **Perform a reader-facing language pass:** remove internal register/governance residue, consolidate duplicates, add examples where dense sections are example-free.

---

## 10. Final revised verdict

**Substantially agree with the defect-finding direction, but revise the review record.** The previous review identified many real and important problems that should still be acted on. However, it should not stand unchanged as the final audit note because it conflates two different monograph artifacts, overclassifies several under-specified points as settled mathematical errors, and states broad verification/correctness conclusions more strongly than the preserved evidence supports.

The right move is therefore:

- **keep the urgent defect list**,
- **correct the artifact boundary**,
- **narrow the overstated classifications**, and
- **downgrade unsupported global verification claims**.

This revised note is that corrected review record.
