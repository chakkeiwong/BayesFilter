# Reply to the Revised BayesFilter Monograph Review

- **Date:** 2026-08-03
- **Reviewed artifact:** `docs/plans/bayesfilter-monograph-main-readonly-review-revised-2026-08-03.md`
- **Decision:** **REVISE, narrowly.** I agree with the revised review's substantive mathematical reclassifications and most of its retained defect list. Four material record issues remain, plus one unsupported repair-list item.
- **Metadata/source date:** 2026-08-03

## Findings

### 1. GenUT is incorrectly assigned to the committed PDF

**Severity:** material provenance error

The headline table classifies finding 7 as applying to “current source and committed PDF.” That is wrong.

At the PDF commit, `41f2aa4f263d96e5575a6448d89bdd93bb262035`, `ch32c` had 1,637 lines and contained no Ebeigbe/GenUT section. The current file has 3,032 lines; the Ebeigbe/whitened-moment material is in the post-PDF addition. The Ebeigbe bibliography entry was also added after the PDF commit.

**Required correction:** classify finding 7 as **current-source-only**. It is a confirmed source-faithfulness defect in the current source, not a defect rendered in `docs/main.pdf`.

### 2. The build defect is not current-source-only

**Severity:** material provenance error

The headline table classifies finding 10 as current-source-only. The narrower statement “the current source tree cannot rebuild” is true, but the underlying missing-figure defect already existed in the PDF-era tracked source:

- `ch28a` is byte-identical between the PDF commit and current `HEAD`.
- At the PDF commit it already referenced all five `.pdf` figures.
- The same commit tracked only the five `.png` files; none of the requested PDFs existed at those Git paths.

Therefore the reproducibility defect belongs to **both the PDF-era tracked source and the current source**. What remains unproved is how the committed PDF was produced: untracked PDFs, a different build environment, conversion, or another unrecorded mechanism are possibilities, not established history.

**Required correction:** use wording such as:

> The tracked source corresponding to the committed PDF, and the current source, are not reproducibly buildable from their tracked figure inputs. The historical mechanism by which `docs/main.pdf` was produced is not established.

### 3. The `Supersedes` pointer has the wrong provenance

**Severity:** record-integrity defect

The revised note says it supersedes `docs/plans/bayesfilter-monograph-main-readonly-review-findings-2026-08-03.md`. That path now contains the independent audit, not Fable's original 201-line review. The original review was overwritten at that path when the requested independent reply was written, and neither file is tracked by Git. As a result, the revised note appears to supersede the audit that prompted the corrections rather than the original Fable review it is revising.

**Required correction:** do not use the current path as the sole identity of the superseded review. State explicitly that the note revises Fable's original review supplied in-session and that the original artifact is no longer preserved at that path. If an original copy exists elsewhere, cite its exact path and checksum. Otherwise record the provenance gap rather than implying an intact file lineage.

### 4. The revised note still lacks a compact evidence/limitations ledger

**Severity:** evidence-classification gap

The revision correctly withdraws unauditable global verification claims, but then calls itself “the corrected review record” without preserving the minimum source-support boundary needed for its remaining literature conclusions. It does not state:

- which primary technical sources were directly inspected for which claims;
- metadata sources and access date for NeuTra, Li, and Hu;
- that no complete backward or forward snowball audit was performed;
- that no comprehensive retraction, erratum, or version-conflict check was completed;
- that no source is currently quarantined;
- what literature completeness is not concluded.

This matters because finding 7 is marked `CONFIRMED source-faithfulness defect`, finding 9 includes metadata corrections, and section 3.3 makes survey-completeness claims. Those claims need an explicit evidence boundary even in a compact review.

**Required correction:** add a short source-support and limitations section. At minimum it should record:

- Ebeigbe as the checked technical seed for finding 7, with the inspected equations/algorithm;
- DOI `10.1109/TSP.2017.2703684`, DOI `10.1002/qj.4028`, and arXiv `1903.03704` as metadata anchors, accessed 2026-08-03;
- backward snowball: not completed book-wide;
- forward snowball: not completed book-wide;
- retraction/erratum audit: not completed comprehensively;
- quarantined sources: none on present evidence;
- nonclaims: bibliography completeness, absence of retractions, and correctness of all citation-led claims are not established.

### 5. `Mass-matrix orientation` appears only as an orphan repair item

**Severity:** moderate internal-consistency defect

Section 8 adds “mass-matrix orientation” to the policy-repair list, but the revised headline table, retained findings, narrowed findings, and support-gap sections never adjudicate such a finding. The phrase is too vague to be actionable: the monograph discusses covariance, precision, HMC mass, and induced metric objects, and an orientation error must identify the exact claimed object and the exact conflicting formula or API.

The same repair-list row uses broad labels “Contract E derivative wording” and “HMC exactness wording” even though the retained findings are more specific. Those should point back to finding 15 and findings 5–6 rather than introduce apparently new defect classes.

**Required correction:** either add an evidence-classified mass-matrix finding with exact source anchors and a covariance-versus-precision derivation, or remove it. Rewrite the other two labels as explicit references to already adjudicated findings.

## Points of Agreement

I agree with the revised review on the following substantive corrections:

- the committed PDF and current 55-chapter source must be treated as distinct artifacts;
- the SR-UKF, LEDH, skip-resampling PF, finite-fallback, ICNN, and two squared-TT defects remain actionable;
- the third squared-TT/Jacobian allegation is under-specified rather than a proved algebraic error;
- the covariance-contraction proposition is not invalidated by coupling nonuniqueness;
- the structural-UKF table is likely reversed under its intended reading but has an ambiguous heading;
- the chunk-policy criticism must be limited to the missing selector/policy identity, stale `K=N` wording, no-override validation, stop-gradient clarity, and missing per-scope tuning rule;
- “incorrect metadata” and “bibliographically unresolved/likely conflated” are the right replacements for “fabricated” and “unfindable”;
- the q=20 threshold roles are unclear, not logically contradictory on the current evidence;
- the HAC theorem claim remains a source-gap blocker;
- the previous exhaustive positive-verification and “principal scientific positions” conclusions were not auditable from the preserved review artifact.

Finding 15 may remain current-source-only as a policy-mismatch finding: the exact-divisor and per-scope tuning policies were added to the active repository governance after the PDF commit, even though the PDF-era `ch32c2` already stored chunk and stop-gradient settings.

## Final Verdict

**Mostly agree, but not yet `AGREE`.** The revision fixes the central mathematical and epistemic problems in the original review. Correct the GenUT and build artifact scopes, repair the supersession provenance, add a compact scholarly evidence boundary, and either substantiate or remove the mass-matrix orientation item. After those changes, I would agree with the revised review's substantive verdict.

## Scholarship Boundary

- **Seed papers/records used here:** Ebeigbe et al. GenUT; Li-Coates DOI record; Hu-van Leeuwen DOI record; NeuTra arXiv record.
- **Source-support summary:** this reply reuses the primary technical and official-metadata checks recorded in the prior independent audit and adds Git-history checks at the PDF commit.
- **Citation/venue summary:** metadata corrections are identity checks, not evidence of mathematical validity.
- **Backward snowball:** not repeated; no book-wide completion claimed.
- **Forward snowball:** not repeated; no book-wide completion claimed.
- **Quarantined sources:** none on present evidence.
- **Top omission risk:** the revised review itself needs the compact source-support/nonclaim ledger described above.
- **What is not concluded:** no comprehensive literature-completeness, retraction, erratum, or all-claims correctness conclusion is made.
