# Verdict on replacing the canonical monograph with the Fable rewrite

- **Date:** 2026-08-06
- **Compared roots:** `docs/main.tex` and `docs/fable-rewrite/monograph/main.tex`
- **Compared rendered artifacts:** `docs/main.pdf` and `docs/fable-rewrite/monograph/main.pdf`
- **Review method:** reachable-source diff, prior-review/provenance audit, targeted mathematical review of load-bearing repairs, active citation/bibliography audit, clean-room LaTeX builds, build-log inspection, PDF text/image inspection, and sampled visual review
- **Decision:** **Do not replace the canonical tree with the rewrite as-is. Repair the rewrite, then promote its curated source diff into `docs/`.**

## Executive verdict

The rewrite is materially better than the current canonical source and should become the basis of the canonical monograph. The current source is not a defensible alternative: a clean build stops at a missing PDF figure, and it retains confirmed mathematical, target-definition, citation, label, and policy defects that the rewrite repairs or states more honestly.

However, a direct replacement today would be premature. The rewrite still contains one replacement-blocking mathematical inconsistency in the squared-TT retained-coordinate derivation, an incomplete SR-UKF derivative lane, explicit literature-support gaps, a small prose defect in a load-bearing equation transition, a bibliography regression and unused speculative entries, and substantial typesetting debt. It is also packaged as a 20 MB standalone snapshot containing generated files and copied research artifacts that should not be merged wholesale.

The correct disposition is therefore:

1. preserve the rewrite branch;
2. repair the blockers below in the rewrite source;
3. rerun a clean build and focused mathematical/source checks;
4. promote only the curated chapter, appendix, bibliography, and required figure changes into canonical `docs/`;
5. retain the old canonical PDF and rewrite audit records as historical evidence, not as live source inputs.

This is a **conditional yes** to replacement, not approval of the current rewrite bits unchanged.

## Comparison baseline

The committed PDFs are not contemporaneous and must not be compared as if they were two builds of the same source state:

| Artifact | Build date | Pages | Extracted words | Status |
|---|---:|---:|---:|---|
| `docs/main.pdf` | 2026-07-20 | 431 | about 151,600 | stale relative to current `docs/main.tex`; includes 54 rather than the current 55 chapters |
| `docs/fable-rewrite/monograph/main.pdf` | 2026-08-06 | 493 | about 180,900 | matches a clean rebuild of the rewrite source |

The two `main.tex` files and two preambles are identical. The rewrite changes 25 of 55 active chapters and four of seven appendices, with 407 inserted and 304 deleted chapter lines plus eight inserted and 21 deleted appendix lines. Thirty active chapter/appendix files are unchanged. It is therefore best understood as a targeted repair branch, not a wholesale rewrite or redesign.

The 62-page increase is not evidence of 62 pages of newly written source. It reflects current-source inclusion, pagination shifts, bibliography changes, figure handling, and the repaired build closure. Page count and compilation success are explanatory diagnostics, not correctness criteria.

## Findings

### 1. Blocker: the squared-TT retained-coordinate repair is internally inconsistent

The rewrite declares the adjacent-state order as

```tex
r_t=(x_t,x_{t-1})
```

and explicitly says that the retained current state is the **first** block. It then claims that the displayed left contractions leave this first block explicit. But the displayed construction contracts cores `1,...,D-1` into `M_{<D-1}` and evaluates `C_D` at `b_D(z_cur)`. Algebraically, that leaves the **last** coordinate/core explicit, not the first.

The claimed target and the displayed computation are therefore different. This is wrong relative to the stated retained-first convention. Renaming `z_D` to `z_cur` did not repair the contraction orientation.

Required repair: choose and use one convention throughout.

- For retained-first order `(x_t,x_{t-1})`, integrate the trailing previous-state block with right contractions and leave the first retained block explicit.
- Alternatively, restore a retained-last order and update the concrete branch, fitting-point handoff, next-step query rule, and prose consistently.

The retained-first convention appears to match the rewrite's intended concrete implementation story, so a right-contraction repair is the more natural route. It needs a small explicit scalar/vector derivation and a numerical contraction check before promotion.

**Classification:** `wrong relative to the stated target`; replacement blocker.

### 2. Major: the SR-UKF value-side repair is good, but the signed analytical derivative is still incomplete

The rewrite correctly repairs the filtered covariance factor target by including the negative covariance-weight deviation stack as well as the `K S K^T` downdate. That is a substantive mathematical improvement over the canonical text.

It also honestly states that the monograph does not derive the ordered signed rank-one update/downdate derivative needed to admit every negative-weight analytical-score branch, and Appendix C remains a placeholder. This is preferable to the canonical text's false promise that Chapter 12 already supplies that calculus.

The remaining limitation must constrain the book's release claim: the rewrite is a safer development monograph, but it is not a complete in-book derivation of the advertised analytical SR-UKF score on all signed branches. Before publication-grade release, either:

- add the primitive, branch assumptions, derivative recurrence, and reconstruction proof; or
- keep the lane explicitly conditional everywhere it is summarized or recommended.

There is also a literal prose fragment, `the implemented branch:`, immediately before the filtered-factor derivative equation. It should read, for example, `The first derivative must reconstruct the implemented branch:`.

**Classification:** value-side repair `correct`; derivative completeness `not checked/incomplete`; prose fragment `defect`; major but not necessarily a canonical-development blocker if the conditional status is retained.

### 3. Major: scholarly support is cleaner but still not publication-closed

The rewrite itself identifies load-bearing source gaps in Appendix E: exact theorem-level HAC support, durable local source support and bibliographic resolution for Li-Coates, and remaining survey/source-faithfulness audits. The HAC passage is responsibly downgraded to an implementation and audit direction rather than a closed theorem, which is an improvement, but the support gap still exists.

The literature audit for this verdict was bounded and local. It verified active citation resolution and inspected the rewrite's own source-support declarations; it did not complete full primary-source technical anchoring, backward snowballing, forward snowballing, retraction checks, or an omission register for the 152-entry bibliography. No network metadata lookup was needed to decide replacement readiness. Accordingly:

- active citations compile and resolve;
- several load-bearing claims are now properly qualified;
- publication-grade literature completeness remains **unsupported**, not established.

This does not justify retaining the worse canonical text. It means promotion should be called canonical development promotion, not scholarly final publication, unless the remaining ledgers and source anchors are completed.

**Classification:** acceptable for qualified canonical development; blocker for an unqualified publication-ready claim.

### 4. Major: bibliography resolution improved, but curation regressed in places

The canonical source cites eight absent bibliography keys. The rewrite closes all active citation-key failures and corrects the first names for Yunpeng Li and Chih-Chi Hu. Those are real improvements.

The rewrite also introduces two avoidable bibliography problems:

- it replaces the complete ICML/PMLR NeuTra conference record with a weaker arXiv/symposium record even though the canonical published record is already present and more precise;
- it appends eleven entries that are not cited anywhere in the active rewrite. Several are sparse 2025-2026 records with `and others`, generic `arXiv preprint` fields, or incomplete volume/page/identifier metadata.

Only the Acevedo-de Wiljes-Reich entry among that appended cluster is active. Unused speculative records should not be promoted merely because BibTeX ignores them. Restore the published NeuTra entry, keep only verified active additions, and move unresolved candidates to a literature ledger rather than `references.bib`.

**Classification:** active citation mechanics `correct`; bibliography curation `needs revision` before promotion.

### 5. Major: the current source is unbuildable; the rewrite is reproducibly buildable

Clean-room commands used:

```bash
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/bayesfilter-main-audit main.tex

cd docs/fable-rewrite/monograph
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/bayesfilter-rewrite-audit main.tex
```

Results:

- canonical source: fails at the first missing `ssl-lstm-launch-traces-z.pdf` input and produces no PDF;
- rewrite source: produces a 493-page PDF;
- rewrite: no undefined active citations, undefined references, or duplicate-label build warnings;
- the fresh rewrite PDF text matches the committed rewrite PDF apart from expected metadata-level differences.

The rewrite fixes the five figure paths to consume available PNGs. The five PNGs are actual build dependencies and render correctly. This is a decisive reason to use the rewrite as the migration base.

**Classification:** canonical build `broken`; rewrite build `passes`.

### 6. Major: the standalone rewrite tree must not be copied wholesale

The rewrite subtree is about 20 MB, but its build closure is much smaller. It contains:

- about 7.7 MB of generated `.mathdevmcp/latex_index.json`;
- about 7.8 MB of copied experiment-plan artifacts, although the LaTeX build consumes only five PNGs from that tree;
- generated `main.aux`, `.bbl`, `.blg`, `.fdb_latexmk`, `.fls`, `.log`, `.out`, `.toc`, and the built PDF;
- inactive predecessor chapters copied into the standalone snapshot.

A directory replacement would duplicate evidence, obscure canonical artifact provenance, and import build products. Promotion should apply the reviewed diffs to canonical files and place only required figures at stable canonical asset paths. Existing canonical files not in the active rewrite closure should be handled separately, not silently deleted or overwritten.

**Classification:** packaging/migration defect; blocker for wholesale copy, not for curated promotion.

### 7. Moderate: typesetting remains below final-release quality

The clean rewrite log contains:

- 193 overfull box warnings;
- 769 underfull box/vertical-box warnings;
- one remaining `amsmath` warning for `\atopwithdelims` in the structural-dynamics chapter.

Sampled pages show that ordinary derivation pages are readable, but long running heads, very wide equations/tables, and tiny labels in multi-panel figures need attention. The title page and overall visual system are unchanged; this is a mathematical repair branch, not a visual redesign.

These warnings do not invalidate the mathematics or block canonical development promotion after the higher-severity fixes, but they do block a claim of polished publication readiness. At minimum, fix the largest overfull boxes, the old-TeX command warning, clipped content if any, and figure-label legibility.

**Classification:** correctness-neutral release debt.

### 8. Moderate: the monograph still mixes textbook exposition with repository-state reporting

The rewrite improves the Reader Map and removes some internal scaffolding, but much of the unchanged monograph remains a hybrid of textbook, design specification, experiment report, and governance ledger. Phrases such as lane identifiers, promotion/admission language, exact repository policy identifiers, and dated experiment results are useful to maintainers but make the book age quickly and interrupt a first-year-reader narrative.

This is a structural editorial issue rather than a reason to prefer the current text, which has the same problem more severely. A later edition should separate durable method exposition from live project-status appendices or companion reports.

**Classification:** editorial/maintainability issue; not a blocker for canonical development promotion.

## Material improvements to preserve

The following rewrite changes should be retained during repair and promotion:

- reproducible PNG figure inputs and a complete clean build;
- active citation and duplicate-label cleanup;
- corrected Reader Map and navigation;
- corrected finite-fallback target semantics;
- the distinction between exact endpoint-MH correction and surrogate/wrong-value dynamics;
- weighted post-decision particle-filter recursion when resampling is skipped;
- corrected LEDH centered information offset;
- narrower, more accurate GenUT attribution;
- SR-UKF signed-stack filtered covariance repair and honest derivative limitation;
- squared-TT defensive `e^{-c_t}` scaling and explicit coordinate/Jacobian ownership, after fixing the contraction orientation;
- corrected fixed-`varphi` ICNN explanation and explicit schematic status;
- explicit canonical Contract E, derivative-composition, chunk-policy, and per-scope tuning identifiers;
- qualified statistical and source-support language instead of unsupported promotion claims.

## Recommended promotion gate

Promotion into canonical `docs/` should require the following bounded gate, in order:

1. **Repair the squared-TT contraction orientation.** Preserve a derivation note and a small numerical identity check for scalar and multi-coordinate retained blocks.
2. **Fix the SR-UKF sentence fragment and freeze its release posture.** Either complete the signed derivative primitive or keep all summary claims explicitly conditional.
3. **Curate `references.bib`.** Restore the published NeuTra record, verify the active Acevedo entry, remove or ledger unused speculative additions, and record unresolved source gaps.
4. **Move the five figures to a stable canonical asset location.** Avoid making the book depend on a copied full experiment-artifact tree.
5. **Apply source diffs, not a subtree copy.** Promote the 25 changed chapters, four changed appendices, curated bibliography changes, and required assets. Preserve unrelated canonical and staging-root files.
6. **Clean-build the canonical root twice through `latexmk`.** Require no fatal error, undefined citation, undefined reference, or duplicate active label.
7. **Run focused checks.** Recheck SR-UKF reconstruction wording, PF carried-weight algebra, LEDH offset reduction, squared-TT contraction identity, canonical Contract E identifiers, and figure inclusion.
8. **Perform a bounded visual pass.** Inspect pages associated with the largest overfull boxes and all five raster figures; fix the foreign-command warning.
9. **Record the promotion.** Write a short migration note identifying the old PDF as historical, the promoted rewrite commit, remaining nonclaims, and the exact build command.

Full publication readiness would additionally require completion of the remaining primary-source anchors and literature ledgers plus a broader typesetting/editing pass. Those are not required merely to stop using the known-broken canonical source as the development baseline.

## Decision table

| Decision dimension | Status | Reason |
|---|---|---|
| Prefer rewrite content over current canonical content | **Yes** | It repairs confirmed defects and is reproducibly buildable |
| Replace canonical tree immediately with current rewrite bits | **No** | Squared-TT contradiction, incomplete/conditional lanes, bibliography and packaging debt |
| Use rewrite as the canonical-development migration base | **Yes, after bounded repair** | The remaining blockers are localized and the current source is materially worse |
| Call the rewrite publication-ready | **No** | Source-support and typesetting gates remain open |
| Copy `docs/fable-rewrite/monograph/` wholesale | **No** | It includes generated state, duplicated artifacts, and inactive snapshot files |
| Preserve old canonical artifact | **Yes, as history** | It records the July 20 PDF-era state but is stale relative to current source |

## Scholarly-audit status

```text
decision: conditional promotion after repair; no as-is replacement
metadata_date: 2026-08-06
seed_papers: active sources implicated by the repair set, especially Li-Coates, Ebeigbe et al., NeuTra, Acevedo-de Wiljes-Reich, and the HAC references
source_support_summary: active citations resolve, several claims are responsibly downgraded, but explicit load-bearing source gaps remain
citation_venue_summary: no live citation-count or venue-rank audit was needed; published NeuTra metadata should be restored
backward_snowball_summary: not completed for the full bibliography
forward_snowball_summary: not completed; no network metadata query was required for this replacement decision
quarantined_sources: none established in this bounded audit
top_omission_risks: exact HAC theorem support, durable Li-Coates technical anchor, unresolved survey-neighbor coverage
claim_support_gaps: signed SR-UKF derivative primitive; publication-level HAC support; some survey/source-faithfulness claims
next_required_actions: repair squared-TT orientation, curate bibliography, close or preserve explicit nonclaims, then perform focused promotion checks
what_is_not_concluded: global mathematical correctness, full literature completeness, publication readiness, or correctness of every unchanged paragraph
```

## Final answer

**Do not keep the current canonical monograph as the primary development version. Do not copy the standalone rewrite into its place unchanged. Fix the localized rewrite blockers, then promote a curated rewrite diff into `docs/`.**

That route preserves the rewrite's real mathematical and engineering gains, avoids blessing a remaining wrong contraction, and avoids importing standalone-snapshot debris into the canonical source tree.
