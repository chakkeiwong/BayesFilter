# Review request: monograph audit, verification verdict, and rewrite plan

Date: 2026-08-02

metadata_date: 2026-08-02

## Purpose

Request one thorough, independent review of the three documents below
**together**, before the rewrite plan is executed. Under current governance
this is the single material plan review for the campaign; review is advisory
except where it surfaces a material scientific, content-loss, numerical,
provenance, or destructive-action defect, which is a real blocker.

## Documents under review

1. **Original audit (read-only):**
   `docs/plans/docs-main-tex-read-only-audit-2026-08-01.md`
   — chapter-by-chapter findings, tooling assessment, tiered fix queue.
2. **Verification verdict:**
   `docs/plans/docs-main-tex-audit-verification-verdict-2026-08-02.md`
   — independent verification of the audit's falsifiable claims; four
   corrections to the audit's suggested fixes; five defects the audit missed;
   an explicit `not checked` list.
3. **Rewrite plan:**
   `docs/plans/docs-main-tex-monograph-rewrite-plan-2026-08-02.md`
   — phased point-by-point execution plan (P0–P5), verification harness,
   owner decision points D1–D8, skeptical self-audit.

Ground truth for all spot-checks is the LaTeX source tree at
`docs/main.tex` + `docs/chapters/` + `docs/appendices/` in the current
worktree (commit `fb9a0679` plus local modifications;
`docs/chapters/ch32c_entropic_ot_sinkhorn.tex` is locally modified, so line
numbers cited in documents 1–2 refer to the worktree, not `HEAD`).

## Reviewer standard

- Review as a skeptical developer and as a mathematical editor. Do not accept
  either earlier document as authority: documents 1 and 2 can both be wrong,
  and document 2 explicitly marks large classes of document 1 as
  `not checked`.
- Do not return `AGREE` while any material unexamined default remains: every
  inherited or convenient choice in the plan should either be justified,
  downgraded to a hypothesis, or recorded as a risk with a diagnostic.
- Spot-check against the LaTeX source, not against the documents' own quotes.
  A minimum spot-check set is proposed below; expand it where suspicion
  arises.
- Separate clearly in your findings: what you verified against source, what
  you judged editorially, and what you did not check.

## Specific review questions

### A. On the verification verdict (document 2)

- **A1.** Re-verify a sample of the "verified correct" claims directly
  against source (suggested minimum: the four duplicate labels and the claim
  that **no** `\ref` targets them; the ch37 `e_i`/`h_i`/`e_j` analysis; the
  ch02 absence of any `a_t` law; the ch20-versus-ch18 promoted-route
  conflict). Any failure here undermines the whole verdict — say so plainly.
- **A2.** Check the verdict's four *disagreements* with the audit's suggested
  fixes:
  1. ch37: does the proposed `h_m`/`e_i` (or bound-index) repair introduce
     any new collision elsewhere in ch37 or in chapters that reference these
     indicator equations?
  2. ch32c2: is "local exact Daum–Huang" in fact the literature-faithful
     LEDH expansion as used by Li–Coates and by ch19b, so that scoping
     "exact" (not renaming) is the correct repair? Check ch19b's definition
     and, if needed, the locally stored Li–Coates source.
  3. ch16: does the "Approximation Boundary" section genuinely cover the
     quadrature-exactness versus filter-accuracy distinction, so a
     cross-reference suffices?
  4. ch12: is the `Q_t` versus `Q_t^P` distinction correctly characterized as
     a superscript-separated near-collision rather than a strict collision?
- **A3.** The verdict claims the ch32b cell coupling is exactly the monotone
  coupling and therefore optimal for 1D quadratic cost. **Re-derive this
  independently** (masses `(1/2,1/3,1/6)` to uniform `(1/3,1/3,1/3)` on
  sorted supports `{-2,0,3}`; northwest-corner construction; monotone-
  rearrangement optimality for convex cost on the line). If the derivation
  does not hold, P3.6 must be struck from the plan.
- **A4.** Probe the verdict's completeness the same way it probed the audit:
  run at least one independent defect scan the verdict did *not* report
  (suggestions: multiply-defined labels in the *non-included* shadow files
  that would fire if they are ever re-included; `\citep`/`\cite` keys missing
  from `references.bib`; `\ref` to nonexistent labels; other `^star`-style
  missing-backslash superscripts; other machine-specific strings such as
  `chakwong` or `anaconda3`). Report anything found as new work items.

### B. On the original audit (document 1), in the light of document 2

- **B1.** Document 2 left the audit's "no material issue" verdicts
  (ch05, ch21, ch25, ch29, ch31, app_b, app_d, app_f) essentially unchecked
  and found one inconsistency (ch22). Spot-check at least two of those
  chapters against the audit's own issue classes (provenance leaks, stale
  content, unsupported claims, derivation gaps). Do the clean verdicts hold?
- **B2.** Sample two or three of the audit's *editorial* findings that
  document 2 classified as `not checked` (e.g., ch18b organization, ch19c
  density, ch26b overstated breadth) and judge whether the finding and the
  plan's corresponding item are proportionate — neither cosmetic-only nor
  over-engineered.
- **B3.** Does the audit's tier structure (Tier 1/2/3) put anything in the
  wrong tier once document 2's amendments are applied? In particular: should
  the realized staging-file divergence be Tier 1 rather than Tier 3?

### C. On the rewrite plan (document 3)

- **C1. Baseline and gates.** Is the relative promotion criterion ("no *new*
  compile issues versus the Phase 0 baseline") acceptable, or does any item
  require the stronger absolute gate? Are the vetoes (content loss; unproved
  claim changes; out-of-scope edits) checkable in practice from the per-phase
  diffs?
- **C2. Content-preservation rule.** The plan permits deletion only of
  verbatim (or near-verbatim) duplication and requires moves otherwise
  (owner directive: expansion, not compression). Check the four items that
  invoke the duplication exemption (P1.1, P2.19, P4.5, P4.7): is each
  justification actually a duplication case? Is the page-count guardrail
  sufficient to catch silent compression?
- **C3. Code-truth prerequisites.** P1.6 (ch28a `\dot z` vs `\dot z^+`) and
  P1.7 (ch28 `N(0,0.2)` parameterization) change mathematical content based
  on prerequisite checks (local re-derivation; harness code). Are the
  prerequisites specific enough that a first-year-PhD-level worker cannot
  satisfy them by guessing? Is there any other plan item that silently
  changes a mathematical statement without such a prerequisite?
- **C4. Ordering and drift.** Phases 1–3 (small edits) run before Phase 4
  (splits/moves). Does any Phase 2/3 item edit text that a Phase 4 split will
  relocate, creating wasted or conflicting work? If so, propose re-ordering
  (candidates to inspect: P2.18 vs P4.3; P2.28 vs P4.4; P3.8/P3.9 vs
  P4.1/P4.2).
- **C5. Split proposals.** Are the proposed split boundaries (P4.1 ch32c
  three-way; P4.3 ch18b; P4.4 ch28a) coherent with the part structure and
  with the reader-map rewrite (P2.1/P4.18)? Flag any split that would orphan
  cross-references or separate a derivation from the notation it depends on.
- **C6. Archive safety.** P4.16/P4.17 archive shadow and staging files after
  a git-history check (P0.2). Is the stop condition ("if a divergent staging
  file contains unique newer material, stop") sufficient, or should the plan
  require a recorded content diff for all seven staging files regardless?
- **C7. Governance proportionality.** Under current policy, does the plan
  carry any ceremony that should be removed, or is any required record
  (result note, reset memo, manifest fields) missing? The plan intends: one
  review (this one), phase-level semantic commits, result note + reset memo
  at the end.
- **C8. Silent defaults.** List every default the plan adopts without
  target-specific justification (e.g., "keep the first ch32c copy because it
  carries citations"; the proposed symbol `\mathsf{U}_t^P` in P2.11; the
  specific FD acceptance heuristic in P3.5; archive-over-delete). For each:
  reasonable, or should it be a decision point?

### D. Cross-document consistency

- **D1.** Every defect asserted in document 2 (including the five audit
  misses) must map to exactly one plan item. Verify the mapping and report
  any orphan finding or plan item with no finding behind it.
- **D2.** Where document 2 *corrected* the audit's suggested fix (ch37,
  ch32c2, ch16, ch12), confirm the plan implements the corrected version,
  not the audit's original.
- **D3.** Confirm the plan's standing-directive section faithfully reflects
  the owner's recorded preferences (expansion-not-compression with page-count
  guardrail; first-year-PhD implementability; panel feedback on worked
  examples in the quadrature/sparse-grid chapters).

## What NOT to spend time on

- Re-auditing all 55 chapters line-by-line; sample per B1/B2.
- The audit's tooling assessments of `~/MathDevMCP`, `~/research-assistant`,
  `~/DynareMCP` (not load-bearing for this campaign).
- Formatting preferences with no correctness or readability consequence.
- Launch-ceremony or approval-token mechanics; a plain-language owner
  go-ahead plus this review is the full authorization path.

## Requested deliverable

A single review note at
`docs/plans/docs-main-tex-rewrite-plan-review-2026-08-XX.md` containing:

1. Per-question findings for A1–A4, B1–B3, C1–C8, D1–D3, each classified as
   `correct` / `wrong relative to the stated target` / `unsupported` /
   `not checked` / `heuristic only`, with source anchors for anything
   verified.
2. A verdict per document: audit, verdict, plan — each `AGREE`,
   `AGREE-WITH-CHANGES` (list the changes), or `DISAGREE` (state the
   material defect).
3. A list of any **new** defects or work items found (A4, B1), tier-assigned.
4. An explicit statement of what was sampled versus exhaustively checked.
5. Blocking findings, if any, separated from advisory suggestions.

Disagreement about purely procedural formatting must not block execution;
material findings (content loss risk, wrong mathematics, wrong baseline,
provenance misstatement, unsafe archive step) do.
