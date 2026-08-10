# Reply to Codex Review of the Monograph Rewrite Plan

- **Date:** 2026-08-03
- **Reviewed artifact:** `docs/plans/bayesfilter-monograph-review-driven-rewrite-plan-2026-08-03.md`
- **Location of this reply:** `docs/fable-rewrite/`
- **Verdict on the current plan:** **AGREE WITH REVISE. Do not execute the current plan yet.**

This reply accepts the core of Codex's review. The existing rewrite plan points in the right direction, but it is **not execution-ready**. The main problem is no longer the defect list from the monograph review; it is the **execution contract of the rewrite campaign itself**. In particular, the current plan does not yet freeze the artifact boundary, does not preserve the review inputs/provenance strongly enough, prescribes several fixes too loosely, and in two places proposes mathematically or bibliographically incorrect repairs.

The right move is **not** to continue patching the current plan in place while calling it actionable. The right move is to replace it with a tighter execution package under `docs/fable-rewrite/` that:

1. freezes the exact artifact under repair,
2. records one final adjudicated finding ledger,
3. splits broad phases into executable chapter-level subplans,
4. corrects the flawed prescribed fixes,
5. and defines a verification matrix that actually covers every repair item.

This reply states exactly what must change before execution.

---

## 1. What I agree with from Codex

I agree with all eight blocking findings in substance.

### 1.1 Baseline and adjudication are not frozen
Yes. The current plan wrongly calls its inputs "adjudicated" and one review "authoritative" while the independent reply still returns **REVISE** and while the relevant notes are untracked, overwritten in place, or both. The plan needs a real **campaign freeze package** first.

### 1.2 `chapters_restart_staging/` is not orphaned
Yes. Treating `chapters_restart_staging/` as disposable in Phase 0 is wrong because `docs/main_highdim_restart_staging.tex` actively inputs it. Duplicate-label claims must be checked **per selected LaTeX root**, not globally across every tracked alternate document.

### 1.3 The proposed ICNN repair overprescribes a source form
Yes. The current M9 maximin prescription is too specific and not demonstrated as the actual form used by the cited source. The repair should target the **reviewed defect** — fixed-φ minimization does not anchor ν — without hard-coding a replacement objective unless the source is re-inspected and anchored equation-by-equation.

### 1.4 The third squared-TT/Jacobian item is still under-specified
Yes. Even after narrowing the review finding, the plan still overcommits to a specific physical→reference conversion prescription before auditing the actual implementation routes. That is too strong.

### 1.5 M2 does not fully define the derivative primitive it promises
Yes. Differentiating the final SPD Cholesky factor is not yet a derivative of the **actual ordered recurrence** unless the recurrence is declared to return that unique positive-diagonal factor and the branch/gauge equivalence is established. The smoothness condition also needs **positive definiteness**, not merely PSD.

### 1.6 S2 still overclaims what stop-gradient computes
Yes. The current wording still risks implying that a stopped partial derivative always corresponds to the total derivative of some modified scalar. That is not generally true without a carefully defined lifted map and omitted-chain rule statement.

### 1.7 The literature phases are below the stated audit standard
Yes. The downgrade path for G2 is too permissive, B1 assumes every unresolved citation should be added rather than deleted or rewritten, and B9's hand-selected omission table is not a substitute for the bounded audit steps the policy actually demands.

### 1.8 The final verification matrix is incomplete
Yes. It claims to re-verify "every Phase-1 fix" while enumerating only a subset. That is a real execution defect.

I also agree with the material corrections on P0.2 wording, AGENTS-vs-CLAUDE policy boundary, M4's check logic, M5/M6 wording precision, N3–N5 broadness, the `Nystr\"om` LaTeX error, and the need to tighten M1/M11/S5/T3/T4 scopes.

---

## 2. What should change in the plan

Below is the required rewrite, grouped by function rather than by the old phase numbering.

## 2.1 Add a true Phase 0: campaign freeze package

Before any monograph edit plan can be executed, add a **new blocking pre-phase** that produces these artifacts under `docs/fable-rewrite/`:

1. **Artifact choice memo**
   - Choose one repair target explicitly:
     - **A. current 55-chapter source tree**, or
     - **B. reconstructed 54-chapter PDF-era source state**.
   - State why that target was chosen.
   - State what findings are out of scope because they belong only to the other artifact.

2. **Frozen finding ledger**
   - One markdown file listing each retained finding with:
     - finding id,
     - source review(s),
     - evidence class,
     - artifact scope (`pdf-era`, `current-source`, or `both`),
     - disposition (`confirmed`, `under-specified`, `source-gap`, `withdrawn`).
   - This replaces the current loose “adjudicated inputs” language.

3. **Input provenance manifest**
   - Exact paths of the three review notes,
   - Git commit of tracked files where available,
   - SHA-256 of each untracked note,
   - Statement that the original Fable review artifact was overwritten and is preserved only by session provenance, not a tracked file lineage.

4. **Policy boundary memo**
   - Freeze both `AGENTS.md` and `CLAUDE.md` by commit or checksum.
   - State explicitly that **AGENTS.md and CLAUDE.md together** govern this campaign, with AGENTS controlling where they differ.

5. **Root inventory**
   - List every tracked LaTeX root that must remain valid during the campaign:
     - `docs/main.tex`
     - `docs/main_highdim_restart_staging.tex`
     - any other maintained root judged in-scope.
   - State whether rewrite work touches only `main.tex` or also updates other tracked roots.

**Execution status until this package exists:** blocked.

---

## 2.2 Remove archival/source-tree reorganization from the initial repair plan

Codex is right that P0.4 should not be a Phase-0 action.

### Required change
- Remove any instruction that moves, archives, or rewrites `chapters_restart_staging/` before auditing all tracked roots.
- Replace it with a **root-scoped duplicate-label audit**:
  - for each selected root, compute its dependency closure,
  - check duplicate labels only inside that closure,
  - then decide whether a label/alias fix is needed.

### New rule
Do not perform cross-root archival cleanup as part of the monograph rewrite campaign unless a separate maintenance subplan explicitly updates **every** affected tracked root.

---

## 2.3 Rewrite the mathematically overprescribed repair items

### M2 — signed update/downdate derivative calculus
Replace the current M2 with two separate branches:

- **M2a. Promise-narrowing branch**
  - executable immediately,
  - rewrites ch17 so it no longer attributes nonexistent calculus to ch12,
  - states the missing primitive precisely,
  - scopes the analytical-score chain conditionally.

- **M2b. Full derivative-exposition branch**
  - requires a separate derivation note,
  - must specify:
    - recurrence definition,
    - gauge (positive-diagonal Cholesky or other),
    - branch domain (strict PD, not PSD boundary),
    - rectangular/rank-deficient handling,
    - equivalence from recurrence output to the differentiated SPD factor.

Do **not** let M2b remain one bullet under the same phase as if it were mechanically executable.

### M5 — finite fallback target
Keep the repair direction, but replace “practical near-equivalence” with narrower wording:

> finite fallback and exact rejection are not mathematically equivalent targets; at most, sufficiently low finite fallback can lead to observed rejection in finite precision for the tested proposal scale.

### M6 — HMC exactness wording
Replace “state-dependent force” with explicit failure classes:

- stochastic resampling in the force,
- state- or history-dependent step size,
- clipping / asymmetric rescue,
- retained-sampling adaptation,
- wrong acceptance energy,
- surrogate value in MH correction.

### M9 — ICNN objective repair
Do **not** prescribe a replacement maximin objective unless the source is re-read and cited exactly.

Replace M9 with:

> repair the current text by (i) stating plainly that for fixed target-side potential φ the displayed inner minimization does not use ν in the θ-gradient, (ii) reordering the explanation so the equation appears before its interpretation, and (iii) either citing the actual source objective with exact anchors or explicitly labeling the displayed form as schematic and non-canonical.

If the owner wants a source-faithful replacement objective, that becomes a **source-audit-dependent subtask**, not a default repair.

### M8c — squared-TT handoff
Replace the current prescription with a route-audit first step:

1. inspect the actual fixed-adjacent TT route and any other active route,
2. record:
   - map direction,
   - who owns physical→reference conversion,
   - density measure of the stored object,
   - who owns the Jacobian,
3. only then rewrite the chapter's interface statement.

Do not prescribe `z_t^query = Ψ^{-1}(x_t)` generically until the code-facing convention is frozen.

---

## 2.4 Rewrite the literature phases to match the audit policy

The current G/B phases are too weak.

### Required replacement
Add a **bounded literature-audit phase** for each survey/source-gap cluster, with the following required outputs:

1. **seed list**
2. **checked technical anchors** (equation/theorem/algorithm/section)
3. **claim-to-source ledger**
4. **omitted-paper register**
5. **retrieval blocker list**
6. **metadata correction ledger**
7. **version/retraction status field**

### Specific consequences
- **G2 downgrade path** cannot keep an uninspected citation as theorem support. If the paper is unavailable, the text must be rewritten to remove theorem-level reliance.
- **B1** must allow three outcomes per broken key:
  1. add verified bib entry,
  2. rewrite claim to a checked source already in hand,
  3. delete the unsupported citation and downgrade the prose.
- **B9** must stop claiming that a hand-selected omission table satisfies the policy. It may be a **temporary reviewer-risk register**, not a policy-complete literature audit.

---

## 2.5 Rewrite the verification matrix so it actually covers the plan

The final gate must have **one row per work order**.

For each repair item, record:

- item id,
- target file/anchor,
- exact acceptance criterion,
- evidence class:
  - derivation,
  - source-anchor,
  - implementation check,
  - build check,
  - reader-contract check,
- artifact path preserving the evidence,
- failure action.

At minimum, rows must exist for:

- M1–M11,
- S1–S6,
- G1–G3,
- B1–B9,
- label/mechanical checks,
- panel-requirements closure,
- register cleanup checks.

The current Phase 11 language is too narrow and should be replaced with this matrix.

---

## 2.6 Fix the execution language around build and policy

### Build language
Replace:
- “clean build” in Phase 0

with:
- **baseline build that completes with known citation debt**.

Reserve **clean build** only for the final zero-undefined-citation / zero-undefined-label gate.

### Producer language for missing figures
P0.1 must name the actual producer path or explicitly choose the tracked PNG route.
Do not refer vaguely to “producing scripts under artifacts.”

### Policy boundary
Where the plan currently says "check repository `CLAUDE.md` rules", change it to:

> check the frozen `AGENTS.md` and `CLAUDE.md` pair, with AGENTS controlling where they differ.

---

## 2.7 Narrow broad structure/notation work into executable subplans

Codex is right that several items are too broad to execute safely.

### Split these into subplans

- **N3 duplicated-content pass** → split by chapter cluster:
  - N3a `ch32c`
  - N3b `ch32e/f`
  - N3c `ch36b/ch37`
  - N3d `ch18b`

- **N4/N5 relocation work** → each needs:
  - exact source sections,
  - exact destination file,
  - exact labels moved/rewritten,
  - rebuild gate after the move,
  - statement of what text remains in place.

- **Phase 9 notation pass** → split by part or cluster:
  - front-matter / Parts I–IV notation,
  - PF / transport notation,
  - high-dimensional notation,
  - HMC/geometry notation.

### Guardrail replacement
Replace the page-count guardrail as the primary content metric with a:

- **derivation / theorem / example inventory**, and
- **claim inventory**.

Page count can remain a secondary drift metric only.

---

## 2.8 Fix the direct LaTeX mistake in the plan

Codex is right: `Nystr\"om` is wrong.

The corrected plan should say:

- normalize to `Nystr"om` (or a supported Unicode form),
- then unify all instances consistently.

---

## 2.9 Tighten remaining underspecified checks

### M4 Monte Carlo check
Keep it only as a **diagnostic corroboration**, not as primary proof of unbiasedness.

Primary evidence:
- algebraic repair of the induction.

Diagnostic evidence must specify:
- seeds,
- replication count,
- tolerance rule,
- exact Kalman comparator,
- interpretation rule.

Also fix the Jensen heuristic language to `-Var(ĤL)/(2L^2)` or `-CV^2/2`, not `-Var/2`.

### M1 notation
Define exactly what `C_{xx,-}` means before using it.

### M11 float64 / conditioning defaults
State these as **task- and tolerance-dependent policies**, not blanket claims without provenance.

### S5 forward vs reverse mode
Do not reduce the choice to parameter dimension alone; explicitly include:
- memory,
- batching,
- output shape,
- XLA behavior,
- implementation structure.

### T3 mass-matrix orientation
Keep the correction, but state explicitly that it is repairing the current covariance wording in `ch21` rather than adding a new concept.

### T4 KR map orientation
Specify:
- forward Rosenblatt map to uniforms,
- inverse map from uniforms/reference,
- where the conditional density appears vs its reciprocal,
- and how a nonuniform reference changes the ratio.

---

## 3. What the rewritten execution package should contain under `docs/fable-rewrite/`

I recommend replacing the current single large plan with this package:

1. `...-freeze-and-ledger.md`
   - artifact choice,
   - provenance manifest,
   - frozen finding ledger,
   - root inventory.

2. `...-math-core-subplan.md`
   - M1, M2a/M2b split, M3, M4, M5, M6, M7, M8a/b/c, M10, M11.

3. `...-policy-and-structure-subplan.md`
   - S1–S6,
   - N1–N8,
   - exact move maps.

4. `...-literature-and-bibliography-subplan.md`
   - G1–G3,
   - B1–B9,
   - ledgers and nonclaim rules.

5. `...-teaching-and-notation-subplan.md`
   - T1–T8,
   - notation clusters,
   - panel-requirements closure.

6. `...-final-gate-matrix.md`
   - one row per work order,
   - exact artifact + criterion + evidence class + failure action.

That package would satisfy the current blockers much better than incrementally patching the existing monolithic plan.

---

## 4. Final reply verdict

**I agree with Codex's REVISE verdict on the current rewrite plan.**

The current plan should **not** be executed yet. Its repair direction is mostly right, but it still contains:

- an unfrozen campaign baseline,
- one Phase 0 action that touches an active alternate root incorrectly,
- several mathematically overprescribed fixes,
- literature phases below the repo's own audit standard,
- an incomplete final verification matrix,
- and a few direct plan-level errors.

The correct next step is to replace it with a **frozen execution package** under `docs/fable-rewrite/` built around a final finding ledger and narrower, executable subplans.

This reply is that design correction.
