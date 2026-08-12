# BayesFilter Monograph Rewrite Plan v2

- **Date:** 2026-08-03
- **Location:** `docs/fable-rewrite/`
- **Supersedes for execution purposes:** `docs/plans/bayesfilter-monograph-review-driven-rewrite-plan-2026-08-03.md`
- **Reviewed inputs:**
  - `docs/plans/bayesfilter-monograph-main-readonly-review-revised-2026-08-03.md`
  - `docs/plans/bayesfilter-monograph-main-readonly-review-revised-reply-2026-08-03.md`
  - `docs/plans/bayesfilter-monograph-main-readonly-review-findings-2026-08-03.md`
  - `docs/fable-rewrite/bayesfilter-monograph-rewrite-plan-reply-2026-08-03.md`
- **Current status:** **PLAN ONLY — DO NOT EXECUTE YET.** Execution begins only after the **Freeze Gate** in Workstream 0 passes.
- **Governing policy freeze:** this campaign is governed by the frozen pair `AGENTS.md` and `CLAUDE.md`, not `CLAUDE.md` alone.

---

## 0. Purpose of this rewrite

This is a corrected rewrite plan for the monograph-repair campaign. It replaces the earlier plan's loose execution contract with:

1. an explicit **artifact freeze package**,
2. a **root-scoped** LaTeX inventory,
3. narrower and mathematically safer repair prescriptions,
4. literature/audit workstreams that match the repo's stated standards,
5. and a **verification matrix that covers every repair item**.

This document is intentionally written as a master execution package. It does not yet modify the monograph, and it does not assume that the prior review notes are final in a file-provenance sense.

---

## 1. Campaign question and nonclaims

### Question
What exact sequence of documentation, bibliography, derivation, and build repairs is required to move the monograph from its current review state to a state that is:

- reproducibly buildable from tracked inputs,
- internally consistent across its stated target/algorithm contracts,
- bibliographically resolvable in all load-bearing sections,
- readable linearly by the declared first-year-PhD audience,
- and suitable for a bounded re-review?

### What this campaign does **not** conclude

Even if all workstreams pass, this campaign does **not** certify:

- the unnamed remainder of the monograph as mathematically correct,
- bibliography completeness in the full snowball sense,
- absence of retractions/errata for every cited source,
- or promotion of any scientific claim beyond what the repaired text explicitly supports.

This is a **repair campaign**, not a proof of global correctness.

---

## 2. Freeze Gate (blocking; required before any monograph edit)

Execution is blocked until every item below exists under `docs/fable-rewrite/`.

### FZ1. Artifact choice memo
Choose one repair target and freeze it explicitly:

- **Option A (recommended):** current 55-chapter source tree.
- **Option B:** reconstructed 54-chapter PDF-era tracked source state.

The memo must state:
- chosen target,
- why it was chosen,
- which findings are `pdf-era only`, `current-source only`, or `both`.

### FZ2. Input provenance manifest
Create a manifest containing:
- exact file paths of the review inputs,
- git commit IDs for tracked inputs where available,
- SHA-256 checksums for untracked review notes,
- a plain statement that the original Fable review is no longer preserved at a stable tracked path and survives only by session provenance.

### FZ3. Final adjudicated finding ledger
Create one ledger with one row per retained finding:
- finding id,
- short title,
- source review(s),
- evidence class (`confirmed`, `under-specified`, `source-gap`, `withdrawn`),
- artifact scope (`pdf-era`, `current-source`, `both`),
- downstream repair workstream id(s).

This ledger is the only operative defect source for the campaign.

### FZ4. Root inventory
List every tracked LaTeX root that must remain valid during the campaign:
- `docs/main.tex`
- `docs/main_highdim_restart_staging.tex`
- any other tracked root intentionally kept in scope

For each root, record:
- inclusion closure,
- whether the campaign edits it directly,
- whether labels only need to be unique within that root's closure.

### FZ5. Policy freeze note
Freeze the active text of:
- `AGENTS.md`
- `CLAUDE.md`

Record both by commit or checksum and state explicitly:
> Where they differ, `AGENTS.md` controls.

### Freeze Gate success criterion
All FZ1–FZ5 artifacts exist and are self-consistent.

**Only after that may any monograph edits begin.**

---

## 3. Root-scoped execution rules

These rules apply to every workstream.

1. **Never treat `chapters_restart_staging/` as orphaned by default.** It is actively used by `docs/main_highdim_restart_staging.tex`.
2. **Never move or archive files in Phase 0.** Repository-structure cleanup is a later, separately scoped maintenance action only if every affected tracked root is updated together.
3. **Every label audit is root-scoped.** Duplicate labels are defects only relative to a selected root's dependency closure.
4. **Expansion, not compression.** Mathematical derivations are not deleted. Duplicated prose/scaffolding may be consolidated by cross-reference, but derivation/theorem/example inventory, not page count alone, is the primary guardrail.
5. **Every nontrivial mathematical repair needs an evidence artifact.** Scripts, matrices, seeds, or derivation notes must be preserved under `docs/plans/` or `docs/fable-rewrite/`.

---

## 4. Workstream map

This plan is divided into seven execution workstreams:

- **W0** — buildable baseline and mechanical inventory
- **W1** — confirmed mathematical and target-definition repairs
- **W2** — policy-text and contract wording repairs
- **W3** — literature, bibliography, and source-support repairs
- **W4** — structure, navigation, and chapter-scope repairs
- **W5** — teaching, examples, and notation repairs
- **W6** — final verification gate and bounded re-review

Each workstream below lists exact tasks and checks.

---

## W0. Buildable baseline and mechanical inventory

This is the first executable workstream after the Freeze Gate.

### W0.1 Missing-figure repair choice
**Anchor:** `ch28a_neural_network_state_space_model_applications.tex:1002, 1012, 1031, 1077, 1098`

Choose one route and record it:
- **Route PNG (default):** change `\includegraphics` to the tracked `.png` files.
- **Route PDF regeneration:** regenerate the `.pdf` files from the exact producing command(s) and track them.

If PDF regeneration is chosen, record the exact producer command and inputs. Do **not** say only "scripts under artifacts."

### W0.2 Baseline build that completes with known citation debt
Run a full clean rebuild for the chosen root.

Success for W0 is:
- build completes past the old fatal missing-figure point,
- chapter count matches the chosen artifact,
- undefined citations and labels are recorded,
- but this is **not** yet called a clean build.

Known citation debt may remain at this stage.

### W0.3 Mechanical baseline snapshot
Record:
- page count,
- chapter count,
- undefined citations,
- multiply-defined labels,
- undefined references,
- overfull count,
- extracted-PDF grep for `[?]` and `[????]`.

### W0 gate
A **baseline build that completes** exists with known debt recorded.

---

## W1. Confirmed mathematical and target-definition repairs

These are the confirmed core defects from the adjudicated finding ledger.

### W1.M1 — SR-UKF filtered-factor identity
Repair `ch17` so the filtered factor reconstructs the stated covariance on negative-weight branches.

**Constraint:** define the semantics of any new object such as `C_{xx,-}` explicitly before use.

**Evidence class:** derivation + numerical branch check.

### W1.M2a — Promise-narrowing repair for signed update/downdate calculus
Before any full derivative exposition, repair the false attribution in `ch17` that points to nonexistent `ch12` content.

This branch must:
- stop promising content that is not there,
- identify the missing primitive precisely,
- mark the analytical-score chain as conditional if the primitive remains absent.

### W1.M2b — Optional full derivative-exposition branch
If pursued, this is a separate heavy subtask.

It must specify:
- the actual recurrence being differentiated,
- the returned factor gauge,
- strict PD branch conditions (not merely PSD),
- what happens at singular or rank-deficient boundaries,
- and whether the recurrence is proven equivalent to the differentiated Cholesky target.

This branch is not mechanically implied by a final-matrix Φ-operator derivation alone.

### W1.M3 — LEDH offset repair
Repair `ch19b:550–554` and reconcile it explicitly with the later Li–Coates source-form display.

**Required check:**
- algebraic linear-case reduction,
- preserved numerical endpoint test versus Kalman mean.

### W1.M4 — Skip-resampling PF recursion repair
Repair the `ch19` algorithm/proof pair.

Primary evidence is **algebraic**, not Monte Carlo.

The Monte Carlo check may remain as a corroborating diagnostic only if it records:
- seeds,
- replication count,
- tolerance rule,
- exact comparator,
- interpretation rule.

The Jensen heuristic must be written as a coefficient-of-variation or normalized-variance statement, not an unqualified `-Var/2` phrase.

### W1.M5 — Finite-fallback target repair
Repair `ch03`/`ch23` so the modified target is defined explicitly.

Do **not** imply equivalence to the exact target. Any finite-precision discussion must be framed as observed proposal-rejection behavior, not invariant-law identity.

### W1.M6 — Same-scalar HMC wording repair
Repair `ch03`/`ch13` to separate:
- exact endpoint-MH correction,
- unadjusted dynamics,
- wrong-value acceptance,
- stochastic/asymmetric/non-involutive forces.

Do **not** use vague phrases like "state-dependent force" as if that alone were the failure class.

### W1.M7 — GenUT attribution repair
Repair `ch32c` in a way consistent with the actual checked Ebeigbe equations.

If the repo code follows a different convention, the text must either:
- match the code and label itself a local variant, or
- match the paper and note any code mismatch separately.

### W1.M8a/b/c/d — Squared-TT lane
Split into four separate tasks:
- **M8a:** missing `e^{-c_t}`
- **M8b:** retained-coordinate ordering contradiction
- **M8c:** under-specified physical/reference handoff — route audit first, no overprescribed conversion before code-facing convention is frozen
- **M8d:** companion clarifications (`M_{<j}` naming, representability condition, realized stop index, initialization contradiction)

### W1.M9 — ICNN objective/explanation repair
Do **not** prescribe a replacement source objective unless it is re-read and anchored exactly.

The minimal repair is:
- fixed-φ minimization does not use ν in the θ-gradient,
- the explanation comes after the equation,
- the displayed form is either source-faithfully cited or explicitly labeled schematic.

### W1.M10 — Structural-UKF comparison row
Repair the transposed row in `ch18b`, naming the misuse target explicitly.

### W1.M11 — Batch of smaller confirmed identity/hypothesis fixes
Keep as a batch, but retain target-specific provenance for:
- float64 finite-difference policy,
- float32 conditioning defaults,
- role classification of R-hat,
- threshold-role clarification in ch28a,
- taxonomy cell addition in ch19e,
- and every small equation-hypothesis correction.

### W1 gate
Every W1 item has:
- a precise diff target,
- a preserved evidence artifact,
- and a pass/fail check result.

---

## W2. Policy-text and contract wording repairs

### W2.S1 — DPF chunk policy wording
Repair `ch32c2` so it records the canonical selector and no-override rule without implying caller freedom.

### W2.S2 — Stop-gradient wording
Replace the current overclaim with precise partial-derivative language:

- define **derivative-neutral** as omitted chain contribution equal to zero,
- state that stopped autodiff returns a partial derivative with held-constant arguments,
- do **not** generally claim it is the exact gradient of another scalar unless a lifted scalar is explicitly defined.

The route audit must distinguish the stopped log-sum-exp shift from any stopped upstream cotangent paths.

### W2.S3 — LEDH per-scope tuning rule
Add the missing tuning-scope boundary and warm-start-only language.

### W2.S4 — Filter-choice route wording
Repair `ch20` so the promoted route is named unambiguously and stale navigation is removed.

### W2.S5 — Forward-vs-reverse derivative mode
Do not decide only by parameter dimension. The chapter text must state the actual discriminants:
- memory,
- batching,
- output shape,
- XLA behavior,
- implementation structure.

### W2.S6 — Comparison-stage statistical-evidence sentence
Add the ranking-vs-descriptive boundary where currently missing.

### W2 gate
No remaining policy sentence contradicts the frozen `AGENTS.md` + `CLAUDE.md` pair.

---

## W3. Literature, bibliography, and source-support repairs

This workstream must satisfy a bounded version of the stated literature-audit standard.

### Required ledgers for W3
Create under `docs/fable-rewrite/` or `docs/plans/`:

1. **source-support ledger**
2. **claim-to-source ledger**
3. **metadata correction ledger**
4. **omitted-paper register**
5. **retrieval blocker ledger**

### W3.G1 — HAC theorem source gap
Obtain and inspect the needed theorem text or explicitly downgrade the claim.

### W3.G2 — Acevedo–de Wiljes–Reich anchor
If the source remains unavailable, remove theorem-level reliance. Do not leave an uninspected definition-level citation as support.

### W3.G3 — Li–Coates durable local copy
Store a local tractable copy before treating it as a settled load-bearing source.

### W3.B1–B9 — bibliography repairs
Retain the old B1–B9 directions, but with this rule:

For each broken or unresolved citation, allowed outcomes are:
1. add verified entry,
2. rewrite to an already checked source,
3. remove the citation and downgrade the prose.

### W3 gate
Every literature-facing fix is backed by a ledger row or an explicit nonclaim.

---

## W4. Structure, navigation, and chapter-scope repairs

This workstream must be split into exact move maps.

### W4.N1 — Reader map rewrite
### W4.N2 — Running-cell/bootstrap claim resolution
Requires owner decision D1 before execution.

### W4.N3a–N3d — deduplication subplans
Split by cluster:
- `ch32c`
- `ch32e/f`
- `ch36b/ch37`
- `ch18b`

Each subplan must state:
- source section(s),
- retained destination,
- labels affected,
- cross-reference rewrites,
- rebuild gate.

### W4.N4 — `ch32c` appended research-note material
Must specify exact destination and what remains in `ch32c`.

### W4.N5 — `ch32c2` scope split
Must specify exact destination for certification records and OPG content.

### W4.N6 — fixed-center section placement
### W4.N7 — title/honesty repairs
### W4.N8 — misplaced synthesis subsection

### W4 gate
All move maps executed with root-scoped rebuild checks and no derivation loss.

---

## W5. Teaching, examples, and notation repairs

### W5.T1–T8
Keep the old directions, but with these corrections:

- **T3:** explicitly repair the existing covariance wording in `ch21`; do not present the mass-matrix orientation as a new standalone claim.
- **T4:** define KR direction carefully:
  - forward Rosenblatt map,
  - inverse map,
  - where the conditional density appears,
  - what changes under nonuniform reference.
  - add an exact source anchor for the proposition cited.
- **T6:** keep the quadrature closure plan, but note that requirement 6 is not satisfied until an explicit 3D test function is numerically approximated in the text.
- **T7:** derivative-spine worked example must preserve explicit matrices or formulas so the check is auditable later.

### W5 notation pass
The old monolithic notation pass is too broad.

Split it into:
- **W5.NT-front** (Parts I–IV)
- **W5.NT-pf-transport** (Parts V–VIII)
- **W5.NT-highdim** (Parts IX)
- **W5.NT-hmc-geometry** (Part X)

Use derivation/claim inventory, not page count, as the primary completeness metric.

### W5 gate
Panel-requirements audit for quadrature chapters rescored; notation clusters pass their local grep checks.

---

## W6. Final verification gate and bounded re-review

This workstream must use a **full verification matrix**.

### Required matrix columns
- item id
- target file/anchor
- acceptance criterion
- evidence class
- artifact path
- failure action

### Minimum covered rows
Every work order from:
- FZ1–FZ5
- W0.1–W0.3
- W1.M1–W1.M11
- W2.S1–W2.S6
- W3.G1–W3.G3
- W3.B1–W3.B9
- W4.N1–W4.N8 (or their split subplans)
- W5.T1–W5.T8 and notation subpasses

No claim that "every Phase-1 fix" was reverified is allowed unless every corresponding row is present and passed.

### W6 gate
- zero undefined citations,
- zero undefined references,
- zero multiply-defined labels within the selected root(s),
- every matrix row either passed or explicitly blocked with recorded reason,
- result note written.

---

## 5. Recommended next artifact set under `docs/fable-rewrite/`

To replace the old monolithic plan, create these files next:

1. `bayesfilter-monograph-rewrite-freeze-and-ledger-2026-08-03.md`
2. `bayesfilter-monograph-rewrite-math-core-subplan-2026-08-03.md`
3. `bayesfilter-monograph-rewrite-policy-structure-subplan-2026-08-03.md`
4. `bayesfilter-monograph-rewrite-literature-bibliography-subplan-2026-08-03.md`
5. `bayesfilter-monograph-rewrite-teaching-notation-subplan-2026-08-03.md`
6. `bayesfilter-monograph-rewrite-final-gate-matrix-2026-08-03.md`

These should be treated as the actual execution package.

---

## 6. Final verdict

**I agree with Codex's REVISE verdict on the current rewrite plan.**

The current plan should not be executed. Its repair direction is still mostly right, but the execution contract is not yet safe or precise enough. The next step is not to keep patching the old monolithic plan while pretending it is executable. The next step is to replace it with the frozen execution package outlined above under `docs/fable-rewrite/`.

This reply states that replacement boundary clearly.
