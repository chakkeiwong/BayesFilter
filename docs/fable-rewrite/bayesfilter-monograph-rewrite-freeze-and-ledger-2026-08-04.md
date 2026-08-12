# BayesFilter Monograph Rewrite Freeze and Finding Ledger

- **Date:** 2026-08-04
- **Purpose:** execute the blocking **Freeze Gate** from `docs/fable-rewrite/bayesfilter-monograph-rewrite-plan-v2-2026-08-03.md` so later rewrite work can proceed from a frozen artifact boundary, frozen governing policy, and one adjudicated finding ledger.
- **Status:** **Freeze Gate PASSED for documentation planning only.** This authorizes preparation of subplans and scoped documentation repairs against the chosen artifact, but it does **not** by itself authorize broad monograph rewrites beyond the frozen scope and workstreams below.

---

## 1. Artifact choice (D0)

### Decision
**D0 = Option A.** The repair target is the **current 55-chapter source tree** rooted at:

- `docs/main.tex`
- current repository commit: `6a11b689295bfb0e58de6e6d2f84918671b5a685`

### Why this option
This is the safest progress-making choice because:

1. the user asked to execute the rewrite plan now, not to reconstruct historical artifacts first;
2. several high-priority findings concern material that exists **only** in the current source tree, including:
   - new `ch26c_hnn_surrogate_hmc.tex`,
   - expanded `ch32c_entropic_ot_sinkhorn.tex`,
   - expanded `ch32c2_ledh_pfpf_ot_custom_gradient.tex`,
   - changed `ch19e_dpf_hmc_target_suitability.tex`,
   - changed `references.bib`,
   - changed `main.tex`;
3. the current-source contradictions and build failures are the most actionable defects.

### Consequence of D0
All execution work under `docs/fable-rewrite/` is scoped to the **current source tree** unless a subplan explicitly says otherwise.

### Out-of-scope for the current execution package
The following are **not** repair targets in this campaign unless a later owner decision expands scope:

- reproducing the exact tracked source state that produced the committed 431-page `docs/main.pdf`;
- byte-for-byte forensic reconstruction of how the committed PDF was historically produced;
- structural cleanup of alternate tracked LaTeX roots beyond what is required to keep them valid.

---

## 2. Provenance manifest

### 2.1 Review inputs frozen by checksum

These are the operative review artifacts for the rewrite campaign.

| Artifact | SHA-256 |
|---|---|
| `docs/plans/bayesfilter-monograph-main-readonly-review-findings-2026-08-03.md` | `d3cea1c94a51b377bac46aedf7b42c9c144846dc9692afbc5e2eb2b1626deb2f` |
| `docs/plans/bayesfilter-monograph-main-readonly-review-revised-2026-08-03.md` | `d0142fb186c70af260b3175d5077a2ad6f44589ae083b38cae733b97dc6ee773` |
| `docs/plans/bayesfilter-monograph-main-readonly-review-revised-reply-2026-08-03.md` | `3aa2d5fe45a63a650cb573058826961d248c104b5e1efb68bb0ed7b121b2d015` |
| `docs/plans/bayesfilter-monograph-review-driven-rewrite-plan-2026-08-03.md` | `a2890f4ec5a6661d416af2fc0da40a964ea57be368734d47786502ac0474d08d` |
| `docs/fable-rewrite/bayesfilter-monograph-rewrite-plan-reply-2026-08-03.md` | `a67eee58c784dd105c0696189845e50a8f1b45cefbbaec2a10ed82bf1ab79f0a` |
| `docs/fable-rewrite/bayesfilter-monograph-rewrite-plan-v2-2026-08-03.md` | `1bc7a27c284a729bfdf11e21f7d83e80c7cbd362f59e734961206f4c53337327` |

### 2.2 Governing policy files frozen by checksum

| Policy file | SHA-256 |
|---|---|
| `AGENTS.md` | `af04f7205dcb17b9992992579d4b09c23de3b04a90bec9403e969d394d18dc20` |
| `CLAUDE.md` | `d0196a417e253262e322bf59352039e149e02db4632555f14d4460ac2f2769f5` |

### 2.3 Original-review provenance gap

The original Fable review that prompted the independent audit is **not preserved at a stable tracked repository path**. The path

- `docs/plans/bayesfilter-monograph-main-readonly-review-findings-2026-08-03.md`

now contains the later independent audit, not the original review artifact. Therefore:

- the current campaign revises the original review **by session provenance and content**, not by a clean file-to-file lineage;
- the three review notes listed above are the operative adjudication record for execution.

### 2.4 Worktree state at freeze time

Repository commit at freeze time:

- `6a11b689295bfb0e58de6e6d2f84918671b5a685`

Working tree is **dirty** with unrelated active research changes outside the monograph-rewrite campaign. Therefore all rewrite work must preserve unrelated changes and touch only the files explicitly called out by downstream subplans.

---

## 3. Governing policy boundary

This campaign is governed by the frozen pair:

- `AGENTS.md`
- `CLAUDE.md`

### Priority rule
Where they differ, **`AGENTS.md` controls**.

### Practical consequences for this campaign

1. This is a **trusted local academic repository**; governance must emphasize scientific validity, reproducibility, bounded compute, and progress, not production-service ceremony.
2. A concise documentation-repair plan with preserved evidence is sufficient; no hash-token or launch-claim machinery is required.
3. Literature-facing repairs must meet the repository's own **scholarly audit** expectations: checked technical anchors where a paper materially supports a theorem- or algorithm-level claim, local copies for load-bearing sources where required, and explicit nonclaims where sources remain unavailable.
4. The repository has active policy text on:
   - Contract E canonical LEDH reset semantics,
   - LEDH per-scope tuning,
   - DPF transport chunk selector,
   - GPU memory policy,
   - NeuTra execution/training policy.

Any monograph wording rewrite that touches these topics must be checked against both frozen policy files.

---

## 4. Root inventory

### In-scope tracked LaTeX roots

| Root | Status in this campaign | Notes |
|---|---|---|
| `docs/main.tex` | **Primary repair root** | Current 55-chapter monograph source; all chapter/appendix rewrite work is keyed to this root. |
| `docs/main_highdim_restart_staging.tex` | **Protected alternate root** | Must remain valid if touched indirectly by shared-file edits. `chapters_restart_staging/` is **not orphaned**. |

### Root-scoped label rule
Duplicate labels are defects only relative to a selected root's dependency closure. Therefore:

- no global archival move or duplicate-label cleanup may be performed until each affected root has been audited in its own closure;
- `chapters_restart_staging/` cannot be moved, archived, or rewritten casually as part of a Phase-0 cleanup.

### Root counts at freeze time

- `docs/main.tex` direct `\input{}` count: **63**
- `docs/main_highdim_restart_staging.tex` direct `\input{}` count: **60**

These are inventory facts only, not chapter counts.

---

## 5. Adjudicated finding ledger

This is the only operative defect source for the rewrite campaign.

| ID | Finding | Evidence class | Artifact scope | Disposition | Downstream workstream |
|---:|---|---|---|---|---|
| 1 | SR-UKF filtered factor omits negative-weight downdates | confirmed | both | retain | W1.M1 |
| 2 | SR-UKF signed update/downdate derivative chain missing | confirmed incomplete exposition | both | retain | W1.M2a / W1.M2b |
| 3 | Chapter-derived LEDH offset is wrong | confirmed algebraic defect | both | retain | W1.M3 |
| 4 | Skip-resampling PF likelihood estimator wrong as stated | confirmed algorithm/proof defect | both | retain | W1.M4 |
| 5 | Same-scalar surrogate-gradient rule overstates HMC bias | confirmed wording/logic defect | split: early chapters both, refuting theorem current-source-only | retain with artifact split | W1.M6 |
| 6 | Finite fallback changes the target | confirmed target-definition defect | split: early chapters both, direct contradiction with `ch26c` current-source-only | retain with artifact split | W1.M5 |
| 7 | GenUT misattributed to whitened moments | confirmed source-faithfulness defect | current-source-only | retain | W1.M7 |
| 8a | Missing `e^{-c_t}` in squared-TT contraction | confirmed | both | retain | W1.M8a |
| 8b | Retained-last vs retained-first contradiction | confirmed | both | retain | W1.M8b |
| 8c | Missing Jacobian / normalizer-shift claim | under-specified | both | narrow, do not treat as confirmed error | W1.M8c |
| 9 | Broken/corrupted bibliography | broadly confirmed | split | retain with narrower wording | W3 |
| 10 | Tracked source for committed PDF and current source are not reproducibly buildable from tracked figure inputs | confirmed build/reproducibility defect | both | retain | W0 |
| 11 | Mis-anchored and duplicate labels | confirmed | both | retain | W4 labels/mechanics |
| 12 | Structural-UKF comparison row likely transposed under intended reading | likely defect / heading ambiguous | both | retain with narrower classification | W1.M10 |
| 13 | Filter-choice register recommends a demoted route | confirmed ambiguous/stale guidance | both | retain | W2.S4 |
| 14 | ICNN target-anchoring explanation wrong | confirmed | both | retain | W1.M9 |
| 15 | `ch32c2` chunk/derivative policy contradiction | partly confirmed | current-source-only | retain in narrowed form | W2.S1–S3 |
| 16 | Running-cell and bootstrap/SIR cross-chapter claims stale | confirmed | both | retain | W4.N1 / W4.N2 |
| 17 | HAC consistency theorem claim overbroad | source-gap blocker | both | retain as blocker, not confirmed theorem error | W3.G1 |

### Additional retained secondary findings

The campaign also retains these as real but non-headline items:
- `app_b` symmetry-hypothesis gap,
- local Taylor/integrability gap in structural-UKF accuracy proposition,
- floored Sinkhorn fixed-point semantics,
- Nyström norm-assumption gap,
- Zhao–Cui preconditioning compatibility omission,
- Vehtari citation gap,
- citation deserts in Parts I–III and ch34/ch35,
- internal register leakage,
- uneven chapter structure,
- alternate-root alias hazards.

These will be assigned inside workstream subplans rather than in this master freeze note.

---

## 6. Immediate execution consequences

Because the Freeze Gate now passes, the following are authorized next steps:

1. prepare execution subplans under `docs/fable-rewrite/`;
2. perform root-scoped baseline build/inventory work;
3. begin work on the **current-source** rewrite package;
4. preserve all new evidence artifacts generated by repair checks.

The following are **still blocked** until their dedicated subplans exist:

- broad monograph text edits,
- any move/archive of `chapters_restart_staging/` or other alternate-root content,
- any overprescribed mathematical rewrite whose source basis has not been re-audited,
- any claim that the campaign has already produced a clean build or final gate.

---

## 7. Next documents to create

The next execution-package files to generate under `docs/fable-rewrite/` are:

1. `bayesfilter-monograph-rewrite-math-core-subplan-2026-08-04.md`
2. `bayesfilter-monograph-rewrite-policy-structure-subplan-2026-08-04.md`
3. `bayesfilter-monograph-rewrite-literature-bibliography-subplan-2026-08-04.md`
4. `bayesfilter-monograph-rewrite-teaching-notation-subplan-2026-08-04.md`
5. `bayesfilter-monograph-rewrite-final-gate-matrix-2026-08-04.md`

These should replace the old monolithic execution plan for actual work.

---

## 8. Final freeze verdict

**Freeze Gate passed for planning.**

The campaign now has:
- a chosen repair artifact,
- frozen review-input checksums,
- a frozen governing policy boundary,
- a root inventory,
- and one adjudicated finding ledger.

That is enough to start writing the narrower execution subplans under `docs/fable-rewrite/`.
