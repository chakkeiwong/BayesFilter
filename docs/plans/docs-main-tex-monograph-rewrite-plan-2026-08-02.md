# Monograph rewrite plan for `docs/main.tex` (point-by-point)

Date: 2026-08-02

metadata_date: 2026-08-02

## Context

- The read-only audit
  `docs/plans/docs-main-tex-read-only-audit-2026-08-01.md` identified LaTeX
  integrity defects, notation defects, provenance leaks, stale content,
  policy-compliance gaps, and structural overload across the monograph.
- The verification verdict
  `docs/plans/docs-main-tex-audit-verification-verdict-2026-08-02.md`
  confirmed essentially all falsifiable audit findings, corrected four of the
  audit's suggested fixes, and added defects the audit missed (two extra
  absolute paths, the ch28a `^star` typo, the ch22 unanchored test reference,
  and realized divergence of four restart-staging files).
- This plan turns the amended finding set into an executable, phased rewrite.
- Repository state at planning: commit `fb9a0679` with local uncommitted
  modifications (`docs/chapters/ch32c_entropic_ot_sinkhorn.tex` is among
  them). **Line numbers in this plan are worktree line numbers from
  2026-08-02 and will drift during execution; every item therefore also gives
  a text/label anchor. Workers must re-locate by anchor, not by line number.**

### Standing owner directives that constrain every edit

1. **Expansion, not compression.** Readability means more derivation and more
   worked examples, not replacing mathematics with prose. The only text that
   may be *deleted* is verbatim (or near-verbatim) duplication; everything
   else that leaves a chapter must be *moved* (appendix, new chapter, result
   note) with a forward reference. Track page count as a guardrail.
2. **First-year-PhD implementability.** New exposition must be implementable
   by junior PhD-level workers without reopening source papers.
3. **Panel feedback on quadrature/sparse-grid sections stands:** motivate with
   simple examples, carry a concrete test function through the 3D sparse-grid
   walkthrough, and thread examples through the formal sections.
4. **Mathematical discipline:** no categorical mathematical claim may be added
   without a local derivation or an exact source anchor (paper
   section/equation) checked via the locally stored copy (ResearchAssistant
   where available).
5. **Worktree hygiene:** touch only `docs/` files listed in this plan;
   preserve all unrelated dirty worktree changes (many `bayesfilter/` code
   files are currently modified — do not stage, revert, or reformat them).

## Question

Can `docs/main.tex` be brought to a review-ready monograph state —
mechanically sound (labels, references, paths), provenance-anchored,
policy-aligned, and pedagogically linear — while preserving all mathematical
content and honoring the expansion-not-compression directive?

## Scope

### In scope
- All Phase 0–5 items listed below, across `docs/main.tex`,
  `docs/chapters/*.tex`, `docs/appendices/*.tex`, and the disposition of
  shadow/staging chapter files.
- Compile and structural verification after each phase.
- A result note and reset memo at completion.

### Out of scope
- Any change to `bayesfilter/` code, tests, or benchmarks.
- New scientific experiments or reinterpretation of recorded results
  (summaries may be *weakened* to match evidence, never strengthened).
- Filling `app_c` with full factor-derivative proofs (relabel only; proof
  work is a separate campaign).
- A full derivation-tree audit of every monograph equation (separate
  MathDevMCP campaign; this plan repairs the specific verified items).
- Bibliography restructuring beyond adding citations required by items below.

## Evidence contract (document-work form)

- **Baseline:** worktree at `fb9a0679` (+ local modifications), plus the
  Phase 0 scan artifacts recorded before any edit.
- **Primary promotion criterion:** at completion, `main.tex` compiles with
  zero multiply-defined labels, no *new* undefined references or warnings
  relative to the Phase 0 baseline log, zero absolute filesystem paths in
  included sources, zero swept stale markers (list in Phase 0), and all
  Phase 1–3 items closed; Phase 4 items closed per owner-approved decisions.
- **Vetoes (any one blocks phase completion):**
  - a new compile error or new multiply-defined/undefined-reference warning;
  - content loss: an equation, derivation, proof, or worked example deleted
    rather than moved (checked in per-phase diff review; verbatim-duplicate
    removal exempt);
  - a mathematical statement changed without a local derivation or exact
    source anchor;
  - edits outside `docs/`, or staging/reverting unrelated dirty files.
- **Explanatory diagnostics (non-gating):** page-count delta per phase,
  per-chapter diff sizes, count of remaining `MacroFinance` mentions.
- **Non-claims:** completing this plan does not certify the mathematical
  correctness of every monograph statement; it repairs the verified defect
  set and editorial structure. No claim strength may increase except where an
  item adds a *derived or cited* statement (e.g., P3.6 optimality).

## Verification harness

Run at Phase 0 (baseline) and after every phase; store outputs under
`docs/plans/artifacts/main-tex-rewrite-2026-08/` (create it):

```bash
cd /home/chakwong/BayesFilter/docs
# V1: compile (record full log; do not fail the phase on pre-existing issues)
latexmk -pdf -interaction=nonstopmode main.tex; cp main.log <artifact-dir>/
# V2: duplicate labels across everything main.tex inputs (must be empty)
for f in $(grep -oP '(?<=\\input\{)[^}]+' main.tex); do
  grep -oP '(?<=\\label\{)[^}]+' "$f.tex"; done | sort | uniq -d
# V3: absolute paths (must be empty)
grep -rn '/home/' main.tex preamble.tex chapters/ appendices/
# V4: stale markers (target: empty by end of Phase 2)
grep -rn 'p31-\|p32-\|p34-\|p38-\|p50 lane\|source note\|MacroFinance note\|Phase 2B literature gate\|current MathDevMCP audit' chapters/ appendices/
# V5: page count guardrail
pdfinfo main.pdf | grep Pages
```

Notes: V1 baseline may contain pre-existing warnings — record them; later
phases are judged on *new* issues only. V4's `p3x-` patterns must be tuned in
Phase 0 to the actual label forms found (e.g. `subsec:p32-...`).

---

## Phase 0 — Preflight and baseline (no edits)

- **P0.1** Run V1–V5; store artifacts. If V1 fails to produce a PDF, record
  the failure mode as pre-existing baseline state and proceed (structural
  scans V2–V4 do not need the PDF).
- **P0.2** Record `git log -1 --format=%H -- docs/chapters_restart_staging/`
  and the same for the four divergent included chapters (`ch35b`, `ch36b`,
  `ch37`, `ch38`) to confirm the included side is the newer, authoritative
  side (filesystem timestamps already indicate this: included 2026-06-29 vs
  staging 2026-06-20). If git history contradicts this, stop and surface to
  the owner before P4.17.
- **P0.3** Inventory `\ref`/`\cref` uses of every label this plan renames or
  deletes (P1.2, P1.3, P2.32): already verified zero refs for the four
  duplicated labels; re-verify at execution time.
- **P0.4** Confirm decision defaults D1–D8 (below) with the owner, or record
  plain-language approval to proceed on the stated defaults.

## Phase 1 — Mechanical correctness fixes

Small, unambiguous, cross-reference-safe edits. Each item: defect → edit →
check.

- **P1.1 ch32c duplicate subsection.** Anchor: the two
  `\subsection{Exact teacher versus exact engineering route}` headers with
  label `sec:bf-eot-teacher-versus-engineering` (worktree lines 125 and 197).
  Edit: diff the two subsection bodies exactly (lines 125–155 vs 197–227 at
  planning time); keep the **first** copy (it carries the
  `\citep{feydy2019geomloss,charlier2021keops,...}` citation line the second
  lacks); delete the second copy entirely. Then repair the transition: the
  streaming paragraph that preceded the second copy ("This is exactly why the
  derivation below matters...") must now flow into
  `\section{Why the optimizer factorizes}` — adjust one linking sentence if
  needed. Check: V2 empty for this label; the diff of the two copies shows no
  unique mathematical content in the deleted copy (if any is found, merge it
  into the kept copy first); compile.
- **P1.2 ch32e stray label.** Anchor:
  `\section{Structured variants and boundaries}` carrying both
  `sec:bf-neural-ot-direct-structured` and
  `sec:bf-neural-ot-direct-scalable-boundary` (line 506). Edit: delete the
  stray second label at the section; keep the label on
  `\subsection{Direct-map families as a different scalable-OT answer}`
  (line 552). Zero refs exist; check V2.
- **P1.3 ch32f stray labels.** Anchors: `\section{Richer-object audit...}`
  carries `sec:bf-neural-ot-dynamic-object` plus stray
  `...-dynamic-same-scalar` (line 166); `\section{Target-changing boundary
  families}` carries `...-dynamic-boundary-families` plus stray
  `...-dynamic-sliced-localized` (line 356). Edit: delete both stray second
  labels; the canonical targets remain `\section{Same-scalar rule revisited}`
  (line 504) and `\subsection{Sliced, subspace, and localization families...}`
  (line 432). Zero refs exist; check V2. (Content merge of the overlapping
  sliced/localization sections is deferred to P4.5.)
- **P1.4 ch37 index repair.** Anchor: section
  `sec:bf-hd-fixed-branch-fd-contract`, equations
  `eq:bf-hd-ttkr-branch-indicator`, `eq:bf-hd-ttkr-copied-core-indicator`,
  `eq:bf-hd-ttkr-decreasing-window`. Defect: quotient `D_i(h)` uses direction
  `e_i`; the audit-protocol block re-uses `i` as the ladder index (`h_i`) and
  writes direction `e_j` with `j` unbound. **Do not apply the original
  audit's fix (`e_j`→`e_i`) alone — it collides with `h_i`.** Edit: rename
  the ladder index to `m` throughout the protocol block (`h_i`→`h_m`,
  `W_i`→`W_m`, `\min_{i,\pm}`→`\min_{m,\pm}`, `e(h_{i+1})`→`e(h_{m+1})`,
  etc.) and use direction `e_i` consistently; where an indicator depends on
  the direction, bind it explicitly (e.g. `I^{(i)}_\pm(h_m)`). Introduce the
  ladder with one sentence ("for a step ladder `h_1 > h_2 > \dots` indexed by
  `m`, applied per direction `i`"). Check: grep the chapter (and V4-style
  repo grep) for any other `h_i`/`W_i` uses tied to these equations; compile.
- **P1.5 absolute paths (4 sites).**
  - ch16 line 49 and ch17 line 367
    (`/home/chakwong/python/src/dsge_hmc/filters/CUTSRUKF.py`): replace with a
    descriptive reference ("the source project's CUT-SR-UKF module") plus a
    pointer to the source map appendix; if `app_f_source_map` lacks a
    corresponding entry, add one there (repo-relative/descriptive form).
  - ch26c line 935 (`/home/chakwong/python`): replace with "the source
    project" plus the app_f pointer.
  - app_e line 14
    (`/home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph`):
    replace with the repo-relative `\path{.research/ra-bayesfilter-monograph}`.
  - Check: V3 empty.
- **P1.6 ch28a observation-tangent equation.** Anchor:
  `eq:bf-ssl-lstm-observation-state-jvp` (line 1249, `\dot y^star=C\dot z.`).
  Two defects: (i) `^star` missing backslash — fix to `^\star`
  unconditionally; (ii) `\dot z` vs `\dot z^+`. Prerequisite before edit
  (ii): re-derive locally from the surrounding recursion — the state JVP
  block produces `(\dot z^+,\dot a^+,\dot c^+)` and the innovation tangent is
  added to those "+" objects, so if the observation map acts on the updated
  latent the equation must read `\dot y^\star = C\dot z^+`; confirm against
  the definition of the observation JVP
  (`eq:bf-ssl-lstm-observation-jvp-definition`) and every other
  `\dot y^\star` use in the chapter. If the derivation instead shows a
  deliberate pre-update convention, add a one-sentence redefinition instead
  of changing the symbol. Record which branch was taken.
- **P1.7 ch28 distribution parameterization.** Anchor: "defaults are
  `x_1\sim N(0,0.2)`" (line 95). Prerequisite: determine the actual
  parameterization from the harness that defines this Model B/C fixture
  (search `bayesfilter/testing/`, `tests/`, and `docs/benchmarks/` for the
  univariate nonlinear model with `0.2`); then rewrite as
  `x_1 \sim N(0, \sigma_0^2)` with the explicit numeric value and state
  whether `0.2` is the variance or the standard deviation. **Do not guess;
  the code is the ground truth.** If no code fixture pins it, state the
  chosen convention explicitly and mark it as the declared convention.
- **P1.8 app_a backend wording.** Anchor: "TensorFlow or JAX execution"
  (line 46). Edit: "TensorFlow execution or another explicitly approved
  backend". Check: grep app_a for other JAX-as-default phrasing.
- **P1.9 audit-memo typo.** The 2026-08-01 audit file's own stray `n-` at its
  line 532: leave unchanged (historical artifact policy); recorded here so it
  is not re-reported.

**Phase 1 exit:** V1 no new issues; V2 empty; V3 empty; diff review confirms
no content loss; optional semantic commit (D8).

## Phase 2 — Provenance, staleness, and wording repairs

Per-chapter targeted edits. Global rule for donor-project language: either
(a) add an exact anchor (source-map entry, result-note path, or paper
section/equation), or (b) restate as BayesFilter-local content with a short
provenance footnote. Never delete the underlying technical statement.

- **P2.1 ch01 reader map.** Rewrite `\section{Reader Map}` against the actual
  11-part structure, explicitly naming Parts V–IX (particle foundations,
  differentiable resampling/transport, learned transport, HMC target
  interpretation/verification/filter choice, high-dimensional filtering) and
  correcting the HMC/JIT/diagnostics part to its true position. Keep the
  "stable core outward" framing. (Re-check after Phase 4 splits: P4.18.)
- **P2.2 ch02 auxiliary block (D6).** After the transition contract display,
  add the auxiliary completion law
  `a_t = T_a(s_t, d_t, x_{t-1};\theta)` with one sentence: `T_a` is a
  declared deterministic completion (lags, bookkeeping, backend summaries)
  recorded in the partition metadata, and models with no auxiliary block set
  `a_t` empty. Mirror the same completion in the packed display at the
  contract (both `pack` sites, lines 37 and 120 at planning time).
- **P2.3 ch03 finite-failure policy.** Add one sentence: the smoothness
  guarantee is branchwise — valid away from branch-switch boundaries — unless
  a separate support/boundary construction is declared.
- **P2.4 ch04 fixed-center curvature.** Rewrite the section as an abstract
  curvature-initialization contract; move concrete lane/function names into
  `app_f_source_map` (add entries) with a pointer.
- **P2.5 ch06 diagnostics.** Define the three trigger states locally (one
  sentence each: no trigger / candidate trigger / confirmed trigger as
  telemetry states of the SVD-fallback monitor); recast "MacroFinance-backed
  telemetry" as generic backend telemetry with a provenance footnote.
- **P2.6 ch07 implementation policy.** Replace "MacroFinance tests explicitly
  check the policy" with the policy restated as a BayesFilter requirement
  plus a provenance footnote; add a concrete artifact anchor only if one
  exists in-repo.
- **P2.7 ch08.** (a) Replace raw fixture names (`baseline_10x3x5`, ...) with
  prose scenario descriptions; move raw names to a small appendix/source-map
  table. (b) Define or cross-reference "medium and strict HMC recovery"
  tiers where first used. (c) Rewrite "Derivative Consolidation Hypotheses"
  as a forward-looking handoff (what a future consolidation must establish),
  removing live to-do phrasing.
- **P2.8 ch09.** (a) Replace "The source note derives the innovation
  derivatives" with a two-to-three-line local derivation in BayesFilter
  notation (differentiate `v_t = y_t - Z_t a_{t|t-1} - d_t` and
  `S_t = Z_t P_{t|t-1} Z_t^\top + H_t` term-by-term with the product rule).
  (b) In "Gain and Update Derivatives", add the branch qualification: the
  filtered-covariance derivative is for the simplified update, is not the
  Joseph form, and matches only on the same algebraic branch;
  Joseph/square-root implementations must validate reconstruction to the same
  propagated covariance. (c) Recast "Evidence Status" in durable terms
  (what is established, by which artifact class), moving tool-run specifics
  to a footnote or result-note pointer.
- **P2.9 ch10.** (a) Add a compact product-rule roadmap for the Hessian
  recursions (which first-derivative objects each second-derivative term
  consumes). (b) Add a mnemonic list for the solve-form Hessian: log-det
  term, innovation-linear term, quadratic solve term. (c) Move the embedded
  "current MathDevMCP audit ... mismatch" report out of the theory chapter —
  summarize the evidence state generically and point to app_d or a result
  note.
- **P2.10 ch11.** (a) Open "Provider Map" with a miniature worked provider
  record (one parameter, one matrix, its first/second derivative entries),
  then generalize. (b) Present the parameter-transform chain rule directly
  in BayesFilter notation (it is two displays), replacing provenance-only
  justification. (c) For stationary initial conditions, either add the
  second-derivative formulas or add an explicit deferral sentence naming
  where they will live (app_c roadmap entry); default: explicit deferral.
- **P2.11 ch12.** (a) Add a short derivation or standard citation for the
  lower-triangular projection operator in the Cholesky derivative (derive:
  differentiate `L L^\top = \Sigma`, left/right multiply by `L^{-1}`,
  `L^{-\top}`, split into lower/upper parts — three displays; cite a standard
  matrix-derivative source if present in `references.bib`, else the
  derivation suffices). (b) Add the QR bridge sentence: `\Gamma^{(ij)}`
  arises from second derivatives of `Q^\top Q = I` plus triangularity of
  `R`. (c) Rename the orthogonal QR factor symbol (currently `Q_t^P`,
  colliding visually with process covariance `Q_t`) to a distinct symbol
  (suggest `\mathsf{U}_t^P`); sweep every use in the chapter and any
  cross-chapter refs. (d) Restate backend parity gates as BayesFilter-local
  requirements with a provenance footnote.
- **P2.12 ch13.** (a) In "Same-Scalar Contract", add the plain verdict
  sentence: a gradient of a different surrogate scalar is wrong relative to
  the claimed HMC target unless a corrected algorithm (with its own validity
  argument) is explicitly defined; distinguish same-scalar from
  different-target gradients in one display or sentence. (b) At the start of
  "Hessian and Observed-Information Path", add the explicit coordinate map
  between `u` and `\psi` (define `\psi` there or state `\psi = u` if they
  coincide — determine from the section's usage before writing).
- **P2.13 ch14.** (a) "MacroFinance currently supplies examples" → timeless
  phrasing ("existing project tests include...") with provenance footnote.
  (b) Condense "Audit Record" into a summary table (content preserved in the
  table cells). (FD acceptance heuristic is P3.5.)
- **P2.14 ch15.** (a) Display both covariance-update forms explicitly
  (simplified and Joseph) with a one-line stability remark and, if available
  in `references.bib`, a canonical EKF citation. (b) Open "Likelihood
  Status" with: HMC with an EKF likelihood targets the EKF posterior, not the
  exact nonlinear posterior.
- **P2.15 ch16.** (a) (path fixed in P1.5.) (b) After the CUT degree-five
  exactness discussion, add one cross-reference sentence: quadrature
  exactness does not by itself establish filtering or posterior accuracy —
  see `\ref{sec:bf-sigma-point-approximation-boundary}`. (c) Restate the
  evidence ladder as BayesFilter's adopted ladder with a historical
  provenance footnote.
- **P2.16 ch17.** (a) Add a one-line contrast (or two-row mini-table):
  factor-propagating SR-UKF versus covariance-reconstructing "square-root"
  implementations. (b) Add a roadmap paragraph at the head of the
  factor-propagating score-contract section; optional end-of-section
  validation recap.
- **P2.17 ch18.** Sweep transient provenance phrases ("source-project
  audit", unnamed notes): replace with exact anchors or clearly labeled
  project-evidence footnotes; keep the central value-robustness versus
  derivative-fragility lesson untouched.
- **P2.18 ch18b (wording layer only; split is P4.3).** (a) Chapter intro:
  "proves the law-level reason" → "derives and illustrates the law-level
  reason", plus one sentence enumerating which statements are formally proved
  (the propositions) versus illustrated. (b) Qualify the
  original-UT-pattern claim to local moment-approximation improvement under
  stated smoothness. (c) Move the actual-SV digression into a remark or
  footnote with a cross-reference to the actual-SV treatment elsewhere.
  (d) Restate the exact pushforward-equivalence condition as a displayed,
  labeled condition and forward-reference it. (e) Add a bold remark to the
  UT-accuracy proposition: local moment statement, not a global
  filtering-accuracy theorem. (f) Add provenance for numeric witness values
  in Worked Example A (script path if script-generated; otherwise mark
  hand-computed and re-verify by hand at execution). (g) Convert the
  "does not guarantee" items in "Structural Correctness Boundary" into a
  boxed checklist (formatting only).
- **P2.19 ch19.** (a) Remove the unnumbered duplicate of the
  marginal-likelihood factorization (keep the labeled equation
  `eq:bf-pf-marginal-factorization`; the first display becomes one linking
  sentence — verbatim-duplication removal, exempt from the no-deletion rule).
  (b) Add a three-step proof roadmap before the unbiasedness proof.
  (c) Add one sentence distinguishing severe scaling pressure (weight
  collapse results) from impossibility of all proposal/transport
  improvements.
- **P2.20 ch19b.** (a) Opening: state explicitly that this is a
  derivation-led foundations-from-the-literature chapter, not a survey.
  (b) Stiffness/discretization section: add citations if suitable entries
  exist in `references.bib`; otherwise mark the section explicitly as
  BayesFilter's implementation-facing consequence of the earlier ODE
  structure. (1D EDH micro-example is P4.15.)
- **P2.21 ch19f.** State the autodiff-topology rule first as a boxed
  principle; re-frame the long numerical trace as the worked example.
- **P2.22 ch20.** Replace recommended-order step (iv) ("only then attempt
  DSGE-scale SVD sigma-point HMC") with the promoted strict-SPD
  principal-square-root route as the scale route, demoting SVD sigma-point to
  a diagnostic/comparison arm. Prerequisite: cite the in-document basis
  (ch18's promoted-route statements) in the edit; if any plan/policy file
  contradicts this promotion, stop and surface.
- **P2.23 ch26.** Replace "The Phase 2B literature gate accepted NeuTra..."
  with a direct technical status statement (NeuTra is adopted as transport
  and geometry surrogate under the stated contract), with an optional
  footnote naming the historical decision note.
- **P2.24 ch26b.** (a) Weaken the cross-geometry summary language to match
  the evidence classes actually present; separate common-protocol evidence
  from heterogeneous historical context explicitly. (Catalog table is
  P4.16.)
- **P2.25 ch26c.** (a) State warm-up handling plainly: warm-up archived but
  excluded from posterior estimates (align with the canonical NeuTra policy).
  (b) Add a decision/inference-status table to the corrected-kernel result
  presentation (rows: hard veto screen; statistically supported ranking;
  descriptive-only differences; default-readiness; next evidence needed).
  (Absolute path fixed in P1.5.)
- **P2.26 ch27.** Tie "strongest local evidence stream" to concrete artifact
  anchors (result-note paths under `docs/plans/`, test module names).
- **P2.27 ch28.** (a) Rename "reference oracle" to "dense Gaussian-moment
  comparator" and sweep the chapter for consistency (an approximate
  comparator must not be called an oracle). (b) Anchor "score parity is
  certified" claims to the exact tests/result notes and state the branch
  conditions. (Parameterization fix is P1.7.)
- **P2.28 ch28a (wording layer; split is P4.4).** Add artifact anchors
  (result-note paths) for the quantitative experiment-history claims; add the
  "test construction roadmap" paragraph for the predictive-feature and
  dependence-aware sections.
- **P2.29 ch30.** Add local result-note/source-map anchors for the DZ5
  fixed-center curvature numerical claims.
- **P2.30 ch33.** Replace "current"/"previous" lane sequencing words with
  stable route names throughout.
- **P2.31 ch34.** Add the exact Jia–Xin–Cheng source anchor
  (section/equation) for the Gaussian-approximation block; consult the
  locally stored paper (ResearchAssistant) — do not cite from memory.
- **P2.32 ch35.** (a) Rename all stale `p31/p32/p34/p38` label forms
  (~19 sites, e.g. `subsec:p32-source-theorems`) to stable chapter-local
  labels; first inventory refs repo-wide (`grep -rn 'p3[1248]' docs/`) and
  retarget them. (b) Add exact source anchors (theorem/equation) for the
  "source exactness and UKF-relation results", or derive the narrow claims
  in BayesFilter notation; the existing scoping paragraph is honest and
  stays. (3D test-function pedagogy is P4.9.)
- **P2.33 ch35b.** Rewrite stale planning phrases in "Lane-Specific
  Validation Burden" in timeless language. (Altitude split is P4.10.)
- **P2.34 ch36.** Add exact Zhao–Cui section/algorithm anchors for the TT
  sequential-learning and conditional-KR-transport material, from the locally
  stored paper. (Running example is P4.11.)
- **P2.35 ch36b.** Replace stale workflow markers ("operational middle of
  the p50 lane") with chapter-local names. (Initialization unification is
  P3.4; toy branch is P4.12.)
- **P2.36 ch38.** Remove "restarted" migration language; present the
  architecture as live. (Inference-status rule is P3.3.)
- **P2.37 ch32a.** State the provenance of the pedagogical soft-resampling
  interpolation rule explicitly: check the cited source's rule and classify
  ours as either a direct simplification (with anchor) or a BayesFilter toy
  surrogate (with a non-claim). (Verification-contract addition is P3.7.)
- **P2.38 ch22 (audit miss).** Recast "MacroFinance mass-matrix tests
  check..." (line 22) with an artifact anchor or as a BayesFilter-local
  requirement plus provenance footnote — same standard as P2.6.

**Phase 2 exit:** V1 no new issues; V4 empty for the swept marker list; diff
review confirms restatements preserve technical content; optional semantic
commit.

## Phase 3 — Policy alignment and small mathematical/pedagogical additions

- **P3.1 ch32 production checklist.** Add a manifest-and-evidence-contract
  subsection: git commit, exact command, environment/conda env, CPU/GPU
  status (and TF GPU memory mode), data version, seeds, wall time, artifact
  paths, plan/result file paths, baseline/comparator, promotion criterion,
  veto diagnostics, explicit nonclaims. Mirror the current governance list;
  keep checklist form.
- **P3.2 app_g experiment templates.** Extend each template with required
  fields: question/comparator, promotion criterion, vetoes (promotion vs
  continuation), nonclaims, budget/stop conditions, artifact root,
  command/env/seeds/hardware, plan/result paths, decision table.
- **P3.3 ch38 inference-status rule.** Add the explicit rule/table:
  runtime/ESS/tail metrics are descriptive only unless a predeclared
  uncertainty analysis supports ranking; include the five-row
  inference-status table format.
- **P3.4 ch36b initialization unification.** Make the stored-object
  walkthrough's "initial cores → deterministic constant-channel
  initialization" line conditional and consistent with the earlier policy:
  constant-channel only when no scout warm start is declared; otherwise the
  branch record freezes the scout-derived warm start. State the branch-record
  field explicitly.
- **P3.5 ch14 finite-difference acceptance heuristic.** Add a minimal
  operational rule (e.g., accept a row when the central-difference error
  exhibits a decreasing window across three ladder steps and the best two
  steps agree within the declared tolerance), cross-referencing ch37's
  ladder indicators (`I`, `K`, `W`) so both chapters use one convention.
- **P3.6 ch32b coupling optimality.** After the feasibility check of
  `\Pi^{\mathrm{cell}}`, add two sentences plus (if a suitable OT text is in
  `references.bib`) a citation: the exhibited coupling is the monotone
  coupling of the sorted supports, and for one-dimensional strictly convex
  cost the monotone coupling is optimal, so `\Pi^{\mathrm{cell}} =
  \Pi_0^\star` for this cell. Also add the bridging sentence from the
  unregularized-OT baseline section back to the running cell. (This is an
  addition of a *derived* claim; the derivation was checked in the
  verification verdict.)
- **P3.7 ch32a verification contract.** Add the missing check: compare soft
  resampling against hard resampling under the same fixed randomness.
- **P3.8 ch32c bias wording.** In "Why the current reset can accumulate
  bias", replace unproved directional "bias" language with "finite-`N`
  approximation error" except where a bias direction is actually derived.
- **P3.9 ch32c2 LEDH scoping (replaces the audit's rename).** At the chapter
  opening, keep the literature-faithful name and add one scoping sentence:
  in LEDH, "exact" refers to the closed-form solution of the local
  linearized (Gaussian) flow equation at each particle, not to exact
  posterior transport; cross-reference ch19b's LEDH definition. Add the
  bullet roadmap before the row-quotient/pullback/normalization/residual
  equations.
- **P3.10 ch23 finite invalid returns.** Split into two displayed cases:
  mathematically declared out-of-support states (ordinary rejection
  semantics) versus implementation/numerical failures (diagnostics, not
  ordinary rejections); one paragraph each.
- **P3.11 ch24 CPU/GPU separation.** Separate CPU-only import discipline
  (CPU-hiding variable set before framework import; artifact records
  intentional hiding) from the GPU memory-growth rule (applies only to GPU
  processes; growth vs logical-device limit mutually exclusive). Align
  wording with the repository TensorFlow GPU memory rule.
- **P3.12 app_c relabel (D7).** Retitle/relabel as an outline-and-roadmap
  appendix; add an explicit non-claim sentence (no proof herein is complete)
  and keep the obligations list as the roadmap. Filling proofs remains out of
  scope.

**Phase 3 exit:** V1 no new issues; added claims each carry a derivation or
anchor; optional semantic commit.

## Phase 4 — Structural restructuring (each item gated by its decision)

Execution rule for every split/move: create the new file(s), move content
verbatim (then adapt transitions), update `main.tex` `\input` order, keep all
labels stable (or add `\label` aliases at new locations), re-run V1/V2, and
confirm zero content loss by a move-accounting diff (every removed block
appears in a new location).

- **P4.1 ch32c split (D3).** Proposed boundaries: (a) `ch32c` retains
  entropic-OT/Sinkhorn foundations through the factorization/updates
  material; (b) new chapter "Barycentric covariance loss and Contract E reset
  families"; (c) new chapter "Higher-moment Contract E and TT-teacher
  extensions" (absorbing the algorithm-design section). Part VI ordering:
  ch32b → ch32c → (b) → (c) → ch32c2.
- **P4.2 ch32c2 split.** Keep the custom-VJP chapter focused; move the
  Contract E–TP proposal dossier and certification material to its own
  chapter or appendix (owner choice at D3 review; default: new chapter after
  P4.1(c)).
- **P4.3 ch18b split (D4).** Proposed: (a) core chapter — structural
  predictive law, standard-vs-structural UKF, pushforward-equivalence
  condition, propositions; (b) worked-examples chapter or appendix —
  Examples A and B with the repeated failure logic abstracted into one
  reusable lemma referenced by both; (c) "Structural degeneracy, numerical
  degeneracy, and SVD" promoted to its own chapter (PSD-degenerate accuracy,
  collapsed-law diagnostics, UKF/HMC implications); (d) source-project lesson
  → case-study appendix; (e) validation gates → explicitly labeled
  BayesFilter validation-policy block or policy appendix.
- **P4.4 ch28a split (D5).** Proposed: (a) model definition +
  predictive-equivalence theory; (b) experiment chronology/results +
  matrix-free derivative design. Minimal alternative if the owner declines
  the split: strong roadmap + explicit section-role transitions only.
- **P4.5 ch32f overlap merge.** Merge the overlapping sliced/subspace/
  localization material between "Target-changing boundary families" (line
  356 block) and "Sliced, subspace, and localization families..." (line 432
  block): one authoritative treatment, genuinely repeated statements deduped,
  unique material preserved; one label survives (already ensured by P1.3).
- **P4.6 ch32d survey reduction.** Keep the warm-start core; reduce the
  mid-chapter broader-family survey to a classification table plus forward
  references; move the detailed survey prose to an appendix or a
  clearly-referenced note (moved, not deleted).
- **P4.7 ch19c restructure.** Split the Li–Coates Algorithm-1 contract block
  into four subsections (covariance lifecycle; auxiliary vs actual flow
  paths; determinant accumulation; resampling payload). Compress the
  PF-PF-inheritance subsection into a handoff lemma with a forward reference
  to ch32b (the removed prose is near-duplicate of ch32b content — verify
  before removal; otherwise move).
- **P4.8 ch33 relocations.** Move the actual-SV SR-UKF augmented-noise
  adapter derivation to the sigma-point lane (ch17 appendix or a new
  appendix), keeping the target-mismatch verdict in ch33 with a pointer.
  Move or contract-summarize the LEDH–PFPF–OT compact sensitivity score
  section toward the transport lane (ch32c2 vicinity), keeping an
  exported-contract summary in ch33.
- **P4.9 ch35 3D test function (panel feedback).** Carry one explicit
  `F(\xi_1,\xi_2,\xi_3)` (simple polynomial/exponential mix) through the
  dense-27-point versus sparse-grid walkthrough: evaluate both clouds on `F`,
  show the reduction and the retained accuracy numerically (hand-checkable
  values; state provenance per P2.18(f) convention).
- **P4.10 ch35b altitude split.** Keep oracle + derivative + same-branch
  contract in the chapter; move I/O contract, defaults, and full algorithm
  listing to an appendix or technical note with pointers.
- **P4.11 ch36 running example.** Add a tiny two-coordinate retained-object
  example and thread it through the TT/KR sections (panel-feedback pattern).
- **P4.12 ch36b toy branch.** Introduce a very small toy branch (two points,
  rank-1) before the consolidated fixed least-squares formulation.
- **P4.13 ch37 toy table.** Add the three-row toy `I/K/W` table
  (branch-identity failure; copied-core failure; decreasing-window success)
  using the P1.4 index convention.
- **P4.14 ch26b catalog table.** Compress
  "Additional historical model tests" into a summary table; move per-model
  details to an appendix or result note (moved, not deleted).
- **P4.15 ch19b micro-example.** Add the 1D linear-Gaussian EDH micro-example
  (closed-form flow for scalar Gaussian prior/likelihood) before the general
  Gaussian-closure derivation.
- **P4.16 shadow and staging disposition (D1, D2).** After P0.2 confirms the
  included side is authoritative: create `docs/chapters_archive/` with a
  README naming this plan; `git mv` the four non-included shadow chapters,
  the seven `docs/chapters_restart_staging/` files, and
  `docs/main_highdim_restart_staging.tex` into it (or delete outright if the
  owner prefers — git history preserves them either way; default: archive).
  Result: `docs/chapters/` contains only included files; no name collisions
  remain.
- **P4.17 duplicate-content audit of archived vs included ch34–ch38.** Before
  archiving, diff the three identical and four divergent staging files; if a
  divergent staging file contains unique *newer* material (contradicting
  P0.2), stop and surface to the owner.
- **P4.18 reader-map re-check.** After all Phase 4 moves, re-verify P2.1's
  reader map against the final part/chapter structure and update.

**Phase 4 exit:** V1–V5 clean per the promotion criterion; move-accounting
diff shows zero content loss; page count not decreased except by verbatim
dedup (P1.1, P2.19, P4.5, P4.7); semantic commit per approved policy.

## Phase 5 — Final verification and records

- **P5.1** Full harness run (V1–V5); store artifacts; compare against
  Phase 0 baseline (new-issue count must be zero; duplicate labels zero;
  paths zero; stale markers zero).
- **P5.2** Red-team pass over the edited chapters for: claims strengthened
  without derivation/anchor; provenance footnotes that assert more than the
  artifact shows; split transitions that orphan a forward reference.
- **P5.3** Write the result note
  (`docs/plans/docs-main-tex-monograph-rewrite-result-<date>.md`) with the
  decision table (decision; primary criterion status; veto status; main
  uncertainty; next justified action; non-claims) and the run/edit manifest
  (commit(s), commands, artifact paths).
- **P5.4** Write/update a reset memo recording the new chapter structure,
  archive location, decisions D1–D8 as resolved, and remaining known debt
  (app_c proofs; full derivation-tree audit; any Phase 4 items deferred).

## Decision points for the owner

| ID | Decision | Default in this plan |
|---|---|---|
| D1 | Shadow chapters (4 non-included files) | Archive to `docs/chapters_archive/` |
| D2 | `chapters_restart_staging/` + `main_highdim_restart_staging.tex` | Archive (after P0.2 confirmation) |
| D3 | ch32c split boundaries | Three-way split per P4.1 |
| D4 | ch18b split boundaries | Split per P4.3 |
| D5 | ch28a split vs roadmap-only | Two-way split per P4.4 |
| D6 | ch02 `a_t`: explicit law vs explicit omission | Explicit completion law |
| D7 | app_c: relabel vs fill proofs now | Relabel as roadmap appendix |
| D8 | Commit policy | One semantic commit per phase, if committing is approved |

A plain-language owner approval of the defaults is sufficient authorization;
individual Phase 1–3 items need no per-item approval.

## Skeptical plan self-audit (pre-execution)

- **Line-number drift:** heavy — mitigated by requiring anchor-based
  relocation and running Phase 1 before Phase 4 (the big movers).
- **Baseline compile may already be dirty:** the promotion criterion is
  relative (no *new* issues), so pre-existing warnings cannot silently gate
  or excuse; Phase 0 records them explicitly.
- **Code-truth dependencies:** P1.6 and P1.7 change equations/parameters and
  therefore carry prerequisites (local re-derivation; harness code check).
  Guessing is prohibited by the item text.
- **Tension between the audit's "compress/shorten" suggestions and the
  expansion directive:** resolved by the global rule — delete only verbatim
  duplication; otherwise move with pointers. Items P2.19, P4.5, P4.7, P4.14
  each name their duplication justification.
- **Wrong-baseline risk in P2.22 (ch20):** the promoted route is asserted by
  ch18 in-document; the item requires stopping if any policy file
  contradicts it.
- **Proxy-metric risk:** compile success and empty scans are *mechanical*
  gates only; they do not certify exposition quality — that is what the
  Phase 5 red-team pass and the external review (see the companion review
  request) address.
- **Archive risk (P4.16):** destructive-adjacent; gated on P0.2 git-history
  confirmation and D1/D2 approval; `git mv` (not `rm`) as default.

Audit outcome: no material flaw found under the above mitigations; the plan
may be executed after owner confirmation of D1–D8 and the companion review
(`docs-main-tex-rewrite-plan-review-request-2026-08-02.md`) — reviewer
findings are advisory except where they surface a material scientific,
content-loss, or provenance defect.
