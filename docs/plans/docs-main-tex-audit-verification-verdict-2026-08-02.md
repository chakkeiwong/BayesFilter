# Verification verdict on the 2026-08-01 read-only audit of `docs/main.tex`

Date: 2026-08-02

metadata_date: 2026-08-02

## Provenance

- Document under verification: `docs/plans/docs-main-tex-read-only-audit-2026-08-01.md`
  (the "audit" below).
- Repository state at verification: commit `fb9a0679` with local uncommitted
  modifications; `docs/chapters/ch32c_entropic_ot_sinkhorn.tex` is among the
  modified files. All line numbers below refer to the 2026-08-02 worktree
  state, not to `HEAD`.
- Mode: read-only verification of the audit's claims against the LaTeX
  sources, plus independent scans (duplicate labels, absolute paths, stale
  markers, shadow/staging file diffs) and one local mathematical derivation
  (ch32b coupling optimality).

## Method

Three verification layers were applied:

1. **Mechanical claims** (document structure, duplicate labels, shadow files,
   absolute paths, stale markers, phrase-level provenance leaks): verified by
   independent scans over every file `\input` by `docs/main.tex` (preamble,
   55 chapters, 7 appendices), not by re-running the audit's own searches.
2. **Mathematical and notation claims** (ch02 auxiliary block, ch37 index
   mismatch, ch28 parameterization, ch28a tangent, ch32b coupling, ch16
   degree-five scope, ch32c2 naming): verified by reading the cited source
   sections in context and, where needed, deriving the claim locally.
3. **Editorial-judgment claims** (chapter splits, density, roadmaps): the
   underlying facts were spot-checked where falsifiable; the judgments
   themselves are classified as `not checked` (they are matters of editorial
   policy, though several match standing panel feedback).

## Verdict summary

**The audit is substantially correct and unusually precise.** Every
mechanically falsifiable claim that was checked verified, most to the exact
line number. The audit's weaknesses are on the other side: four of its
*suggested corrections* are wrong or suboptimal, and it *misses* several
instances of its own issue classes, so its Tier 1 queue must not be treated as
an exhaustive defect list.

## Claims verified correct

### Structure (all exact)

- 11 parts, 55 chapter `\input`s, 7 appendix `\input`s, `plainnat`
  bibliography: exact.
- The four non-included shadow chapters are exactly the four the audit names
  (`ch34_highdim_gaussian_and_sparse_quadrature.tex`,
  `ch35_highdim_particle_transport_tensor_filters.tex`,
  `ch36_nonlinear_ssm_hmc_research_program.tex`,
  `ch37_highdim_filtering_candidate_synthesis.tex`).
- `docs/chapters_restart_staging/` and `docs/main_highdim_restart_staging.tex`
  exist as claimed.
- Duplicate labels: an independent scan over everything `main.tex` inputs
  found **exactly** the audit's four duplicates and no others. The audit's
  "at least four" is in fact the complete list. All eight line numbers are
  correct as of the current worktree:
  - `sec:bf-eot-teacher-versus-engineering` — ch32c lines 126 and 198;
  - `sec:bf-neural-ot-direct-scalable-boundary` — ch32e lines 506 and 552;
  - `sec:bf-neural-ot-dynamic-same-scalar` — ch32f lines 166 and 504;
  - `sec:bf-neural-ot-dynamic-sliced-localized` — ch32f lines 356 and 432.
- Additional structural fact established during verification: **no `\ref`
  anywhere in the document targets any of the four duplicated labels**, so all
  label repairs are cross-reference-safe.

### Tier 1 defect claims

- **ch32c duplicate subsection: correct.** "Exact teacher versus exact
  engineering route" appears twice (lines 125–155 and 197–227) with identical
  title, identical label, and near-identical body. Precision added by this
  verification: the copies are *not* byte-identical — the first copy carries a
  citation line (`\citep{feydy2019geomloss,charlier2021keops,...}`) that the
  second lacks, and the two copies continue into different follow-on sections
  ("The current BayesFilter bottleneck" at 156 versus "Why the optimizer
  factorizes" at 228). The repair is a careful merge, not a mechanical block
  deletion.
- **ch37 `e_i`/`e_j` mismatch: correct.** The finite-difference quotient
  (line 114) uses direction `e_i` with scalar step `h`; the branch-validity
  indicators (lines 142, 147) use `\theta_0 \pm h_i e_j` where `i` has been
  silently re-used as the step-ladder index and the direction index `j` is
  left unbound on the indicator's left-hand side.
- **ch02 under-specified auxiliary block: correct.** The structural nonlinear
  transition contract gives laws for `s_t` and `d_t`, then packs
  `(s_t, d_t, a_t)`; the only statement about `a_t` in the chapter is the role
  description at line 41. No completion or update rule for `a_t` exists
  anywhere in ch02.
- **Absolute paths in ch16/ch17: correct but under-inclusive** (see "Defects
  the audit missed" below). Confirmed at ch16 line 49 and ch17 line 367.

### Provenance, staleness, and policy claims (verbatim confirmations)

- ch01 reader map stale: **correct and understated.** The map describes a
  six-part book and calls "HMC, JIT, and diagnostic contracts" Part V; in the
  current `main.tex` that content is Part X of eleven, and Parts V–IX
  (particle foundations, differentiable transport, learned transport, HMC
  target interpretation, high-dimensional filtering) are absent from the map
  entirely.
- ch06 lines 110–111: trigger taxonomy ("no trigger, candidate trigger,
  confirmed trigger") present without local definition. Correct.
- ch09 line 48: "The source note derives the innovation derivatives" —
  present, unanchored. Correct.
- ch10 line 283: "The current MathDevMCP audit of the MacroFinance solve
  differentiated Kalman..." — transient audit language embedded in a theory
  chapter. Correct.
- ch13: chapter opens in `u` coordinates (lines 15–68); the
  Hessian/observed-information section (line 88 onward) switches to `\psi`
  (first appearance line 95) with no explicit map. Correct.
- ch19 lines 249–260: the marginal-likelihood factorization is displayed
  twice, back-to-back, the second time as
  `eq:bf-pf-marginal-factorization`. Correct.
- ch20 "Recommended Order" step (iv) recommends "DSGE-scale SVD sigma-point
  HMC": **correct, and the conflict is supported from inside the document** —
  ch18 repeatedly names the "promoted strict-SPD principal-square-root
  derivative path" (e.g. ch18 lines 673, 729, 768, 789, 1239).
- ch26 line 40: "The Phase 2B literature gate accepted NeuTra..." — internal
  campaign jargon. Correct.
- ch28 line 59 ("reference oracle" naming a dense one-step Gaussian-moment
  comparator) and line 95 (`x_1 \sim N(0,0.2)` ambiguous between variance and
  standard deviation): both correct.
- ch32 production checklist: **zero** occurrences of git commit, seeds, or
  manifest language — the audit's policy-completeness gap is confirmed against
  the run-manifest requirements in current governance.
- ch35: 19 occurrences of stale internal markers `p31/p32/p34/p38`
  (e.g. `subsec:p32-source-theorems`). Correct.
- ch36b initialization inconsistency: correct. Lines 409–410 and 524–528
  prefer scout-derived warm starts and say the branch record "should freeze a
  warm start derived from that scout", while the stored-object walkthrough at
  line 736 hard-codes "deterministic constant-channel initialization".
- app_a line 46: "TensorFlow or JAX" wording conflicts with the current
  backend policy (JAX is exception-only). Correct.
- app_c: literally self-describes as "a placeholder" for the factor-derivative
  proofs. Correct verbatim.
- MacroFinance appears in 15 chapter files, supporting the audit's #1 global
  finding that donor-project provenance leaks are widespread.
- The ch35 3D-walkthrough finding (concrete explicit test function still
  missing) independently matches standing scholarship-panel feedback, which
  corroborates the audit's judgment there.

### Mathematical claims checked by local derivation

- **ch32b running-cell coupling: the audit is correct, and the situation is
  sharper than it states.** The exhibited coupling `\Pi^{\mathrm{cell}}` is
  exactly the monotone (northwest-corner on sorted supports) coupling for
  source masses `(1/2, 1/3, 1/6)` and uniform target `(1/3, 1/3, 1/3)`.
  For one-dimensional quadratic cost, the monotone coupling **is** optimal
  (standard monotone-rearrangement optimality for convex costs on the line).
  So the chapter currently asserts strictly less ("a feasible coupling") than
  what is provably true; the audit's "add a short optimality argument" branch
  is the correct fix and is available in one or two sentences plus a standard
  citation.

## Audit claims that are imprecise, or whose suggested fix is wrong

1. **ch37 suggested fix would create a new defect** — classification: finding
   `correct`, suggested correction `wrong relative to the stated target`.
   "Use one index consistently, almost certainly `e_i`" collides with the
   step-ladder index already spelled `h_i` in the same equations:
   `\theta_0 \pm h_i e_i` would denote the wrong object. The correct repair
   disambiguates both indices (rename the ladder index, e.g. `h_m`, and keep
   direction `e_i`; or bind the direction index explicitly on the indicator,
   e.g. `I_\pm^{(i)}(h_m)`).
2. **ch32c2 rename suggestion conflicts with source-faithful naming** —
   classification: finding partially correct, primary suggested correction
   `wrong relative to the stated target`. "Local exact Daum–Huang" is the
   spelled-out literature acronym LEDH, which the monograph itself defines
   ("The local EDH (LEDH) construction...", ch19b lines 484–487) and which
   names the repository's production route (LEDH-PFPF-OT). Renaming to
   "Daum–Huang-type" would reduce fidelity to the literature and to repo
   policy naming. Only the audit's fallback — explicitly scope what "exact"
   means (exact solution of the local linearized flow equation, not exact
   posterior transport) — is the right fix.
3. **ch16 degree-five finding is locally true but overstated as a chapter
   gap.** The chapter's own "Approximation Boundary" section, two sections
   after the CUT proof, already draws precisely the requested distinction
   ("If the sigma-point likelihood is used in the target, the chain targets
   the sigma-point posterior..."). The right fix is a one-line
   cross-reference from the CUT exactness discussion to that section, not a
   new claim.
4. **ch12 `Q_t` overload is milder than implied.** The orthogonal factor is
   written `Q_t^P` (superscripted, line 194), not bare `Q_t` (process
   covariance, line 185). Confusing in one section, but not a strict symbol
   collision. The audit hedged this one as "possible notation confusion", so
   it is fairly presented; renaming remains worthwhile.

## Defects the audit missed (instances of its own issue classes)

1. **Two more absolute filesystem paths**, beyond the ch16/ch17 pair listed in
   Tier 1 item 4:
   - `docs/chapters/ch26c_hnn_surrogate_hmc.tex:935` (`/home/chakwong/python`);
   - `docs/appendices/app_e_researchassistant_workflows.tex:14`
     (`/home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph`).
2. **A LaTeX rendering bug in the very equation the audit examined**:
   `docs/chapters/ch28a_neural_network_state_space_model_applications.tex:1249`
   reads `\dot y^star = C\dot z.` — the missing backslash typesets as a
   superscript "s" followed by literal text "tar". (The audit's `C\dot z`
   versus `C\dot z^+` finding at the same line is confirmed as a genuine
   ambiguity; this second defect went unnoticed.)
3. **An inconsistent clean verdict**: ch22 is certified "no material issue",
   yet `ch22_mass_matrices.tex:22` ("MacroFinance mass-matrix tests check
   eigenvalue regularization...") is exactly the unanchored donor-project test
   reference the audit flags as a defect in ch07. (By contrast, ch19e's
   MacroFinance mentions are legitimately topical — model-class naming, not
   provenance leakage — so that clean verdict stands.)
4. **The staging-file drift risk is already realized, not hypothetical.**
   `docs/chapters_restart_staging/` contains seven files whose names collide
   with included chapters; four of them (`ch35b`, `ch36b`, `ch37`, `ch38`)
   have **diverged** from the included copies, and three (`ch34`, `ch35`,
   `ch36`) are identical stale copies. Filesystem timestamps indicate the
   included chapters are the newer side (2026-06-29 versus 2026-06-20 for the
   staging copies); this should be confirmed via `git log` before archiving.
5. Trivially: the audit file itself has a stray-character typo at its line 532
   (`n- **Issue class:**`).

## Not checked

The following audit content was **not** independently verified and should be
treated as `not checked` rather than confirmed:

- The chapter-splitting, density, and roadmap recommendations as judgments
  (their underlying facts were spot-checked where falsifiable; the judgments
  are editorial policy, though several match standing panel feedback).
- The "no material issue" verdicts for ch05, ch21, ch25, ch29, ch31, app_b,
  app_d, app_f (beyond the provenance greps that exposed the ch22 issue).
- The chapter-internal claims for ch18b (beyond its section inventory), ch23,
  ch24, ch26b, ch27, ch30 individually.
- The audit's tooling assessments of `~/MathDevMCP`, `~/research-assistant`,
  and `~/DynareMCP`.

## Classification table

| Audit component | Classification |
|---|---|
| Document-structure claims | correct (verified exactly) |
| Duplicate-label claims | correct and complete (verified exhaustively) |
| Tier 1 defect findings | correct; item 4 under-inclusive (two missed paths) |
| ch37 suggested fix | wrong relative to the stated target (index collision) |
| ch32c2 rename suggestion | wrong relative to source-faithful naming; fallback fix correct |
| ch16 degree-five gap | correct locally, overstated at chapter level |
| Provenance/staleness spot-checks (ch01, ch06, ch09, ch10, ch13, ch19, ch20, ch26, ch28, ch32, ch35, ch36b, app_a, app_c) | correct (verbatim) |
| ch32b coupling finding | correct; optimality is provable, so the stronger fix is available |
| "No material issue" verdicts | mostly not checked; ch22 verdict inconsistent with the audit's own standard |
| Editorial split/density recommendations | not checked (judgment); consistent with panel feedback |

## Bottom line

Trust the audit's findings and prioritization. Before executing a rewrite,
amend it in four places: (i) expand Tier 1 item 4 with the two missed absolute
paths and the ch28a `^star` typo; (ii) repair the ch37 fix so it does not
collide with the ladder index; (iii) replace the ch32c2 rename with an
explicit scoping sentence for "exact" in LEDH; (iv) add the realized
staging-file divergence and the ch22 unanchored test reference to the
drift/provenance work items. The companion rewrite plan
(`docs-main-tex-monograph-rewrite-plan-2026-08-02.md`) incorporates all four
amendments.
