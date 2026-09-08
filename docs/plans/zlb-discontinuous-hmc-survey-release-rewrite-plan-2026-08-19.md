# ZLB Discontinuous HMC Survey: Release Rewrite Plan

Date: 2026-08-19. Status: approved for execution (owner instruction: execute
with minimal stopping unless a real blocker appears).

## Objective

Rewrite `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.md`
and its companion package for an **internal release v1.1**, adopting the
findings of `zlb_discontinuous_hmc_survey_audit_20260819.md` (all P0 and P1
findings verified against the workspace on 2026-08-19), and repairing the
package-wide factual error the audit found in the manuscript but did not fix
in the companion ledgers.

This is an internal release, not a publication-facing release: the package is
saturated with workspace paths and fixture names, and `hostile_review.md`'s
own decision line requires human reading before any publication claim.

## Verified inputs

The following facts were independently re-verified on 2026-08-19 before this
plan was written; the rewrite may rely on them without re-checking:

1. `/home/ubuntu/workspace/python/src/dsge_hmc` exists (full package tree). The
   survey's "dsge_hmc does not exist" claim is true only of the wrong path
   `/home/ubuntu/workspace/dsge_hmc`. Anchors confirmed at exactly:
   - `models/bgs_restricted_surface_generated.py:225` (notional rule `rn`),
   - `models/bgs_restricted_surface_generated.py:228` (placeholder residual
     `(endogenous['r']) - (endogenous['rn'])`),
   - `models/bgs_restricted_surface_tf_coefficients.py:17` (`ROW_TAGS` rows
     `NRP-026`/`NRP-028` tagged `OBC_ZLB_NO_RUN_GUARD`),
   - `/home/ubuntu/workspace/python/docs/plans/actual-bgs-restricted-surface-port-master-program-v2-2026-07-09.md`
     ("no OBC/ZLB logic is active in the witness likelihoods"; evidence
     contract excludes OBC/ZLB estimation).
2. MacroFinance facts as stated in survey Sec. 13.1 are correct (softplus map
   at `two_currency_double_zlb_math.py:278,303`; `DOMESTIC_CONFIG`/
   `FOREIGN_CONFIG` alphas 1.5e-3/1.0e-3 at
   `two_currency_double_zlb_fixtures.py:44-45`; `DEFAULT_QUADRATURE_ORDER=40`;
   purely linear-Gaussian transition, no regime branch).
3. The audit's P0-2 (fully-adapted overclaim), P0-3 (C0/C1/S1 conflation),
   P0-5 (mass-matrix event kernel), P1-2 (identically-zero root; 6-dim joint
   cell probability), P1-3 (signed likelihood difference) were re-derived and
   confirmed.
4. Pericoli and Taboga (2022 JFEC, "Nearly Exact Bayesian Estimation of
   Non-linear No-Arbitrage Term-Structure Models") full text is locally
   available at `/tmp/ra-zlb-audit-20260819/local_research/papers/raw/`
   (555 KB PDF). This gates the W5 "frontier" claim repair.
5. Rendering toolchain present: pandoc 2.9.2.1 and pdflatex.

## Decisions taken in this plan (owner may override later)

- **C0's role:** the release *specifies* the root-aware C0 integration
  experiment contract but does not run it. Running C0-vs-C1 is Phase-2
  implementation work, not survey text.
- **Jump-variant geometry:** Sec. 13.3 keeps the kink classification for the
  model as coded and continues to name the three jump-restoring extensions.
  No jump-variant `target_id` is created now, because no owner instruction
  adopts regime-dependent dynamics; the target table states the trigger for
  re-derivation instead.
- **Literature triage:** only items that invalidate statements currently in
  the text block release (Pericoli--Taboga; Genz/Botev as *conditions* on the
  (84)--(87) language; Cuba-Borda et al. and Holden--Paetz as citations).
  The remaining audit-table rows stay in `omitted_papers.md` as
  implementation-phase gates with updated status.
- **Bounded Claude review retry:** attempted once in W7; a second
  `probe_timeout` is recorded, not treated as a blocker.

## Work packages

### W1 — Package-wide fact repair (dsge_hmc identity)

Files: `zlb_discontinuous_hmc_survey.md` (header line 10; Secs. 11, 12.3
trailing sentence, 14 intro, 16 Phase 5, 17), `README.md`,
`hostile_review.md` (append correction note; do not rewrite history),
`survey_enhancement_plan_20260819.md` (append correction note),
`project_roadmap.md`, and `experiment_plan.md` (both assert absence and must
be corrected; found in plan review, 2026-08-19).

Replacement wording (from the audit, verbatim in substance):

> The `dsge_hmc` package exists under `/home/ubuntu/workspace/python/src/dsge_hmc`.
> Its validated BGS restricted surface has the linear placeholder `r = rn`
> (rows tagged `OBC_ZLB_NO_RUN_GUARD`) and is not a true OBC/ZLB model. A
> genuine BGS ZLB target requires a new, source-anchored solution, selection,
> filtering, and inference architecture.

Rules: historical review documents (`hostile_review.md`, the enhancement
plan's preserved original text) get dated correction appendices, not silent
edits. The survey manuscript itself is edited in place (it is the release
artifact).

Exit: `grep -rn "does not exist\|did not exist\|was not present\|was absent"`
over the package returns no uncorrected assertion about dsge_hmc.

### W2 — Section 13 target-ladder refactor (largest edit)

1. Insert a target table before current Sec. 13.3:

   | `target_id` | Definition | Allowed claim |
   |---|---|---|
   | `mf_s1_k40_softplus` | current finite softplus program | smooth-model inference |
   | `mf_c1_k40_hardmax` | same nodes/weights, hard max | exact relative to the finite hard-quadrature model |
   | `mf_c0_root_integral_hardmax` | root-aware continuous-maturity hard integral | exact relative to the declared integral/tolerance contract |

   State: C1 is the first hard-bound counterfactual (one-operation change to
   current source); C0 differs from C1 (moving-boundary cancellation under a
   transverse root makes C0 potentially smoother than C1); tangency, endpoint
   root, and identically-bound cases are separate; "exact" is reserved for
   inference relative to a named target. State the jump-variant re-derivation
   trigger.
2. Rewrite Sec. 13.4: rename (84)--(87) an **ideal fully adapted proposal
   identity**. Add the three implementability caveats: polytope masses `Z_B`
   are numerical (Genz--Bretz), not closed form; a finite Pakman--Paninski
   trajectory is an invariant kernel, not an independent exact draw, so the
   point-independent weight (87) does not survive it; pruning to
   "non-negligible" cells changes `sum_B Z_B` and breaks unbiasedness unless
   omitted masses are certified zero. Make the primary hard-bound authority a
   **bootstrap PF** with exact Gaussian transition draws and exact
   observation-density weights (nonnegative, unbiased; PMMH-exact wrapper).
   Keep the cell decomposition as diagnostic/proposal layer; if pruned, its
   actual proposal density and importance ratio must be used.
3. Fix the crossing-lemma exceptions in Sec. 13.3: add the identically-zero
   case `L-\ell=0, S=0, C=0`; correct "at most three-dimensional" to
   per-country three-dimensional with the joint two-country/FX probability
   generically six-dimensional unless a factorization is proved.
4. Fix Sec. 13.5 P1-3: keep the one-sided *map* shift, replace the one-signed
   *likelihood* accumulation claim with the signed difference identity
   `(y-h_max)^T R^{-1} Delta - Delta^T R^{-1} Delta/2` and the FX-row caveat.
   Qualify the `epsilon = O(sqrt(alpha))` claim as a local stability
   heuristic (P1-4).
5. Rewrite the Sec. 13.7 route table against the target ladder: route (iii)
   = C1 bootstrap-PF + PMMH (authority); route (iv) = C1 joint non-centered
   HMC (co-equal exact route; agreement gate); cell-adapted proposal demoted
   to conditional on Genz/Botev inspection; rename "censored-measurement
   particle filter" to "hard-bound observation-map particle filter" (P1-7),
   reserving "censored measurement" for the Tobit noise ordering.
6. Narrow the Sec. 13.2 frontier claim per W5's Pericoli--Taboga inspection.

### W3 — General-methods corrections

1. **Sec. 2 (P0-4):** add the Markov transition kernel
   `K_theta(x, A, r) = int 1{R(x,eps,theta)=r} 1{F_r(x,eps,theta) in A} phi_theta(d eps)`,
   state when a density exists w.r.t. a declared dominating measure; require
   `pi_r(q)` in (5a) to name its branch coordinate and carry Jacobians; fix
   the (4)-starts-at-`x_0` vs (12a)-starts-at-`x_1` mismatch; fix the (53)
   numerator/proposal coordinate mismatch (propose shocks directly or include
   the transformation Jacobian).
2. **Sec. 5.1 (P1-6):** scope the proposition to a finite (or locally finite)
   piecewise-regular partition with deterministic branch gradients and a
   boundary convention; explicitly exclude implicit multi-root solvers.
3. **Sec. 6 (P0-5):** replace the unit-mass event rule with the general-`M`
   kernel: `a = n^T M^{-1} n`, `s_- = n^T M^{-1} p_- / sqrt(a)`; cross when
   `s_-^2 > 2 Delta U` with `p_+ = p_- + n (s_+ - s_-)/sqrt(a)`,
   `s_+ = sign(s_-) sqrt(s_-^2 - 2 Delta U)`; reflect otherwise with
   `p_+ = p_- - 2 n (n^T M^{-1} p_-)/(n^T M^{-1} n)`. Add boundary
   orientation and grazing/equality conventions; keep unit-mass as the
   special case.
4. **Sec. 10.2 (P0-6):** state Alenlov--Doucet--Lindsten's four assumptions;
   state that an unbiased bootstrap PF supports PMMH but not PM-HMC by
   itself; differentiable-resampling gradients are approximations or a
   different extended target until their full law and Metropolis correction
   are derived.
5. **Sec. 4.3.2 (P1-1):** require `logcdf`/`logsf`/log-density arithmetic for
   Mills ratios; variance clamps documented as roundoff safeguards only.
6. **Sec. 14.2 (P1-5):** add parametric-LCP regularity conditions (fixed
   finite horizon, continuous `q(theta)`, `M(theta)`, P-matrix over the
   declared domain) before continuity of the solution map is used.

### W4 — Section 14 split (pedagogical + BGS layers)

Retitle Sec. 14 so the NK/LCP material is explicitly pedagogical. Add a new
BGS status subsection with the verified anchors from W1 and the audit's
six-item documentation list (source/version, bound, policy equation,
expectations and terminal condition, measurement system; active-constraint
alterations incl. QE feedback; source-anchored solution operator and
verification loop; uniqueness/nonexistence domain and selection law;
likelihood value on no-verified-path; source/Dynare parity before
particle/HMC promotion). State: the first BGS deliverable is a
transition/solution kernel, not a particle filter; the restricted-surface
parity program is a no-binding baseline and must not be relabelled as ZLB
evidence. Update Sec. 16 Phase 5 accordingly.

### W5 — Literature triage

1. Inspect the local Pericoli--Taboga PDF (Secs./method level: what model,
   what inference, is it Bayesian nonlinear no-arbitrage with MCMC). Add the
   citation to Sec. 13.2 and `references.bib`; narrow the frontier claim to:
   the *inspected* shadow-rate papers predominantly use single-Gaussian
   closure; the broader Bayesian nonlinear term-structure literature
   (Pericoli--Taboga) requires comparison before any frontier language.
2. Add citations (metadata level, clearly labelled): Cuba-Borda, Guerrieri,
   Iacoviello, Zhong (inversion-filter likelihood line, named baseline in
   Sec. 14.4); Holden and Paetz (2012) (original anticipated-shock route);
   Genz and Bretz (2009) and Botev (2017) (conditions on cell-adapted
   proposals); Deligiannidis, Doucet, Pitt (2018) correlated pseudo-marginal
   (PMMH comparator, explicitly not a PM-HMC differentiability fix).
3. Update `omitted_papers.md`: move newly cited items to resolved/cited
   status; keep the remaining audit-table rows as implementation-phase gates.

### W6 — Hygiene pass

1. **(Amended in plan review, 2026-08-19.)** Do NOT renumber the existing
   equation sequence: textual `(N)` references are interleaved with citation
   years and page refs, so a global rewrite risks silent corruption, and the
   file already has an established letter-suffix convention (5a, 56a, 70a).
   Instead: insert all new W2/W3 equations with letter suffixes at their
   insertion points; scripted verification that every `\tag{...}` is unique
   and that every equation tag referenced in `claim_support.md` and
   `omitted_papers.md` still exists in the manuscript.
2. Update `references.bib` for all new citations.
3. Update `README.md` (reader path, scope paragraph, central-findings
   paragraph — the "exactly weighted particle authority available in closed
   form per step" sentence must change per W2).
4. Rerun `derivation_check_ukf_section_20260819.py` and
   `derivation_check_contract_sections_20260819.py`; if equation renumbering
   breaks their comments, update comments only (not checks).
5. Regenerate HTML and PDF with the same pandoc invocation family recorded in
   `hostile_review.md` (tex_math_single_backslash, pdflatex).

### W7 — Verification and release note

1. `pdftotext` smoke check of the regenerated PDF (title, new target table,
   references present).
2. Rerun the derivation-check scripts (already in W6; record logs).
3. Append a dated release note to `hostile_review.md` and `README.md`:
   which corrections were applied, which certifications passed, and which
   are structural-only. MathDevMCP and the bounded Claude review are
   *attempted* if their tools are reachable from this session; unreachable
   tools are recorded as not re-run, which is a gap statement, not a blocker.
4. Acceptance checklist = the audit's checklist, executed item by item and
   recorded in the release note.

## Sequencing and stopping rules

Order: W1 -> W5.1 (Pericoli--Taboga inspection feeds W2.6) -> W2 -> W3 ->
W4 -> W5.2--5.3 -> W6 -> W7.

Real blockers (stop and ask): a workspace fact contradicting the verified
inputs above; a rendering failure not fixable by escaping/markup; discovery
that a companion ledger makes claims the rewrite would falsify in a way not
covered by this plan. Everything else (wording choices, table layout,
equation-number collisions) is resolved in place and recorded in the release
note.

## Non-claims

This rewrite does not implement any sampler, does not run C0-vs-C1, does not
select a BGS solution engine, and does not establish posterior correctness,
HMC convergence, or production readiness. It preserves the survey's existing
non-claims discipline and the repository governance rules (TF/TFP backend,
GPU-default, tuning-scope, chunk-policy rules are untouched by this
documentation task).

## Execution record (2026-08-20)

All work packages executed without a blocker; no stop was required.

- **W1 done.** dsge_hmc identity corrected in the survey (header; Sec. 11
  layer assignment; Sec. 12.3 trailing sentence; Sec. 14 correction block;
  Sec. 16 Phase 5; Sec. 17 conclusion), `README.md`, `project_roadmap.md`,
  `experiment_plan.md`, `source_support.md`; dated correction appendices
  added to `hostile_review.md` and `survey_enhancement_plan_20260819.md`.
  Package-wide grep confirms no uncorrected absence assertion remains.
- **W5.1 done first, as sequenced.** Pericoli--Taboga inspected from the
  recovered local PDF (2018 Banca d'Italia WP 1189; blockwise random-walk
  Metropolis data augmentation, no moment-closure filter, sub-basis-point NN
  pricing surrogate; anchors Secs. 2, 4--5, 9). Note: the recovered artifact
  is the 2018 working paper, not the reported later journal version; cited
  accordingly.
- **W2 done.** Target-ladder table inserted (S1/C1/C0, with the jump-variant
  re-derivation trigger); Sec. 13.4 rewritten (bootstrap authority eq. (82a);
  (84)--(87) demoted to ideal fully adapted proposal identity with the three
  caveats); crossing-lemma exceptions (identically-zero curve; 6-dim joint
  cell mass); Sec. 13.5 signed-difference identity (90a) and O(sqrt(alpha))
  qualification; Sec. 13.7 rewritten as a five-route table with target IDs;
  Sec. 13.2 frontier claim narrowed; terminology renamed to hard-bound
  observation-map.
- **W3 done.** Kernel (4a) + measure/initialization/coordinate coherence in
  Secs. 2 and 4.3.6; Sec. 5.1 scoped to piecewise-regular partitions;
  general-M event maps (60a)--(60c) (re-derived during execution) with
  Sec. 12.1 pointer; PM-HMC four assumptions + correlated-PM comparator in
  Sec. 10.2; tail-stable Mills requirement in Sec. 4.3.2; parametric-LCP
  regularity in Sec. 14.1.
- **W4 done.** Sec. 14 split: 14.1--14.4 labelled pedagogical; 14.5 scoped to
  the OBC/ZLB target; new 14.6 BGS layer with the six-item source
  documentation list and the verified anchors.
- **W5.2--5.3 done.** Citations added (Cuba-Borda et al. 2019, Holden--Paetz
  2012, Genz--Bretz 2009, Botev 2017, Deligiannidis--Doucet--Pitt 2018, all
  labelled metadata-cited; Pericoli--Taboga 2018 inspected). 51 references in
  the manuscript and `references.bib`. `omitted_papers.md` rows updated plus
  a release-revision additions table.
- **W6 done, as amended.** No renumbering; letter-suffix insertions only.
  One collision found in review and fixed: the bootstrap equation initially
  tagged (83a) preceded (83) in document order and was renamed (82a)
  everywhere. Scripted audit: 109 tags, unique, monotone; ledger references
  resolve. Both derivation-check scripts re-run (tf-gpu env; system python
  lacks NumPy): ALL CHECKS PASSED, logs refreshed. HTML and 34-page PDF
  regenerated; pdftotext smoke checks pass. `claim_support.md` rows 41/43/44/
  45/47 updated where the rewrite changed their status.
- **W7 done.** Release note appended to `hostile_review.md`; README reader
  path and scope updated; acceptance-checklist greps recorded (all present).
  The bounded Claude review was retried as a read-only subagent run of
  `claude_review_bundle.md` against the revised manuscript: `VERDICT:
  REVISE`, one material in-scope finding (the isolated refraction map is not
  volume preserving; unit Jacobian holds for the reflection map and for the
  composed flight--event--flight step), verified and applied to survey
  Secs. 6.1 and 12.1, `claim_support.md` row 18, and roadmap WP3-D.
  MathDevMCP was **not** re-run (tool not reachable from this session) and is
  recorded as a gap in the release note; per the plan's stopping rules this
  is a gap statement, not a blocker.

Residual items for a publication-facing revision: MathDevMCP re-run scoped to
(4a), (60a)--(60c), (82a), (90a) and the rewritten Sec. 13 text; the
remaining `omitted_papers.md` expansion rows; journal-version metadata for
Pericoli--Taboga; human reading of the rendered manuscript. (The bounded
Claude review retry is complete; its one finding is applied.)
