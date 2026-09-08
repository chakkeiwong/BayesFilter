# Hostile Scholarly Review

Review date: 2026-08-18. Review scope: the survey, project roadmap, six
literature ledgers, local source corpus, rendered PDF/HTML, and MathDevMCP
reports. This is a bounded local review; it is not a human publication review.

## Findings

### 1. Deterministic-regime support error

The first draft recommended a regime-only mixed-HMC move for a regime that could
be a deterministic threshold. Equation (5a) now makes the counting-measure
support explicit and the project recommendation is split into deterministic and
stochastic targets. This finding is closed in the current manuscript.

### 2. OBC solution and uniqueness coverage

Holden's existence/uniqueness work, Holden/Paetz solution methods, Boehl, and the
Cuba-Borda likelihood line are not fully audited. They are named in
`omitted_papers.md` and `forward_snowball.md`. No section claims global
uniqueness, and Phase 0 blocks implementation until a selection law is written.
This remains an expansion requirement before application claims.

### 3. Recent differentiable-filter coverage

The bounded search fully checks Ścibior--Wood and Corenflos et al. but does not
claim exhaustive 2022--2026 coverage. Newer differentiable resampling, score,
and particle-gradient papers remain an omission risk. The survey labels those
methods as comparison arms and does not promote them to exact HMC forces.

### 4. Metadata and venue evidence

OpenAlex counts and retraction flags are dated in `citation_venue_metadata.md`;
duplicate preprint/proceedings records are not summed. Venue rankings were not
available and are not presented as zeros. ResearchAssistant's provider bootstrap
blocker is preserved. These are adequate for coverage triage, not for claims of
impact or completeness.

### 5. Mathematical and rendering checks

- Pandoc 2.9 rendered HTML and a 13-page PDF successfully using the
  `tex_math_single_backslash` extension and `pdflatex`.
- `pdftotext` inspection found the title, equations, citations, roadmap, limits,
  and complete reference list in the rendered output.
- `git diff --check` reported no whitespace errors.
- MathDevMCP `audit-math-document-rigor` selected **zero labelled equations**
  and therefore reported zero scoped semantic issues; its report explicitly
  says this is not a proof certificate. This is a scope/coverage fact, not a
  substantive certification of every derivation in the manuscript.
- MathDevMCP `audit-applied-math-document --mode deep` completed with limits and
  warned that no code paths were supplied, so implementation alignment was not
  checked.
- A one-path Claude review was attempted using
  `claude_review_bundle.md`; the governed gate returned
  `REVIEW_STATUS=probe_timeout` and `VERDICT=NONE`, so no Claude agreement is
  claimed. The bundle and structured run record are preserved for a later
  retry.

### 6. Section 4.3 recovery and UKF-family sources (19 August 2026)

The piecewise/truncated/mixture UKF section was added on 19 August 2026 after
the writing session was interrupted by a provider rate-limit error. Recovery
facts:

- The interrupted write had lost inline-math delimiters throughout Section 4.3;
  every affected span was repaired, and the manuscript's equations were
  renumbered into one sequence (1)--(74), with cross-references updated in the
  manuscript, `claim_support.md`, and `omitted_papers.md`.
- Four of the nine newly cited works were inspected as local full text
  (Kandepu et al. 2008; Teixeira et al. 2010; Zhang et al. 2020; Wang et al.
  2020). The other five (both García-Fernández et al. 2012 papers,
  Gokce--Kuzuoglu 2015, Alspach--Sorenson 1972, Blom--Bar-Shalom 1988) are
  pay-walled: they are cited from verified OpenAlex/Crossref metadata as family
  orientation only, the manuscript says so explicitly, and no equation-level
  content is imported from them.
- The section's closed forms (truncated branch moments, censored predictive
  density, mixture collapse, IMM mixing) were checked numerically by
  `derivation_check_ukf_section_20260819.py`; all checks passed and the log is
  preserved. This is a diagnostic spot-check, not a proof certificate.
- MathDevMCP was not re-run after the addition, so its earlier zero-scope
  statement does not cover Section 4.3. The adaptive Gaussian-splitting
  recommendation for nonlinear boundaries rests on project reasoning without an
  inspected primary source; this is registered in `omitted_papers.md`.

### 7. Application-contract expansion (19 August 2026, second pass)

The owner-requested enhancement plan was executed: Sections 13 (MacroFinance
shadow-rate contract), 14 (dsge_hmc contract), 15 (solver-induced
discontinuities), and 5.1 (kinked-HMC validity) were added, with twenty new
sources and equations (56a) and (75)--(98). A hostile reading records:

- **Inspection basis.** The MacroFinance contract is derived from five
  inspected code modules plus the `dz5` campaign files; `dsge_hmc` still does
  not exist, so Section 14 is a forward contract. Sixteen of the twenty new
  sources were inspected as local full text with recorded anchors; Black,
  Allik et al., Pakman--Paninski, and Aruoba--Cuba-Borda--Schorfheide are
  metadata-cited only, and the manuscript's claims are scoped accordingly.
- **Geometry reclassification risk.** Section 13.3 classifies the coded
  MacroFinance target as a kink target. This contradicts a casual reading of
  the project brief ("genuine discontinuous regime") and is deliberately
  stated as a property of the code as of 2026-08-19, with the three
  extensions that would restore genuine jumps named explicitly. If the owner
  intends one of those extensions, Section 13.3's classification must be
  re-derived, not quoted.
- **Derivation checks.** The crossing lemma, the cell-evidence identity, the
  softplus bounds, and the lagged-rule multiplicity example are spot-checked
  numerically in `derivation_check_contract_sections_20260819.py` (log
  preserved; all pass, with the crossing bound and both softplus bounds
  attained). The Section 5.1 validity claim and the general LCP continuity
  discussion are argued, not machine-verified.
- **Known remaining gaps.** The inversion-filter/Cuba-Borda line, ACS model
  internals, post-2015 shadow-rate refinements, orthant/truncated-sampling
  computational references, and general-boundary event-HMC theory remain
  unaudited (see `omitted_papers.md`). Kim--Priebsch (2013) has no indexed
  metadata record. The van der Merwe author-order discrepancy between the
  inspected copy and the proceedings index is recorded in the source ledger.
- **MathDevMCP re-run.** The rigor audit was re-run on a labeled-equation
  LaTeX export, focused on the 25 new equations (56a, 75--98):
  four records, of which one actionable proposal --- state invertibility of
  the eq. (85) inverse operand --- was applied to the manuscript; the other
  three are formalization-status diagnostics on model-definition equations.
  Report archived as `mathdevmcp_audit_20260819.md` and
  `mathdevmcp_applied_audit/rigor_audit_v2_20260819.json`. The tool's own
  boundary statement applies: this is a scoped exposition check, not a proof
  certificate. The bounded Claude review was not retried.

## Decision

`bounded_local_survey_with_explicit_gaps`.

The package is suitable for internal project scoping and source-grounded
implementation planning. It is not a claim of literature completeness,
publication acceptance, posterior correctness, HMC convergence, or production
readiness. A later publication-facing revision needs human reading of the
rendered manuscript and expansion of the OBC uniqueness and recent-gradient
literatures.

## Correction appendix (19 August 2026, release revision)

The statement in Finding 7 that "`dsge_hmc` still does not exist" is false as
a workspace statement and is corrected here without altering the historical
text above. The originally named path `/home/ubuntu/workspace/dsge_hmc` is absent, but
the package exists at `/home/ubuntu/workspace/python/src/dsge_hmc`. Its validated BGS
restricted surface is a no-binding linear placeholder (`r = rn`;
`models/bgs_restricted_surface_generated.py:225,228`; rows tagged
`OBC_ZLB_NO_RUN_GUARD` in `models/bgs_restricted_surface_tf_coefficients.py:17`),
and the package's master program excludes OBC/ZLB estimation from its
evidence contract. Survey Section 14 was mis-framed as a greenfield forward
contract and has been split into a pedagogical NK/LCP layer and a
source-anchored BGS layer (Secs. 14.5--14.6) in the release revision. The
full defect list and its verification record are in
`zlb_discontinuous_hmc_survey_audit_20260819.md` and the release rewrite plan
`docs/plans/zlb-discontinuous-hmc-survey-release-rewrite-plan-2026-08-19.md`.

## Release note (20 August 2026, v1.1 internal release revision)

Executed under
`docs/plans/zlb-discontinuous-hmc-survey-release-rewrite-plan-2026-08-19.md`,
implementing `zlb_discontinuous_hmc_survey_audit_20260819.md`. All P0 and P1
audit findings were applied; the audit's factual claims were independently
re-verified against the workspace before execution (dsge_hmc anchors at the
exact claimed lines; MacroFinance softplus/alpha/quadrature/transition facts).

**Corrections applied to the manuscript.**

- P0-1: dsge_hmc identity corrected in the header, Secs. 11, 12.3, 14
  (retitled with a dated correction block, pedagogical/BGS layers split, new
  Sec. 14.6 with verified source anchors), 16 Phase 5, and 17; also in
  `README.md`, `project_roadmap.md`, `experiment_plan.md`,
  `source_support.md`, and via dated appendices here and in the enhancement
  plan.
- P0-2: eqs. (84)--(87) renamed an ideal fully adapted proposal identity with
  the three implementability caveats (numerical polytope masses; a finite
  Pakman--Paninski trajectory is not an exact draw; pruning breaks
  unbiasedness). New eq. (82a) bootstrap PF is the C1 likelihood authority.
- P0-3: target ladder `mf_s1_k40_softplus` / `mf_c1_k40_hardmax` /
  `mf_c0_root_integral_hardmax` inserted before Sec. 13.3; C0-vs-C1
  sensitivity experiment specified (not run); route table extended to five
  routes with per-target claims.
- P0-4: Markov transition kernel (4a), declared dominating measures, branch
  coordinates for (5a), the x_0/x_1 initialization conventions, and the (53)
  shock-chart/Jacobian obligation added.
- P0-5: general-mass-matrix event maps (60a)--(60c) added with orientation
  and grazing conventions; Sec. 12.1 now points at them.
- P0-6: Alenlov--Doucet--Lindsten's four PM-HMC assumptions stated; bootstrap
  PF supports PMMH, not PM-HMC; correlated pseudo-marginal cited as a
  comparator, not a differentiability fix.
- P1-1 through P1-7: tail-stable Mills ratios; identically-zero root case and
  the six-dimensional joint cell probability; signed likelihood-difference
  identity (90a) replacing the one-signed accumulation claim; O(sqrt(alpha))
  qualified as a local heuristic; parametric-LCP regularity conditions;
  Sec. 5.1 scoped to piecewise-regular partitions excluding implicit
  multi-root solvers; censored-measurement terminology renamed hard-bound
  observation-map throughout, reserving "censored" for the Tobit ordering.
- Literature: Pericoli--Taboga (2018 WP 1189) inspected as local full text and
  the Sec. 13.2 "current frontier" claim narrowed (closure-free Bayesian
  estimation already exists; the UKF+HMC route may claim a gradient-sampler
  frontier within the closure-filter family only). Cuba-Borda et al. (2019),
  Holden--Paetz (2012), Genz--Bretz (2009), Botev (2017), and Deligiannidis--
  Doucet--Pitt (2018) added as labelled metadata citations. 51 references;
  `references.bib` updated in lockstep. `omitted_papers.md` and
  `claim_support.md` rows updated where the rewrite changed their status.

**Verification.**

- Both derivation-check scripts re-run on 2026-08-20 (tf-gpu env python):
  ALL CHECKS PASSED; logs refreshed in place.
- Equation-tag audit: 109 tags, all unique, monotone in document order
  (letter suffixes used for insertions; no renumbering of the historical
  sequence). Companion-ledger equation references resolve against the
  manuscript.
- HTML and 34-page PDF regenerated (pandoc 2.9.2.1,
  tex_math_single_backslash, pdflatex); pdftotext smoke checks pass (title,
  target ladder, new equations, Pericoli--Taboga reference present).
- The general-mass event maps (60b)--(60c) were re-derived during execution:
  energy conservation and the unit-mass reduction were checked algebraically.

**Bounded Claude review retried and applied (20 August 2026).** The
`claude_review_bundle.md` prompt was re-run as a bounded read-only subagent
review of the revised manuscript. It returned `VERDICT: REVISE` with one
material finding in scope, which was verified and applied: the survey (and
Afshar--Domke attribution) claimed unit absolute Jacobian for the *event
map*, but the isolated refraction map (59)/(60b) has absolute Jacobian
|s_-/s_+| != 1 for a nonzero potential jump; volume preservation attaches to
the composed flight--event--flight trajectory step (the reflection map alone
is unit-Jacobian). Sections 6.1 and 12.1, `claim_support.md` row 18, and the
roadmap WP3-D test list were corrected so that correctness tests target the
composed step or account for the event Jacobian explicitly. The review found
the deterministic-versus-stochastic support distinction and the resulting
kernel recommendations otherwise coherent.

**Not re-run.** MathDevMCP was not re-run in this revision (tool not
reachable from the execution session). The 19 August MathDevMCP rigor re-run
covered eqs. (56a), (75)--(98) as then numbered; it does not cover the
release revision's new equations (4a), (60a)--(60c), (82a), (90a) or the
rewritten Secs. 13.3--13.7 text. This is a recorded gap, not a
certification.

**Decision unchanged:** `bounded_local_survey_with_explicit_gaps`, now at
release revision v1.1. Publication-facing release still requires human
reading of the rendered manuscript, the remaining `omitted_papers.md`
expansion rows, and a MathDevMCP re-run scoped to the revised equations. The
bounded Claude review retry is complete (one finding, applied, above).

## Rendering addendum (20 August 2026, revised)

The human-facing primary reading artifact is now the standalone LaTeX source
`zlb_discontinuous_hmc_survey.tex`, built from the canonical Markdown by
`build_survey_tex.py` (in this directory) and compiled with pdflatex (two
passes) to the 44-page `zlb_discontinuous_hmc_survey.pdf`. The build script
fixes the defects of a naive pandoc conversion: literal heading numbers are
stripped so LaTeX's own section numbering reproduces the survey's 1--17
scheme exactly (a naive pandoc pass produced double numbering, 0.1/0.2 TOC
labels); markdown pipe tables are converted to booktabs `tabularx` tables
with inline math intact and long `target_id` code names breakable at
underscores; `hypertarget` wrappers are dropped. Display math passes through
pandoc's `tex_math_single_backslash` unchanged, so every equation renders as
LaTeX math with its original `\tag{}` numbering. Two display equations ((90)
and (96)) were reflowed in the Markdown for line width; no mathematical
content changed. The compile has zero overfull boxes above 20pt; visual
page inspection confirmed the title/TOC page, a math-dense contract page
(Sec. 13.2--13.3 with eq. (80) and the target-ladder table), and the
five-route table all render correctly. The Markdown remains the canonical
content source; the `.tex` is deterministic build output --- regenerate it
with `python build_survey_tex.py && pdflatex` (twice), do not hand-edit.

## Rendering addendum II (20 August 2026): human-facing LaTeX manuscript

The owner rejected the pandoc-converted LaTeX as unreadable: it preserved the
internal governance register (file paths, inspection dates, "claim-bearing"/
"evidence contract" vocabulary, code identifiers) and did not teach. The
LaTeX manuscript `zlb_discontinuous_hmc_survey.tex` is now a full prose
rewrite for a human reader, produced under a written style contract:

- audience is a graduate reader with no codebase knowledge; every section
  opens with motivation and derivations carry their connecting steps;
- all file paths, project/repository names, fixture names, inspection dates,
  and governance vocabulary removed; the two applications are self-contained
  case studies (a two-country shadow-rate term-structure model; a New
  Keynesian model with an occasionally binding policy rate) whose models are
  specified from scratch in the text;
- the three target identifiers appear as $\mathcal{S}_1/\mathcal{C}_1/
  \mathcal{C}_0$, mapping one-to-one to `mf_s1_k40_softplus` /
  `mf_c1_k40_hardmax` / `mf_c0_root_integral_hardmax` in the internal
  ledgers; equation labels reuse the Markdown's tag names (`eq:82a` etc.) so
  the ledger anchors remain translatable;
- ALL mathematics is retained, including the release-revision corrections
  (target ladder, ideal-versus-bootstrap positioning, general-mass event
  maps with the composed-step volume statement, PM-HMC assumptions, signed
  softplus-likelihood identity, LCP parametric regularity);
- dropped as internal-only: the dsge_hmc package-status subsection (old
  Sec. 14.6), phase/work-package numbering, GPU/memory rules, and the
  correction-history blocks. The internal Markdown remains the complete
  record of those; the audit ledgers continue to anchor to the Markdown.

Verification: 46-page pdflatex compile with zero errors, zero unresolved
references, zero overfull boxes above 20pt; rendered-page inspection of the
title/TOC, the censored-likelihood derivation (old Sec. 4.3.3), and both
case-study openings; a leak sweep over the rendered text for file paths,
project names, dates, and governance vocabulary found zero occurrences.
The manuscript was written section-by-section against the style contract
(preserved at the execution scratchpad) with per-section drop reports; the
prose is new, so a human read of the full PDF remains the outstanding
publication gate, now with a document actually written for that reader.

## Rendering addendum III (20 August 2026): editorial pass under written policy

The owner asked whether the manuscript rewrite had followed
`claudecodex/policies`; it had not been consulted, which was a process
failure. The "Reader-Facing Scientific Prose" section of
`global-scientific-coding-agent-policy.md` was then applied, together with
the newly installed `humanizer-ai-writing-patterns.md` (from
github.com/blader/humanizer, MIT, installed with a precedence header making
it a diagnostic subordinate to the global policy).

Measured defects and repairs on the LaTeX manuscript: 133 body em dashes
reduced to 0 (en dashes in ranges and names untouched); section-opening
monotony (20/55 sections opening "The ...", one uniform
motivation-math-forward-link shape) diversified so no opener word exceeds
4/55; document-narrating transitions ("we turn next to...") removed in
favor of subject-driven transitions; a repeated hedging connector reduced
from 9 to 3 occurrences; the one stock word removed.

Policy-required baseline comparison after the pass: 109/109 equation bodies
byte-identical, 166/166 labels, 183/183 eqrefs, 55/55 sections, 51/51
reference items preserved; three-pass pdflatex compile with zero errors and
zero unresolved references; 46 pages.

Per the same policy, author or model self-review cannot certify a human
voice: human reading of the rendered PDF remains the acceptance gate, and
this revision is provisional until that reading happens.

## Rendering addendum IV (20 August 2026): defensive-register repair

Owner feedback on the previous revision: the prose was still over-defensive,
"making a nonclaim instead of explaining"; rigor should teach rather than
avoid liability. A diagnostic pass found ~65 sentences of defensive register
(negative definitions, prohibitions without mechanisms, a moralizing
"honest"/"silently" tic, two defensive section headings).

Repair, per the governing policy's "Rigor In Natural Prose" clauses (state
the choice positively; positive result first; qualify only where a stronger
wrong inference is plausible): negative definitions were rewritten so the
mechanism carries the boundary (e.g. the fixed-branch HMC limitation now
derives from the zero-density proposal under eq. (5a) instead of being
asserted); prohibitions became consequences (the P-matrix/continuity caution
now explains that pointwise uniqueness and between-point continuity are
different questions before stating the parametric condition); "honest" (7)
and "silently" (4) were replaced by the substantive property (unbiased,
explicit, quantified); the headings "Why this is not a nonlinear ZLB filter"
and "...why it is not implementable, and the bootstrap filter as the honest
workhorse" became "Exactness conditional on a linearization" and "An exact
cell identity and the bootstrap filter that carries the likelihood".
Remaining negative statements are substantive mathematics (theorem
conditions, the complementarity-form scope remark, the LCP continuity
distinction) and were kept.

Baseline comparison after the pass: 109/109 equation bodies byte-identical;
166 labels unchanged; zero body em dashes; three-pass pdflatex compile with
zero errors and zero unresolved references; 46 pages. The editing agent was
interrupted once by API capacity errors; its edits were verified complete
against the diagnostic list before installation. Human reading remains the
acceptance gate.

## Rendering addendum V (21 August 2026): standard test-model ladder

Owner feedback: the manuscript prescribed validation procedures but never
named the reference problems. A "ladder of standard test models" subsection
and table (manuscript Sec. 16 / Table 3; Markdown Phase 1a) were added,
organized by the source of the independent answer: closed forms
(linear-Gaussian Kalman; the survey's own censored-scalar model (29)--(35);
truncated-Gaussian moments (21)--(28); Pakman--Paninski/Botev reference
draws), exact enumeration (two-state switching LGSSM with 2^T paths;
Gordon--Salmond--Smith growth model with a grid-filter reference),
ZLB-specific anchors (Gorovoi--Linetsky closed-form shadow-rate bond
prices; a one-node K=1 reduction of the C1 case study that collapses to the
Tier-1 scalar model; the lagged-rule economy's analytic multiplicity
boundary; published OccBin and Aruoba et al. references), and
model-agnostic harnesses (Geweke joint-distribution tests; simulation-based
calibration). Four metadata-cited references added
(Gordon--Salmond--Smith 1993; Gorovoi--Linetsky 2004; Geweke 2004; Talts et
al. 2018), bringing the list to 55; references.bib updated in lockstep.
Three-pass pdflatex compile clean (49 pages, zero unresolved references,
zero significant overfull boxes); table page visually inspected. The
Gorovoi--Linetsky item also upgrades the omitted-papers register's
"acceptable" row for that lineage to an actively cited test anchor.
