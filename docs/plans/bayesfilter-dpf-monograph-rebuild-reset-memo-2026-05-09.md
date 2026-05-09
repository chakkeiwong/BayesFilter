# Reset memo: BayesFilter DPF monograph rebuild

## Date

2026-05-09

## Context

This reset memo is dedicated to the reader-facing differentiable particle filter
(DPF) monograph rebuild in BayesFilter.  It is intentionally separate from the
historical shared monograph reset memo and from the student experimental-baseline
workstream so that concurrent agents do not write into the same execution log.

The governing program for this work is:

- `docs/plans/bayesfilter-dpf-monograph-rebuild-master-program-2026-05-09.md`

The user rejected the earlier DPF chapter-writing pass as insufficiently
mathematical, insufficiently self-contained, and too governance-heavy.  The
corrected goal is to rebuild the DPF material as a mathematically rigorous,
literature-first monograph block that can stand beside the stronger CIP
monograph chapters.

## Scope of this reset memo

This memo tracks only:

- the DPF monograph rebuild planning and reader-facing chapter rewrite work;
- literature survey, architecture, and chapter-design phases;
- actual chapter rewrite execution passes;
- audits, tests, interpretations, and stop/go decisions for this workstream.

It does not track:

- the student DPF baseline experimental code path;
- unrelated analytic-derivative or SGU structural work;
- any other agent's execution stream.

## Execution policy

Each phase follows:

```text
plan for the phase -> execute -> test -> audit -> tidy -> update reset memo
```

Continue automatically only when the phase's primary criterion is met and no
veto diagnostic has fired.  If a veto diagnostic fires or a next phase is no
longer justified, stop and ask for direction.

## Program status at split time

### Program creation

Created the controlling plan set under `docs/plans/`:

- `bayesfilter-dpf-monograph-rebuild-master-program-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m0-preflight-plan-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m1-literature-survey-plan-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m2-architecture-plan-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m3-particle-flow-theory-plan-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m4-resampling-ot-plan-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m5-hmc-suitability-plan-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m6-drafting-execution-plan-2026-05-09.md`
- `bayesfilter-dpf-monograph-rebuild-phase-m7-integration-audit-plan-2026-05-09.md`

### Independent plan audit

Audit note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-plan-audit-2026-05-09.md`

Audit conclusion:
- proceed, with additions for:
  1. citation-key/source-identity audit;
  2. load-bearing theorem/equation obligation register.

These additions were folded into Phase M1.

## Phase M0: preflight and supersession audit

Phase plan:
- classify existing DPF artifacts by role;
- determine which remain governing inputs and which are only source/provenance or
  experimental comparison artifacts;
- explicitly mark the earlier DPF chapter block as a reader-facing draft to be
  replaced or heavily rewritten.

Execution:
- Reviewed the current DPF program note, the earlier Phase 1 audit plan, the
  earlier DPF plan review, the student baseline consolidation plan, and the new
  master rebuild program.
- Confirmed that the earlier DPF chapter-writing pass should not govern future
  reader-facing prose; it remains useful only for provenance, scope decisions,
  and failure history.
- Confirmed that the high-level mathematical ladder remains valid:
  EDH -> EDH/PF with importance correction -> differentiable/soft resampling ->
  OT resampling -> LEDH + neural OT.
- Confirmed that student materials remain critique/coverage inputs rather than
  monograph authorities.

Tests:
- checked current worktree state with `git status --short`;
- confirmed all new program files are in `docs/plans/` and no destructive action
  was required.

Audit:
- No blocker prevented proceeding to the literature survey.
- The main correction was made explicit: later phases must produce a
  mathematically rigorous monograph treatment, not governance-heavy prose.

Next phase justified?
- Yes, provided the literature phase records both citation-identity and
  theorem/equation obligation audits.

## Phase M1: literature survey and source-grounding audit

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m1-literature-survey-result-2026-05-09.md`

Execution summary:
- initialized ResearchAssistant workspace under `/tmp/ra-bayesfilter-dpf-rebuild`
  and ran `doctor` in offline mode;
- confirmed local student-source availability:
  - `2026MLCOE/full_report.pdf`, earlier part-1 report PDF, README;
  - `advanced_particle_filter` README, requirements, notebooks index,
    differentiable-resampling notebooks, amortized-OT architecture note;
- inspected the main CIP DPF/HMC chapters and related nonlinear-filtering
  context;
- recorded:
  1. citation-key/source-identity audit;
  2. method-family map;
  3. student-coverage comparison;
  4. theorem/equation obligation register.

Interpretation:
- the literature base is broad enough to support an architecture phase;
- the student sources add genuine mathematical and implementation coverage,
  especially on OT/amortized OT and DPF-HMC experiments;
- the key unresolved issue remains the HMC target question.

Audit:
- this phase satisfied the two additions required by the independent plan audit;
- no blocker prevented the next phase.

Next phase justified?
- Yes.  The literature map is now strong enough to support a deliberate
  mathematical chapter architecture.

## Phase M2: mathematical architecture and chapter-map design

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m2-architecture-result-2026-05-09.md`

Execution summary:
- concluded that the rebuilt DPF block should likely be organized as six
  mathematical units:
  1. particle-filter foundations;
  2. particle-flow foundations;
  3. PF-PF and proposal correction;
  4. differentiable resampling and OT;
  5. learned/amortized OT and implementation mathematics;
  6. HMC target correctness and structural-model suitability;
- recorded a dependency graph and replacement map for the current DPF draft
  artifacts;
- classified `ch19b/ch19c/ch19d` as drafts to retire as final exposition and
  mine only for provenance or structural ideas.

Interpretation:
- the topic is too rich for only three new chapters if treated rigorously;
- mathematical separation is now explicit enough to prevent the earlier collapse
  of survey, implementation, and HMC questions into the same prose block.

Audit:
- no blocker prevented the next phase.

Next phase justified?
- Yes.  The architecture is explicit enough to support a detailed mathematical
  plan for the particle-filter / particle-flow theory block.

## Phase M3: particle-filter and particle-flow theory plan

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m3-particle-flow-theory-result-2026-05-09.md`

Execution summary:
- split the theory block into at least:
  1. particle-filter foundations;
  2. particle-flow foundations;
  3. PF-PF / proposal-correction chapter-equivalent unit;
- recorded section-level mathematical goals and explicit load-bearing derivation
  obligations;
- made the comparison obligations among bootstrap PF, EDH, LEDH, and PF-PF
  explicit.

Interpretation:
- particle-filter likelihood estimation, flow approximation mathematics, and
  proposal-corrected PF-PF logic must be treated as distinct layers.

Audit:
- no blocker prevented the next phase.

Next phase justified?
- Yes.  The resampling and OT material can now be planned independently.

## Phase M4: differentiable resampling and OT plan

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m4-resampling-ot-result-2026-05-09.md`

Execution summary:
- recommended a dedicated chapter for differentiable resampling and OT
  foundations;
- recommended a further chapter-equivalent unit for learned/amortized OT and
  implementation-facing mathematical constraints;
- recorded the section map for soft resampling, transport/projection language,
  entropic OT, Sinkhorn structure, and bias analysis;
- recorded the comparison template across standard resampling, soft resampling,
  OT/EOT, and learned/amortized OT.

Interpretation:
- differentiability in the resampling step is purchased by either relaxation or
  approximation, and this must be stated mathematically in the final monograph.

Audit:
- no blocker prevented the next phase.

Next phase justified?
- Yes.  The HMC-target assessment can now be planned as its own problem.

## Phase M5: HMC target correctness and DSGE/MacroFinance suitability plan

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m5-hmc-suitability-result-2026-05-09.md`

Execution summary:
- defined the HMC-assessment chapter structure around:
  1. framing the HMC question;
  2. rung-by-rung target analysis;
  3. DSGE structural-model difficulties;
  4. MacroFinance structural-model difficulties;
  5. BayesFilter practical interpretation;
- recorded the rung-by-rung HMC assessment table and structural-model stress
  table;
- made explicit that HMC suitability attaches to a particular value-gradient
  construction, not to a method-family name.

Interpretation:
- exact targets, unbiased-estimator paths, relaxed targets, and learned
  surrogates must be separated explicitly in the final monograph.

Audit:
- no blocker prevented the next phase.

Next phase justified?
- Yes.  The program now has explicit source, architecture, theory, resampling,
  and HMC planning artifacts.

## Phase M6: drafting-execution and equation-audit protocol

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m6-drafting-execution-result-2026-05-09.md`

Execution summary:
- defined the per-chapter execution cycle:
  `source map -> derivation map -> draft -> tool audit -> revise -> compile -> claim audit`;
- recorded the per-chapter checklist, required audit artifacts, and tidy-up
  requirements;
- made explicit that the earlier failed DPF pass must be retired through a
  replacement map rather than silently polished.

Interpretation:
- for this topic, equations and claim ledgers must lead prose.

Audit:
- no blocker prevented the next phase.

Next phase justified?
- Yes.  The drafting protocol is explicit enough to support final integration
  planning.

## Phase M7: integration and final audit protocol

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m7-integration-audit-result-2026-05-09.md`

Execution summary:
- defined the integration checklist covering chapter order, `docs/main.tex`,
  bibliography integration, source-map integration, and cross-reference
  coherence;
- defined the final audit checklist covering undefined references/citations,
  duplicate labels, unsupported high-stakes claims, drift back toward repo
  commentary, and proper use of student-material discussion;
- recorded the completion condition for the rebuild.

Interpretation:
- the planning program is complete; what remains is the actual monograph rewrite.

Audit:
- no blocker prevented the first reader-facing rewrite phase.

Next phase justified?
- Yes.  The program is complete enough to support actual chapter rewriting.

## 2026-05-09 update: first reader-facing DPF rewrite execution pass started

### Phase R1: particle-filter foundations rewrite

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r1-particle-foundations-plan-2026-05-09.md`

Phase plan:
- replace the current short gateway chapter
  `docs/chapters/ch19_particle_filters.tex` with a mathematically serious
  particle-filter foundations chapter;
- cover the nonlinear state-space setup, empirical filtering measures,
  sequential importance sampling, sequential importance resampling, the
  bootstrap PF likelihood estimator, and degeneracy/ESS;
- keep HMC claims out of this first rewrite phase;
- compile, audit, tidy, and update this memo before any later flow or
  resampling chapter rewrite proceeds.

Primary criterion:
- the rewritten chapter must be mathematically self-contained and materially
  stronger than the old gateway version.

Veto diagnostics:
- if the chapter still reads mainly as program commentary rather than a
  derivation-based monograph chapter, do not proceed to the next rewrite phase;
- if the likelihood-estimator status cannot be stated with clear source support,
  stop and record the blocker;
- if the rewrite introduces unresolved notation or citation failures, fix them
  before declaring the phase complete.

Execution status:
- chapter rewritten substantially in `docs/chapters/ch19_particle_filters.tex`;
- compile/audit still active because bibliography support for the wider DPF block
  is incomplete and the phase has not yet cleared its veto diagnostics.

### Phase R1 result

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r1-particle-foundations-result-2026-05-10.md`

Execution:
- Rewrote `docs/chapters/ch19_particle_filters.tex` substantially.
- The chapter now includes:
  1. nonlinear state-space model and exact filtering recursion;
  2. predictive and filtering laws in BayesFilter notation;
  3. marginal-likelihood factorization;
  4. empirical filtering measures and test-function approximation;
  5. sequential importance sampling with explicit trajectory/proposal factoring;
  6. sequential importance resampling and the bootstrap proposal choice;
  7. the bootstrap PF likelihood estimator;
  8. a proposition-level statement of its status as an unbiased estimator of the
     marginal likelihood under standard assumptions;
  9. ESS and degeneracy discussion as the mathematical baseline for later DPF
     chapters;
  10. an explicit chapter-boundary section stating what is not yet claimed.
- Added the missing DPF bibliography keys needed by the rewritten chapter and
  the wider DPF block:
  `daumhuang2008`, `li2017particle`, `hu2021particle`,
  `zhumurphyjonschkowski2020`, `corenflos2021differentiable`,
  `jonschkowski2018differentiable`, and `karkus2018particle`.
- Tidied the trailing blank-line issue in `docs/references.bib`.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed and
  reported all targets up to date after the bibliography/key cleanup.
- `git diff --check` passed after trimming the trailing blank line in
  `docs/references.bib`.
- The wording scan found `unbiased` and `HMC-ready` in the chapter; these were
  audited and judged acceptable because:
  - `unbiased` appears only in the scoped estimator-status statement;
  - `HMC-ready` appears only in a negative boundary statement saying the chapter
    does not yet claim such a backend.

Interpretation:
- The particle-filter baseline chapter now functions as a mathematically serious
  opening to the rebuilt DPF block.
- The key conceptual improvement is that later DPF constructions can now be
  compared against a clearly defined exact filtering recursion and a clearly
  identified classical Monte Carlo estimator, rather than against a vague verbal
  baseline.

Audit:
- Primary criterion: satisfied.
- Commentary-drift veto: cleared for this chapter.
- Likelihood-status-support veto: cleared for the bootstrap PF statement.
- Notation/citation veto: cleared for this phase after bibliography cleanup.

Next phase justified?
- Yes.  The next rewrite phase is justified because the particle-filter baseline
  is now strong enough to support a mathematically serious particle-flow
  foundations rewrite.
- Recommended next phase: rewrite the particle-flow foundations chapter
  (EDH/LEDH/homotopy/continuity-equation structure), while keeping PF-PF
  proposal correction separate unless the derivation proves that one combined
  chapter remains readable.
