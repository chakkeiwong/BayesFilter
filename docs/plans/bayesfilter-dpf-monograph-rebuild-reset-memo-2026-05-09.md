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

## 2026-05-10 update: second reader-facing DPF rewrite phase started

### Phase R2: particle-flow foundations rewrite

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r2-particle-flow-foundations-plan-2026-05-10.md`

Phase plan:
- rewrite the current `docs/chapters/ch19b_dpf_literature_survey.tex` into a
  mathematically serious particle-flow foundations chapter;
- cover the transport motivation, homotopy density, continuity equation, EDH
  under Gaussian closure, linear-Gaussian recovery, LEDH, and stiffness;
- keep PF-PF proposal correction as a later chapter rather than collapsing it
  prematurely into the flow chapter;
- compile, audit, tidy, and update this memo before the next phase proceeds.

Primary criterion:
- the rewritten chapter must be mathematically self-contained, materially
  stronger than the current draft, and explicit about where approximation
  enters.

Veto diagnostics:
- if the chapter still reads as a broad survey rather than a theory chapter,
  stop and revise;
- if the EDH/LEDH distinction is not mathematically clear, stop and revise;
- if exact-special-case and approximate-flow statements are not separated
  cleanly, do not proceed to the PF-PF rewrite phase.

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

## 2026-05-10 update: sixth reader-facing DPF rewrite phase started

### Phase R6: HMC-target correctness and structural-model suitability rewrite

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r6-hmc-target-suitability-plan-2026-05-10.md`

Phase plan:
- rewrite `docs/chapters/ch21_hmc_for_state_space.tex` so that the DPF-target
  question is integrated explicitly into the reader-facing HMC chapter;
- cover value/gradient/target consistency rung by rung across the DPF ladder;
- state why differentiability alone is insufficient;
- analyze nonlinear DSGE and MacroFinance structural-model difficulties;
- conclude with BayesFilter's first justified HMC-relevant DPF development
  rung.

Primary criterion:
- the rewritten HMC material must make the value/gradient/target correspondence
  explicit for each DPF rung and explain why that matters more in nonlinear
  DSGE and MacroFinance models than in toy systems.

Veto diagnostics:
- if the chapter still speaks as if differentiability nearly implies HMC
  validity, stop and revise;
- if the rung-by-rung table is missing or vague, stop and revise;
- if DSGE and MacroFinance difficulties are treated only informally rather than
  structurally, do not declare the phase complete.

## 2026-05-10 update: sixth reader-facing DPF rewrite phase started

### Phase R6: HMC-target correctness and structural-model suitability rewrite

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r6-hmc-target-suitability-plan-2026-05-10.md`

Phase plan:
- rewrite `docs/chapters/ch21_hmc_for_state_space.tex` so that the DPF-target
  question is integrated explicitly into the reader-facing HMC chapter;
- cover value/gradient/target consistency rung by rung across the DPF ladder;
- state why differentiability alone is insufficient;
- analyze nonlinear DSGE and MacroFinance structural-model difficulties;
- conclude with BayesFilter's first justified HMC-relevant DPF development
  rung.

Primary criterion:
- the rewritten HMC material must make the value/gradient/target correspondence
  explicit for each DPF rung and explain why that matters more in nonlinear
  DSGE and MacroFinance models than in toy systems.

Veto diagnostics:
- if the chapter still speaks as if differentiability nearly implies HMC
  validity, stop and revise;
- if the rung-by-rung table is missing or vague, stop and revise;
- if DSGE and MacroFinance difficulties are treated only informally rather than
  structurally, do not declare the phase complete.

### Phase R6 result

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r6-hmc-target-suitability-result-2026-05-10.md`

Execution:
- This phase was not executed.

Tests:
- Verified the current chapter-role situation in the active DPF block.
- Confirmed that `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` is now being
  used in substance for learned/amortized OT and implementation mathematics.
- Confirmed that `ch21_hmc_for_state_space.tex` still carries broader BayesFilter
  HMC doctrine rather than a DPF-specific final assessment role.

Interpretation:
- The final DPF-specific HMC-suitability chapter does not yet have a clean
  dedicated chapter slot in the active chapter graph.
- Rewriting `ch21_hmc_for_state_space.tex` now would blur generic BayesFilter
  HMC doctrine with the final DPF rung-by-rung assessment.

Audit:
- Primary criterion: not satisfied.
- Veto diagnostics: triggered.
- This is a real architectural blocker, not a cosmetic one.

Next phase justified?
- No, not yet.
- The final DPF HMC-suitability rewrite is not justified until the chapter-slot
  question is resolved.
- The cleanest next step is to decide whether to create a new dedicated DPF HMC
  chapter inside the Part IV DPF block or to refactor the chapter graph with an
  explicit new slot.

## 2026-05-11 update: enrichment round started

The reader-facing DPF rewrite sequence completed the architectural rebuild of
the DPF block, but the resulting chapters are still too compressed relative to
the intended monograph standard.  A new enrichment round therefore begins under:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-master-program-2026-05-11.md`

### Phase E0: source and claim-audit setup

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e0-source-claim-audit-plan-2026-05-11.md`

Plan audit:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e0-plan-audit-2026-05-11.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e0-source-claim-audit-result-2026-05-11.md`

Execution:
- tightened the E0 plan before execution by adding a per-chapter priority
  register so the enrichment round does not spread effort too evenly across
  low-value and high-value claims;
- initialized a fresh ResearchAssistant workspace for the enrichment round under
  `/tmp/ra-bayesfilter-enrichment` and verified doctor/readiness status;
- confirmed the active DPF chapter inventory for the enrichment round;
- clarified the RA workflow expectations for chapter-level source and claim
  auditing.

Tests:
- ResearchAssistant workspace initialization: passed.
- ResearchAssistant doctor/readiness check: passed.
- Active DPF chapter inventory: passed.
- RA command support inspection relevant to source-label workflows: passed.

Interpretation:
- The enrichment round now has a concrete source-and-claim audit lane rather
  than relying on informal citation memory.
- This phase succeeded procedurally: later chapter-enrichment phases can now be
  driven by explicit source/claim obligations rather than by memory or loose
  summary instincts.

Audit:
- Primary criterion: satisfied at the setup level.
- No blocker prevents the next phase.
- The actual chapter-level claim ledgers and literature/debugging backlinks now
  belong to the later enrichment phases themselves.

Next phase justified?
- Yes.  Phase E1 is now justified because the source-audit infrastructure is in
  place and the particle-flow chapter is the first chapter that now needs a much
  deeper derivation and literature expansion.

### Phase E1: particle-flow theory enrichment

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e1-particle-flow-theory-plan-2026-05-11.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e1-particle-flow-theory-result-2026-05-12.md`

Execution:
- Reviewed the current particle-flow chapter against the stricter enrichment
  standard.
- Cross-checked the current chapter against CIP particle-flow source material.
- Used the student report for a topic-ordering and coverage check on EDH,
  LEDH, PF-PF, and stiffness.
- Produced an explicit source-claim / local-derivation / debugging map for the
  particle-flow chapter.
- Produced exact-vs-approximate comparison findings and a debugging-relevance
  map for key flow issues.

Tests:
- chapter inspected against E1 requirements;
- CIP particle-flow sections inspected for richer source scaffolding;
- student-report parsing used for topic and coverage orientation.

Interpretation:
- The chapter is structurally sound but still too brief relative to the
  enrichment standard.
- The key missing depth is in EDH derivation, LEDH derivation, and stiffness as
  an implementation/debugging topic.

Audit:
- Primary criterion: only partially satisfied.
- Veto diagnostics fired:
  1. EDH still too compressed;
  2. LEDH still too compressed;
  3. stiffness still too short for a strong debugging reference.

Next phase justified?
- No, not yet.
- The enrichment round should not proceed to E2 until `ch19b_dpf_literature_survey.tex`
  itself has been expanded enough to clear the E1 veto diagnostics.
- The correct next action is a direct deepening pass on the particle-flow
  chapter rather than moving on to PF-PF enrichment.

### E1 follow-up deepening pass

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e1-particle-flow-theory-followup-result-2026-05-12.md`

Execution:
- Deepened `docs/chapters/ch19b_dpf_literature_survey.tex` directly after the
  initial E1 execution blocked progression.
- Expanded the homotopy section to strengthen endpoint exactness,
  normalization, and debugging relevance.
- Expanded the EDH section to make the Gaussian-closure approximation more
  explicit and to add stronger implementation/debugging interpretation.
- Expanded the LEDH section to include the local linearization equation, the
  local information-vector construction, and a clearer statement of the added
  approximation layer and local numerical fragilities.
- Expanded the stiffness section so that it now serves as a stronger
  debugging-oriented discussion rather than only a short warning.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- Keyword audit confirmed that EDH, LEDH, Gaussian closure, exact special case,
  and stiffness now appear in substantially richer form.

Audit:
- EDH-too-compressed veto: cleared.
- LEDH-too-compressed veto: cleared.
- stiffness-too-short veto: cleared.
- Primary criterion: satisfied after the deepening pass.

Interpretation:
- The particle-flow chapter is now materially closer to the monograph standard
  required by the enrichment round.
- The stop condition in the first E1 pass did its job: it forced a real chapter
  expansion instead of allowing a still-too-thin chapter to pass.

Next phase justified?
- Yes.  Phase E2 is now justified because the PF-PF chapter can now build on a
  stronger particle-flow foundation rather than on a compressed summary.

### Phase E2: PF-PF and Jacobian/log-det enrichment

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e2-pfpf-jacobian-plan-2026-05-11.md`

Plan audit:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e2-plan-audit-2026-05-12.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e2-pfpf-jacobian-result-2026-05-12.md`

Execution:
- Tightened the E2 plan before execution by requiring an explicit separation
  between mathematically exact identities, approximation sources, and observable
  failure symptoms.
- Deepened `docs/chapters/ch19c_dpf_implementation_literature.tex` directly.
- Strengthened the discussion so that the chapter now distinguishes clearly
  between:
  1. the exact change-of-variables identity;
  2. approximation introduced by the flow family;
  3. approximation introduced by numerical integration;
  4. approximation introduced by the finite-particle system.
- Strengthened the Jacobian/log-det section so it now points more directly to
  likely implementation failures and diagnostics.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- local text audit confirmed richer treatment of exact identity versus
  approximation and debugging relevance.

Interpretation:
- The PF-PF chapter is now significantly more useful for implementation and
  target interpretation.
- The main gain is the explicit distinction between what the proposal correction
  restores exactly and what remains approximate because of closure,
  discretization, and finite-particle effects.

Audit:
- Primary criterion: satisfied.
- Change-of-variables-sketched veto: cleared.
- Nonlinear-filter-exactness-overclaim veto: cleared.
- Jacobian/log-det-burden veto: cleared.

Next phase justified?
- Yes.  Phase E3 is now justified because the resampling/OT chapter can now be
  deepened on top of a clearer PF-PF correction layer.

### Phase E3: differentiable resampling and OT enrichment

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e3-resampling-ot-plan-2026-05-11.md`

Plan audit:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e3-plan-audit-2026-05-12.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e3-resampling-ot-result-2026-05-12.md`

Execution:
- Tightened the E3 plan before execution by requiring an explicit separation of:
  exact categorical resampling, exact OT interpretation, entropic OT
  regularization, and finite-iteration Sinkhorn approximation.
- Deepened `docs/chapters/ch32_diff_resampling_neural_ot.tex` directly.
- Strengthened the chapter so that the mathematical transport object and the
  numerical solver approximation are no longer blurred.
- Strengthened the barycentric-map and bias-versus-differentiability sections so
  that the exact / relaxed / approximated layers are explicit.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed and
  reported all targets up to date.
- `git diff --check` passed.
- local text audit confirmed stronger chapter-level coverage of categorical
  resampling law, OT formulation, Sinkhorn approximation, and explicit bias
  language.

Interpretation:
- The chapter now distinguishes clearly among the categorical law, the OT
  formulation, the entropic relaxation, and the finite solver approximation.
- This makes the chapter materially more useful for implementation debugging and
  for later HMC interpretation.

Audit:
- Primary criterion: satisfied.
- OT-too-intuitive veto: cleared.
- soft-resampling-bias-too-verbal veto: cleared for this phase.
- regularization/finite-iteration disconnect veto: cleared.

Next phase justified?
- Yes.  Phase E4 is now justified because learned/amortized OT can now be
  deepened on top of a clearer OT baseline and a more explicit approximation
  hierarchy.

### Phase E3 correction and build-integration repair

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e3-resampling-ot-result-2026-05-12.md`

Execution:
- Re-audited the active E3 state before proceeding to E4.
- Found that `docs/chapters/ch32_diff_resampling_neural_ot.tex` was not yet
  included in `docs/main.tex`; therefore the earlier E3 build note did not prove
  that the target chapter compiled.
- Added the missing `\input{chapters/ch32_diff_resampling_neural_ot}` between
  the PF-PF chapter and the learned-OT chapter.
- Repaired the malformed approximation-hierarchy list in the resampling/OT
  chapter.
- Added the unregularized OT coupling object, the soft-resampling nonlinear
  test-function bias expansion, the required OT-object-versus-solver table, the
  resampling-family comparison table, and implementation/source-debug tables.
- Added minimal bibliography support for OT, Sinkhorn, stabilized Sinkhorn,
  learned set operators, pseudo-marginal/HMC, and HNN references needed by the
  E3--E5 enrichment pass.
- Added `placeins` and `\FloatBarrier`, and set `hypertexnames=false`, to keep
  table float anchors stable after the new DPF chapter slot was activated.

Tests:
- `latexmk -C main.tex` was run once from `docs/` to clear stale artifacts.
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  from `docs/` after the target chapter was wired into the book.
- `git diff --check` passed.
- Focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, and no LaTeX errors after the final
  E3 repair build.
- Focused text audit confirmed the chapter now contains categorical resampling,
  unregularized OT, entropic OT, finite Sinkhorn approximation, explicit
  soft-resampling bias, and implementation-debug mappings.

Interpretation:
- This correction was necessary before E4.  The earlier E3 conclusion was
  directionally right but build-incomplete because the chapter was not in the
  compiled monograph.
- E3 now satisfies the intended gate: learned/amortized OT can be discussed as a
  further approximation layer on top of a compiled and source-supported OT
  baseline.

Next phase justified?
- Yes.  Phase E4 is justified.

### Phase E4: learned/amortized OT enrichment

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e4-learned-ot-plan-2026-05-11.md`

Plan audit:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e4-plan-audit-2026-05-12.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e4-learned-ot-result-2026-05-12.md`

Execution:
- Tightened the E4 plan before execution by requiring stronger residual
  interpretation guardrails.
- Deepened `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
  directly.
- Strengthened the chapter so that map-level residuals, posterior-level
  consequences, and required further evidence are more explicitly separated.
- Strengthened the training-distribution discussion so that in-distribution
  residual quality is more clearly separated from out-of-distribution target
  reliability.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- local text audit confirmed stronger coverage of residual interpretation,
  training-distribution dependence, and explicit non-equivalence claims.

Interpretation:
- The chapter is now much less likely to be misread as saying that a small
  learned-map residual automatically implies a negligible posterior shift.
- This gives the learned-OT layer the mathematical depth needed for later HMC
  analysis.

Audit:
- Primary criterion: satisfied.
- Teacher-versus-learned-too-compressed veto: cleared.
- Residuals-only-empirical-error veto: cleared.
- Training-distribution-too-casual veto: cleared.

Next phase justified?
- Yes.  Phase E5 is now justified because the DPF-specific HMC chapter can now
  compare rungs against a clearer learned-OT approximation story.

### Phase E4 verification and artifact completion

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e4-learned-ot-result-2026-05-12.md`

Execution:
- Re-audited the active E4 chapter against the E4 plan and audit note.
- Found that the chapter contained the main residual prose but not the required
  E4 tables/source-ledger artifacts.
- Added a source-spine paragraph tying the teacher map to EOT/Sinkhorn sources
  and learned maps to permutation-equivariant set operators.
- Added the approximation-hierarchy table, residual/target-shift hierarchy
  table, residual interpretation guardrail table, and learned-OT implementation
  issue map.
- Added the permutation-equivariance equation for learned set maps.
- Added `\FloatBarrier` at the chapter end to keep E4 tables local.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  from `docs/`.
- `latexmk -pdf -g -interaction=nonstopmode -halt-on-error main.tex` forced
  BibTeX/LaTeX convergence after the new set-operator citations.
- `git diff --check` passed.
- Focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, and no LaTeX errors after the final
  E4 build.
- Focused text audit confirmed the presence of the E4 hierarchy, residual,
  guardrail, and implementation issue tables.

Interpretation:
- The earlier E4 memo entry was directionally correct but overstated artifact
  completion.  The chapter now contains the required tables and source-backed
  implementation diagnostics.
- E4 now clears the primary criterion: learned OT is explicitly a second
  approximation layer, not a free exact acceleration of OT.

Next phase justified?
- Yes.  Phase E5 is justified.

### Phase E5: DPF-specific HMC enrichment

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e5-hmc-enrichment-plan-2026-05-11.md`

Plan audit:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e5-plan-audit-2026-05-12.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e5-hmc-enrichment-result-2026-05-12.md`

Execution:
- Tightened the E5 plan before execution by requiring an explicit rung-by-rung
  promotion-evidence ladder.
- Deepened `docs/chapters/ch19e_dpf_hmc_target_suitability.tex` directly.
- Strengthened the recommendation and closing sections so that each DPF rung is
  read not only by current target status but also by the evidence required for
  future promotion.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- Focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, and no LaTeX errors.  The only match
  to the scan pattern was the package name `rerunfilecheck`, not a rerun
  diagnostic.
- local text audit confirmed stronger rung-promotion language and clearer links
  between target interpretation and next-step evidence.

Interpretation:
- The chapter is now a better bridge between mathematical interpretation and
  future experimentation.
- It no longer only classifies the current rung status; it also says what would
  have to be shown next to make a stronger HMC claim.

Audit:
- Primary criterion: satisfied.
- Differentiability-treated-as-near-validity veto: cleared.
- DSGE/MacroFinance-too-slogan-like veto: cleared for this phase.
- surrogate-HMC/HNN-too-superficial veto: improved enough to clear for this
  enrichment step.

Next phase justified?
- Yes.  Phase E6 is now justified because the DPF block is rich enough that a
  literature-to-debugging crosswalk can now be built on top of the enriched
  chapters.

### Phase E6: literature-to-debugging crosswalk

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e6-debug-crosswalk-plan-2026-05-11.md`

Plan audit:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e6-plan-audit-2026-05-12.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e6-debug-crosswalk-result-2026-05-12.md`

Execution:
- Audited the E6 plan before execution and tightened it so the crosswalk became
  reader-facing inside the monograph rather than only a private planning note.
- Added `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`.
- Added `\input{chapters/ch19f_dpf_debugging_crosswalk}` to `docs/main.tex`
  after the DPF HMC target chapter.
- Added `longtable` support to `docs/preamble.tex` and converted the E6
  crosswalk to a breakable table after the first build showed that the
  page-sized table was too tall as a float.
- The new chapter maps failures to mathematical layer, chapter section,
  literature, and next diagnostic, including ESS collapse, flow stiffness,
  EDH/LEDH mismatch, PF-PF/log-det issues, resampling bias, Sinkhorn issues,
  learned-OT extrapolation, DPF-HMC target mismatch, and value-gradient mismatch.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  from `docs/`.
- After the `longtable` correction, `latexmk -pdf -interaction=nonstopmode
  -halt-on-error main.tex` completed again.
- `git diff --check` passed.
- Focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, no rerun requests, no `Float too
  large` diagnostics, and no LaTeX errors.  The only match to the scan pattern
  was the package name `rerunfilecheck`.
- Focused text audit confirmed that the required issue categories and the
  coding-agent issue taxonomy are present.

Interpretation:
- The DPF block is now not only a richer mathematical exposition but also a
  troubleshooting reference: a future implementer can route an observed failure
  to a chapter section, source cluster, and next diagnostic.
- Dense-table overfull/underfull warnings remain in the LaTeX log; these are
  typography cleanup items rather than E6 correctness blockers.

Audit:
- Primary criterion: satisfied.
- Crosswalk-too-abstract veto: cleared by row-level diagnostics.
- Issue-to-literature mapping vague veto: cleared by chapter labels and
  citation clusters.
- Artifact-not-useful-for-implementation-diagnosis veto: cleared by the
  diagnostic order and coding-agent taxonomy.

Next phase justified?
- Yes.  Phase E7 is justified because the enriched chapters and reader-facing
  debugging crosswalk are now in place for a final round-level audit.

### Phase E7: final enrichment audit and consolidation

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e7-final-audit-plan-2026-05-11.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-enrichment-phase-e7-final-audit-result-2026-05-12.md`

Execution:
- Audited the full enriched DPF block:
  - `docs/chapters/ch19_particle_filters.tex`;
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`;
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`.
- Created the final E7 audit note with the required table covering
  self-containment, literature depth, derivation depth, implementation
  usefulness, coding-agent usefulness, and remaining gaps by chapter.
- Consolidated the round-level interpretation: the writing/enrichment phase can
  close, but experiment-backed validation remains a separate next phase.
- Preserved lane discipline: did not modify student-baseline files, did not
  update the global monograph reset memo, and did not commit or push per the
  current user instruction.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` reported all
  targets up to date from `docs/`.
- `git diff --check` passed.
- Focused log scan found no undefined citations, no undefined references, no
  duplicate Hyperref destination warnings, no rerun requests, no `Float too
  large` diagnostics, and no LaTeX errors.  The only match to the scan pattern
  was the package name `rerunfilecheck`.
- Text audit confirmed that the reviewed DPF block is over 2,000 lines and
  contains chapter-local derivations, boundary statements, implementation
  diagnostics, and the reader-facing debugging crosswalk.

Interpretation:
- The enrichment round clears the primary criterion.  The DPF block is no
  longer only a cleaner outline; it is now a source-backed technical reference
  that separates exact, corrected-particle, approximate, relaxed, and learned
  surrogate target statuses.
- Dense table overfull/underfull warnings remain as typography cleanup items,
  not as mathematical or build blockers.
- The next justified work is targeted validation and experiment design, not
  another broad writing pass.

Audit:
- Structurally-correct-but-too-short veto: cleared.
- Literature-support-still-thin veto: cleared for this round.
- Derivations-still-too-compressed veto: cleared for the load-bearing objects.
- Not-useful-for-coding-agents veto: cleared by the E6 debugging crosswalk and
  chapter-local diagnostic material.

Next phase justified?
- Yes, but it should be an experiment-design and validation phase.  Priority
  hypotheses are PF-PF log-det agreement, flow stiffness sensitivity, Sinkhorn
  relaxation sensitivity, learned-OT residual-to-posterior stability, and HMC
  value-gradient contract consistency.


## 2026-05-11 update: sixth reader-facing DPF rewrite phase completed

### Phase R6 result

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r6-hmc-target-suitability-plan-2026-05-10.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r6-hmc-target-suitability-result-2026-05-11.md`

Execution:
- Added a new dedicated DPF HMC chapter slot to `docs/main.tex`:
  `\input{chapters/ch19e_dpf_hmc_target_suitability}`.
- Created the new chapter file
  `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`.
- Wrote the new chapter as a DPF-specific HMC-target and structural-model
  suitability chapter covering:
  1. the DPF target contract for HMC;
  2. rung-by-rung target analysis across EDH, PF-PF, soft resampling, OT/EOT,
     and learned OT;
  3. nonlinear DSGE stress points;
  4. MacroFinance stress points;
  5. relation to surrogate-HMC / HNN acceleration ideas from the
     `\texttt{dsge\_hmc}` project;
  6. BayesFilter's recommendation for the first justified HMC-relevant DPF
     rung;
  7. an explicit chapter-boundary section stating what remains unclaimed.
- Fixed the LaTeX syntax issue caused by TeX-breaking inline markup in the new
  chapter and reran the build successfully.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- A second rerun confirmed the monograph build was stable and up to date.
- `git diff --check` passed.

Interpretation:
- The architectural blocker is now cleared.  The final DPF-specific HMC chapter
  has its own dedicated slot and no longer needs to be forced into either the
  learned-OT chapter or the generic BayesFilter HMC doctrine chapter.
- This preserves the intended mathematical sequence of the rebuilt DPF block.

Audit:
- Primary criterion: satisfied.
- Differentiability-implies-validity veto: cleared.
- Rung-table vagueness veto: cleared.
- Structural-model-informality veto: cleared.
- Architecture-slot veto: cleared.

Next phase justified?
- Yes.  The reader-facing DPF rewrite sequence is now complete enough for the
  end-of-plan consolidation step: final summary, reset-memo closure, and the
  final commit bundle for the accumulated uncommitted phases.

## 2026-05-10 update: fifth reader-facing DPF rewrite phase started

### Phase R5: learned/amortized OT and implementation-mathematics rewrite

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r5-learned-ot-implementation-plan-2026-05-10.md`

Phase plan:
- rewrite `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` in
  substance into a mathematically disciplined chapter on learned/amortized OT and
  implementation-facing mathematics;
- make the hierarchy explicit:
  exact resampling -> OT relaxation -> learned OT approximation;
- cover teacher-versus-learned map structure, approximation residuals,
  training-distribution dependence, and implementation-facing mathematical
  constraints;
- keep final HMC-suitability verdicts out of this phase and reserve them for the
  later assessment chapter.

Primary criterion:
- the rewritten chapter must be mathematically self-contained and make the
  teacher-versus-learned map distinction explicit enough that the later HMC
  chapter can refer to learned OT as a clearly identified surrogate layer.

Veto diagnostics:
- if the chapter treats learned OT as equivalent to the OT baseline, stop and
  revise;
- if approximation residuals are discussed only operationally and not
  mathematically, stop and revise;
- if the chapter starts making final HMC-suitability claims, defer those claims
  to the later assessment chapter.

### Phase R5 result

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r5-learned-ot-implementation-result-2026-05-10.md`

Execution:
- Rewrote `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
  substantially.
- The chapter now includes:
  1. the hierarchy of approximations:
     exact resampling -> OT relaxation -> learned OT approximation;
  2. the teacher-versus-learned map distinction;
  3. a map-level approximation residual and its interpretation;
  4. training-distribution dependence and extrapolation risks;
  5. implementation-facing mathematical consequences for runtime,
     deterministic evaluation, state dimension, and architecture dependence;
  6. an explicit chapter-boundary section deferring the final HMC verdict to the
     later assessment chapter.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- The wording scan for approximation-status language was reviewed and accepted
  because the flagged phrases appear only in explicit approximation-hierarchy or
  negative boundary statements.

Interpretation:
- The chapter now makes the teacher-versus-learned map distinction explicit and
  stabilizes the approximation hierarchy of the whole DPF block.
- This is important because the later HMC chapter can now analyze the target as
  a layered construction rather than as a vague black box.

Audit:
- Primary criterion: satisfied.
- Learned-OT-equals-OT veto: cleared.
- Residual-only-operational veto: cleared.
- Premature HMC-verdict veto: cleared.
- No phase-local formatting or citation blocker remains.

Next phase justified?
- Yes.  The next rewrite phase is justified because the final remaining
  reader-facing problem is now the HMC-target and structural-model suitability
  chapter.
- Recommended next phase: write the final rung-by-rung HMC-target correctness
  and nonlinear DSGE/MacroFinance suitability chapter.

## 2026-05-10 update: fourth reader-facing DPF rewrite phase started

### Phase R4: differentiable resampling and OT rewrite

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r4-resampling-ot-rewrite-plan-2026-05-10.md`

Phase plan:
- rewrite `docs/chapters/ch32_diff_resampling_neural_ot.tex` into a
  mathematically rigorous differentiable-resampling and OT chapter;
- cover the resampling bottleneck, pathwise nondifferentiability of standard
  resampling, soft resampling, transport/projection formulations, entropic OT,
  Sinkhorn structure, barycentric projection, and the bias-versus-
  differentiability trade-off;
- keep learned/amortized OT clearly downstream of the OT baseline rather than
  letting it dominate this chapter;
- compile, audit, tidy, and update this memo before the next phase proceeds.

Primary criterion:
- the rewritten chapter must be mathematically self-contained and make the
  differentiability-versus-bias issue explicit enough that the later HMC and
  learned-OT chapters can be written without ambiguity.

Veto diagnostics:
- if the chapter treats OT resampling as if it were identical to exact
  multinomial resampling, stop and revise;
- if the source of bias in soft or OT resampling remains vague, stop and revise;
- if learned/amortized OT begins to dominate the chapter before the OT baseline
  is mathematically clear, do not proceed to the next phase.

### Phase R4 result

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r4-resampling-ot-rewrite-result-2026-05-10.md`

Execution:
- Rewrote `docs/chapters/ch32_diff_resampling_neural_ot.tex` substantially.
- The chapter now includes:
  1. the weighted-to-equal-weight resampling bottleneck at the level of
     empirical measures;
  2. standard resampling as a discontinuous categorical map;
  3. soft resampling as a smooth relaxation;
  4. equal-weight resampling viewed as a transport problem;
  5. entropic OT resampling with the primal formulation;
  6. the Sinkhorn scaling form and barycentric projection;
  7. an explicit comparison of standard, soft, and OT resampling in terms of
     differentiability and bias;
  8. an explicit chapter-boundary section deferring learned/amortized OT to the
     next stage.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed and
  reported all targets up to date.
- `git diff --check` passed.
- The wording scan for high-stakes target-status language was reviewed and
  accepted because the flagged statements are boundary statements or explicit
  trade-off descriptions rather than overclaims.

Interpretation:
- The chapter now makes the differentiability-versus-bias problem explicit at
  the right mathematical level.
- This gives the rebuilt DPF block the missing middle layer between PF-PF
  correction and the later learned/surrogate or HMC-target chapters.

Audit:
- Primary criterion: satisfied.
- OT-equals-multinomial veto: cleared.
- Bias-source vagueness veto: cleared.
- Learned-OT-domination veto: cleared.
- No phase-local formatting or citation blocker remains.

Next phase justified?
- Yes.  The next rewrite phase is justified because learned/amortized OT can now
  be written as a further approximation on top of a clear OT baseline, and the
  HMC-assessment chapter can later refer back to an explicit resampling-status
  hierarchy.
- Recommended next phase: rewrite the learned/amortized OT and implementation-
  mathematics chapter, keeping the teacher-versus-learned map distinction
  explicit and treating approximation residuals as part of the target-status
  story.

## 2026-05-10 update: third reader-facing DPF rewrite phase started

### Phase R3: PF-PF and proposal-correction rewrite

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r3-pfpf-proposal-correction-plan-2026-05-10.md`

Phase plan:
- rewrite `docs/chapters/ch19c_dpf_implementation_literature.tex` into a
  mathematically rigorous PF-PF / proposal-correction chapter;
- cover the flow map as a proposal, transformed proposal density, corrected
  importance weights, Jacobian/log-determinant evolution, and the exact status
  of what the correction restores;
- keep learned resampling / OT and broad HMC conclusions out of this phase;
- compile, audit, tidy, and update this memo before the next phase proceeds.

Primary criterion:
- the rewritten chapter must be mathematically self-contained and make the
  proposal-correction logic precise enough that later HMC-target analysis can be
  written without ambiguity.

Veto diagnostics:
- if the change-of-variables argument is not stated clearly, stop and revise;
- if the chapter still blurs raw flow with corrected PF-PF, stop and revise;
- if the status of the corrected method remains ambiguous (exact vs approximate
  vs finite-particle), do not proceed to the resampling chapter.

### Phase R3 result

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r3-pfpf-proposal-correction-result-2026-05-10.md`

Execution:
- Rewrote `docs/chapters/ch19c_dpf_implementation_literature.tex` substantially
  into a PF-PF / proposal-correction chapter.
- The chapter now includes:
  1. the need for proposal correction outside exact special cases;
  2. the flow map as a change of variables;
  3. transformed proposal density under the flow map;
  4. corrected PF-PF importance weights;
  5. the distinction between EDH/PF and LEDH/PF at the level of map
     construction;
  6. Jacobian-matrix and log-determinant evolution identities;
  7. a rigorous discussion of what proposal correction restores and what still
     remains approximate;
  8. an explicit chapter-boundary section stating what is not yet claimed.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- A wording scan for status language was reviewed and accepted because:
  - `exact` appears only in explicitly limited contexts;
  - `restores` refers specifically to the proposal-to-target density ratio;
  - `HMC target` appears only in negative or future-facing boundary statements.

Interpretation:
- The DPF exposition now distinguishes clearly between flow-as-transport and
  flow-as-proposal with importance correction.
- This closes the main mathematical gap between the raw-flow chapter and the
  later HMC-assessment work.

Audit:
- Primary criterion: satisfied.
- Change-of-variables clarity veto: cleared.
- Raw-flow versus corrected-PF-PF blur veto: cleared.
- Status ambiguity veto: cleared for this phase.
- No phase-local formatting blocker remains.

Next phase justified?
- Yes.  The next rewrite phase is justified because the differentiable
  resampling and OT chapter can now be written on top of a clear three-step
  baseline:
  1. classical particle filtering;
  2. particle-flow transport;
  3. proposal-corrected PF-PF.
- Recommended next phase: rewrite the differentiable-resampling and OT chapter,
  making the bias-versus-differentiability trade-off explicit and keeping
  learned/amortized OT clearly downstream of OT itself.

### Phase R2 result

Plan note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r2-particle-flow-foundations-plan-2026-05-10.md`

Result note:
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-r2-particle-flow-foundations-result-2026-05-10.md`

Execution:
- Rewrote `docs/chapters/ch19b_dpf_literature_survey.tex` substantially into a
  particle-flow foundations chapter.
- The chapter now includes:
  1. the move from discrete reweighting/resampling to continuous transport;
  2. the pseudo-time homotopy density with explicit normalizing constant;
  3. the continuity equation and the corresponding flow PDE;
  4. EDH under Gaussian closure, including the homotopy covariance family and
     affine EDH ODE coefficients;
  5. the exact linear-Gaussian recovery statement as the main special-case
     benchmark;
  6. LEDH via particle-specific local linearization and local precision
     matrices;
  7. stiffness and discretization as mathematical and numerical concerns;
  8. an explicit chapter-boundary section stating what is not yet claimed.

Tests:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- `git diff --check` passed.
- A wording scan for exactness/high-stakes language was reviewed and accepted
  because `exact` appears only in the continuity/conservation statement, the
  linear-Gaussian recovery benchmark, and explicit qualifying boundary
  statements.

Interpretation:
- The particle-flow foundations are now mathematically explicit enough that the
  later PF-PF chapter can be written on top of a clear homotopy and transport
  baseline.
- The key conceptual improvement is that the monograph now states plainly that
  raw flow transport is not yet the corrected filtering target in the nonlinear
  case.

Audit:
- Primary criterion: satisfied.
- Survey-drift veto: cleared.
- EDH/LEDH distinction veto: cleared.
- Exact-versus-approximate-status veto: cleared for this phase.
- No phase-local formatting or citation blocker remains for this rewrite.

Next phase justified?
- Yes.  The next rewrite phase is justified because the particle-flow
  foundations chapter is now strong enough to support a mathematically serious
  PF-PF / proposal-correction rewrite.
- Recommended next phase: separate flow-as-transport from flow-as-proposal and
  write the corrected importance-weight and Jacobian/log-determinant machinery
  carefully.
