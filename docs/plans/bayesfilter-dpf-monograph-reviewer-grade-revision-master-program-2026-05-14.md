# Master program: reviewer-grade DPF monograph revision

## Date

2026-05-14

## Status

This program supersedes the prior DPF monograph rebuild and enrichment programs
as the governing plan for the next reader-facing revision round.

Earlier work improved the chapter architecture and added useful mathematical
material.  That work is not sufficient for the required audience.  The next
round must be treated as a demanding reviewer-grade revision, not as another
summary rewrite.

Relevant predecessor artifacts:

- `docs/plans/bayesfilter-dpf-monograph-rebuild-master-program-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-final-summary-2026-05-11.md`
- `docs/plans/bayesfilter-dpf-monograph-enrichment-master-program-2026-05-11.md`
- `docs/plans/bayesfilter-dpf-monograph-reboot-handoff-2026-05-13.md`

Current target chapters:

- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch19b_dpf_literature_survey.tex`
- `docs/chapters/ch19c_dpf_implementation_literature.tex`
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

## Audience and standard

The intended review setting is not a casual technical report review.  The
audience includes mathematically mature academics and senior technical reviewers
who may be physicists, mathematicians, chemists, engineers, or other scientists,
not necessarily economists or filtering specialists.  They should be assumed to
be skeptical, rigorous, and unwilling to search the literature to repair missing
steps.

The document must therefore satisfy a high external-review standard:

1. It must be self-contained enough that a mathematically mature non-specialist
   can follow the argument without reading the cited papers first.
2. It must be conservative about every claim concerning differentiable particle
   filtering for banking, because the application is new and likely to attract
   skepticism.
3. It must show derivations rather than merely asserting them.
4. It must state assumptions, approximation layers, and failure modes before
   making recommendations.
5. It must distinguish clearly between theorem, approximation, heuristic,
   implementation device, and empirical hypothesis.
6. It must be useful both to a rigorous reviewer and to a coding agent asked to
   implement or debug the methods.

## Governing diagnosis

The current DPF block is structurally improved but still reads too much like a
compressed technical report.  It has many correct ingredients, but it does not
yet force the skeptical reader through enough explicit mathematics, literature
context, and claim boundary control.

The next round is therefore not a polishing round.  It is a substantial
revision.  The working hypothesis is that every DPF chapter needs some
combination of:

- a stronger local introduction for non-specialists;
- a notation table and mathematical object inventory;
- explicit assumptions before the main equations;
- derivations expanded from endpoint formulas into auditable steps;
- a deeper literature synthesis;
- a claim-status table separating proven facts from approximations and
  hypotheses;
- a reviewer-facing limitations section;
- and implementation checks tied to the formulas.

## Non-negotiable revision principles

### Principle 1: no casual-report prose

Do not write as if the reader already accepts the relevance of differentiable
particle filters.  The text must earn the claim.  Sentences such as "this gives
a differentiable filter" or "this is suitable for HMC" are unacceptable unless
the local scalar objective, gradient path, target interpretation, and
approximation status have already been defined.

### Principle 2: self-contained before citation-dependent

Citations may support and locate the argument, but they must not carry missing
reasoning.  A skeptical reviewer should not have to open Daum-Huang, PF-PF,
Corenflos, Cuturi, Andrieu, or Neal to understand the local mathematical
argument.  The chapter should explain enough locally, then cite the source for
provenance, variants, and deeper results.

### Principle 3: derivations before verdicts

Every central verdict must come after the derivation that justifies it.  For
example:

- the EDH formula must follow from the homotopy density and Gaussian closure;
- LEDH must be presented as a local linearization with explicit dependence on
  particle-local Jacobians;
- PF-PF weights must follow from change of variables and proposal correction;
- log-determinant evolution must follow from the flow Jacobian ODE;
- soft resampling bias must be shown using test functions, not merely asserted;
- Sinkhorn/EOT must be derived from the constrained regularized transport
  problem;
- learned OT must be treated as a student approximation to a teacher map;
- HMC target claims must follow from value-gradient consistency and target
  invariance logic.

### Principle 4: source-grounded skepticism

The revision must make it easy for a reviewer to see what is established in the
literature, what is an interpretation for BayesFilter, and what remains an
engineering hypothesis.  No new banking or HMC claim may be presented as
settled unless the evidence is explicitly identified.

### Principle 5: implementation relevance is mandatory

The mathematics must be connected to what code must compute:

- densities and log weights;
- flow ODEs and Jacobians;
- log determinants;
- resampling maps;
- Sinkhorn residuals and regularization;
- learned-map residuals;
- likelihood scalars supplied to samplers;
- gradients supplied to samplers;
- and diagnostic checks for numerical failure.

The chapter should not merely sound rigorous; it should constrain an
implementation.

## Worktree and branch safety

Before each execution phase:

1. run `git branch --show-current`;
2. run `git status --short`;
3. inspect whether dirty files belong to the DPF monograph lane or another lane;
4. do not edit student-baseline planning artifacts unless explicitly directed;
5. keep this reviewer-grade DPF revision separate from student experimental
   workstreams.

The live state is not fixed by this master document.  The first execution
artifact supersedes the original snapshot:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-result-2026-05-15.md`

That result records `main` as the active branch, `main` ahead of `origin/main`
by two commits, unrelated dirty student-baseline files, and untracked
reviewer-grade DPF planning artifacts.  Every later phase must rerun the branch
and worktree checks because the dirty set may change between agents.

## Required tool lanes

### ResearchAssistant lane

Use the local ResearchAssistant MCP before and during source-facing phases.
Current status established by P0/P2:

- workspace root: `/home/ubuntu/python/ResearchAssistant`;
- mode: read-only;
- offline mode: enabled;
- generic DPF keyword searches returned no stored paper summaries;
- P2 found no local ResearchAssistant paper summaries for the DPF source
  families and did not ingest sources because write/fetch workflows were not
  authorized.

This means the reviewer-grade revision must not describe DPF claims as
ResearchAssistant-reviewed unless a later artifact records a reviewed local
source summary.  Until then, source grounding is bibliography-spine plus local
chapter/source-map review, and load-bearing mathematical claims must be derived
locally or explicitly weakened.

Required ResearchAssistant outputs for this program:

- a source availability register;
- a claim-support register for every major method family;
- a list of source gaps that prevent strong claims;
- review status of any local paper summaries used.

If ResearchAssistant remains empty for a source family, the phase result must
say so and must not use "source-grounded" to mean paper-reviewed support.

### MathDevMCP lane

Use MathDevMCP for bounded derivation checks, label lookup, document/code
search, and proof-obligation decomposition where applicable.  It is not a
substitute for mathematical judgment, but it must be used to reduce unsupported
formula drift.

Required MathDevMCP outputs for this program:

- derivation-obligation lists by chapter;
- audit evidence for selected load-bearing equations;
- conservative notes for obligations that cannot be mechanically verified;
- implementation-brief checks where a chapter makes code-facing claims.

Every derivation-heavy phase must record either MathDevMCP evidence or a
specific reason that the obligation was unsuitable for the tool and was checked
manually.  A phase cannot pass by silently skipping the mandated tool lane.

### Bibliography and chapter-source lane

Use `docs/references.bib`, existing chapter citations, and predecessor planning
artifacts as the local source spine.  Do not add citation padding.  Every
important citation must answer:

- What exact question does this source address?
- What assumptions does it use?
- What result, algorithm, or warning does it support?
- How does that result translate into BayesFilter notation?
- What does it not prove for the banking DPF application?

## Global execution cycle

Every phase must follow:

```text
preflight -> phase plan -> execute -> compile/test -> audit -> tidy -> update reset memo
```

Continue automatically only if:

- the phase primary criterion is satisfied;
- no veto diagnostic fires;
- and the next phase remains justified by the evidence collected.

If a veto diagnostic fires, stop the phase, record the blocker, and either
repair it or ask for direction.

## Global veto diagnostics

A phase is not complete if any of the following remain true:

1. A central equation appears without definitions for all symbols.
2. A central formula is asserted without derivation, source equation, or proof
   obligation.
3. A claim says or implies "exact", "unbiased", "consistent", "HMC-ready",
   "validated", "robust", or "production" without explicit assumptions and
   limitations.
4. A citation is present but the text does not say what the citation contributes.
5. A non-specialist mathematical reader would need to search papers to fill in a
   load-bearing step.
6. A chapter makes a banking or structural-model claim without separating
   mathematical fact from modeling judgment.
7. A chapter recommends an implementation route without diagnostic tests tied to
   the relevant formulas.
8. A table compresses a claim that should be derived in prose.
9. A section reads like a survey summary rather than a monograph explanation.
10. The revised text would not materially improve a coding agent's
    implementation or debugging behavior.

## Deliverables

This program must produce:

1. a reviewer-grade source and claim audit;
2. a derivation-obligation register;
3. chapter-local revision plans;
4. substantially revised DPF chapters;
5. claim-status tables embedded in the chapters or adjacent notes;
6. a skeptical-reviewer limitations section for each major method family;
7. an implementation-test map tied to equations and approximation layers;
8. a compile and cross-reference audit;
9. an independent final reviewer-readiness audit;
10. an updated DPF reset memo or a new reviewer-grade reset memo;
11. a final supervisor audit showing that the master program and phase subplans
    agree on taxonomy, artifacts, source limits, build gates, and lane safety.

Result artifact naming rule:

- Plan filenames keep their original creation date.
- Result, audit, and closeout artifacts use the actual completion date in the
  filename.
- Existing completed results are:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-result-2026-05-15.md`;
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`;
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`.
  Future subplans should name result artifacts as `{YYYY-MM-DD}` patterns unless
  the artifact already exists.

## Required claim-status vocabulary

The chapters must use a controlled vocabulary:

- **Exact model identity**: an identity for the stated probabilistic model under
  stated assumptions.
- **Unbiased particle estimator**: a Monte Carlo estimator whose expectation is
  the stated likelihood or normalizing constant under stated sampling rules.
- **Consistent approximation**: a finite-particle or numerical construction with
  a stated limiting interpretation.
- **Approximate closure**: a Gaussian, local-linear, moment, or other closure
  that changes the mathematical object.
- **Relaxed target**: a deliberately smoothed or regularized target different
  from the original categorical or exact object.
- **Learned surrogate**: a trained approximation to another computational
  object, with training-distribution dependence.
- **Engineering hypothesis**: a plausible implementation route not yet validated
  by BayesFilter-specific experiments.
- **Unsupported claim**: a statement that must be removed, weakened, or held
  pending source or experiment evidence.

## Required reviewer-facing chapter pattern

Each main DPF chapter should be revised toward this pattern, unless a local
reason is recorded:

1. problem statement for non-specialists;
2. notation and object inventory;
3. assumptions and what is being conditioned on;
4. baseline exact or classical object;
5. derivation of the proposed construction;
6. claim-status table;
7. literature synthesis with source contributions and limits;
8. implementation implications and diagnostic tests;
9. skeptical-reviewer limitations;
10. transition to the next chapter.

## Governing phase taxonomy

The executable taxonomy for this reviewer-grade round is **P0-P13**.  The
earlier R0-R12 outline in predecessor discussion is superseded and must not be
used as a parallel execution plan.  If any older note mentions an R phase, map
it through the table below and then follow the P-phase subplan.

| Governing phase | Artifact | Status | Role |
|---|---|---|---|
| P0 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-plan-2026-05-14.md` | completed; result dated `2026-05-15` | worktree, build, source, label, and evidence inventory |
| P1 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-plan-2026-05-14.md` | completed; result dated `2026-05-15` | skeptical-reader argument map and claim ledger |
| P2 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-plan-2026-05-14.md` | completed; result dated `2026-05-15` | source roles, source gaps, and ResearchAssistant-empty fallback |
| P3 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p3-smc-baseline-expansion-plan-2026-05-14.md` | next gating phase | classical filtering and SMC baseline expansion |
| P4 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p4-edh-ledh-derivation-plan-2026-05-14.md` | provisional result dated `2026-05-15`; requires P3 impact reconciliation | EDH/LEDH derivation expansion |
| P5 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p5-pfpf-jacobian-audit-plan-2026-05-14.md` | provisional result dated `2026-05-15`; requires P3 impact reconciliation | PF-PF proposal correction and Jacobian audit |
| P6 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p6-resampling-ot-sinkhorn-plan-2026-05-14.md` | blocked until P3 passes, P4/P5 reconciliation passes, and any existing `ch32` partial diff is audited | differentiable resampling and OT/Sinkhorn expansion |
| P7 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p7-learned-ot-defensibility-plan-2026-05-14.md` | pending P6 gate | learned/amortized OT defensibility expansion |
| P8 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p8-hmc-banking-target-plan-2026-05-14.md` | pending P7 gate | HMC target correctness and banking suitability expansion |
| P9 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p9-debugging-verification-contract-plan-2026-05-14.md` | pending P8 gate | equation-indexed debugging and verification contract |
| P10 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-plan-2026-05-14.md` | pending P9 gate | cross-chapter notation, claim-status, and literature consolidation |
| P11 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p11-derivation-audit-plan-2026-05-14.md` | pending P10 gate | mathematical derivation audit |
| P12 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p12-hostile-reader-audit-plan-2026-05-14.md` | pending P11 gate | build, PDF review, and hostile-reader audit |
| P13 | `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p13-final-readiness-plan-2026-05-14.md` | pending P12 gate | final readiness report and reset memo update |

Historical crosswalk:

- R0 maps to P0 and P2.
- R1 maps to P1.
- R2 maps to P3.
- R3 maps to P4.
- R4 maps to P5.
- R5 maps to P6.
- R6 maps to P7.
- R7 maps to P8.
- R8 maps to P9.
- R9 maps to P2 and P10.
- R10 maps to P11 and P12.
- R11 maps to P12.
- R12 maps to P13.

## Common execution contract for P3-P13

Every remaining phase must begin by recording:

1. `git branch --show-current`;
2. `git status --short`;
3. whether any dirty or untracked file is outside the reviewer-grade DPF lane;
4. the exact allowed write set for that phase.

Allowed write sets:

- P3-P9 may edit only their named DPF chapter plus their phase result artifact,
  unless the phase result records a specific reason to touch a shared file such
  as `docs/references.bib`.
- P10-P13 may edit reviewer-grade DPF planning artifacts, the reviewer-grade
  reset memo, and DPF chapters as required by audit repair.
- Student-baseline files under `docs/plans/bayesfilter-student-dpf-baseline-*`
  are out of lane and must not be edited, staged, reverted, or summarized as
  DPF monograph evidence unless a later user instruction explicitly changes the
  lane.

Source-grounding contract:

- P2 is the governing source register until superseded by a later source-intake
  artifact.
- ResearchAssistant currently has no local DPF paper summaries.  Therefore
  P3-P9 must treat source roles as bibliography-spine support, not
  ResearchAssistant-reviewed support.
- A phase may strengthen a mathematical claim only if the local text derives it
  in BayesFilter notation or the phase result records a reviewed primary-source
  path with assumptions and limits.
- If source support is unavailable, the claim must be weakened, marked as an
  engineering hypothesis, or recorded as unresolved.

Out-of-order P4/P5 reconciliation:

- P4 and P5 result artifacts dated `2026-05-15` are present in the worktree.
  They are in-lane artifacts and must not be reverted merely because this master
  now makes P3 the next gating phase.
- Because P3 baseline definitions were not completed before those results, P4
  and P5 remain provisional until P3 is completed and a bounded reconciliation
  note confirms that P3 did not change baseline notation, estimator status, or
  source boundaries in a way that invalidates P4/P5.
- If P3 changes any symbol, definition, estimator-status claim, or
  differentiability boundary used by P4/P5, repair P4/P5 before P6.
- P6 may start only after the P3 result and the P4/P5 reconciliation note both
  pass their veto checks.

Existing P6 partial-diff rule:

- `docs/chapters/ch32_diff_resampling_neural_ot.tex` may already contain
  in-lane partial P6 edits without a P6 result artifact.  Do not revert those
  edits solely because they predate the tightened plan.
- The P6 phase must begin by auditing the existing `ch32` diff, deciding whether
  to adopt, repair, or supersede it, and recording that decision in the P6
  result.
- P6 is not complete until it has a dated result artifact, build/layout status,
  derivation-obligation evidence, and a disposition for the pre-existing
  partial diff.

Tool-evidence contract:

- P3-P11 must create an explicit derivation-obligation table in the phase
  result for load-bearing formulas touched in that phase.
- For each obligation, record `MathDevMCP`, `manual`, or `blocked` as the audit
  method.  If MathDevMCP is not used for a listed obligation, the result must
  explain why the obligation was unsuitable for bounded tool checking.
- ResearchAssistant queries must be recorded in source-facing phases.  Empty
  results are valid evidence only for source unavailability, not for claim
  support.

Build and layout contract:

- P3, P4, P5, P6, P8, P9, P10, P12, and P13 must run the established LaTeX
  build or record why it could not be run.
- P6 and P10 must additionally inspect table/layout risks because P0 already
  found DPF-local table readability warnings.
- A phase that adds substantial tables may not defer layout truth until P12
  unless its result records a bounded reason and a follow-up gate.

Result artifact contract:

- P3-P13 result artifacts must use the actual completion date in the filename:
  `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p{N}-<slug>-result-{YYYY-MM-DD}.md`
  when the phase has a numbered phase result.
- Audit and closeout artifacts may use their descriptive names, but still use
  the actual completion date.

## Historical R-phase outline superseded by P0-P13

The remaining R-phase text below is retained only as historical design context.
It is not executable.  When it conflicts with the P0-P13 taxonomy or the common
execution contract above, the P-phase subplan and the common contract govern.

### Historical R0: preflight and evidence inventory

Purpose:

Establish what exists, what is dirty, what sources are locally available, and
where the current chapters are weakest relative to the required audience.

Tasks:

1. Verify branch and dirty files.
2. Record the DPF chapter line counts, section maps, labels, citations, and
   theorem-like environments.
3. Query ResearchAssistant for available local DPF, particle-flow, SMC, OT,
   HMC, DSGE, and MacroFinance paper summaries.
4. Search `docs/references.bib` and existing chapters for source coverage.
5. Use MathDevMCP to search the DPF chapters for central labels and candidate
   derivation obligations.
6. Produce a source availability register and a gap register.

Primary criterion:

The next phase has a concrete map of chapter weaknesses, source coverage, and
derivation obligations.

Veto diagnostics:

- source availability is assumed rather than checked;
- dirty worktree files are not classified;
- central chapters are revised before the gap register exists.

### Historical R1: skeptical-reader architecture and chapter contract

Purpose:

Redesign the DPF block as an argument for skeptical non-economist academics,
not as a field-internal report.

Tasks:

1. Write a chapter-by-chapter contract specifying what a skeptical reader must
   know before accepting the next step.
2. Define the exact dependency ladder:
   classical filtering -> SMC -> particle flow -> PF-PF correction ->
   differentiable resampling -> EOT/Sinkhorn -> learned OT -> HMC target
   interpretation -> banking suitability.
3. Decide whether the current filenames remain acceptable or whether a later
   renaming phase is needed.
4. Create a local "claim ladder" showing which claims are exact, approximate,
   relaxed, learned, or empirical.

Primary criterion:

The DPF block has a reviewer-facing argument structure that prevents later
chapters from relying on missing earlier facts.

Veto diagnostics:

- the architecture still assumes reader familiarity with filtering economics;
- target-status distinctions are introduced only late in the sequence;
- learned or relaxed methods are discussed before the exact baseline is clear.

### Historical R2: classical filtering and SMC foundation deepening

Target:

`docs/chapters/ch19_particle_filters.tex`

Purpose:

Make the particle-filter baseline strong enough that later differentiable
constructions can be judged against it.

Required expansions:

1. Define the state-space model, filtering law, prediction law, update law, and
   marginal likelihood in a notation table.
2. Derive the nonlinear filtering recursion from Bayes' rule.
3. Derive sequential importance weights from a general proposal.
4. Specialize to bootstrap/SIR weights.
5. State the bootstrap likelihood estimator and explain its unbiasedness status
   with assumptions and limits.
6. Explain pathwise non-differentiability of resampling without overstating the
   failure of classical SMC.
7. Add a reviewer-facing table separating exact filtering identities,
   finite-particle Monte Carlo approximations, and differentiability issues.

Primary criterion:

A non-specialist mathematical reader can identify the exact target object and
the Monte Carlo estimator before seeing any DPF modification.

Veto diagnostics:

- likelihood unbiasedness is asserted without assumptions;
- resampling is described only informally;
- later DPF claims rely on terms not defined here.

### Historical R3: EDH and LEDH particle-flow deepening

Target:

`docs/chapters/ch19b_dpf_literature_survey.tex`

Purpose:

Turn the particle-flow chapter from a compressed survey into a derivation-led
monograph chapter.

Required expansions:

1. Introduce homotopy densities with normalizing constants and state what
   changes with artificial time.
2. Derive the continuity equation and connect it to deterministic particle
   transport.
3. Derive the EDH affine flow under Gaussian closure in BayesFilter notation.
4. Work through the linear-Gaussian special case explicitly and show what is
   exact there.
5. Derive LEDH as a particle-local linearization and state the resulting local
   approximation status.
6. Add conditions under which the flow map is invertible or numerically fragile.
7. Explain stiffness, step-size sensitivity, and Jacobian spectra as
   implementation concerns.
8. Add a literature synthesis separating Daum-Huang flow origins, PF-PF use,
   later particle-flow variants, and what none of them prove for banking HMC.

Primary criterion:

The reader can reproduce the EDH/LEDH equations and understand exactly which
assumptions make them exact or approximate.

Veto diagnostics:

- EDH is presented as a formula rather than a derivation;
- LEDH is treated as a minor variant without local-linearization assumptions;
- raw flow is implied to be a corrected posterior sampler.

### Historical R4: PF-PF proposal correction and Jacobian deepening

Target:

`docs/chapters/ch19c_dpf_implementation_literature.tex`

Purpose:

Make PF-PF the first fully auditable correction rung, not merely a conceptual
bridge.

Required expansions:

1. Define the pre-flow proposal density, the deterministic flow map, and the
   post-flow proposal density.
2. Derive the change-of-variables formula step by step.
3. Derive corrected importance weights for the post-flow particles.
4. Derive log-determinant evolution from the flow Jacobian ODE and state sign
   conventions.
5. Separate global EDH/PF and local LEDH/PF weight formulas.
6. Explain what proposal correction restores and what it does not restore:
   finite-particle variance, approximate flow construction, resampling, and
   HMC target smoothness remain separate issues.
7. Add implementation tests: affine-flow density check, finite-difference
   Jacobian determinant check, log-weight normalization check, and seed-fixed
   value-gradient check.

Primary criterion:

A reviewer can audit PF-PF as a change-of-variables and importance-sampling
construction without trusting informal particle-flow intuition.

Veto diagnostics:

- the Jacobian determinant appears without derivation;
- "corrected" is used without specifying the corrected object;
- PF-PF is promoted to HMC-ready without variance and smoothness caveats.

### Historical R5: differentiable resampling, OT, and Sinkhorn deepening

Target:

`docs/chapters/ch32_diff_resampling_neural_ot.tex`

Purpose:

Make the resampling and OT chapter rigorous enough for reviewers who know
measure, optimization, or numerical analysis but not DPF.

Required expansions:

1. Define weighted and equal-weight empirical measures.
2. Formalize hard resampling as a random, discontinuous ancestor-selection map.
3. Derive soft resampling as a relaxed map and show bias with test functions.
4. State the transport problem from weighted to equal-weight empirical measures.
5. Derive entropic regularization, the Gibbs kernel form, and Sinkhorn scaling
   equations.
6. Explain barycentric projection and its status as a deterministic map derived
   from a coupling.
7. Separate four objects: categorical resampling, exact OT, entropic OT, and
   finite-iteration Sinkhorn.
8. Add numerical-analysis limits: regularization bias, iteration residual,
   underflow, log-domain stabilization, memory cost, and gradient-through-solver
   choices.
9. Add a reviewer-facing warning that differentiability is purchased by
   changing the object unless a precise correction or limiting argument is
   supplied.

Primary criterion:

The chapter makes the exact price of differentiability mathematically visible.

Veto diagnostics:

- soft or OT resampling is described as a harmless replacement for categorical
  resampling;
- Sinkhorn is presented as a black-box differentiable layer;
- the finite solver path is not distinguished from the mathematical EOT
  problem.

### Historical R6: learned and amortized OT deepening

Target:

`docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

Purpose:

Revise the learned-OT chapter so that neural transport is treated as a
source-grounded approximation hierarchy, not as a technological shortcut.

Required expansions:

1. Define the teacher object: exact OT, EOT, or finite Sinkhorn map.
2. Define the student map, training distribution, loss, and invariance
   requirements.
3. Derive the residual hierarchy: teacher-object error, solver error,
   supervised learning error, distribution shift, and downstream likelihood
   error.
4. Explain permutation equivariance/invariance requirements for particle sets.
5. Separate amortization speed claims from target-correctness claims.
6. Add failure modes: sharp weights, dimension shift, particle-number shift,
   model-region extrapolation, and learned-map non-invertibility or collapse.
7. State what evidence would be required before a bank-facing learned-OT DPF
   claim becomes credible.

Primary criterion:

The chapter prevents a reviewer from interpreting learned OT as an exact or
validated replacement for the underlying filter.

Veto diagnostics:

- learned OT is sold through speed or differentiability before target status is
  defined;
- training-distribution dependence is vague;
- student residuals are not connected to filtering or HMC quantities.

### Historical R7: DPF-HMC target correctness and banking suitability deepening

Target:

`docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

Purpose:

Make the HMC chapter rigorous enough for reviewers who know Markov chains,
Hamiltonian dynamics, numerical analysis, or statistical mechanics.

Required expansions:

1. Define the scalar target, gradient, Hamiltonian, numerical integrator, and
   Metropolis correction contract.
2. Explain why HMC correctness attaches to a specific scalar target and
   gradient path, not to a method name.
3. Compare exact likelihood HMC, pseudo-marginal methods, noisy-gradient
   methods, relaxed-target HMC, and learned-surrogate HMC.
4. For each DPF rung, state the value object, gradient object, target status,
   assumptions, and required diagnostics.
5. Add a conservative banking suitability discussion:
   structural-model likelihoods, long panels, latent dimensions, parameter
   constraints, determinacy regions, filtering degeneracy, and model-risk
   governance.
6. State explicitly what has not been validated for banking use.
7. Define the minimum experimental evidence required before recommending each
   rung in production research.

Primary criterion:

A skeptical HMC-literate reviewer can see exactly why PF-PF is only the first
serious candidate, why relaxed/learned DPF is surrogate-target HMC, and what
evidence is missing for bank-facing claims.

Veto diagnostics:

- "HMC-ready" is used as a method label;
- pseudo-marginal and differentiable-surrogate arguments are blurred;
- banking suitability is argued from plausibility rather than evidence.

### Historical R8: debugging crosswalk and implementation contract revision

Target:

`docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

Purpose:

Turn the crosswalk into a reviewer- and implementation-facing verification
contract.

Required expansions:

1. Map every major mathematical object to a concrete implementation diagnostic.
2. Add minimal reproducible tests for each layer:
   filtering recursion, EDH endpoint, LEDH local linearization, PF-PF weights,
   Jacobian log determinants, soft-resampling bias, Sinkhorn residuals,
   learned-map residuals, and HMC value-gradient consistency.
3. Add expected failure signatures and what they do, and do not, imply.
4. Add evidence thresholds for promoting a method from exploratory to credible.
5. Link every diagnostic back to chapter equations and literature sources.

Primary criterion:

The crosswalk can serve as an implementation audit checklist for a coding agent
or reviewer.

Veto diagnostics:

- diagnostics remain qualitative;
- tests are not tied to equations;
- implementation advice ignores target-status distinctions.

### Historical R9: literature synthesis and source gap closure

Purpose:

Build a coherent literature spine that a skeptical non-specialist can use
without performing their own literature search.

Tasks:

1. Expand chapter bibliographic discussions from citation lists into source
   contribution paragraphs.
2. For each major source, record:
   - method family;
   - assumptions;
   - main contribution;
   - relevance to BayesFilter;
   - limitation for banking DPF/HMC.
3. Identify missing source coverage that blocks claims.
4. Add or correct bibliography entries only when source identity has been
   checked.

Primary criterion:

The DPF block reads as literature-grounded mathematical exposition, not as a
report with citations attached.

Veto diagnostics:

- citations are added without synthesis;
- source identity or metadata is uncertain;
- local claims exceed the cited source.

### Historical R10: claim boundary and skeptical-reviewer audit

Purpose:

Audit the revised chapters as a skeptical academic panel would.

Tasks:

1. Search for overclaim terms:
   `exact`, `unbiased`, `consistent`, `validated`, `robust`, `HMC-ready`,
   `production`, `optimal`, `guarantee`, `solves`, `proves`.
2. For each occurrence, confirm that assumptions and limitations are local.
3. Check whether every central formula has definitions for all symbols.
4. Check whether every theorem-like or proposition-like claim has either a
   derivation, proof sketch, source theorem, or conservative qualification.
5. Check whether every banking claim is labeled as fact, hypothesis, or needed
   experiment.
6. Record unresolved limitations honestly.

Primary criterion:

The text survives a hostile reading without relying on reader goodwill.

Veto diagnostics:

- bold claims survive because they sound plausible;
- caveats are hidden in later chapters instead of local to the claim;
- limitations are softened to preserve narrative momentum.

### Historical R11: build, reference, and consistency audit

Purpose:

Ensure that the heavily revised material compiles and remains internally
consistent.

Tasks:

1. Compile the document using the repo's established build route.
2. Resolve DPF-local missing labels, duplicate labels, undefined citations, and
   broken references.
3. Check notation consistency across DPF chapters.
4. Check that chapter transitions match the revised dependency ladder.
5. Confirm that tables do not replace necessary derivations.

Primary criterion:

The revised DPF block compiles and is internally navigable.

Veto diagnostics:

- unresolved DPF-local references remain;
- notation changes across chapters without warning;
- compilation succeeds but the chapter argument is inconsistent.

### Historical R12: final reviewer-readiness report

Purpose:

Close the round with an honest account of what has and has not been achieved.

Required final report sections:

1. summary of chapter changes;
2. source-support and source-gap status;
3. derivation-obligation status;
4. claim-boundary audit result;
5. remaining reviewer risks;
6. minimum experiments still required for bank-facing validation;
7. recommended next engineering or writing phase.

Primary criterion:

The final report allows the next agent or human reviewer to know exactly whether
the DPF block is ready, partially ready, or still blocked.

Veto diagnostics:

- the report presents an optimistic closeout not supported by the audits;
- remaining experimental gaps are hidden;
- unresolved mathematical weaknesses are rephrased as future polish.

## Governing execution order

The first execution pass must follow completed evidence setup before chapter
rewrites.  The governing order is:

1. P0 preflight and evidence inventory: completed.
2. P1 skeptical-reader argument map and claim ledger: completed.
3. P2 source-grounding and literature synthesis: completed.
4. P3 classical filtering and SMC baseline expansion: next gating phase.
5. P4/P5 impact reconciliation because P4 and P5 provisional results already
   exist out of the tightened prerequisite order.
6. P6 differentiable resampling and OT/Sinkhorn expansion.
7. P7 learned/amortized OT defensibility expansion.
8. P8 HMC target correctness and banking suitability expansion.
9. P9 debugging crosswalk into verification contract.
10. P10 cross-chapter notation, claim-status, and literature consolidation.
11. P11 mathematical derivation audit.
12. P12 build, PDF review, and hostile-reader audit.
13. P13 final readiness report and reviewer-grade reset memo update.

P3 remains the next gate because P1 identified baseline definitions and
estimator status as prerequisites for later target-status claims.  Existing P4
and P5 work should be preserved, but it cannot unlock P6 until P3 confirms or
repairs the baseline assumptions they depend on.

## Definition of success

This reviewer-grade revision succeeds only if the DPF block becomes something a
skeptical academic reviewer could read without feeling that the central
mathematics and claim boundaries were outsourced to papers they were expected to
find themselves.

It is acceptable for the final document to say that some DPF-for-banking claims
remain unvalidated.  It is not acceptable for the document to imply validation
through polished prose, citation density, or architectural neatness.
