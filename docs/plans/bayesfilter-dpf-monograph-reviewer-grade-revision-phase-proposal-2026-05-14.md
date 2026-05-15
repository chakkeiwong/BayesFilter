# Proposed phases for reviewer-grade DPF monograph revision

## Date

2026-05-14

## Governing program

This proposal operationalizes:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`

## Execution-safety amendment

This proposal uses the governing P0-P13 taxonomy.  Any older R-phase language
is historical only and must be mapped through the master program before use.

Completed setup artifacts:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-result-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`

Current execution consequence:

- P3 is the next gating phase.
- P4 and P5 provisional result artifacts dated `2026-05-15` are present in the
  worktree.  Preserve them as in-lane work, but reconcile them after P3 before
  starting P6.
- `docs/chapters/ch32_diff_resampling_neural_ot.tex` may already contain
  partial in-lane P6 edits without a P6 result artifact.  P6 must begin by
  auditing and adopting, repairing, or superseding that diff.
- P2 established that ResearchAssistant has no local DPF paper summaries in the
  current read-only/offline workspace.  Later phases must treat source support
  as bibliography-spine plus local derivation unless a later artifact records
  reviewed source intake.
- Every chapter-edit phase must repeat branch/worktree classification, keep
  student-baseline files out of lane, record the allowed write set, record
  MathDevMCP/manual derivation-obligation evidence, and run the established
  LaTeX build unless the phase result records a bounded reason it could not.

It is based on a direct pass over the current DPF chapter block:

- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch19b_dpf_literature_survey.tex`
- `docs/chapters/ch19c_dpf_implementation_literature.tex`
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

## Current-state diagnosis

The current document is organized correctly but is not yet written at the
standard required for a skeptical academic review panel.  The chapters do many
right things: they separate particle filtering, particle flow, PF-PF,
differentiable resampling, learned OT, HMC target analysis, and debugging.  They
also contain useful cautionary language around exactness, approximation, and
target drift.

The remaining weakness is not chapter architecture.  It is that many
reviewer-critical arguments are still too compressed.  Several central formulas
appear as endpoint expressions, proof sketches, or interpretive tables where a
skeptical mathematical reader would expect derivations, assumptions, and claim
boundaries.

The phase plan below therefore treats the next round as a deepening and
defensibility round, not a structural rewrite.

## Evidence from the current document

Current DPF chapter sizes:

- `ch19_particle_filters.tex`: 295 lines.
- `ch19b_dpf_literature_survey.tex`: 382 lines.
- `ch19c_dpf_implementation_literature.tex`: 252 lines.
- `ch32_diff_resampling_neural_ot.tex`: 372 lines.
- `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`: 299 lines.
- `ch19e_dpf_hmc_target_suitability.tex`: 324 lines.
- `ch19f_dpf_debugging_crosswalk.tex`: 127 lines.

This is not enough space for a self-contained reviewer-grade treatment of the
full DPF ladder.  The most urgent gaps are qualitative:

1. `ch19b` states EDH/LEDH formulas but does not yet walk the reader through a
   full enough derivation from the homotopy, Gaussian closure, affine flow, and
   linear-Gaussian special case.
2. `ch19c` gives the PF-PF correction and log-determinant identities, but a
   reviewer would still want a more careful target/proposal construction,
   trajectory-level notation, and sign-convention audit.
3. `ch32` has the right OT/Sinkhorn layers, but needs more optimization detail,
   dual/scaling derivation, coupling/barycentric interpretation, and
   numerical-analysis caveats.
4. `ch19e` is strong conceptually but still too compact for HMC-literate
   reviewers: it needs a fuller HMC correctness contract, pseudo-marginal
   comparison, noisy-gradient distinction, and banking-model governance
   limitations.
5. The literature synthesis is still mostly citation-spine plus brief
   contribution statements.  It is not yet a self-contained survey for readers
   who will not search the papers themselves.
6. The debugging crosswalk is useful, but it is too short to be an audit
   contract.  It should become an equation-indexed verification plan.

## Proposed phase sequence

### Phase P0: worktree, build, and source-inventory preflight

Purpose:

Establish a clean baseline before any writing pass.

Tasks:

1. Verify branch, dirty files, and divergence from remote.
2. Classify unrelated student-baseline dirty files and keep them out of the DPF
   lane.
3. Compile or at least run the repo's established LaTeX build check if
   available.
4. Record DPF-local undefined references/citations, duplicate labels, and
   overfull table risks.
5. Query ResearchAssistant for local paper summaries, review items, and
   source-code links relevant to SMC, particle flow, PF-PF, OT/Sinkhorn, learned
   OT, HMC, DSGE, and MacroFinance.
6. Use MathDevMCP or direct LaTeX search to build a label/equation inventory for
   every DPF chapter.

Deliverable:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-result-2026-05-15.md`

Acceptance gate:

- No chapter editing begins until source availability, build status, and
  dirty-worktree scope are recorded.

### Phase P1: skeptical-reader argument map and claim ledger

Purpose:

Make the whole DPF block read as a cumulative argument rather than a sequence of
method summaries.

Tasks:

1. Build a claim ledger with one row per major claim:
   exact identity, unbiased estimator, approximation, relaxation, learned
   surrogate, implementation hypothesis, or banking hypothesis.
2. Build a chapter dependency map:
   classical filtering -> SMC likelihood estimator -> flow homotopy ->
   EDH/LEDH closure -> PF-PF correction -> resampling relaxation ->
   Sinkhorn/EOT -> learned OT -> HMC target -> banking suitability.
3. Identify where a later chapter relies on a definition or assumption not yet
   established earlier.
4. Decide whether chapter names/filenames should be cleaned up later or left
   unchanged for now.

Deliverable:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`

Acceptance gate:

- Every later rewrite phase must know which claims it is allowed to strengthen,
  weaken, or leave as hypotheses.

### Phase P2: source-grounding and literature synthesis pass

Purpose:

Prevent citation padding and give skeptical non-specialists enough literature
context inside the document.

Tasks:

1. For each source currently cited in the DPF chapters, write a source-role
   note: what it contributes, assumptions, relevance, and limits.
2. Check source identity for core references in `docs/references.bib`.
3. Identify missing literature for:
   - particle-flow derivations and variants;
   - PF-PF proposal correction;
   - degeneracy and high-dimensional particle filtering;
   - differentiable resampling;
   - OT, EOT, Sinkhorn, and stabilization;
   - set/permutation architectures for learned maps;
   - HMC, pseudo-marginal methods, noisy gradients, and surrogate HMC.
4. Draft chapter-local literature paragraphs that summarize contributions rather
   than merely list citations.

Deliverable:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`

Acceptance gate:

- No major claim may remain supported only by a bare citation.

### Phase P3: classical filtering and SMC baseline expansion

Target:

- `docs/chapters/ch19_particle_filters.tex`

Purpose:

Make the baseline exact object and bootstrap likelihood estimator sufficiently
clear before DPF modifications begin.

Required revisions:

1. Add a notation/object table for state, observations, parameter, filtering
   law, predictive law, likelihood factor, proposal, empirical measure, and
   likelihood estimator.
2. Expand the filtering recursion derivation from Bayes' rule.
3. Expand sequential-importance weights with trajectory notation and then
   specialize to the bootstrap filter.
4. Strengthen the likelihood-estimator proposition:
   assumptions, filtration, conditional-expectation steps, and precise meaning
   of unbiasedness.
5. Add a short discussion distinguishing unbiased likelihood estimation from
   pathwise differentiability and from unbiased score estimation.
6. Add an "audit table" separating exact identities, Monte Carlo estimator
   statements, asymptotic/variance statements, and implementation diagnostics.

Acceptance gate:

- A non-specialist reviewer can say exactly what the true likelihood is, what
  the bootstrap estimator estimates, and why the differentiable variants must be
  compared to that baseline.

### Phase P4: EDH/LEDH derivation expansion

Target:

- `docs/chapters/ch19b_dpf_literature_survey.tex`

Purpose:

Make particle flow derivation-led rather than formula-led.

Required revisions:

1. Add a notation/object table for homotopy density, normalizer, velocity field,
   flow map, pseudo-time, Gaussian closure, affine coefficients, and local
   linearization.
2. Derive the continuity equation and log-density transport equation more
   explicitly, including regularity assumptions and non-uniqueness of the
   velocity field.
3. Derive EDH from Gaussian homotopy precision and affine velocity matching,
   rather than only stating the final $A(\lambda),b(\lambda)$ expressions.
4. Work through the linear-Gaussian recovery case in enough detail to recover
   the Kalman mean and covariance endpoints.
5. Derive LEDH local information vector and local precision carefully from the
   observation linearization.
6. Add a reviewer-facing section on exactness:
   homotopy endpoint exactness, continuity equation exactness, Gaussian-closure
   approximation, local-linearization approximation, and numerical integration
   error.
7. Add source-synthesis paragraphs for Daum-Huang, particle-flow variants, and
   PF-PF connections.

Acceptance gate:

- EDH and LEDH no longer appear as named algorithms with formulas; they appear
  as consequences of stated density paths, approximations, and velocity-field
  choices.

### Phase P5: PF-PF proposal correction and Jacobian audit

Target:

- `docs/chapters/ch19c_dpf_implementation_literature.tex`

Purpose:

Make PF-PF defensible as a proposal-corrected importance construction.

Required revisions:

1. Define the one-step and trajectory-level target/proposal objects, including
   conditioning on ancestors and observations.
2. Derive post-flow proposal density from change of variables step by step.
3. Derive corrected weights for prior proposal, generic proposal, EDH/PF, and
   LEDH/PF cases.
4. Expand the log-determinant derivation:
   Jacobian ODE, Jacobi formula, trace identity, affine simplification, sign
   convention, and forward/inverse map convention.
5. Add a finite-particle likelihood-estimator status discussion: what is
   corrected, what is unbiased or not claimed, and what remains numerical.
6. Add implementation-audit tests:
   affine map density check, determinant finite-difference check, corrected
   weight normalization, seed-fixed value/gradient consistency.

Acceptance gate:

- A reviewer can audit PF-PF without accepting "flow improves particles" as a
  mathematical argument.

### Phase P6: differentiable resampling and OT/Sinkhorn expansion

Target:

- `docs/chapters/ch32_diff_resampling_neural_ot.tex`

Purpose:

Make the price of differentiability mathematically visible.

Required revisions:

1. Add notation/object table for weighted empirical measure, equal-weight
   measure, categorical ancestor map, soft map, coupling, OT objective, EOT
   objective, dual/scaling variables, barycentric projection, and finite solver.
2. Expand hard-resampling discontinuity with a simple one-dimensional example.
3. Expand soft-resampling bias derivation beyond first-order notation and
   clarify which test functions are preserved.
4. Derive the OT coupling constraints and explain why this is not categorical
   resampling.
5. Derive EOT optimality conditions leading to Sinkhorn scaling, including
   marginal constraints and entropy convention.
6. Clarify barycentric projection as a deterministic map induced by a coupling,
   not the coupling itself.
7. Add numerical-analysis subsection:
   regularization limit, kernel underflow, log-domain stabilization,
   differentiation through finite iterations versus implicit differentiation,
   memory/runtime scaling.
8. Add a claim-status table covering categorical, OT, EOT, finite Sinkhorn, and
   autodiff-through-solver objects.

Acceptance gate:

- The chapter prevents a reviewer from concluding that differentiable resampling
  is simply classical resampling made smooth.

### Phase P7: learned/amortized OT defensibility expansion

Target:

- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

Purpose:

Make learned OT read as an approximation hierarchy with explicit evidence
requirements.

Required revisions:

1. Define teacher variants:
   unregularized OT, EOT, finite Sinkhorn, and barycentric map.
2. Define student map, loss, training distribution, permutation equivariance,
   particle-count assumptions, and state-dimension assumptions.
3. Expand residual hierarchy:
   teacher-object error, solver error, learning error, distribution shift,
   filtering-summary error, posterior error, HMC value-gradient error.
4. Add a source-grounded architecture discussion: DeepSets, attention/set
   transformers, and why symmetry is necessary but not sufficient.
5. Add examples of failure regimes:
   sharp weights, high dimension, shifted particle count, shifted epsilon,
   nonlinear structural state regimes, and out-of-distribution clouds.
6. Add an evidence ladder for learned OT in banking:
   teacher agreement, stress tests, posterior comparison, gradient checks,
   governance limits.

Acceptance gate:

- Learned OT is impossible to read as a speed trick that preserves target
  correctness automatically.

### Phase P8: HMC target correctness and banking-review expansion

Target:

- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

Purpose:

Make the HMC/banking chapter credible to reviewers trained in Hamiltonian
dynamics, MCMC, numerical analysis, or scientific computing.

Required revisions:

1. Add a compact but explicit HMC target contract:
   target density, potential, Hamiltonian, leapfrog/integrator, Metropolis
   correction, and value-gradient consistency.
2. Separate exact HMC, pseudo-marginal MCMC, noisy-gradient methods,
   delayed-acceptance/surrogate methods, relaxed-target HMC, and learned-target
   HMC.
3. For every DPF rung, state:
   scalar value, gradient path, target status, assumptions, possible correction,
   diagnostics, and what remains unvalidated.
4. Replace "first serious candidate" style phrases with reviewer-safe language:
   "first rung with a defensible proposal-correction interpretation, pending
   finite-particle and numerical validation."
5. Expand DSGE and MacroFinance sections as skeptical-reviewer limitations:
   determinacy, pruning, invalid regions, latent dimension, long panels,
   missing data, volatility curvature, model-risk governance, and auditability.
6. Add a minimum evidence table for banking research use versus production use.

Acceptance gate:

- The chapter cannot be read as claiming that DPF-HMC for banking is validated.
  It must define exactly what evidence would be needed.

### Phase P9: debugging crosswalk into verification contract

Target:

- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

Purpose:

Turn the short crosswalk into a reviewer-visible implementation audit plan.

Required revisions:

1. Add an equation-to-test matrix linking each load-bearing equation to a
   diagnostic test.
2. Specify minimal controlled examples:
   linear-Gaussian recovery, affine-flow determinant, nonlinear scalar
   observation, soft-resampling bias, Sinkhorn marginal residual, learned-map
   permutation test, HMC finite-difference gradient test.
3. Add failure interpretation rules:
   what each failed test means and what it does not prove.
4. Add promotion thresholds for exploratory, credible research, and
   bank-facing claims.
5. Add route from every diagnostic to source literature and chapter sections.

Acceptance gate:

- A coding agent could use the chapter as a concrete verification checklist, not
  merely as a conceptual routing table.

### Phase P10: cross-chapter notation, claim-status, and literature consolidation

Purpose:

Make the revised block read as one monograph argument.

Tasks:

1. Normalize notation for particles, weights, proposals, flow maps, couplings,
   likelihood estimates, scalar HMC targets, gradients, and approximation
   statuses.
2. Insert or harmonize claim-status tables across chapters.
3. Ensure every chapter has a skeptical-reader limitations section.
4. Ensure source-synthesis paragraphs do not conflict across chapters.
5. Check that chapter transitions do not smuggle in unproved claims.

Deliverable:

- DPF-local notation and claim-status audit note.

Acceptance gate:

- A reader moving across chapters should not have to infer whether a symbol or
  claim-status word has changed meaning.

### Phase P11: mathematical audit with MathDevMCP and manual obligation checks

Purpose:

Audit the load-bearing formulas after the writing passes.

Tasks:

1. Build a derivation-obligation register for:
   filtering recursion, SIS weights, likelihood estimator, homotopy derivative,
   continuity equation, EDH coefficients, LEDH linearization, PF-PF weights,
   log-determinant ODE, soft-resampling bias, EOT scaling, barycentric map,
   learned residual hierarchy, and HMC value-gradient contract.
2. Use MathDevMCP where labels and bounded obligations are suitable.
3. Record which obligations were checked mechanically, checked manually, or
   conservatively left as source-supported but not verified.
4. Downgrade or qualify any claim whose derivation does not survive audit.

Deliverable:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-derivation-audit-{YYYY-MM-DD}.md`

Acceptance gate:

- No central equation remains as "obvious" if it is load-bearing for target
  interpretation or implementation.

### Phase P12: build, PDF review, and hostile-reader audit

Purpose:

Verify the revised document as a document, not only as source files.

Tasks:

1. Compile the PDF.
2. Review DPF chapters in PDF order, including tables and page breaks.
3. Search for overclaim terms:
   exact, unbiased, consistent, validated, robust, HMC-ready, production,
   optimal, guarantee, solves, proves, suitable, credible, first serious.
4. For each hit, confirm local assumptions and limits.
5. Check that tables do not replace derivations.
6. Check that citations are synthesized and not ornamental.

Deliverable:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-hostile-reader-audit-{YYYY-MM-DD}.md`

Acceptance gate:

- The block can survive a skeptical read without relying on reader goodwill or
  outside paper searches.

### Phase P13: final readiness report and reset memo update

Purpose:

Close the round honestly.

Tasks:

1. Summarize chapter changes.
2. Report source support and source gaps.
3. Report derivation audit status.
4. Report build/PDF status.
5. Report remaining reviewer risks.
6. Separate writing-complete items from experiment-required items.
7. Update the DPF reset memo or create a reviewer-grade reset memo.

Deliverable:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-final-readiness-report-{YYYY-MM-DD}.md`

Acceptance gate:

- The final report must be willing to say "not ready" for any claim or chapter
  that does not meet the skeptical-reviewer standard.

## Recommended execution order

The execution order should be:

1. P0 preflight.
2. P1 claim ledger.
3. P2 source grounding.
4. P3 SMC baseline.
5. P4/P5 impact reconciliation if provisional P4/P5 results remain present.
6. P6 resampling and OT/Sinkhorn, beginning with any existing `ch32` partial
   diff audit.
7. P7 learned OT.
8. P8 HMC and banking suitability.
9. P9 debugging crosswalk.
10. P10 consolidation.
11. P11 derivation audit.
12. P12 hostile-reader PDF audit.
13. P13 final readiness report.

P3 now precedes the high-risk DPF chapter rewrites because P1 identified exact
filtering targets, estimator status, and differentiability distinctions as
prerequisites for later target-status claims.

## Stop conditions

Stop and ask for direction if any of the following occur:

1. Source support for a central claim cannot be established.
2. A derivation cannot be completed without changing the claim.
3. A chapter would need to become too long for the current book structure,
   requiring a split or appendix.
4. The build system is broken in a way unrelated to the DPF lane.
5. Unrelated dirty worktree files block safe editing.

## Practical guidance for the next agent

The next agent should not start by rewriting prose.  It should start with P0
and P1, then rewrite the highest-risk mathematical chapters with the claim
ledger open.  The writing standard is not "clear summary."  It is "a skeptical
mathematically mature reviewer can audit the argument locally."
