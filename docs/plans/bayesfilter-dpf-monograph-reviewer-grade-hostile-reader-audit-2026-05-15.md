# Phase P12 result: build, PDF review, and hostile-reader audit

## Date

2026-05-15

## Governing inputs

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p12-hostile-reader-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-result-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-derivation-audit-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane DPF chapter files already modified before P12:
  - `docs/chapters/ch19_particle_filters.tex`;
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`;
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`.
- In-lane reviewer-grade plan/result artifacts remain untracked.
- Out-of-lane dirty/untracked student-baseline and controlled-baseline files
  remain present and were not edited, staged, reverted, or used as monograph
  evidence.

## Allowed write set used

- This P12 hostile-reader audit artifact only.

No chapter file, student-baseline file, controlled-baseline file, git history
operation, deletion, push, merge, rebase, reset, restore, or staging operation
was performed in P12.

## Build and PDF status

Build command:

```text
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Result:

- Passed.
- `latexmk` reported all targets up to date.
- PDF: `docs/main.pdf`.
- PDF pages: 218.

Targeted log scan:

- No undefined DPF-local citations found.
- No undefined references found by the targeted scan.
- No rerun warnings found by the targeted scan.
- No multiply defined label warnings found by the targeted scan.

PDF review inputs:

- `pdfinfo docs/main.pdf`.
- `pdftotext -layout` on DPF pages around Chapters 20--26.
- Rendered page review with `pdftoppm` output under `/tmp` for pages covering
  resampling/OT, learned OT, HMC, and debugging-contract tables.
- Visual spot checks included representative pages in the DPF block, including
  pages 131, 135, 141, 145, 151, 161, and 166.

## PDF layout finding

The DPF block is readable but dense.  The main presentation weakness is table
pressure and long running headers:

- Chapter 20 notation/status and baseline-audit tables create underfull boxes.
- Chapter 23 comparison/debug tables are dense but readable in the rendered PDF.
- Chapter 24 starts with a dense object table but remains legible.
- Chapter 25 target-status and banking-evidence tables create table pressure.
- Chapter 26 equation-to-test, diagnostic-contract, and promotion-threshold
  tables create the most visible layout pressure.
- Long chapter titles create overfull running headers, especially for Chapters
  24--26.

This is a layout caution, not a P12 veto.  The rendered pages inspected are not
unreadable, the tables do not replace the local derivations, and the text around
the tables states the claim boundaries.

## Hostile-reader audit by chapter

| Chapter | Problem statement | Object inventory | Assumptions | Derivations | Source synthesis | Claim boundaries | Implementation diagnostics | Result |
|---|---|---|---|---|---|---|---|---|
| Chapter 20, Particle-Filter Foundations | Clear: define the classical SMC reference before DPF relaxations. | Passed: state, observation, parameter, proposal, weights, ancestors, empirical law, and likelihood estimator are named. | Passed: dominated kernels, support, finite integrals, conditionally unbiased resampling. | Passed: filtering recursion, SIS ratio, bootstrap likelihood-estimator status. | Passed with P2 caveat: bibliography-spine only, not RA-reviewed. | Passed: likelihood unbiasedness is not log-likelihood, score, pathwise-gradient, or HMC readiness. | Passed: ESS, ancestor count, Kalman-reference and finite-difference checks. | Pass. |
| Chapter 21, Particle-Flow Foundations | Clear: build homotopy/flow mathematics before PF-PF correction. | Passed: density path, normalizer, velocity field, flow map, EDH/LEDH objects. | Passed: positivity, finite normalizer, differentiability, boundary flux, Gaussian/local-linear closure. | Passed: homotopy derivative, continuity equation, EDH covariance/mean, linear-Gaussian recovery, LEDH local information vector. | Passed with P2 caveat. | Passed: raw EDH/LEDH exactness restricted to stated regimes. | Passed: stiffness, pseudo-time resolution, Jacobian and endpoint diagnostics. | Pass. |
| Chapter 22, PF-PF and Proposal Correction | Clear: explain why flow must be treated as proposal transformation. | Passed: pre-flow/post-flow particles, proposal density, flow map, inverse, Jacobian, target density, corrected weight. | Passed: differentiable bijection, nonzero determinant, finite densities, ancestor conditioning. | Passed: change of variables, corrected weight, Jacobian ODE, log-det trace identity. | Passed with P2 caveat. | Passed: proposal correction is not exact nonlinear filtering, not finite-particle validation, and not HMC readiness. | Passed: affine-flow, determinant-sign, corrected-weight, and path-consistency tests. | Pass. |
| Chapter 23, Differentiable Resampling and OT | Clear: identify the resampling bottleneck and distinguish categorical, soft, OT, EOT, and finite Sinkhorn objects. | Passed: empirical measures, ancestors, couplings, costs, entropy, scalings, barycentric output, finite plan. | Passed: normalized weights, positive marginals for EOT, finite costs, epsilon, solver budget. | Passed: categorical discontinuity, soft-resampling bias, OT feasible set, EOT KKT/scaling, Sinkhorn residuals, barycentric dimensions. | Passed with P2 caveat. | Passed: relaxed/transport objects are not categorical resampling or original posterior claims. | Passed: residuals, epsilon sensitivity, stabilization, unrolled/implicit gradient route, runtime/memory. | Pass with layout caution. |
| Chapter 24, Learned Transport Operators | Clear: learned OT is a student map trained against an already relaxed teacher. | Passed: categorical baseline, OT/EOT/finite teachers, barycentric teacher, student, training distribution, residuals. | Passed: teacher variant, training envelope, permutation action, residual metric. | Passed: teacher/student definitions, residual hierarchy, equivariance/invariance equations. | Passed with P2 caveat. | Passed: symmetry, speed, and low residual are not posterior or HMC correctness. | Passed: teacher residual, student residual, OOD stress, posterior comparison, compiled repetition. | Pass with layout caution. |
| Chapter 25, HMC Target Correctness | Clear: decide what scalar HMC targets before any banking claim. | Passed: transformed coordinates, scalar log target, potential, momentum, Hamiltonian, integrator, gradient, accept/reject value, target status. | Passed: differentiability on valid region, same-coordinate gradient, proposal correction or target status required. | Passed: value-gradient contract, target-status taxonomy, pseudo-marginal versus surrogate distinction. | Passed with P2 caveat. | Passed: no nonlinear DSGE/MacroFinance validation and no production or bank-governance approval. | Passed: same-scalar gradient checks, invalid-region policy, posterior sensitivity, compiled repeatability. | Pass with layout caution. |
| Chapter 26, Debugging Contract | Clear: convert mathematics into equation-indexed diagnostics. | Passed: equations, examples, implementation quantities, tolerances, failure interpretations. | Passed: diagnostics require declared inputs, tolerances, seed/solver policy, scalar target. | Passed as a test contract rather than a mathematical proof. | Passed with P2 caveat. | Passed: diagnostics do not validate beyond the equation/local claim tested. | Passed: promotion thresholds distinguish exploratory, credible research, bank-facing research, and production/governance claims. | Pass with layout caution. |

## Overclaim audit

Terms searched across the DPF chapter block:

- `exact`;
- `unbiased`;
- `consistent`;
- `validated`;
- `robust`;
- `HMC-ready`;
- `production`;
- `optimal`;
- `guarantee`;
- `solves`;
- `proves`;
- `suitable`;
- `credible`;
- `first serious`.

Disposition:

- `exact` is bounded to model identities, named mathematical objects, or
  recovery/special cases.  Relaxed, learned, finite-solver, and banking claims
  are not promoted to exactness.
- `unbiased` is restricted to likelihood-estimator status and conditionally
  unbiased resampling, with explicit denials for log likelihood, score, and
  pathwise gradients.
- `consistent` appears in value-gradient, sign-convention, or diagnostic
  consistency contexts, not as a blanket consistency theorem for DPF.
- `validated` appears only in explicit non-claims or validation-burden language.
- `robust` is not used to promote DPF correctness beyond evidence.
- `HMC-ready` appears only in negative or boundary statements.
- `production` and governance language appears only as evidence-gated claim
  categories or explicit non-claims.
- `optimal` is used for optimal-transport terminology and optimizer objects,
  not as a generic method-quality claim.
- `guarantee` appears in denied or qualified contexts.
- `solves` and `proves` appear in local mathematical-object language or
  explicit denials of overclaiming.
- `suitable` appears in suitability-audit and disallowed-shortcut contexts.
- `credible` is restricted to a research-implementation evidence level or to a
  necessary-but-not-sufficient diagnostic statement.
- `first serious` is absent from the chapter block as a promotion phrase; it has
  been replaced by weaker proposal-correction language.

No overclaim veto fired.

## Source-support status

P12 confirms the P2 source boundary:

- The DPF source spine is present in `docs/references.bib` and cited in the
  chapters.
- ResearchAssistant did not provide local reviewed paper summaries for the DPF
  source families in earlier phases.
- The chapters use citations as provenance and literature positioning, while
  load-bearing claims are locally derived or explicitly downgraded.
- No chapter text claims ResearchAssistant-reviewed source support.

## Tables versus derivations

Tables summarize but do not replace the load-bearing arguments:

- Chapter 20 derives recursion and likelihood-estimator status before the audit
  table.
- Chapter 21 derives homotopy, continuity, EDH/LEDH, and recovery statements
  before the exactness ledger.
- Chapter 22 derives proposal densities and log-det equations before the audit
  table.
- Chapter 23 derives categorical discontinuity, soft bias, OT/EOT, Sinkhorn,
  and barycentric projection before comparison/debug tables.
- Chapter 24 defines teacher/student maps and residuals before evidence tables.
- Chapter 25 states the HMC scalar/gradient contract before target and banking
  ladders.
- Chapter 26 is intentionally a verification contract; its tables are the object
  of the chapter rather than substitutes for previous derivations.

## Veto diagnostic review

- PDF cannot be built: no.
- DPF-local references or citations unresolved: no targeted unresolved matches.
- Overclaim terms lack local assumptions or limitations: no.
- Derivation gaps survive because citations exist: no; P11 recorded 20 manual
  derivation audits.
- Banking suitability reads stronger than evidence supports: no.
- Source-support language conflicts with P2: no.
- Student-baseline files edited, staged, or treated as monograph evidence: no.
- Tables have become substitutes for derivations: no.
- Non-specialist mathematical reviewer must search papers for load-bearing
  derivation steps: no for the central obligations audited in P11; yes only for
  deeper literature provenance beyond this monograph's bibliography-spine
  support, which is already recorded as a source-support limitation.

## Exit gate

Passed with layout caution.

The revised DPF block can proceed to P13 final readiness.  The remaining P12
risk is presentational density in long tables and running headers, not a
mathematical, source-support, or banking-overclaim veto.

## Next recommended phase

Proceed to P13:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p13-final-readiness-plan-2026-05-14.md`.
