# Phase P0 result: worktree, build, and source-inventory preflight

## Date

2026-05-15

## Governing plan

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`

## Branch and worktree state

Branch:

- `main`

Recent graph:

- `a75e18f (HEAD -> main) Merge branch 'dpf-monograph-rebuild'`
- `30097bd (dpf-monograph-rebuild) Plan reviewer-grade DPF monograph revision`
- `074b90a (origin/main, origin/HEAD) Merge branch 'dpf-monograph-rebuild'`

Remote divergence:

- `main` is ahead of `origin/main` by two commits.

Tracked dirty files:

- `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`
- `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`

Classification:

- These are student-baseline workstream files, not DPF reviewer-grade monograph
  files.  They must not be edited, staged, or reverted by the DPF monograph
  lane.

Untracked DPF reviewer-grade planning files:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-phase-proposal-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p3-smc-baseline-expansion-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p4-edh-ledh-derivation-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p5-pfpf-jacobian-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p6-resampling-ot-sinkhorn-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p7-learned-ot-defensibility-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p8-hmc-banking-target-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p9-debugging-verification-contract-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p11-derivation-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p12-hostile-reader-audit-plan-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p13-final-readiness-plan-2026-05-14.md`

Classification:

- These are DPF reviewer-grade planning artifacts.  They belong to the current
  lane.

Untracked student-baseline files:

- `docs/plans/bayesfilter-student-dpf-baseline-master-program-and-final-subplans-audit-2026-05-13.md`
- `docs/plans/bayesfilter-student-dpf-baseline-mp5-clean-room-implementation-plan-2026-05-13.md`
- `docs/plans/bayesfilter-student-dpf-baseline-mp6-clean-room-fixed-grid-execution-plan-2026-05-13.md`
- `docs/plans/bayesfilter-student-dpf-baseline-mp7-clean-room-comparison-audit-plan-2026-05-13.md`
- `docs/plans/bayesfilter-student-dpf-baseline-mp8-final-archive-and-closeout-plan-2026-05-13.md`

Classification:

- Student-baseline lane.  Do not edit, stage, or revert during DPF monograph
  work.

## Build status

Established build route from prior reset notes:

```text
cd docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Result:

- Build completed successfully.
- Output PDF: `docs/main.pdf`
- Output size: 200 pages.
- `latexmk` reported all targets up to date after reruns.

Build warnings relevant to DPF-local or adjacent review:

- Undefined citations were reported in non-DPF Chapter 35:
  - `kitagawa1996montecarlo`
  - `arulampalam2002tutorial`
- These keys exist in `docs/references.bib`, so the warning appears to be a
  build/rerun or bibliography-state issue rather than missing bibliography
  entries.  It is not DPF-local, but P12 should recheck it.
- DPF-local layout warnings appear in:
  - `docs/chapters/ch19_particle_filters.tex`, lines around 150 and 259--267;
  - `docs/chapters/ch19b_dpf_literature_survey.tex`, lines around 280--283;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`, especially tables around
    lines 267--343;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`, especially
    tables around lines 44--221;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`, especially tables
    around lines 132--283;
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`, especially the longtable
    around lines 42--67.

Interpretation:

- The current DPF block builds, but table readability is already a DPF-local
  presentation risk.  Later phases that add reviewer-grade tables must redesign
  table structure rather than expanding already tight tables.

## DPF chapter structure

Current DPF chapter sizes:

- `docs/chapters/ch19_particle_filters.tex`: 295 lines.
- `docs/chapters/ch19b_dpf_literature_survey.tex`: 382 lines.
- `docs/chapters/ch19c_dpf_implementation_literature.tex`: 252 lines.
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`: 372 lines.
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`: 299 lines.
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`: 324 lines.
- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`: 127 lines.

Section map:

1. `ch19_particle_filters.tex`
   - nonlinear state-space model and filtering recursion;
   - empirical filtering measures;
   - sequential importance sampling;
   - sequential importance resampling and bootstrap filter;
   - bootstrap particle-filter likelihood estimator;
   - degeneracy and effective sample size;
   - chapter boundary.
2. `ch19b_dpf_literature_survey.tex`
   - transport motivation;
   - homotopy path;
   - continuity equation and transport PDE;
   - EDH under Gaussian closure;
   - linear-Gaussian recovery;
   - LEDH and local linearization;
   - stiffness and discretization;
   - chapter boundary.
3. `ch19c_dpf_implementation_literature.tex`
   - why proposal correction is needed;
   - change of variables;
   - proposal-corrected PF-PF weights;
   - EDH/PF and LEDH/PF;
   - Jacobian determinant and log-determinant evolution;
   - correction scope;
   - HMC bridge;
   - chapter boundary.
4. `ch32_diff_resampling_neural_ot.tex`
   - resampling bottleneck;
   - standard resampling as discontinuous map;
   - soft resampling;
   - resampling as transport;
   - entropic OT resampling;
   - bias versus differentiability;
   - implementation debug map;
   - learned OT transition;
   - chapter boundary.
5. `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
   - hierarchy of approximations;
   - teacher and learned maps;
   - residuals and target shift;
   - training-distribution dependence;
   - implementation-facing mathematics;
   - no HMC verdict;
   - chapter boundary.
6. `ch19e_dpf_hmc_target_suitability.tex`
   - DPF target contract for HMC;
   - rung-by-rung target analysis;
   - structured target maps;
   - nonlinear DSGE issue;
   - MacroFinance issue;
   - surrogate-HMC relation;
   - recommendation;
   - chapter boundary.
7. `ch19f_dpf_debugging_crosswalk.tex`
   - how to read a DPF failure;
   - failure-mode crosswalk;
   - diagnostic order;
   - coding-agent issue taxonomy;
   - crosswalk boundary.

## Label and equation inventory

The DPF block has labels for all current chapters and major sections.  Central
equation labels include:

- SMC baseline:
  - `eq:bf-pf-transition-density`
  - `eq:bf-pf-observation-density`
  - `eq:bf-pf-predictive-law`
  - `eq:bf-pf-filtering-law`
  - `eq:bf-pf-predictive-observation`
  - `eq:bf-pf-marginal-factorization`
  - `eq:bf-pf-empirical-measure`
  - `eq:bf-pf-sis-recursion`
  - `eq:bf-pf-bootstrap-likelihood-estimator`
  - `prop:bf-pf-bootstrap-likelihood-status`
  - `eq:bf-pf-ess`
- Particle flow:
  - `eq:bf-pff-log-homotopy`
  - `eq:bf-pff-homotopy-normalizer`
  - `eq:bf-pff-log-homotopy-derivative`
  - `eq:bf-pff-continuity-equation`
  - `eq:bf-pff-log-density-continuity`
  - `eq:bf-pff-flow-pde`
  - `eq:bf-pff-homotopy-covariance`
  - `eq:bf-pff-homotopy-cov-derivative`
  - `eq:bf-pff-edh-A`
  - `eq:bf-pff-edh-b`
  - `eq:bf-pff-kalman-endpoint`
  - `eq:bf-pff-local-information-vector`
  - `eq:bf-pff-ledh-A`
  - `eq:bf-pff-ledh-b`
- PF-PF:
  - `eq:bf-pfpf-flow-map`
  - `eq:bf-pfpf-postflow-density`
  - `eq:bf-pfpf-target-density`
  - `eq:bf-pfpf-weight`
  - `eq:bf-pfpf-jacobian-ode`
  - `eq:bf-pfpf-logdet-ode`
  - `eq:bf-pfpf-logdet-affine`
- Resampling and OT:
  - `eq:bf-dr-weighted-empirical`
  - `eq:bf-dr-equal-weight-empirical`
  - `eq:bf-dr-categorical-selection`
  - `eq:bf-dr-soft-resampling`
  - `eq:bf-dr-soft-test-bias`
  - `eq:bf-dr-unregularized-ot`
  - `eq:bf-dr-eot-primal`
  - `eq:bf-dr-sinkhorn-form`
  - `eq:bf-dr-barycentric-map`
- Learned OT:
  - `eq:bf-learned-ot-teacher-map`
  - `eq:bf-learned-ot-student-map`
  - `eq:bf-learned-ot-training-objective`
  - `eq:bf-learned-ot-equivariance`
  - `eq:bf-learned-ot-residual`
- HMC:
  - `eq:bf-dpf-hmc-contract`

Initial derivation-obligation candidates:

- bootstrap likelihood-estimator proof;
- homotopy derivative and normalizer role;
- continuity/log-density transport;
- EDH covariance and affine coefficient derivation;
- linear-Gaussian mean and covariance recovery;
- LEDH local information vector;
- PF-PF post-flow density and corrected weight;
- Jacobian ODE and log-determinant trace identity;
- soft-resampling test-function bias;
- EOT scaling/Sinkhorn form;
- barycentric projection;
- learned-map residual hierarchy;
- permutation equivariance;
- HMC value-gradient contract and rung classification.

## Citation and bibliography inventory

Current DPF citation keys include:

- `gordon1993novel`
- `doucet2001sequential`
- `andrieu2010particle`
- `bengtsson2008curse`
- `snyder2008obstacles`
- `daumhuang2008`
- `li2017particle`
- `hu2021particle`
- `zhumurphyjonschkowski2020`
- `corenflos2021differentiable`
- `villani2003topics`
- `reich2013nonparametric`
- `cuturi2013sinkhorn`
- `peyre2019computational`
- `schmitzer2019stabilized`
- `zaheer2017deep`
- `lee2019set`
- `neal2011mcmc`
- `andrieu2009pseudo`
- `betancourt2017conceptual`
- `greydanus2019hamiltonian`

All of these keys appear in `docs/references.bib`.

## ResearchAssistant status

ResearchAssistant MCP status:

- workspace root: `/home/ubuntu/python/ResearchAssistant`;
- mode: read-only;
- offline mode: enabled;
- providers disabled;
- no hosted service;
- no default workflow sends papers or notes to external providers.

Queries run:

- `sequential Monte Carlo particle filter`;
- `particle flow EDH LEDH particle filter`;
- `optimal transport Sinkhorn differentiable resampling particle filter`;
- `Hamiltonian Monte Carlo pseudo marginal particle MCMC`;
- `Deep Sets Set Transformer permutation equivariant`.

Result:

- no local paper summaries or review records were returned.

Interpretation:

- The current execution cannot treat ResearchAssistant as containing reviewed
  local DPF source records.
- P2 must build a source-role register from `docs/references.bib`, existing
  chapter citations, local PDFs where present, and any available local source
  material.  It must mark ResearchAssistant source coverage as absent rather
  than silently assuming paper-level review.

## MathDevMCP status

MathDevMCP is available.  Tool matrix reports support for:

- LaTeX search and context extraction;
- document/code consistency comparisons;
- derivation-backed claim checks;
- bounded proof obligations;
- document-grounded implementation briefs.

MathDevMCP searches successfully located:

- EDH/homotopy labels in `ch19b_dpf_literature_survey.tex`;
- PF-PF corrected weight and Jacobian labels in
  `ch19c_dpf_implementation_literature.tex`.

Interpretation:

- MathDevMCP can support P1/P11 label-local audits and obligation extraction.
- It does not replace manual mathematical review for probabilistic or
  source-assumption claims.

## Overclaim and reviewer-risk signals

High-risk terms already present include:

- `exact`;
- `unbiased`;
- `correct`;
- `valid`;
- `HMC-ready`;
- `production-ready`;
- `first serious`;
- `guarantee`;
- `validated`.

Important occurrences requiring P1 classification:

- bootstrap likelihood estimator described as unbiased;
- homotopy and continuity equations described as exact;
- EDH exactness in linear-Gaussian/Gaussian-closure regimes;
- PF-PF described as corrected and as a first serious DPF-HMC candidate;
- EOT and unregularized OT described as exact for their own chosen objectives;
- learned OT residuals and guarantees;
- HMC target correctness and production-readiness caveats;
- nonlinear DSGE and MacroFinance suitability statements.

## P0 audit result

P0 primary criterion:

- The next phase has a concrete map of chapter weaknesses, source coverage, and
  derivation obligations.

Status:

- Passed with cautions.

No P0 veto diagnostic fired:

- branch state was recorded;
- dirty files were classified;
- source availability was checked rather than assumed;
- build status was tested;
- DPF labels, citations, tables, and overclaim signals were inventoried;
- MathDevMCP availability was confirmed.

Cautions carried forward:

1. ResearchAssistant has no local DPF paper records in the current searchable
   workspace.
2. DPF-local table readability is already weak in several chapters.
3. Existing source support is bibliography/chapter-spine support, not
   ResearchAssistant-reviewed paper support.
4. P1 must classify claims before chapter rewriting begins.

## Exit decision

Proceed to P1.

