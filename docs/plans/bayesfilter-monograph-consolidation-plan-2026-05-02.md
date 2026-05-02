# Plan: BayesFilter Monograph Consolidation

## Date
2026-05-02

## Purpose
Create a dedicated BayesFilter monograph under `~/BayesFilter/docs` that
consolidates the filtering, HMC, analytic derivative, numerical-stability, and
industrial-scale model work now scattered across:

- `~/python/docs/monograph.tex`
- `~/python/docs/chapters/*.tex`
- `~/latex/CIP_monograph/main.tex`
- `~/latex/CIP_monograph/chapters/*.tex`
- `~/MacroFinance/analytic_kalman_derivatives.tex`
- related implementation notes, tests, and reset memos in `~/MacroFinance` and
  `~/python/docs/plans`

This is a documentation architecture plan, not yet a content migration. The
goal is to prevent the BayesFilter project from inheriting duplicated,
inconsistent, or project-specific chapters without a governing structure.

## Motivation
The SVD sigma-point filter debugging work exposed a strategic problem: HMC and
filtering infrastructure is now a shared industrial asset, not a side component
inside either `dsge_hmc` or `MacroFinance`.

The documents currently mix several layers:

- DSGE model theory and perturbation solvers
- CIP and affine macro-finance model construction
- Kalman, square-root, SVD, sigma-point, and particle filtering
- HMC/NUTS, transport maps, NeuTra, and surrogate geometry
- analytic Kalman score/Hessian recursions
- numerical diagnostics, XLA constraints, and code-level stability gates

BayesFilter should own the generic filtering and HMC-safe likelihood layer.
The monograph should therefore explain the reusable numerical technology
independently from NK, EZ, SGU, CIP, AFNS, or future NAWM applications.

## Tooling Policy
Use the new local tools during planning and audit:

- `MathDevMCP`
  - Codex MCP server name: `mathdevmcp`
  - CLI fallback:
    `PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli ...`
  - Use for LaTeX label search, derivation audit, code-document consistency,
    shape/notation checks, and numeric diagnostic suggestions.

- `ResearchAssistant`
  - CLI:
    `ra --root WORKSPACE ...`
  - Use for literature ingest, source-paper inspection, citation neighborhoods,
    local paper summaries, and claim-support audits.
  - Current local query for `"Hamiltonian Monte Carlo Kalman filter differentiable filtering"`
    returned no hits, so the first literature phase must ingest or approve the
    required paper set rather than assuming it is already available.

## Source Inventory

### DSGE/HMC Monograph
Root:
- `~/python/docs/monograph.tex`

Important chapters:
- `ch08_kalman_filter.tex`
- `ch09_sr_ukf.tex`
- `ch09b_svd_filters.tex`
- `ch13_mass_matrix.tex`
- `ch14_hmc.tex`
- `ch15_advanced_hmc.tex`
- `ch15b_position_dependent_geometry.tex`
- `ch16_transport_foundations.tex`
- `ch17_transport_literature.tex`
- `ch18_transport_training.tex`
- `ch16_xla_ops.tex`
- `app_d_numerical_stability.tex`

Role in BayesFilter:
- reusable HMC/filtering algorithms and failure modes
- SVD sigma-point lessons
- XLA and finite-gradient engineering policy
- transport and geometry context

Exclude or isolate:
- NK/EZ/SGU model chapters except as application case studies
- DSGE perturbation solver internals except where they define a state-space
  interface contract

### CIP / Macro-Finance Monograph
Root:
- `~/latex/CIP_monograph/main.tex`

Important chapters:
- `ch11_state_space_recursions.tex`
- `ch12_identification.tex`
- `ch16_kalman_filter.tex`
- `ch17_nonlinear_filtering.tex`
- `ch18_mixed_frequency.tex`
- `ch20_hmc.tex`
- `ch21_advanced_hmc.tex`
- `ch26_differentiable_pf.tex`
- `ch27_ledh_pfpf_neural_ot.tex`
- `ch32_diff_resampling_neural_ot.tex`
- `ch33_analytical_validation.tex`
- `ch35_pipeline_synthesis.tex`
- appendices on TensorFlow/TFP design, affine computation, practical
  identification, and MCP cookbook

Role in BayesFilter:
- large macro-finance state-space motivations
- mixed-frequency and missing-data filtering
- differentiable particle-filter frontier
- decision framework for Kalman, sigma-point, and particle-filter/HMC pipelines

Exclude or isolate:
- CIP economics, basis sign conventions, asset-pricing model chapters, and
  portfolio-management presentation material except as downstream examples.

### MacroFinance Analytic Derivative Note
Root:
- `~/MacroFinance/analytic_kalman_derivatives.tex`

Related implementation:
- `~/MacroFinance/filters/differentiated_kalman.py`
- `~/MacroFinance/filters/solve_differentiated_kalman.py`
- `~/MacroFinance/filters/qr_sqrt_differentiated_kalman.py`
- `~/MacroFinance/filters/tf_qr_sqrt_differentiated_kalman.py`
- `~/MacroFinance/filters/masked_*`
- `~/MacroFinance/perf_ad_vs_analytic_kalman_gradient.py`
- `~/MacroFinance/tests/test_*differentiated_kalman*.py`
- `~/MacroFinance/tests/test_large_scale_lgssm*.py`

Role in BayesFilter:
- primary mathematical spine for analytic likelihood score/Hessian
- evidence that nested autodiff is not the right large-model default
- implementation reference for QR/Cholesky derivative recursions

## Intended Monograph Product

Create:

- `~/BayesFilter/docs/main.tex`
- `~/BayesFilter/docs/preamble.tex`
- `~/BayesFilter/docs/references.bib`
- `~/BayesFilter/docs/chapters/*.tex`
- `~/BayesFilter/docs/appendices/*.tex`
- `~/BayesFilter/docs/plans/*.md`
- optional `~/BayesFilter/docs/source_map.yml`

Working title:

> BayesFilter: HMC-Safe Filtering, Analytic Likelihood Derivatives, and
> Industrial Bayesian State-Space Inference

Target size:
- several hundred pages after consolidation
- modular enough that chapters can be built and audited independently

## Proposed Chapter Architecture

### Part I: Scope and Interfaces
1. Introduction: why filtering is the bottleneck for Bayesian state-space HMC
2. State-space model contracts: linear, nonlinear, missing-data, mixed-frequency
3. Posterior targets and HMC requirements: value, score, curvature, diagnostics
4. BayesFilter API design: model maps, filter backends, derivative providers

### Part II: Linear Gaussian Filtering
5. Prediction-error decomposition and exact Kalman likelihood
6. Numerically stable covariance, Cholesky, QR, and solve formulations
7. Missing data, masks, ragged panels, and mixed-frequency observations
8. Large-scale LGSSM implementation and memory/computation complexity

### Part III: Analytic Derivatives
9. First-order Kalman score recursions
10. Second-order Hessian and observed-information recursions
11. Structural derivatives: parameter maps into state-space objects
12. QR and Cholesky factor derivatives
13. Custom-gradient wrappers for HMC and XLA
14. Validation: finite differences, AD parity, invariants, and stress tests

### Part IV: Nonlinear Filtering
15. EKF and local-linearization filters
16. UKF/CKF sigma-point filters
17. Square-root sigma-point filters
18. SVD sigma-point filters: benefits, risks, and eigen-gap pathology
19. Particle filters, differentiable resampling, and pseudo-marginal HMC
20. Decision rules for filter choice as dimension and nonlinearity grow

### Part V: HMC, Geometry, and Diagnostics
21. HMC/NUTS for state-space likelihoods
22. Mass matrices, MAP curvature, and observed information
23. Boundary handling, finite target values, and safe gradients
24. XLA/JIT constraints and TensorFlow/TFP implementation discipline
25. Divergences, eigen-gap telemetry, condition numbers, and failure taxonomy
26. Transport maps, NeuTra, and surrogate geometry for large models

### Part VI: Industrial Applications and Case Studies
27. Small LGSSM and nonlinear SSM validation ladder
28. NK DSGE as a small nonlinear/perturbation case study
29. CIP/AFNS macro-finance state-space case study
30. Toward NAWM-scale DSGE and large multi-country asset-pricing systems
31. Production checklist: when a filter/HMC backend is industrial-ready

### Appendices
A. Notation and shape conventions
B. Matrix calculus identities
C. QR/Cholesky/SVD derivative proofs
D. MathDevMCP audit workflows
E. ResearchAssistant literature workflows
F. Code-to-document source map
G. Experiment templates and reset-memo templates

## Content Ownership Rules

1. Generic filter and HMC infrastructure belongs in BayesFilter chapters.
2. DSGE, CIP, AFNS, and NAWM specifics should appear only as applications or
   motivating examples.
3. Do not copy chapters wholesale without rewriting local notation and removing
   project-specific assumptions.
4. Preserve source provenance for every imported section in `source_map.yml`.
5. Every mathematical claim should be classified as:
   - derivation
   - implementation contract
   - empirical diagnostic
   - literature claim
6. No claim about HMC convergence should rely on smoke tests or clean compile
   checks.
7. Any filter backend proposed for HMC must state its derivative policy:
   - autodiff through primitive operations
   - custom analytic score
   - stopped-gradient approximation
   - pseudo-marginal estimator
   - not HMC-safe

## Execution Phases

### Phase 0: Repository and Build Scaffold
Create:
- `docs/main.tex`
- `docs/preamble.tex`
- `docs/references.bib`
- `docs/chapters/`
- `docs/appendices/`
- `docs/source_map.yml`
- `docs/plans/`

Pass criteria:
- `pdflatex` or `latexmk` builds a skeleton PDF
- no copied source content yet except short placeholder abstracts
- plan file committed first

### Phase 1: Source Index and Provenance Map
Build a machine-readable source map.

Suggested fields:
- source project
- source file
- source chapter/section/label
- destination chapter
- import status: `candidate`, `drafted`, `audited`, `superseded`, `excluded`
- reason
- required rewrite notes

Use MathDevMCP commands such as:

```bash
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli \
  search-latex "Kalman filter likelihood score Hessian" \
  --root /home/chakwong/python/docs
```

```bash
PYTHONPATH=/home/chakwong/MathDevMCP/src python -m mathdevmcp.cli \
  search-latex "mixed frequency nonlinear filtering HMC" \
  --root /home/chakwong/latex/CIP_monograph
```

Pass criteria:
- every candidate source chapter has a destination or exclusion reason
- duplicated sections are identified before copying

### Phase 2: Literature Workspace
Create a dedicated ResearchAssistant workspace, for example:

```bash
ra --root /tmp/ra-bayesfilter-monograph init
ra --root /tmp/ra-bayesfilter-monograph doctor
```

Seed it with core papers and local PDFs/source when available:
- Kalman filter and prediction-error decomposition
- square-root filtering
- unscented/cubature Kalman filtering
- differentiable filtering and analytic Kalman derivatives
- particle MCMC and pseudo-marginal HMC
- HMC/NUTS and Riemannian/geometry diagnostics
- transport maps and NeuTra

Pass criteria:
- approved local summaries for the core literature
- citation-neighborhood notes for frontier nonlinear filtering/HMC topics
- no unsupported literature claims in the monograph outline

### Phase 3: Linear Gaussian Spine
Draft Parts I--III first. These are the foundation for BayesFilter as an
independent package.

Primary source:
- `~/MacroFinance/analytic_kalman_derivatives.tex`

Secondary sources:
- `~/python/docs/chapters/ch08_kalman_filter.tex`
- `~/latex/CIP_monograph/chapters/ch16_kalman_filter.tex`

Pass criteria:
- analytic score/Hessian notation is unified
- implementation contracts for state-space objects and derivative providers are
  explicit
- MathDevMCP derivation/code audit is run against the MacroFinance
  differentiated Kalman implementation

### Phase 4: HMC-Safe Derivative Policy
Write the central design chapter explaining why BayesFilter should not depend
on raw autodiff through fragile decompositions for large models.

Must cover:
- eigenvalue/eigenvector derivative instability
- SVD sigma-point filter lessons from `dsge_hmc`
- why NAWM-scale HMC needs structured derivatives
- custom-gradient contract: returned score must match the log target
- finite boundary rejection and gradient policy

Pass criteria:
- explicit decision table for each backend: Kalman, QR sqrt, SVD sigma-point,
  EKF, UKF/CKF, particle filter
- no backend is labeled HMC-safe without a derivative and diagnostics policy

### Phase 5: Nonlinear Filtering and SVD Lessons
Consolidate:
- `~/python/docs/chapters/ch09_sr_ukf.tex`
- `~/python/docs/chapters/ch09b_svd_filters.tex`
- `~/latex/CIP_monograph/chapters/ch17_nonlinear_filtering.tex`
- `~/latex/CIP_monograph/chapters/ch26_differentiable_pf.tex`
- `~/latex/CIP_monograph/chapters/ch27_ledh_pfpf_neural_ot.tex`
- `~/latex/CIP_monograph/chapters/ch32_diff_resampling_neural_ot.tex`

Pass criteria:
- SVD filter is framed as robust likelihood evaluation plus carefully bounded
  HMC use, not the default industrial gradient path
- eigen-gap and condition telemetry become first-class diagnostics
- particle-filter/HMC alternatives are positioned clearly

### Phase 6: HMC and Geometry
Consolidate:
- `~/python/docs/chapters/ch13_mass_matrix.tex`
- `~/python/docs/chapters/ch14_hmc.tex`
- `~/python/docs/chapters/ch15_advanced_hmc.tex`
- `~/python/docs/chapters/ch15b_position_dependent_geometry.tex`
- `~/python/docs/chapters/ch16_transport_foundations.tex`
- `~/python/docs/chapters/ch17_transport_literature.tex`
- `~/python/docs/chapters/ch18_transport_training.tex`
- `~/latex/CIP_monograph/chapters/ch20_hmc.tex`
- `~/latex/CIP_monograph/chapters/ch21_advanced_hmc.tex`
- `~/latex/CIP_monograph/chapters/ch34_hnn_surrogate.tex`

Pass criteria:
- short HMC diagnostics are not described as convergence evidence
- mass-matrix and observed-information chapters refer back to analytic
  derivative recursions
- large-model geometry strategies are tied to practical gates

### Phase 7: Case Studies
Write compact case-study chapters rather than importing full model monographs.

Case studies:
- controlled LGSSM
- generic nonlinear SSM
- NK SVD sigma-point debugging and hardening
- CIP/AFNS large LGSSM and analytic derivative validation
- NAWM-scale design target

Pass criteria:
- each case study explains what BayesFilter learns from it
- project-specific economics remains in source projects
- BayesFilter only owns generic algorithms and interfaces

### Phase 8: Build, Audit, and Release Gates
For each drafted part:

1. Build the PDF.
2. Run MathDevMCP label/search checks.
3. Run a claim-support pass for literature-heavy chapters.
4. Run code-document consistency checks for implementation-linked chapters.
5. Update `source_map.yml`.
6. Add a reset memo or result note under `docs/plans`.

Suggested build command:

```bash
cd /home/chakwong/BayesFilter/docs
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Initial Deliverables

1. This plan.
2. A skeleton LaTeX book under `~/BayesFilter/docs`.
3. `source_map.yml` with initial source inventory.
4. A reset memo recording setup and decisions.
5. A first buildable PDF skeleton.

## Risks

- The monograph becomes a copy-paste archive instead of a coherent book.
- Project-specific economics leaks into BayesFilter as generic policy.
- SVD filter hardening is overgeneralized into a false industrial guarantee.
- Literature claims are imported from old text without re-audit.
- Build complexity becomes unmanageable if preambles and labels are copied
  wholesale.
- The source map is skipped, causing future maintenance drift.

## Stop Rules

- Stop content migration if the skeleton cannot build.
- Stop importing a chapter if its notation conflicts with the BayesFilter
  notation contract.
- Stop claiming an implementation is HMC-safe if its gradient policy is not
  explicit and tested.
- Stop treating any source chapter as authoritative if its code references are
  stale.

## Next Action
Execute Phase 0 only:

1. create the skeleton LaTeX book;
2. create `source_map.yml`;
3. add a reset memo;
4. build the empty/skeleton PDF;
5. commit the scaffold.

Do not migrate hundreds of pages until Phase 0 and Phase 1 are complete.

## Independent Audit Addendum

Date:
- 2026-05-02

Audit stance:
- Treat this as another developer's pre-execution review.
- The plan is sensible, but the later phases are too large to execute safely
  without explicit gates. The safe autonomous path is to execute scaffold,
  source-map, and literature-workspace setup first, then stop before content
  migration if the gates are not met.

Required modifications before execution:

1. Original documents must be read-only inputs.
   - Do not edit `~/python/docs`, `~/latex/CIP_monograph`, or
     `~/MacroFinance` while building the BayesFilter monograph.
   - All consolidation output belongs under `~/BayesFilter/docs`.

2. Add a LaTeX namespace policy.
   - New labels should use a BayesFilter prefix such as `bf:`, `ch:bf-`,
     `sec:bf-`, `eq:bf-`, `alg:bf-`, and `tab:bf-`.
   - Imported labels from source documents should not be copied verbatim until
     they are rewritten or mapped, because duplicate labels are likely.

3. Add a bibliography policy.
   - Start with a small BayesFilter `references.bib`.
   - Merge source bibliographies only through an explicit key-conflict audit.
   - Literature-heavy claims require ResearchAssistant or manual source notes.

4. Add a build-artifact policy.
   - Commit source `.tex`, `.bib`, `.yml`, `.md`, and selected generated PDFs
     only if deliberately requested.
   - Ignore `.aux`, `.log`, `.out`, `.toc`, `.bbl`, `.blg`, `.fdb_latexmk`,
     `.fls`, and similar LaTeX byproducts.

5. Clarify Phase 2 as a gate.
   - If ResearchAssistant has no approved summaries for the core literature,
     Phase 2 is incomplete.
   - It is still useful to create the workspace and seed list, but drafting
     literature-heavy chapters is not justified until sources are ingested or
     explicitly waived.

6. Clarify content-migration stop rule.
   - Do not run Phases 3--7 as full writing phases until Phase 0 and Phase 1
     pass, and Phase 2 is either passed or explicitly scoped down.
   - Placeholder chapter files are allowed in Phase 0; they are not content
     migration evidence.

7. Add a reset-memo cadence.
   - After each executed phase, record commands, outputs, interpretation, and
     whether the next phase remains justified in:
     `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`.

Audit conclusion:
- Proceed with Phase 0.
- Proceed with Phase 1 after Phase 0 builds.
- Proceed with Phase 2 setup after Phase 1 source-map creation.
- Stop before Phases 3--7 unless the source-map and literature gates are
  satisfied or the user explicitly approves drafting from existing internal
  documents without a completed literature workspace.

## Execution Addendum: Phase 2 Gate Result

Date:
- 2026-05-02

Executed scope:
- Phase 0 scaffold.
- Phase 1 source-map/provenance pass.
- Phase 2 ResearchAssistant workspace setup and seed-list creation.

Gate result:
- Phase 0 passed.
- Phase 1 passed.
- Phase 2 setup passed, but Phase 2 did not satisfy the full literature gate:
  no approved ResearchAssistant summaries or citation-neighborhood notes exist
  yet for the core literature set.

Plan modification:
- Do not execute Phases 3--7 as content migration in this pass.
- Treat the next work as a dedicated Phase 2B literature-ingestion and
  claim-support audit, or as a narrow internal-only drafting pass that avoids
  literature-heavy claims.

Rationale:
- The source map shows several chapters whose correctness depends on external
  literature support, especially nonlinear filtering, differentiable particle
  filters, pseudo-marginal HMC, transport maps, NeuTra, and large-scale
  production thresholds.
- Drafting those chapters now would likely reproduce unsupported claims from
  source monographs rather than producing an auditable BayesFilter text.
