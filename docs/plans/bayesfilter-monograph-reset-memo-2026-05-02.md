# Reset memo: BayesFilter monograph consolidation

## Date
2026-05-02

## Context
The user wants a new BayesFilter monograph under `~/BayesFilter/docs` that
consolidates filtering, HMC, analytic Kalman derivatives, SVD-filter numerical
lessons, and industrial-scale state-space inference material now scattered
across:

- `~/python/docs/monograph.tex`
- `~/python/docs/chapters/*.tex`
- `~/latex/CIP_monograph/main.tex`
- `~/latex/CIP_monograph/chapters/*.tex`
- `~/MacroFinance/analytic_kalman_derivatives.tex`
- related implementation notes, tests, and reset memos

The original documents are source/provenance material and should not be edited
as part of this monograph consolidation. Work should happen inside
`~/BayesFilter`.

## Initial repo state
- Repo: `/home/chakwong/BayesFilter`
- Branch: `main`
- Initial commit before this work: `a09e8d1 first commit`
- Initial tracked file: `README.md`
- Existing untracked file before this execution pass:
  `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`

## User request
1. Update the reset memo.
2. Audit the plan as another developer.
3. Execute each phase using a plan/execute/test/audit/tidy/update-memo cycle.
4. After each phase, update the memo with results, interpretation, and whether
   the next phase is justified.
5. Continue without human intervention if no major issue blocks the next phase.
6. Commit modified files after the whole plan finishes.
7. Update the reset memo on completion.
8. Provide a detailed result summary and next-phase hypotheses.

## Execution policy
- Do not edit the original source documents in `~/python`, `~/latex`, or
  `~/MacroFinance`.
- Use `MathDevMCP` where useful for source discovery, derivation/code auditing,
  and notation/shape review.
- Use `ResearchAssistant` where useful for literature workspace setup and
  claim-support planning.
- Stop content migration until the BayesFilter skeleton builds and the initial
  source map exists.

## Current status
This memo starts the execution pass. The plan audit and Phase 0 scaffold are
next.

## Plan audit

Plan audited:
- `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`

Audit outcome:
- The plan is directionally correct and should create a new BayesFilter
  monograph without changing original source documents.
- The major risk is uncontrolled content migration: hundreds of pages could be
  copied before notation, bibliography, provenance, and build policies are
  settled.

Amendments made:
- Added an independent audit addendum to the plan.
- Required original source docs to remain read-only.
- Added label namespace policy for BayesFilter labels.
- Added bibliography and build-artifact policy.
- Clarified that Phase 2 literature setup is a gate for literature-heavy
  chapters.
- Clarified that Phases 3--7 should not run as content migration until Phase 0
  and Phase 1 pass, and Phase 2 is either passed or explicitly scoped down.

Interpretation:
- Phase 0 is justified.
- Phase 1 is justified after Phase 0 builds.
- Phase 2 setup is justified after Phase 1.
- Full content migration is not yet justified.

## Phase 0: Repository and build scaffold

Phase plan:
- Create a standalone BayesFilter LaTeX book skeleton under
  `/home/chakwong/BayesFilter/docs`.
- Keep original source documents read-only.
- Commit LaTeX sources and plans only; treat generated build byproducts as
  local artifacts.

Execution results:
- Created `docs/main.tex`, `docs/preamble.tex`, `docs/references.bib`, and
  `docs/source_map.yml`.
- Created placeholder chapters `docs/chapters/ch01_*.tex` through
  `docs/chapters/ch32_*.tex`.
- Created placeholder appendices `docs/appendices/app_a_*.tex` through
  `docs/appendices/app_g_*.tex`.
- Added `.gitignore` entries for LaTeX byproducts.
- Built the skeleton from `/home/chakwong/BayesFilter/docs` with:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.

Test results:
- Skeleton PDF build succeeded.
- Generated `docs/main.pdf` was 83K and 51 pages.
- The only expected warning was an empty bibliography, because the skeleton has
  no citations yet.
- LaTeX byproducts are ignored by Git. `docs/main.pdf` is also ignored as a
  generated local artifact unless a later release phase explicitly requests a
  committed PDF.

Audit:
- Phase 0 satisfies the build-scaffold pass criteria.
- No source content was migrated, beyond short placeholders.
- Original source projects were not edited.

Interpretation:
- The scaffold is healthy enough to support source indexing.
- Phase 1 is justified.

## Phase 1: Source index and provenance map

Phase plan:
- Build a machine-readable source map before importing any prose.
- Classify each major source chapter as `candidate`, `superseded`, or
  `excluded`.
- Identify duplicate groups and sources whose claims require a literature gate.

Execution results:
- Replaced the initial `docs/source_map.yml` stub with a detailed provenance
  map covering:
  - DSGE/HMC monograph source chapters and SVD reset notes.
  - CIP monograph source chapters and appendices.
  - MacroFinance analytic Kalman note, bibliography, implementation files, and
    tests.
- Recorded destination chapters for reusable material.
- Marked application-specific economics chapters as excluded or case-study
  only.
- Marked transport, particle-filter, nonlinear-filtering, and quantitative
  production claims as literature/audit gated.
- Built temporary MathDevMCP LaTeX indexes:
  - `/home/chakwong/python/docs`: 3615 blocks, 1065 labels.
  - `/home/chakwong/latex/CIP_monograph`: 2607 blocks, 1016 labels.
  - `/home/chakwong/MacroFinance`: 225 blocks, 140 labels.
- Ran targeted MathDevMCP searches for:
  - Kalman likelihood, score, and Hessian material.
  - Mixed-frequency nonlinear filtering and HMC.
  - SVD sigma-point/eigen-gap material.
  - Particle-filter, differentiable resampling, and pseudo-marginal HMC
    material.
  - XLA, `GradientTape`, HMC, and finite-gradient policy.

Test results:
- `docs/source_map.yml` parses successfully with `yaml.safe_load`.
- The LaTeX skeleton remains buildable with `latexmk`.
- The source map now includes phase gates showing Phase 1 passed, Phase 2
  pending, and content migration blocked until literature/provenance gates are
  satisfied.

Audit:
- The map avoids wholesale import of original chapters.
- It distinguishes generic BayesFilter material from downstream economics.
- It explicitly identifies duplicate source groups:
  - Kalman likelihood basics.
  - HMC foundations and diagnostics.
  - Nonlinear Gaussian filters.
  - Particle-filter frontier.
- Bibliography sources are listed but not merged, preserving the key-conflict
  audit requirement.

Interpretation:
- Phase 1 satisfies its pass criteria.
- Phase 2 literature workspace setup is justified.
- Full content migration remains blocked because literature-heavy claims and
  advanced nonlinear-filter/HMC claims do not yet have approved local source
  summaries.

## Phase 2: Literature workspace setup

Phase plan:
- Create a dedicated ResearchAssistant workspace.
- Confirm local tool readiness.
- Search for existing approved summaries.
- Create a durable seed list under BayesFilter so the phase can continue after
  reboot even if the temporary workspace is removed.

Execution results:
- Initialized ResearchAssistant workspace:
  `/tmp/ra-bayesfilter-monograph`.
- `doctor` reported status `ok`, offline mode enabled, providers disabled, and
  local PDF parsing tools available (`pdftotext`, `markitdown`,
  `marker_single`, and `magic-pdf`).
- `find --query "Hamiltonian Monte Carlo Kalman filter differentiable
  filtering"` returned no existing local hits.
- `find --query "Kalman"` returned no existing local hits.
- `review-list` returned no approved reviews.
- Added durable seed/audit checklist:
  `docs/plans/bayesfilter-literature-seed-list-2026-05-02.md`.
- Inspected source bibliography sizes:
  - `/home/chakwong/python/docs/references.bib`: 1459 lines.
  - `/home/chakwong/latex/CIP_monograph/references.bib`: 3984 lines.
  - `/home/chakwong/MacroFinance/analytic_kalman_derivatives.bib`: 73 lines.
- Confirmed many relevant candidate keys exist locally, but they are not yet
  reviewed or merged.

Test results:
- ResearchAssistant workspace initialization passed.
- ResearchAssistant `doctor` passed.
- Local reviewed-summary search found no approved source material.

Audit:
- Phase 2 setup is useful and reproducible.
- Phase 2 full pass criteria are not met because there are no approved local
  summaries or citation-neighborhood notes.
- The seed list preserves the next literature task without falsely treating
  raw `.bib` entries as evidence.

Interpretation:
- Phase 2 is setup-only, not passed as a literature gate.
- Phases 3--7 content migration are not justified in this autonomous pass.
- The next justified phase is Phase 2B: literature ingestion, review, and
  claim-support audit, or a deliberately scoped internal-only drafting pass
  that avoids external literature claims.

## Stop decision before Phases 3--7

Decision:
- Stop autonomous content migration before Phases 3--7.

Reason:
- The plan's audited gate requires approved literature summaries or an explicit
  scoping decision before drafting literature-heavy chapters.
- Phase 2 created the tooling workspace and durable seed list, but did not
  produce reviewed summaries.
- Proceeding into the linear/nonlinear/HMC chapters now would risk turning
  BayesFilter into a copy-paste archive of source monographs and unsupported
  claims.

What is still safe without additional approval:
- Continue Phase 2B literature ingestion and claim-support review.
- Draft purely internal scaffolding material: notation conventions,
  source-map appendix, implementation-interface placeholders, and reset-memo
  templates, provided no unsupported external claims are introduced.

What is not justified yet:
- Drafting full Parts III--VI.
- Claiming any SVD sigma-point filter path is HMC-safe at large scale.
- Importing particle-filter, pseudo-marginal HMC, transport-map, NeuTra, or
  production-threshold claims from source chapters without source review.

## Final validation before commit

Commands:
- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`
  from `/home/chakwong/BayesFilter/docs`.
- `python -c "import yaml; yaml.safe_load(open('docs/source_map.yml'));
  print('source_map yaml ok')"` from `/home/chakwong/BayesFilter`.
- `git status --short --ignored` from `/home/chakwong/BayesFilter`.

Results:
- LaTeX skeleton is up to date and buildable.
- `docs/source_map.yml` parses successfully.
- LaTeX byproducts and `docs/main.pdf` are ignored as generated artifacts.

Completion state:
- Completed: plan audit, Phase 0, Phase 1, Phase 2 setup.
- Stopped by design before content migration.
- Initial scaffold commit for this work was `6bafcc5`, then amended during
  reset-memo finalization.

## Commit checkpoint

Scaffold commit before the final reset-memo correction:
- `373bb8e` (`Scaffold BayesFilter monograph`)

Note:
- A later commit may update only this reset memo. Use
  `git log --oneline -2` to inspect the exact final repository state.

Committed files:
- `.gitignore`
- `docs/main.tex`
- `docs/preamble.tex`
- `docs/references.bib`
- `docs/source_map.yml`
- `docs/chapters/*.tex`
- `docs/appendices/*.tex`
- `docs/plans/bayesfilter-monograph-consolidation-plan-2026-05-02.md`
- `docs/plans/bayesfilter-monograph-reset-memo-2026-05-02.md`
- `docs/plans/bayesfilter-literature-seed-list-2026-05-02.md`

Generated but intentionally uncommitted:
- `docs/main.pdf`
- LaTeX build byproducts (`.aux`, `.log`, `.toc`, `.bbl`, `.blg`,
  `.fdb_latexmk`, `.fls`, `.out`)
