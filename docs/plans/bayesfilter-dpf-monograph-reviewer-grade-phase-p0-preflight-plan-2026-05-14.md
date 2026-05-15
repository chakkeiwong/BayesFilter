# Phase P0 plan: worktree, build, and source-inventory preflight

## Date

2026-05-14

## Governing program

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-phase-proposal-2026-05-14.md`

## Purpose

Establish a reliable baseline before any reviewer-grade writing pass begins.
This phase exists to prevent two common failures: editing over unrelated dirty
work and rewriting mathematical chapters before the source, label, citation,
and build state is known.

## Target scope

Primary DPF chapters:

- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch19b_dpf_literature_survey.tex`
- `docs/chapters/ch19c_dpf_implementation_literature.tex`
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`

Supporting files:

- `docs/main.tex`
- `docs/preamble.tex`
- `docs/references.bib`
- `docs/source_map.yml` if used by the repo's documentation workflow.

## Implementation instructions

1. Verify branch and repository state:
   - run `git branch --show-current`;
   - run `git status --short --branch`;
   - run `git log --oneline --decorate --graph -12`.
2. Classify dirty files:
   - DPF monograph lane;
   - student-baseline lane;
   - unrelated BayesFilter implementation lane;
   - unknown.
3. Do not edit unknown or unrelated dirty files.  Record them in the P0 result
   note instead.
4. Inspect the DPF chapter structure:
   - line counts;
   - chapter/section/subsection map;
   - equation labels;
   - tables;
   - theorem-like environments;
   - citation keys.
5. Run the repo's established LaTeX build route if available.  If the build
   requires unavailable tools, record the exact blocker and run the nearest
   feasible static checks.
6. Search for DPF-local undefined-looking references:
   - `\\ref{...}`;
   - `\\eqref{...}`;
   - `\\citep{...}`;
   - `\\citet{...}`;
   - labels duplicated across DPF chapters.
7. Query ResearchAssistant in read-only mode for local paper availability:
   - SMC and particle filters;
   - particle flow and EDH/LEDH;
   - PF-PF and proposal correction;
   - differentiable resampling;
   - OT/EOT/Sinkhorn;
   - learned set maps and amortized OT;
   - HMC, pseudo-marginal MCMC, surrogate/noisy-gradient MCMC;
   - DSGE and MacroFinance filtering.
8. Query MathDevMCP or use direct LaTeX search to identify derivation labels and
   candidate proof obligations.

## Required result artifact

Create:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-result-2026-05-15.md`

This artifact already exists and is the governing P0 result unless superseded
by a later explicitly dated preflight result.

The result artifact must include:

1. branch and commit state;
2. dirty-file classification;
3. build status and exact command used;
4. DPF chapter section map;
5. label and citation inventory;
6. source availability register;
7. initial derivation-obligation candidates;
8. blockers, if any.

## Audit rules

- Do not infer source availability from citations; verify local availability or
  record the source as bibliography-only.
- Do not treat a successful compile as mathematical readiness.
- Do not proceed to chapter edits if dirty files make the DPF lane ambiguous.
- Do not hide missing build capability; record it as a reproducibility risk.

## Veto diagnostics

The phase fails if:

- branch state is not recorded;
- dirty worktree files are not classified;
- DPF chapter labels and citations are not inventoried;
- ResearchAssistant availability is not checked;
- MathDevMCP or direct derivation-obligation inventory is skipped;
- build status is assumed rather than tested or explicitly blocked.

## Exit gate

Proceed to P1 only when the P0 result note gives a clear baseline for the
current document, source availability, and build/reference state.
