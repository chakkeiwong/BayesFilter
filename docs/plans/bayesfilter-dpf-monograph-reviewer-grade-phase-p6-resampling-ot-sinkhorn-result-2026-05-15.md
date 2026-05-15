# Phase P6 result: differentiable resampling and OT/Sinkhorn expansion

## Date

2026-05-15

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane DPF chapter changes present:
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`.
- In-lane reviewer-grade planning/result artifacts present as untracked files.
- User-owned in-lane governance edit remains present:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`.
- Out-of-lane student-baseline files remain present and were not touched.

## Allowed write set used

- `docs/chapters/ch32_diff_resampling_neural_ot.tex`.
- This result artifact.

No student-baseline file, git history operation, deletion, push, merge, rebase,
or reset was performed.

## Chapter changes completed

P6 expanded the differentiable-resampling chapter so that categorical
resampling, soft resampling, unregularized OT, entropic OT, finite Sinkhorn, and
solver-differentiated objectives are explicitly separated.

Completed items:

1. Added an object/convention inventory for weighted and equal-weight empirical
   measures, ancestor indices, soft-resampling interpolation, couplings, the OT
   feasible set, cost matrix, entropy convention, Sinkhorn scalings,
   barycentric output particles, and finite-iteration plans.
2. Added source-role limits: citations support provenance for differentiable
   resampling, OT, and Sinkhorn, but do not certify categorical-law
   preservation, pseudo-marginal validity, or HMC target validity.
3. Added a two-particle discontinuity example for categorical resampling.
4. Expanded soft-resampling mean preservation, affine-test preservation, and
   nonlinear test-function bias, including dimension and smoothness assumptions.
5. Defined the OT coupling feasible set `eq:bf-dr-coupling-set` and clarified
   that unregularized OT is exact for the chosen transport projection, not for
   categorical resampling.
6. Added the entropy convention `eq:bf-dr-entropy-convention`.
7. Derived EOT stationarity, the Gibbs factorization, Sinkhorn scaling form,
   marginal equations, and scaling updates.
8. Expanded barycentric projection dimensions and clarified that the
   barycentric map is a deterministic support-cloud construction, not the
   coupling itself and not an ancestor draw.
9. Added a numerical-analysis section for epsilon limits, kernel underflow,
   log-domain stabilization, finite-iteration residuals, unrolled versus
   implicit differentiation, and memory/runtime scaling.
10. Expanded status/debug tables to include solver-differentiated objectives
    and iteration-budget gradient sensitivity.
11. Added an explicit non-claim that finite Sinkhorn is not a black-box layer
    whose gradient has the same status as the original categorical likelihood.

## ResearchAssistant evidence

ResearchAssistant remains available only as a read-only/offline local workspace.

Queries run:

- `differentiable resampling optimal transport Sinkhorn particle filter Corenflos Cuturi`;
- `soft resampling differentiable particle filters Zhu Murphy Jonschkowski`.

Both returned no local paper summaries.  Therefore this phase does not describe
P6 source support as ResearchAssistant-reviewed.  Source support remains
bibliography-spine plus local derivation and local claim-boundary text.

## Derivation-obligation audit

| Obligation | Chapter location | Method | Result |
|---|---|---|---|
| Soft-resampling mean preservation | `eq:bf-dr-soft-mean-preserved` | MathDevMCP scalar equality plus manual dimension review | Passed manually; scalarized equality `alpha*xbar + (1-alpha)*xbar = xbar` certified by SymPy. |
| Soft-resampling nonlinear test-function bias | `eq:bf-dr-soft-test-bias` | MathDevMCP label lookup/audit plus manual Taylor review | Passed manually; MathDevMCP remained diagnostic-only and requested dimensions/regularity, which were added. |
| OT feasible set and unregularized objective | `eq:bf-dr-coupling-set`, `eq:bf-dr-unregularized-ot` | Manual constraint and object-status audit | Passed; rows and columns, target weights, and non-categorical status are explicit. |
| EOT entropy convention and stationarity | `eq:bf-dr-entropy-convention`, `eq:bf-dr-eot-stationarity` | MathDevMCP audit plus scalarized derivative check | Passed manually; scalarized derivative simplification was certified by SymPy. |
| Sinkhorn scaling equations and updates | `eq:bf-dr-sinkhorn-marginals`, `eq:bf-dr-sinkhorn-updates` | MathDevMCP audit plus scalarized row-update check | Passed manually; row-update scalar identity was certified by SymPy. |
| Barycentric map dimensions | `eq:bf-dr-barycentric-map` | MathDevMCP label/audit plus scalar normalization check | Passed manually; scalar normalization `M*(1/M)=1` was certified by SymPy. |
| Finite-solver residual status | `eq:bf-dr-sinkhorn-residuals` | Manual numerical-analysis audit | Passed; residuals are necessary diagnostics, not proof of categorical-law preservation. |

MathDevMCP full derivation audits for the main labels remained `unverified` or
`inconclusive` because the obligations involve vector/matrix constraints,
Taylor remainders, optimization KKT conditions, and prose-level regularity
assumptions outside the bounded algebraic backend.  This is not treated as
mechanical certification.

## Local checks run

- `git branch --show-current`.
- `git status --short --branch`.
- `rg` checks for:
  - `exact`;
  - `optimal`;
  - `categorical`;
  - `finite`;
  - `implicit`;
  - `unrolled`;
  - `black-box`;
  - new P6 equation labels.
- ResearchAssistant queries listed above.
- MathDevMCP:
  - `search_latex`;
  - `latex_label_lookup`;
  - `audit_derivation_v2_label`;
  - scalar `check_equality` and `check_proof_obligation` calls.
- LaTeX build:
  - working directory: `docs`;
  - command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;
  - result: passed; final `latexmk` status reports all targets up to date;
  - final PDF size: 206 pages.
- Post-build log scan:
  - no undefined citations;
  - no undefined references;
  - no rerun warnings;
  - no multiply defined label warnings found by targeted `rg`.

## Layout inspection

P6 adds table/layout pressure.  The final build passes, but the log contains
P6-local warnings including:

- underfull warning in the object inventory around lines 47--49;
- page-header overfull warnings for Chapter 23 pages;
- underfull table warnings around comparison tables near lines 483--510;
- an overfull warning around the family comparison table near lines 499--510;
- underfull warnings in the debug/source maps around lines 532--555;
- an overfull warning around the chapter-boundary list near lines 575--578.

These are not mathematical vetoes, but P10 must revisit P6 table layout rather
than waiting until final PDF review only.

## Veto diagnostic review

- Differentiability appears free: no.
- OT and categorical resampling remain blurred: no.
- Finite Sinkhorn is not separated from EOT: no.
- Barycentric projection is not explained: no.
- Numerical stabilization is mentioned without mathematical context: no.

## Exit gate

Passed with layout caution.

A reviewer can now identify the object whose gradient is computed for each
resampling variant: categorical law has no pathwise realized-ancestor gradient;
soft resampling differentiates a shrinkage surrogate; EOT differentiates the
regularized transport optimizer; finite Sinkhorn differentiates a numerical
solver path or fixed-point system.

## Next recommended phase

Proceed to P8 under the user's execution order:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p8-hmc-banking-target-plan-2026-05-14.md`.
