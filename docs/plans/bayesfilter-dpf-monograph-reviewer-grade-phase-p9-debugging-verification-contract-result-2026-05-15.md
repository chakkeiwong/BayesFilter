# Phase P9 result: debugging crosswalk into verification contract

## Date

2026-05-15

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane DPF chapter changes present before or during this phase:
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`;
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`.
- In-lane reviewer-grade planning/result artifacts are present as untracked
  files.
- User-owned in-lane governance edit remains present:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`.
- Out-of-lane student-baseline and controlled-baseline experiment files remain
  present and were not touched.

## Allowed write set used

- `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`.
- This result artifact.

No student-baseline file, controlled-baseline experiment file, git history
operation, deletion, push, merge, rebase, reset, or staging operation was
performed.

## Chapter changes completed

P9 replaced the qualitative debugging crosswalk with an equation-indexed
verification contract.

Completed items:

1. Added a diagnostic order that requires scalar-target and value-gradient
   checks before tuning, posterior comparisons, or sampler interpretation.
2. Added an equation-to-test matrix covering filtering recursion, bootstrap
   likelihood estimation, homotopy derivatives, EDH/LEDH endpoints, PF-PF
   weights, log-determinant ODEs, categorical discontinuity, soft-resampling
   bias, EOT/Sinkhorn marginals, barycentric projection, learned-map residuals,
   and the HMC value-gradient contract.
3. Added minimal controlled examples: linear-Gaussian recovery, synthetic
   affine flow, scalar nonlinear observation, two-particle soft-resampling
   bias, small Sinkhorn problem, permutation-equivariance learned-map test, and
   finite-difference HMC gradient test.
4. Added a diagnostic specification requiring equation or claim, inputs,
   expected output, tolerance, failure interpretation, and non-implication.
5. Added a source/chapter/equation/implementation map for classical PF,
   particle flow, PF-PF correction, resampling/OT, learned OT, and HMC target
   diagnostics.
6. Added promotion thresholds for exploratory implementation, credible research
   implementation, bank-facing research claim, and production/governance claim.
7. Added a boundary statement restricting `validate` to equation-local checks
   or separately governed model-risk processes.

## ResearchAssistant evidence

ResearchAssistant was not used to upgrade source support in P9.  P2 found no
local ResearchAssistant summaries for the DPF source spine, and this phase
therefore keeps source support at bibliography-spine provenance plus local
equation/diagnostic obligations.  No P9 sentence describes DPF source support
as ResearchAssistant-reviewed.

## Derivation and diagnostic-obligation audit

| Obligation | Chapter location | Method | Result |
|---|---|---|---|
| Equation-indexed diagnostics | `tab:bf-dpf-debug-equation-test-matrix` | Manual equation-to-test audit plus label existence checks | Passed. Each diagnostic row points to named equations or a named HMC contract and states a controlled example, implementation quantity, pass signal, and narrow failure interpretation. |
| Classical PF checks | rows for filtering recursion and bootstrap likelihood estimator | MathDevMCP label lookup from prior P9 work plus manual review | Passed. The referenced PF equations exist and are connected to Kalman-reference controlled examples. |
| Particle-flow checks | rows for homotopy derivative, EDH endpoint, and LEDH local endpoint | Manual derivation-routing audit | Passed. Failures are limited to homotopy algebra, flow ODE, Gaussian closure, local linearization, or integration issues. |
| PF-PF correction checks | rows for PF-PF weight and log-determinant ODE | MathDevMCP label lookup from prior P9 work plus manual review | Passed. Density-ratio and log-determinant diagnostics are separated from resampling and HMC target claims. |
| Resampling/OT checks | rows for categorical discontinuity, soft-resampling bias, Sinkhorn marginals, and barycentric projection | MathDevMCP label lookup from prior P9 work plus manual review | Passed. The chapter separates categorical target behavior, relaxed surrogate bias, finite Sinkhorn residuals, and barycentric projection shape/normalization. |
| Learned-map checks | learned-map residual row and controlled examples | MathDevMCP label lookup from prior P9 work plus manual review | Passed. Teacher and student diagnostics are separated, and learned residuals are not allowed to refute the teacher map. |
| HMC value-gradient check | `eq:bf-dpf-hmc-contract` row | MathDevMCP label lookup from prior P9 work plus manual review | Passed. The chapter prohibits HMC tuning before scalar value-gradient mismatch is resolved. |
| Promotion thresholds | `tab:bf-dpf-debug-promotion-thresholds` | Manual claim-boundary audit | Passed. Bank-facing and production/governance claims require evidence beyond chapter prose or benchmark success. |

MathDevMCP was useful for label-level support on several referenced equations.
The phase still relies on manual review for diagnostic meaning because the
obligations are test-contract and claim-boundary obligations rather than
closed-form scalar identities.

## Local checks run

- `git branch --show-current`.
- `git status --short --branch`.
- `git log --oneline -5`.
- Fixed-string `rg` checks for `\label{` and `\eqref{` in the P9 chapter.
- Text audit `rg` for:
  - `validat`;
  - `prove`;
  - `evidence`;
  - `target`;
  - `gradient`;
  - `tuning`.
- Label existence spot checks for:
  - `eq:bf-pf-predictive-law`;
  - `eq:bf-dpf-hmc-contract`.
- LaTeX build:
  - working directory: `docs`;
  - command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;
  - result: passed after required reruns;
  - final PDF size: 214 pages.
- Post-build log scan:
  - no undefined citations;
  - no undefined references;
  - no rerun warnings;
  - no multiply defined label warnings found by targeted `rg`.

## Text-audit findings

Flagged words remain only in controlled contexts:

- `validate` appears in the opening and boundary statements to deny prose-level
  validation and reserve the term for equation-local or model-risk processes.
- `evidence` appears in narrow diagnostic or promotion-threshold contexts.
- `target`, `gradient`, and `tuning` appear in the required order that places
  target/value-gradient checks before tuning.
- `prove` appears only in non-implication cells that state what a diagnostic
  does not prove.

No flagged phrase promotes PF-PF, relaxed DPF, learned OT, HMC, or banking
deployment beyond the recorded evidence.

## Layout inspection

The build passes, but P9 introduces dense table pressure.  P9-local warnings
include:

- underfull/overfull warnings in the equation-to-test matrix around
  lines 60--139, including an overfull alignment of about 23.8pt;
- underfull warnings in the controlled-example table around lines 156--193;
- overfull alignments of about 11.8pt in the diagnostic-contract table around
  lines 215--263;
- small overfull alignments of about 2.7pt in the source/implementation map
  around lines 270--312;
- underfull warnings in the promotion-threshold table around lines 330--350;
- page-header overfull warnings for the long chapter title.

These are not mathematical vetoes, but P10/P12 should revisit longtable layout
and chapter-title header pressure for P7--P9.

## Veto diagnostic review

- Diagnostics remain qualitative: no.
- Equations are not linked to tests: no.
- Failure interpretations are overbroad: no.
- Target-status distinctions disappear from implementation advice: no.
- Crosswalk remains too short to guide a coding agent: no.
- Student-baseline or controlled-baseline files edited or staged: no.
- Bibliography-spine source support described as ResearchAssistant-reviewed:
  no.
- MathDevMCP/manual diagnostic-obligation evidence omitted: no.
- Build or table-readability status omitted: no.

## Exit gate

Passed with layout caution.

The chapter can now serve as a concrete DPF implementation and reviewer-audit
checklist, with every major diagnostic tied to a mathematical layer, controlled
example, implementation quantity, and narrow failure interpretation.

## Next recommended phase

Proceed to P10:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-plan-2026-05-14.md`.
