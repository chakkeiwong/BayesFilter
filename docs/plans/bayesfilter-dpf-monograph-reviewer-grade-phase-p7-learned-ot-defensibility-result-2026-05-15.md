# Phase P7 result: learned and amortized OT defensibility expansion

## Date

2026-05-15

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane DPF chapter changes present before or during this phase:
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`.
- In-lane reviewer-grade planning/result artifacts are present as untracked
  files.
- User-owned in-lane governance edit remains present:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`.
- Out-of-lane student-baseline files remain present and were not touched.

## Prerequisite note

P7's subplan lists P3 as a prerequisite.  The user's explicit execution order
places P3 only after P7 and only if needed.  P7 proceeded under the user's
current order, using the passed P6 result and conservative target-status
language.  No learned-OT claim relies on a passed P3 gate.

## Allowed write set used

- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`.
- This result artifact.

No student-baseline file, git history operation, deletion, push, merge, rebase,
reset, or staging operation was performed.

## Chapter changes completed

P7 rewrote the learned-OT chapter so that learned transport is presented as a
teacher-student surrogate with explicit residual, extrapolation, and banking
evidence gates.

Completed items:

1. Added an object inventory covering categorical baseline, unregularized OT
   teacher, EOT teacher, finite Sinkhorn teacher, barycentric teacher output,
   learned student map, training distribution, loss function, permutation
   action, and residuals.
2. Defined teacher variants explicitly:
   - unregularized OT teacher `eq:bf-learned-ot-unregularized-teacher`;
   - EOT teacher `eq:bf-learned-ot-eot-teacher`;
   - finite Sinkhorn teacher `eq:bf-learned-ot-finite-teacher`;
   - barycentric teacher output `eq:bf-learned-ot-barycentric-teacher`.
3. Defined the learned student map with explicit dependence on particle count,
   state dimension, regularization parameter, and architecture choices.
4. Expanded supervised and empirical training objectives and stated that these
   objectives do not prove filtering error, posterior error, HMC correctness,
   or out-of-distribution validity.
5. Added permutation-equivariance and permutation-invariance equations and
   distinguished symmetry as necessary but not target-sufficient.
6. Expanded the residual hierarchy: teacher-object error, finite-solver error,
   supervised learning error, distribution shift, filtering-summary error,
   posterior error, and HMC value-gradient error.
7. Added failure regimes for sharp weights, high dimension, particle-count
   shift, epsilon shift, structural-model regions, learned-map collapse, and
   compiled repetition.
8. Added an evidence ladder for banking-scale use: teacher residual tests,
   student residual tests, filtering stress tests, posterior comparison,
   gradient/HMC checks, and governance limitations.
9. Added explicit non-claims: learned OT is not equivalent to the teacher, OT is
   not categorical resampling, finite Sinkhorn is not exact EOT, map residuals
   do not imply posterior error, symmetry does not imply target correctness,
   runtime improvement is not posterior evidence, and learned OT is not
   bank-facing or production-governance ready.

## ResearchAssistant evidence

ResearchAssistant remains available as a read-only/offline local workspace.

Query run:

- `learned amortized optimal transport Sinkhorn particle filter DeepSets Set
  Transformer differentiable particle filter`.

Result: no local paper summaries.  Therefore P7 does not describe learned-OT
support as ResearchAssistant-reviewed.  Source support remains
bibliography-spine plus local definitions, derivations, and claim-boundary text.

## Derivation-obligation audit

| Obligation | Chapter location | Method | Result |
|---|---|---|---|
| Teacher variant definitions | `eq:bf-learned-ot-unregularized-teacher`, `eq:bf-learned-ot-eot-teacher`, `eq:bf-learned-ot-finite-teacher` | Manual object/status audit | Passed. The chapter distinguishes unregularized OT, EOT, finite Sinkhorn, and the teacher actually used for training. |
| Barycentric teacher dimensions | `eq:bf-learned-ot-barycentric-teacher` | MathDevMCP typed diagnostic plus scalar normalization check | Passed manually with MathDevMCP assistance. Typed diagnostic was consistent; scalar check `q/(1/n)=n*q` was certified by SymPy for nonzero `n`. |
| Student-map scope | `eq:bf-learned-ot-student-map` | Manual dimensional and scope audit | Passed. Inputs include `X`, `w`, epsilon, particle count, dimension, and architecture. |
| Training objective scope | `eq:bf-learned-ot-training-objective`, `eq:bf-learned-ot-empirical-training-objective` | Manual objective audit | Passed. The text states what the objectives define and what they do not prove. |
| Permutation equivariance | `eq:bf-learned-ot-equivariance` | MathDevMCP typed obligation diagnostic plus manual review | Passed manually. MathDevMCP marked the obligation dimensionally consistent under the supplied context. |
| Permutation invariance | `eq:bf-learned-ot-invariance` | Manual symmetry audit | Passed. Scalar summary symmetry is separated from output-cloud equivariance. |
| Map residual definition | `eq:bf-learned-ot-residual` | MathDevMCP typed obligation diagnostic plus scalar residual check | Passed. MathDevMCP marked the residual equation dimensionally consistent; scalar check `T-T=0` was certified by SymPy as a sanity check. |
| Residual hierarchy and banking evidence | `tab:bf-learned-ot-residual-hierarchy`, `tab:bf-learned-ot-banking-evidence` | Manual claim-boundary audit | Passed. Every residual has a direct object, a "does not prove" boundary, and a next diagnostic. |

## Local checks run

- `git status --short --branch`.
- `rg` checks for:
  - `guarantee`;
  - `works`;
  - `valid`;
  - `correct`;
  - `target`;
  - `does not prove`;
  - `runtime`;
  - `bank`;
  - `production`;
  - `teacher`;
  - `student`;
  - `residual`;
  - `training distribution`;
  - `equivariance`;
  - `invariance`.
- ResearchAssistant query listed above.
- MathDevMCP:
  - `typed_obligation_label` for `eq:bf-learned-ot-equivariance`;
  - `typed_obligation_label` for `eq:bf-learned-ot-residual`;
  - `typed_obligation_label` for `eq:bf-learned-ot-barycentric-teacher`;
  - scalar `check_equality` calls for residual and barycentric-normalization
    sanity checks.
- LaTeX build:
  - working directory: `docs`;
  - command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;
  - result: passed after required reruns;
  - final PDF size: 212 pages.
- Post-build log scan:
  - no undefined citations;
  - no undefined references;
  - no rerun warnings;
  - no multiply defined label warnings found by targeted `rg`.

## Text-audit findings

Flagged words remain only in controlled contexts:

- `guarantee` appears in a denied target-guarantee statement.
- `works` appears only as the rejected phrase "`the learned map works`".
- `valid` and `correct` appear in correction, HMC-contract, or explicit
  non-claim language.
- `target`, `bank`, and `production` appear in evidence-gated language or
  explicit non-claims.
- `runtime` appears only with the statement that runtime is not target evidence.

No flagged phrase promotes learned OT beyond the recorded evidence.

## Layout inspection

The build passes, but the P7 rewrite adds table pressure.  P7-local warnings
include:

- underfull warnings in the object inventory around lines 46--88;
- page-header overfull warnings for the long chapter title;
- underfull warnings in teacher-variant table rows around lines 155--175;
- a small overfull paragraph around lines 246--250;
- underfull warnings in the residual hierarchy around lines 310--340;
- underfull/overfull warnings in the failure-regime table around lines 394--420;
- underfull warnings in the banking-evidence table around lines 461--489.

These are not mathematical vetoes, but P10/P12 should revisit longtable layout
for P7 and P8.

## Veto diagnostic review

- Teacher actually being approximated is ambiguous: no.
- Training distribution is vague: no.
- Learned residuals are not connected to filtering and HMC quantities: no.
- Banking credibility is argued from speed or smoothness: no.
- Architecture constraints are treated as software-only details: no.
- Student-baseline files edited or staged: no.
- Bibliography-spine source support described as ResearchAssistant-reviewed:
  no.
- MathDevMCP/manual derivation-obligation evidence omitted: no.
- Build status omitted: no.

## Exit gate

Passed with layout caution.

Learned OT is now clearly presented as a layered surrogate with explicit
teacher, student, residual, extrapolation, and target-status risks.

## P3 need assessment

P3 remains useful but is not required as an immediate blocker before P9.  The
current P4--P8 rewrites repeatedly state categorical, SMC, PF-PF, relaxed, and
learned-target boundaries.  No P7 or P8 veto requires a P3 backfill before the
debugging/verification contract phase.

## Next recommended phase

Proceed to P9 under the user's execution order:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p9-debugging-verification-contract-plan-2026-05-14.md`.
