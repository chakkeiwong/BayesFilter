# Phase P8 result: HMC target correctness and banking suitability expansion

## Date

2026-05-15

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane DPF chapter changes present before or during this phase:
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`.
- In-lane reviewer-grade planning/result artifacts are present as untracked
  files.
- User-owned in-lane governance edit remains present:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`.
- Out-of-lane student-baseline files remain present and were not touched.

## Governed-order exception

The P8 subplan lists a passed P7 result as a prerequisite.  The user's current
execution instruction explicitly orders P8 before P7.  This phase therefore
proceeded under the user's explicit order and did not rely on a passed P7 gate.
No claim in the rewritten chapter depends on P7 being complete.

## Allowed write set used

- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`.
- This result artifact.

No student-baseline file, git history operation, deletion, push, merge, rebase,
reset, or staging operation was performed.

## Chapter changes completed

P8 rewrote the HMC target chapter around named scalar targets, value-gradient
consistency, pseudo-marginal distinctions, and banking evidence gates.

Completed items:

1. Added a notation/object inventory for sampler coordinates, structural
   transforms, scalar log target, potential, momentum, Hamiltonian, integrator,
   gradient vector field, accept/reject value, and target-status category.
2. Expanded the HMC target contract with explicit differentiability and
   dimension assumptions:
   `g(u) = \nabla_u \ell_\star(u)`.
3. Separated exact-likelihood HMC, pseudo-marginal MCMC, noisy-gradient methods,
   delayed-acceptance or surrogate-corrected MCMC, relaxed-target HMC, and
   learned-surrogate HMC.
4. Rewrote the rung-by-rung analysis for EDH/LEDH, PF-PF, soft resampling,
   OT/EOT, and learned OT with scalar value, gradient path, target status, and
   missing validation.
5. Replaced risky promotion language.  The old phrases `first serious
   DPF-HMC candidate`, `strong target story`, and `production-ready HMC path`
   are no longer used.
6. Expanded the pseudo-marginal versus surrogate distinction so that unbiased
   estimator logic is not confused with deterministic relaxed-gradient HMC.
7. Expanded nonlinear DSGE limitations: determinacy and invalid regions,
   pruning/approximation convention, structural timing, latent-state semantics,
   and local Jacobian fragility.
8. Expanded MacroFinance limitations: latent dimension, long panels, missing
   or mixed-frequency data, volatility/noise curvature, compiled repeatability,
   and model-risk governance.
9. Added evidence gates for exploratory, credible research, bank-facing
   research, and production/governance language.
10. Added explicit non-claims: no PF-PF validation, no transfer from
    pseudo-marginal validity to relaxed gradients, no nonlinear DSGE or
    MacroFinance validation, and no production or bank-governance approval.

## ResearchAssistant evidence

ResearchAssistant remains available as a read-only/offline local workspace.

Query run:

- `Hamiltonian Monte Carlo pseudo-marginal surrogate noisy gradient MCMC
  differentiable particle filter target correctness`.

Result: no local paper summaries.  Therefore P8 does not describe HMC,
pseudo-marginal, or surrogate-HMC support as ResearchAssistant-reviewed.
Source support remains bibliography-spine plus local derivation and local
claim-boundary text.

## Derivation-obligation audit

| Obligation | Chapter location | Method | Result |
|---|---|---|---|
| Value-gradient consistency contract | `eq:bf-dpf-hmc-contract` | MathDevMCP label extraction and typed obligation diagnostic plus manual review | Passed manually. MathDevMCP confirmed the differentiability constraint is explicitly satisfied after the assumption sentence was added, but routed the vector equation to human review because it is a definition-level target contract. |
| Potential/log-target sign convention | target-object inventory | MathDevMCP scalar equality | Scalar check `-(-l(u)) = l(u)` certified by SymPy. |
| Posterior factorization from log components | target-object setup | MathDevMCP scalar equality | Scalar check `exp(logp + logL + logJ) = exp(logp)*exp(logL)*exp(logJ)` certified by SymPy. |
| Pseudo-marginal versus surrogate distinction | Section `sec:bf-dpf-hmc-pm-surrogate` | MathDevMCP proof-obligation attempt plus manual review | Passed manually. MathDevMCP marked the conditional-expectation statement not encodable for SymPy, which is appropriate because the obligation is measure-theoretic rather than scalar algebra. |
| Rung target-status table | Table `tab:bf-dpf-hmc-rungs` | Manual target-status audit | Passed. Each rung names scalar value, gradient path, target status, and missing validation. |
| Banking evidence gates | Table `tab:bf-dpf-hmc-banking-evidence` | Manual claim-boundary audit | Passed. Production/governance language appears only as a claim-level category and explicit non-claim. |

## Local checks run

- `git branch --show-current`.
- `git status --short --branch`.
- `git log --oneline -5 --decorate`.
- `rg` checks for:
  - `HMC-ready`;
  - `production-ready`;
  - `first serious`;
  - `strong target story`;
  - `validated`;
  - `suitable`;
  - `correct`;
  - `exact`;
  - `pseudo-marginal`;
  - `surrogate`;
  - `target`.
- ResearchAssistant query listed above.
- MathDevMCP:
  - `extract_latex_context` for `eq:bf-dpf-hmc-contract`;
  - `typed_obligation_label` for `eq:bf-dpf-hmc-contract`;
  - scalar `check_equality` calls;
  - conditional-expectation proof-obligation attempt for the pseudo-marginal
    distinction.
- LaTeX build:
  - working directory: `docs`;
  - command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;
  - result: passed after required reruns;
  - final PDF size: 209 pages.
- Post-build log scan:
  - no undefined citations;
  - no undefined references;
  - no rerun warnings;
  - no multiply defined label warnings found by targeted `rg`.

## Text-audit findings

Flagged words remain only in controlled contexts:

- `correct` appears in HMC-correction and non-claim language.
- `exact` appears in exact-likelihood categories, recovery-case restrictions,
  and explicit denials of exactness for relaxed or learned rungs.
- `validated` appears only in an explicit non-claim.
- `suitable` appears only in a disallowed-shortcut cell.
- `production` and governance language appears only as evidence-gated claim
  categories or explicit non-claims.

No flagged phrase is used to promote PF-PF, relaxed DPF, learned OT, or banking
deployment beyond the recorded evidence.

## Layout inspection

The build passes, but the P8 rewrite adds table pressure.  P8-local warnings
include:

- underfull warnings in the object inventory around lines 61--96;
- overfull page-header warnings for the long chapter title;
- overfull/underfull warnings in the rung target-status longtable around
  lines 273--316;
- a small overfull paragraph around lines 368--371;
- underfull warnings in the method-family comparison around lines 427--459;
- underfull warnings in the banking-evidence table around lines 474--502;
- a small overfull paragraph around lines 536--538.

These are not mathematical vetoes, but P10/P12 should revisit table layout and
chapter-title header pressure.

## Veto diagnostic review

- Method labels substitute for scalar target definitions: no.
- PF-PF is oversold: no.
- Relaxed or learned DPF is implied to target the original posterior: no.
- Banking claims exceed evidence: no.
- Pseudo-marginal and differentiable-surrogate logic remain blurred: no.
- Student-baseline files edited or staged: no.
- Bibliography-spine support described as ResearchAssistant-reviewed: no.
- MathDevMCP/manual derivation-obligation evidence omitted: no.
- Build status omitted: no.

## Exit gate

Passed with layout caution.

A skeptical HMC-literate reviewer can now identify the scalar target sampled by
each rung, the correction mechanism if one is claimed, and the evidence still
missing before banking or production language would be defensible.

## Next recommended phase

Proceed to P7 under the user's execution order:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p7-learned-ot-defensibility-plan-2026-05-14.md`.
