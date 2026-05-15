# Reviewer-grade DPF monograph full re-audit result

## Date

2026-05-15

## Governing request

The user requested a fresh audit of the whole reviewer-grade DPF monograph work,
with modifications where missing gaps are identified, and with MathDevMCP used
where possible.

Governing reviewer-grade files:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-derivation-audit-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-hostile-reader-audit-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-final-readiness-report-2026-05-15.md`

## Branch and worktree classification

- Branch at audit start: `main`.
- Recent history included merge commit `a75e18f Merge branch 'dpf-monograph-rebuild'`.
- In-lane modified DPF chapter files present before this re-audit:
  - `docs/chapters/ch19_particle_filters.tex`;
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`;
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`.
- In-lane reviewer-grade plan/result files are untracked and remain in scope.
- Out-of-lane dirty/untracked student-baseline and controlled-baseline files
  were present and were not edited, staged, restored, or used as monograph
  evidence.

## Tool status

MathDevMCP:

- `doctor` succeeded.
- Available backend: SymPy.
- Unavailable: LaTeXML, Pandoc, Lean, Sage, Lean Dojo.
- Consequence: MathDevMCP provided parser, provenance, dimensional, and
  obligation diagnostics, but no formal proof certificate is claimed.

ResearchAssistant:

- Workspace: `/home/ubuntu/python/ResearchAssistant`.
- Mode: read-only, offline.
- Providers and live LLM calls disabled.
- Search for DPF, particle-flow, PF-PF, Sinkhorn, HMC, and pseudo-marginal
  material returned no local paper summaries.
- Consequence: source support remains bibliography-spine plus local derivation,
  not ResearchAssistant-reviewed paper evidence.

## Fresh audit checks run

- `git branch --show-current`.
- `git status --short`.
- `git log --oneline -5`.
- DPF chapter and reviewer-grade plan inventory with `rg --files`.
- Chapter line-count and diff-stat checks.
- Direct label, equation, align, and proposition inventories across the DPF
  chapter block.
- Overclaim search for terms including `exact`, `unbiased`, `consistent`,
  `validated`, `robust`, `HMC-ready`, `production`, `guarantee`, `credible`,
  `bank-facing`, `ResearchAssistant-reviewed`, and `first serious`.
- Direct source reading of Chapters 20--26:
  - particle-filter foundations;
  - particle-flow foundations;
  - PF-PF proposal correction;
  - differentiable resampling and OT;
  - learned transport operators;
  - HMC target correctness;
  - debugging verification contract.
- Final LaTeX build from `docs`:
  `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`.
- Final log scan for unresolved references, unresolved citations, multiply
  defined labels, and rerun warnings.
- Final overclaim scan across the DPF chapter block for `HMC-ready`,
  `production-ready`, `bank-facing`, `valid posterior`, `guarantee`,
  `unbiased gradient`, `exact posterior`, `first serious`, and
  `ResearchAssistant-reviewed`.
- MathDevMCP diagnostic audits on:
  - `prop:bf-pf-bootstrap-likelihood-status`;
  - `eq:bf-pff-log-homotopy-derivative`;
  - `eq:bf-pff-homotopy-cov-derivative`;
  - `eq:bf-pfpf-weight`;
  - `eq:bf-dr-soft-test-bias`;
  - `eq:bf-dr-eot-stationarity`;
  - `eq:bf-dr-barycentric-map`;
  - `eq:bf-dpf-hmc-contract`.

## Repairs made in this re-audit

### Bootstrap likelihood-estimator proof

File: `docs/chapters/ch19_particle_filters.tex`.

The proof of `prop:bf-pf-bootstrap-likelihood-status` was strengthened.  The
previous proof stated the standard Feynman--Kac induction but compressed the
operator-level step.  The revised proof now defines:

- the bootstrap Feynman--Kac operator `Q_t`;
- the unnormalized filtering functional `gamma_t`;
- the relation `gamma_t(varphi)=gamma_{t-1}(Q_t varphi)`;
- the pre-propagation empirical measure after any resampling decision;
- the conditional unbiasedness identity for resampling;
- the induction for
  `E[(prod_s Z_s^N) hat pi_t^N(varphi)] = gamma_t(varphi)`;
- the likelihood-unbiasedness conclusion by taking `varphi = 1`.

This does not promote log-likelihood, score, pathwise-gradient, or HMC claims.
It makes the normalizing-constant argument more self-contained for a rigorous
non-specialist.

### EDH matrix assumptions and implementation enumeration

File: `docs/chapters/ch19b_dpf_literature_survey.tex`.

The EDH Gaussian-closure section now states the positive-definiteness and
dimension assumptions that make the inverse and matrix products meaningful:

- `P_{t|t-1}` and `R_t` are symmetric positive definite;
- `H_t`, `P_{t|t-1}`, and `R_t` have conformable dimensions;
- implementation should use solves and conditioning diagnostics rather than
  treating displayed inverses as numerical instructions.

A prose defect was also corrected: the implementation paragraph now says
"four tasks" before listing four tasks.

### PF-PF change-of-variables assumptions

File: `docs/chapters/ch19c_dpf_implementation_literature.tex`.

The flow-map assumption was tightened from a generic differentiable bijection
to the local diffeomorphism condition needed by the density formula:

- `D Phi_t(x_{t,0})` is an `n_x x n_x` nonsingular matrix along proposal paths;
- the change-of-variables calculation applies on the support actually used by
  the proposal.

This directly answers MathDevMCP log-determinant/domain diagnostics.

### Soft-resampling bias derivation

File: `docs/chapters/ch32_diff_resampling_neural_ot.tex`.

The bias derivation around `eq:bf-dr-soft-test-bias` now explicitly introduces
`d_i=(1-alpha)(bar x-x_i)`, expands `varphi(x_i+d_i)` with a second-order
remainder, and states the bounded-Hessian condition behind the
`O((1-alpha)^2)` term.

The soft-resampling differentiability boundary was also tightened.  The chapter
now states that the interpolation term is smooth, but the sampled ancestor
`a_j` remains a discrete random index unless the method also replaces or
conditions on that draw.  Therefore soft resampling is described as a smoother
surrogate path, not as an exact pathwise derivative of categorical resampling.

### EOT/Sinkhorn KKT derivation

File: `docs/chapters/ch32_diff_resampling_neural_ot.tex`.

The EOT stationarity derivation now writes the full Lagrangian with row and
column constraints before differentiating with respect to `Pi_ij`.  This makes
the sign convention and multiplier role explicit before the Gibbs/Sinkhorn
factorization.

### Barycentric projection dimensions

File: `docs/chapters/ch32_diff_resampling_neural_ot.tex`.

The barycentric projection section now states the general formula
`tilde x_j = u_j^{-1} sum_i Pi_ij x_i` for a positive column mass, then derives
the displayed `N sum_i Pi_ij x_i` formula as the equal-weight case
`u_j=1/N`.

### HMC coordinate and mass-matrix assumptions

File: `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`.

The HMC target-contract section now states:

- the parameter transform is differentiable on the sampler-valid region;
- `DT(u)` is the square Jacobian for the displayed change-of-variables term, or
  must be replaced by the appropriate volume term for a more general
  parameterization;
- the ordinary separable HMC mass matrix is symmetric positive definite with
  dimension `d_theta x d_theta`.

This answers MathDevMCP dimension and inverse diagnostics without strengthening
any DPF-HMC validity claim.

## Mathematical audit conclusion

No fatal mathematical overclaim was found after the repairs.  The central
status boundaries remain conservative:

- exact statements are limited to model identities, change-of-variables
  identities, EOT/OT objects for their declared objectives, and linear-Gaussian
  recovery cases;
- the bootstrap particle filter is presented as likelihood-unbiased, not as
  log-likelihood-, score-, pathwise-gradient-, or HMC-unbiased;
- EDH and LEDH remain Gaussian/local-linear closure methods outside exact
  recovery cases;
- PF-PF restores an explicit proposal-to-target density ratio for the declared
  flow proposal, not exact nonlinear filtering or production HMC status;
- soft resampling, OT/EOT, finite Sinkhorn, and learned OT remain relaxed,
  numerical, or learned surrogate layers;
- HMC claims are organized around the scalar value-gradient contract and target
  status;
- banking and production language remains evidence-gated and explicitly
  unachieved.

## Build and final scan result

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed
  successfully from `docs`.
- Output: `docs/main.pdf`, 220 pages, 925136 bytes on the final run.
- Final log scan for unresolved references/citations and rerun warnings found
  no substantive warning.  The only matching line was the package-identification
  line for `rerunfilecheck`.
- Remaining LaTeX log noise consists of overfull/underfull box warnings,
  primarily from dense longtables and long running headers.  This remains a
  presentation issue, not a mathematical blocker.
- The final DPF chapter overclaim scan found only negative, boundary, or
  evidence-gating contexts for risky phrases such as `HMC-ready`,
  `bank-facing`, and `exact posterior`; no promotional use was found in the
  DPF chapter block.

## Residual limitations

- MathDevMCP did not certify formal proofs.  Its role in this re-audit was
  diagnostic because only SymPy was available and the obligations involve
  measure-theoretic filtering, transport, and HMC target interpretation.
  A final MathDevMCP pass on the repaired labels still returned
  diagnostic-only/unverified statuses, with human-formalization and
  dimension/shape obligations for measure-theoretic and target-contract
  statements.
- ResearchAssistant did not provide local reviewed source summaries.  No claim
  should be described as ResearchAssistant-reviewed.
- No implementation tests were run in this re-audit.  The debugging chapter
  defines the required diagnostics but does not execute them.
- The PDF layout caution from P12 remains: the DPF block is table-heavy, with
  dense longtables and long running headers in later chapters.  This is a
  presentation caution, not a mathematical veto.

## Exit decision

Passed for rigorous internal reading and manual-audit purposes, with documented
limitations.

The re-audit identified and repaired several rigor and presentation gaps.  No
remaining blocker was found for a rigorous internal reading of the DPF monograph
block, provided the limitations above are kept attached to the work.  This exit
decision does not close source-review, formal-proof, implementation-validation,
posterior-comparison, or bank-facing evidence gates.
