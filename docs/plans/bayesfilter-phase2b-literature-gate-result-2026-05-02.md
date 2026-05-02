# BayesFilter Phase 2B Literature Gate Result

## Date
2026-05-02

## Purpose
This note records the first conservative literature-ingestion gate for the
BayesFilter monograph. The goal was not to draft chapters yet, but to decide
which claims are safe to use as foundations for a publication-quality monograph
and which claims still require primary-source, derivation, or implementation
audit.

The operating standard was intentionally conservative: exact arXiv source
packages and locally inspected derivations count as stronger evidence than PDF
metadata matches; ResearchAssistant download mismatches count as rejected
evidence; MathDevMCP inconclusive audits do not certify a derivation.

## Research Workspace
Persistent ignored workspace:

```text
/home/chakwong/BayesFilter/.research/ra-bayesfilter-monograph
```

Git policy:
- Commit only durable notes, source maps, and monograph source files.
- Do not commit `.research/`, downloaded papers, inbox PDFs, or generated
  `docs/main.pdf`.

## Sources Accepted For Bounded Claims

| Source | Local evidence | Status | Claims supported | Limits |
| --- | --- | --- | --- | --- |
| Betancourt, conceptual HMC, arXiv `1701.02434` | `source-fetch` exact arXiv LaTeX, paper id `betancourt2017conceptual_hmc` | Accepted for HMC foundations | Hamiltonian lift, Hamiltonian dynamics, mass/metric role, energy diagnostics including E-BFMI | Does not validate any particular BayesFilter likelihood implementation |
| Hoffman and Gelman, NUTS, arXiv `1111.4246` | `source-fetch` exact arXiv LaTeX, paper id `hoffman_gelman2014_nuts` | Accepted for NUTS foundations | Leapfrog update, no-U-turn stop criterion, dual-averaging adaptation, ESS definitions | Does not remove the need for finite differentiable log target tests |
| Hoffman et al., NeuTra, arXiv `1903.03704` | `source-fetch` exact arXiv LaTeX, paper id `hoffman2019neutra` | Accepted only as transport/geometry support | Transport maps may improve HMC geometry and can be discussed as acceleration infrastructure | Not a correctness substitute; failed or tail-poor maps can harm sampling |
| Ionescu et al., Matrix Backprop, arXiv `1509.07838` | `source-fetch` exact arXiv LaTeX, paper id `ionescu2015_matrix_backprop`; RA theorem extraction for `prop:svd` | Accepted for SVD/eigen differentiation risk | SVD variations contain factors like `1 / (sigma_i^2 - sigma_j^2)`; repeated or close singular/eigen values create non-differentiability or derivative blow-up risk | Supports caution and mitigation requirements, not a ready-made BayesFilter custom gradient |
| Kitagawa, Kalman score/Hessian, arXiv `2011.09638` | `source-fetch` exact arXiv LaTeX, paper id `kitagawa2020_kalman_score_hessian`; labels include `Eq_log-lk`, `Eq_gradient_ell`, and gradient filter recursions | Provisionally accepted for linear Gaussian score/Hessian literature | Linear Gaussian Kalman likelihood, score recursions, and Hessian extension are legitimate source material | Needs BayesFilter notation rewrite plus MathDevMCP/human derivation and code audit before certified formulas |
| Corenflos et al., differentiable PF via OT, arXiv `2102.07850` | `source-fetch` exact arXiv LaTeX, paper id `corenflos2021differentiable_pf_ot` | Accepted for differentiable PF frontier only | Entropy-regularized OT resampling gives a differentiable approximate PF with stated convergence/bias assumptions | Does not certify generic pseudo-marginal HMC or arbitrary differentiable resampling |
| Jacob, Chopin, and Robert discussion/commentary, arXiv `0911.0985` | `source-fetch` exact arXiv LaTeX, paper id `pmcmc_discussion_comments_candidate` | Context only | Useful commentary around PMCMC computational cost and model-choice issues | Not the primary Andrieu-Doucet-Holenstein PMCMC source |

## Rejected Or Unapproved Evidence

| Query or fetch | Result | Gate decision |
| --- | --- | --- |
| `download-paper --query "A Conceptual Introduction to Hamiltonian Monte Carlo Betancourt"` | Downloaded the rank-normalized R-hat paper, not Betancourt HMC | Rejected mismatch |
| `download-paper --query "1701.02434"` | Downloaded the same wrong R-hat paper | Rejected mismatch |
| `download-paper --query "Particle Markov chain Monte Carlo methods Andrieu Doucet Holenstein"` | Downloaded *Sequential Monte Carlo Methods in Practice* | Rejected mismatch |
| `download-paper --query "10.1111/j.1467-9868.2009.00736.x"` | Again resolved to *Sequential Monte Carlo Methods in Practice* | Rejected mismatch |
| `download-paper --query "Unscented Filtering and Nonlinear Estimation Julier Uhlmann"` | Downloaded an ensemble Kalman/unscented-transform paper, not Julier-Uhlmann | Rejected mismatch |
| `source-fetch --arxiv-id 1011.0419 --paper-id pseudo_marginal_efficiency_candidate` | Fetched an unrelated dissipative billiard paper | Rejected mismatch |
| Kitagawa PDF ingest | Text extraction is useful, but metadata candidates were low-confidence and mismatched | Use exact arXiv source fetch as authority; treat PDF metadata as unapproved |

## MathDevMCP Audit Result

MathDevMCP was used as a conservative reviewer, not as decoration.

Useful findings:
- `search-latex --root /home/chakwong/MacroFinance "gradient Hessian Kalman likelihood"`
  found the local solve-form likelihood, score, and Hessian section including
  `eq:solve_score_proved`, `eq:first_w_derivative`, and Hessian components.
- `audit-derivation-label eq:solve_score_proved --root /home/chakwong/MacroFinance`
  extracted the relevant derivation context but returned `inconclusive` because
  the matrix calculus obligations were outside the bounded backend.
- `audit-derivation-label eq:first_w_derivative --root /home/chakwong/MacroFinance`
  also returned `inconclusive` for the same reason.
- `audit-kalman-recursion /home/chakwong/MacroFinance/filters/solve_differentiated_kalman.py`
  extracted a large AST operation graph with Cholesky solves, trace-like helper
  calls, log determinant via Cholesky diagonals, Kalman updates, gradient, and
  Hessian operations. The strict required-operation query reported `mismatch`
  because the tool records `np.linalg.solve` as `inverse_or_solve`, and because
  explicit shape/covariance guards were not detected.

Interpretation:
- The MacroFinance analytic derivative path is a strong candidate source, but
  it is not yet mechanically certified for BayesFilter.
- The next audit phase should either formalize the matrix obligations into
  MathDevMCP-supported proof obligations or perform a documented human derivation
  review plus code-test parity review.
- The missing guard finding should be treated as a real production-readiness
  prompt: BayesFilter should require explicit shape, SPD, finite-value, and
  derivative-consistency guards at public filter boundaries.

## Claim Gate Decisions

Passed for foundational drafting:
- HMC foundations and diagnostics.
- NUTS algorithm and adaptation foundations.
- SVD/eigen differentiation pathology and the need for eigen-gap safeguards.

Partially unblocked:
- Analytic Kalman score/Hessian: literature source and local implementation
  source exist, but the BayesFilter monograph must not present final formulas as
  certified until derivation/code parity is audited.
- NeuTra/transport: safe to discuss as geometry-improvement infrastructure, not
  as a correctness or convergence guarantee.
- Differentiable PF with OT resampling: safe as a frontier method under its
  assumptions, not as a generic pseudo-marginal HMC result.

Still blocked:
- Primary PMCMC and pseudo-marginal support.
- Primary UKF/sigma-point literature support for Julier-Uhlmann style claims.
- Any claim that autodiff through SVD sigma-point filtering is industrial-scale
  HMC safe for NAWM-sized models.
- Any claim that current BayesFilter derivative formulas are peer-review ready
  without a dedicated derivation/code audit.

## Recommended Next Phase

Phase 2C should be an analytic-gradient and filter-contract audit before broad
chapter drafting:

1. Formalize the linear Gaussian solve-form likelihood, score, and Hessian in
   BayesFilter notation.
2. Cross-check the formulas against Kitagawa and MacroFinance.
3. Use MathDevMCP where encodable; record human proof obligations where not.
4. Audit implementation contracts for shape, SPD, finite log target, finite
   gradient, and static-shape/JIT constraints.
5. Write a small BayesFilter derivative certification appendix before importing
   large amounts of prose.

Explicit hypotheses for the next technical phase:
- H1: Analytic/custom-gradient filtering will be more robust for HMC than raw
  tape gradients through spectral decompositions when singular values or
  eigenvalues approach each other.
- H2: A solve-form or square-root linear Gaussian backend can be certified first
  and used as the regression oracle for nonlinear filters.
- H3: SVD sigma-point filtering should be treated as a numerically stabilized
  likelihood approximation whose gradient path requires safeguards or a custom
  derivative, especially for large DSGE systems.
- H4: Transport or NeuTra-like methods are geometry accelerators that must be
  validated against exact or controlled approximate likelihood targets, not a
  substitute for target correctness.

## Bottom Line

Phase 2B partially unblocks the monograph: we can now draft HMC/NUTS foundations
and SVD differentiation-risk sections with proper evidence. It does not yet
justify full content migration or industrial-scale BayesFilter claims. The next
best work is a focused analytic-gradient/filter-contract audit.
