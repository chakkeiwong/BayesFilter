# BayesFilter literature blocker register

## Date
2026-05-03

## Purpose
Record the remaining citation and source-support blockers after the first
BayesFilter monograph writing pass. This note is a continuation of:

- `docs/plans/bayesfilter-literature-seed-list-2026-05-02.md`
- `docs/plans/bayesfilter-phase2b-literature-gate-result-2026-05-02.md`

It is not a bibliography and should not be treated as claim support by itself.

## Current accepted bounded sources

Already accepted for bounded drafting:

- Betancourt HMC foundations, arXiv `1701.02434`
- Hoffman--Gelman NUTS foundations, arXiv `1111.4246`
- Hoffman et al. NeuTra, arXiv `1903.03704`, only as transport/geometry support
- Ionescu et al. Matrix Backprop, arXiv `1509.07838`, for SVD/eigen derivative
  risk
- Kitagawa Kalman score/Hessian, arXiv `2011.09638`, provisionally for
  literature support but not final BayesFilter-certified formulas
- Corenflos et al. differentiable PF via OT, arXiv `2102.07850`, as frontier
  support only

## Blockers

### UKF and sigma-point primary support

Blocked claims:
- detailed historical claims about the UKF;
- claims about exact polynomial order beyond what is locally derived and
  tested;
- claims that a specific sigma-point rule is generally HMC-safe.

Required action:
- fetch and verify the primary Julier--Uhlmann source or an equivalent primary
  source package;
- write a claim-neighborhood note for sigma-point moment matching and limits;
- update `docs/references.bib` only after metadata is verified.

### PMCMC and pseudo-marginal methods

Blocked claims:
- particle-filter likelihood estimators define valid pseudo-marginal targets;
- pseudo-marginal HMC correctness;
- unbiased particle likelihood claims beyond conservative local wording.

Required action:
- fetch the Andrieu--Doucet--Holenstein PMCMC primary source without metadata
  mismatch;
- fetch pseudo-marginal foundations, such as Andrieu--Roberts, with verified
  metadata;
- write a note distinguishing PMMH, particle Gibbs, differentiable particle
  filters, and any HMC-specific claim.

### Square-root filtering bibliography

Blocked claims:
- final literature prose comparing Cholesky, QR, UD, and SVD square-root
  filters;
- historical priority claims.

Required action:
- ingest primary square-root Kalman/UKF sources;
- map claims to BayesFilter's implementation-level backend contract.

### Analytic Kalman derivative certification

Blocked claims:
- BayesFilter score/Hessian formulas are peer-review ready;
- the current MathDevMCP audit mechanically proves the MacroFinance
  derivations.

Required action:
- formalize matrix-calculus obligations not covered by the current tool;
- audit code/test parity against MacroFinance implementations;
- add explicit shape, SPD, finite-value, and covariance guards where the public
  BayesFilter API requires them.

### Industrial SVD-HMC safety

Blocked claims:
- SVD sigma-point tape gradients are safe for NAWM-sized HMC;
- NK SVD-HMC has converged;
- EZ, SGU, Rotemberg, or NAWM inherit NK production-XLA evidence.

Required action:
- complete strict LGSSM and nonlinear SSM recovery;
- pass NK clean stress gate;
- run NK medium and strict convergence audits;
- add model-specific production-XLA and clean-stress gates before any other
  DSGE model is promoted.

## Drafting policy until blockers close

- Keep particle-filter and sigma-point chapters conservative.
- Mark frontier methods as frontier methods.
- Use project-decision language for TFP NUTS and BayesFilter-owned HMC.
- Preserve reset-memo links and source-map provenance.
- Do not remove blocker language merely to make the monograph look complete.
