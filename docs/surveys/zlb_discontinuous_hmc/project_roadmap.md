# Project Roadmap: Nonlinear ZLB Inference

## Objective

Build a BayesFilter-owned TensorFlow/TFP inference package that preserves the
declared hard-ZLB target in nonlinear state-space models. MacroFinance supplies
the application model; Dynare/OccBin supplies the piecewise-linear benchmark.
The `dsge_hmc` package exists at `/home/ubuntu/workspace/python/src/dsge_hmc`
(corrected 19 August 2026: the originally named `/home/ubuntu/workspace/dsge_hmc`
path is wrong), but its validated BGS restricted surface is a no-binding
`r = rn` placeholder; no OBC/ZLB integration is planned until a true BGS
OBC/ZLB target discharges the survey's Section 14.5 checklist against the
actual source per Section 14.6. Survey Sections 13 and 14 (19 August 2026)
are the binding application contracts: WP0 is discharged for MacroFinance by
Section 13, and the future dsge_hmc true-OBC model must discharge the
Section 14.5 checklist, including a computed LCP uniqueness verdict and a
declared selection law, before WP1 starts for it.

## Work packages

### WP0: Statistical target contract

For each model, record the transition and observation densities, ZLB inequality,
regime timing, equilibrium solver, one-sided density behavior, and solution law
when roots are multiple. Classify the target as kink, deterministic jump,
stochastic regime, or undefined multiple-root model.

**Exit criterion:** the finite log-density program and its base measure are
written without an implicit solver tie-break. A deterministic regime-only move
must be shown to have zero support in the toy contract.

### WP1: Dynare/OccBin and PKF benchmark

Port the conditional matrices and regime iteration into a BayesFilter reference
lane and compare them with the inspected Dynare example and
`+occbin/kalman_update_algo_3.m`.

**Tests:** innovation, covariance, gain, log-likelihood contribution, inferred
shock, regime inequality, missing observation, fixed-point cycle, and multiple
fixed-point fixtures.

**Exit criterion:** same-object agreement with Dynare on frozen synthetic
fixtures. This establishes only the piecewise-linear benchmark.

### WP2: Nonlinear particle authority

Implement the resample-propagate bootstrap filter, retained genealogy, an
unbiased likelihood estimate, PMMH, and PGAS. Add the Aruoba COPF only for models
that satisfy its PLC canonical assumptions. For MacroFinance, the
claim-bearing authority is the bootstrap particle filter for the
`mf_c1_k40_hardmax` target (exact Gaussian transition draws, exact
observation-density weights; survey Sec. 13.4); the per-step cell
decomposition of survey eqs. (84)--(87) is an ideal fully adapted proposal
identity whose implementation requires controlled polytope-probability and
truncated-draw numerics and is a proposal/diagnostic layer, not the default
authority. The joint kinked-target HMC route of survey Sec. 13.7(iv) targets
the same exact model and must agree with the bootstrap authority on matched
fixtures. Optionally
add the Section 4.3 branch-split mixture UKF as a fast approximate diagnostic
and as a proposal inside the corrected particle filter; the particle weights
must keep the exact shock density in the numerator as in survey eqs.
(52)--(53), and a UKF-only likelihood is never claim-bearing.

**Tests:** exact enumeration for a tiny switching model; analytic Kalman
likelihood in a linear-Gaussian limit; repeated likelihood mean with Monte Carlo
uncertainty; PGAS smoothing frequencies versus enumeration; zero-likelihood and
particle-collapse cases; truncated-moment unit tests against the affine closed
forms (21)--(28).

**Exit criterion:** bounded toy evidence that PMMH and PGAS target the enumerated
posterior within predeclared Monte Carlo uncertainty.

### WP3-D: Deterministic-threshold kernel

Start with an exact non-gradient joint-state kernel or PMMH. Develop an event
oracle that returns the earliest full-model regime crossing and both one-sided
posterior values. Only then implement reflection/refraction or Laplace DHMC.

**Tests:** event time against analytic affine cases; multiple crossings in one
step; grazing; reversible forward/backward trajectory; one-sided energy;
Jacobian/volume check of the composed flight--event--flight step (the isolated
refraction map is not volume preserving; survey Sec. 6.1); comparison with
exact toy posterior.

**Veto:** unavailable first event, nondeterministic branch solve, unmodelled
multiple root, or energy computed from only the policy-rate equation.

### WP3-S: Stochastic-regime kernel

Compose fixed-path HMC with a discrete Metropolis or PGAS path update. Establish
this baseline before implementing Zhou-style mixed HMC.

**Tests:** detailed balance by finite-state flux; exact enumeration; block and
single-site path proposals; no-movement and zero-support checks; equivalence of
the mixed-HMC and baseline stationary marginals on the toy model.

**Veto:** reusing this kernel for a deterministic indicator without a supported
joint proposal.

### WP4: Approximation and gradient arms

Implement softplus, Concrete, stop-gradient score, or entropy-regularized
transport only as separately named targets/estimators. Measure approximation
error against the exact toy authority. If an active DPF transport route is
created, it must use BayesFilter policy
`dpf_transport_exact_divisor_cap3000_v1`.

### WP5: MacroFinance application

Freeze a model-specific target contract, parameterization, data window, priors,
and diagnostic thresholds. Tune within the execution scope, then run a held-out
claim campaign under a versioned output root and bounded budget.

## Engineering constraints

- Algorithmic code uses TensorFlow and TensorFlow Probability; NumPy is limited
  to explicitly diagnostic/reference files.
- XLA JIT defaults on for algorithmic and differentiable routes.
- Serious execution targets GPU and configures verified memory growth before
  device initialization.
- Every serious run records command, environment, seeds, hardware, wall time,
  target identity, model identity, and versioned artifact path.
- New work stays in BayesFilter until explicit integration authority is given.

## Recommended first vertical slice

Use a one-dimensional nonlinear state with a deterministic threshold and a
Gaussian observation, chosen so both one-sided likelihoods and the exact
posterior can be integrated numerically. Implement PKF, bootstrap PF/PMMH, and a
non-gradient joint-state baseline first. In parallel, use a second toy model
with a stochastic two-state regime to validate PGAS and mixed HMC. Keeping these
two toys separate is the simplest way to prevent a support mistake from being
hidden by code reuse.
