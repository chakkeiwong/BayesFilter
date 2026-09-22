# Observation-aware TT repair: bounded end-to-end master run (2026-09-13)

## Research question

Can the missing Zhao--Cui observation-aware lifecycle be executed on the
existing scalar stochastic-volatility fixture: retain a posterior particle
bank, use an SGQF auxiliary update from the current observation, fit a fixed
square-root TT to the one-step transition/observation target, construct a TT
conditional proposal, and apply the exact transition/observation importance
correction over several time steps?

The run is a bounded diagnostic of the lifecycle mechanics.  It is not a
paper-scale Zhao--Cui production route and cannot establish TT superiority,
posterior correctness, HMC readiness, or source-faithful high-dimensional
performance.

## Evidence contract

* **Target and data:** the repository `StochasticVolatilitySSM` with fixed
  `gamma=0.65`, `beta=0.4`, `sigma=1`, and a deterministic simulated scalar
  observation sequence of length four.
* **Comparator:** the same recursion with a transition-prior Gaussian proposal
  (the unadjusted SIR proposal).  Both arms use the same retained bank and
  exact correction formula where applicable.
* **Promotion/pass criterion:** every TT step returns finite particles,
  positive normalized weights, a finite TT fit residual, a monotone conditional
  CDF whose inverse reconstructs each requested probability, an ESS above 4
  particles, and a finite exact importance correction; the fit residual must
  be below 1.0; the SGQF guide must change when the
  observation is changed while the predictive moments are fixed.
* **Vetoes:** non-finite values, non-positive covariance, failed CDF monotonicity
  or normalization, failed TT fit status, shape/hash mismatch, or a missing
  manifest field veto the run.
* **Explanatory diagnostics:** effective sample size, TT holdout residual,
  guide mean shift, proposal/transition log-density gap, and comparison of
  posterior means are descriptive only.
* **Heuristic adversaries:** prior-proposal SIR and the SGQF-only moment update
  are constructed cheap baselines.  They are evaluated at ordinary, near-zero,
  and tail observations; no baseline is tuned against the TT output.
* **Non-claims:** one seed, four steps, and a scalar state do not support a
  ranking, convergence, production readiness, or a claim that the finite
  conditional interpolant is the author’s full paper-scale KR implementation.
* **Artifact:** each launch writes a unique directory under
  `docs/benchmarks/artifacts/observation_aware_tt_repair_full_master_20260913/`
  containing `result.json`, `run_manifest.json`, per-step diagnostics, and the
  exact command log.

## Implementation classification

The SGQF likelihood-weighted guide, its retained-ancestor row cloud in the TT
regression, fixed-rank `FixedTTFitter` call, retained bank, and exact
transition/observation ratio follow the repository contracts.
The scalar conditional sampler uses a numerically integrated squared-TT density
on a fixed physical grid with piecewise-linear CDF inversion and a declared
SGQF defensive mixture.  That sampler is an explicit
`extension_or_invention` diagnostic used to make the lifecycle executable; it
does not close the source-faithfulness gate for the full Zhao--Cui route.

## Default and assumption audit

The 9-point standard-normal GHQ rule is inherited from the repository SGQF
implementation and is recorded as a baseline quadrature choice; its first
diagnostic is the observation-response test.  A degree-4 Legendre basis,
rank-2 TT, 129-point conditional grid, and 32 retained particles are bounded
fixture settings chosen for a short CPU run; the regular fixed design spans the
complete `[-8,8]^2` basis domain, while fit residual and CDF checks expose
failure from the remaining conveniences.  The 70% SGQF defensive mixture is a
declared robustness extension, not a tuned promotion default.  The simulated
data seed and proposal quantiles are fixed so that a rerun is reproducible.

Systematic resampling is deterministic and identity-preserving, triggered only
when ESS falls below half the retained bank; the same trigger is applied to the
prior-proposal comparator.

## Skeptical audit and pre-mortem

Before execution we checked the call chain to `FixedTTFitter.fit`,
`FunctionalTT.evaluate`, `StochasticVolatilitySSM.transition_log_density`,
`StochasticVolatilitySSM.observation_log_density`, and
`tf_standard_normal_ghq_level_rule`.  The old master stopped after the SGQF
smoke and never retained a recursive bank or fitted a proposal; this program
adds those stages explicitly.  A successful command could still mislead if the
low-rank fit is too poor, if a bounded grid truncates the Gaussian tails, or if
the discrete particle ratio is read as a continuous-density theorem.  The
earliest checks for those failures are the holdout residual, CDF/normalization
veto, and comparison with the prior-proposal SIR baseline.  A failure of this
fixture rejects the implementation attempt, not the broader research idea.

## Budget and stop conditions

Each launch is bounded to at most four filtering steps, 32 particles, 129
conditional grid points, two TT sweeps, and no network or package changes.
Stop immediately on any veto or on a nonzero subprocess exit.  CPU-only
execution is deliberate; the artifact records `CUDA_VISIBLE_DEVICES=-1`;
`jit_compile=False` is an explicit bounded diagnostic exception.  Implementation
repairs were kept under this same bound and preserved in versioned run
directories; `run-15/failure.txt` records the only failed final-stage attempt,
and `run-20` is the terminal run used for interpretation.
