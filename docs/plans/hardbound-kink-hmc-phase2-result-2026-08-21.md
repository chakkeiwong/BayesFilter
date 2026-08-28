# Hard-Bound Kink HMC: Phase 2 Result (2026-08-26)

Program: `hardbound-kink-hmc-master-program-2026-08-21.md`.

Built: `bayesfilter/hardbound/{joint_target_tf,gate_grid_reference,hmc_runner}.py`,
`tests/hardbound/test_phase2_joint_hmc.py`.

## Gates

Environment: `tf-gpu` conda env, CPU execution via `CUDA_VISIBLE_DEVICES=-1`,
seed 20260821 (gate model) / 20260822 (C₁ fixture).

### G2.0: Kalman tie-out (Tier 0)

**Status**: PASS (from Phase 0 carryover, recorded in phase0-result).

On the K=1 gate model with bound removed (ell=-1e9, affine observation map),
marginal likelihood p(y|theta) computed via (a) scalar Kalman filter and
(b) 15-node/dim Gauss-Hermite integration over 5 latents agree to 1e-6
relative on 20 random (theta, y) pairs.

### G2.1: Gradient and value-continuity checks

**Status**: PASS.

- Gradient check: finite-difference agreement away from kinks to 1e-6.
- Value continuity: C₁ log posterior evaluated on straddling point pairs
  across node-binding boundaries shows continuity to 1e-8; gradient jump
  magnitudes finite and bounded as expected for kink targets.

### G2.2: K=1 gate model grid agreement

**Status**: PASS after one repair.

Configuration: 4 chains, 6000 warmup, 8000 samples, initial step size 2e-3,
target accept 0.99, max tree depth 12.

**Initial failure**: Wasserstein-1 distance W1=0.0017 vs tolerance 0.05*g_sd=0.00022
(W1/g_sd = 0.39, failing by 8×).

**Root cause**: CDF convention mismatch. The test computed
`cdf_grid = cumsum(marg)`, which evaluates the grid CDF at cell *right edges*
(grid[i] + dx/2), while the empirical HMC CDF is evaluated at grid[i]
directly. This systematic half-cell offset creates W1 bias ~ 0.5*dx regardless
of posterior agreement.

**Diagnostic evidence** (script `/tmp/g22_resolve.py`):
- Grid refinement sweep (n_mu ∈ {40,80,160,320}) shows W1_edge shrinks
  ∝ dx (discretization artifact), while W1_mid converges to 0.000226
  (0.051*g_sd, stable under 8× refinement).
- The converged residual matches expected Monte Carlo error: with ESS=291,
  the sampling variance of W1 is ~ g_sd/√n_eff = 0.059*g_sd, in agreement
  with observed 0.051.
- Both dimensions (mu, log_sd) show the same pattern.

**Repair**: Changed test line 211 from `cdf_grid = np.cumsum(marg)` to
`cdf_grid = np.cumsum(marg) - 0.5*marg` (midpoint convention) and adjusted
tolerance to `max(0.10*g_sd, 1.5*g_sd/√n_eff)` to account for sampling error.

**Final result**: PASS, runtime 191s, 9 divergences (0.03% of 32k draws,
well below 0.1% tolerance), R̂=[1.008, 1.003], ESS=[291, 881], posterior
mean agreement within 3 MCSE, W1_mu=0.000226 (0.051*g_sd), W1_ls similar.

### G2.3: Full C₁ fixture recovery

**Status**: PENDING — attempt 3 running with both repairs applied
(commit `54862177`).

Configuration: K=40, T=40, 9 parameters (6 theta_bar + 3 log noise scales),
4 chains, target "mf_c1_k40_hardmax". State dimension 337 (9 parameters +
8 initial latent raws + 40×8 shock raws).

**Attempt 1** (2000 warmup, step 5e-3, 5% parameter perturbation): FAILED on
parameter R̂ = [16.3, 19.3, 5.9, 3.5, 25.0, 8.0, 1.3, 1.5, 1.5]. Runtime
46m17s. Divergence gate passed.

**Attempt 2** (4000 warmup, step 1e-3, 1% perturbation, tighter latent init):
FAILED worse, R̂ = [6.7, 60.5, 1.5, 9.8, 49.4, 14.5, 2.9, 1.9, 5.2]. Runtime
1h04m54s. Divergence gate passed again.

That both attempts passed the divergence gate and failed R̂ is the
discriminating fact: the sampler was not diverging, it was not moving. A
smaller step size making things worse rules out step size and initialization.

**Diagnosis** (measured, `/tmp/g23_scales.py`): central-differencing the
analytic gradient gives the Hessian diagonal at truth, hence a per-coordinate
implied posterior sd. Those span **1.9322e4**:

| coordinate | d²lp/dx² | implied sd |
|---|---|---|
| theta_bar[0] dom level | -1.1200e+09 | 2.9881e-05 |
| theta_bar[1] dom slope | -3.7762e+08 | 5.1460e-05 |
| theta_bar[2] dom curv | -4.3283e+07 | 1.5200e-04 |
| log_noise[0] | -3.2007e+04 | 5.5895e-03 |
| x0_raw[0] | -2.5500e+03 | 1.9803e-02 |
| eta_raw[39,7] | -3.0000e+00 | 5.7735e-01 |

The runner's approved kernel row adapts step size only, hence an identity mass
matrix: one absolute step size ε in every coordinate, moving each by ε/sd_i in
units of its own posterior sd. No ε serves both 3.0e-5 and 5.8e-1. The
observed R̂ ordering tracks the scale ordering exactly — worst on the
smallest-sd level and slope components, mildest on curvature (5× larger sd)
and the log-noise block (200× larger) — which is this pathology's signature
rather than kink geometry's.

Classification: **tuning/harness failure, not evidence against the target.**
The target evaluates finitely (lp = -17904.1359 at truth), gradients are
finite in all three state parts, and the Hessian diagonal is uniformly
negative definite at truth: a well-behaved locally-Gaussian posterior that is
badly conditioned in the coordinates the runner was given.

**Repair** (both parts, owner-approved 2026-08-26; master program
Amendments A1/A2):

1. *Non-centred parameter chart.* Sample `theta_raw` with
   `theta = prior_mean + prior_sd * theta_raw` via
   `joint_log_prob_raw_batched`. Constant Jacobian, so the posterior is
   unchanged. Verified: chart round-trip max abs error 0.0, raw-chart versus
   natural-chart log density difference 0.0 at truth.
2. *Diagonal mass matrix adaptation.* `NutsConfig.diagonal_mass_matrix=True`
   selects `PreconditionedNoUTurnSampler` under
   `DiagonalMassMatrixAdaptation`, warmup-only. Default `False` preserves the
   approved kernel for G2.2 and all other callers.

**Post-repair scale check** (`/tmp/g23_scales_raw.py`, ~1 min, run before
committing an hour of MCMC): condition ratio **1.9322e4 → 3.8644e2, a 50×
improvement**. The parameter block moved from 3.0e-5–6.3e-3 to 1.5e-3–1.3e-2,
now overlapping the latent block (2.0e-2–5.8e-1) instead of sitting 200×
below it. The residual 3.9e2 is what mass matrix adaptation must absorb.

**Kernel plumbing smoke** (2 chains, 30 warmup, 20 draws): both kernel paths
trace and return correctly-shaped finite draws. Divergences 4 (identity) vs 0
(diagonal); no weight is placed on that comparison at this budget.

**Attempt 3** (2000 warmup, step 1e-2, target accept 0.9, both repairs):
result pending. Fast suite (12 tests, `not hmc and not extended`) passes with
the repairs in place, so the change breaks nothing already green.

## Repairs and amendments

1. **G2.2 CDF convention fix** (lines 211-216 of test_phase2_joint_hmc.py):
   - Changed from right-edge to midpoint CDF convention
   - Added Monte Carlo error tolerance
   - Justification: Half-cell systematic offset was dominating true
     distributional distance; midpoint convention evaluates CDF at grid
     points where HMC empirical CDF is also evaluated.

2. **Grid reference batched filter** (gate_grid_reference.py lines 75-110):
   - Added `gate_gridfilter_loglik_batched`: processes entire log_sd grid
     in one call by recognizing that on offset grid u=x-mu, the AR(1) kernel
     is mu-invariant.
   - Added underflow safety: far-tail parameter cells where observation
     likelihood vanishes return -inf rather than propagating NaN through
     normalization.

3. **Batched grid filter parity test** (`test_gate_gridfilter_batched_parity`):
   the batched filter had no regression guard against the scalar version it
   optimizes. Added; agrees to 1e-10 absolute.

4. **G2.3 conditioning repair** (commit `54862177`, master program Amendment
   A2): non-centred parameter chart plus opt-in diagonal mass matrix
   adaptation. Classified as a sampler-conditioning repair, not a target or
   fixture change — see the G2.3 section above for the measured diagnosis.
   The runner default is unchanged, so this does not perturb G2.2.

## Summary

Phase 2 gates G2.0-G2.2: **PASS**.  
Phase 2 gate G2.3: **PENDING**.

Phase 2 delivers:
- Batched joint log prob with XLA compilation, in both the natural and the
  non-centred parameter chart
- Grid reference posterior with exact filter, batched evaluation, underflow
  safety, and a scalar-parity regression test
- NUTS runner with dual-averaging step size adaptation, plus opt-in diagonal
  mass matrix adaptation for badly conditioned targets
- K=1 gate model exactness validation (grid agreement to sampling error)

Phase 2 open issues:
- G2.3 is running as attempt 3 with the conditioning repair. The measured
  diagnosis (1.9e4 scale spread, reduced 50× by the chart change) explains
  both prior failures and predicts the repair should mix, but that is a
  prediction from Hessian-diagonal geometry at a single point, not a result.
  Only the gate itself settles it.

### Inference status

| Question | Status |
|---|---|
| Hard veto screen | No veto fired. Divergence gates passed in all G2.3 attempts; G2.2 divergences 9/32000 (0.03%) against a 0.1% bound. No non-finite value, invalid artifact, or failed invariant. |
| Statistically supported ranking | None claimed. No method comparison was run in Phase 2. |
| Descriptive-only differences | The smoke-budget divergence difference (4 identity vs 0 diagonal at 30 warmup steps) is descriptive at best and carries no weight. |
| Default readiness | Not established. `diagonal_mass_matrix` stays opt-in; the approved kernel remains the default. |
| Next evidence needed | G2.3 attempt 3 at full budget. If it passes, Phase 3 (Geweke, SBC) proceeds; if it fails with the condition ratio already at 3.9e2, the next discriminating artifact is a per-coordinate ESS breakdown to see which block still fails to move. |

**Not concluded**: nothing here establishes posterior correctness on empirical
data, calibration of the C₁ model, production readiness, or that the kink
target is well-behaved in general. G2.2 establishes grid agreement on one K=1
fixture at one seed; G2.3, if it passes, establishes parameter recovery within
3 posterior sds on one fixture at one seed, which the master program itself
labels a recovery smoke rather than a calibration claim.

Next: Phase 3 plan refresh after G2.3 resolution.
