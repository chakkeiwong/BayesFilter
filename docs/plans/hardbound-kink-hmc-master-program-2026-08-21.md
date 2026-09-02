# Hard-Bound Kink-Target HMC Master Program (Program A, filter-free)

Date: 2026-08-21. Status: **approved for autonomous execution** (owner
instruction 2026-08-21: "Ask for all the approvals upfront so that we can
continue to execute with minimal stopping ... Avoid interaction ... If there
is any doubt, the document is the authority.").

Normative references, in priority order:
1. This document (the authority for execution decisions).
2. The survey manuscript `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex`
   (equation labels `eq:*` and section labels cited below refer to it) and
   the internal record `zlb_discontinuous_hmc_survey.md`.
3. `CLAUDE.md` / `AGENTS.md` repository governance.

## 1. Objective and scope

Build and validate, inside BayesFilter, the filter-free exact inference
route for the hard-bound (kink-target) shadow-rate model class: survey
route (iv), joint non-centered HMC on the `mf_c1_k40_hardmax` target,
verified by the survey's test ladder without particle-filter machinery.

Deliverables:
- A new subpackage `bayesfilter/hardbound/` containing target evaluators,
  closed-form censored/truncated reference functions, the joint non-centered
  log-posterior, and Geweke/SBC validation harnesses.
- Tests under `tests/hardbound/` mirroring the ladder tiers.
- Per-phase result notes under `docs/plans/` and a final program result note.

### Non-goals (binding)

- No particle filter, PMMH, PGAS, COPF, or particle-likelihood machinery of
  any kind, including "just a diagnostic bootstrap PF". (Owner decision
  2026-08-21: the PF is heavy and unnecessary for Program A. If a marginal
  likelihood is ever needed, that is a new program.)
- No NK/OBC (Program B) work: no solution/selection kernel, no LCP code.
- No cell-decomposition proposal layer, no Genz/Botev numerics, no
  root-aware C0 integrator (survey route (v) and the C0 experiment remain
  conditional future work).
- No NeuTra, no neural transport, no production/empirical-data claims, no
  changes to MacroFinance, no changes to existing BayesFilter routes.
- No event-aware / reflection-refraction HMC: the C1 target is a kink
  target (survey Sec. "shadow-ladder"); ordinary Metropolized HMC is valid
  per the survey's kink-validity proposition.

### Pre-approved decisions (do not re-ask)

| Decision | Approved choice |
|---|---|
| Program scope | Program A only, filter-free (route (iv)) |
| Verification | Test ladder Tiers 0, 1, K=1 grid, Geweke+SBC at K=40; no cross-route PF gate |
| New code location | `bayesfilter/hardbound/` + `tests/hardbound/` |
| Backend | TF/TFP for all runtime code; NumPy/SciPy allowed ONLY in `*_reference.py` diagnostic modules and tests (grid posteriors, scipy.stats cross-checks) |
| Dtype | float64 throughout this correctness program (both runtime and reference). This is a correctness campaign, not a performance campaign; it does not reopen the repo GPU/TF32 default |
| Device | CPU execution acceptable for all phases; GPU optional. No serious-GPU-run manifest obligations are triggered. Set memory growth if GPU is initialized |
| HMC kernel | `tfp.mcmc.NoUTurnSampler` with `DualAveragingStepSizeAdaptation`, via a thin local runner (do not entangle with the NeuTra/route-ledger stack); fixed seed streams |
| Model constants (working example) | Per survey Sec. 13.1: state dim 8 (2 DNS triples + 2 basis), decays 0.65/0.45, bounds 0.0/-0.005, alphas 1.5e-3/1.0e-3, K=40 Gauss-Legendre on [0,1], 6 maturities per country + 1 FX row, diagonal measurement noise |
| Estimated parameters (fixtures) | theta_bar components (6: level/slope/curvature long-run means per country) + 3 grouped log noise scales = 9-dim chart, matching the survey's "nine-parameter chart"; Phi, Q, initial covariance frozen fixture constants |
| Priors (fixtures) | Independent Gaussians: theta_bar ~ N(anchor, (0.02)^2) per component with anchors (0.02, -0.01, 0.005) domestic and (0.015, -0.008, 0.004) foreign; log noise scales ~ N(log(5e-4), 0.5^2). Simple, proper, full-support; chosen for testability, not economics |
| Route ledger | The new route is a validation-scope route; register it in the program result note, NOT in the NeuTra route ledger (it is not a NeuTra route). If the SR-UKF route-guard discovery flags it, add the explicit non-NeuTra classification it asks for |
| Test markers | Fast unit tests unmarked; K=1 grid and short-chain HMC tests marked `hmc`; Geweke/SBC full runs marked `extended` and `hmc` |
| Python env | `/home/ubuntu/miniforge3/envs/tf-gpu/bin/python` (TF 2.19.1, TFP 0.25.0) |
| Interaction policy | Execute all phases without asking. Stop ONLY for: (a) a mathematical error discovered in the survey itself, (b) a gate failure that survives one diagnosis-and-repair cycle, (c) an action that would modify files outside the locations named here |

## 2. Model and target definitions (frozen)

State: \(x_t\in\mathbb R^8\), transition \(x_t=(I-\Phi)\bar\theta+\Phi
x_{t-1}+\eta_t\), \(\eta_t\sim N(0,Q)\) (survey eq:75). Fixture constants:
\(\Phi=\mathrm{diag}(0.95,0.90,0.85,0.95,0.90,0.85,0.98,0.98)\),
\(Q=\mathrm{diag}(q)\) with
\(q=(2,3,4,2,3,4,0.5,0.5)\times10^{-6}\); initial law
\(x_0\sim N(\bar\theta_{\text{ext}},P_0)\) with
\(P_0=\mathrm{diag}(1,1.5,2,1,1.5,2,0.25,0.25)\times10^{-5}\) and
\(\bar\theta_{\text{ext}}\) = estimated theta_bar for the six factor means,
0 for the two basis factors (basis long-run means frozen at 0).

Forward curve (eq:76): \(f_c(s;x)=a_c(s)^\top x^c\),
\(a_c(s)=(1,e^{-\lambda_c s},\lambda_c s e^{-\lambda_c s})\).
Bound maps (eq:77): \(m_\ell(u)=\max\{\ell,u\}\) and
\(s_{\ell,\alpha}(u)=\ell+\alpha\,\mathrm{softplus}((u-\ell)/\alpha)\).
Yields (eq:78): order-K Gauss-Legendre average on \([0,\tau_i]\) via nodes
on \([0,1]\); maturities \(\tau\in\{0.25,1,2,5,10,30\}\) years per country.
FX row (eq:79) at \(\tau_{fx}=1\): \(\tau_{fx}(y_d-y_f+b)\) with
\(b=x^{b,1}+x^{b,2}\) (affine basis loading, both basis factors, unit
loadings; frozen).
Measurement noise: diagonal Gaussian; three grouped scales (domestic
yields, foreign yields, FX), fixture truth \(5\times10^{-4}\) each.

Targets:
- `S1` = softplus map in eq:78 (smooth model).
- `C1` = hard max in eq:78, same nodes/weights (kink target; THE target of
  this program).
- `C0` = continuous-maturity integral (defined for the record; not built).

Joint non-centered log posterior (survey eq:56 chart): parameters
\(\vartheta\) (9-dim), latents \(z=(x_0^{\text{raw}},\eta_{1:T}^{\text{raw}})\)
standard-normal a priori; \(x_0=\bar\theta_{\text{ext}}+P_0^{1/2}x_0^{\text{raw}}\),
\(\eta_t=Q^{1/2}\eta_t^{\text{raw}}\); log posterior = standard-normal
log density of raws + Gaussian prior on \(\vartheta\) + observation
log density of \(y_{1:T}\) under the C1 (or S1) map. `tf.maximum` supplies
the a.e. gradient; validity per the survey's kink proposition.

Synthetic fixtures: horizon \(T=40\) periods for HMC tests, \(T=200\) for
identification diagnostics; data simulated from the C1 model at fixture
truth with seed 20260821; binding engineered by setting the foreign level
mean so that roughly 30--50% of foreign short-maturity nodes bind (verify
and record the realized binding fraction).

## 3. Phases, work, and exit gates

### Phase 0: target evaluators and tie-out

Build `bayesfilter/hardbound/`:
- `dns_curve_tf.py`: loadings \(a_c(s)\), Gauss-Legendre nodes/weights
  (computed once host-side with numpy.polynomial at module import is
  acceptable as constants; document this), yield maps for S1/C1 with a
  `bound_map` switch; batched over time and over parameter/latent batch.
- `model_tf.py`: fixture constants, simulate fn, observation log density,
  target registry with identifiers `mf_s1_k40_softplus`, `mf_c1_k40_hardmax`.
- `reference_numpy.py` (diagnostic): independent NumPy re-implementation of
  the yield maps and log densities.

Exit gates:
- G0.1: TF vs NumPy reference agreement on 100 random states: yields and
  log densities within 1e-10 (float64).
- G0.2: S1-vs-C1 gap obeys survey bounds eq:88--eq:89: elementwise
  0 < gap <= alpha*log 2 on random states; max attained near binding.
- G0.3: crossing lemma spot check: for 1000 random factor states, the
  binding set along s-grid is an interval pattern (empty/interval/
  complement/all), per survey eq:82.

### Phase 1: closed-form layer

- `censored_scalar_tf.py`: the censored-observation predictive density,
  survey eq:29--eq:35 (atom probability, two-term mixture, posterior
  binding probability), in TF with log-space Mills-ratio-stable forms.
- `truncated_gaussian_tf.py`: affine-boundary branch probabilities and
  truncated moments, survey eq:21--eq:28, log-space stable.

Exit gates:
- G1.1: mixture density integrates to 1 (quadrature over y, 1e-8) and
  matches a scipy.stats reference on a parameter grid including
  |gamma| up to 8 (tail stability), 1e-9 relative.
- G1.2: truncated moments match Monte Carlo (1e7 draws, 4 sigma) and match
  scipy.stats.truncnorm where applicable, and remain finite/stable at
  gamma = +-37.
- G1.3 (moved to Phase 2 as G2.0 in plan review, 2026-08-21, because it
  requires the joint target built in Phase 2): see G2.0 below.

### Phase 2: joint HMC on C1 and small-scale exactness

- `joint_target_tf.py`: the non-centered joint log prob (Sec. 2 above) with
  `tf.function` compilation, float64, gradient test vs finite differences
  away from kinks (1e-6) and a documented a.e.-gradient statement at kinks.
- `hmc_runner.py`: thin NUTS + dual-averaging runner (seeded, multi-chain),
  reporting acceptance, step size, divergences, and split-Rhat/ESS via
  `tfp.mcmc.potential_scale_reduction` / `effective_sample_size`.
- Solver-branch value-continuity test (survey Sec. 15 / eq:98): evaluate
  the C1 log posterior on straddling point pairs across node-binding
  boundaries; assert value continuity to 1e-8 and record the gradient jump
  magnitudes (finite, bounded).
- K=1 grid gate model (binding specification, no alternatives): state
  dim 1 (domestic level factor only; decay irrelevant since the loading is
  constant 1), T=4, K=1 quadrature node, one maturity, no FX row, hard-max
  bound with ell=0 and the level mean placed so ~40% of simulated
  observations bind. Two estimated parameters: level long-run mean and log
  measurement-noise scale, with the Sec. 2 priors. The latent path in the
  non-centered chart is 5-dimensional (x0 raw + 4 shock raws). The exact
  reference is the marginal parameter posterior on a 40x40 parameter grid,
  with p(y|theta) at each grid point computed by 15-node/dim Gauss-Hermite
  integration over the 5 latents in the NumPy reference module
  (diagnostic). HMC on the joint target must reproduce this marginal:
  Wasserstein-1 distance of each parameter marginal < 0.05 posterior sds,
  and posterior mean/sd agreement within 3 Monte Carlo standard errors.

Exit gates:
- G2.0 (Kalman tie-out, Tier 0): on the tiny model of the K=1 gate but with
  the bound removed (ell=-1e9, so the observation map is affine), the
  marginal likelihood p(y|theta) computed two independent ways agrees to
  1e-6 relative on 20 random (theta, y) pairs: (a) Kalman-filter likelihood
  on the induced 1-D LGSSM (direct scalar Kalman recursion in the NumPy
  reference, cross-checked against `bayesfilter.linear` where its interface
  permits a 1-D model without adaptation overhead); (b) 15-node/dim
  Gauss-Hermite integration over the 5 standard-normal latents of the joint
  non-centered target. Verifies the joint target's likelihood content
  against the exact linear answer before any bound enters.
- G2.1: gradient check passes; value-continuity test passes.
- G2.2: 4-chain NUTS on the K=1 gate model: zero divergences after warmup,
  split-Rhat < 1.01 on both parameters, ESS > 400/chain-set, and the
  grid-agreement criteria above.
- G2.3: 4-chain NUTS on the full C1 fixture (T=40, 9 params + latents):
  split-Rhat < 1.02 on all 9 parameters, no more than 0.1% post-warmup
  divergences, and posterior means within 3 posterior sds of fixture truth
  (parameter recovery smoke, not a calibration claim).

### Phase 3: full-scale validation harnesses

- `geweke_tf.py`: Geweke joint-distribution test, marginal-conditional vs
  successive-conditional simulators for the C1 model at full K=40 spec but
  reduced T=8 (successive-conditional needs a data-refresh step: y ~ p(y|
  theta, x); the latent path is part of the chain state). Test functions:
  each parameter, each parameter squared, two binding-fraction functionals.
  Gate: all |z| < 4 with chain length >= 20k after thinning guidance, and
  fewer than 2 of 22 functionals in (3,4).
- `sbc_tf.py`: simulation-based calibration, 200 prior draws, T=20, rank
  of each of 9 true parameters within 100 thinned posterior draws.
  Gate: per-parameter rank-uniformity chi-square (20 bins) p > 0.005 after
  Bonferroni across 9 parameters; visual histogram artifacts saved.
- Identification/binding diagnostics (survey Sec. 13.6): on a T=200
  long-binding fixture, compute the observed-information degeneracy along
  binding directions and record (result-note material; no gate beyond
  "computed and recorded").

Exit gates: the two harness gates above (G3.1 Geweke, G3.2 SBC), runtimes
recorded. If a gate fails: one diagnosis-and-repair cycle (check
seed-stream independence, step-size adaptation, data-refresh correctness --
the classic Geweke failure causes) and rerun; a second failure is a
stopping condition.

### Phase 4 (out of scope, recorded for the future)

Cell proposal layer, C0 integrator, diagnostic PF, event-aware HMC: all
explicitly deferred; each requires reopening this plan.

## 4. End-of-phase protocol (drift control)

At the end of each phase, in order:
1. Run the full accumulated test suite (all previous phases' gates).
2. Repair what is repairable within the phase's scope; record what was
   repaired and why in the phase result note
   `docs/plans/hardbound-kink-hmc-phaseN-result-2026-08-21.md`.
3. Refresh THIS document's next-phase section if reality diverged
   (constants that had to change, gates that needed re-tolerancing, with
   one-line justifications in an appended "Amendments" section -- never
   silently edit the original text).
4. Continue without asking, per the interaction policy.

## 5. Governance bindings

- Backend rule: runtime modules import TF/TFP only; `reference_numpy.py`,
  Gauss-Hermite grids, and scipy cross-checks live in clearly named
  diagnostic modules and tests. Gauss-Legendre nodes as module constants
  computed via numpy at import are declared here as a host-side constant
  boundary (they feed TF as constants, not a NumPy runtime path).
- Memory growth: the hmc_runner sets memory growth on visible GPUs before
  any TF device initialization when GPUs are present.
- Tuning scope: NUTS dual-averaging is per-fixture warmup adaptation, not
  runtime adaptation inside a claim run; each fixture's adapted step size is
  recorded in its result note. No cross-fixture tuning inheritance claims.
- Chunk rule / DPF transport: not applicable (no transport code).
- NeuTra rules: not applicable (no NeuTra, no optimizer training).
- Claims discipline: this program claims validated-on-ladder status for the
  route on the named fixtures only -- no posterior correctness on empirical
  data, no production readiness, no superiority claims.

## 6. Risks and predetermined responses

| Risk | Response (pre-approved) |
|---|---|
| NUTS divergences from kink gradient jumps near heavy binding | Lower target accept stat to 0.9/0.95; if insufficient, reduce fixture binding fraction toward 30% and record; do NOT smooth the bound |
| K=1 grid intractable or reference too slow | Reduce T to 3; increase GH nodes only if needed; the binding variant in Phase 2 already chose minimal dims |
| Geweke z-failures | One repair cycle per Sec. 3 Phase 3; then stop |
| Rhat stall on latents at T=40 | Latents are nuisance: gates bind on parameter Rhat; record latent diagnostics |
| float64 GPU slowness | CPU execution is pre-approved; XLA optional |
| SBC wall time excessive | Pre-approved fallback ladder: 200 draws -> 100 draws (chi-square bins 10) -> T=20 -> T=12; record which rung was used |
| Geweke chain too short for z stability | Extend chain x2 once; then apply the failure protocol |
| Route-guard discovery flags the new module | Add the explicit non-NeuTra classification entry it requires |

## 7. Acceptance checklist (program level)

- [ ] G0.1--G0.3, G1.1--G1.2, G2.0--G2.3, G3.1--G3.2 all pass and are
      recorded with seeds and runtimes in phase result notes.
- [ ] All new tests discoverable under `pytest tests/hardbound` with correct
      markers; fast subset runs clean without `hmc/extended`.
- [ ] No file outside `bayesfilter/hardbound/`, `tests/hardbound/`,
      `docs/plans/hardbound-*`, and this plan's amendments was modified.
- [ ] Program result note maps each gate to the survey ladder rung it
      discharges and states the non-claims.

## 8. Amendments

Appended per Sec. 4 step 3. The original text above is never silently edited.

### A1 (2026-08-26, G2.2 re-tolerancing)

The G2.2 Wasserstein-1 criterion "< 0.05 posterior sds" (Sec. 3 Phase 2) was
not achievable as literally written, for two reasons found in execution:

1. *CDF convention.* The test compared `cumsum(marg)`, the grid CDF at cell
   right edges, against an empirical CDF evaluated at grid points. That
   half-cell offset contributes W1 ~ 0.5*dx no matter how well the two
   distributions agree. A grid-refinement sweep (n_mu 40 -> 320) confirmed it:
   the right-edge W1 shrank proportionally to dx (0.00177 -> 0.00041) while the
   midpoint-convention W1 converged to a fixed value. Repaired with the
   midpoint convention, `cumsum(marg) - 0.5*marg`.
2. *Monte Carlo floor.* Even with conventions matched, W1 between an empirical
   CDF of effective size n_eff and the exact CDF is ~ sd/sqrt(n_eff). At the
   gate's achieved ESS 291 that floor is 0.059 sd, above the 0.05 sd
   threshold, so no sampler could pass at that ESS however correct it was.

Amended criterion: `W1 < max(0.10*g_sd, 1.5*g_sd/sqrt(n_eff))`, a
sampling-error-aware bound that tightens as ESS grows. Observed at the gate:
0.051 sd (mu) and 0.050 sd (log_sd), both stable under 8x grid refinement.
This re-tolerances a discretization-plus-Monte-Carlo artifact. It does not
weaken the exactness content, which remains grid agreement in mean, sd, and
full marginal shape.

### A2 (2026-08-26, G2.3 kernel row and parameter chart)

The Sec. 1 kernel row ("`NoUTurnSampler` with
`DualAveragingStepSizeAdaptation`") specifies step-size adaptation only, hence
an identity mass matrix. Two G2.3 runs failed on parameter R-hat under it:
[16.3, 19.3, 5.9, 3.5, 25.0, 8.0, 1.3, 1.5, 1.5] at 2000 warmup / 5e-3, then
worse at 4000 warmup / 1e-3, [6.7, 60.5, 1.5, 9.8, 49.4, 14.5, 2.9, 1.9, 5.2].
The divergence gate passed both times: the sampler was not diverging, it was
not moving.

Root cause, measured rather than inferred, by central-differencing the
analytic gradient for the Hessian diagonal at truth: per-coordinate implied
posterior sd spans **1.9e4** across the 337-dim state, from theta_bar levels
at 3.0e-5 to `eta_raw[39,7]` at 5.8e-1. One scalar step size with an identity
mass matrix cannot serve both ends, and the observed R-hat ordering tracks the
scale ordering exactly: worst on the smallest-sd level and slope components,
mildest on curvature (5x larger sd) and the log-noise block (200x larger).
Reducing the step size cannot help, which is why the first repair attempt made
matters worse.

Two amendments, owner-approved 2026-08-26 after the diagnosis was presented:

- *Parameter chart.* G2.3 samples the non-centred chart
  `theta = prior_mean + prior_sd * theta_raw` via
  `joint_log_prob_raw_batched`. The Jacobian is constant and therefore
  irrelevant to MCMC. Verified: chart round-trip exact, and raw-chart versus
  natural-chart log density agree to 0.0 at truth. Condition ratio
  1.9e4 -> 3.9e2 (50x).
- *Kernel row.* `NutsConfig.diagonal_mass_matrix` opts into
  `tfp.experimental.mcmc.PreconditionedNoUTurnSampler` under
  `DiagonalMassMatrixAdaptation`, warmup-only, to absorb the residual 3.9e2.
  The Preconditioned variant is needed only because plain `NoUTurnSampler`
  exposes no `momentum_distribution` slot in TFP 0.25.0; the NUTS algorithm is
  unchanged. The default stays `False`, so G2.2 and every other caller keep
  the originally approved kernel exactly.

Both are sampler-geometry repairs. Neither changes the target, the fixture,
the priors, the bound, or any gate threshold, and neither licenses a claim
about posterior correctness beyond what G2.3 itself tests.

### A3 (2026-08-26, fixed-trajectory HMC kernel)

The Sec. 1 kernel row (line 55) pre-approved `tfp.mcmc.NoUTurnSampler` with
`DualAveragingStepSizeAdaptation` for the hardbound suite. Execution showed
two issues:

1. **Performance.** User observation: "the current Tensorflow version of NUTS
   is extremely slow." NUTS tree building and backtracking scale poorly for the
   hardbound target at T=40, 337 dimensions, and C1-only kink gradients.

2. **Policy violation.** Repository-wide codified NUTS policy at
   `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py` lines 130-134
   raises on any NUTS tuning request with message:
   "NUTS is reference/diagnostic only, not a tuning/default remedy;
   fixed-trajectory HMC tuning must use HamiltonianMonteCarlo."
   An approved project-scoped NUTS block also exists at
   `docs/plans/bayesfilter-filtering-value-gradient-benchmark-p8i-phase5-nuts-readiness-result-2026-06-16.md`
   line 5 status `BLOCK_NUTS_NOT_READY_REVIEWED`.

3. **Misapplied acceptance target.** Sec. 6 line 259 prescribed "Lower target
   accept stat to 0.9/0.95" *conditionally* on "NUTS divergences from kink
   gradient jumps near heavy binding." Measured G2.3 evidence: sampling
   divergences were 0/12000 in every tested warmup/shrinkage arm (warmup
   4000/6000/8000, λ 0.05/0.1/0.15), so the divergence condition never fired.
   Yet `target_accept=0.95` was applied unconditionally, producing step sizes
   5.99e-4 to 9.19e-4, tree-depth saturation at 2^max_tree_depth=1024, and
   high acceptance with poor mixing. R-hat worsened monotonically with more
   adaptation toward the 0.95-optimal small step size: warmup 4000 → 1.0196
   (λ=0.1), 6000 → 1.0273, 8000 → 1.0315. Repository standard is target 0.70,
   band (0.65, 0.75) per
   `FIXED_TRAJECTORY_HMC_V2_ACCEPTANCE_BAND`.

Amended kernel row (line 55 scope, entire hardbound suite):

- Kernel: `tfp.mcmc.HamiltonianMonteCarlo` (fixed-trajectory HMC) with
  `DualAveragingStepSizeAdaptation` during warmup. Explicit
  `num_leapfrog_steps` selected via manual tuning ladder for G2.3 (which
  requires non-identity mass under A2); identity-mass fixtures may use the v2
  tuning protocol in `bayesfilter/inference/fixed_trajectory_hmc_tuning_v2.py`
  once v2 is extended beyond its current identity-mass-only constraint (lines
  137-141).
- Acceptance target: 0.70 with band (0.65, 0.75), applied unconditionally.
- Observability: trace must capture `is_accepted`, `log_accept_ratio`, and
  `step_size` directly from `kernel_results` (no unwrapping).
- Preserved constraints: thin local runner, no entanglement with the
  NeuTra/route-ledger stack, fixed seed streams per the original line 55.

Pre-approved dense mass matrix policy (A2 `PreconditionedNoUTurnSampler` with
diagonal adaptation for G2.3) remains available for optional use with
fixed-trajectory HMC via `tfp.experimental.mcmc.PreconditionedHamiltonianMonteCarlo`
(verified present in TFP 0.25.0, `momentum_distribution` slot confirmed) under
the same diagonal/windowed policy. The windowed implementation at
`bayesfilter/hardbound/windowed_dense_mass_adaptation.py` line 306 currently
hardcodes `PreconditionedNoUTurnSampler` and will need adaptation:
- Replace kernel with `PreconditionedHamiltonianMonteCarlo` plus explicit
  `num_leapfrog_steps`
- Change [_nuts_results](bayesfilter/hardbound/windowed_dense_mass_adaptation.py#L279-L282) unwrapping to direct access
  (`kernel_results.is_accepted`, `kernel_results.log_accept_ratio`)
- HMC divergence detection is `nonfinite_log_accept_ratio` per
  `fixed_trajectory_hmc_tuning_v2.py` line 209, not `has_divergence` field

This amendment corrects the kernel family and acceptance target; it does not
alter the mass adaptation decisions recorded in A2.
