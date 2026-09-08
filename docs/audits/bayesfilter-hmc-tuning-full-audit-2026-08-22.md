# BayesFilter HMC Tuning: Full Code-Trace Audit and Literature Survey

Date: 2026-08-22
Method: three independent end-to-end code traces (ordinary tuner pipeline,
fixed-transport tuner stack, mass adaptation + diagnostics), each with
file:line evidence, plus a literature/library survey (GIST, ChEES/SNAPER,
nutpie/Walnutpie, nested R-hat, NeuTra follow-ups, flowMC, BlackJAX).
Scope: the tuning subsystem behind the two active routes
`tune_hmc_kernel` and `tune_fixed_transport_hmc_kernel`.
Classification per repo evidence discipline; line anchors verified during
the traces (note `tune_hmc_kernel` def is at hmc_kernel_tuning.py:13420 and
the private signature at :13753 — earlier docs cite stale anchors).

## A. Correctness-critical findings (`wrong relative to the stated target`)

### A1. Final admission does not gate on R-hat; the guard is dead code
- `SequentialRHatHMCVerifier.run` requires R-hat <= 1.01 only for the
  early-pass break (hmc.py:5195-5201). At the draw cap with
  promotion-eligible acceptance evidence but R-hat above threshold, it
  exits `passed=False, cap_hit=True` (5204-5212) — and the attempt
  classifier (`_classify_phase7_acceptance_evidence_verification`,
  hmc_kernel_tuning.py:23591-23781) admits the kernel from acceptance
  evidence + min-retained alone, never reading `passed`,
  `all_finite_rhat_at_or_below_threshold`, or `cap_hit`. The final payload
  then stamps `"fresh_fixed_kernel_verification_passed": True` (25463).
- The intended R-hat-cap retry is unreachable: the trigger strings
  `verification_rhat_above_threshold_or_cap_hit` /
  `verification_rhat_cap_hit` are consumed (17692-17695) but produced
  nowhere; `verification_only_retry_bundle` is always None, deadening
  ~200 lines (11250-11445).
- Three contradictory role labels coexist in one call path:
  `historical_explanatory_only_not_stopping_or_admission` (23294) vs
  `fixed_kernel_convergence_gate_not_candidate_ranking` (23333, hmc.py:5226),
  plus a nonclaim describing the statistic as "unsplit" R-hat (hmc.py:5329)
  when it is split rank-normalized.
- Verdict: wrong relative to the claim "a final kernel is emitted only when
  the Phase 7 fresh fixed-kernel verification passes" (12283-12284).

### A2. No ESS criterion anywhere in the canonical admission path
- A complete, TFP-parity-tested bulk/tail ESS implementation exists
  (hmc_convergence.py:224-358; thresholds 1000/400 declared at 19-24) but
  is never called by canonical verification; the only tuner computing ESS
  (`hmc_robust_broad_grid.py:361`) sets `bulk_ess_min=1.0, tail_ess_min=1.0`
  — thresholds disabled, ESS reduced to a ranking key.
- Combined with the operational verification budget (4 chains x 64 retained
  draws), the R-hat<=1.01 early-pass is an extremely weak screen (split
  half-chains of ~8 draws). Mixing quality of admitted kernels is
  unmeasured. Verdict: `unsupported` for any mixing-related reading of
  "verification passed".

## B. Latent correctness hazards (fail-closed today, one edit from live)

### B3. Unvalidated triangularity assumption in the coordinate transform
`AffineCoordinateTransform.theta_to_latent` uses
`triangular_solve(..., lower=True)` (hmc_coordinates.py:239; also
hmc_warmup.py:4364-4373) but `__post_init__` validates only nonsingularity
(152-157), and `PrecomputedMassArtifact` accepts any F with FF^T=Sigma
(hmc.py:112-121). A symmetric square-root factor would make
`theta_to_latent`/`latent_to_theta` a silently non-inverse pair.
`LatentAffineHMCTransform.position_to_latent` uses full solve (hmc.py:473)
— two transform classes, different inverse semantics for the same field.
Mitigated today by construction + downstream round-trip probes; untested.
One-line fix: assert `np.allclose(np.triu(F,1),0)` or use full solve.

### B4. Artifact signatures depend on eigensolver bit patterns
`signature_payload()` embeds full eigenvalue tuples (hmc.py:287-301)
computed by TF `eigvalsh` on one path (mass_matrix.py:228-243) and numpy
`eigvalsh` on another; `from_payload` discards the persisted summary and
recomputes with numpy (hmc.py:387). Same covariance, different
backend/LAPACK build => different signature => spurious fail-closed
rejection of valid artifacts across environments. Also: no schema/version
label in the hash preimage; int-vs-float type flips silently change every
signature; `json.dumps(allow_nan=True)` would serialize Infinity.

## C. Statistical-evidence weaknesses (heuristic where a criterion is claimed)

- **C5. Verification-time epsilon repair never revalidates L** (ordinary):
  scalar/bracket repairs change epsilon up to x2 with frozen L
  (11175-11249, `skips_phase4_phase5_phase6`), so admitted trajectory
  length tau can be half the geometry target with no re-screen and no
  label of the concession.
- **C6. Sequential looks without multiplicity control**: R-hat and the 90%
  acceptance CI are recomputed on cumulative draws every 64-draw chunk with
  stop-at-first-decision semantics (hmc.py:5042-5202); repeated interval
  tests inflate directional-repair error rates; number of looks recorded
  but not flagged.
- **C7. Adaptation/evaluation coupling**: "fresh" verification chains start
  from a 4-point bank greedily selected from the same run's final warmup
  window (hmc_warmup.py:3920-3935), limiting chain-start overdispersion —
  the premise of split R-hat. Worse in the fixed-transport tuner: default
  initial state is `tf.zeros` for all chains (fixed_transport_hmc_tuning_tf.py:946-956),
  the provided z0 is validated but never seeds any chain, and
  `initial_state_all_zero` is recorded but never vetoed — including under
  the "modern rank-normalized verification" promotion screen.
- **C8. Tiny acceptance budgets at band edges**: fixed-transport screens
  (16 results x 4 chains, 4 burn-in) make in-band membership seed-dependent
  near edges; the bootstrap repairs off a 16-draw single-chain binary rate.
- **C9. Silent degradation off the serious preset**: non-passed bootstrap
  (budget exhausted / cap saturation) still hands its kernel onward for
  smoke/diagnostic/standard presets (12778 gate is serious-only); top-level
  status can read "passed" over a never-in-band bootstrap.
- **C10. Self-certified XLA readiness** (fixed-transport): config flag
  `use_xla` becomes the adapter's readiness stamp (390-397) and mechanics
  jit-compile without consulting capability — sibling routes require
  accepted XLA authority first (candidate_discovery_tf.py:1240-1242).

## D. Labeling and duplication debt

- **D11. Legacy windowed route**: offline replay labeled with adaptive
  vocabulary — `dual_averaging_reset` events with no dual averaging and no
  reset (hmc_tuning.py:1547-1555, 1846-1856); semantic checks verify only
  telemetry presence.
- **D12. "Shrinkage" means three things** under one config field
  (`mass_shrinkage`): shrink-toward-target-mass (legacy), correlation-shrink
  toward diagonal (operational), TFP step-size prior (dual averaging).
- **D13. Forked implementations**: three acceptance-evidence
  implementations with two semantics (mean-probability vs binary in
  neutra_hmc.py:382-390); two rank-normalized R-hat codepaths; three
  acceptance-band classifiers; three L grids ((5,10,15,20,25) tuner /
  (3,5,9,13,18,25) discovery / (2,3,5,8,10,12,16,20,25) grid policy) with
  no cross-reference; repair-factor logic in three places with subtly
  different bracket rules; triplicated `_json_ready`/seed/validation
  helpers; `hmc_kernel_tuning.py` at 27k lines with ~200 lines of dead
  retry machinery and reachable `_..._TEST_BUDGET_SCHEDULE=(3,6,12)`
  production defaults for direct ladder callers (19314-19318).
- **D14. Dead config knobs** (fixed-transport):
  `step_repair_min_directional_factor` and
  `fixed_grid_fallback_acceptance_max` validated and serialized but never
  read — artifacts advertise nonexistent policy.
- **D15. Minor**: late artifact-collision check discards completed tuning
  work (FileExistsError after the ladder, 459-464); per-round
  `initial_step_size` records the post-repair value; per-call
  `tf.function(jit_compile=True)` recompiles per (candidate x round) while
  the ReusableRunner sits unused; NumPy Gaussian-harmonic warm-start pilot
  on the tuning control path (Backend Rule borderline, labeled
  warm_start_only).

## E. What checked out clean (`correct` by trace)

- Fixed-transport transformed-target math: log-det enters the MH-relevant
  value exactly; pullbacks are exact analytic VJPs (proposal-efficiency
  only); frozen-transport guarantee enforced by zero-gradient closure; no
  autodiff tape on the path; batch-native enforced (rank-2 bank, raises on
  rank-1).
- Mass-contract algebra: covariance role preserved at every audited
  boundary; identity-momentum convention enforced structurally; operational
  metric updates genuinely reset dual averaging and re-qualify epsilon;
  composition back to base coordinates probe-verified at 1e-10.
- R-hat implementation is a faithful Vehtari-2021
  max(rank-normalized-split, folded) with scipy/TFP reference tests; ESS is
  honest Geyer initial-positive with declared TFP parity.
- Acceptance evidence has genuine chain-level MCSE (Student-t over 4 chain
  means, df=3) with role separation and fail-closed provenance cross-checks.
- Identity/artifact checks are consistently fail-closed, never fail-open.

## F. Literature and library comparison

1. **Trajectory-length tuning is a generation behind.** BayesFilter screens
   fixed L grids by acceptance bands. The field: ChEES-HMC (Hoffman, Radul,
   Sountsov, AISTATS 2021) and SNAPER-HMC (Sountsov & Hoffman, 2021) adapt
   trajectory length by gradient ascent on ensemble criteria across
   parallel chains (GPU-native — directly relevant to this repo's GPU
   default target); GIST (Bou-Rabee, Carpenter, Marsden; Statistics
   Surveys 2026) gives locally adaptive path-length sampling with exactness
   by construction, unifying NUTS, and has a step-size extension (J. Chem.
   Phys. 2025). BlackJAX ships ChEES adaptation. BayesFilter's
   acceptance-band L screens are heuristic gates with no analogue of either
   family; the geometry anchor tau = pi/(2 median omega) is a reasonable
   warm start but is then frozen (and C5 lets epsilon repairs silently
   halve realized tau).
2. **Mass adaptation ignores score information the repo already computes.**
   Windowed empirical covariance (Stan-style, with correlation shrinkage)
   is the only estimator. nutpie's Fisher-divergence adaptation (geometric
   midpoint of draw covariance and score covariance) converges in far fewer
   warmup draws and dominates covariance-only adaptation across posteriordb;
   low-rank-plus-diagonal structure handles correlated/hierarchical targets
   at K^2 N cost; Walnutpie now runs the same idea online. BayesFilter
   evaluates exact scores at every draw and discards them for mass
   estimation. This is the highest-leverage algorithmic upgrade available.
3. **Diagnostics regime mismatch.** Canonical verification is 4 chains x 64
   retained draws — the many-short-chains regime where classical split
   R-hat is known to be weakly informative. Nested R-hat (Margossian et al.,
   Bayesian Analysis 2025; `posterior::rhat_nested`) was designed exactly
   for this and is absent. Given the repo's GPU default and batch-native
   rules, the natural evolution is many-chain verification with nested
   R-hat, which would also repair A1/A2's weak-screen problem at constant
   wall time.
4. **NeuTra program risk.** Nabergoj & Strumbelj (Machine Learning, 2025;
   10k+ experiments) find train-once NeuTra-HMC frequently worse than
   linearly-preconditioned HMC on second-moment estimates; their follow-up
   (arXiv:2511.02345) attributes it to train-once SVI + overparameterized
   flows and proposes linear + conditional-flow factorization with cyclic
   warmup (re-fit flow between cycles, per-chain dual averaging).
   Implications here: (i) Gate-2 comparisons of enhanced-vs-plain NeuTra
   should include the linear-preconditioner baseline rung the literature
   now treats as the one to beat (the baseline-ladder policy already
   requires this in spirit); (ii) the frozen-transport tuner's design is
   sound for its role, but the program should not assume plain NeuTra is a
   strong baseline; (iii) per-chain step sizes (their finding for avoiding
   stuck chains) contradict the current shared-scalar-epsilon-per-bank
   design — worth a discriminating experiment. flowMC's local-global jump
   architecture is the main non-preconditioning alternative if Gate-2
   enhancements stall.
5. **Provenance.** Constants are enforced and echoed into payloads but
   almost never cited: 0.70/(0.65,0.75) bands (the Beskos et al. 0.651
   optimality neighborhood is the presumable source), Stan-shaped window
   schedules, dual-averaging gamma/t0/kappa silently TFP defaults on every
   path, movement/resonance thresholds bare literals. Only the Student-t
   critical value self-documents. Under the numerical-provenance policy
   most of these are unlabeled inherited/convenience choices.

## G. Contract-gap bridge (feeds the deferred §6a decision)

The fixed-transport trace resolved exactly what the frozen-mass-as-
fixed-transport reformulation needs: an eight-method TF transport-protocol
adapter (`forward/log_abs_det_jacobian/pullback_score/
log_abs_det_jacobian_score` + `_batch` variants + `manifest_payload` + dim)
constructed from `(center, factor)` with constant log-det and zero
log-det-score. No such bridge exists (`LatentAffineHMCTransform` has wrong
method names/signature, no log-det, NumPy backend; the nearest semantics,
`_AffineComponent` in neutra_artifacts.py:428-486, is private and
manifest-bound). One semantic decision required: the transport protocol
adds the log-det to the target value while the affine path omits the
constant — both MH-exact, but cross-route `target_log_prob` traces differ
by a constant. Estimated scope: one small TF class + tests. This makes the
§6a "fixed-transport reformulation" option concrete and cheap to prototype.

## H. Recommended repair order (proposal, not executed)

1. A1: make the classifier consume the verifier's `passed`/R-hat fields, or
   re-wire the dead retry triggers; reconcile the three role labels.
   Smallest discriminating artifact: a test forcing R-hat>1.01 at cap and
   asserting non-admission.
2. A2: enable bulk/tail ESS thresholds in canonical verification (they are
   already implemented and tested); pick thresholds scaled to the
   verification budget or raise the budget.
3. B3/B4: one-line triangularity assert; hash the covariance bytes rather
   than solver-derived eigen summaries (or pin one backend + version the
   preimage).
4. C7: overdispersed verification starts (target-independent dispersal or
   inflated bank covariance); fixed-transport: actually use z0 and require
   distinct starts like the discovery route.
5. C5: re-screen L (or re-anchor tau) after any verified epsilon repair.
6. Literature upgrades as gated experiments, highest leverage first:
   score-informed mass estimation (nutpie-style) inside the existing
   windowed machinery; many-chain verification with nested R-hat; ChEES/
   SNAPER-style L adaptation as a Gate-2-compatible enhancement arm.
7. Dead-code/duplication sweep (D13-D15) after the correctness items.

Nonclaims: this audit establishes code-level findings and literature
context only. No empirical performance claim about any repair is made;
each upgrade in (6) requires its own experiment plan, baseline ladder, and
evidence contract under the repo's gate discipline.
