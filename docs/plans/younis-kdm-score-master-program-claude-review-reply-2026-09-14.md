# Claude Review Reply: Younis KDM Score Master Program

**Reviewer:** Claude Opus 5 (1M context)  
**Review date:** 2026-09-14  
**Plan under review:** `younis-kdm-score-master-program-2026-09-14.md`  
**Handoff document:** `younis-kdm-score-master-program-claude-review-handoff-2026-09-14.md`  
**Repository commit at review:** `5cc59cfa`  
**Repository branch:** `surrogate-hmc`

---

## Executive summary

The master program is **approved for execution** subject to three mandatory repairs and one material scope clarification. The program defines a valid scientific question, specifies exact mathematical targets with source grounding, and respects the existing governance and evidence-contract framework. The diagnostic KDM routes, oracle ladder, and LGSSM test harness are implemented and tested. The program correctly treats Phase 6 (degenerate transitions) as unresolved and outside current scope.

**Three mandatory repairs before Phase 1 execution:**

1. **Add explicit initial-law tangent verification** to Phase 1 gates (lines 421–440). The current Phase 1 scope mentions "parameter-coordinate checks" but does not explicitly require initial-law derivative parity with the oracle. The Kalman oracle (`kalman_oracle_value_and_score`) computes the complete score including `initial_mean` and `initial_covariance` tangents; the Phase 1 gate must verify that the LEDH analytical score matches those components.

2. **Clarify OT reset role** in the target definition (lines 191–265) and evidence contract (lines 266–298). The OT reset (`ledh_contract_e_reset_tf.py`) is a deterministic, measure-preserving particle transformation—it changes the particle representation but does not change the filtering distribution, introduce new randomness, or create a parameter-dependent resampling law. It is not a "reset to a new proposal" in the Younis sense. The program must state explicitly whether the OT step's Jacobian determinant (currently accumulated as part of the forward pass) is treated as part of the normalizer `Z_hat` or as a separate geometric correction, and whether the target `ATOM-FINITE` includes or excludes that determinant term.

3. **State the UKF covariance update observability requirement** in Phase 2 (lines 442–520) and the governance checklist (lines 735–753). The UKF per-particle covariances are internal state; the current score recursion uses them but does not emit them as diagnostic artifacts. If Phase 2 or later phases need to inspect, validate, or compare UKF covariances (e.g., for KDM bandwidth calibration, representation study, or failure-mode diagnosis), the implementation must expose them. If they remain internal-only, state that explicitly and justify why covariance inspection is unnecessary for the planned comparisons.

**One material scope clarification (not a blocker, but must be recorded):**

4. **Compute budget is open-ended.** The program states "Budget and time are not a problem" (line 301) and makes no particle-count, model-horizon, replica-count, or wall-clock limit visible. This is the owner's prerogative, but it must be acknowledged: the program authorizes a potentially expensive nine-phase factorial study with no declared GPU-hour cap, no per-phase attempt limit, and no cost-based stop rule. If a budget ceiling exists, record it now; if not, state that the campaign is uncapped and that cost overruns are an accepted risk.

---

## Stage 1 audit: Handoff document and cited paths

All 35 cited paths exist and are readable. The handoff correctly distinguishes the research snapshot (`804616e3`) from the current main-branch implementation (`5cc59cfa`). The Stage 2 mathematical sources (6 local papers, 1 arXiv preprint) are present in `.localresources/`. The implementation modules and test files match the claimed line ranges.

The handoff correctly identifies that the master program does not cover disturbance-coordinate filtering (Phase 6 degenerate transitions). That work belongs to the separate degenerate-model-score-reassessment artifact and is not a dependency for Phases 1–5 or 7–9.

---

## Stage 2 audit: Mathematical definitions and source grounding

### Target definitions (lines 191–265)

The program defines four exact targets with clear mathematical content:

1. **EXACT-LGSSM-SCORE** (lines 195–215): The Kalman marginal log-likelihood and its gradient with respect to θ. Grounded in Kalman recursion (no contested identity). Implemented in `ledh_kalman_oracle_tf.py` as `kalman_oracle_value_and_score`.

2. **ATOM-FINITE** (lines 217–227): The log of the normalized particle weight product, as a finite-N realization of the unknown continuous expectation. This is the existing canonical LEDH value program. The definition is clear as stated, but see **Mandatory Repair 2** above: the program must clarify whether the OT reset determinant is part of this target or treated as a separate correction.

3. **KDM-FINITE** (lines 229–253): The integrated-observation finite-N estimator, replacing the point-observation factor `p(y_t | x_t^i)` with its linear-Gaussian integral under a declared Gaussian kernel centered at `x_t^i`. This is the Younis (2021) construction (Definition 3.1, equation 3.2) extended to the LEDH setting. The definition correctly notes that this is a *different* finite-N program from ATOM-FINITE, not merely a variance-reduction technique for the same target.

4. **Unnormalized and normalized scores** (lines 255–265): The program distinguishes `D_hat / Z_hat` (the ratio of unnormalized tangent to unnormalized value) from `d/dθ log Z_hat` (the score of the normalized value). This distinction is load-bearing and correctly reflects the bias introduced by normalizing a ratio estimator. The degenerate-reassessment artifact (`degenerate-model-score-reassessment.md`, decision table) confirms this is not a tuning detail—it is a structural estimand question.

**Verdict:** The four targets are mathematically well-defined. The EXACT target is uncontroversial. The ATOM-FINITE and KDM-FINITE targets are explicitly labeled as distinct finite-N programs. The unnormalized/normalized distinction is correct and will be testable in Phase 3.

**One clarification required (Mandatory Repair 2):** The OT reset role must be stated explicitly. The current program says "canonical LEDH reset" (line 219) without defining whether the reset's Jacobian determinant is part of the value `Z_hat` or a separate term. The implementation (`ledh_canonical_score_tf.py`, `ledh_contract_e_reset_tf.py`) accumulates the determinant as part of the forward pass, but the target definition should say so explicitly to avoid later confusion about what "the ATOM-FINITE value" means.

### Source grounding (handoff lines 91–157)

The handoff cites six primary papers:

- **Younis (2021, PhD thesis)**: Sections 3.1–3.4 define the Gaussian-mixture marginalized observation factor (Definition 3.1, equation 3.2), the backward information filter (Section 3.3), and the IWSG resampling gradient (equation 3.10). The thesis does not prove that the marginalized gradient is unbiased for the unmixed model score—it is a proposal-informed estimator that changes the finite-N program. The master program correctly treats KDM-FINITE as a distinct target.

- **Li et al. (2017, Algorithm 1)**: The UKF-PF-PF forward filter with per-particle covariances. The LEDH implementation (`ledh_ukf_lifecycle_tf.py`) is semantically grounded in the reviewed `experiments/dpf_implementation/.../ledh_pfpf_alg1_ukf_tf.py` (June 2026 campaign). The master program correctly identifies this as the baseline particle law.

- **Murray et al. (2013, disturbance-state-space particle filtering)**: Cited in the handoff for Phase 6 degenerate models. The master program correctly excludes disturbance coordinates from the current scope (Phase 6, lines 630–651, is marked unresolved).

- **Scibior & Wood (2021, resampling gradient correction)**: The pathwise derivative of a parameter-dependent resampling law. The degenerate-reassessment artifact confirms that the correction is required for parameter-dependent proposal laws and that a pathwise-only derivative is insufficient. The master program does not yet specify where this correction enters (Phase 3? Phase 4B? Phase 5?), but the principle is acknowledged.

- **Poyiadjis, Doucet, Singh (2011, smoothing and the Fisher identity)**: The backward information recursion for the model score. The master program correctly notes that this identity assumes non-degenerate transitions (Phase 6 blocker) and that smoothing-based estimators are a separate route class.

- **Corenflos et al. (2021, differentiable particle filtering)**: General framework. The master program does not claim to implement the full Corenflos taxonomy; it is cited as background.

**Verdict:** The source grounding is adequate for Phases 1–5 and 7–9. The unresolved Phase 6 question (degenerate transitions) is correctly flagged and excluded from the current scope.

---

## Stage 3 audit: Implementation and call-chain verification

### Implemented routes and tests

The diagnostic KDM modules exist and are tested:

- `bayesfilter/highdim/ledh_younis_kdm_tf.py` (ROUTE_ID: `ledh_younis_kdm_algebra_diagnostic_v1`): Implements the Gaussian KDM kernel with complete tangent propagation. Tests in `test_ledh_younis_kdm_tf.py` verify:
  - Complete tangent matches finite difference (line 122)
  - Conditional KDM uses the full ancestor mixture (line 250)
  - IWSG anchor/tangent match fixed-proposal difference (line 292)
  - Anchored PF-PF KDM correction freezes the proposal denominator (line 355)
  - Linear-Gaussian KDM normalizer complete tangent (line 474)
  - Subspace KDM handles rank-deficient support (line 665)
  - KDM atom sidecar reads the canonical endpoint trace (line 736)

- `bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py` (ROUTE_ID: `ledh_younis_kdm_integrated_observation_weighting_v1`): Phase 4A integrated-observation score. Feeds the exact linear-Gaussian integral into the canonical Contract-E/GenUT/dual-cap executor. This is the KDM-FINITE target implementation.

- `bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py` (ROUTE_ID: `ledh_contract_e_then_younis_iwsg_resampling_reference_v2`): Phase 4B full-mixture IWSG resampler after the canonical LEDH reset. This defines the separate RESKDM-IWSG-FINITE target, not another derivative implementation for ATOM-FINITE.

- `bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py`: Independent LGSSM references (exact Kalman, bootstrap fixed-stream). These are the Phase 0.2 oracle ladder. Tests in `test_ledh_younis_kdm_lgssm_reference_tf.py` verify scalar and matrix LGSSM references match finite-difference (lines 20, 78) and that the bootstrap score matches finite-difference (line 142).

- `bayesfilter/highdim/ledh_kalman_oracle_tf.py`: The EXACT-LGSSM-SCORE oracle. Tests in `test_ledh_kalman_oracle_tf.py` verify scalar LGSSM (line 16), score vs finite-difference (line 57), and deterministic replay (line 124).

**Call-chain audit (Implementation Audit Call-Chain Rule, AGENTS.md):**

The handoff claims "registry call-chain repair is complete" (handoff line 138). I verified:

- `ledh_canonical_score_tf.py` defines `canonical_value_and_analytical_score` (line 682), which wraps `_value_and_analytical_score_impl` (line 67). The wrapper "deliberately offers no observation-factor substitution" (line 717) and invokes the shared executor with the model's actual observation density.

- The integrated-observation route (`ledh_younis_kdm_integrated_tf.py`) calls `_value_and_analytical_score_impl` with an `observation_factor_override` that supplies the KDM-integrated weights. This is the correct call chain for a diagnostic route that changes the observation factor while reusing the Contract-E/GenUT/dual-cap executor.

- The resampling route (`ledh_younis_kdm_resampling_tf.py`) calls `_value_and_analytical_score_impl` with a `post_reset_transform` that applies the full-mixture IWSG resampler. This is the correct call chain for a diagnostic route that changes the resampling step while reusing the upstream flow and observation steps.

**Verdict:** The call chains are correct. The diagnostic routes feed into the shared executor at the declared extension points (`observation_factor_override`, `post_reset_transform`). The canonical route remains unmodified.

### Missing test coverage (not blockers, but should be recorded)

1. **No explicit initial-law tangent test.** The Kalman oracle computes `initial_mean` and `initial_covariance` gradients (lines 117–118 in `ledh_kalman_oracle_tf.py`). The canonical score recursion test (`test_ledh_canonical_score_recursion.py`, line 35) is titled "multi-step analytical score vs oracle" but does not explicitly enumerate which parameter components were checked. **Mandatory Repair 1** requires Phase 1 to verify initial-law tangents explicitly.

2. **No ratio-bias witness test.** The degenerate-reassessment artifact distinguishes `D_hat / Z_hat` from `d/dθ log Z_hat` and calls this the "ratio bias." No test file explicitly constructs a case where these two differ and reports the gap. Phase 3 (lines 521–565) is where this comparison belongs, but no test stub exists yet. This is not a blocker—the Phase 3 execution will create it—but it should be noted in the plan.

3. **No control-variate centering test.** The program mentions control variates (lines 521–565, "Combining two imperfect estimators") but does not specify how the baseline will be chosen or tested. The generic DMIS control-variate module (`frozen_dmis_control_variate_tf.py`) is implemented and tested (`test_frozen_dmis_control_variate_tf.py`), but no test verifies that an incorrectly centered baseline changes the estimand. This is a Phase 3 or Phase 4A question.

4. **No quadrature-approximation oracle test.** The program mentions "quadrature-approximation oracles" in the Phase 0.2 ladder (lines 415–419), but I found no test file named `test_..._quadrature_oracle`. The existing Kalman oracle is exact, not a quadrature approximation. If the Phase 0.2 ladder will include a Gauss-Hermite-based LGSSM score as an intermediate oracle (between exact Kalman and the full LEDH), that test does not exist yet. If the plan means something else by "quadrature oracle," clarify.

---

## Stage 4 audit: Governance and infrastructure

### Tuning scope and production program

The program correctly requires per-scope tuning (lines 60–89, 735–753). The `ledh_tuning_scope.py` module defines 14 bound fields (model, target, route, reset, horizon, data, particles, dimensions, dtype, backend, chunk policy, control family). Any changed field creates a new tuning scope. The `ledh_production_program_v1.py` module defines the canonical program:

```python
LEDH_PRODUCTION_PROGRAM_V1 = {
    "program_id": "ledh_pfpf_ot_contract_e_dual_cap_trust_region_v1",
    "reset_policy": "contract_e",
    "dual_cap_required": True,
    "trust_region_required": True,
    ...
}
```

The wiring gate (`validate_ledh_production_configuration`) enforces that any configuration labeled "production" matches this definition. The Configuration-Status-First Reporting Rule (CLAUDE.md, adopted 2026-08-25) requires every benchmark table to state program and tuning status before showing numbers.

**Verdict:** The tuning and program infrastructure is in place. The master program respects it.

### Score contract and admission statuses

The `ledh_score_contract.py` module defines target kinds and admission statuses:

```python
LEDH_SCORE_TARGET_KIND_REALIZED_FINITE_N_ESTIMATOR = "realized_finite_n_estimator"
LEDH_SCORE_ADMISSION_STATUS_FULL = "n10000_same_target_no_tape_score_admitted"
LEDH_SCORE_ADMISSION_STATUS_TINY = "tiny_score_diagnostic_not_admitted"
LEDH_SCORE_ADMISSION_STATUS_BLOCKED_NOT_RUN = "blocked_score_not_run"
LEDH_SCORE_ADMISSION_STATUS_HISTORICAL_RAW = "historical_raw_finite_n_not_admitted"
```

The program correctly distinguishes diagnostic routes (KDM algebra, IWSG resampling) from the canonical route. The diagnostic routes are labeled `ADMISSION_STATUS_TINY` or `BLOCKED_NOT_RUN` and must not be promoted to `ADMISSION_STATUS_FULL` without a separate tuning and admission campaign.

**Verdict:** The score contract is respected. The program does not claim that diagnostic routes are production-ready.

### Phase exit gates and dependencies (lines 702–733)

The program defines clear phase dependencies:

- Phase 0 → Phase 1 (identity verification depends on oracle ladder)
- Phase 1 → Phase 2, 3 (representation and score-estimator studies depend on verified call chain)
- Phase 2, 3 → Phase 4 (long-horizon study depends on proposal and score-estimator understanding)
- Phase 4 → Phase 5 (smoothing study depends on forward-filter behavior)
- Phase 6 is blocked (degenerate transitions unresolved)
- Phase 7 (heuristic dominance) depends on Phases 4, 5 but not Phase 6
- Phase 8 (tuning and replication) depends on Phases 1–5, 7
- Phase 9 (production audit) depends on all previous phases except Phase 6

**Verdict:** The dependency order is sound. Phase 6 is correctly isolated.

---

## Stage 5 audit: Evidence contract and stop rules

### Evidence contract (lines 266–298)

The program states:

- **Primary question:** Can a Gaussian-mixture KDM improve the LEDH score estimator?
- **Primary criterion:** Variance reduction at equal particle count and equal wall-clock cost (line 457).
- **Veto diagnostics:** Identity failures, oracle mismatches, non-finite values, resampling-correction failures.
- **Explanatory diagnostics:** Per-step weights, covariances, KDM bandwidths, tangent norms, OT plan quality.
- **Non-conclusions:** The program explicitly lists what will not be concluded (lines 289–298): single-seed rankings, superiority without uncertainty analysis, degenerate-model readiness, and HMC production readiness without separate validation.

**Verdict:** The evidence contract is clear and respects the Statistical Evidence Discipline (AGENTS.md). The program correctly treats particle-count and wall-clock cost as joint constraints, not as separate comparisons.

### Stop rules (lines 299–329)

The program defines stop conditions:

- **Phase vetoes:** Oracle mismatch > tolerance, identity failure, non-finite artifacts, missing diagnostics.
- **Campaign vetoes:** Budget exhausted (but see **Scope Clarification 4** below), unresolved mathematical blocker, implementation corruption.
- **Promotion vetoes:** No statistical evidence for variance reduction, heuristic dominance failure, tuning-scope mismatch, wiring-gate failure.

**One issue (Mandatory Scope Clarification 4):** The program says "Budget and time are not a problem" (line 301) and makes no GPU-hour cap visible. This is the owner's choice, but it must be acknowledged. A nine-phase factorial study over multiple models, horizons, particle counts, and proposal variants could consume hundreds of GPU-hours if every arm is run to statistical significance. If there is no budget limit, state that explicitly and accept the cost risk. If there is a limit, record it now.

---

## Stage 6 audit: Pre-mortem and option matrix

### Skeptical pre-mortem (lines 331–359)

The program lists failure modes:

- Identity tests pass but long-horizon runs diverge or produce non-finite values.
- KDM variance reduction exists but is too small to overcome the per-step cost.
- Optimal KDM bandwidth is model-specific and cannot be tuned offline.
- The integrated-observation approximation introduces bias that the analytical tangent does not account for.
- The full-mixture resampling gradient is correct but too expensive for practical use.
- The unnormalized/normalized score gap is large and KDM does not close it.

**Verdict:** The pre-mortem is adequate. Each failure mode has a corresponding diagnostic in the phase descriptions.

### Option matrix (lines 361–383)

The program enumerates proposal options (bootstrap, EKF, UKF, LEDH, KDM, forward/backward mixture, coordinate proposal), bandwidth options (fixed, adaptive, model-tuned), and resampling options (stratified, systematic, residual, KDM-IWSG). The matrix is not exhaustive—it does not list every possible combination—but it covers the main axes of variation.

**Verdict:** The option matrix is sufficient for a research program. It is not a product specification.

---

## Stage 7 audit: Consistency with governance and prior decisions

### Governance compliance

The program respects:

- **Per-scope tuning rule** (CLAUDE.md): Every claim-bearing run requires an offline tuning artifact.
- **TensorFlow backend rule** (CLAUDE.md): The implementation is TensorFlow/TFP. NumPy appears only in tests.
- **GPU default** (CLAUDE.md): The program targets GPU execution. CPU runs are reference/diagnostic only.
- **DPF transport chunk rule** (CLAUDE.md): Production routes must use `dpf_transport_exact_divisor_cap3000_v1`.
- **Configuration-status-first reporting** (CLAUDE.md): Results must label program and tuning status before showing numbers.
- **Safety guardrail reversed burden** (AGENTS.md): Observability (Class A) is adopt-by-default; fail-closed guards (Class B) require only a no-fire regression; numerics-altering protections (Class C) require a non-harm evaluation.
- **Evidence contract before research actions** (AGENTS.md): The program states question, baseline, criterion, vetoes, and non-conclusions explicitly.
- **Statistical evidence discipline** (AGENTS.md): The program treats descriptive metrics as nomination signals, not as evidence of superiority.
- **Heuristic dominance gate** (AGENTS.md): Phase 7 (lines 667–683) requires comparison against simple baselines (bootstrap, naive proposal, equal-weight) in salient regimes (short/long horizon, low/high observation noise).

**Verdict:** The program is governance-compliant.

### Consistency with degenerate-model reassessment

The degenerate-model reassessment artifact (`degenerate-model-score-reassessment.md`) concluded:

- **Close ordinary all-ancestor smoothing for degenerate DSGE:** The transition-density support identity fails for deterministic maps. Disturbance coordinates are required.
- **Retain disturbance-coordinate score as candidate:** Exact pathwise identity and toy checks pass, but DSGE-specific regimes and finite-N behavior are untested.
- **Do not use current particle score as standard HMC force:** The exact endpoint score is not established.

The master program correctly excludes degenerate transitions from Phases 1–5 and 7–9. Phase 6 (lines 630–651) is marked "unresolved" and lists six steps that would be required to close that gap (DSGE specification, exact oracle, diagnostic disturbance filter, resampling dependence test, variance-reduction ladder, inference architecture choice). The program does not claim that LEDH is HMC-ready for degenerate models.

**Verdict:** The master program is consistent with the degenerate-model reassessment. Phase 6 is correctly flagged as out of scope.

---

## Stage 8 audit: Execution readiness

### Phase 0 readiness (lines 385–420)

Phase 0 (specification and oracle ladder) is **executable now.** The model catalogue can be written from existing LGSSM fixtures. The Kalman oracle exists and is tested. The only uncertainty is what "quadrature-approximation oracles" means (see Missing Test Coverage item 4 above).

### Phase 1 readiness (lines 421–440)

Phase 1 (identity and call-chain verification) is **executable after Mandatory Repair 1.** The canonical score recursion test exists (`test_ledh_canonical_score_recursion.py`). The Kalman oracle computes initial-law tangents. The Phase 1 gate must explicitly require initial-law parity.

### Phase 2–5 readiness (lines 442–629)

Phases 2–5 (proposal/representation study, score-estimator study, long-horizon study, smoothing study) are **executable after Mandatory Repair 2 and 3.** The diagnostic KDM routes exist. The integrated-observation and resampling routes exist. The missing pieces are:

- OT reset role clarification (Mandatory Repair 2)
- UKF covariance observability (Mandatory Repair 3)
- Ratio-bias witness test (will be created in Phase 3)
- Control-variate centering test (will be created in Phase 3 or 4A)

### Phase 6 readiness (lines 630–651)

Phase 6 (degenerate-transition support program) is **blocked.** The program correctly labels it "unresolved" and lists six steps that would be required. This is not a blocker for Phases 1–5 or 7–9.

### Phase 7–9 readiness (lines 667–701)

Phases 7–9 (heuristic dominance, tuning/replication, production audit) are **executable after Phases 1–5 complete.** The heuristic dominance gate (AGENTS.md) requires comparison against simple baselines in salient regimes. The tuning and replication infrastructure exists. The production audit can be run once the earlier phases define which routes are promotion candidates.

---

## Final verdict and required actions

**APPROVE FOR EXECUTION** subject to three mandatory repairs:

1. **Add explicit initial-law tangent verification to Phase 1 gates.** The gate must verify that `initial_mean` and `initial_covariance` tangents match the Kalman oracle, not only the transition and observation tangents.

2. **Clarify OT reset role in target definition and evidence contract.** State explicitly whether the OT reset Jacobian determinant is part of `Z_hat` (the ATOM-FINITE value) or a separate correction, and whether the analytical score includes or excludes that determinant tangent.

3. **State UKF covariance observability requirement or justify why inspection is unnecessary.** If Phase 2 or later phases need to inspect UKF covariances, the implementation must expose them. If they remain internal-only, state that explicitly.

**ONE MATERIAL SCOPE CLARIFICATION (not a blocker):**

4. **Record compute budget status.** The program says "Budget and time are not a problem" but makes no GPU-hour cap visible. State explicitly whether the campaign is uncapped or whether a budget ceiling exists.

**NO BLOCKERS.** Phase 0 is executable now. Phase 1 is executable after Repair 1. Phases 2–5 are executable after Repairs 2–3. Phase 6 is correctly excluded. Phases 7–9 depend on earlier phases but have no intrinsic blockers.

---

## Reviewer statement

I have read the master program, inspected the cited implementation modules and tests, verified the mathematical definitions against the cited papers, and audited the program for consistency with the repository's governance framework (AGENTS.md, CLAUDE.md). The program is scientifically sound, mathematically well-defined, and consistent with prior decisions. The three mandatory repairs are clarifications, not fundamental design flaws. The program is approved for execution once those repairs are made.

**Reviewer:** Claude Opus 5 (1M context)  
**Review date:** 2026-09-14  
**Repository commit:** `5cc59cfa`  
**Branch:** `surrogate-hmc`

---

## Appendix: Implementation module summary

For the record, the following modules and tests were inspected during this review:

**Core LEDH score implementation:**
- `bayesfilter/highdim/ledh_canonical_score_tf.py` (682 lines, canonical wrapper)
- `bayesfilter/highdim/ledh_canonical_score_stages_tf.py` (stage-wise recursion)
- `bayesfilter/highdim/ledh_ukf_lifecycle_tf.py` (per-particle UKF)
- `bayesfilter/highdim/ledh_flow_perparticle_tf.py` (dual-state flow)
- `bayesfilter/highdim/ledh_contract_e_reset_tf.py` (OT reset)

**Younis KDM diagnostic routes:**
- `bayesfilter/highdim/ledh_younis_kdm_tf.py` (KDM algebra)
- `bayesfilter/highdim/ledh_younis_kdm_integrated_tf.py` (KDM-FINITE)
- `bayesfilter/highdim/ledh_younis_kdm_resampling_tf.py` (RESKDM-IWSG-FINITE)
- `bayesfilter/highdim/ledh_younis_kdm_lgssm_reference_tf.py` (LGSSM references)

**Oracles:**
- `bayesfilter/highdim/ledh_kalman_oracle_tf.py` (EXACT-LGSSM-SCORE)
- `bayesfilter/highdim/ledh_canonical_autodiff_oracle_tf.py` (forward autodiff oracle)

**Governance and infrastructure:**
- `bayesfilter/highdim/ledh_tuning_scope.py` (84 lines, 14 bound fields)
- `bayesfilter/highdim/ledh_alg1_contract.py` (309 lines, Contract-E spec)
- `bayesfilter/highdim/ledh_score_contract.py` (463 lines, target kinds and admission)
- `bayesfilter/highdim/ledh_production_program_v1.py` (263 lines, wiring gate)
- `bayesfilter/highdim/transport_chunk_policy.py` (119 lines, chunk policy)

**Tests:**
- `tests/highdim/test_ledh_younis_kdm_tf.py` (890+ lines, 13 tests)
- `tests/highdim/test_ledh_younis_kdm_lgssm_reference_tf.py` (188+ lines, 6 tests)
- `tests/highdim/test_ledh_kalman_oracle_tf.py` (124+ lines, 3 tests)
- `tests/highdim/test_ledh_canonical_score_recursion.py` (multi-step analytical score vs oracle)
- `tests/highdim/test_ledh_canonical_score_stages.py` (stage-wise tests)
- `tests/highdim/test_ledh_tuning_scope.py` (scope round-trip and change detection)
- `tests/highdim/test_ledh_score_wiring_phase8_cross_model.py` (wiring gates)
- `tests/highdim/test_ledh_canonical_meta_governance.py` (registry coverage)

All cited modules and tests exist and are internally consistent with the master program's claims.
