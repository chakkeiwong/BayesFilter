# SQMC 4-Route Oracle Comparison: Master Program v2

**Date:** 2026-09-09  
**Status:** AUTHORITATIVE - Supersedes v1  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Supersedes:** `sqmc-oracle-comparison-master-program-2026-09-09.md` (v1, 896-cell direct approach)

---

## Executive Summary

This program measures **accuracy** of 4 SQMC transport routes against Kalman filter oracles on LGSSM and KSC-SV models. 

**Key differences from v1:**
1. **Phased execution:** Characterization first (8 cells), then expand based on results
2. **Tuning strategy:** Warm-start controls + adaptive state map (no per-model tuning required)
3. **Decision gates:** Route distinguishability determines full campaign scope
4. **Timeline:** Hours to decision (not weeks)

---

## Background

**Austria SIR precedent:** Completed SQMC comparison used warm-start controls (not per-model tuned) and found all 4 routes **statistically indistinguishable**.

**Oracle comparison goal:** Test if oracle reveals accuracy differences that internal consistency (Austria SIR) did not.

**Parameter analysis:** Mathematical analysis shows only MAP location/scale truly needs per-model tuning, but adaptive state map eliminates this requirement. All other parameters have validated defaults.

**See:** `docs/plans/sqmc-parameters-tuning-analysis-2026-09-09.md` for mathematical justification.

---

## Models and Oracles

### Model 1: LGSSM (Linear Gaussian State-Space Model)

**State dimension:** 10  
**Observation dimension:** 10  
**Oracle:** Kalman filter (exact marginal log-likelihood and score)  
**Implementation:** `bayesfilter/highdim/ledh_kalman_oracle_tf.py` ✓ Verified Phase 0

**Horizons:**
- T=20 (characterization baseline)
- T=50 (moderate scaling, optional expansion)
- T=360 (full-year stress test, optional expansion)

### Model 2: KSC-SV (Kim-Shephard-Chib Stochastic Volatility)

**State dimension:** 1 (scalar log-volatility)  
**Observation dimension:** 1  
**Oracle:** Dense Kalman filter  
**Implementation:** `bayesfilter/highdim/sv_mixture_cut4.independent_panel_sv_mixture_kalman_filter` ✓ Verified Phase 0

**Horizons:**
- T=10 (standard SV horizon)
- T=20, T=50, T=120 (optional expansion)

---

## Production Algorithm Configuration

**Code path:** `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.finite_value_standard_score_initial_rqmc`

**Reset:** Contract-E with GenUT  
**Dual-cap:** Enabled (diagonal + pairwise + radial + coordinate caps)  
**Trust-region:** Enabled (Levenberg-Marquardt damping)  
**Transport:** Streaming (K≤3000 chunk rule compliant)  
**Execution:** TF32 GPU, float32 tensors

**This matches CLAUDE.md Default Execution Target.** ✓ Verified

---

## Tuning Strategy: Warm-Start + Adaptive

**Key decision:** Use **warm-start controls with adaptive state map** to avoid 56 per-model tuning artifacts.

### Transport Parameters (warm-start from Austria SIR)
```
epsilon = 8.0              # Entropic regularization
sinkhorn_steps = 8         # Sinkhorn iterations
balance_steps = 8          # Balancing iterations
ridge = 1e-5               # Numerical regularization
```

### Dual-Cap Parameters (validated defaults from four-model evidence)
```
diagonal_steps = 4         # Diagonal moment correction iterations
diagonal_strength = 0.2    # Step size (Austria SIR value)
pairwise_steps = 4         # Pairwise moment correction iterations
pairwise_strength = 0.02   # Step size (LGSSM/KSC-SV value)
radial_cap = 2.0           # Radial RMS cap (universal default)
coordinate_cap = 0.98      # Coordinate cap (universal default)
coordinate_cap_power = 8   # Cap smoothness
```

**Exception:** `repaired_permutation` route uses slightly different values (inherited from Austria SIR):
```
diagonal_steps = 3, diagonal_strength = 0.15
pairwise_steps = 3, pairwise_strength = 0.01
radial_cap = 1.5, coordinate_cap = 0.97, power = 6
```

### Trust-Region Parameters (conservative Austria SIR values)
```
lm_damping = 1e-2          # Conservative (vs tuned 1e-3)
lm_scale_floor = 1e-4      # Numerical floor
trust_radius = 0.5         # Conservative (vs tuned 0.1)
```

### State Map Policy (CRITICAL)
```
state_map_policy = "adaptive_empirical"  # No MAP tuning needed
hilbert_bits = 12
```

**Justification:**
- Adaptive state map eliminates need for per-model MAP tuning (56 artifacts avoided)
- Warm-start dual-cap/trust-region values validated across 4 models
- Austria SIR found routes equivalent even with warm-start (not tuned)
- Conservative trust-region (damping 1e-2 vs tuned 1e-3) prioritizes stability

**Documentation requirement:**
Every result must state: *"Configuration uses warm-start controls with adaptive state map (no per-model tuning). Comparison measures relative route accuracy, not absolute optimally-tuned performance."*

---

## Routes Tested

The 4 SQMC routes are **transport ancestry policies**:

| Route | Ancestry Policy | Description |
|---|---|---|
| `iid_dual_cap` | `existing_one_to_one` | No reordering, identity ancestors |
| `previous_inverse_cdf` | `hilbert_inverse_cdf` | Hilbert curve + inverse CDF sampling |
| `repaired_fixed_previous_controls` | `hilbert_permutation_one_to_one` | Hilbert curve + fixed permutation |
| `repaired_permutation` | `hilbert_permutation_one_to_one` | Hilbert curve + permutation (different controls) |

**Shared:** Reset (Contract-E), transport algorithm (Sinkhorn), trust region  
**Different:** Ancestry selection, dual-cap hyperparameters, state map scale

---

## Phased Execution Plan

### Phase 0: Pre-execution Verification ✓ COMPLETE

**Deliverables:**
1. ✓ Verify LGSSM Kalman oracle (`ledh_kalman_oracle_tf.py`)
2. ✓ Verify KSC-SV dense Kalman oracle
3. ✓ Smoke test all horizons

**Artifact:** `docs/benchmarks/smoke_test_oracles_sqmc_comparison.py`  
**Result:** PASSED - all oracles functional, all horizons feasible  
**Date:** 2026-09-09

---

### Phase 1: Characterization (LGSSM T=20, N=1008)

**Purpose:** Test if oracle reveals route differences before committing to full campaign.

**Configuration:**
- Model: LGSSM T=20 (10D state/obs)
- Particle count: N=1008
- Routes: All 4
- Seeds: **2 only** (97701, 97702)
- Controls: Warm-start + adaptive state map

**Total cells:** 4 routes × 2 seeds = **8 cells**  
**Estimated time:** ~15 minutes GPU

**Metrics:**
- Value error: |SQMC_value - Kalman_value|
- Score L2 error: ||SQMC_score - Kalman_score||_2
- Per-route mean/std across 2 seeds

**Success criteria:**
- All cells complete with finite SQMC and oracle values
- Errors computable and reasonable (<1e6)
- Route distinguishability assessed

**Decision gates:**

#### Gate 1A: Routes Indistinguishable (Likely)
If route-to-route error range < seed-to-seed std:
- **Conclusion:** Austria SIR finding confirmed with oracle
- **Action:** Optionally expand to 16 seeds for statistical confidence, then proceed to Phase 2 (other models/horizons)
- **Tuning:** Not needed - warm-start sufficient

#### Gate 1B: Clear Route Winner (Possible)
If one route consistently lower error across both seeds:
- **Conclusion:** Oracle reveals accuracy differences
- **Action:** Expand to 16 seeds to confirm statistical significance
- **Tuning:** Optional - targeted tuning for winning route only

#### Gate 1C: Marginal/Inconclusive (Unlikely)
If results noisy or routes marginally different:
- **Conclusion:** Need more seeds for statistical power
- **Action:** Expand to 16 seeds before deciding
- **Tuning:** Defer until statistical pattern clear

**Deliverables:**
- `docs/benchmarks/run_sqmc_oracle_characterization.py` - execution runner
- `docs/benchmarks/artifacts/sqmc-oracle-characterization-20260909/result.json`
- `docs/plans/sqmc-oracle-characterization-result-2026-09-09.md` - analysis and decision

**Timeline:** 1-2 days (implementation + execution + analysis)

---

### Phase 2: Expansion (Conditional)

**Trigger:** Phase 1 decision gate outcome

#### Scenario A: Routes Equivalent (Most Likely)

If Phase 1 confirms routes indistinguishable:

**Option A1: Report findings, no further testing**
- Document: "All 4 routes statistically equivalent with oracle (confirming Austria SIR)"
- Timeline: Complete
- Total cells: 8 (characterization only)

**Option A2: Expand to validate across models/horizons**
- LGSSM T=20 → 16 seeds (64 cells total)
- Optional: LGSSM T=50 (64 cells), KSC-SV T=10 (64 cells)
- Purpose: Build confidence that finding generalizes
- Timeline: +1-2 days per model/horizon

#### Scenario B: Routes Differ (Possible)

If Phase 1 shows clear winner:

**Phase 2B.1: Statistical Confirmation**
- LGSSM T=20, N=1008, 4 routes × 16 seeds = 64 cells
- Compute bootstrap CIs, pairwise tests
- Confirm statistical significance

**Phase 2B.2: Targeted Tuning (Optional)**
- If winner has tuning potential, run focused tuning grid
- Compare tuned winner vs warm-start winner
- Document improvement magnitude

**Phase 2B.3: Model Generalization**
- Test winner on LGSSM T=50, T=360
- Test winner on KSC-SV T=10
- Assess if ranking holds across models

**Timeline:** +1-2 weeks (statistical confirmation + optional tuning + generalization)

---

### Phase 3: Full Campaign (Optional, Conditional)

**Trigger:** User decision to complete full matrix

**Scope:**
- LGSSM: T=20/50/360 × N=1008/2016 × 4 routes × 16 seeds = 384 cells
- KSC-SV: T=10/20/50/120 × N=1008/2016 × 4 routes × 16 seeds = 512 cells
- **Total: 896 cells**

**Timeline:** ~30-40 hours GPU time (distributed or sequential)

**Purpose:** Complete characterization matrix, test:
- Horizon scaling (does ranking change with T?)
- Particle count effect (does N=2016 change ranking?)
- Model dependence (LGSSM vs KSC-SV consistent?)

**Decision:** Only execute if Phase 1-2 justify the investment (i.e., routes differ and understanding scaling is scientifically valuable).

---

## Implementation Status

### Complete ✓
- Phase 0 verification (oracles tested)
- Parameter analysis (mathematical roles identified)
- Production code verification (correct path confirmed)
- Characterization runner skeleton (`run_sqmc_oracle_characterization_step1.py`)

### In Progress
- Phase 1 implementation: LGSSM model construction (next step)

### Blocked
- Phase 2+: Awaiting Phase 1 results

---

## Timeline Summary

| Phase | Description | Cells | GPU Time | Wall Time |
|---|---|---|---|---|
| 0 | Pre-execution verification | 14 (smoke) | <5 min | ✓ Complete |
| 1 | Characterization (LGSSM T=20, 2 seeds) | 8 | ~15 min | 1-2 days* |
| 2 | Expansion (conditional) | 64-256 | 1-4 hours | +days/weeks |
| 3 | Full campaign (optional) | 896 | 30-40 hours | +weeks |

*Wall time includes implementation (6-8 hours) + execution (15 min) + analysis (1 hour)

---

## Success Criteria

### Characterization Success (Phase 1)
- ✓ All 8 cells complete with finite values
- ✓ Oracle values match expected ranges
- ✓ Errors computable (not NaN/Inf)
- ✓ Route distinguishability assessed
- ✓ Decision gate outcome documented

### Campaign Success (if Phase 2/3 executed)
- All cells complete with finite values
- Statistical analysis complete (bootstrap CIs, pairwise tests)
- Route ranking documented with uncertainty
- Horizon/particle-count effects characterized
- Model-dependence assessed

### Scientific Success
- Answer: "Do SQMC routes differ in oracle accuracy?"
- If yes: Which route is most accurate? Does ranking generalize?
- If no: Confirm Austria SIR finding extends to oracle comparison

---

## Risk Mitigation

### Risk 1: Implementation Complexity
**Mitigation:** Phased approach - characterization first (8 cells) validates integration before scaling

### Risk 2: Routes Equivalent (No Tuning Needed)
**Impact:** Positive - validates warm-start strategy, no further work needed  
**Response:** Document findings, optionally expand for confidence

### Risk 3: Routes Differ (Tuning May Help)
**Impact:** Neutral - targeted tuning becomes scientifically justified  
**Response:** Phase 2B targeted tuning for winning route only (not all 56 artifacts)

### Risk 4: LGSSM T=360 Infeasible
**Mitigation:** Test T=50 first; if timing reasonable, proceed to T=360; otherwise skip

---

## Documentation Requirements

Every result artifact must include:

**Configuration disclosure:**
```
Configuration: Warm-start controls + adaptive state map
- No per-model tuning artifacts
- Transport: ε=8.0, 8 steps (Austria SIR warm-start)
- Dual-cap: Validated defaults from four-model evidence
- Trust-region: Conservative values (damping 1e-2, radius 0.5)
- State map: Adaptive empirical (no MAP tuning)
```

**Interpretation boundaries:**
```
Comparison measures:
- ✓ Relative route accuracy (routes vs each other)
- ✓ Oracle error magnitude (SQMC vs ground truth)
- ✗ Absolute optimally-tuned performance
- ✗ Production readiness without model-specific tuning
```

**Non-claims:**
```
This comparison does NOT establish:
- That warm-start controls are optimal for these models
- That routes would remain equivalent with model-specific tuning
- That oracle accuracy predicts HMC/NeuTra performance
- Production leaderboard rankings (requires per-model tuning per CLAUDE.md)
```

---

## References

1. **Production verification:** `docs/plans/sqmc-oracle-production-verification-2026-09-09.md`
2. **Parameter analysis:** `docs/plans/sqmc-parameters-tuning-analysis-2026-09-09.md`
3. **User decision:** `docs/plans/sqmc-oracle-decision-summary-2026-09-09.md`
4. **Phase 0 artifact:** `docs/benchmarks/smoke_test_oracles_sqmc_comparison.py` (PASSED)
5. **Austria SIR precedent:** `docs/plans/sqmc-rerun-corrected-filter-2026-09-06.md`
6. **Dual-cap evidence:** `docs/genut-dual-cap-default-algorithm-integration-note-2026-08-07.md`
7. **Trust-region evidence:** `docs/memos/ledh-trust-region-phase3-complete-2026-09-02.md`

---

## Version History

**v2 (2026-09-09):** Authoritative version
- Phased execution (characterization → expansion)
- Warm-start + adaptive state map strategy
- Decision gates based on distinguishability
- Timeline: hours to decision (not weeks)
- User-approved Option 3

**v1 (2026-09-09, superseded):** Initial master program
- Direct 896-cell campaign
- No tuning strategy specified
- No characterization phase
- Status: Superseded, not executed

