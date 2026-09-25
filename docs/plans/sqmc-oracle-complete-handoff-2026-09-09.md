# SQMC Oracle Comparison: Complete Recovery and Next Steps

**Date:** 2026-09-09  
**Status:** RECOVERY COMPLETE - Ready for implementation  
**Branch:** `rqmc-sqmc-4route-comparison`

---

## Recovery Summary

**Previous Problem:** Agent created planning documents but stalled before execution. Root cause: 896-cell scope too large to implement in one step.

**Recovery Actions:**
1. ✓ Verified production code path (LEDH PFPF-OT with Contract-E, dual-cap, trust-region)
2. ✓ Verified both oracles work (Phase 0 smoke test passed)
3. ✓ Analyzed parameter tuning requirements mathematically
4. ✓ User approved Option 3: Characterization first, tune if needed
5. ✓ Created characterization runner skeleton

---

## Key Findings

### Production Code: ✓ VERIFIED CORRECT

**File:** `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`  
**Function:** `finite_value_standard_score_initial_rqmc`

This is the genuine production LEDH PFPF-OT code matching CLAUDE.md requirements:
- Contract-E reset with GenUT
- Dual-cap covariance stabilization
- Trust-region Levenberg-Marquardt damping
- Streaming/chunked transport (K≤3000 rule)
- TF32 GPU execution

### Parameter Tuning Analysis: ✓ COMPLETE

**Critical insight:** Only **MAP location/scale** truly needs per-model tuning, but we can **avoid this by using adaptive state map**.

All other parameters have validated defaults from four-model evidence (LGSSM, KSC-SV, Austria SIR, Predator-Prey):
- Transport (ε, steps): Numerical convergence parameters with known behavior
- Dual-cap (radial/coordinate caps, strengths): Validated across 4 models
- Trust-region (damping, radius): Conservative values work, fine-tuning gives ~26% improvement but doesn't change route rankings

**See:** `docs/plans/sqmc-parameters-tuning-analysis-2026-09-09.md` for mathematical analysis.

### User Decision: Option 3 Approved

Run characterization first (8 cells, 2 seeds) with warm-start controls + adaptive state map. If routes equivalent → done. If routes differ → targeted tuning.

---

## Current State

### What Exists

**Phase 0 verification:**
- `docs/benchmarks/smoke_test_oracles_sqmc_comparison.py` ✓ PASSED
- Both LGSSM Kalman and KSC-SV dense Kalman oracles verified working
- All horizons (T=20/50/360 for LGSSM, T=10/20/50/120 for KSC-SV) feasible

**Documentation:**
- `docs/plans/sqmc-oracle-production-verification-2026-09-09.md` — Production code verification
- `docs/plans/sqmc-parameters-tuning-analysis-2026-09-09.md` — Parameter roles and tuning criticality
- `docs/plans/sqmc-oracle-decision-summary-2026-09-09.md` — User decision record
- `docs/plans/sqmc-oracle-recovery-plan-2026-09-09.md` — Recovery diagnosis

**Characterization runner:**
- `docs/benchmarks/run_sqmc_oracle_characterization_step1.py` — Skeleton implemented, tested
- Status: Works up to LGSSM model construction, then raises NotImplementedError

---

## Next Implementation Step

### Immediate: LGSSM Model Construction

**Task:** Adapt `diagonal_lgssm_callbacks()` for 10D state/obs, T=20 horizon.

**Current:** `bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py` has 3D diagonal LGSSM with 5 parameters.

**Needed:** 10D diagonal LGSSM with parameters suitable for Kalman oracle gradient tracking.

**Approach:**
1. Study existing `diagonal_lgssm_callbacks()` (lines 237-260)
2. Identify parameter structure: theta → (F_diag, Q_diag, R_diag)
3. Extend to 10D: theta → (F_diag[10], log(Q_scale), log(R_scale))
4. Wire up initial_log_density, transition_log_density, observation_log_density
5. Create callbacks object with 10D observation matrix

**Files to modify:**
- `run_sqmc_oracle_characterization_step1.py`: Implement `create_lgssm_model()`

**Estimated time:** 2-3 hours

---

### After LGSSM Construction: SQMC Integration

**Task:** Call `finite_value_standard_score_initial_rqmc` with LGSSM callbacks.

**Components needed:**
1. Generate RQMC point sets (initial, process, ancestors)
2. Create design matrix for Contract-E
3. Call SQMC with warm-start controls
4. Extract value, score, diagnostics

**Reference:** Austria SIR runner `_row()` function (lines 297-405)

**Estimated time:** 2-3 hours

---

### After SQMC Integration: Run Characterization

**Task:** Execute 8-cell characterization, analyze results.

**Cells:** 4 routes × 2 seeds = 8
**Runtime:** ~10-15 minutes GPU
**Outputs:**
- Per-cell: SQMC value/score, oracle value/score, errors
- Aggregate: Mean errors per route, statistical distinguishability

**Decision gates:**
- Routes indistinguishable → Austria SIR finding confirmed, proceed to full campaign
- Routes differ → Targeted tuning for differentiating configurations
- Implementation issues → Fix before scaling

**Estimated time:** 30 minutes (execution + analysis)

---

## Configuration: Warm-Start Controls

**From:** Austria SIR runner + four-model dual-cap evidence

```python
# Transport (warm-start)
epsilon = 8.0
sinkhorn_steps = 8
balance_steps = 8
ridge = 1e-5

# Dual-cap (validated defaults)
diagonal_steps = 4
diagonal_strength = 0.2  # Austria SIR
pairwise_steps = 4
pairwise_strength = 0.02  # LGSSM/KSC-SV
radial_cap = 2.0  # Universal
coordinate_cap = 0.98  # Universal
coordinate_cap_power = 8

# Trust-region (conservative)
lm_damping = 1e-2  # More conservative than tuned 1e-3
lm_scale_floor = 1e-4
trust_radius = 0.5  # More conservative than tuned 0.1

# State map (CRITICAL: adaptive to avoid MAP tuning)
state_map_policy = "adaptive_empirical"
hilbert_bits = 12
```

**Documentation requirement:**
> "Configuration uses warm-start controls with adaptive state map (no per-model tuning). Comparison measures relative route accuracy, not absolute optimally-tuned performance."

---

## Timeline Estimate

From current state to characterization results:

| Step | Description | Time |
|---|---|---|
| 1 | Implement LGSSM model construction | 2-3 hours |
| 2 | Integrate SQMC calling code | 2-3 hours |
| 3 | Test single cell end-to-end | 30 min |
| 4 | Run 8-cell characterization | 15 min |
| 5 | Analyze results, create report | 30 min |
| **Total** | **6-8 hours to decision point** | |

After decision:
- If routes equivalent: Expand to other models/horizons (hours/days)
- If routes differ: Targeted tuning (days/weeks)

---

## Files Created This Session

1. `smoke_test_oracles_sqmc_comparison.py` ✓ PASSED
2. `sqmc-oracle-production-verification-2026-09-09.md`
3. `sqmc-parameters-tuning-analysis-2026-09-09.md`
4. `sqmc-oracle-decision-summary-2026-09-09.md`
5. `sqmc-oracle-recovery-plan-2026-09-09.md`
6. `phase2b-step1-characterisation-tests-plan-20260909.md`
7. `run_sqmc_oracle_characterization_step1.py` (skeleton, tested)
8. `sqmc-oracle-complete-handoff-2026-09-09.md` (this file)

All committed to branch `rqmc-sqmc-4route-comparison`.

---

## Summary for User

**Recovery Complete:** ✓

**Production code verified:** ✓ Correct LEDH PFPF-OT path

**Oracles verified:** ✓ Both LGSSM and KSC-SV work

**Parameter analysis:** ✓ Mathematical roles understood, warm-start strategy validated

**User decision:** ✓ Option 3 approved (characterization with warm-start)

**Next:** Implement LGSSM model construction (2-3 hours), then SQMC integration (2-3 hours), then run characterization (15 min).

**Total time to decision:** 6-8 hours from current state.

