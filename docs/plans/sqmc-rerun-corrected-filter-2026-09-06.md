# SQMC Re-Run with Corrected GenUT DCTR Filter: Execution Plan

**Date:** 2026-09-06  
**Status:** READY FOR OWNER APPROVAL  
**Supersedes:** None (new campaign, not a revision)  
**Related:** bayesfilter-genut-sqmc-particle-count-trust-region-result-2026-08-17.md (August baseline)

## Motivation

The August 17 SQMC campaign tested Gerber–Chopin sequential quasi-Monte Carlo with:
- Hilbert-curve ancestry ordering
- Four policies: `iid_dual_cap`, `previous_inverse_cdf`, `repaired_fixed_previous_controls`, `repaired_permutation`
- Particle counts: N = 1008, 2016, 4032
- 16 randomized seeds per cell

**Critical finding**: The trust-region repair achieved 192/192 valid rows, while the legacy dual-cap produced 0/64 valid rows at Austria-SIR T=20, N=1008.

**Remaining uncertainty**: The August campaign compared the **repaired trust-region filter** to the **legacy dual-cap filter**. But both were tested **before** recent filter corrections (post-Aug 17 commits). Owner requests: **re-run the exact same experiment with the current corrected filter**.

---

## Research Question

**Does the current corrected GenUT DCTR filter (post-Aug 17) preserve the SQMC variance scaling properties observed in August, and does it improve numerical validity or variance reduction further?**

### Success Criteria

1. **Validity**: All cells complete with finite values (192/192 for trust-region DCTR)
2. **Variance scaling**: Variance ratios at N=2016, 4032 vs N=1008 approach theoretical landmarks (1/√2 ≈ 0.707, 1/2 = 0.500)
3. **Comparison to August**: Mean values stable (±1 SD of August), variance comparable or improved

### Non-goals

- This is **not** testing initialization effects (that was V1/V2)
- This is **not** adding new models or arms
- This is a **replication** of the August scope with the current filter

---

## Exact Scope (Matching August 17)

### Models
- **Austria-SIR T=20** (primary discriminating case)
- Optional: Six-model T=6, N=72 port smoke (if requested)

### Arms (SQMC Ancestry Policies)
1. `iid_dual_cap` (baseline, no Hilbert ordering)
2. `previous_inverse_cdf` (Hilbert + inverse CDF ancestry)
3. `repaired_fixed_previous_controls` (Hilbert + fixed controls)
4. `repaired_permutation` (Hilbert + permutation)

### Particle Counts
- N = 1008, 2016, 4032

### Seeds
- 16 randomized seeds: 97701-97716 (same as August)

### Filter Route
- `genut_column_scaled_lm_smooth_rms_trust_v1` (current corrected version)
- Trust-region controls: LM damping 1e-2, scale floor 1e-4, trust radius 0.5

### Total Runs
4 arms × 3 particle counts × 16 seeds = **192 cells**

### Point-Set Configuration (from sqmc_tf.py)
- Hilbert implementation: `skilling_transpose_tf_lexicographic_int30_v2`
- Ancestor CDF: `empirical_inverse_cdf_right_open_v1`
- Point set: `tfp_owen2017_randomized_halton_v1` (Owen-scrambled Halton)
- State map: `componentwise_logistic_empirical_mean_std_floor_v2`

---

## Primary Diagnostics

### 1. Validity Check
Count finite rows per arm and particle count. Compare to August:
- August trust-region: 192/192 valid
- August legacy: 0/64 valid
- Target: Current filter ≥ 192/192 valid

### 2. Variance Scaling
For each arm, compute:
```
SD_ratio(N) = SD_at_N / SD_at_1008
```

Theoretical landmarks:
- N=2016 (2×): ideal ratio = 1/√2 ≈ 0.707
- N=4032 (4×): ideal ratio = 1/2 = 0.500

Report:
- Per-arm SD ratios for j0, j1, j2, total score
- Deviation from landmarks
- Comparison to August ratios

### 3. Mean Stability
For each (arm, N) cell:
```
mean_current - mean_august
```

Expected: within ±1 SD of August (RQMC targets same distribution, means should agree)

If |difference| > 2 SD: flag as potential regression or improvement

### 4. Wall-Time Per Cell
Track execution time to estimate total campaign cost and detect performance regressions.

---

## Execution Protocol

### Phase 0: Smoke Test (1 hour budget)
- Run 1 seed for each (arm, N) at Austria-SIR T=20
- 4 arms × 3 N = 12 cells
- Verify all finite, compare first-seed values to August
- If any cell fails: STOP, debug, do not proceed

### Phase 1: Full Campaign (8-12 hour budget)
- Run all 192 cells (4 arms × 3 N × 16 seeds)
- Parallel execution where possible (device permitting)
- Stop conditions:
  - Validity rate < 95% after 50% of cells
  - Wall time exceeds 15 hours
  - Device memory/thermal issues

### Phase 2: Analysis (2 hour budget)
- Compute variance scaling ratios
- Compare to August baseline
- Generate comparison tables and diagnostic plots

### Phase 3: Result Documentation (1 hour)
- Write result memo with verdict
- Archive artifacts
- Update SQMC ledger

---

## Comparison to August Baseline

### What Changed Since August 17

**Filter corrections** (post-Aug 17):
- Check git log from 2026-08-17 to 2026-09-06 for `ledh_contract_e_reset_tf.py`, `cubature_genut_filter.py`
- Document any numerical stability, trust-region, or reset fixes

**Same as August**:
- Seeds: 97701-97716
- Particle counts: 1008, 2016, 4032
- Route ID: `genut_column_scaled_lm_smooth_rms_trust_v1`
- Point-set: Owen-scrambled Halton
- Model: Austria-SIR T=20

### August Results Summary (for comparison)

From `bayesfilter-genut-sqmc-particle-count-trust-region-result-2026-08-17.md`:

**Validity:**
- Trust-region: 192/192 valid
- Legacy: 0/64 valid

**Variance scaling (repaired_permutation arm, j0 component):**
- N=1008: SD = 599.71
- N=2016: SD = 529.55 → ratio 0.883 (ideal 0.707)
- N=4032: SD = 313.35 → ratio 0.522 (ideal 0.500)

**Conclusion**: Larger N helps, but not a clean 1/N law. The j0 problem persists even at 4× particles.

**Current campaign aim**: Check whether filter corrections since August improve the variance scaling or numerical stability.

---

## Evidence Contract

### Promotion Criteria (Current Filter is Production-Ready)
- Validity ≥ 192/192 (matches or exceeds August)
- Variance scaling at N=4032: SD ratio ≤ 0.55 for at least one arm (improved or matches August)
- Mean values: |current - august| ≤ 2 SD for all arms

### Regression Veto (Current Filter is Worse Than August)
- Validity < 180/192 (>6% failure rate)
- Variance scaling at N=4032: SD ratio > 0.65 for all arms (significantly worse than August)
- Mean values: |current - august| > 3 SD for any arm (distributional shift)

### Neutral Verdict (Inconclusive or Mixed)
- Validity 180-192/192 (minor failures)
- Variance scaling mixed (some arms better, some worse)
- Mean values stable but variance unchanged

---

## Artifact Schema

Each `result.json` must include:

```json
{
  "schema_version": "sqmc_rerun_v1",
  "model": "austria_sir_T20",
  "arm": "repaired_permutation",
  "particle_count": 1008,
  "seed": 97701,
  "status": "complete",
  "value": -681.8125,
  "components": {
    "j0": -450.23,
    "j1": -120.45,
    "j2": -111.12
  },
  "wall_seconds": 285.4,
  "route_id": "genut_column_scaled_lm_smooth_rms_trust_v1",
  "point_set_id": "tfp_owen2017_randomized_halton_v1",
  "hilbert_impl_id": "skilling_transpose_tf_lexicographic_int30_v2",
  "git_commit": "...",
  "filter_corrections_since_aug17": ["commit_sha_1", "commit_sha_2"],
  "device": "4080_SUPER",
  "comparison_to_august": {
    "august_value": -681.65,
    "difference": 0.1625,
    "august_sd": 0.5389
  }
}
```

Phase 3 analyzer must:
- Verify all cells have matching `route_id`, `point_set_id`, `hilbert_impl_id`
- Compute variance scaling ratios per arm
- Compare to August baseline with difference statistics
- Generate comparison tables

---

## Runner Specification

### Entry Point
- Use existing `ledh_pfpf_genut_initial_rqmc_tf.py` or create a new runner if needed
- Ensure ancestry policies match August: `ANCESTRY_POLICIES = ("existing_one_to_one", "hilbert_inverse_cdf", "hilbert_systematic_equal_weight", "hilbert_permutation_one_to_one")`

### Key Parameters
```python
MODEL = "austria_sir_T20"
ARMS = ["iid_dual_cap", "previous_inverse_cdf", "repaired_fixed_previous_controls", "repaired_permutation"]
PARTICLE_COUNTS = [1008, 2016, 4032]
SEEDS = range(97701, 97717)  # 16 seeds
ROUTE_ID = "genut_column_scaled_lm_smooth_rms_trust_v1"
TRUST_REGION_CONTROLS = {
    "higher_moment_lm_damping": 1e-2,
    "higher_moment_lm_scale_floor": 1e-4,
    "higher_moment_trust_radius": 0.5,
}
```

### Device
- Primary: 4080 SUPER (CUDA_VISIBLE_DEVICES=1)
- Fallback: 5080 if needed

### Estimated Cost
- 192 cells × ~5 min/cell = 960 min ≈ **16 hours** (sequential)
- With 3-4 parallel workers: **4-6 hours**

---

## Decision Point

**This plan is not approved for execution.** Owner review required on:

1. Confirm scope matches August exactly (models, arms, N, seeds)
2. Approve 16-hour budget (or request cost reduction)
3. Confirm device availability (4080 SUPER)
4. Approve Phase 0 smoke test before full campaign

Once approved, proceed to:
1. Document filter corrections since Aug 17 (git log)
2. Implement runner or verify existing runner matches spec
3. Execute Phase 0 smoke test
4. Execute Phase 1 full campaign if smoke passes
5. Phase 2 analysis and Phase 3 documentation

---

## Relationship to V1/V2 Campaigns

**V1 campaign (what I audited today)**:
- Tested initialization at t=0 only
- Arms: sobol_owen, halton_owen, genut_guided
- N=1008 only, 3 seeds
- Estimand: mean terminal log-likelihood
- Found: process-noise confound, genut identity issue
- Status: invalid, confounded

**V2 campaign (proposed redesign)**:
- Tested initialization at t=0 only (same mechanism as V1)
- Arms: mc, sobol_owen, halton_owen, halton_reverse, lhs
- N=1008, 10 seeds
- Estimand: mean terminal log-likelihood
- Status: design complete, awaiting owner decision

**SQMC re-run (this plan)**:
- Tests ancestry ordering at every resampling step (Gerber–Chopin mechanism)
- Arms: 4 Hilbert policies
- N=1008/2016/4032, 16 seeds
- Estimand: variance scaling across N
- Status: awaiting owner approval

These are **three independent experiments** testing different mechanisms. The SQMC re-run is **not** a replacement for V1/V2 — it tests a completely different hypothesis.

---

## References

- August baseline: `docs/plans/bayesfilter-genut-sqmc-particle-count-trust-region-result-2026-08-17.md`
- Gerber–Chopin paper: `.localresources/papers/gerber-chopin-sequential-quasi-monte-carlo-arxiv-1402.4039.pdf`
- Implementation: `bayesfilter/highdim/sqmc_tf.py`, `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`
- Filter route: `bayesfilter/highdim/genut_shape_lm_tf.py` (route ID definition)
