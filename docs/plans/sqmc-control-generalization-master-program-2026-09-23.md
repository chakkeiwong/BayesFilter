# SQMC Control Generalization: Dimension and Horizon Transfer

**Date:** 2026-09-23  
**Status:** DRAFT - Ready for Execution  
**Branch:** `rqmc-sqmc-4route-comparison`  
**Worktree:** `/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909`

---

## Research Question

**Primary:** Do exact-scope tuned SQMC transport controls (epsilon, Sinkhorn steps, correction strengths) generalize across state dimensions (3D → 10D) and time horizons (T=20 → T=120)?

**Secondary:** 
- Is the dimension-generic P44 LGSSM infrastructure compatible with SQMC tuning?
- What particle counts satisfy the `N % (2*D) == 0` sigma-point constraint across dimensions?
- Do the four ancestry routes remain indistinguishable when dimension varies?

---

## Background and Motivation

### Completed Prior Work

**3D LGSSM T=20 N=1008 Campaign** (`sqmc-master-program-2026-09-12.md`):
- ✓ All 4 routes tuned to exact-scope controls (Sep 13-14, 2026)
- ✓ Tuned vs UNTUNED comparison complete
- ✓ Oracle: exact Kalman gradient
- ✓ Result: Tuning improved L2 error by ~1.8% while maintaining direction quality
- ✓ Artifact: `docs/tuning/sqmc-lgssm-t20-n1008-{route}-20260912/`

**Key findings from 3D baseline:**
- Tuned controls: `epsilon=16.0`, `sinkhorn_steps=8`, `balance_steps=8`, `correction_strength=0.15-0.2`, `pairwise_strength=0.03`
- Cosine similarity: 0.9995+ (excellent direction)
- L2 error: improved from 1.50 (UNTUNED) to 1.48 (TUNED)
- All 4 routes statistically indistinguishable after tuning

### Why This Question Matters

**LEDH Per-Scope Tuning Rule:** "Every claim-bearing run requires an offline tuning artifact for the exact model/target, route/reset family, horizon, particle count, dimensions, dtype/backend, and chunk policy. Any changed bound field is a new tuning scope."

**Practical concern:** If we must retune for every (D, T, N) combination, the tuning cost is prohibitive. But if controls transfer with predictable adjustments, one tuning campaign can seed many scopes.

**Scientific question:** Are transport quality metrics (Sinkhorn convergence, correction stability) dimension-invariant or dimension-sensitive?

### Infrastructure Gap Discovered

**Frozen 3D canonical model limitation:**
- Current tuning uses `diagonal_lgssm_canonical_model` (hard-coded 3D)
- Frozen observations are 3D-specific
- Cannot test dimension transfer without dimension-generic infrastructure

**P44 LGSSM exists but not integrated:**
- `tests/highdim/test_p44_lgssm_exact_baseline.py:83-113` - `_structural_model(theta, dim)`
- Leaderboard already tests dims 1, 2, 3
- Has exact Kalman oracle via `_lgssm_kalman_value(dim)`
- **Not yet connected to SQMC tuning infrastructure**

---

## Research Phases

### Phase 0: Infrastructure Integration ⏸️ READY

**Purpose:** Make SQMC tuning dimension-generic by integrating P44 LGSSM

**Tasks:**
1. Create dimension-parameterized data generator using P44 LGSSM structure
2. Integrate P44 Kalman oracle into SQMC evaluation
3. Fix hard-coded `dimension=3` in `_evaluate_controls` (already done in `run_sqmc_tuning.py:286`)
4. Document particle count constraint `N % (2*D) == 0`
5. CPU smoke test: 1D, 3D, 10D with N=20, T=2, one seed

**Success criteria:**
- ✓ P44 LGSSM generates D-dimensional data
- ✓ Kalman oracle produces D-dimensional exact score
- ✓ SQMC evaluator accepts `state_dim` parameter
- ✓ Smoke test passes for all three dimensions

**Artifacts:**
- `docs/benchmarks/run_sqmc_generic_lgssm.py` - dimension-generic runner
- `docs/benchmarks/artifacts/sqmc-generic-smoke-20260923/` - smoke results

**Budget:** 1 hour implementation + testing

**Status:** ⏸️ READY

---

### Phase 1: 3D Replication Check ⏸️ PENDING PHASE 0

**Purpose:** Verify P44 LGSSM gives comparable results to frozen canonical on 3D

**Design:**
- Dimensions: 3
- Horizon: T=20
- Particles: N=1008 (satisfies 1008 % 6 = 0)
- Seeds: 4 (50001-50004, same as tuning seeds)
- Controls: Tuned `iid_dual_cap` controls from Phase 2.1 pilot
- Oracle: P44 Kalman vs frozen canonical Kalman

**Metrics:**
- L2 error difference between P44 and frozen canonical (should be < 0.05)
- Cosine similarity difference (should be < 0.0001)
- If P44 results are within noise of frozen canonical → infrastructure validated

**Decision gate:**
- If P44 ≈ frozen canonical → proceed to dimension scaling
- If P44 differs substantially → investigate model mismatch before scaling

**Budget:** 30 minutes

**Status:** ⏸️ PENDING PHASE 0

---

### Phase 2: Dimension Scaling - Same Controls ⏸️ PENDING PHASE 1

**Purpose:** Test whether 3D tuned controls work at 10D without retuning

**Design:**
- **3D baseline** (reference): T=20, N=1008, tuned controls, 4 seeds
- **10D test** (transfer): T=20, N=1000, same controls, same 4 seeds
  - N=1000 chosen: `1000 % (2*10) = 0` ✓ (N=1008 fails: `1008 % 20 = 8`)

**Controls transferred (from 3D `iid_dual_cap` tuning):**
- `reset_epsilon`: 16.0
- `reset_sinkhorn_steps`: 8
- `reset_balance_steps`: 8
- `correction_steps`: 4
- `correction_strength`: 0.2
- `pairwise_steps`: 4
- `pairwise_strength`: 0.03

**Routes tested:** All 4 (iid_dual_cap, previous_inverse_cdf, repaired_permutation, repaired_permutation_ablation)

**Evidence contract:**
- **Promotion criterion:** None (this is a transfer diagnostic)
- **Promotion veto:** Catastrophic failure (NaN, divergence, cosine < 0.99)
- **Continuation veto:** If >50% of cells fail validity checks, stop before T=120
- **Repair trigger:** Dimension mismatch, particle count violation, or graph compilation failure

**Metrics table:**

| Dimension | N | Oracle L2 | SQMC L2 | Cosine | Interpretation |
|-----------|---|-----------|---------|--------|----------------|
| 3D | 1008 | [baseline] | 1.48 | 0.9995+ | Tuned reference |
| 10D | 1000 | ? | ? | ? | Controls transferred |

**Interpretation scenarios:**
1. **10D L2 ≈ 3D L2 (± 10%)** → Controls transfer well, dimension-invariant
2. **10D L2 moderately worse (10-50%)** → Controls suboptimal but usable, retuning may help
3. **10D L2 much worse (>50%) or direction degrades** → Controls do not transfer, must retune per dimension
4. **10D better than 3D** → Unexpected, investigate (may indicate dimension-dependent convergence)

**What this phase does NOT establish:**
- ❌ Optimality of transferred controls at 10D
- ❌ Statistical superiority of any route
- ❌ Production readiness

**Artifacts:**
- `docs/benchmarks/artifacts/sqmc-dimension-transfer-20260923/dim3_baseline/`
- `docs/benchmarks/artifacts/sqmc-dimension-transfer-20260923/dim10_transferred/`

**Budget:** 2 hours (4 routes × 4 seeds × 2 dims × ~7 min/cell)

**Status:** ⏸️ PENDING PHASE 1

---

### Phase 3: Horizon Scaling - Same Controls ⏸️ PENDING PHASE 2

**Purpose:** Test whether T=20 tuned controls work at T=120 without retuning

**Design:**
- **3D T=20** (reference): N=1008, tuned controls, 4 seeds [already have from Phase 2]
- **3D T=120** (transfer): N=1008, same controls, same 4 seeds
- **10D T=120** (double transfer): N=1000, same controls, same 4 seeds

**Evidence contract:**
- **Promotion criterion:** None (transfer diagnostic)
- **Promotion veto:** Catastrophic failure at either horizon
- **Continuation veto:** If T=120 fails validity at both dimensions, stop before cross-analysis
- **Repair trigger:** Numerical instability, memory exhaustion, or timeout

**Metrics table:**

| (D, T) | N | Oracle L2 | SQMC L2 | Cosine | Per-step error |
|--------|---|-----------|---------|--------|----------------|
| (3, 20) | 1008 | 1.48 | 1.48 | 0.9995+ | baseline |
| (3, 120) | 1008 | ? | ? | ? | ? |
| (10, 20) | 1000 | ? | ? | ? | [from Phase 2] |
| (10, 120) | 1000 | ? | ? | ? | ? |

**Interpretation:**
- **Per-step error** = (SQMC L2) / sqrt(T) — tests whether error scales with horizon
- If per-step error is stable across T → controls are horizon-invariant
- If per-step error grows with T → longer horizons may need different controls

**What this phase does NOT establish:**
- ❌ Optimal controls for T=120
- ❌ Whether T=120 needs more particles than T=20
- ❌ Generalization to T > 120

**Artifacts:**
- `docs/benchmarks/artifacts/sqmc-horizon-transfer-20260923/3d_t120/`
- `docs/benchmarks/artifacts/sqmc-horizon-transfer-20260923/10d_t120/`

**Budget:** 3 hours (longer horizon takes more time per cell)

**Status:** ⏸️ PENDING PHASE 2

---

### Phase 4: Cross-Analysis and Decision ⏸️ PENDING PHASE 3

**Purpose:** Synthesize transfer evidence and make retuning recommendations

**Analysis:**
1. **Dimension transfer summary** - plot L2 error vs dimension at T=20
2. **Horizon transfer summary** - plot L2 error vs horizon for 3D and 10D
3. **Joint transfer surface** - (D, T) → L2 error heatmap
4. **Route stability** - do all 4 routes show similar transfer patterns?

**Decision criteria:**

| Transfer quality | L2 degradation | Recommendation |
|------------------|----------------|----------------|
| Excellent | < 10% | Controls transfer, no retuning needed |
| Good | 10-25% | Controls usable as warm-start, retuning optional |
| Moderate | 25-50% | Retuning recommended for claim-bearing work |
| Poor | > 50% or direction fails | Retuning required per scope |

**Deliverables:**
1. Transfer quality table for each (D, T) → recommendation
2. Updated LEDH tuning scope guidance - when to retune vs reuse
3. Particle count selection table for common dimensions
4. Plot: control transfer landscape

**Artifacts:**
- `docs/plans/sqmc-control-transfer-decision-2026-09-23.md`
- `docs/benchmarks/artifacts/sqmc-transfer-analysis-20260923/`

**Budget:** 2 hours analysis + documentation

**Status:** ⏸️ PENDING PHASE 3

---

## Particle Count Constraint

**Mathematical requirement:** The sigma-point design in `_reset_design` requires `N % (2*D) == 0`.

**Recommended particle counts by dimension:**

| State dim | Valid N (examples) | Used in this program |
|-----------|-------------------|---------------------|
| 1D | 20, 100, 500, 1000, 1008 | - |
| 2D | 20, 100, 500, 1000, 1008 | - |
| 3D | 24, 120, 504, 1002, 1008 | **1008** (T=20, T=120) |
| 10D | 20, 100, 500, 1000, 1020 | **1000** (T=20, T=120) |

**Design note:** N=1008 was chosen for 3D because it was used in Austria SIR comparison. N=1000 for 10D is the closest round number satisfying the constraint.

**Scope limitation:** This constraint is specific to the sigma-point reset design. Other SQMC reset mechanisms may have different requirements.

---

## Two-Track Infrastructure: Frozen 3D vs Generic P44

**Frozen 3D canonical model** (existing):
- Purpose: Exact reproducibility, stable baselines
- Model: `diagonal_lgssm_canonical_model`
- Data: `_lgssm_frozen_observations()[:T]`
- Theta: Fixed 5-parameter `[0.9, 0.8, 0.7, 0.6, 0.8]`
- Dimensions: 3/3 state/obs
- Use case: Authoritative 3D reference for all campaigns

**Generic P44 LGSSM** (this program):
- Purpose: Dimension-parameterized testing
- Model: `_structural_model(theta, dim)` from P44 test
- Data: Generated per (dim, T, seed) using P44 structure
- Theta: Compatible 4-parameter `[rho_param, log_q_scale, log_r_scale, initial_mean_scale]`
- Dimensions: Arbitrary (1, 2, 3, 10, ...)
- Use case: Dimension/horizon generalization research

**Relationship:**
- Both use Kalman oracle (exact)
- Both are LGSSM (linear-Gaussian)
- Phase 1 verifies they give comparable 3D results
- Frozen 3D remains authoritative; P44 enables scaling research

**When to use which:**
- Use **frozen 3D** for: reproducible baselines, regression checks, authoritative 3D claims
- Use **P44 generic** for: dimension studies, horizon studies, architecture research
- Never mix within one comparison

---

## Evidence Standards

### Hard Vetoes (Correctness)

These must NEVER be violated:
- Cosine similarity ≥ 0.999 (direction quality, per principled metrics)
- Relative norm error ≤ 5% (magnitude quality)
- All Fisher-scaled errors ≤ 1.0 (component quality)
- No NaN, Inf, or divergence

### Descriptive Evidence (This Program)

All phases use 4 seeds with no uncertainty intervals:
- Can screen validity
- Can nominate transfer patterns
- **Cannot support statistical ranking or superiority claims**

### Transfer Quality Thresholds

Based on measured noise in 3D baseline (seed std dev ~ 0.17 on L2):

| Degradation | Threshold | Interpretation |
|-------------|-----------|----------------|
| Noise level | < 0.2 | Indistinguishable from seed variation |
| Small | 0.2-0.5 | Detectable but minor |
| Moderate | 0.5-1.0 | Clear degradation, retuning may help |
| Large | > 1.0 | Substantial degradation, retuning recommended |

---

## Scope Boundaries

**This program establishes:**
- ✓ Whether 3D T=20 tuned controls work at (10D, T=20)
- ✓ Whether 3D T=20 tuned controls work at (3D, T=120)
- ✓ Whether 3D T=20 tuned controls work at (10D, T=120)
- ✓ P44 LGSSM integration with SQMC tuning
- ✓ Particle count guidance for dimensions 1-10

**This program does NOT establish:**
- ❌ Optimal controls at 10D or T=120 (would require separate tuning)
- ❌ Generalization beyond LGSSM (nonlinear models untested)
- ❌ Statistical route ranking at any (D, T)
- ❌ Production readiness or HMC convergence guarantees
- ❌ Generalization to other particle counts
- ❌ Transfer to float32/TF32 backend (all tests are float64)

**LEDH tuning rule compliance:**
This program tests control transfer, not optimality. Any claim-bearing work at (10D, T=120) still requires exact-scope tuning per the rule. Transfer quality informs whether that tuning can use 3D controls as a warm-start.

---

## Non-Claims

Even after completion, this program does NOT establish:

❌ **Universal control portability** - Evidence limited to LGSSM, 3D→10D, T=20→T=120  
❌ **Optimality at transferred scopes** - Only tested whether controls work, not whether they're best  
❌ **Statistical certainty** - 4 seeds provide descriptive evidence only  
❌ **Production readiness** - Transfer diagnostic, not validation campaign  
❌ **Nonlinear model transfer** - LGSSM only  
❌ **Float32/TF32 transfer** - All tests float64  
❌ **Particle count scaling** - Fixed N per dimension, not optimized  

---

## Configuration Status

All cells labeled: `generic_p44_lgssm_sqmc_float64_transfer_diagnostic`

**Differences from production default:**
- Backend: float64 (matching 3D tuning baseline) vs production float32/TF32
- Model: P44 generic LGSSM vs frozen canonical 3D
- Data: Generated per (D,T,seed) vs frozen canonical observations
- Purpose: Transfer diagnostic vs claim-bearing comparison

**Production mechanisms active:**
- Contract-E reset ✓
- GenUT dual-cap correction ✓  
- Trust-region damping ✓
- UKF covariance lifecycle ✓
- Validity guards ✓
- Reset source-marginal checks ✓

---

## Artifacts

### Pending (This Program)

**Phase 0:**
- `docs/benchmarks/run_sqmc_generic_lgssm.py` - dimension-generic runner
- `docs/benchmarks/artifacts/sqmc-generic-smoke-20260923/` - CPU smoke

**Phase 1:**
- `docs/benchmarks/artifacts/sqmc-p44-3d-replication-20260923/` - P44 vs frozen comparison

**Phase 2:**
- `docs/benchmarks/artifacts/sqmc-dimension-transfer-20260923/dim{3,10}_t20/`

**Phase 3:**
- `docs/benchmarks/artifacts/sqmc-horizon-transfer-20260923/{3d,10d}_t{20,120}/`

**Phase 4:**
- `docs/plans/sqmc-control-transfer-decision-2026-09-23.md`
- Transfer landscape plots

### References (Prior Work)

**3D T=20 tuning baseline:**
- `sqmc-master-program-2026-09-12.md` - completed campaign
- `docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json`

**P44 LGSSM infrastructure:**
- `tests/highdim/test_p44_lgssm_exact_baseline.py:83-113`
- `docs/benchmarks/benchmark_two_lane_lowdim_leaderboard.py`

---

## Execution Commands

### Phase 0: Infrastructure (FIRST)

```bash
cd /home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909
source ~/anaconda3/bin/activate tftwogpu

# Create dimension-generic runner
# (to be implemented based on run_sqmc_tuning.py)
python docs/benchmarks/run_sqmc_generic_lgssm.py \
  --mode smoke \
  --dimensions 1 3 10 \
  --horizon 2 \
  --particles 20 \
  --seeds 50001 \
  --output-dir docs/benchmarks/artifacts/sqmc-generic-smoke-20260923
```

### Phase 1: 3D Replication

```bash
export CUDA_VISIBLE_DEVICES=1  # 4080 SUPER
python docs/benchmarks/run_sqmc_generic_lgssm.py \
  --mode p44-replication \
  --dimension 3 \
  --horizon 20 \
  --particles 1008 \
  --seeds 50001 50002 50003 50004 \
  --controls docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json \
  --output-dir docs/benchmarks/artifacts/sqmc-p44-3d-replication-20260923
```

### Phase 2: Dimension Transfer

```bash
# 3D baseline (reference from Phase 1)
# 10D transfer
python docs/benchmarks/run_sqmc_generic_lgssm.py \
  --mode dimension-transfer \
  --dimension 10 \
  --horizon 20 \
  --particles 1000 \
  --seeds 50001 50002 50003 50004 \
  --routes iid_dual_cap previous_inverse_cdf repaired_permutation repaired_permutation_ablation \
  --controls docs/tuning/sqmc-lgssm-t20-n1008-iid-dual-cap-20260912/tuning_artifact.json \
  --output-dir docs/benchmarks/artifacts/sqmc-dimension-transfer-20260923/dim10_t20
```

### Phase 3: Horizon Transfer

```bash
# 3D T=120
python docs/benchmarks/run_sqmc_generic_lgssm.py \
  --mode horizon-transfer \
  --dimension 3 \
  --horizon 120 \
  --particles 1008 \
  --seeds 50001 50002 50003 50004 \
  --routes iid_dual_cap previous_inverse_cdf repaired_permutation repaired_permutation_ablation \
  --controls docs/tuning/sqmc-lgssm-t20-n1008-*/tuning_artifact.json \
  --output-dir docs/benchmarks/artifacts/sqmc-horizon-transfer-20260923/3d_t120

# 10D T=120
python docs/benchmarks/run_sqmc_generic_lgssm.py \
  --mode horizon-transfer \
  --dimension 10 \
  --horizon 120 \
  --particles 1000 \
  --seeds 50001 50002 50003 50004 \
  --routes iid_dual_cap previous_inverse_cdf repaired_permutation repaired_permutation_ablation \
  --controls docs/tuning/sqmc-lgssm-t20-n1008-*/tuning_artifact.json \
  --output-dir docs/benchmarks/artifacts/sqmc-horizon-transfer-20260923/10d_t120
```

### Phase 4: Analysis

```bash
python docs/benchmarks/analyze_sqmc_control_transfer.py \
  --phase2 docs/benchmarks/artifacts/sqmc-dimension-transfer-20260923/ \
  --phase3 docs/benchmarks/artifacts/sqmc-horizon-transfer-20260923/ \
  --output docs/benchmarks/artifacts/sqmc-transfer-analysis-20260923/
```

---

## Budget

| Phase | Description | Duration | Cumulative |
|-------|-------------|----------|------------|
| 0 | Infrastructure + smoke | 1 hour | 1 hour |
| 1 | 3D P44 replication | 30 min | 1.5 hours |
| 2 | 10D T=20 transfer (4 routes) | 2 hours | 3.5 hours |
| 3 | 3D+10D T=120 transfer | 3 hours | 6.5 hours |
| 4 | Analysis + decision | 2 hours | 8.5 hours |
| **Total** | | | **~9 hours** |

**Calendar time:** 1-2 days with checkpoints for review

**Comparison to prior work:**
- 3D T=20 tuning campaign: ~18 hours (tune + compare + analyze)
- This program: ~9 hours (test transfer only, no retuning)

---

## Decision Gates and Approval

### Required Approvals

**Phase 0 execution:**
- Plain-language: "Execute Phase 0" or "Start generalization program"
- No magic tokens or hash-bound approvals (per Academic Research Governance policy)

**Phase 1-3 continuation:**
- Conditional on Phase 0 success (infrastructure working)
- Can proceed without separate approval (within campaign scope)
- Stop if any phase hits a continuation veto

**Phase 4 (analysis):**
- Automatic after Phase 3 completion
- Delivers transfer quality assessment and retuning recommendations

### Checkpoints for Review

1. After Phase 0: Infrastructure smoke test passed?
2. After Phase 1: P44 matches frozen 3D baseline?
3. After Phase 2: 10D dimension transfer quality?
4. After Phase 3: T=120 horizon transfer quality?
5. After Phase 4: Final recommendations on control portability

---

## Recovery Protocol

### If Session Interrupted

**Current state indicators:**
- Branch: `rqmc-sqmc-4route-comparison`
- Worktree: `/home/chakwong/BayesFilter/.claude/worktrees/kdm-score-campaign-20260909`
- Master program: `docs/plans/sqmc-control-generalization-master-program-2026-09-23.md`

**Check execution status:**
```bash
# Check for completed phases
ls docs/benchmarks/artifacts/sqmc-generic-smoke-20260923/ 2>/dev/null && echo "Phase 0 done"
ls docs/benchmarks/artifacts/sqmc-p44-3d-replication-20260923/ 2>/dev/null && echo "Phase 1 done"
ls docs/benchmarks/artifacts/sqmc-dimension-transfer-20260923/ 2>/dev/null && echo "Phase 2 done"
ls docs/benchmarks/artifacts/sqmc-horizon-transfer-20260923/ 2>/dev/null && echo "Phase 3 done"
ls docs/plans/sqmc-control-transfer-decision-2026-09-23.md 2>/dev/null && echo "Phase 4 done"
```

**Resume from:**
- No artifacts → Start Phase 0
- Smoke only → Start Phase 1
- Phase 1 complete → Start Phase 2
- Phase 2 complete → Start Phase 3
- Phase 3 complete → Start Phase 4

**Context recovery:**
1. Read this master program
2. Read prior 3D baseline: `sqmc-master-program-2026-09-12.md`
3. Check reset memo: `docs/reset-memos/sqmc-4route-campaign-reset-20260922.md`
4. Review P44 infrastructure: `tests/highdim/test_p44_lgssm_exact_baseline.py`

---

## Open Questions and Risks

### Q1: Will P44 LGSSM match frozen 3D results?

**Risk:** Model structure differences cause numerical divergence  
**Mitigation:** Phase 1 explicitly tests this before scaling  
**Contingency:** If mismatch > 5%, investigate and repair model adapter

### Q2: Will 10D need more particles than 1000?

**Risk:** Curse of dimensionality requires N ~ O(D) or O(D²)  
**Mitigation:** N=1000 is 3.3× larger than 3D's N=1008/3 per dimension  
**Contingency:** If 10D fails validity, try N=2000 or N=5000

### Q3: Will T=120 reveal numerical instability?

**Risk:** Error accumulation over 6× more steps  
**Mitigation:** All production safety mechanisms (dual-cap, trust-region) active  
**Contingency:** If T=120 diverges, investigate per-step error growth

### Q4: Do all 4 routes transfer equally?

**Risk:** Some routes dimension/horizon-sensitive  
**Mitigation:** Test all 4 in Phases 2-3  
**Expected:** Routes remain indistinguishable (per Austria SIR precedent)

### Q5: Backend mismatch with production?

**Risk:** float64 results don't transfer to float32/TF32  
**Limitation:** Acknowledged in scope boundaries  
**Future work:** Separate float32/TF32 transfer study if needed

---

## Success Criteria

**Minimum success (Phase 0-1):**
- ✓ P44 LGSSM infrastructure integrated
- ✓ 3D replication within 5% of frozen baseline
- ✓ Dimension-generic runner operational

**Full success (Phases 0-4):**
- ✓ All phases complete within budget
- ✓ Transfer quality quantified for (3D→10D) and (T=20→T=120)
- ✓ Clear recommendations on when to retune vs reuse controls
- ✓ Particle count guidance documented
- ✓ All 4 routes tested at all (D, T) combinations

**Stretch goal:**
- Controls transfer with < 25% degradation → retuning optional
- Per-step error stable across horizons → horizon-invariant controls

---

## Changelog

- 2026-09-23: Program created
- 2026-09-23: Ready for execution after thorough review

---

## References

**Prior campaigns:**
- `sqmc-master-program-2026-09-12.md` - 3D T=20 tuning complete
- `sqmc-oracle-comparison-master-program-v2-2026-09-09.md` - UNTUNED baseline
- `sqmc-4route-comparison-final-report-2026-09-09.md` - Austria SIR comparison

**Infrastructure:**
- `tests/highdim/test_p44_lgssm_exact_baseline.py` - P44 LGSSM reference
- `docs/benchmarks/benchmark_two_lane_lowdim_leaderboard.py` - Leaderboard precedent
- `bayesfilter/testing/lgssm_generic_target_adapter_tf.py` - Generic SSM adapters

**Governance:**
- `CLAUDE.md` - LEDH per-scope tuning rule
- `docs/plans/sqmc-oracle-principled-score-metrics-2026-09-11.md` - Metrics standard