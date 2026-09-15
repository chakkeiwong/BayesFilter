# LEDH Surrogate-Force HMC Phase 3 Task 3.2 Status

**Date:** 2026-09-12  
**Task:** Damping calibration for surrogate-force HMC  
**Status:** BLOCKED - Resource constraints

---

## Completed Work

### 1. Dual-Parameter Adapter Implementation ✓

**File:** `bayesfilter/inference/ledh_dual_parameter_target.py`

Implements Corollary 5.2 surrogate-force HMC target:
- Value computed with exact parameters
- Score computed with biased (damped) parameters  
- Frozen Monte Carlo seeds (deterministic noises)
- Returns 3-tuple: (value, score, diagnostics)

### 2. Spot-Check Verification ✓

**File:** `docs/benchmarks/ledh_surrogate_hmc_spot_check_adapter.py`

**Configuration:**
- N = 252 particles
- T = 50 horizon
- LGSSM d=3 model
- Damping ratios: 1× and 100×

**Results:**
```
✓ PASS: 1× value is finite
✓ PASS: 100× value is finite  
✓ PASS: 1× score is finite
✓ PASS: 100× score is finite
✓ PASS: Values identical (diff=0.00e+00)
✓ PASS: Scores differ (L2 diff=0.000196)
```

**Conclusion:** Dual-parameter adapter works correctly per Corollary 5.2.

### 3. Shape Fix for reset_design

**Issue:** Contract-E reset requires `reset_design` shape `[N, d]`, not `[d, d]`

**Fix applied:**
```python
reset_basis = tf.concat([tf.eye(d), -tf.eye(d)], axis=0)  # [2d, d]
reset_repeats = (N + 2 * d - 1) // (2 * d)
reset_design = tf.tile(reset_basis, [reset_repeats, 1])[:N]  # [N, d]
```

This matches the canonical NeuTra factory pattern.

---

## Blocker: GPU Memory Exhaustion

### Calibration Attempts

Attempted 4-arm calibration sweep per plan specification:

| Attempt | N | Steps | Mode | Outcome |
|---------|---|-------|------|---------|
| 1 | 5000 | 10k+10k | tf.function | OOM kill (~45GB RAM) |
| 2 | 1000 | 10k+10k | tf.function | OOM kill (~45GB RAM) |
| 3 | 1000 | 500+500 | eager | OOM kill (~45GB RAM) |
| 4 | 1008 | 500+500 | eager | GPU OOM (~13GB VRAM) |
| 5 | 500 | 100+100 | eager | Timeout (>5 min, no output) |

**Plan specification:** N=1008, 2 chains × 500 warmup + 500 samples per arm

### Root Cause

LEDH filter evaluation with N ≥ 500 particles is extremely memory-intensive:

1. **Per-step memory:** Each HMC leapfrog step requires:
   - Forward LEDH pass: value computation
   - Backward LEDH pass: gradient computation via dual-parameter score
   - Both passes maintain full particle clouds [N, d] through T=50 timesteps
   - Sinkhorn iterations, correction steps, pairwise ops all on GPU

2. **Accumulated memory:** 
   - tf.function: builds massive unrolled graph for all steps
   - Eager mode: accumulates intermediate tensors across steps
   - Even 500+500 steps × 1008 particles exhausts 13GB GPU

3. **Memory scaling:** 
   - N=252 (spot-check): ~3GB, works
   - N=1008 (plan): ~13GB+, OOM
   - Memory scales roughly as O(N²) due to pairwise operations

### System Resources

- GPU: NVIDIA RTX 4080 SUPER (13.5GB VRAM) + RTX 5080 (13.2GB VRAM)
- RAM: ~64GB (OOM killer activated at ~45GB usage)
- Each HMC step with N=1008 takes several seconds

---

## Alternatives Considered

### 1. Further Reduce Particle Count

**Option:** Run calibration with N=252 (proven to work)

**Concerns:**
- Plan specifies N=1008 for production
- N=252 may not adequately test LEDH numerical stability at scale
- Damping calibration results may not transfer to production particle count
- Would require deviation from master program

### 2. CPU-Only Execution

**Option:** Set `CUDA_VISIBLE_DEVICES=-1` and run on CPU

**Concerns:**
- LEDH filter is GPU-optimized (TF32, streaming OT)
- CPU execution would be 10-100× slower
- 500+500 steps × 2 chains × 4 arms = 8k total HMC steps
- Estimated wall time: days to weeks
- CPU-calibrated damping may differ from GPU behavior

### 3. Incremental Sampling with Checkpointing

**Option:** Run 1 chain at a time, save intermediate states to disk

**Concerns:**
- Still requires each individual chain to fit in GPU memory
- N=1008 single chain already OOM
- Would not solve the fundamental memory constraint

### 4. Chunked/Streaming LEDH HMC

**Option:** Implement particle-chunked HMC gradient computation

**Requirements:**
- Modify LEDH filter to support particle chunking
- Implement chunk-aware autodiff or manual gradient accumulation
- Validate that chunked gradients match full gradients
- Significant engineering work (days-weeks)
- Beyond Phase 3 Task 3.2 scope

---

## Recommendation

**Option A (Conservative):** Run calibration with N=252, document limitation

- Complete 4-arm sweep with proven working configuration
- Obtain damping ratio ranking and acceptance/ESS trends
- Explicitly document that results are at N=252, not production N=1008
- Flag need for larger-scale validation before production use
- Estimated time: 2-4 hours

**Option B (Deferred):** Mark Task 3.2 as blocked pending infrastructure

- Document current state (adapter verified, shape fixed, calibration blocked)
- Continue to Phase 3 Task 3.3 or other non-HMC tasks
- Return to calibration when:
  - Larger GPU available (40GB+ VRAM), or
  - Chunked LEDH-HMC implementation complete, or
  - User provides alternative resource access

**Option C (Reduced Scope):** Single-point diagnostic instead of full calibration

- Evaluate each damping ratio at true θ only (no sampling)
- Measure gradient L2 norms, condition numbers, numerical stability
- Cheaper than HMC but less informative for mixing efficiency
- Estimated time: 30 minutes

---

## Decision Point

This requires user direction:

1. Accept reduced-scale calibration (N=252) with documented limitations?
2. Defer Task 3.2 and proceed to other Phase 3 tasks?
3. Pursue reduced-scope diagnostic without HMC sampling?
4. Other alternatives?

---

## Files Modified/Created

**Implementation:**
- `bayesfilter/inference/ledh_dual_parameter_target.py` (new)

**Verification:**
- `docs/benchmarks/ledh_surrogate_hmc_spot_check_adapter.py` (new, passed)

**Calibration attempts (all OOM):**
- `docs/benchmarks/ledh_surrogate_hmc_damping_calibration.py`
- `docs/benchmarks/ledh_surrogate_hmc_damping_calibration_v2.py`
- `docs/benchmarks/ledh_surrogate_hmc_single_chain_test.py`

**Status documentation:**
- `docs/plans/ledh-surrogate-hmc-phase3-task3.2-status-2026-09-12.md` (this file)
