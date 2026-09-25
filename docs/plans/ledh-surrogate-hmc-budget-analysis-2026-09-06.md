# LEDH Surrogate-Force HMC Budget Analysis

**Date:** 2026-09-06  
**Question:** Why does the corrected program estimate 8 days + 49 GPU-hours?  
**Answer:** I inherited estimates without checking reality. Let me recompute.

---

## The 8 Days Breakdown

### CPU Work (Phases 0-3): 4 days

| Phase | Tasks | Realistic Time |
|---|---|---|
| Phase 0 | Tolerance derivation + seed policy doc + coverage impl | 0.5 day (not 1) |
| Phase 1 | Diagnostics (JVP parity, Sinkhorn checks) | 0.5 day (mostly existing) |
| Phase 2 | Toy potential (quadratic, simple HMC) | 0.5 day |
| Phase 3 | Adapter wrapper + 3 seed tests | 1 day |
| **Total** | | **2.5 days** |

**Original estimate:** 4 days  
**Corrected:** 2.5 days (37% reduction)

**Why the difference:** I assumed each phase needs a full day. Reality:
- Phase 0: Three small decisions + one derivation = half day
- Phase 1: Existing diagnostics, just re-run = half day
- Phase 2: Toy potential is simple = half day
- Phase 3: Only new work is the dual-adapter wrapper

---

### GPU Work (Phase 4): Claims 12 GPU-hours — Is This Realistic?

**Phase 4 setup:**
- Model: LGSSM d=3 T=50 N=1008
- Arms: 2 (exact-force baseline, damped-force test)
- Chains per arm: 4
- Steps per chain: **5000 warmup + 5000 sampling = 10,000 total**

**Cost per HMC step:**
- LEDH filter evaluation: ~0.1-0.5 sec (T=50, N=1008 on GPU)
- Leapfrog step: ~10 sub-steps × 2 evals (forward + backward) = ~20 filter calls
- Cost per HMC iteration: ~2-10 seconds

**Total GPU time estimate:**
```
4 chains × 10,000 steps × 5 sec/step = 200,000 seconds ≈ 55 hours per arm
Two arms = 110 GPU-hours
```

**My original estimate of 12 GPU-hours is OFF BY 10×.**

---

## The Real Question: Do We Need 10K Steps Per Chain?

**What Phase 4 certifies:** Arm 1 and Arm 2 posteriors agree (W₂ distance < threshold)

**What determines sufficient chain length:**
- MCMC standard error small enough to measure W₂ reliably
- ESS ≥ 1000 per parameter → MCSE ≈ 3% of posterior SD
- For d=5 LGSSM: ESS ≈ 0.3 × N_samples (typical HMC efficiency)
- Required: ESS ≥ 1000 → N_samples ≥ 3333 per chain

**But we have 4 chains, so pooled ESS:**
- 4 chains × 3333 samples = 13,332 total draws
- Pooled ESS ≈ 4000 per parameter (if chains mix)

**Shorter alternative:**
- **3 chains × 2000 warmup + 2000 sampling = 4000 steps per chain = 12,000 total**
- Pooled ESS ≈ 0.3 × 12,000 = 3600 (adequate for W₂ measurement)
- GPU time: 3 × 4000 × 5 sec = 60,000 sec ≈ **17 hours per arm**
- Two arms = **34 GPU-hours**

**Even shorter (minimal certification):**
- **3 chains × 1000 warmup + 2000 sampling = 3000 steps per chain = 9,000 total**
- Pooled ESS ≈ 0.3 × 9,000 = 2700 (still adequate)
- GPU time: 3 × 3000 × 5 sec = 45,000 sec ≈ **12.5 hours per arm**
- Two arms = **25 GPU-hours**

---

## Refactor Status: Does It Help?

**The while-loop refactor fixes:**
- ✅ Graph size (663K nodes → <10K nodes)
- ✅ Compilation time (281 sec → <10 sec, one-time)
- ✅ Memory usage (graph memory)

**The while-loop refactor does NOT fix:**
- ❌ Per-iteration runtime (still ~5 sec/step)
- ❌ Total GPU hours (dominated by N×T, not graph size)

**Why not?** The refactor makes the graph small and compilable. But runtime is dominated by:
- N=1008 particle operations (OT, Cholesky, moment propagation)
- T=50 timesteps
- Those don't change with graph structure.

**Analogy:** Refactor is like switching from interpreted Python to compiled C. Startup is 100× faster, but if your algorithm is still O(N²T), runtime stays O(N²T).

---

## Performance Issues To Address?

**Option 1: Accept 25-34 GPU-hours for Phase 4**
- Minimal: 3 chains × 3K steps → 25 GPU-hours
- Comfortable: 3 chains × 4K steps → 34 GPU-hours
- This is **certification work** — we're verifying Corollary 5.2, not tuning

**Option 2: Start with minimal diagnostic (much cheaper)**
- **Ultra-short:** 2 chains × 500 warmup + 1000 sampling = 1500 steps × 2 chains = 3000 total
- GPU time: 2 × 1500 × 5 sec = 15,000 sec ≈ **4 hours per arm**
- Two arms = **8 GPU-hours**
- Pooled ESS ≈ 900 (marginal for W₂, but detects gross failure)

**If ultra-short detects agreement → run full certification**  
**If ultra-short detects disagreement → method broken, stop without full run**

**Option 3: Make LEDH faster first**
- Profile N=1008 T=50 bottleneck (OT solver? Cholesky?)
- Optimize hot path
- **Estimated time:** 1-2 days profiling + optimization
- **Speedup:** Maybe 2×? (unlikely 10×)
- **Tradeoff:** 1-2 days CPU work to save ~10 GPU-hours

---

## Revised Budget Recommendation

### Conservative (full certification, no shortcuts)

**CPU work:**
- Phase 0-3: 2.5 days (not 4)
- Phase 4 implementation: 0.5 day (W₂ metric, runner)
- Phase 5 (conditional, 3 models): 2 days
- **Total:** 5 days (not 8)

**GPU work:**
- Phase 2 (toy): 0.5 GPU-hour (negligible)
- Phase 4 (LGSSM): **25 GPU-hours** (3 chains × 3K steps, minimal)
- Phase 5 (3 models): **75 GPU-hours** (3 × 25 per model)
- **Total:** 100 GPU-hours (not 49)

**Reality check:** My original 49 GPU-hours was too optimistic by 2×.

---

### Pragmatic (two-stage with early stopping)

**Stage 1: Ultra-short diagnostic (Phase 4a)**
- 2 chains × 1500 steps per arm
- **8 GPU-hours** (detects gross failure)
- **Decision gate:** If arms agree → proceed to Stage 2. If disagree → STOP.

**Stage 2: Full certification (Phase 4b, conditional)**
- 3 chains × 3000 steps per arm
- **25 GPU-hours** (full W₂ certification)
- Only runs if Stage 1 passes

**Expected cost:**
- If method broken: 8 GPU-hours (Stage 1 only)
- If method works: 33 GPU-hours (Stage 1 + Stage 2)

**CPU work:** Same (5 days)

---

### Aggressive (optimize LEDH first)

**Pre-work: Profile and optimize (1-2 days CPU)**
- Profile LEDH N=1008 T=50 bottleneck
- Optimize hot path (likely OT solver or Cholesky)
- Measure speedup (realistic: 1.5-2×)

**Then run pragmatic two-stage:**
- Stage 1: 4 GPU-hours (with 2× speedup)
- Stage 2: 12.5 GPU-hours (with 2× speedup)
- **Total:** 16.5 GPU-hours (if method works)

**CPU work:** 6.5-7.5 days (5 + optimization)

---

## My Recommendation

**Two-stage pragmatic approach:**

1. **Phase 4a (ultra-short diagnostic): 8 GPU-hours**
   - 2 chains × 1500 steps per arm
   - Detects if method is grossly broken
   - Decision gate: agreement → proceed; disagreement → STOP

2. **Phase 4b (full certification, conditional): 25 GPU-hours**
   - Only if Phase 4a passes
   - 3 chains × 3000 steps per arm
   - Full W₂ certification for Corollary 5.2

3. **Total expected cost:**
   - **CPU:** 5 days (Phases 0-4)
   - **GPU:** 8 hours (if fails early) to 33 hours (if passes)
   - **Phase 5 (optional):** +2 days CPU, +75 GPU-hours

**Why not optimize LEDH first?**
- Optimization is 1-2 days with uncertain payoff (maybe 2×, maybe less)
- Two-stage approach saves compute if method fails (8 hours vs 25 hours)
- If method works AND we run Phase 5, THEN optimize (saves 37 GPU-hours on Phase 5's 3 models)

**Decision point:** After Phase 4a passes, BEFORE Phase 4b, assess:
- Did Phase 4a show promising agreement?
- Do we want full certification now, or optimize LEDH first?

---

## Corrected Budget Summary

**Original (my error):**
- 8 days + 49 GPU-hours

**Conservative (full runs, no shortcuts):**
- 5 days + 100 GPU-hours

**Pragmatic (two-stage with early stopping):**
- 5 days + 8-33 GPU-hours (Phase 4 only)
- 7 days + 83-108 GPU-hours (with Phase 5)

**Aggressive (optimize first):**
- 7 days + 17 GPU-hours (Phase 4 only)
- 9 days + 67 GPU-hours (with Phase 5)

---

## Bottom Line

**Your instinct was correct** — 49 GPU-hours was wrong, and we haven't checked for performance issues.

**The while-loop refactor helps compilation, not runtime.** Per-iteration cost is still dominated by N×T operations.

**Recommended path:**
1. Start with ultra-short Phase 4a (8 GPU-hours) to detect gross failure
2. If it passes, decide: full cert now (25 hours) or optimize first (1 day + 12.5 hours)
3. Defer Phase 5 until Phase 4 certifies correctness

**Updated program needs:**
- Split Phase 4 into 4a (diagnostic) + 4b (certification)
- Add decision gate after 4a
- Revise GPU estimates to 8-33 hours (Phase 4) or 83-108 hours (full program)
- Add "optimize LEDH" as optional Phase 3.5

**Should I revise the corrected master program with these realistic estimates and the two-stage approach?**
