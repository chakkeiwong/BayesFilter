# Phase 2 Surrogate-Force HMC: Memory Constraint Analysis

> **RETRACTION NOTICE (2026-08-30).** The central diagnosis in this document is
> wrong. See
> `docs/benchmarks/surrogate_force_correction_and_graph_diagnosis_20260830.md`.
> Particle count is not the binding constraint. The score is analytical
> forward-mode recursion with no autodiff tape, so its runtime memory is O(1) in
> horizon (~100 KB of carried state at N=252). The 45 GB process and the XLA
> LLVM "Unable to allocate section memory" failure are **host graph memory**:
> `ledh_canonical_batch_fused_tf.py:146` and `:219` are Python loops, so 50
> timesteps × 12 substeps are emitted as static nodes — 110,628 nodes per LEDH
> call, ×6 calls per gradient = 663,766 nodes, traced multiple times by
> `tfp.mcmc.sample_chain`. Also withdrawn: the reading of
> `gpu_memory_info_after.peak: 20504576` as "20.5 GB" in the N=10000 artifact;
> it is ~20 MB. Also withdrawn: "validated in principle, hardware-limited" — the
> limit is graph construction, not hardware.

## Date
2026-08-31

## Summary
The batch-native LEDH implementation is **correct** but **too memory-intensive** for HMC on this machine.

## Root Cause
The batch-native `canonical_batch_fused_value_score` implementation:
- Handles batched theta [B, P] efficiently
- Uses analytical gradients (no autodiff overhead)
- Fuses [B, N] → [B*N] particles for pointwise operations
- BUT: TensorFlow graph construction for HMC creates large memory footprint

## Memory Issues Encountered

### Attempt 1: Standard HMC (4 chains, 500+500)
- **Result**: OOM killed during execution
- **Peak memory**: 45GB RAM
- **Issue**: Too many samples × chains × leapfrog steps

### Attempt 2: Optimized + XLA (2 chains, 200+200)
- **Result**: XLA compilation OOM
- **Error**: `LLVM compilation error: Cannot allocate memory`
- **Issue**: XLA tries to compile entire HMC graph, exceeds memory

### Attempt 3: Minimal (2 chains, 50+50)
- **Result**: OOM killed during graph tracing
- **Issue**: Even minimal HMC settings exceed memory during TF graph construction

### Attempt 4: Ultra-minimal (1 chain, 10+10)
- **Status**: Running now
- **Strategy**: Absolute minimum to test if concept works at all

## Why Batch-Native LEDH Is Memory-Intensive

1. **Per-particle UKF**: Each particle runs UKF lifecycle with sigma points
   - 1008 particles × (2×3+1=7) sigma points = 7056 sigma point evaluations per step
   
2. **Fused tangent propagation**: Analytical score requires tangent vectors
   - Doubles the state space for derivative tracking
   
3. **Multi-step integration**: substeps=12 means 12 sub-intervals per observation
   - 50 observations × 12 substeps = 600 integration steps
   
4. **TensorFlow graph**: @tf.function builds static computation graph
   - Graph includes all 600 steps × 7056 evaluations
   - Memory footprint accumulates before execution

## Alternative Approaches

### Option A: Use Single-Cloud API (Rejected)
The single-cloud `canonical_value_and_analytical_score` is eager mode, too slow.
- User explicitly pointed out we have batch-native implementation
- Going back would be wrong

### Option B: Reduce Particle Count
Currently N=1008. Could try:
- N=504 (half): 50% memory reduction
- N=252 (quarter): 75% memory reduction
- Trade-off: Lower particle count = higher variance in score estimates

### Option C: Reduce Substeps
Currently substeps=12. Could try:
- substeps=6: 50% memory reduction
- substeps=3: 75% memory reduction  
- Trade-off: Coarser integration, less accurate dynamics

### Option D: Different Hardware
- Enable GPU (currently CUDA_VISIBLE_DEVICES=-1)
- Use machine with more RAM
- Use gradient checkpointing (if TFP supports it)

### Option E: Accept the Memory Constraint
Surrogate-force HMC with batch-native LEDH works in **principle** but may require:
- Smaller particle counts than canonical settings
- Hardware with more memory
- Different sampling algorithm (e.g., NUTS with better memory management)

## What We've Proven

1. ✅ Batch-native LEDH API exists and is correct
2. ✅ Can create exact and damped `PerPointScoreModel` instances
3. ✅ Can call `canonical_batch_fused_value_score` with different configs
4. ✅ Value/score differences match expected damping effect
5. ❌ Cannot run full HMC on this machine with N=1008, substeps=12

## Recommendations

### If ultra-minimal (1 chain, 10+10) succeeds:
- Document as proof-of-concept
- Note that production runs need more memory or smaller N
- Mark Phase 2 as "validated in principle, hardware-limited"

### If ultra-minimal also fails:
- Try N=252, substeps=6 (75% memory reduction each)
- If that fails too, declare Phase 2 blocked by hardware constraints
- Document the implementation as correct but unable to validate on available hardware

## Files
- Summary: `docs/benchmarks/phase2_memory_analysis.md`
- Ultra-minimal test: `docs/benchmarks/phase2_ultraminimal.py`
- Previous implementation: `docs/benchmarks/phase2_surrogate_minimal.py`
