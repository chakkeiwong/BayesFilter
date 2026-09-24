# BayesFilter High-Dimensional Code: Performance and XLA Issues

**Audit Date:** 2026-09-03  
**Scope:** bayesfilter/highdim/* implementations  
**Context:** Post-LEDH while-loop refactor  
**Method:** Static code analysis (grep, pattern matching)

## Overview

This is a preliminary audit of performance bottlenecks and XLA compilation issues in the high-dimensional filtering codebase, based on code search and pattern analysis. A comprehensive performance audit would require profiling, but several patterns warrant immediate attention.

## Critical Issues Found

### 1. Serial tf.map_fn in Contract E Streaming JVP ⚠️ HIGH SEVERITY

**Location:** `bayesfilter/highdim/ledh_contract_e_streaming_tf.py:1255-1262, 1341-1348`

**Pattern:**
```python
reset_particles = tf.transpose(
    tf.map_fn(
        one_direction,
        tf.range(parameter_count),
        fn_output_signature=output_signature,
        parallel_iterations=1,  # SERIAL EXECUTION
    ),
    [1, 2, 3, 0],
)
```

**Issue:** `parallel_iterations=1` forces serial execution of the parameter-direction loop. For P parameters, this computes P separate Contract E resets sequentially.

**Impact:**
- No parallelization across gradient directions
- P×(single reset cost) instead of amortized parallel cost
- Blocks GPU utilization (sequential kernel launches)
- Cannot leverage multi-core or tensor cores
- **This is in the JVP hot path for NeuTra training**

**Why it exists:** Likely implemented this way to avoid memory explosion or because the original working version was serial.

**Fix strategy:**
1. Convert to `tf.vectorized_map` (same pattern as LEDH refactor)
2. Ensure the reset computation is truly independent across directions
3. Verify memory usage doesn't explode with K concurrent evaluations
4. Add parity test: P=5 serial vs K=5 vectorized
5. Measure wall-clock speedup

**Expected improvement:** 5-10× speedup for P=5 parameters (if memory permits)

**Risk:** Memory usage may increase by factor of P. Need measurement.

### 2. XLA QR Accuracy Issue ⚠️ MEDIUM-HIGH SEVERITY

**Location:** `bayesfilter/highdim/squared_tt_engine_xla_tf.py:63-68`

**Issue:**
```python
# XLA's blocked-Householder QR on CPU is both slow AND inaccurate for
# wide fits (measured P3.3: 2.4e-4 value error at 208 columns);
# CholeskyQR2 gives near-Householder solution accuracy for scaled
# systems...
```

**Context:** The code works around XLA's QR by using CholeskyQR2 (Gram matrix → Cholesky → Q recovery).

**Impact:**
- Cannot use TensorFlow's native `tf.linalg.qr` under XLA
- Custom implementation adds complexity and maintenance burden
- CholeskyQR2 has different numerical properties (condition-number dependent)
- May limit GPU compilation benefits for this kernel
- 2.4e-4 accuracy loss is significant for some numerical algorithms

**Root cause:** XLA's blocked-Householder QR implementation issue on CPU (slow + inaccurate for wide matrices)

**Workarounds in place:**
- CholeskyQR2 with two-pass refinement
- Condition number monitoring via eigenvalue ratio
- Fail-closed on non-finite outputs
- Measured accuracy recovery to near-Householder levels

**Open questions:**
1. Does this affect GPU? Comment mentions "CPU" but doesn't clarify GPU behavior
2. Is this a known TensorFlow/XLA bug? Is it being tracked?
3. What's the upstream timeline for a fix?
4. Should we file a bug if one doesn't exist?

**Next steps:**
1. Test GPU behavior explicitly (may not have the same issue)
2. Document XLA version where issue was observed
3. Create minimal reproducer for upstream bug report
4. Measure CholeskyQR2 vs native QR on GPU

### 3. O(N²) Memory OOM in RQMC Initial Score ⚠️ HIGH SEVERITY

**Location:** `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py:258-270`

**Issue:**
```python
# Sequential XLA While region: one child block's O(block x N)
# intermediates live at a time. An unrolled Python loop here lets XLA
# keep all blocks co-live, which is O(N^2) peak memory and OOMs at
# large N (see the 2026-08-19 result note).
```

**Pattern:** Uses `tf.while_loop` with `TensorArray` to process blocks sequentially, avoiding O(N²) peak memory.

**Context:** Computing inherited parameter scores across N particles requires summing contributions from parent particles. Naïve implementation materializes all N×N pairwise scores simultaneously.

**Impact:**
- Cannot scale to N=10000+ particles without this mitigation
- Memory bound is the limiting constraint, not compute
- Sequential processing is 10-100× slower but avoids OOM
- Blocks use of large particle counts that would otherwise be feasible

**Status:** MITIGATED by while-loop chunking, but still a fundamental bottleneck

**Mathematical question:** Can this be reformulated to avoid the O(N²) materialization entirely?

**Potential approaches:**
1. **Sparse computation:** If parent-child relationships are sparse, only compute non-zero terms
2. **Streaming aggregation:** Accumulate scores incrementally without materializing full matrix
3. **Low-rank approximation:** If score matrix has low-rank structure, exploit it
4. **Hierarchical methods:** Tree-structured aggregation (O(N log N) memory)

**Next steps:**
1. Characterize sparsity structure of parent-child scores
2. Measure actual memory vs N scaling (confirm O(N²))
3. Profile to separate memory-bound vs compute-bound regimes
4. Investigate mathematical reformulations

### 4. Static Shape Requirements Everywhere ⚠️ MEDIUM SEVERITY

**Locations:** 17+ explicit checks found across multiple files

**Pattern:**
```python
if particle_count is None:
    raise ValueError("XLA core requires static particle count")
```

**Files with static shape requirements:**
- `ledh_pfpf_genut_initial_rqmc_tf.py:365`
- `cubature_genut_filter.py:593`
- `cubature_genut_batch_tf.py:1575, 1798`
- `genut_guided_proposal_tf.py:1169, 1186`
- `zhao_cui_moment_teacher.py:1301`
- `ledh_canonical_batch_fused_tf.py` (from recent refactor)
- And many more

**Impact:**
- Every new (B, N, T, dim, K) configuration retraces the graph
- Trace time can be 10s-100s for complex graphs
- Cannot batch different problem sizes together
- Limits flexibility in production settings
- Cold-start latency for new configurations

**Why this exists:**
- Many algorithms have dimension-dependent operations (Cholesky, UKF sigma points)
- XLA optimization benefits from static shapes
- Dynamic shapes add runtime overhead and complexity

**Mitigation strategies not yet used:**
1. Shape polymorphism via `tf.TensorSpec([None, None, ...])`
2. Padding to fixed bucket sizes (e.g., round up to nearest power of 2)
3. Rank polymorphism
4. Shape caching strategies

**Trade-off:** Static shapes enable better XLA optimization at the cost of retracing overhead.

**Key question:** What is the retracing frequency in actual use?
- If shapes change every iteration: polymorphism helps
- If shapes are fixed per campaign: static is fine
- If shapes cluster into a few common sizes: bucketing helps

**Next steps:**
1. Instrument retracing frequency in production/benchmark workloads
2. Measure trace time vs eval time trade-off
3. Evaluate shape polymorphism for high-value kernels
4. Create shape-bucketing strategy if clustering is observed

### 5. Blocked P73_B Optimizer ⚠️ MEDIUM SEVERITY

**Location:** `bayesfilter/highdim/source_route.py:307, 5033, 5369, 5385, 5395, 5406, 5426, 5438`

**Issue:** Multiple routes report `P73_B_OPTIMIZER_BLOCKED` with reason:
```python
P73_B_OPTIMIZER_BLOCKED = "P73_B_OPTIMIZER_BLOCKED_NONLINEAR_OBJECTIVE_NOT_IMPLEMENTED"
```

**Context:** P73_B appears to be a Lane B optimizer that's blocked because nonlinear objectives aren't implemented.

**Impact:**
- Cannot use optimizer-based tuning for these routes
- Limits automatic hyperparameter selection
- May force manual tuning or grid search
- Blocks systematic exploration of parameter space

**Status:** BLOCKED - requires implementation work

**Open questions:**
1. What is the P73_B optimizer? (Not defined in search results)
2. Why is nonlinear objective support missing?
3. What routes depend on this optimizer?
4. Is this blocking any active research directions?
5. How much work is required to implement?

**Next steps:**
1. Read P73 phase documentation to understand optimizer design
2. Identify which routes are blocked and their priority
3. Scope implementation effort for nonlinear objectives
4. Owner decision on priority vs workarounds

### 6. Underspecified tf.map_fn Calls ⚠️ LOW SEVERITY

**Location:** `bayesfilter/highdim/sir_online_score_teacher_tf.py:287, 303`

**Pattern:**
```python
return tf.map_fn(
    lambda x: ...,
    elems,
)
```

**Issue:** Using `tf.map_fn` without `parallel_iterations` or `fn_output_signature` specification.

**Impact:**
- Default parallel_iterations may cause memory issues or underutilization
- Type inference may fail or cause retracing
- Less predictable behavior under XLA
- Harder to debug performance issues

**Recommendation:** Add explicit parameters:
```python
return tf.map_fn(
    lambda x: ...,
    elems,
    parallel_iterations=10,  # or appropriate value
    fn_output_signature=tf.TensorSpec(...),
)
```

**Next steps:** Audit all `tf.map_fn` calls and add specifications

### 7. Limited Use of reduce_retracing ⚠️ LOW SEVERITY

**Observation:** Only ~20 functions use `reduce_retracing=True` despite hundreds of tf.function decorators.

**Pattern found:**
```python
@tf.function(
    input_signature=[tf.TensorSpec([param_dim], DTYPE)],
    reduce_retracing=True,  # Good!
)
```

**Issue:** Most `@tf.function` decorators don't use `reduce_retracing=True`, potentially causing unnecessary retracing on minor input variations.

**Impact:**
- More retracing than necessary
- Slower first calls
- Larger memory footprint (multiple cached graphs)
- Unpredictable cold-start behavior

**Recommendation:** Audit all `@tf.function` calls and add `reduce_retracing=True` where safe.

**Caution:** `reduce_retracing=True` is not always safe. It uses heuristics that may break assumptions about input shapes or values. Use with `input_signature` for safety.

**Next steps:**
1. Create linting rule to check for `reduce_retracing`
2. Audit high-traffic functions first
3. Add to style guide

## Performance Patterns Observed

### Pattern: Serial Direction Loops ⚠️ HIGH SEVERITY

Multiple implementations iterate over parameter directions serially instead of vectorizing. The LEDH refactor fixed this for one kernel; similar patterns likely exist elsewhere.

**Instances found:**
- Contract E streaming JVP (documented above)
- Potentially in other JVP/VJP implementations

**Search needed:** Full audit of all JVP/VJP implementations for serial parameter loops

**Pattern to search:**
```python
for k in range(parameter_count):
    # compute direction k
```

Or:
```python
tf.map_fn(..., parallel_iterations=1)
```

**Fix template:** Convert to `tf.vectorized_map` as done in LEDH refactor

### Pattern: CPU-Only XLA Issues ⚠️ MEDIUM SEVERITY

Several comments mention "XLA on CPU" issues but don't clarify GPU behavior. The repo claims GPU-first execution, so CPU-specific issues might be less critical—but they need documentation.

**Questions:**
- Are these issues GPU-specific, CPU-specific, or both?
- Are tests running on CPU when they should run on GPU?
- What's the actual production target? (GPU per CLAUDE.md)
- Should CPU code paths be deprecated?

**Next steps:**
1. Add "Device: CPU" / "Device: GPU" to all performance-related comments
2. Test all "CPU-only" issues on GPU explicitly
3. Document CPU vs GPU behavior divergence
4. Consider deprecating CPU-specific workarounds if GPU doesn't need them

### Pattern: Memory-Bounded Algorithms ⚠️ MEDIUM-HIGH SEVERITY

Multiple algorithms hit O(N²) or O(N·T) memory walls before compute walls. This suggests:
- Memory optimization is more important than FLOP optimization
- Streaming/chunking strategies are critical
- Device memory (not host memory) is the actual constraint

**Examples:**
- RQMC initial score: O(N²) → sequential blocking
- Transport chunk policy: cap at 3000×3000 for memory
- Contract E streaming: explicit row/column chunking

**Design principle:** Bounded-memory algorithms should be the default, not the fallback.

**Recommendation:** Establish memory-budgeting as a first-class design constraint:
1. Document peak memory scaling for each algorithm
2. Add O(N²) guard rails at implementation time
3. Prefer streaming designs from the start
4. Test with N=10000 as a smoke test

## Missing Diagnostics

Based on the search results, several performance diagnostics appear to be missing:

1. **Graph size measurement:** No systematic graph-size monitoring after XLA optimization
2. **Peak memory tracking:** Limited GPU memory measurement infrastructure
3. **Compilation time tracking:** No systematic trace-time measurement
4. **Retracing detection:** No runtime warnings when graphs retrace
5. **XLA fallback detection:** No detection of ops that fail to compile
6. **Direction-cost profiling:** No measurement of K-direction overhead in vectorized_map
7. **Parallelization efficiency:** No measurement of parallel_iterations utilization

**Recommendation:** Add instrumentation layer for performance monitoring:
```python
@monitored_tf_function(
    name="canonical_batch_fused",
    track_trace_time=True,
    track_peak_memory=True,
    warn_on_retrace=True,
)
@tf.function(...)
def canonical_batch_fused_value_score(...):
    ...
```

## Recommended Actions

### Immediate (High ROI)

1. **Vectorize Contract E streaming JVP** [HIGH PRIORITY]
   - Convert `parallel_iterations=1` to `tf.vectorized_map`
   - Add parity test vs serial version
   - Measure speedup and memory increase
   - Timeline: 1-2 days

2. **Document XLA QR issue** [HIGH PRIORITY]
   - Test GPU behavior explicitly
   - Document TensorFlow/XLA version
   - File upstream bug if needed
   - Timeline: 1 day

3. **Add memory profiling to score artifacts** [HIGH PRIORITY]
   - Instrument GPU memory for all claim-bearing runs
   - Add peak memory to score contract
   - Timeline: 1-2 days

4. **Audit all tf.map_fn calls** [MEDIUM PRIORITY]
   - Add explicit `parallel_iterations` and `fn_output_signature`
   - Create linting rule
   - Timeline: 1 day

### Medium-term (Infrastructure)

1. **Add retracing detection** [MEDIUM PRIORITY]
   - Warn when graphs retrace unexpectedly
   - Log retracing frequency in production
   - Timeline: 2-3 days

2. **Create shape polymorphism strategy** [MEDIUM PRIORITY]
   - Measure retracing frequency
   - Evaluate trade-offs for high-impact kernels
   - Implement for top-3 bottlenecks
   - Timeline: 1 week

3. **Implement P73_B nonlinear optimizer** [DEPENDS ON OWNER]
   - Unblock optimizer-based tuning
   - Scope effort first
   - Timeline: Unknown

4. **Add XLA compilation gates** [LOW PRIORITY]
   - Fail explicitly when XLA is expected but falls back
   - Add XLA verification to test suite
   - Timeline: 2 days

### Long-term (Research)

1. **Investigate O(N²) memory alternatives** [RESEARCH]
   - Can RQMC score avoid full materialization?
   - Sparse computation, streaming, low-rank, hierarchical methods
   - Timeline: 1-2 weeks

2. **Evaluate streaming architectures** [RESEARCH]
   - Can more kernels use bounded-memory streaming?
   - Generalize chunking patterns
   - Timeline: 2-3 weeks

3. **Profile direction-cost scaling** [MEASUREMENT]
   - Measure actual K-direction overhead in vectorized_map
   - Establish K-scaling baselines
   - Timeline: 1 week

4. **XLA compilation study** [MEASUREMENT]
   - Measure compilation time vs optimization benefit trade-offs
   - Characterize retracing costs
   - Timeline: 1 week

## Non-Issues (Cleared)

These patterns were investigated but appear intentional or already mitigated:

1. **while_loop usage:** Most cases are intentional for memory bounding ✅
2. **Static dtype:** DTYPE constants are intentional, not limitations ✅
3. **Explicit TensorSpec:** Used appropriately with input_signature ✅
4. **Module-level constants:** Appropriate for readability ✅
5. **reduce_retracing with input_signature:** Used correctly where present ✅

## Comparison to LEDH Refactor

The LEDH while-loop refactor successfully addressed several patterns that appear elsewhere:

**LEDH refactor wins:**
- ✅ Unrolled horizon loops → bounded while_loop
- ✅ Serial direction loops → tf.vectorized_map
- ✅ Graph size reduction (qualitative: O(10⁶) → O(10³))
- ✅ Form (c) architecture (K tangents share primal)
- ✅ Closure capture for nested functions
- ✅ Constant precomputation outside loops

**Similar patterns in other kernels:**
- Contract E streaming JVP: same serial direction loop
- RQMC initial score: same memory bounding strategy
- Multiple routes: same static shape requirements

**Generalization opportunity:** The refactor patterns are reusable. Consider:
1. Creating a refactor template/checklist
2. Systematic audit of all JVP/VJP implementations
3. Shared helper library for common patterns

## Caveats and Limitations

This audit is based on static code analysis (grep, pattern matching) without:

❌ Profiling data  
❌ Runtime measurements  
❌ GPU memory traces  
❌ Compilation time measurements  
❌ Production workload characterization  
❌ XLA HLO graph dumps  
❌ TensorFlow profiler traces  

A complete performance audit would require:

1. **Representative benchmark suite**
   - Cover common models (LGSSM, SV, SIR, predator-prey)
   - Cover common scales (N=100, 1000, 10000)
   - Cover common horizons (T=10, 100, 1000)

2. **GPU profiling**
   - nvidia-smi dmon during runs
   - nvprof detailed traces
   - TensorBoard profiler
   - Peak memory measurement

3. **TensorFlow profiling**
   - tf.profiler.experimental.Profile
   - Op-level timing
   - Memory timeline analysis

4. **XLA analysis**
   - HLO graph dumps (before/after optimization)
   - XLA compilation logs
   - Fusion analysis

5. **Baseline comparisons**
   - Theoretical FLOP limits
   - Memory bandwidth limits
   - Roofline model analysis

## Priority Matrix

| Issue | Severity | Effort | ROI | Priority |
|-------|----------|--------|-----|----------|
| Contract E serial JVP | HIGH | LOW | HIGH | 1 |
| Memory profiling | HIGH | LOW | HIGH | 2 |
| XLA QR documentation | MED-HIGH | LOW | HIGH | 3 |
| Retracing detection | MEDIUM | MED | MED | 4 |
| O(N²) RQMC research | HIGH | HIGH | MED | 5 |
| Shape polymorphism | MEDIUM | MED | MED | 6 |
| P73_B optimizer | MEDIUM | HIGH | LOW | 7 |
| tf.map_fn audit | LOW | LOW | LOW | 8 |

## Next Steps for Owner

1. Review this audit and prioritize issues
2. Decide on immediate actions (items 1-3 recommended)
3. Allocate research time for O(N²) memory issue
4. Run comprehensive profiling study (with GPU escalation)
5. Create performance regression test suite

## References

- LEDH while-loop refactor: [docs/plans/ledh-while-loop-refactor-program-result-2026-09-03.md](ledh-while-loop-refactor-program-result-2026-09-03.md)
- LEDH reset memo: [docs/plans/ledh-while-loop-refactor-reset-memo-2026-09-03.md](ledh-while-loop-refactor-reset-memo-2026-09-03.md)
- P3 XLA scoping note: `docs/plans/bayesfilter-p3-xla-port-scoping-note-2026-08-18.md` (referenced in code, not on worktree)
- 2026-08-19 RQMC result note: referenced in RQMC memory comment (not found on worktree)
- TensorFlow function optimization guide: https://www.tensorflow.org/guide/function
- XLA architecture: https://www.tensorflow.org/xla/architecture

---

**Audit performed by:** Claude Opus 5  
**Date:** 2026-09-03  
**Method:** Static code analysis via grep/pattern matching  
**Confidence:** Medium (needs profiling validation)

---

## CORRECTION: RQMC Status (2026-09-03)

**Finding:** The O(N²) RQMC memory issue (Issue #3) is **NOT in the production path**.

**Evidence:**
1. Production program definition (`ledh_production_program_v1.py`) specifies:
   - Reset policy: "contract_e" (required)
   - Does NOT mention RQMC
   
2. RQMC module (`ledh_pfpf_genut_initial_rqmc_tf.py`) is only referenced in:
   - Test files (`test_ledh_pfpf_genut_initial_rqmc_*.py`)
   - Benchmark/experiment scripts
   - NOT imported by production score lanes

3. RQMC appears to be an experimental initialization method, not the canonical production route.

**Impact on audit priorities:**

- **Issue #3 severity downgraded: HIGH → LOW-MEDIUM**
  - Still a real limitation for RQMC experiments
  - Not blocking production deployments
  - Research concern, not operational concern

**Updated priority matrix:**

| Issue | Severity | Production Impact | Priority |
|-------|----------|-------------------|----------|
| Contract E serial JVP | HIGH | HIGH | 1 |
| Memory profiling | HIGH | HIGH | 2 |
| XLA QR documentation | MED-HIGH | MEDIUM | 3 |
| Retracing detection | MEDIUM | MEDIUM | 4 |
| Shape polymorphism | MEDIUM | MEDIUM | 5 |
| P73_B optimizer | MEDIUM | LOW | 6 |
| O(N²) RQMC research | LOW-MED | NONE | 7 |
| tf.map_fn audit | LOW | LOW | 8 |

**Recommendation:** Focus immediate performance work on production-path issues (Contract E JVP, memory profiling, XLA QR). RQMC optimization is important for research but doesn't block production claims.

