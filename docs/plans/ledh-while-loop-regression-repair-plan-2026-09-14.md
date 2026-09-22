# LEDH While-Loop Regression Repair Plan

**Date:** 2026-09-14  
**Branch:** surrogate-hmc  
**Defect:** Phase 2B unification (5cc59cfa, 2026-09-11) deleted a working `tf.while_loop` implementation, regressing to Python `range` loops that unroll ~400 stages into the graph.

---

## Executive Summary

**Regression identified:** Commit 5cc59cfa unified the single-cloud and fused-batch LEDH implementations onto one canonical engine (`ledh_canonical_score_tf.py`). The deleted fused-batch adapter (`ledh_canonical_batch_fused_tf.py` at parent commit) contained 14 `tf.while_loop` calls implementing bounded time/substep iteration with multi-direction tangent propagation. The current unified engine has ZERO `tf.while_loop` and THREE Python `range` loops that unroll into the traced graph.

**Evidence:**
- Current engine: 0 `tf.while_loop`, 3 Python `for time_index/stage/step_index in range()`
- Deleted code (5cc59cfa~1): 14 `tf.while_loop`, 0 Python `range` loops
- Phase 0-4 result documents from 2026-09-02/03 all present, claiming completion
- Reset memo (2026-09-03) records "refactor complete at engineering level"
- No measurement of the claimed "O(10⁶) nodes → O(10³) nodes" or "trace time 101.8s → O(10s)" ever recorded in phase results

**Current measured cost law:**
- Time independent of N (×0.99 for 24→252)
- Time independent of substeps (×1.01 for 2→4)
- Time exactly linear in T (×2.00 for 5→50)
- `correction_steps` 4→0 saves 37%, `pairwise_steps` 4→0 saves 32%
- Signature: fixed iteration counts per timestep, insensitive to problem size = many tiny sequential kernels with GPU mostly idle
- Trace time 12× steady state (T=5: trace 30.7s vs steady 2.6s; T=50: trace 485.1s vs steady 39.3s)

**pfor (approach 1) measured and rejected:**
- Small scale (N=24, T=5): sequential 15.7s, pfor 42.0s → ×0.37 (2.7× slower)
- Plan scale (N=252, T=50): sequential 190.6s, pfor 720.2s → ×0.26 (3.8× slower)
- Memory: 12.3 MB → 37.6 MB
- Parity: PASS at both scales (values/scores identical to rtol 1e-12)
- **Verdict:** pfor computes the right answer but is the wrong architecture; vectorization over K directions buys nothing when the problem is dispatch-bound, not compute-bound.

**The user's four-part directive:**
1. Trace through the whole code in detail
2. Create a subplan for the repair phase
3. Repair/amend the master program and review the whole program
4. Execute the repair phase with tests

**This document satisfies part (1) and part (2).**

---

## Trace Through The Whole Code

### File: `bayesfilter/highdim/ledh_canonical_score_tf.py` (919 lines)

**Public entry point:** `canonical_value_and_analytical_score` (lines 682-754)
- Signature: `(model, theta, initial_states, initial_covariances, noises, observations, *, flow_substeps, with_score, return_trace=False, reset_policy="none", ...)`
- Delegates to `_value_and_analytical_score_impl` (line 67) after minimal argument forwarding
- Single theta `[P]`, single direction (implicit), returns scalar value and scalar score

**Core implementation:** `_value_and_analytical_score_impl` (lines 67-679)

**Loop structure (THE DEFECT):**

**Outer loop — TIME (line 216):**
```python
for time_index in range(horizon):
```
- `horizon` is `int(observations.shape[0])` at line 172
- For T=50, this unrolls 50 copies of the timestep body into the graph
- Python `range` → no traced loop, pure unrolling

**Middle loop — ANNEALED STAGES (line 315, inside time loop):**
```python
for stage in range(1, annealed_stages + 1):
```
- `annealed_stages` defaults to 1, so this normally executes once
- When `annealed_stages > 1`, this further multiplies graph size
- Also pure Python unrolling

**Inner loop — FLOW SUBSTEPS (line 787, inside `_flow_substeps_with_tangent`):**
```python
for step_index in range(substeps):
```
- `substeps` (called `flow_substeps` at the public API) is a required kwarg, typical value 8
- This is inside the per-particle flow function called at line 360
- For T=50, substeps=8: 50 × 8 = 400 flow stages unrolled

**Total unrolling:** T × annealed_stages × substeps = 50 × 1 × 8 = 400 graph copies of the substep body.

**Loop carry variables (what would become `loop_vars` under `tf.while_loop`):**

Time loop carries (across timesteps):
- `states` [N, d] — posterior particle locations after timestep `t`, initial at line 199
- `d_states` [N, d] — tangent of states w.r.t. theta
- `covariances` [N, d, d] — posterior particle covariances after timestep `t`
- `d_covariances` [N, d, d, d] — tangent of covariances (NOTE: rank 4, not rank 3)
- `incoming_log_weights` [N] — log-weights entering the next timestep
- `d_incoming_log_weights` [N] — tangent of incoming weights
- `total` scalar — accumulated log-likelihood
- `d_total` scalar — tangent of total (the score)

Reassignments (where carries update):
- Line 356-357: `total +=`, `d_total +=` (annealed telescope contribution)
- Line 458-459: `total +=`, `d_total +=` (main log-likelihood increment)
- Line 607-608: `states, d_states = reset_states, d_reset_states` (Contract-E reset)
- Line 608-612: `covariances, d_covariances = ...` (reset covariances)
- Line 612-613: `incoming_log_weights = uniform_log_weights`, `d_incoming_log_weights = tf.zeros...` (reset weights)
- Line 631-634: `states, d_states = children, d_children` (post-correction resampling)
- Line 633-634: `incoming_log_weights = uniform_log_weights`, `d_incoming_log_weights = ...` (post-correction reset)

Substep loop carries (inside `_flow_substeps_with_tangent`, lines 757-904):
- `actual` [N, d] — current particle cloud during flow
- `d_actual` [N, d] — tangent
- `auxiliary` [N, d, d] — auxiliary covariance matrix
- `d_auxiliary` [N, d, d, d] — tangent (rank 4)
- `log_det` [N] — accumulated log-determinant
- `d_log_det` [N] — tangent

**Diagnostic accumulators:**
- `trace` — a Python list, appended once per timestep (line 673) when `return_trace=True`
- Contains per-timestep diagnostics: predicted means/covs, posterior means/covs, reset transport, correction records
- Incompatible with `tf.while_loop` unless converted to `tf.TensorArray`

**Model callbacks (per-point, must handle [N,d] batches):**
- `model.transition_mean_fn(theta, points)` → [N, d]
- `model.transition_mean_tangent_fn(theta, points, d_points)` → [N, d]
- `model.observation_fn(points)` → [N, obs_dim]
- `model.observation_tangent_fn(points, d_points)` → [N, obs_dim]
- Optional: `observation_log_density_fn`, `observation_log_density_tangent_fn`
- Optional: `process_covariance_tangent_fn(theta)`, `observation_covariance_tangent_fn(theta)`

**Branches inside the time loop:**
- Line 299-365: Annealed telescope (when `annealed_stages > 1`)
- Line 396-478: Main target evaluation (always executed)
- Line 482-615: Contract-E reset (when `reset_policy == "contract_e"`)
- Line 538-636: Higher-moment correction (when `correction_steps > 0`)

**Why `tf.while_loop` is necessary:**
1. The three Python `range` loops unroll into the graph, creating T × substeps = ~400 replicated subgraphs
2. Graph construction time is linear in unrolled body count: T=5 → 30.7s trace, T=50 → 485.1s trace (12× ratio)
3. Steady-state execution is dispatch-bound: time independent of N and substeps, linear in T only
4. XLA cannot fuse 400 separate subgraphs as effectively as one body executed 400 times

### File: `bayesfilter/highdim/ledh_canonical_batch_fused_tf.py` (290 lines, current HEAD)

**Public entry point:** `canonical_batch_fused_value_score` (lines 115-287)
- Signature: `(model, theta, theta_directions, initial_states, initial_covariances, noises, observations, *, substeps, ...)`
- `theta` [B, P], `theta_directions` [B, K, P] or [B, P] (squeezed K=1 case)
- Returns `(value [B], score [B, K] or [B], diagnostics)`
- Added `k_batch_mode="sequential"` parameter (line 144) — controlled pfor vs sequential K evaluation

**Adapter structure:**
- Outer `tf.map_fn` over B batch rows (line 269), `parallel_iterations=1` → sequential
- Inner K-direction evaluation via:
  - `k_batch_mode == "pfor"`: `tf.vectorized_map` (line 246)
  - `k_batch_mode == "sequential"`: `tf.map_fn(..., parallel_iterations=1)` (line 251)
- Each (theta_row, direction) pair calls `_single_cloud_model` (line 46) to bind callbacks
- Then delegates to `canonical_value_and_analytical_score` (single-cloud engine) at line 228

**What this file does NOT contain:** any while-loop. It is a pure adapter that maps batch/direction dimensions onto the single-cloud engine.

**Deleted implementation (5cc59cfa~1, 732 lines):**
- Had 14 `tf.while_loop` calls implementing:
  - Outer time loop with bounded `maximum_iterations=horizon`
  - Inner substep loop with bounded `maximum_iterations=substeps`
  - K-direction tangent loops (7 sites: UKF predict tangent, anchor tangent, flow tangent, observation log tangent, observation tangent, update tangent, plus one inside the per-row map)
- Used `shape_invariants` for loop-varying tensor ranks
- Had closure capture for module-level helpers (`chol_diff`, `gaussian_log_and_tangent`)
- Hoisted constants outside loops (log normalization, precomputed Cholesky factors)
- Form (c) architecture: one primal evaluation, K tangent evaluations sharing that primal

**Why it was deleted:** Phase 2B "unified" the single-cloud and fused-batch implementations to eliminate duplicate algorithm logic. The unified engine lives in `ledh_canonical_score_tf.py` and is single-direction only. The adapter maps B×K onto that single-direction engine.

**The regression:** The deleted fused-batch implementation had `tf.while_loop` in its own body. The unified single-cloud engine does not. So the unification traded "732 lines with while-loops, calling a 617-line single-cloud engine without while-loops" for "290-line adapter without while-loops, calling a 919-line unified engine also without while-loops." Net -715 lines, but the loop structure went backward.

### File: `bayesfilter/highdim/ledh_unified_reset_tf.py` (276 lines)

**Purpose:** Contract-E Sinkhorn transport + covariance carry
- Called by the canonical engine at line 497-515 when `reset_policy == "contract_e"`
- Entry point: `contract_e_reset_with_tangent` (line 34)
- Contains its own Python `for sinkhorn_iter in range(reset_sinkhorn_steps)` loop (not shown in trace, but present)
- Also has balance iterations: `for balance_iter in range(reset_balance_steps)`
- These are additional unrolled loops inside the already-unrolled time loop

### File: `bayesfilter/highdim/ledh_unified_correction_tf.py` (167 lines)

**Purpose:** Higher-moment diagonal/pairwise/coordinate correction
- Called by the canonical engine at lines 544-564 when `correction_steps > 0`
- Entry point: `unified_correction_with_tangent` (line 29)
- Contains `for correction_iter in range(correction_steps)` (line 88)
- Also `for pairwise_iter in range(pairwise_steps)` (line 190)
- More unrolled loops inside the time loop

**Total nested unrolling:** T × annealed_stages × substeps × (reset_sinkhorn + correction + pairwise). For T=50, substeps=8, reset_sinkhorn=8, correction=4, pairwise=4: 50 × 1 × 8 × (8 + 4 + 4) = 6,400 op replications in the worst case (when all features are enabled). The measured cost law shows correction/pairwise contribute 37% and 32% independently, consistent with this.

---

## Carry-Dependency Analysis

### Time loop `tf.while_loop` conversion

**Loop signature:**
```python
def time_cond(t, states, d_states, covariances, d_covariances,
              incoming_log_weights, d_incoming_log_weights, total, d_total):
    return t < horizon

def time_body(t, states, d_states, covariances, d_covariances,
              incoming_log_weights, d_incoming_log_weights, total, d_total):
    # ... timestep logic (lines 217-634) ...
    # returns updated (t+1, states, d_states, ..., total, d_total)

_, states, d_states, covariances, d_covariances, \
    incoming_log_weights, d_incoming_log_weights, total, d_total = tf.while_loop(
    time_cond,
    time_body,
    loop_vars=(
        tf.constant(0, dtype=tf.int32),  # t
        initial_states,
        tf.zeros_like(initial_states),
        initial_covariances,
        tf.zeros_like(initial_covariances),
        uniform_log_weights,
        tf.zeros_like(uniform_log_weights),
        tf.zeros([], dtype),
        tf.zeros([], dtype),
    ),
    shape_invariants=(
        tf.TensorShape([]),      # t: scalar
        tf.TensorShape([None, None]),  # states: [N, d]
        tf.TensorShape([None, None]),  # d_states: [N, d]
        tf.TensorShape([None, None, None]),  # covariances: [N, d, d]
        tf.TensorShape([None, None, None, None]),  # d_covariances: [N, d, d, d]
        tf.TensorShape([None]),  # incoming_log_weights: [N]
        tf.TensorShape([None]),  # d_incoming_log_weights: [N]
        tf.TensorShape([]),      # total: scalar
        tf.TensorShape([]),      # d_total: scalar
    ),
    maximum_iterations=horizon,
)
```

**Per-timestep slices (must be indexed by `t` inside body):**
- `observations[t]` → `observations[time_index]` (line 217)
- `noises[t]` → `noises[time_index]` (line 218)
- If `return_trace`: `trace.append(...)` at line 673 → convert to `tf.TensorArray` writes

**Annealed stages loop (line 315):**
- Normally `annealed_stages=1`, so loop executes once → can be left as Python `range` if that case is dominant
- For `annealed_stages > 1`, would need another `tf.while_loop` inside the time body
- Or: flatten the two loops into one with `effective_horizon = horizon * annealed_stages` and index arithmetic

### Substep loop `tf.while_loop` conversion (inside `_flow_substeps_with_tangent`)

**Current function signature (line 757):**
```python
def _flow_substeps_with_tangent(
    actual, d_actual, auxiliary, d_auxiliary,
    model, theta, process_chol, d_process_chol,
    substeps, jitter, dtype,
):
```

**Loop signature:**
```python
def substep_cond(s, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det):
    return s < substeps

def substep_body(s, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det):
    # ... lines 788-899 ...
    return s+1, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det

_, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det = tf.while_loop(
    substep_cond,
    substep_body,
    loop_vars=(
        tf.constant(0, dtype=tf.int32),
        actual,  # [N, d]
        d_actual,  # [N, d]
        auxiliary,  # [N, d, d]
        d_auxiliary,  # [N, d, d, d]
        tf.zeros([N], dtype),  # log_det
        tf.zeros([N], dtype),  # d_log_det
    ),
    shape_invariants=(...),
    maximum_iterations=substeps,
)
```

**No per-step slices:** all inputs to the substep loop are already per-timestep scalars/tensors. The flow does not index into observations or noises.

### Sinkhorn/balance/correction/pairwise loops

These live in `ledh_unified_reset_tf.py` and `ledh_unified_correction_tf.py`. They are additional `for` loops that would also need `tf.while_loop` conversion. However, their iteration counts are typically smaller (8, 4, 4) and they execute conditionally (only when `reset_policy` or `correction_steps` are enabled). **Defer these to a later phase.** The time and substep loops are the 50× and 8× multipliers that dominate cost.

---

## Blockers and Compatibility Issues

### 1. `return_trace` accumulator

**Problem:** `trace` is a Python list appended at line 673. `tf.while_loop` cannot append to Python containers.

**Solution:** Use `tf.TensorArray` with `size=horizon, dynamic_size=False`.
```python
trace_array = tf.TensorArray(dtype=tf.float32, size=horizon, element_shape=...)
# inside body:
trace_array = trace_array.write(t, step_record_as_tensor)
# after loop:
trace = trace_array.stack()  # or .read(t) for each t
```

**Complication:** `step_record` is a Python dict with heterogeneous dtypes (floats, integers, bools). TensorArray requires uniform dtype. Options:
- (a) Separate TensorArrays per field
- (b) Pack into one large tensor with fixed structure
- (c) Make `return_trace` incompatible with `tf.while_loop`, require `return_trace=False` for the refactored kernel

**Recommendation:** Option (c) for Phase 1. `return_trace` is a diagnostic feature, not used in production HMC. Surrogate-force HMC does not request traces. Add a validation:
```python
if return_trace and USE_WHILE_LOOP:
    raise ValueError("return_trace requires the unrolled fallback; set USE_WHILE_LOOP=False")
```

### 2. `observation_factor_override` and `post_reset_transform` callables

**Problem:** These are Python callables passed as optional arguments (lines 286-295, 576-581). If they are not `tf.function`-decorated, they cannot be called inside a traced `tf.while_loop` body.

**Solution:** Document that these must be `tf.function`-compatible. Most uses already are (they're typically simple scaling functions). Add a runtime check or accept the traceback on first call.

### 3. `annealed_stages > 1`

**Problem:** The annealed telescope loop (line 315) is inside the time loop. Nested `tf.while_loop` is supported but adds complexity.

**Solution (Phase 1):** Require `annealed_stages=1` for the `tf.while_loop` path. The telescope feature is rarely used. Add validation:
```python
if annealed_stages != 1 and USE_WHILE_LOOP:
    raise ValueError("annealed_stages > 1 requires the unrolled fallback")
```

**Future:** Flatten into one loop with `effective_t = stage * horizon + time_index`.

### 4. Shape polymorphism vs static shapes

**Current:** All shapes are dynamic (determined at runtime from input tensors).

**`tf.while_loop` requirement:** `shape_invariants` must be specified. Unknown dimensions can be `None`, but the rank must be fixed.

**Solution:** Use `tf.TensorShape([None, None, ...])` for batch-varying dimensions. The deleted implementation (5cc59cfa~1) already did this correctly at lines 500-711 (fused adapter's time loop).

### 5. Model callback compatibility

**Current:** Callbacks are plain Python functions passed as `model.transition_mean_fn`, etc.

**Requirement:** These must be traceable by AutoGraph. Most model callbacks are already TensorFlow-only (no NumPy, no Python conditionals).

**Solution:** Document that custom models must use `tf.function`-compatible callbacks. Existing models (`_diagonal_lgssm_fused_model`, KSC, Austria) already satisfy this.

---

## Master Program Amendment

The 2026-08-30 master program `ledh-while-loop-refactor-master-program-2026-08-30.md` is **superseded by this plan**. That program was written for a different codebase state (worktree `ledh-canonical-rebuild`, 866-line fused adapter). The current state (surrogate-hmc branch, 919-line unified engine) requires a different decomposition.

**New phases:**

### Phase 0: Baseline Measurement and Regression Documentation (COMPLETE, this document)
- Measured pfor ×0.26 at plan scale → approach 1 rejected
- Traced current engine structure: 3 Python `range` loops, 8 carry variables, ~400 unrolled stages
- Identified deleted reference implementation (5cc59cfa~1, 732 lines, 14 `tf.while_loop`)
- Located Phase 2B unification commit that caused the regression
- **Artifact:** This plan document

### Phase 1: Time Loop `tf.while_loop` Restoration
- **Objective:** Convert `for time_index in range(horizon)` to `tf.while_loop` with bounded `maximum_iterations=horizon`
- **Scope:** `ledh_canonical_score_tf.py`, function `_value_and_analytical_score_impl`
- **Constraints:** 
  - `annealed_stages=1` only (validate and reject annealed_stages > 1)
  - `return_trace=False` only (validate and reject return_trace=True)
  - `observation_factor_override=None` and `post_reset_transform=None` (validate)
- **Parity gate:** Single-cloud value and score at rtol 5e-4 vs current unrolled baseline
- **Deliverable:** Working time loop, substeps still unrolled (400 → 50 graph copies)

### Phase 2: Substep Loop `tf.while_loop` Restoration
- **Objective:** Convert `for step_index in range(substeps)` to `tf.while_loop` inside `_flow_substeps_with_tangent`
- **Scope:** `ledh_canonical_score_tf.py`, function `_flow_substeps_with_tangent` (lines 757-904)
- **Constraints:** None beyond Phase 1
- **Parity gate:** Single-cloud value and score at rtol 5e-4 vs Phase 1 baseline
- **Deliverable:** Bounded time and substep loops (50 → ~1 traced body executed 400 times)

### Phase 3: Measurement and Integration
- **Objective:** Measure graph size, trace time, steady-state time vs current unrolled baseline
- **Scope:** `docs/benchmarks/ledh_execution_mode_matrix.py` (already written, never run)
- **Evidence:** 
  - Graph node count (expect O(10³) vs current ~10⁶ estimate)
  - Trace time (expect <20s vs current 485s at T=50)
  - Steady-state time (expect ≤2× current 39.3s, target competitive)
- **XLA evaluation:** Run matrix script with `compile_mode="xla"`, record compatibility and performance
- **Deliverable:** Measurement artifact under `docs/plans/artifacts/ledh-loop-architecture-20260914/`

### Phase 4: Adapter Multi-Direction Batch Restoration (if needed)
- **Objective:** Restore multi-direction tangent support in the canonical engine (currently single-direction only)
- **Scope:** Add K-direction tangent state to the canonical engine, OR keep the adapter's `tf.map_fn` over K and accept sequential K evaluation
- **Decision gate:** If Phase 3 shows the time loop alone achieves target performance, K-batching may be unnecessary (5 sequential calls at 10s each = 50s total, acceptable for gradient evaluation)
- **Deliverable:** TBD based on Phase 3 results

---

## Success Criteria

### Engineering Criteria (required for Phase 1-2 completion)
1. **Parity:** Value and score match the current unrolled baseline at rtol 5e-4 (op-order tolerance)
2. **Graph compilation:** `@tf.function` with fixed `input_signature` succeeds, no runtime errors
3. **Tests pass:** Existing test suite under `tests/highdim/test_ledh_canonical_*` remains green
4. **No silent behavior change:** All validation gates, assertions, and diagnostics remain active

### Performance Criteria (required for Phase 3 acceptance)
5. **Graph size reduction:** Measured node count O(10³), not O(10⁶)
6. **Trace time reduction:** T=50 first-call <50s (current 485s)
7. **Steady-state competitive:** T=50 steady-state ≤80s (current 39.3s, allow 2× for loop dispatch overhead)
8. **XLA compatibility:** `jit_compile=True` succeeds, no numerical regression

### Integration Criteria (required for Phase 4, if executed)
9. **Surrogate-force HMC integration:** Adapter routes through the refactored engine, no API break
10. **K-direction support:** Either (a) restored inside the engine, or (b) sequential K calls remain acceptable

---

## What This Plan Does Not Repair

- **Sinkhorn/balance loops:** `ledh_unified_reset_tf.py` still has Python `range` loops (smaller iteration counts, conditional execution)
- **Correction/pairwise loops:** `ledh_unified_correction_tf.py` still has Python `range` loops
- **Annealed telescope:** `annealed_stages > 1` remains unrolled (rarely used)
- **Trace accumulation:** `return_trace=True` remains incompatible with the refactored kernel
- **Shape polymorphism:** All shapes remain static (different B/N/T configs retrace)
- **GPU memory measurement:** Deferred to a separate escalated experiment
- **HMC convergence validation:** Deferred to end-to-end sampler validation

These are explicitly out of scope. The repair targets the dominant 50× and 8× multipliers (time and substeps). The conditional smaller loops can be addressed in a later phase if measurement shows they matter.

---

## Execution Approach

**Phase 1 and 2 will be executed by referencing the deleted implementation (5cc59cfa~1) as the authority.** That code had working `tf.while_loop` structures, correct `shape_invariants`, proper closure capture, and passed parity tests. The regression repair is not a fresh design; it is a restoration of a known-good implementation onto the current unified engine.

**Differences from the 2026-08-30 master program:**
- That program targeted a 866-line fused adapter with duplicate algorithm logic
- This plan targets the 919-line unified canonical engine that already exists
- That program's Phase 0 test audit found 21 test failures; current branch tests are green
- That program's execution model required worktree isolation; this repair happens on surrogate-hmc branch in place
- That program allowed `tf.vectorized_map`; this plan uses only `tf.while_loop` (pfor is measured and rejected)

**Why this is not duplication:** The deleted fused adapter had `tf.while_loop` in its own 732-line body, calling a 617-line single-cloud engine without loops. The current state has a 290-line adapter without loops, calling a 919-line unified engine also without loops. The loops must now go into the 919-line unified engine, which is the new single source of algorithm truth.

---

## Next Step

**User decision required:** Approve this plan and authorize Phase 1 execution, or revise scope/constraints.

Phase 1 will convert the time loop only (50× unrolling), leaving substeps unrolled (8×). This is the smallest measurable increment and allows early validation that the `tf.while_loop` approach works on the current unified engine before proceeding to Phase 2.
