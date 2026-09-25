# Phase 3.5.2: Substep Loop Restoration

**Date:** 2026-09-14  
**Branch:** surrogate-hmc  
**Parent Phase:** Phase 3.5 (LEDH While-Loop Regression Repair)  
**Prerequisites:** Phase 3.5.1 (time loop) ✓, Phase 1.5 (XLA enabled) ✓

---

## Objective

Restore `tf.while_loop` for the substep loop in `ledh_canonical_score_tf.py` line 787, completing the while-loop regression repair and meeting the master program's Phase 3.5 success criteria.

---

## Current State (After Phase 1 + 1.5)

**Completed:**
- Phase 3.5.1 (= Phase 1): Time loop restored, 5.5× speedup
- Phase 1.5: Batch adapter XLA enabled, 600× speedup
- Oracle contract: 8/8 tests passing

**Remaining Python loops in canonical engine:**
1. ~~Line 216: `for time_index in range(horizon)`~~ — **FIXED** in Phase 1
2. Line 315: `for stage_index in range(annealed_stages)` — **DEFERRED** (constraint: annealed_stages=1)
3. **Line 787: `for step_index in range(substeps)`** — **TARGET for Phase 3.5.2**

**Current performance (N=252, T=50, substeps=8):**
- Trace time: 485s (target: <50s)
- Steady state: 39.3s (target: ≤80s) ✓
- Graph size: ~400 flow stages (T=50 copies × substeps=8 each) — target: O(10³) nodes

---

## Scope

### In Scope

Convert the substep loop at line 787 in `_flow_substeps_with_tangent()` from:
```python
for step_index in range(substeps):
    # Flow substep body (~150 lines)
```

To:
```python
def substep_body(index, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det):
    # Flow substep body
    return (index + 1, updated_actual, updated_d_actual, ...)

_, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det = tf.while_loop(
    cond=lambda i, *_: i < substeps,
    body=substep_body,
    loop_vars=(tf.constant(0), actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det),
)
```

### Out of Scope

- Annealed telescope loop (line 315) — remains deferred, constraint enforced
- TensorArray trace support — remains deferred, `return_trace=False` only
- Batch adapter changes — Phase 1.5 already complete

### Constraints (Preserved from Phase 1)

- `annealed_stages=1` only (ValueError for > 1)
- `return_trace=False` only (no TensorArray trace)
- `observation_factor_override=None`
- `post_reset_transform=None`

---

## Loop Analysis

### Substep Loop Carry Variables

The substep loop (`_flow_substeps_with_tangent`, lines 757-904) carries:

**Forward values:**
- `actual` [N, d] — particle cloud during flow
- `auxiliary` [N, d, d] — auxiliary covariance matrix
- `log_det` [N] — accumulated log-determinant

**Tangents:**
- `d_actual` [N, d]
- `d_auxiliary` [N, d, d, d] — **rank 4**
- `d_log_det` [N]

**Loop control:**
- `step_index` — loop counter (0 to substeps-1)

### Reassignment Sites

Inside the substep body:
- Line 851-852: `actual, d_actual` updated (flow increment)
- Line 896-897: `log_det, d_log_det` updated (Jacobian)
- Line 903-904: `auxiliary, d_auxiliary` updated (covariance tracking)

### Function Calls Inside Loop

- `model.transition_mean_fn(theta, actual)` — [N, d] → [N, d]
- `model.transition_mean_tangent_fn(theta, actual, d_actual)` — tangent
- `tf.linalg.qr()` — line 896 (analytical tangent implemented)
- Various linear algebra ops (matmul, solve, cholesky)

All operations are TensorFlow-native with known shapes — no Python control flow inside the substep body.

---

## Implementation Plan

### Step 1: Extract Substep Body (2 hours)

**Task:** Isolate the substep body into a separate function with explicit carries.

**Deliverable:** New function `_flow_substep_body_impl()` with signature:
```python
def _flow_substep_body_impl(
    step_index: tf.Tensor,
    actual: tf.Tensor,
    d_actual: tf.Tensor,
    auxiliary: tf.Tensor,
    d_auxiliary: tf.Tensor,
    log_det: tf.Tensor,
    d_log_det: tf.Tensor,
    # ... constants (theta, substeps, noise, correction_steps, etc.)
) -> Tuple[tf.Tensor, ...]:
    """Single flow substep with analytical tangent.
    
    Returns:
        (step_index + 1, updated_actual, updated_d_actual, ...)
    """
```

**Verification:** Inline call in the existing Python loop, verify oracle contract still passes.

### Step 2: Convert to tf.while_loop (2 hours)

**Task:** Replace the Python `range` loop with `tf.while_loop`.

**Deliverable:** Updated `_flow_substeps_with_tangent()`:
```python
def substep_cond(i, *_):
    return i < substeps

initial_loop_vars = (
    tf.constant(0, dtype=tf.int32),
    actual,
    d_actual,
    auxiliary,
    d_auxiliary,
    log_det,
    d_log_det,
)

_, actual, d_actual, auxiliary, d_auxiliary, log_det, d_log_det = tf.while_loop(
    cond=substep_cond,
    body=lambda i, *args: _flow_substep_body_impl(i, *args, theta, ...),
    loop_vars=initial_loop_vars,
)
```

**Verification:** Oracle contract must pass (8/8 tests, rtol 5e-4).

### Step 3: Measurement (1 hour)

**Task:** Measure trace time, steady state, and graph size.

**Script:** Modify `docs/benchmarks/ledh_execution_mode_matrix.py` to measure:
- Trace time (first call with known input_signature)
- Steady state time (average of 5 calls)
- Graph node count (via `tf.profiler` or manual inspection)

**Test scales:**
- Small: N=24, T=5, substeps=2
- Plan: N=252, T=50, substeps=8

**Target metrics (from master program):**
- Trace time: <50s (currently 485s)
- Steady state: ≤80s (currently 39.3s, already passes)
- Graph size: O(10³) nodes (currently ~400 × substep_body_size)

### Step 4: XLA Verification (1 hour)

**Task:** Verify XLA compilation still works with the substep while_loop.

**Test:** Run the canonical engine (not the batch adapter) with `jit_compile=True`:
```python
@tf.function(jit_compile=True)
def evaluate_canonical(theta, initial, covs, noises, observations):
    return canonical_value_and_analytical_score(
        model=model,
        theta=theta,
        initial_states=initial,
        initial_covariances=covs,
        noises=noises,
        observations=observations,
        flow_substeps=8,
    )
```

**Success:** Compilation succeeds, parity maintained.

---

## Success Criteria

From master program Phase 3.5:

1. **Parity:** Value/score rtol 5e-4 vs baseline (oracle contract: 8/8 tests pass)
2. **Performance:**
   - Graph: O(10³) nodes (50 × substeps collapsed → ~1 body per timestep)
   - Trace: <50s (vs current 485s)
   - Steady: ≤80s (vs current 39.3s, already passing)
3. **Integration:** Tests pass, HMC adapter compatible

---

## Risk Assessment

### Low Risk

- Substep body is pure TensorFlow ops with known shapes
- No Python control flow inside the body
- QR factorization (line 896) already has analytical tangent implemented
- Phase 1 time loop restoration used the same pattern successfully

### Medium Risk

- Substep body is ~150 lines — larger than time loop body
- `auxiliary` covariance tracking has rank-4 tangent `d_auxiliary`
- Correction and pairwise steps (inner loops) may need attention

### Mitigation

- Extract body first, verify inline before converting to while_loop
- Run oracle contract after each step
- Measure incrementally: extraction → conversion → measurement

---

## Timeline

- Step 1 (extract body): 2 hours
- Step 2 (convert to while_loop): 2 hours
- Step 3 (measurement): 1 hour
- Step 4 (XLA verification): 1 hour

**Total:** 6 hours (~1 day)

---

## Deliverables

1. Modified `bayesfilter/highdim/ledh_canonical_score_tf.py` with substep while_loop
2. Oracle contract passing (8/8 tests)
3. Measurement artifact from Step 3
4. Result summary: `docs/plans/ledh-phase-3.5.2-substep-loop-result-2026-09-14.md`

---

## Approval

Phase 3.5.2 is explicitly part of the master program Phase 3.5. The master program was approved, so no additional approval is needed for execution.

---

## Next After Phase 3.5.2

Upon successful completion:
- Phase 3.5.3: **Already partially complete** (Phase 1.5 measured XLA at 600×)
- Phase 3.5.4: **Skipped** (Phase 1.5 already fixed K-batch with whileloop in batch adapter)
- **Proceed to Phase 4a:** Damping calibration diagnostic (2 GPU-hours)

If Phase 3.5.2 measurements show further optimization needed, reassess before Phase 4a.
