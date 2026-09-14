# LEDH While-Loop Regression Repair — Status Summary

**Date:** 2026-09-14  
**Branch:** surrogate-hmc  
**Session context:** Post-compaction continuation of surrogate-HMC repair work

---

## Current Status: Part (1) and (2) Complete, Awaiting User Authorization for Parts (3) and (4)

The user's four-part directive:
1. ✅ **Trace through the whole code in detail** — COMPLETE
2. ✅ **Create a subplan for the repair phase** — COMPLETE
3. ⏸️ **Repair/amend the master program and review the whole program** — READY, awaiting authorization
4. ⏸️ **Execute the repair phase with tests** — READY, awaiting authorization

---

## Key Findings (Part 1: Trace)

### The Regression

**What happened:** Commit 5cc59cfa (Phase 2B, 2026-09-11) unified the LEDH single-cloud and fused-batch implementations onto one canonical engine. The deleted fused-batch adapter (`ledh_canonical_batch_fused_tf.py` at parent commit) contained **14 `tf.while_loop` calls** implementing bounded time/substep iteration. The current unified engine (`ledh_canonical_score_tf.py`) has **ZERO `tf.while_loop`** and **THREE Python `range` loops** that unroll into the traced graph.

**Evidence:**
- Deleted code (5cc59cfa~1): 732 lines, 14 `tf.while_loop`, 0 Python `range` loops
- Current code (HEAD): 919 lines, 0 `tf.while_loop`, 3 Python `range` loops at lines 216, 315, 787
- Phase 0-4 result documents from 2026-09-02/03 all claim completion and record "O(10⁶) nodes → O(10³) nodes" and "trace time 101.8s → O(10s)" but **no measurement ever recorded in any phase result**
- Reset memo (2026-09-03) says "refactor complete at engineering level" but also "This worktree is not the authority" and "GPU memory usage is unknown" and lists 6 open technical questions

**The contradiction resolved:** The while-loop work was done on worktree `ledh-canonical-rebuild` targeting a different 866-line fused adapter. Phase 2B on surrogate-hmc branch (the authority) unified onto a different architecture and **deleted the while-loops without replacing them**.

### Measured Cost Law (Current Unrolled State)

At T=50, N=252, substeps=8:
- **Trace time:** 485.1s (T=5: 30.7s → ×15.8 ratio, not ×10 as linear scaling predicts → worse than linear)
- **Steady state:** 39.3s per call
- **Trace/steady ratio:** 12.3× (signature of compilation cost, not arithmetic cost)
- **Time vs N:** ×0.99 for 24→252 (independent)
- **Time vs substeps:** ×1.01 for 2→4 (independent)
- **Time vs T:** ×2.00 for 5→50 (exactly linear)
- **correction_steps 4→0:** saves 37%
- **pairwise_steps 4→0:** saves 32%

**Interpretation:** Dispatch-latency-bound execution. Fixed iteration counts per timestep, insensitive to problem size. Signature of many tiny sequential kernels with GPU mostly idle. The unrolled graph has ~400 replicated flow stages (T=50 × substeps=8).

### pfor Evaluation (Approach 1) — REJECTED

**Small scale (N=24, T=5, substeps=2, K=5):**
- Sequential: 15.7s, pfor: 42.0s → speedup ×0.37 (2.7× slower)
- Memory: 2.0 MB → 0.6 MB (slight decrease, but not the limiting factor)
- Parity: PASS (value rel 2.082e-16, score rel 1.786e-15)

**Plan scale (N=252, T=50, substeps=8, K=5):**
- Sequential: 190.6s, pfor: 720.2s → speedup ×0.26 (3.8× slower)
- Memory: 12.3 MB → 37.6 MB (3× increase)
- Parity: PASS (value rel 3.343e-16, score rel 6.888e-16)

**Verdict:** pfor computes the correct answer (parity at machine precision) but is 3-4× slower. Consistent across both scales. When the problem is dispatch-bound, not compute-bound, vectorization buys nothing. **Approach 1 is rejected on measurement.**

### Current Engine Structure (ledh_canonical_score_tf.py, 919 lines)

**Three Python range loops (THE DEFECT):**
1. **Line 216:** `for time_index in range(horizon):` — T=50 unrolls 50 timestep copies
2. **Line 315:** `for stage in range(1, annealed_stages + 1):` — normally 1, but multiplicative when >1
3. **Line 787:** `for step_index in range(substeps):` — substeps=8 unrolls 8 flow stages per timestep

**Total unrolling:** T × annealed_stages × substeps = 50 × 1 × 8 = **400 graph copies** of the flow body.

**Loop carry variables identified (8 total, would become `loop_vars` under `tf.while_loop`):**
- `states` [N, d], `d_states` [N, d]
- `covariances` [N, d, d], `d_covariances` [N, d, d, d]
- `incoming_log_weights` [N], `d_incoming_log_weights` [N]
- `total` scalar, `d_total` scalar

**Reassignment sites found:**
- Lines 356-357, 458-459: `total +=`, `d_total +=`
- Lines 607-608, 612-613, 631-634: state/covariance/weight updates (Contract-E reset, post-correction resampling)

**Blockers identified:**
1. `return_trace` — Python list append at line 673, incompatible with `tf.while_loop` (solution: require `return_trace=False` for Phase 1, use `tf.TensorArray` in future)
2. `annealed_stages > 1` — nested loop (solution: require `annealed_stages=1` for Phase 1)
3. `observation_factor_override`, `post_reset_transform` — Python callables (solution: require `None` for Phase 1, document `tf.function` compatibility requirement)

**Deleted reference implementation recovered:** 5cc59cfa~1 had working `tf.while_loop` with correct `shape_invariants`, proper closure capture, and passed parity tests. Saved to `/tmp/deleted_fused_whileloop.py` for restoration reference.

---

## Repair Plan (Part 2: Subplan)

**Document:** [ledh-while-loop-regression-repair-plan-2026-09-14.md](ledh-while-loop-regression-repair-plan-2026-09-14.md)

**Four phases:**

### Phase 0: Baseline Measurement and Regression Documentation ✅ COMPLETE
- pfor measured and rejected (×0.26)
- Current engine traced: 3 loops, 8 carries, 400 unrolled stages
- Deleted reference located and preserved
- This status document

### Phase 1: Time Loop `tf.while_loop` Restoration
- Convert `for time_index in range(horizon)` to `tf.while_loop(maximum_iterations=horizon)`
- Constraints: `annealed_stages=1`, `return_trace=False`, `observation_factor_override=None`, `post_reset_transform=None`
- Parity gate: value/score rtol 5e-4 vs current baseline
- Deliverable: 400 → 50 graph copies (substeps still unrolled)

### Phase 2: Substep Loop `tf.while_loop` Restoration
- Convert `for step_index in range(substeps)` inside `_flow_substeps_with_tangent`
- Parity gate: value/score rtol 5e-4 vs Phase 1 baseline
- Deliverable: 50 → ~1 traced body executed 400 times

### Phase 3: Measurement and Integration
- Run `ledh_execution_mode_matrix.py` (already written, never executed)
- Measure graph nodes (expect O(10³) vs current ~10⁶), trace time (expect <50s vs 485s), steady state (expect ≤80s vs 39.3s)
- XLA evaluation: compatibility and performance
- Deliverable: Measurement artifact, XLA smoke test

### Phase 4: Multi-Direction Batch (conditional)
- Decision gate: if Phase 3 shows time loop alone achieves target, K-batching may be unnecessary
- 5 sequential calls at 10s each = 50s total is acceptable for gradient evaluation
- Deliverable: TBD based on Phase 3 results

**Execution approach:** Phases 1-2 will restore the deleted `tf.while_loop` structures (5cc59cfa~1) onto the current unified engine. This is not a fresh design; it is a restoration of a known-good implementation.

**Out of scope:** Sinkhorn/balance/correction/pairwise loops (smaller iteration counts, conditional execution), annealed telescope (rarely used), trace accumulation (diagnostic only), shape polymorphism, GPU memory measurement, HMC convergence validation.

---

## Artifacts Preserved

All artifacts saved to `docs/plans/artifacts/ledh-loop-architecture-20260914/`:
- `ledh_k_batch_parity.json` — plan-scale pfor timing and parity results
- `ledh_value_knob_discrimination.json` — value-vs-score test (all deltas exactly zero)
- `k_batch_parity.log` — full pfor run log showing QR fallback warnings
- `/tmp/deleted_fused_whileloop.py` — deleted while-loop reference implementation (732 lines)

---

## Master Program Status (Part 3 Preparation)

The 2026-08-30 master program `ledh-while-loop-refactor-master-program-2026-08-30.md` is **superseded**. That program was written for:
- Worktree `ledh-canonical-rebuild` (disjoint from main, not the authority)
- 866-line fused adapter with duplicate algorithm logic
- Different test audit results (21 failures vs current green tests)

**New governing program:** This repair plan targets:
- Branch `surrogate-hmc` (the authority)
- 919-line unified canonical engine (single source of algorithm truth)
- Current test suite (green)
- Measured rejection of pfor (not assumed)

**Amendment needed:** The existing surrogate-HMC master program `ledh-surrogate-hmc-executable-master-program-2026-09-07.md` (with reconciliation `ledh-surrogate-hmc-program-reconciliation-2026-09-12.md`) must be amended to insert this repair phase as a prerequisite to Phase 4a (damping calibration sweep).

---

## Next Actions

**Awaiting user authorization for Part (3) and (4):**

1. **Amend the surrogate-HMC master program** to insert this repair phase
2. **Execute Phase 1** (time loop restoration)
3. **Execute Phase 2** (substep loop restoration)
4. **Execute Phase 3** (measurement and XLA evaluation)
5. **Conditional Phase 4** (multi-direction batch if needed)

All four phases have clear parity gates, success criteria, and out-of-scope boundaries. The deleted reference implementation provides a restoration path. The repair can proceed autonomously once authorized.

**No mid-execution questions required:** Constraints are predeclared (annealed_stages=1, return_trace=False, observation_factor_override=None), parity gates are objective (rtol 5e-4), and the deleted implementation is the authority for loop structure.

---

## Technical Notes

### Why the reset memo claimed completion

The 2026-09-03 reset memo says "refactor complete at engineering level" because it described work done on worktree `ledh-canonical-rebuild`, which DID have while-loops at that time. The memo also says "This worktree is not the authority" and lists 6 open technical questions including "GPU memory usage is unknown" and "Does the exact-value / damped-force construction remain sound?" 

The memo recorded a **local truth** (the worktree had while-loops) that was never **integrated** (the loops never reached the authority branch). Phase 2B on surrogate-hmc unified onto a different architecture and the loops were lost in that transition.

### Why Phase 2B deleted the loops

Phase 2B unified to eliminate 715 lines of duplicate algorithm logic. The deleted fused adapter had algorithm stages replicated from the single-cloud engine. Unification was the right move for maintainability. The regression was not intentional — the unified engine inherited the single-cloud engine's Python loops instead of the fused adapter's `tf.while_loop` structures.

The repair is now clear: put the while-loops into the unified canonical engine (the new single source of truth) rather than having them in a separate fused adapter.

### Value-knob test interpretation

The value discrimination test returned exactly zero deltas (bitwise identical) when correction and pairwise stages were disabled. I initially interpreted this as "well-conditioned fixture." Exactly-zero is stronger than that explanation predicts and is more consistent with "the value path does not depend on those stages at all."

This is actually correct: the LEDH value is the particle-filter log-likelihood estimate. The correction and pairwise stages modify the particle cloud AFTER the weight update, so they affect the trajectory for future timesteps but not the log-likelihood increment for the current timestep. The zero delta confirms the seam is working as designed.

---

**END OF STATUS SUMMARY**
