# Phase 4: Integration and Measurement — Subplan

**Program**: `ledh-while-loop-refactor-2026-08-30`
**Phase**: 4
**Status**: NOT STARTED (blocked on Phase 3)
**Created**: 2026-08-30
**Corrected**: 2026-09-01 (removed surrogate-force HMC smoke test per Codex correction #10)

---

## Objective

Measure the refactored kernel's graph size, trace cost, and tf.function compilation behavior. Record what the refactor did and did not establish. This phase closes the program.

**Out of scope** (Codex correction #10, 2026-09-01): Surrogate-force HMC integration, acceptance-rate measurement, sampler validation. Those belong to a separate program with their own evidence contract. This refactor establishes graph representation and parity only.

---

## Preconditions

1. Phase 3 complete
2. Phase 1–3 result notes on file
3. Full parity suite passing (fused and non-fused tests)
4. Campaign authorization in effect

---

## Work items

### 4.1 Stable input signature (Codex correction #6, 2026-09-01)

The refactor contract requires `tf.function` with an explicit, stable `input_signature`. The kernel reads `int(theta.shape[0])`, `int(initial_states.shape[0])`, `int(initial_states.shape[1])`, `int(observations.shape[0])`, and `int(observations.shape[1])` as Python ints, so shapes must be static.

Pin the signature to exact static shapes for the target configuration rather than using `None` extents. `None` would make those `int(...)` calls fail. Record the pinned configuration and state plainly that a different `(B, N, dim, horizon, obs_dim, K)` tuple retraces — that is a real limitation of this kernel, not something to paper over.

**Implementation**: Add `input_signature` parameter to `@tf.function` decorator on the refactored kernel, with explicit TensorSpec shapes for all inputs.

### 4.2 Retracing check

Call the wrapped function repeatedly with the same signature and assert the concrete-function count does not grow. A silent retrace per HMC leapfrog step would erase every gain from Phases 1–3, and it is the single most likely way for this refactor to look successful in a benchmark and fail in the sampler.

Add this as a test, not just a diagnostic — it is a permanent regression guard.

**Implementation**: Test calls the kernel 10 times, checks `len(func._list_all_concrete_functions_for_serialization())` stays at 1.

### 4.3 XLA smoke test (Codex correction #6, 2026-09-01)

Attempt one parity test with `jit_compile=True`. If it succeeds, record XLA-compatible. If it fails with compatibility error, record XLA-incompatible and document the limitation.

**XLA promotion is out of scope** — this is diagnostic only. An XLA failure does not block phase completion.

### 4.4 Final diagnostics

Re-run graph-size and trace-time measurements and build the before/after table against the Phase 0 baseline. If diagnostic scripts are missing (D1), skip those modes, document the limitation, continue with available diagnostics.

Target measurements (actual values to be captured):
- Graph nodes, 1 call, B=1, horizon=50
- GraphDef bytes
- Trace + first eval time
- Host RSS during tracing

Compare against Phase 0 baseline to quantify graph-size reduction.

### 4.5 Coverage close-out

Final coverage of `ledh_canonical_batch_fused_tf` and `ledh_canonical_batch_tf` against the Phase 0 baseline. If coverage fell, identify which lines lost coverage and either add a test or
record why the line is unreachable in the CPU float64 test lane.

### 4.7 Program result note and reset memo

The program result note aggregates the four phase notes into: what was
established, what was measured, what changed in the API, what was rejected on
evidence, what remains unvalidated, and the next program.

The reset memo records the state a future agent needs: that the refactor lives
on `worktree-ledh-canonical-rebuild`, what the kernel's contract now is, which
tests gate it, and that surrogate-force HMC validation is the open next step.

---

## Deliberately out of scope

**GPU device-memory validation.** The whole test lane is CPU-only float64 by
design: every test file sets `CUDA_VISIBLE_DEVICES=-1` before importing
TensorFlow, and `scripts/run_phase_tests.sh` exports it before Python starts.
Measuring GPU device memory requires an escalated GPU run under the GPU/CUDA
policy, which would mean a mid-execution escalation prompt — exactly what the
zero-interruption requirement forbids.

This is why master-program success criterion 5 is restated as host RSS, graph
node count, and GraphDef bytes on CPU. GPU device-memory validation is a
separate, owner-scheduled escalated step, recorded as an explicit open item in
the program result note. The refactor's host-side gains are measurable on CPU;
the device-side consequence is not claimed.

**Sampler validity.** No posterior-correctness, convergence, or acceptance claim
is in scope. Phase 4 delivers a kernel, not a validated sampler.

**Multi-parameter tuning.** Per the LEDH per-scope tuning rule, any claim-bearing
run needs its own tuning artifact for its exact scope. This program produces no
claim-bearing run and therefore no tuning artifact, and none of its numbers may
be read as tuned performance.

---

## Verification

| Step | Command | Pass condition |
|---|---|---|
| 1 | `bash scripts/run_phase_tests.sh parity` | all pass |
| 2 | `bash scripts/run_phase_tests.sh parity-fused` | all pass |
| 3 | `bash scripts/run_phase_tests.sh score-suite` | no new failures vs Phase 0 |
| 4 | `bash scripts/run_phase_tests.sh canonical` | no new failures vs Phase 0 |
| 5 | `bash scripts/run_phase_tests.sh surrogate-hmc` | completes, all finite, no retracing |
| 6 | `bash scripts/run_phase_tests.sh graph-size` | final table populated |
| 7 | `bash scripts/run_phase_tests.sh eval-time` | final table populated |
| 8 | `bash scripts/run_phase_tests.sh direction-cost` | final table populated |
| 9 | `bash scripts/run_phase_tests.sh coverage` | ≥ Phase 0 baseline |

---

## Repair scope

Repairable without asking:

- Driver wiring and signature mismatches
- Pinned-signature shape corrections
- Retracing caused by a fixable Python-side non-tensor argument
- Smoke-test configuration too large for the CPU lane — reduce it
- Missing coverage on newly added lines — add tests

Not repairable — stop conditions:

- Any test that passed in Phase 0 and fails now
- Retracing that cannot be eliminated with a pinned signature
- Non-finite values in the CPU smoke path
- Discovering that the exact-value / damped-force asymmetry was broken by the
  refactor. This is a correctness stop, not a tuning issue: report it, do not
  work around it.

---

## Deliverables

1. Updated `step1_true_surrogate_force.py`
2. Retracing regression test
3. `ledh-while-loop-refactor-phase4-result-2026-08-30.md` with the final
   before/after table and the CPU smoke result plus its non-claims
4. `ledh-while-loop-refactor-program-result-2026-08-30.md`
5. `ledh-while-loop-refactor-reset-memo-2026-08-30.md`
6. Semantic commit

---

## Success criteria

- [ ] Driver updated; exact-value/damped-force asymmetry verified intact by reading
- [ ] `tf.function` with pinned static `input_signature`; pinned configuration recorded
- [ ] Retracing regression test added and passing
- [ ] CPU smoke completes with all values finite and no retracing
- [ ] Final before/after table populated from measurement
- [ ] Coverage ≥ Phase 0 baseline
- [ ] Program result note and reset memo written
- [ ] GPU device-memory validation recorded as an open owner-scheduled item
- [ ] Committed

---

## What this program will have established when complete

- The fused LEDH batch kernel executes through bounded `tf.while_loop` bodies
  instead of `horizon × substeps` unrolled subgraphs
- Measured host-side graph size, trace time, warm eval time, and RSS before and
  after
- Whether one multi-direction call beats P swept calls, answered by measurement
  either way
- Parity with the single-cloud authority preserved at rtol 5e-4 throughout
- A kernel ready for the surrogate-force HMC validation program

## What it will not have established

- That surrogate-force HMC is correct, valid, or convergent
- Any GPU device-memory or GPU throughput result
- Any tuned-performance number for any model
- Any posterior or statistical claim
