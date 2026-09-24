# Phase 4: Integration and Surrogate-Force Readiness — Subplan

**Program**: `ledh-while-loop-refactor-2026-08-30`
**Phase**: 4
**Status**: NOT STARTED (blocked on Phase 3)
**Created**: 2026-08-30

---

## Objective

Wire the refactored kernel into the surrogate-force gradient driver, verify the
whole path end to end on CPU, and record what the refactor did and did not
establish. This phase closes the program. It does **not** run or validate
surrogate-force HMC as a sampler — that is the next program, and this phase's
job is to hand it a working, measured, honestly documented kernel.

---

## Preconditions

1. Phase 3 complete and committed
2. Phase 1–3 result notes on file with their run manifests
3. Full parity and score suites passing

---

## Work items

### 4.1 Update the surrogate-force driver

`docs/benchmarks/step1_true_surrogate_force.py` currently implements
`exact_value_damped_grad` with `@tf.custom_gradient`, producing 6 traced calls
per gradient at P=5: one exact value at λ=δ=0 and five swept damped directions
at λ=δ=1e-3.

If Phase 2 succeeded, restructure to two calls: one exact value, one K=5 damped
multi-direction call. If Phase 2 was rejected on measured evidence, leave the
6-call structure and take the Phase 1/3 gains only. Both outcomes are
acceptable completions of this phase.

Preserve exactly, and verify by reading rather than assuming:

- The MH accept/reject term uses the **exact** log-density at λ=δ=0
- The leapfrog force uses the **damped** score at λ=1e-3 on process covariance,
  δ=1e-3 on observation covariance
- `@tf.custom_gradient` returns the exact value with the damped gradient — this
  asymmetry is the entire point of the construction and must not be
  "simplified" into consistency

### 4.2 Stable input signature

The refactor contract requires `tf.function` with an explicit, stable
`input_signature`. The kernel reads `int(theta.shape[0])`, `int(initial_states.shape[0])`,
`int(initial_states.shape[1])`, `int(observations.shape[0])`, and
`int(observations.shape[1])` as Python ints, so shapes must be static.

Pin the signature to exact static shapes for the target configuration rather
than using `None` extents. `None` would make those `int(...)` calls fail. Record
the pinned configuration and state plainly that a different `(B, N, dim, horizon,
obs_dim, K)` tuple retraces — that is a real limitation of this kernel, not
something to paper over.

### 4.3 Retracing check

Call the wrapped function repeatedly with the same signature and assert the
concrete-function count does not grow. A silent retrace per HMC leapfrog step
would erase every gain from Phases 1–3, and it is the single most likely way for
this refactor to look successful in a benchmark and fail in the sampler.

Add this as a test, not just a diagnostic — it is a permanent regression guard.

### 4.4 CPU end-to-end smoke

Run a short surrogate-force HMC chain on CPU at reduced size — small D, small N,
short horizon, a handful of leapfrog steps. The purpose is exercising the whole
path for finiteness, shape correctness, and absence of retracing. It is a smoke
test.

State explicitly in the result note that this establishes none of: posterior
correctness, sampler validity, convergence, acceptance-rate adequacy, or the
soundness of the exact-value/damped-force construction. Those need the separate
surrogate-force validation program with its own evidence contract.

### 4.5 Final diagnostics

Re-run `graph-size`, `eval-time`, and `direction-cost` and build the
before/after table against the Phase 0 baselines:

| Metric | Baseline (Phase 0) | Final (Phase 4) |
|---|---|---|
| Graph nodes, 1 call, B=1, horizon=50 | 110,628 | |
| GraphDef bytes | 9.98 MB | |
| Trace + first eval | 16.6 s | |
| Warm eval per call | 0.185 s (swept) | |
| Host RSS during tracing | 1,336 MB | |
| Calls per P=5 gradient | 6 | |
| Nodes per P=5 gradient | 663,766 | |
| RSS per P=5 gradient | 5,245 MB | |

### 4.6 Coverage close-out

Final coverage of `ledh_canonical_batch_fused_tf` against the Phase 0 baseline.
If coverage fell, identify which lines lost coverage and either add a test or
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
