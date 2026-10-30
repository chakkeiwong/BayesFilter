# Registered LEDH seeded value execution repair

Continue checkpoint aabd2b167/04668 under the authorized execution-only repair.
The question is whether the registered single-cloud value endpoint can execute
its existing seeded RNG and filter recurrence in one XLA owner, preserve healthy
numerics and seeded scheduling, and expose the already-qualified rejection
boundary. Full canonical LEDH reconstruction, analytical-score substitution,
training and HMC remain excluded. Main stays unmerged.

The old registered endpoint mixes NumPy SeedSequence/PCG64, a stateful Philox
generator and Python time/stage loops. The supplied-input value owner and the
compatibility RNG primitives have CPU/GPU evidence, but their composition and
actual public wiring do not. Existing callbacks are configuration closures;
caching by callback identity would silently freeze mutations read afresh by the
old endpoint. Use an explicit caller-owned fixed-configuration seeded factory
with stable observation/seed-word tensor signatures, and a one-shot convenience
wrapper that builds a fresh owner. No automatic global cache. Repeated callers
may retain the factory only while configuration and closures are fixed; mutable
TensorFlow variables remain graph operands/resources under TensorFlow semantics.
Time-dependent callbacks must accept a scalar int32 Tensor time index. Check
actual registered consumers and time-dependent test callbacks, with no Python
callback fallback.

Preserve the existing SeedSequence mixing, initial Philox draw, one process draw
per time step, and independent PCG64(resample_seed + time_index) stage uniforms.
Implement carry across seed words with TensorFlow control flow and call the
existing RNG authorities. Large integer seeds and word-boundary crossings must
match NumPy reference streams exactly for integer/uniform operations; keep the
already declared FP64 1e-12/FP32 1e-5 normal-transform bounds. All random inputs
and filtering must execute under the enclosing XLA function, with observations
and both seed encodings as dynamic inputs. Preparing an integer configuration
as words and attaching strings/trimming completed histories are host boundaries.
No alternate numerical kernel, NumPy runtime, scalar-row mapper or host numerical
loop is permitted. Eager/stateful RNG may remain only in the explicitly named
private diagnostic compatibility helper used by independent tests.

Use the post-invalidation frozen 9d8202b77 value/flow diagnostic authority and
independent NumPy/TF stateful random references. Current shared reset arithmetic
is unchanged. N=8,d=2,T=1/3 and K=N fixtures preserve existing controls; they test
execution, not calibrated method quality. Compare complete healthy records at
the existing atol=rtol=1e-6; separately preserve invalid raw differences and
require NaN public value, false validity, code/index, histories and NaN masks.
Never demand an equivalent usable likelihood from a rejected reset. Seeded
inputs can change the fixture's validity, so record the reference disposition
before judging the candidate. Retain fixed-cloud healthy/no-fire regressions.

Execution order and evidence:

1. Verify offset streams (including 32/64/128/160-bit carry), complete draw
   scheduling, replay and changed seeds against independent references on CPU.
2. Implement and qualify the registered wrapper and reusable owner on CPU:
   registry resolution, healthy/rejected/nonfinite cases, changed observations
   and seeds, mutable closure refresh between one-shot calls, time-dependent
   callbacks, one trace, stable signatures/HLO, no host callbacks and collection.
3. Recheck eligible GPUs and qualify the same cases with verified growth,
   placement, TF32/XLA and trusted-session provenance. GPU remains default.
4. Only after numerical gates pass, measure matched frozen-original eager,
   candidate graph-reference and candidate XLA costs on CPU/GPU in isolated
   workers. Record cold compile/first call, warm calls, host RSS/peak, allocator
   current/peak, release/exit behavior and exact timing boundaries. Include
   one-shot setup cost and reusable-owner amortization. HLO export occurs after
   timed/memory samples. Record device sharing; shared runs are descriptive.
5. Run affected native/random/endpoint policy checks; extend source coverage
   without allowing numerical loops or NumPy. Archive results and update F07
   as partial until score/public-consumer and terminal evidence gates close.

Run via the existing allowlisted interpreter and
`scripts/run_filter_repair_campaign.py test --group <registered group>
--device CPU|GPU --test-timeout-seconds 300` (900 only for a justified cost unit).
Use unique run directories under the existing raw campaign artifact root and
save commands, source identities, complete comparisons, environment and charges.
Allocate at most 24 workers, 5400 CPU seconds and 3600 GPU seconds from the
remaining 25.934150 CPU/25.293292 GPU hours; the global 56/52-hour caps do not
change. One numerical worker at a time, at most two localized harness retries.
No unrelated jobs may be stopped. Stop the current qualification on unexplained
healthy mismatch, seed/ancestry drift, callback incompatibility, missing
diagnostics, device/memory-policy failure or budget exhaustion; localize before
changing the implementation. Failed attempts consume the same allocation.

Skeptical review before execution: a supplied-input pass cannot prove public
seed scheduling, and an automatically cached closure can return stale values.
Both are explicit gates here. The old finite-only status is not a validity
authority; raw invalid diagnostics cannot enter likelihood fallback. A tiny
healthy fixture does not prove scientific/canonical readiness. A newly
compiled one-shot wrapper may be expensive despite a fast reusable owner, so
both cost boundaries must be reported. Timing rankings require valid numerics
and uncertainty support; otherwise timings are descriptive. The baseline is
post-invalidation and uses the same current reset dependency; no old tuning or
claim evidence is promoted. Preserving inherited controls is an execution
comparison, not a scientific justification for those controls. No tolerances,
dtype, reset, training, architecture, tuning policy or analytical score change
is authorized by this unit. Review performed by the primary agent; no
independent review is claimed.

Qualification checkpoint04672: CPU stream7/endpoint8 and GPU stream7 checks
pass. GPU endpoint has7 passes and one healthy dual-trust failure: seed123
second-step ESS differs by1.31069892e-5 against the unchanged1e-6 gate. Keep the
failure and hold costs/promotion. Within the same allocation, first compare
(a) fused seeded XLA, (b) supplied-input XLA with independently drawn normals,
(c) supplied-input XLA with native generated normals and (d) frozen eager.
This distinguishes random-transform rounding, enclosing compilation context
and the existing reset recurrence. Preserve full records, all validity and
conditions, generated inputs and pairwise errors. At most two initial focused
300-second GPU diagnostics; these are explanatory, not equivalence passes.
No tolerance, dtype or rejection criteria change is authorized by a healthy
mismatch. Further repair must follow the localized evidence.

04673 excludes RNG/fused-owner wiring for this case: normal draws are bitwise
identical and both supplied-input owners match the seeded owner exactly.
Unused non-annealed uniforms differ by construction and do not execute.
04674 failed in diagnostic tf.function argument binding before comparisons:
TensorFlow sanitizes leading-underscore keyword defaults. Use a factory closure
to bind function/options instead. This is the first permitted localized harness
retry, charged to the same unit; numerical code and evidence gates are unchanged.

04675 localizes the discrepancy to the existing FP32 reset, and separately to
its trust correction on identical inputs. Compiling only that reset reproduces
the failing trajectory. Against FP64 eager, first-reset particle errors are
2.21e-4 for FP32 eager and1.42e-6 for FP32 XLA; FP64 execution modes agree within
4.9e-15. All three resets are valid, with recorded scaled-system conditions
about8--196. Severe ill-conditioning is not established. The large eager error
and closer XLA answer nominate TF32 arithmetic as a mechanism, not a conclusion.
Allocate up to four further300-second GPU diagnostics inside the original unit
budget: toggle TF32 only in an explicitly diagnostic reference and inspect the
first affected shared primitive if this isolates the cause. Restore TF32 after
each diagnostic. No runtime TF32/dtype/gate change or replacement of the frozen
comparison follows automatically; any revised comparator needs explicit
evidence and a reviewed rationale preserving the original failure.

04678/04679 qualify the shared LM repair; renewed CPU04680 and GPU04681 each
pass all8 endpoint checks.04682 broad CPU regression passes46 and fails one
existing static pfor ban in `ledh_canonical_batch_fused_tf.py` (untouched by
this unit, default sequential; explicit pfor branch already existed at the
starting checkpoint). Preserve the failing test and record it under F14;
inspect its approval scope and callable admission in the terminal phase.
Do not change the test or count the broad group as passing. GPU renewal can
run the numerical subset (random, fixed-input value, analytical score) while
the static F14 failure remains open. This is not a numerical failure of the
registered single-cloud value/LM repair and does not block its bounded costs.

Capacity review: composing supplied-input authorities currently materializes
all process normals with shape[T,N,d] before filtering. This uses O(T*N*d)
random-buffer storage versus the old per-time O(N*d) generator buffer. Tiny
N=8,T=3 measurements cannot establish long-horizon memory readiness. Keep a
streaming seeded-owner/capacity repair as an explicit next phase; no whole
endpoint capacity claim is allowed from the current unit. Costs also need an
eligible uncontended GPU for attribution; shared observations are descriptive
and cannot close that gate.
