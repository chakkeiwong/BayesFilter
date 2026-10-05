# C2 K=1 complete preparation execution result

The K=1 frozen-proposal compiler now runs its numerical time recurrence,
UKF covariance feedback, sampling and exact-prefix weight feedback inside a
retained TensorFlow owner with a fixed signature and XLA enabled by default.
Observations, reference parameters and seed remain live operands. Host entry
validation and completed diagnostic/manifest serialization remain explicit
boundaries. The model retains at most four owners. The numerical method,
Philox stream, shared analytical-score authority and public signature are
preserved. Fixed-mixture and defensive-mixture compilers remain open.

Evidence contract and review: see
[the preparation plan](filter_gradient_c2_preparation_execution_20260930.md).
Runs05401--05419 use the existing runner, unique raw directories and recorded
commands, source hashes, environment, seeds, elapsed time, trusted GPU placement
and verified memory growth. Original source is Git385a348b9. The missing ignored
fixture was recovered from its original seeded diagnostic recipe in05401;
SHA256 is2957a6faeaaea0de893b010a5fd8d66b5e1fae82fb75e0be1645a2524dde603c.
This does not claim identity with unavailable historical JSON bytes.

05402/05403 preserve six complete original CPU/GPU records before runtime edits.
05405/05406 pass13 checks each, including full branch and diagnostic records,
manifest settings, exact ancestor decisions, analytical values/scores, negative
and large seeds, changed observations/theta, original error ordering, enclosing
XLA While/no callbacks and one trace. Maximum absolute differences are
1.2079226507921703e-13 CPU and5.684341886080802e-14 GPU, within the unchanged
1e-10 comparison bounds. Existing strict recomposition/finite-difference checks
also pass. Shared full/prefix evaluation agrees with original sliced programs.

05407 preserves159 passes/two harness failures: missing GPU-group device
registration and an overly strict total graph-node equality assumption.05408
shows only Const/Fill representation changes between horizons, with stable
computational operations. Corrected preflight05409 passes161 checks. Neither
numerical bounds nor execution allowances were relaxed for those failures.

05410--05418 are three counterbalanced fresh-process original/graph/XLA blocks
on the same uncontended GPU. Each measures the complete public preparation
at T3/N16/D4/seed9104, with20 synchronized warm calls.05419 terminal readback
passes full numerical comparisons, source/fixture/device comparability and
worker-exit observations. Medians across the three workers are:

| Complete public preparation | Original | Current graph reference | Current XLA |
|---|---:|---:|---:|
| Warm call | 6049.940 ms | 50.110 ms | 18.067 ms |
| Cold first call | 7.343 s | 5.199 s | 7.572 s |
| Warm host RSS | 2915.723 MiB | 1242.352 MiB | 1308.191 MiB |
| TF device allocator peak | 632.5 KiB | 9366.75 KiB | 573.5 KiB |

Paired geometric warm ratios for XLA are0.00299262 against original
(95% paired log-t interval0.00292570--0.00306107) and0.363492 against graph
(0.237061--0.557353). Thus this measured scope improves substantially on the
original public recurrence. Relative to graph, XLA adds about66MiB host RSS
and2.37s cold cost while reducing warm latency and device peak. This bounded
tradeoff is accepted; graph is an explicit reference exception.

Each XLA worker completes128 additional public reuses with one owner/one trace.
The last64 calls add32768--36864 bytes RSS; live allocator usage stays32512
bytes and process GPU reservation stays513802240 bytes. Clean process exits
are observed. This qualifies the measured reuse scope, not arbitrary capacity
or release of TensorFlow's native allocations after owner eviction.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Nonclaim |
|---|---|---|---|---|---|
| Close K=1 preparation unit | Complete numerical/API/execution and scoped resource checks pass | No outstanding failure in this unit; harness failures preserved | Tiny D4/T3 cost scope; broader families untested | Preserve checkpoint and repair fixed/defensive mixture compilers | No whole-C2/repository closure, scientific admission, or main merge |

Terminal primary-agent review: the strongest misleading explanation would be
timing an inner kernel while excluding host prefix rebuilding. Costs invoke
the complete public compiler and include its completed branch/manifest work;
the original full records were frozen before edits. Three process blocks give
limited cost uncertainty, and128 reuses do not establish universal capacity.
No independent reviewer was launched under the campaign's no-subagent rule.
The original eager reference is not represented as an XLA-compliant baseline.
