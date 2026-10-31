# Native LEDH flow execution costs

The shared flow now passes frozen-source CPU/GPU checks at 1e-12 after
preserving the old binary32 coefficient rounding. This unit measures that
qualified dependency, not the still-unintegrated full value/score endpoints.
The known FP32 dual-trust full-value comparison remains a veto and is excluded
from performance claims. No canonical LEDH or scientific admission follows.

Question: how do Python-unrolled graph flow, native graph flow and native XLA
flow differ in compilation, warm time, host RSS and TensorFlow allocator usage?
Baseline is exact `9d8202b77:bayesfilter/highdim/ledh_flow_perparticle_tf.py`.
All arms load identical source authorities and inputs before measurement;
the old eager flow supplies an independent frozen-source check after timing.
Use 32 particles, d=o=2, 24 substeps, float64, seed 37 and the existing linear
fixture (model seed 13). These small diagnostic controls are not tuned LEDH
production settings. The flow has no OT/chunk operation.

Run three interleaved fresh-process repeats of prior graph / native graph /
native XLA on each CPU reference and one pinned non-desktop GPU. At most 18
workers / 5400 charged seconds, one at a time, within the cumulative 56 CPU /
52 GPU-hour authorization. Each worker is bounded to 300 seconds. No runtime,
test or runner edits during a worker. Use the stable campaign runner and
unique `run-*` directories under the existing campaign output root. CPU runs
hide GPUs; GPU runs use verified memory growth and uncontended preflight plus
in-run process observations. Preserve failed attempts; at most three localized
retries per unchanged arm. Source drift, nonfinite values, failed device
provenance or numerical mismatch stops the cost matrix pending localization.

All six flow output fields must match frozen eager within atol=rtol=1e-12;
changed inputs must also agree, replay must be exact and one trace must serve
all calls. These are validity vetoes before any speed ratio is eligible.
Measure cold construction/trace/first synchronized call and 15 warm synchronized
calls. Sample process RSS every 10 ms during costs; report baseline, sampled
peak and end RSS plus process-lifetime high-water mark separately. Record
TensorFlow current/peak allocator bytes, not nvidia-smi reservation. Export
graph node counts and native HLO only after measured intervals to avoid charging
diagnostic export to cold/warm cost. Missing CPU allocator counters are explicit;
GPU counters are required. GPU process samples cannot prove exclusivity between
samples. Cleanup/GC snapshots do not prove native executable eviction.

Report each repetition and descriptive medians; three repeats do not establish
population performance rankings. Investigate any >20% median warm slowdown,
>2x cold time, or host/allocator incremental peak increase >2x with at least
100 MiB host / 16 MiB allocator excess. These are explanatory repair triggers,
not reasons to relax numerical gates or demote GPU/XLA policy. No full-filter,
score, training, capacity, main-promotion or posterior claim follows.

Skeptical review: comparing only XLA to uncompiled Python would exaggerate
speedup, so the primary baseline is the old TensorFlow graph. Reference
evaluation and HLO export occur after measurements. Identical imports, pinned
hardware, explicit synchronization and fresh processes reduce attribution
confounds. Tiny dimensions limit generality and are stated as a nonclaim.
The first CPU/GPU repeat qualifies this exact scope before the repeat ladder.
Primary-agent review; no subagent or independent reviewer was used.

03832--03835 pass the three CPU arms and prior GPU graph. GPU native graph
03836 fails before timing: TensorFlow has no GPU AddN kernel for uint64,
which the explicit-rounding helper uses. XLA itself supports it. Preserve
these initial attempts and repair the helper to use signed int64 bit operations:
all nonnegative stage fractions have zero sign bit, so arithmetic and masks
are exactly equivalent. Signed AddN is supported on both backends. Rerun
the matched matrix after this source change; prior measurements remain
explanatory only. Extend this localized-repair unit to 24 workers / 7200
charged seconds (18 accepted rows plus up to six preserved/retry rows) within
the same global budget. No method, dtype, fixture, criterion, device class or
scientific boundary changes. This is the automatic localized retry authorized
by the active academic governance profile.

First matched cohort 03837--03842 passes all frozen/changed-input/replay gates
under one source closure and verified device provenance. Graph size drops
2628 -> 359 nodes. Descriptive first-repeat warm medians: CPU prior/native
graph/XLA 5.244/6.482/0.701 ms; GPU 10.096/48.478/2.918 ms. CPU incremental
sampled RSS is 52.44/25.68/177.45 MiB, so the XLA host-memory trigger fires.
GPU peaks are 135.94/157.02/130.27 MiB, with live allocator peaks only
92416/35072/30976 bytes. Native non-XLA GPU graph cold time and warm time also
trigger investigation; it is the explicit reference exception, not the default.
No source edits between matched workers. Continue the declared two additional
interleaved repeats; then independently analyze medians and preserve the
CPU compiler-memory issue without asserting an unmeasured cause or eviction.
