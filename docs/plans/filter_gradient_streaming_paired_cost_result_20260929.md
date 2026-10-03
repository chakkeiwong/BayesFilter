# Paired streaming cost result

Twenty fresh CPU-reference processes compare five buffered/streaming pairs at
each of T32/T128,N64,d2,float64. Every shared numerical/status/condition field
matches exactly; the streaming RNG diagnostics also match across pairs. All
owners have one enclosing XLA trace, no host callbacks and the expected absence
of the T-by-N-by-d process-noise buffer. Sources, environment, thread settings,
affinity, inputs and controls match across the cohort. Final readback04749
passes161 checks. These are CPU reference measurements, not GPU-default costs.

| Horizon | Buffered warm median | Streaming warm median | Paired geometric ratio | Conditional95% ratio interval | Median observed RSS difference |
|---|---:|---:|---:|---:|---:|
|32|9.770ms|10.419ms|1.08239|1.04897–1.11687|−11.02MiB|
|128|35.669ms|38.432ms|1.09678|1.04053–1.15608|−10.77MiB|

Warm medians in the table are medians across the five process medians; paired
ratios use within-pair times, so they are not ratios of those table medians.
Each process has three conditioning calls and30 measured calls. The order was
randomized/counterbalanced before execution with seed81130. The independent
replication unit is the process pair, not the thirty correlated calls. Both
exact two-sided sign-flip tests give p=.0625, their minimum with five pairs.
The t intervals assume independent approximately normal log effects and are
reported with that limitation. Both upper limits exceed10%, triggering focused
profiling under the predeclared plan; the runtime cost is not accepted yet.

Median cold compilation/first-call cost is5.104s buffered versus5.375s streaming
atT32, and4.985s versus5.449s atT128. Median RSS growth from before compilation
to after warm calls is537.39/526.56MiB (buffered/streaming) atT32 and
537.25/526.51MiB atT128. Additional median RSS after the first call is
0.215/0.117MiB and0.449/0.102MiB, respectively. Primary memory samples precede
compiler exports and the comparison owner. The large compile-associated host
residency remains far greater than the32/128KiB random buffer savings; that is
consistent with earlier compiler/context observations, not proof of a leak or
proof that all observed RSS changes are attributable to the removed buffer.

04728 is a preserved failed pilot: it compared dictionaries with different
schemas. All shared numerical errors were zero, but the streaming record adds
three RNG fields. The reviewed repair compares every shared field exactly and
requires/records exactly those three extras. The complete20-worker cohort then
restarted at04729 and ends04748. No timing sample was trimmed or retried for its
value; no numerical tolerance, runtime, RNG stream or gate changed.04728's
timing is excluded solely for the documented harness-schema failure.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
|Keep numerical/memory repair evidence|All20 workers have exact complete shared records and expected buffer removal|All declared numerical/source/graph checks pass|Native/compiler residency remains large|Retain explicit reuse/process containment and GPU capacity gate|Leak freedom or universal capacity|
|Do not yet accept runtime cost|Both ratio intervals extend above1.10|No numerical failures in the measured cohort|Only five pairs; compiler/RNG interaction not attributed|Execute filter_gradient_streaming_profile_20260929.md|Uncontended GPU cost or global speed ranking|

| Inference status | Finding |
|---|---|
|Hard veto screen|Exact numerics, provenance, frozen source, one trace and HLO pass|
|Statistically supported ranking|None at a distribution-free5% threshold; five-pair limitation explicit|
|Descriptive-only differences|Streaming is slower in every pair; geometric effects8.24%/9.68%; RSS about11MiB lower|
|Default readiness|No performance acceptance or whole-master closure|
|Next evidence|Focused saved-HLO/component localization and then a full-filter intervention if justified|

Skeptical terminal review: the initial19% slowdown is not reproduced at that
magnitude, but the new effect still activates the predeclared continuation.
Neither small sample uncertainty nor memory savings is a reason to waive it.
The observations support the measured CPU scope only. Primary-agent review;
no independent review asserted.

The unit uses22 workers including the failed pilot and final readback,
449.669978 CPU seconds, no GPU seconds. Through04749 the remaining allocation
is25.467088 CPU/25.021431 GPU hours. Archive docs/plans/artifacts/filter-gradient-repair-20260917/streaming-paired-04749-evidence.tar.gz
contains142 reopened/verified members and9968524 bytes; SHA256
5ab3ee9719113d9af0450e4949942bd5abd0ca7f9f0be53ba2a46785e7c2aae8. The adjacent verification JSON binds each member;
the archived result precedes this receipt paragraph. No worker remains active.
